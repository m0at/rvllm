//! GPTNeoXForCausalLM and StableLmForCausalLM architectures.
//!
//! GPT-NeoX uses LayerNorm (not RMSNorm), rotary embedding on the full head_dim
//! (non-interleaved), parallel attention + MLP residual streams, optional
//! attention QKV bias, and GELU activation in the MLP.
//!
//! StableLM shares the same graph but uses different weight prefixes and may
//! omit the attention bias.

use half::f16;
use tracing::trace;

use crate::bridge::{
    AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result,
};
use crate::input::ModelInput;
use crate::layers::activation::gelu;
use crate::layers::linear::LinearLayer;
use crate::layers::norm::LayerNorm;
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, embed_tokens, get_or_zeros, lm_head};
use super::Architecture;

// ---------------------------------------------------------------------------
// Shared config & layer structs
// ---------------------------------------------------------------------------

struct NeoXConfig {
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    layer_norm_eps: f32,
    use_parallel_residual: bool,
}

struct NeoXLayer {
    // Pre-attention LayerNorm (weight + bias).
    ln_attn_weight: GpuBuffer<f16>,
    ln_attn_bias: GpuBuffer<f16>,
    // Pre-MLP LayerNorm (weight + bias).
    ln_mlp_weight: GpuBuffer<f16>,
    ln_mlp_bias: GpuBuffer<f16>,
    // Attention projections.
    q_proj: GpuBuffer<f16>,
    k_proj: GpuBuffer<f16>,
    v_proj: GpuBuffer<f16>,
    // Optional QKV biases.
    q_bias: Option<GpuBuffer<f16>>,
    k_bias: Option<GpuBuffer<f16>>,
    v_bias: Option<GpuBuffer<f16>>,
    // Output projection.
    o_proj: GpuBuffer<f16>,
    o_bias: Option<GpuBuffer<f16>>,
    // MLP: dense_h_to_4h (up), dense_4h_to_h (down).
    dense_h_to_4h_weight: GpuBuffer<f16>,
    dense_h_to_4h_bias: Option<GpuBuffer<f16>>,
    dense_4h_to_h_weight: GpuBuffer<f16>,
    dense_4h_to_h_bias: Option<GpuBuffer<f16>>,
}

// ---------------------------------------------------------------------------
// GPTNeoXForCausalLM
// ---------------------------------------------------------------------------

/// GPT-NeoX causal language model (Pythia, GPT-NeoX-20B, etc.).
pub struct GPTNeoXForCausalLM {
    config: NeoXConfig,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<NeoXLayer>,
    final_ln_weight: GpuBuffer<f16>,
    final_ln_bias: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

impl GPTNeoXForCausalLM {
    /// Construct from loaded weights and runner config.
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let cfg = NeoXConfig {
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            layer_norm_eps: 1e-5,
            use_parallel_residual: true,
        };

        let h = cfg.hidden_size;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        let embed_tokens = weights
            .get_as_buffer("gpt_neox.embed_in.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, h]));

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("gpt_neox.layers.{}", i);
            layers.push(NeoXLayer {
                ln_attn_weight: get_or_zeros(&weights, &format!("{p}.input_layernorm.weight"), &[h]),
                ln_attn_bias: get_or_zeros(&weights, &format!("{p}.input_layernorm.bias"), &[h]),
                ln_mlp_weight: get_or_zeros(&weights, &format!("{p}.post_attention_layernorm.weight"), &[h]),
                ln_mlp_bias: get_or_zeros(&weights, &format!("{p}.post_attention_layernorm.bias"), &[h]),
                q_proj: get_or_zeros(&weights, &format!("{p}.attention.query_key_value.q_proj.weight"), &[q_dim, h]),
                k_proj: get_or_zeros(&weights, &format!("{p}.attention.query_key_value.k_proj.weight"), &[kv_dim, h]),
                v_proj: get_or_zeros(&weights, &format!("{p}.attention.query_key_value.v_proj.weight"), &[kv_dim, h]),
                q_bias: weights.get_as_buffer(&format!("{p}.attention.query_key_value.q_proj.bias")).ok(),
                k_bias: weights.get_as_buffer(&format!("{p}.attention.query_key_value.k_proj.bias")).ok(),
                v_bias: weights.get_as_buffer(&format!("{p}.attention.query_key_value.v_proj.bias")).ok(),
                o_proj: get_or_zeros(&weights, &format!("{p}.attention.dense.weight"), &[h, q_dim]),
                o_bias: weights.get_as_buffer(&format!("{p}.attention.dense.bias")).ok(),
                dense_h_to_4h_weight: get_or_zeros(&weights, &format!("{p}.mlp.dense_h_to_4h.weight"), &[inter, h]),
                dense_h_to_4h_bias: weights.get_as_buffer(&format!("{p}.mlp.dense_h_to_4h.bias")).ok(),
                dense_4h_to_h_weight: get_or_zeros(&weights, &format!("{p}.mlp.dense_4h_to_h.weight"), &[h, inter]),
                dense_4h_to_h_bias: weights.get_as_buffer(&format!("{p}.mlp.dense_4h_to_h.bias")).ok(),
            });
        }

        let final_ln_weight = get_or_zeros(&weights, "gpt_neox.final_layer_norm.weight", &[h]);
        let final_ln_bias = get_or_zeros(&weights, "gpt_neox.final_layer_norm.bias", &[h]);

        let lm_head_weight = weights
            .get_as_buffer("embed_out.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, h]));

        Ok(Self {
            config: cfg,
            embed_tokens,
            layers,
            final_ln_weight,
            final_ln_bias,
            lm_head_weight,
        })
    }
}

impl Architecture for GPTNeoXForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let h = self.config.hidden_size;
        let eps = self.config.layer_norm_eps;

        let mut hidden = embed_tokens(&self.embed_tokens, &input.token_ids, h);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "gpt_neox layer forward");

            // Pre-attention LayerNorm.
            let normed_attn = LayerNorm::forward(
                &hidden,
                &layer.ln_attn_weight,
                &layer.ln_attn_bias,
                eps,
            )?;

            // QKV with optional bias.
            let q = LinearLayer::forward(&normed_attn, &layer.q_proj, layer.q_bias.as_ref())?;
            let k = LinearLayer::forward(&normed_attn, &layer.k_proj, layer.k_bias.as_ref())?;
            let v = LinearLayer::forward(&normed_attn, &layer.v_proj, layer.v_bias.as_ref())?;

            // RoPE on full head_dim (non-interleaved -- same RotaryEmbedding,
            // GPT-NeoX applies rotation to the entire head dimension).
            let (q_rot, k_rot) = RotaryEmbedding::forward(
                &input.position_ids,
                &q,
                &k,
                self.config.head_dim,
            )?;

            // Attention.
            let attn_out = attention.forward(
                &q_rot,
                &k_rot,
                &v,
                &input.attention_metadata,
                layer_idx,
            )?;

            // Output projection with optional bias.
            let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, layer.o_bias.as_ref())?;

            if self.config.use_parallel_residual {
                // Parallel attention + MLP: both branches computed from the
                // *same* pre-norm hidden state and added to the residual
                // simultaneously.
                let normed_mlp = LayerNorm::forward(
                    &hidden,
                    &layer.ln_mlp_weight,
                    &layer.ln_mlp_bias,
                    eps,
                )?;

                let mlp_out = neox_mlp(
                    &normed_mlp,
                    &layer.dense_h_to_4h_weight,
                    layer.dense_h_to_4h_bias.as_ref(),
                    &layer.dense_4h_to_h_weight,
                    layer.dense_4h_to_h_bias.as_ref(),
                )?;

                // hidden = hidden + attn_out + mlp_out
                add_inplace(&mut hidden, &attn_proj);
                add_inplace(&mut hidden, &mlp_out);
            } else {
                // Sequential residual (some NeoX variants).
                add_inplace(&mut hidden, &attn_proj);

                let normed_mlp = LayerNorm::forward(
                    &hidden,
                    &layer.ln_mlp_weight,
                    &layer.ln_mlp_bias,
                    eps,
                )?;

                let mlp_out = neox_mlp(
                    &normed_mlp,
                    &layer.dense_h_to_4h_weight,
                    layer.dense_h_to_4h_bias.as_ref(),
                    &layer.dense_4h_to_h_weight,
                    layer.dense_4h_to_h_bias.as_ref(),
                )?;

                add_inplace(&mut hidden, &mlp_out);
            }
        }

        // Final LayerNorm.
        let normed_final = LayerNorm::forward(
            &hidden,
            &self.final_ln_weight,
            &self.final_ln_bias,
            eps,
        )?;

        lm_head(&normed_final, &self.lm_head_weight, num_tokens, self.config.vocab_size)
    }
}

// ---------------------------------------------------------------------------
// StableLmForCausalLM
// ---------------------------------------------------------------------------

/// StableLM causal language model (StableLM-3B, StableLM-2, etc.).
///
/// Same computational graph as GPT-NeoX with different weight naming
/// (`model.` prefix, Llama-style layer names) and LayerNorm instead of
/// RMSNorm. StableLM models may use parallel or sequential residual
/// depending on the variant.
pub struct StableLmForCausalLM {
    config: NeoXConfig,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<NeoXLayer>,
    final_ln_weight: GpuBuffer<f16>,
    final_ln_bias: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

impl StableLmForCausalLM {
    /// Construct from loaded weights and runner config.
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let cfg = NeoXConfig {
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            layer_norm_eps: 1e-5,
            use_parallel_residual: true,
        };

        let h = cfg.hidden_size;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, h]));

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{}", i);
            layers.push(NeoXLayer {
                ln_attn_weight: get_or_zeros(&weights, &format!("{p}.input_layernorm.weight"), &[h]),
                ln_attn_bias: get_or_zeros(&weights, &format!("{p}.input_layernorm.bias"), &[h]),
                ln_mlp_weight: get_or_zeros(&weights, &format!("{p}.post_attention_layernorm.weight"), &[h]),
                ln_mlp_bias: get_or_zeros(&weights, &format!("{p}.post_attention_layernorm.bias"), &[h]),
                q_proj: get_or_zeros(&weights, &format!("{p}.self_attn.q_proj.weight"), &[q_dim, h]),
                k_proj: get_or_zeros(&weights, &format!("{p}.self_attn.k_proj.weight"), &[kv_dim, h]),
                v_proj: get_or_zeros(&weights, &format!("{p}.self_attn.v_proj.weight"), &[kv_dim, h]),
                q_bias: weights.get_as_buffer(&format!("{p}.self_attn.q_proj.bias")).ok(),
                k_bias: weights.get_as_buffer(&format!("{p}.self_attn.k_proj.bias")).ok(),
                v_bias: weights.get_as_buffer(&format!("{p}.self_attn.v_proj.bias")).ok(),
                o_proj: get_or_zeros(&weights, &format!("{p}.self_attn.o_proj.weight"), &[h, q_dim]),
                o_bias: weights.get_as_buffer(&format!("{p}.self_attn.o_proj.bias")).ok(),
                dense_h_to_4h_weight: get_or_zeros(&weights, &format!("{p}.mlp.up_proj.weight"), &[inter, h]),
                dense_h_to_4h_bias: weights.get_as_buffer(&format!("{p}.mlp.up_proj.bias")).ok(),
                dense_4h_to_h_weight: get_or_zeros(&weights, &format!("{p}.mlp.down_proj.weight"), &[h, inter]),
                dense_4h_to_h_bias: weights.get_as_buffer(&format!("{p}.mlp.down_proj.bias")).ok(),
            });
        }

        let final_ln_weight = get_or_zeros(&weights, "model.norm.weight", &[h]);
        let final_ln_bias = get_or_zeros(&weights, "model.norm.bias", &[h]);

        let lm_head_weight = weights
            .get_as_buffer("lm_head.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, h]));

        Ok(Self {
            config: cfg,
            embed_tokens,
            layers,
            final_ln_weight,
            final_ln_bias,
            lm_head_weight,
        })
    }
}

impl Architecture for StableLmForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let h = self.config.hidden_size;
        let eps = self.config.layer_norm_eps;

        let mut hidden = embed_tokens(&self.embed_tokens, &input.token_ids, h);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "stablelm layer forward");

            let normed_attn = LayerNorm::forward(
                &hidden,
                &layer.ln_attn_weight,
                &layer.ln_attn_bias,
                eps,
            )?;

            let q = LinearLayer::forward(&normed_attn, &layer.q_proj, layer.q_bias.as_ref())?;
            let k = LinearLayer::forward(&normed_attn, &layer.k_proj, layer.k_bias.as_ref())?;
            let v = LinearLayer::forward(&normed_attn, &layer.v_proj, layer.v_bias.as_ref())?;

            let (q_rot, k_rot) = RotaryEmbedding::forward(
                &input.position_ids,
                &q,
                &k,
                self.config.head_dim,
            )?;

            let attn_out = attention.forward(
                &q_rot,
                &k_rot,
                &v,
                &input.attention_metadata,
                layer_idx,
            )?;

            let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, layer.o_bias.as_ref())?;

            if self.config.use_parallel_residual {
                let normed_mlp = LayerNorm::forward(
                    &hidden,
                    &layer.ln_mlp_weight,
                    &layer.ln_mlp_bias,
                    eps,
                )?;

                let mlp_out = neox_mlp(
                    &normed_mlp,
                    &layer.dense_h_to_4h_weight,
                    layer.dense_h_to_4h_bias.as_ref(),
                    &layer.dense_4h_to_h_weight,
                    layer.dense_4h_to_h_bias.as_ref(),
                )?;

                add_inplace(&mut hidden, &attn_proj);
                add_inplace(&mut hidden, &mlp_out);
            } else {
                add_inplace(&mut hidden, &attn_proj);

                let normed_mlp = LayerNorm::forward(
                    &hidden,
                    &layer.ln_mlp_weight,
                    &layer.ln_mlp_bias,
                    eps,
                )?;

                let mlp_out = neox_mlp(
                    &normed_mlp,
                    &layer.dense_h_to_4h_weight,
                    layer.dense_h_to_4h_bias.as_ref(),
                    &layer.dense_4h_to_h_weight,
                    layer.dense_4h_to_h_bias.as_ref(),
                )?;

                add_inplace(&mut hidden, &mlp_out);
            }
        }

        let normed_final = LayerNorm::forward(
            &hidden,
            &self.final_ln_weight,
            &self.final_ln_bias,
            eps,
        )?;

        lm_head(&normed_final, &self.lm_head_weight, num_tokens, self.config.vocab_size)
    }
}

// ---------------------------------------------------------------------------
// GPT-NeoX MLP: dense_h_to_4h -> GELU -> dense_4h_to_h
// ---------------------------------------------------------------------------

/// NeoX-style MLP with GELU activation (no gating).
fn neox_mlp(
    input: &GpuBuffer<f16>,
    up_weight: &GpuBuffer<f16>,
    up_bias: Option<&GpuBuffer<f16>>,
    down_weight: &GpuBuffer<f16>,
    down_bias: Option<&GpuBuffer<f16>>,
) -> Result<GpuBuffer<f16>> {
    let up = LinearLayer::forward(input, up_weight, up_bias)?;
    let activated = gelu(&up)?;
    LinearLayer::forward(&activated, down_weight, down_bias)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{AttentionMetadata, MockAttentionBackend, CacheEngine, ModelWeights};
    use crate::input::ModelInput;
    use crate::runner::ModelRunnerConfig;

    fn test_config() -> ModelRunnerConfig {
        ModelRunnerConfig {
            num_layers: 2,
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: 32,
            max_position: 128,
            dtype: "float16".into(),
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            architecture: "GPTNeoXForCausalLM".into(),
        }
    }

    fn test_input() -> ModelInput {
        ModelInput {
            token_ids: vec![1, 2, 3],
            position_ids: vec![0, 1, 2],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0, 1, 2],
                context_lens: vec![3],
                block_tables: vec![vec![0]],
                max_context_len: 3,
            },
            is_prefill: true,
        }
    }

    #[test]
    fn gpt_neox_forward_smoke() {
        let cfg = test_config();
        let weights = ModelWeights::default();
        let model = GPTNeoXForCausalLM::new(weights, &cfg).unwrap();
        let input = test_input();
        let cache = CacheEngine::new(cfg.num_layers, 256);
        let attn = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attn).unwrap();
        assert_eq!(logits.shape, vec![3, 32]);
    }

    #[test]
    fn gpt_neox_output_shape() {
        let cfg = test_config();
        let weights = ModelWeights::default();
        let model = GPTNeoXForCausalLM::new(weights, &cfg).unwrap();
        let input = ModelInput {
            token_ids: vec![0],
            position_ids: vec![0],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0],
                context_lens: vec![1],
                block_tables: vec![vec![0]],
                max_context_len: 1,
            },
            is_prefill: true,
        };
        let cache = CacheEngine::new(cfg.num_layers, 256);
        let attn = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attn).unwrap();
        assert_eq!(logits.shape, vec![1, 32]);
        assert_eq!(logits.data.len(), 32);
    }

    #[test]
    fn stablelm_forward_smoke() {
        let mut cfg = test_config();
        cfg.architecture = "StableLmForCausalLM".into();
        let weights = ModelWeights::default();
        let model = StableLmForCausalLM::new(weights, &cfg).unwrap();
        let input = test_input();
        let cache = CacheEngine::new(cfg.num_layers, 256);
        let attn = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attn).unwrap();
        assert_eq!(logits.shape, vec![3, 32]);
    }

    #[test]
    fn stablelm_single_token() {
        let mut cfg = test_config();
        cfg.architecture = "StableLmForCausalLM".into();
        let weights = ModelWeights::default();
        let model = StableLmForCausalLM::new(weights, &cfg).unwrap();
        let input = ModelInput {
            token_ids: vec![5],
            position_ids: vec![0],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0],
                context_lens: vec![1],
                block_tables: vec![vec![0]],
                max_context_len: 1,
            },
            is_prefill: false,
        };
        let cache = CacheEngine::new(cfg.num_layers, 256);
        let attn = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attn).unwrap();
        assert_eq!(logits.shape, vec![1, 32]);
    }

    #[test]
    fn neox_mlp_smoke() {
        let h = 4;
        let inter = 8;
        let input = GpuBuffer::from_vec(
            vec![f16::from_f32(1.0); h],
            vec![1, h],
        );
        let up_w = GpuBuffer::from_vec(
            vec![f16::from_f32(0.1); inter * h],
            vec![inter, h],
        );
        let down_w = GpuBuffer::from_vec(
            vec![f16::from_f32(0.1); h * inter],
            vec![h, inter],
        );
        let out = neox_mlp(&input, &up_w, None, &down_w, None).unwrap();
        assert_eq!(out.shape, vec![1, h]);
        // With non-zero weights, GELU should produce non-zero output.
        let any_nonzero = out.data.iter().any(|v| v.to_f32().abs() > 1e-6);
        assert!(any_nonzero, "MLP should produce non-zero output");
    }
}
