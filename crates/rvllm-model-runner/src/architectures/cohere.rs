//! CohereForCausalLM architecture (Command-R, Command-R+).
//!
//! Key differences from Llama:
//! - LayerNorm (with bias) instead of RMSNorm
//! - QK normalization: L2-normalize query and key vectors before attention
//! - Multi-Query Attention (MQA): single key/value head shared across all query heads
//! - Tied embed/lm_head weights (Command-R uses weight tying)

use half::f16;
use tracing::trace;

use crate::bridge::{
    AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result,
};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::mlp::MLP;
use crate::layers::norm::LayerNorm;
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, embed_tokens, get_or_zeros, lm_head};
use super::Architecture;

/// Cohere Command-R / Command-R+ causal language model.
pub struct CohereForCausalLM {
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    layernorm_eps: f32,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<CohereLayer>,
    norm_weight: GpuBuffer<f16>,
    norm_bias: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

/// A single transformer layer in the Cohere architecture.
struct CohereLayer {
    /// Pre-attention LayerNorm weight.
    input_layernorm_weight: GpuBuffer<f16>,
    /// Pre-attention LayerNorm bias.
    input_layernorm_bias: GpuBuffer<f16>,
    /// Q projection weight.
    q_proj: GpuBuffer<f16>,
    /// K projection weight (single head for MQA).
    k_proj: GpuBuffer<f16>,
    /// V projection weight (single head for MQA).
    v_proj: GpuBuffer<f16>,
    /// Output projection weight.
    o_proj: GpuBuffer<f16>,
    /// QK normalization: per-head layernorm weight for queries.
    q_norm_weight: GpuBuffer<f16>,
    /// QK normalization: per-head layernorm bias for queries.
    q_norm_bias: GpuBuffer<f16>,
    /// QK normalization: per-head layernorm weight for keys.
    k_norm_weight: GpuBuffer<f16>,
    /// QK normalization: per-head layernorm bias for keys.
    k_norm_bias: GpuBuffer<f16>,
    /// MLP gate projection.
    gate_proj: GpuBuffer<f16>,
    /// MLP up projection.
    up_proj: GpuBuffer<f16>,
    /// MLP down projection.
    down_proj: GpuBuffer<f16>,
}

impl CohereForCausalLM {
    /// Construct a new CohereForCausalLM from pretrained weights.
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_dim = config.num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]));

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{}", i);
            layers.push(CohereLayer {
                input_layernorm_weight: get_or_zeros(
                    &weights,
                    &format!("{p}.input_layernorm.weight"),
                    &[config.hidden_size],
                ),
                input_layernorm_bias: get_or_zeros(
                    &weights,
                    &format!("{p}.input_layernorm.bias"),
                    &[config.hidden_size],
                ),
                q_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.q_proj.weight"),
                    &[q_dim, config.hidden_size],
                ),
                k_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.k_proj.weight"),
                    &[kv_dim, config.hidden_size],
                ),
                v_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[kv_dim, config.hidden_size],
                ),
                o_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    &[config.hidden_size, q_dim],
                ),
                q_norm_weight: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[head_dim],
                ),
                q_norm_bias: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.q_norm.bias"),
                    &[head_dim],
                ),
                k_norm_weight: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[head_dim],
                ),
                k_norm_bias: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.k_norm.bias"),
                    &[head_dim],
                ),
                gate_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.mlp.gate_proj.weight"),
                    &[config.intermediate_size, config.hidden_size],
                ),
                up_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.mlp.up_proj.weight"),
                    &[config.intermediate_size, config.hidden_size],
                ),
                down_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.mlp.down_proj.weight"),
                    &[config.hidden_size, config.intermediate_size],
                ),
            });
        }

        let norm_weight = weights
            .get_as_buffer("model.norm.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.hidden_size]));

        let norm_bias = weights
            .get_as_buffer("model.norm.bias")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.hidden_size]));

        // Command-R uses tied embeddings: lm_head shares embed_tokens weight.
        let lm_head_weight = weights
            .get_as_buffer("lm_head.weight")
            .unwrap_or_else(|_| {
                weights
                    .get_as_buffer("model.embed_tokens.weight")
                    .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]))
            });

        Ok(Self {
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads,
            head_dim,
            vocab_size: config.vocab_size,
            layernorm_eps: 1e-5,
            embed_tokens,
            layers,
            norm_weight,
            norm_bias,
            lm_head_weight,
        })
    }
}

impl Architecture for CohereForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let h = self.hidden_size;

        // Embedding lookup.
        let mut hidden = embed_tokens(&self.embed_tokens, &input.token_ids, h);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "cohere layer forward");

            // Pre-attention LayerNorm (not RMSNorm).
            let normed = LayerNorm::forward(
                &hidden,
                &layer.input_layernorm_weight,
                &layer.input_layernorm_bias,
                self.layernorm_eps,
            )?;

            // QKV projections.
            let q = LinearLayer::forward(&normed, &layer.q_proj, None)?;
            let k = LinearLayer::forward(&normed, &layer.k_proj, None)?;
            let v = LinearLayer::forward(&normed, &layer.v_proj, None)?;

            // QK normalization: apply per-head LayerNorm to Q and K.
            let q_normed = qk_norm(
                &q,
                &layer.q_norm_weight,
                &layer.q_norm_bias,
                self.num_heads,
                self.head_dim,
                self.layernorm_eps,
            )?;
            let k_normed = qk_norm(
                &k,
                &layer.k_norm_weight,
                &layer.k_norm_bias,
                self.num_kv_heads,
                self.head_dim,
                self.layernorm_eps,
            )?;

            // RoPE on normalized Q/K.
            let (q_rot, k_rot) = RotaryEmbedding::forward(
                &input.position_ids,
                &q_normed,
                &k_normed,
                self.head_dim,
            )?;

            // Expand shared keys for MQA: replicate the single KV head(s) to match num_heads.
            let k_expanded = expand_kv_heads(
                &k_rot,
                self.num_kv_heads,
                self.num_heads,
                self.head_dim,
                num_tokens,
            );
            let v_expanded = expand_kv_heads(
                &v,
                self.num_kv_heads,
                self.num_heads,
                self.head_dim,
                num_tokens,
            );

            // Attention.
            let attn_out = attention.forward(
                &q_rot,
                &k_expanded,
                &v_expanded,
                &input.attention_metadata,
                layer_idx,
            )?;

            // Output projection.
            let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, None)?;

            // Residual connection for attention.
            add_inplace(&mut hidden, &attn_proj);

            // Cohere uses a single pre-norm for both attention and MLP (parallel style),
            // but the residual from MLP also adds to hidden. We reuse the same normed
            // input for MLP (parallel residual pattern as in Command-R).
            let mlp_out = MLP::forward(
                &normed,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
            )?;
            add_inplace(&mut hidden, &mlp_out);
        }

        // Final LayerNorm.
        let normed_final = LayerNorm::forward(
            &hidden,
            &self.norm_weight,
            &self.norm_bias,
            self.layernorm_eps,
        )?;

        // LM head projection to vocab logits.
        lm_head(&normed_final, &self.lm_head_weight, num_tokens, self.vocab_size)
    }
}

/// Apply per-head LayerNorm to a projected QKV tensor.
///
/// Input shape: [num_tokens, num_heads * head_dim]
/// Norm weight/bias shape: [head_dim]
/// Output shape: [num_tokens, num_heads * head_dim]
///
/// Each head's slice is independently normalized using the shared per-head
/// LayerNorm parameters.
fn qk_norm(
    input: &GpuBuffer<f16>,
    weight: &GpuBuffer<f16>,
    bias: &GpuBuffer<f16>,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<GpuBuffer<f16>> {
    let total_per_token = num_heads * head_dim;
    let num_tokens = input.len() / total_per_token;
    let mut out = vec![f16::ZERO; input.len()];

    let w_f32: Vec<f32> = weight.data.iter().map(|v| v.to_f32()).collect();
    let b_f32: Vec<f32> = bias.data.iter().map(|v| v.to_f32()).collect();
    let inv_n = 1.0f32 / head_dim as f32;

    for t in 0..num_tokens {
        for h in 0..num_heads {
            let offset = t * total_per_token + h * head_dim;
            let row = &input.data[offset..offset + head_dim];
            let dst = &mut out[offset..offset + head_dim];

            // Mean.
            let mut sum = 0.0f32;
            for v in row.iter() {
                sum += v.to_f32();
            }
            let mean = sum * inv_n;

            // Variance.
            let mut var_sum = 0.0f32;
            for v in row.iter() {
                let d = v.to_f32() - mean;
                var_sum += d * d;
            }
            let inv_std = (var_sum * inv_n + eps).sqrt().recip();

            // Normalize + scale + bias.
            for i in 0..head_dim {
                let x = row[i].to_f32();
                dst[i] = f16::from_f32((x - mean) * inv_std * w_f32[i] + b_f32[i]);
            }
        }
    }

    Ok(GpuBuffer::from_vec(out, input.shape.clone()))
}

/// Expand KV heads for MQA/GQA: replicate kv_heads to match num_q_heads.
///
/// If num_kv_heads == num_q_heads, returns input unchanged.
/// Otherwise repeats each KV head (num_q_heads / num_kv_heads) times.
fn expand_kv_heads(
    input: &GpuBuffer<f16>,
    num_kv_heads: usize,
    num_q_heads: usize,
    head_dim: usize,
    num_tokens: usize,
) -> GpuBuffer<f16> {
    if num_kv_heads == num_q_heads {
        return input.clone();
    }

    let repeat = num_q_heads / num_kv_heads;
    let kv_per_token = num_kv_heads * head_dim;
    let q_per_token = num_q_heads * head_dim;
    let mut out = vec![f16::ZERO; num_tokens * q_per_token];

    for t in 0..num_tokens {
        let src_base = t * kv_per_token;
        let dst_base = t * q_per_token;
        for kv_h in 0..num_kv_heads {
            let src_off = src_base + kv_h * head_dim;
            let src_slice = &input.data[src_off..src_off + head_dim];
            for r in 0..repeat {
                let dst_off = dst_base + (kv_h * repeat + r) * head_dim;
                out[dst_off..dst_off + head_dim].copy_from_slice(src_slice);
            }
        }
    }

    GpuBuffer::from_vec(out, vec![num_tokens, q_per_token])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{
        AttentionMetadata, CacheEngine, MockAttentionBackend, ModelWeights,
    };
    use crate::input::ModelInput;
    use crate::runner::ModelRunnerConfig;

    fn make_buf(vals: &[f32], shape: Vec<usize>) -> GpuBuffer<f16> {
        GpuBuffer::from_vec(
            vals.iter().map(|&v| f16::from_f32(v)).collect(),
            shape,
        )
    }

    fn tiny_config() -> ModelRunnerConfig {
        ModelRunnerConfig {
            num_layers: 1,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            intermediate_size: 8,
            vocab_size: 8,
            max_position: 32,
            dtype: "float16".into(),
            architecture: "CohereForCausalLM".into(),
        }
    }

    #[test]
    fn qk_norm_identity() {
        // Constant input -> mean=c, var=0, normalized=0, output=bias.
        let input = make_buf(&[3.0, 3.0, 3.0, 3.0], vec![1, 4]);
        let weight = make_buf(&[1.0, 1.0], vec![2]);
        let bias = make_buf(&[5.0, 5.0], vec![2]);
        let out = qk_norm(&input, &weight, &bias, 2, 2, 1e-5).unwrap();
        for v in out.data.iter() {
            assert!(
                (v.to_f32() - 5.0).abs() < 0.1,
                "expected ~5.0, got {}",
                v.to_f32()
            );
        }
    }

    #[test]
    fn qk_norm_nontrivial() {
        // Two heads of dim 2: [1, -1, 2, -2] for one token.
        let input = make_buf(&[1.0, -1.0, 2.0, -2.0], vec![1, 4]);
        let weight = make_buf(&[1.0, 1.0], vec![2]);
        let bias = make_buf(&[0.0, 0.0], vec![2]);
        let out = qk_norm(&input, &weight, &bias, 2, 2, 1e-5).unwrap();
        // Head 0: [1,-1], mean=0, std=1, normed=[1,-1]
        let v0 = out.data[0].to_f32();
        let v1 = out.data[1].to_f32();
        assert!((v0 - 1.0).abs() < 0.05, "got {}", v0);
        assert!((v1 + 1.0).abs() < 0.05, "got {}", v1);
    }

    #[test]
    fn expand_kv_heads_noop() {
        let input = make_buf(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = expand_kv_heads(&input, 2, 2, 2, 1);
        assert_eq!(out.data.len(), 4);
    }

    #[test]
    fn expand_kv_heads_mqa() {
        // 1 KV head, 2 Q heads, head_dim=2, 1 token: [a, b] -> [a, b, a, b]
        let input = make_buf(&[1.0, 2.0], vec![1, 2]);
        let out = expand_kv_heads(&input, 1, 2, 2, 1);
        assert_eq!(out.data.len(), 4);
        assert!((out.data[0].to_f32() - 1.0).abs() < 0.01);
        assert!((out.data[1].to_f32() - 2.0).abs() < 0.01);
        assert!((out.data[2].to_f32() - 1.0).abs() < 0.01);
        assert!((out.data[3].to_f32() - 2.0).abs() < 0.01);
    }

    #[test]
    fn cohere_forward_smoke() {
        let config = tiny_config();
        let weights = ModelWeights::default();
        let model = CohereForCausalLM::new(weights, &config).unwrap();

        let input = ModelInput {
            token_ids: vec![0, 1, 2],
            position_ids: vec![0, 1, 2],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0, 1, 2],
                context_lens: vec![3],
                block_tables: vec![vec![0]],
                max_context_len: 3,
            },
            is_prefill: true,
        };

        let cache = CacheEngine::new(config.num_layers, 1024);
        let attention = MockAttentionBackend;

        let logits = model.forward(&input, &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![3, config.vocab_size]);
    }

    #[test]
    fn cohere_factory_registration() {
        let config = tiny_config();
        let weights = ModelWeights::default();
        let model = super::super::create_model("CohereForCausalLM", weights, &config);
        assert!(model.is_ok(), "CohereForCausalLM should be registered in factory");
    }
}
