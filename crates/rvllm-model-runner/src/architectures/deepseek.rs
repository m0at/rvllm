//! DeepSeekV2ForCausalLM architecture.
//!
//! Implements DeepSeek-V2 with:
//! - Multi-head Latent Attention (MLA): low-rank KV compression via a latent
//!   bottleneck that reduces KV cache size by ~10x.
//! - Mixture-of-Experts (MoE) FFN layers with a shared expert alongside
//!   routed experts.
//! - RMSNorm, RoPE, and a standard embedding/LM-head.

use half::f16;
use tracing::trace;

use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::mlp::MLP;
use crate::layers::moe::{ExpertFFN, MoELayer};
use crate::layers::norm::RMSNorm;
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, embed_tokens, get_or_zeros, lm_head};
use super::Architecture;

/// DeepSeek-V2 model configuration.
struct DeepSeekConfig {
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    intermediate_size: usize,
    rms_norm_eps: f32,
    /// Latent dimension for MLA KV compression.
    kv_lora_rank: usize,
    /// Dimension for RoPE portion of queries in MLA.
    #[allow(dead_code)]
    qk_rope_head_dim: usize,
    /// Non-RoPE dimension for queries in MLA.
    #[allow(dead_code)]
    qk_nope_head_dim: usize,
    /// Number of routed experts.
    num_experts: usize,
    /// Number of experts activated per token.
    num_experts_per_tok: usize,
    /// Whether a shared expert is present.
    has_shared_expert: bool,
    /// MoE intermediate size (per expert).
    moe_intermediate_size: usize,
    /// Index of first MoE layer (earlier layers use dense MLP).
    first_moe_layer: usize,
}

/// Attention weights for MLA (Multi-head Latent Attention).
struct MLAWeights {
    /// Projects hidden -> compressed KV latent [kv_lora_rank].
    kv_a_proj: GpuBuffer<f16>,
    /// Projects latent -> full K,V [num_kv_heads * head_dim * 2].
    kv_b_proj: GpuBuffer<f16>,
    /// Query projection (full).
    q_proj: GpuBuffer<f16>,
    /// Output projection.
    o_proj: GpuBuffer<f16>,
    /// RMSNorm on the KV latent.
    kv_a_layernorm: GpuBuffer<f16>,
}

/// A single transformer layer, either dense MLP or MoE.
enum DeepSeekFFN {
    /// Standard dense MLP (used in early layers).
    Dense {
        gate_proj: GpuBuffer<f16>,
        up_proj: GpuBuffer<f16>,
        down_proj: GpuBuffer<f16>,
    },
    /// MoE with routed + shared experts.
    MoE(MoELayer),
}

struct DeepSeekLayer {
    input_layernorm: GpuBuffer<f16>,
    post_attention_layernorm: GpuBuffer<f16>,
    attn: MLAWeights,
    ffn: DeepSeekFFN,
}

/// DeepSeek-V2 causal language model.
pub struct DeepSeekV2ForCausalLM {
    config: DeepSeekConfig,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<DeepSeekLayer>,
    norm_weight: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

impl DeepSeekV2ForCausalLM {
    /// Construct from loaded weights and runner config.
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        // DeepSeek-V2 specific hyperparameters with sensible defaults.
        // In production these come from the HF config.json.
        let kv_lora_rank = 512;
        let qk_rope_head_dim = 64;
        let qk_nope_head_dim = config.head_dim.saturating_sub(qk_rope_head_dim);
        let num_experts = 64;
        let num_experts_per_tok = 6;
        let moe_intermediate_size = config.intermediate_size / 4; // per-expert
        let first_moe_layer = 1; // layer 0 is dense

        let cfg = DeepSeekConfig {
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            vocab_size: config.vocab_size,
            intermediate_size: config.intermediate_size,
            rms_norm_eps: 1e-6,
            kv_lora_rank,
            qk_rope_head_dim,
            qk_nope_head_dim,
            num_experts,
            num_experts_per_tok,
            has_shared_expert: true,
            moe_intermediate_size,
            first_moe_layer,
        };

        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, cfg.hidden_size]));

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{}", i);

            // MLA attention weights.
            let attn = MLAWeights {
                kv_a_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
                    &[cfg.kv_lora_rank, cfg.hidden_size],
                ),
                kv_b_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.kv_b_proj.weight"),
                    &[cfg.num_kv_heads * cfg.head_dim * 2, cfg.kv_lora_rank],
                ),
                q_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.q_proj.weight"),
                    &[cfg.num_heads * cfg.head_dim, cfg.hidden_size],
                ),
                o_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    &[cfg.hidden_size, cfg.num_heads * cfg.head_dim],
                ),
                kv_a_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.kv_a_layernorm.weight"),
                    &[cfg.kv_lora_rank],
                ),
            };

            // FFN: dense for early layers, MoE for later layers.
            let ffn = if i < cfg.first_moe_layer {
                DeepSeekFFN::Dense {
                    gate_proj: get_or_zeros(
                        &weights,
                        &format!("{p}.mlp.gate_proj.weight"),
                        &[cfg.intermediate_size, cfg.hidden_size],
                    ),
                    up_proj: get_or_zeros(
                        &weights,
                        &format!("{p}.mlp.up_proj.weight"),
                        &[cfg.intermediate_size, cfg.hidden_size],
                    ),
                    down_proj: get_or_zeros(
                        &weights,
                        &format!("{p}.mlp.down_proj.weight"),
                        &[cfg.hidden_size, cfg.intermediate_size],
                    ),
                }
            } else {
                // Load routed experts.
                let mut experts = Vec::with_capacity(cfg.num_experts);
                for e in 0..cfg.num_experts {
                    experts.push(ExpertFFN {
                        gate_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.mlp.experts.{e}.gate_proj.weight"),
                            &[cfg.moe_intermediate_size, cfg.hidden_size],
                        ),
                        up_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.mlp.experts.{e}.up_proj.weight"),
                            &[cfg.moe_intermediate_size, cfg.hidden_size],
                        ),
                        down_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.mlp.experts.{e}.down_proj.weight"),
                            &[cfg.hidden_size, cfg.moe_intermediate_size],
                        ),
                    });
                }

                // Router gate.
                let gate = get_or_zeros(
                    &weights,
                    &format!("{p}.mlp.gate.weight"),
                    &[cfg.num_experts, cfg.hidden_size],
                );

                // Shared expert (always-on, not routed).
                let shared_expert = if cfg.has_shared_expert {
                    Some(ExpertFFN {
                        gate_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.mlp.shared_experts.gate_proj.weight"),
                            &[cfg.moe_intermediate_size, cfg.hidden_size],
                        ),
                        up_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.mlp.shared_experts.up_proj.weight"),
                            &[cfg.moe_intermediate_size, cfg.hidden_size],
                        ),
                        down_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.mlp.shared_experts.down_proj.weight"),
                            &[cfg.hidden_size, cfg.moe_intermediate_size],
                        ),
                    })
                } else {
                    None
                };

                DeepSeekFFN::MoE(MoELayer {
                    gate,
                    experts,
                    top_k: cfg.num_experts_per_tok,
                    renormalize: true,
                    shared_expert,
                })
            };

            layers.push(DeepSeekLayer {
                input_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.input_layernorm.weight"),
                    &[cfg.hidden_size],
                ),
                post_attention_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[cfg.hidden_size],
                ),
                attn,
                ffn,
            });
        }

        let norm_weight = weights
            .get_as_buffer("model.norm.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.hidden_size]));

        let lm_head_weight = weights
            .get_as_buffer("lm_head.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, cfg.hidden_size]));

        Ok(Self {
            config: cfg,
            embed_tokens,
            layers,
            norm_weight,
            lm_head_weight,
        })
    }

    /// MLA attention: compress KV through a latent bottleneck, then expand.
    ///
    /// 1. hidden -> kv_a_proj -> [kv_lora_rank] latent
    /// 2. RMSNorm on latent
    /// 3. latent -> kv_b_proj -> split into K and V
    /// 4. Standard Q projection + RoPE on Q and K
    /// 5. Attention via backend
    /// 6. Output projection
    fn mla_attention(
        &self,
        hidden: &GpuBuffer<f16>,
        layer: &DeepSeekLayer,
        input: &ModelInput,
        attention: &dyn AttentionBackend,
        layer_idx: usize,
    ) -> Result<GpuBuffer<f16>> {
        let num_tokens = input.num_tokens();

        // Q projection (standard).
        let q = LinearLayer::forward(hidden, &layer.attn.q_proj, None)?;

        // KV compression: hidden -> latent -> normed latent -> expanded K,V.
        let kv_latent = LinearLayer::forward(hidden, &layer.attn.kv_a_proj, None)?;
        let kv_normed = RMSNorm::forward(
            &kv_latent,
            &layer.attn.kv_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        let kv_expanded = LinearLayer::forward(&kv_normed, &layer.attn.kv_b_proj, None)?;

        // Split expanded KV into K and V.
        // kv_expanded shape: [num_tokens, num_kv_heads * head_dim * 2]
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let mut k_data = Vec::with_capacity(num_tokens * kv_dim);
        let mut v_data = Vec::with_capacity(num_tokens * kv_dim);
        for t in 0..num_tokens {
            let base = t * kv_dim * 2;
            k_data.extend_from_slice(&kv_expanded.data[base..base + kv_dim]);
            v_data.extend_from_slice(&kv_expanded.data[base + kv_dim..base + kv_dim * 2]);
        }
        let k = GpuBuffer::from_vec(k_data, vec![num_tokens, kv_dim]);
        let v = GpuBuffer::from_vec(v_data, vec![num_tokens, kv_dim]);

        // RoPE on Q and K.
        let (q_rot, k_rot) =
            RotaryEmbedding::forward(&input.position_ids, &q, &k, self.config.head_dim)?;

        // Attention.
        let attn_out =
            attention.forward(&q_rot, &k_rot, &v, &input.attention_metadata, layer_idx)?;

        // Output projection.
        LinearLayer::forward(&attn_out, &layer.attn.o_proj, None)
    }
}

impl Architecture for DeepSeekV2ForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();

        // Embedding lookup.
        let mut hidden = embed_tokens(
            &self.embed_tokens,
            &input.token_ids,
            self.config.hidden_size,
        );

        // Transformer layers.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "deepseek layer forward");

            // Pre-attention RMSNorm.
            let normed =
                RMSNorm::forward(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            // MLA attention.
            let attn_proj = self.mla_attention(&normed, layer, input, attention, layer_idx)?;

            // Residual.
            add_inplace(&mut hidden, &attn_proj);

            // Post-attention RMSNorm.
            let normed2 = RMSNorm::forward(
                &hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;

            // FFN: dense or MoE depending on layer index.
            let ffn_out = match &layer.ffn {
                DeepSeekFFN::Dense {
                    gate_proj,
                    up_proj,
                    down_proj,
                } => MLP::forward(&normed2, gate_proj, up_proj, down_proj)?,
                DeepSeekFFN::MoE(moe) => moe.forward(&normed2)?,
            };

            // Residual.
            add_inplace(&mut hidden, &ffn_out);
        }

        // Final RMSNorm.
        let normed_final = RMSNorm::forward(&hidden, &self.norm_weight, self.config.rms_norm_eps)?;

        // LM head -> logits.
        lm_head(
            &normed_final,
            &self.lm_head_weight,
            num_tokens,
            self.config.vocab_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{AttentionMetadata, CacheEngine, MockAttentionBackend, ModelWeights};
    use crate::input::ModelInput;
    use crate::runner::ModelRunnerConfig;

    fn small_config() -> ModelRunnerConfig {
        ModelRunnerConfig {
            num_layers: 2,
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: 32,
            max_position: 128,
            dtype: rvllm_core::types::Dtype::Float16,
            rms_norm_eps: 1e-5, rope_theta: 10000.0,
            architecture: "DeepSeekV2ForCausalLM".into(),
            layer_types: vec![],
        }
    }

    fn dummy_input() -> ModelInput {
        ModelInput {
            token_ids: vec![1, 2, 3],
            position_ids: vec![0, 1, 2],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0, 1, 2],
                context_lens: vec![3],
                block_tables: vec![vec![0]],
                query_lens: vec![1],
                    max_context_len: 3,
            },
            is_prefill: true,
        }
    }

    #[test]
    fn deepseek_construction() {
        let config = small_config();
        let weights = ModelWeights::default();
        let model = DeepSeekV2ForCausalLM::new(weights, &config);
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn deepseek_forward_smoke() {
        let config = small_config();
        let weights = ModelWeights::default();
        let model = DeepSeekV2ForCausalLM::new(weights, &config).unwrap();
        let cache = CacheEngine::new(2, 1024);
        let attention = MockAttentionBackend;
        let input = dummy_input();

        let logits = model.forward(&input, &cache, &attention);
        assert!(logits.is_ok());
        let logits = logits.unwrap();
        assert_eq!(logits.shape, vec![3, 32]); // [num_tokens, vocab_size]
    }

    #[test]
    fn deepseek_single_token() {
        let config = small_config();
        let weights = ModelWeights::default();
        let model = DeepSeekV2ForCausalLM::new(weights, &config).unwrap();
        let cache = CacheEngine::new(2, 1024);
        let attention = MockAttentionBackend;

        let input = ModelInput {
            token_ids: vec![5],
            position_ids: vec![0],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0],
                context_lens: vec![1],
                block_tables: vec![vec![0]],
                query_lens: vec![1],
                    max_context_len: 1,
            },
            is_prefill: false,
        };

        let logits = model.forward(&input, &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![1, 32]);
    }

    #[test]
    fn deepseek_layer_types() {
        // Verify first layer is dense, subsequent are MoE.
        let config = small_config();
        let weights = ModelWeights::default();
        let model = DeepSeekV2ForCausalLM::new(weights, &config).unwrap();

        assert!(matches!(model.layers[0].ffn, DeepSeekFFN::Dense { .. }));
        assert!(matches!(model.layers[1].ffn, DeepSeekFFN::MoE(_)));
    }
}
