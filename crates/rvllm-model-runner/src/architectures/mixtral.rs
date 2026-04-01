//! MixtralForCausalLM architecture.
//!
//! Mixtral uses 8 experts with top-2 routing via a sparse MoE FFN replacing
//! the dense MLP in each transformer layer. Attention is standard GQA with
//! RoPE and sliding window support. The architecture is otherwise identical
//! to Mistral: RMSNorm, GQA, residual connections.

use half::f16;
use tracing::trace;

use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::moe::{ExpertFFN, MoELayer};
use crate::layers::norm::RMSNorm;
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, embed_tokens, get_or_zeros, lm_head};
use super::Architecture;

/// Default number of experts in Mixtral.
const NUM_EXPERTS: usize = 8;

/// Default number of experts activated per token.
const TOP_K: usize = 2;

/// Mixtral transformer layer: GQA attention + sparse MoE FFN.
struct MixtralLayer {
    input_layernorm: GpuBuffer<f16>,
    post_attention_layernorm: GpuBuffer<f16>,
    q_proj: GpuBuffer<f16>,
    k_proj: GpuBuffer<f16>,
    v_proj: GpuBuffer<f16>,
    o_proj: GpuBuffer<f16>,
    moe: MoELayer,
}

/// MixtralForCausalLM: sparse MoE transformer with sliding window attention.
pub struct MixtralForCausalLM {
    hidden_size: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<MixtralLayer>,
    norm_weight: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

impl MixtralForCausalLM {
    /// Construct from loaded weights and model config.
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]));

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{}", i);

            // Attention weights.
            let q_proj = get_or_zeros(
                &weights,
                &format!("{p}.self_attn.q_proj.weight"),
                &[config.num_heads * config.head_dim, config.hidden_size],
            );
            let k_proj = get_or_zeros(
                &weights,
                &format!("{p}.self_attn.k_proj.weight"),
                &[config.num_kv_heads * config.head_dim, config.hidden_size],
            );
            let v_proj = get_or_zeros(
                &weights,
                &format!("{p}.self_attn.v_proj.weight"),
                &[config.num_kv_heads * config.head_dim, config.hidden_size],
            );
            let o_proj = get_or_zeros(
                &weights,
                &format!("{p}.self_attn.o_proj.weight"),
                &[config.hidden_size, config.num_heads * config.head_dim],
            );

            // MoE router gate.
            let gate = get_or_zeros(
                &weights,
                &format!("{p}.block_sparse_moe.gate.weight"),
                &[NUM_EXPERTS, config.hidden_size],
            );

            // Load each expert's FFN weights.
            let mut experts = Vec::with_capacity(NUM_EXPERTS);
            for e in 0..NUM_EXPERTS {
                let ep = format!("{p}.block_sparse_moe.experts.{e}");
                experts.push(ExpertFFN {
                    gate_proj: get_or_zeros(
                        &weights,
                        &format!("{ep}.w1.weight"),
                        &[config.intermediate_size, config.hidden_size],
                    ),
                    up_proj: get_or_zeros(
                        &weights,
                        &format!("{ep}.w3.weight"),
                        &[config.intermediate_size, config.hidden_size],
                    ),
                    down_proj: get_or_zeros(
                        &weights,
                        &format!("{ep}.w2.weight"),
                        &[config.hidden_size, config.intermediate_size],
                    ),
                });
            }

            layers.push(MixtralLayer {
                input_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.input_layernorm.weight"),
                    &[config.hidden_size],
                ),
                post_attention_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[config.hidden_size],
                ),
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                moe: MoELayer {
                    gate,
                    experts,
                    top_k: TOP_K,
                    renormalize: true,
                    shared_expert: None,
                },
            });
        }

        let norm_weight = weights
            .get_as_buffer("model.norm.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.hidden_size]));

        let lm_head_weight = weights
            .get_as_buffer("lm_head.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]));

        Ok(Self {
            hidden_size: config.hidden_size,
            head_dim: config.head_dim,
            vocab_size: config.vocab_size,
            rms_norm_eps: 1e-5,
            embed_tokens,
            layers,
            norm_weight,
            lm_head_weight,
        })
    }
}

impl Architecture for MixtralForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let mut hidden = embed_tokens(&self.embed_tokens, &input.token_ids, self.hidden_size);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "mixtral layer forward");

            // Pre-attention RMSNorm.
            let normed = RMSNorm::forward(&hidden, &layer.input_layernorm, self.rms_norm_eps)?;

            // QKV projections.
            let q = LinearLayer::forward(&normed, &layer.q_proj, None)?;
            let k = LinearLayer::forward(&normed, &layer.k_proj, None)?;
            let v = LinearLayer::forward(&normed, &layer.v_proj, None)?;

            // RoPE.
            let (q_rot, k_rot) =
                RotaryEmbedding::forward(&input.position_ids, &q, &k, self.head_dim)?;

            // Attention (sliding window handled by the backend).
            let attn_out =
                attention.forward(&q_rot, &k_rot, &v, &input.attention_metadata, layer_idx)?;

            // Output projection + residual.
            let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, None)?;
            add_inplace(&mut hidden, &attn_proj);

            // Post-attention RMSNorm.
            let normed2 =
                RMSNorm::forward(&hidden, &layer.post_attention_layernorm, self.rms_norm_eps)?;

            // Sparse MoE FFN replaces the dense MLP.
            let moe_out = layer.moe.forward(&normed2)?;
            add_inplace(&mut hidden, &moe_out);
        }

        // Final RMSNorm + LM head.
        let normed_final = RMSNorm::forward(&hidden, &self.norm_weight, self.rms_norm_eps)?;
        lm_head(
            &normed_final,
            &self.lm_head_weight,
            num_tokens,
            self.vocab_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{AttentionMetadata, CacheEngine, MockAttentionBackend, ModelWeights};
    use crate::input::ModelInput;
    use crate::runner::ModelRunnerConfig;

    fn test_config() -> ModelRunnerConfig {
        ModelRunnerConfig {
            num_layers: 1,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            intermediate_size: 4,
            vocab_size: 8,
            max_position: 32,
            dtype: rvllm_core::types::Dtype::Float16,
            rms_norm_eps: 1e-5, rope_theta: 10000.0,
            architecture: "MixtralForCausalLM".into(),
            layer_types: vec![],
        }
    }

    fn test_input() -> ModelInput {
        ModelInput {
            token_ids: vec![1, 2],
            position_ids: vec![0, 1],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0, 1],
                context_lens: vec![2],
                block_tables: vec![vec![0]],
                query_lens: vec![1],
                    max_context_len: 2,
            },
            is_prefill: true,
        }
    }

    #[test]
    fn mixtral_forward_smoke() {
        let config = test_config();
        let weights = ModelWeights::default();
        let model = MixtralForCausalLM::new(weights, &config).unwrap();
        let input = test_input();
        let cache = CacheEngine::new(1, 64);
        let attention = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![2, 8]);
    }

    #[test]
    fn mixtral_via_factory() {
        let config = test_config();
        let weights = ModelWeights::default();
        let model = super::super::create_model("MixtralForCausalLM", weights, &config).unwrap();
        let input = test_input();
        let cache = CacheEngine::new(1, 64);
        let attention = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![2, 8]);
    }

    #[test]
    fn mixtral_multi_layer() {
        let mut config = test_config();
        config.num_layers = 3;
        let weights = ModelWeights::default();
        let model = MixtralForCausalLM::new(weights, &config).unwrap();
        let input = test_input();
        let cache = CacheEngine::new(3, 64);
        let attention = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![2, 8]);
    }

    #[test]
    fn mixtral_single_token() {
        let config = test_config();
        let weights = ModelWeights::default();
        let model = MixtralForCausalLM::new(weights, &config).unwrap();
        let input = ModelInput {
            token_ids: vec![0],
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
        let cache = CacheEngine::new(1, 64);
        let attention = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![1, 8]);
    }
}
