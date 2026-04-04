//! MistralForCausalLM architecture.
//!
//! Identical to Llama in structure (RMSNorm, GQA attention, gated MLP with SiLU).
//! Mistral differs primarily in sliding-window attention and different default
//! hyperparameters, but the forward-pass graph is the same.

use half::f16;
use tracing::trace;

use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::mlp::MLP;
use crate::layers::norm::RMSNorm;
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, embed_tokens, get_or_zeros, lm_head};
use super::Architecture;

pub struct MistralForCausalLM {
    hidden_size: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<MistralLayer>,
    norm_weight: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

struct MistralLayer {
    input_layernorm: GpuBuffer<f16>,
    post_attention_layernorm: GpuBuffer<f16>,
    q_proj: GpuBuffer<f16>,
    k_proj: GpuBuffer<f16>,
    v_proj: GpuBuffer<f16>,
    o_proj: GpuBuffer<f16>,
    gate_proj: GpuBuffer<f16>,
    up_proj: GpuBuffer<f16>,
    down_proj: GpuBuffer<f16>,
}

impl MistralForCausalLM {
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]));

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{}", i);
            layers.push(MistralLayer {
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
                q_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.q_proj.weight"),
                    &[config.num_heads * config.head_dim, config.hidden_size],
                ),
                k_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.k_proj.weight"),
                    &[config.num_kv_heads * config.head_dim, config.hidden_size],
                ),
                v_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[config.num_kv_heads * config.head_dim, config.hidden_size],
                ),
                o_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    &[config.hidden_size, config.num_heads * config.head_dim],
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

        let lm_head_weight = weights
            .get_as_buffer("lm_head.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]));

        Ok(Self {
            hidden_size: config.hidden_size,
            head_dim: config.head_dim,
            vocab_size: config.vocab_size,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            embed_tokens,
            layers,
            norm_weight,
            lm_head_weight,
        })
    }
}

impl Architecture for MistralForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let mut hidden = embed_tokens(&self.embed_tokens, &input.token_ids, self.hidden_size);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "mistral layer forward");

            let normed = RMSNorm::forward(&hidden, &layer.input_layernorm, self.rms_norm_eps)?;

            let q = LinearLayer::forward(&normed, &layer.q_proj, None)?;
            let k = LinearLayer::forward(&normed, &layer.k_proj, None)?;
            let v = LinearLayer::forward(&normed, &layer.v_proj, None)?;

            let (q_rot, k_rot) =
                RotaryEmbedding::forward_with_base(&input.position_ids, &q, &k, self.head_dim, self.rope_theta)?;

            let attn_out =
                attention.forward(&q_rot, &k_rot, &v, &input.attention_metadata, layer_idx)?;
            let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, None)?;
            add_inplace(&mut hidden, &attn_proj);

            let normed2 =
                RMSNorm::forward(&hidden, &layer.post_attention_layernorm, self.rms_norm_eps)?;
            let mlp_out =
                MLP::forward(&normed2, &layer.gate_proj, &layer.up_proj, &layer.down_proj)?;
            add_inplace(&mut hidden, &mlp_out);
        }

        let normed_final = RMSNorm::forward(&hidden, &self.norm_weight, self.rms_norm_eps)?;
        lm_head(
            &normed_final,
            &self.lm_head_weight,
            num_tokens,
            self.vocab_size,
        )
    }
}
