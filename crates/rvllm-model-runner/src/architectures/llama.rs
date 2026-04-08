//! LlamaForCausalLM architecture.

use half::f16;
use tracing::trace;

use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::mlp::MLP;
use crate::layers::norm::RMSNorm;
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::Architecture;

/// Llama-family causal language model.
pub struct LlamaForCausalLM {
    config: LlamaConfig,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<LlamaLayer>,
    norm_weight: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

struct LlamaConfig {
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
}

struct LlamaLayer {
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

impl LlamaForCausalLM {
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let cfg = LlamaConfig {
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            vocab_size: config.vocab_size,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
        };

        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, cfg.hidden_size]));

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{}", i);
            layers.push(LlamaLayer {
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
                q_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.q_proj.weight"),
                    &[cfg.num_heads * cfg.head_dim, cfg.hidden_size],
                ),
                k_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.k_proj.weight"),
                    &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
                ),
                v_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
                ),
                o_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    &[cfg.hidden_size, cfg.num_heads * cfg.head_dim],
                ),
                gate_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.mlp.gate_proj.weight"),
                    &[config.intermediate_size, cfg.hidden_size],
                ),
                up_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.mlp.up_proj.weight"),
                    &[config.intermediate_size, cfg.hidden_size],
                ),
                down_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.mlp.down_proj.weight"),
                    &[cfg.hidden_size, config.intermediate_size],
                ),
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
}

impl Architecture for LlamaForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let h = self.config.hidden_size;

        // Embedding lookup.
        let mut hidden = embed_tokens(&self.embed_tokens, &input.token_ids, h);

        // Transformer layers.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "llama layer forward");

            // Pre-attention norm.
            let normed =
                RMSNorm::forward(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            // QKV projections.
            let q = LinearLayer::forward(&normed, &layer.q_proj, None)?;
            let k = LinearLayer::forward(&normed, &layer.k_proj, None)?;
            let v = LinearLayer::forward(&normed, &layer.v_proj, None)?;

            // RoPE.
            let (q_rot, k_rot) = RotaryEmbedding::forward_with_base(
                &input.position_ids, &q, &k, self.config.head_dim, self.config.rope_theta,
            )?;

            // Attention.
            let attn_out =
                attention.forward(&q_rot, &k_rot, &v, &input.attention_metadata, layer_idx)?;

            // Output projection.
            let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, None)?;

            // Residual connection.
            add_inplace(&mut hidden, &attn_proj);

            // Post-attention norm + MLP.
            let normed2 = RMSNorm::forward(
                &hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;
            let mlp_out =
                MLP::forward(&normed2, &layer.gate_proj, &layer.up_proj, &layer.down_proj)?;
            add_inplace(&mut hidden, &mlp_out);
        }

        // Final norm.
        let normed_final = RMSNorm::forward(&hidden, &self.norm_weight, self.config.rms_norm_eps)?;

        // LM head: project to vocab (in f32 for logits).
        lm_head(
            &normed_final,
            &self.lm_head_weight,
            num_tokens,
            self.config.vocab_size,
        )
    }
}

// -- Helpers shared across architectures --

pub(crate) fn get_or_zeros(weights: &ModelWeights, name: &str, shape: &[usize]) -> GpuBuffer<f16> {
    weights
        .get_as_buffer(name)
        .unwrap_or_else(|_| GpuBuffer::zeros(shape))
}

pub(crate) fn embed_tokens(
    embed: &GpuBuffer<f16>,
    token_ids: &[u32],
    hidden: usize,
) -> GpuBuffer<f16> {
    let mut out = Vec::with_capacity(token_ids.len() * hidden);
    for &tid in token_ids {
        let start = tid as usize * hidden;
        let end = start + hidden;
        if end <= embed.len() {
            out.extend_from_slice(&embed.data[start..end]);
        } else {
            out.extend(std::iter::repeat(f16::ZERO).take(hidden));
        }
    }
    GpuBuffer::from_vec(out, vec![token_ids.len(), hidden])
}

pub(crate) fn add_inplace(a: &mut GpuBuffer<f16>, b: &GpuBuffer<f16>) {
    for (x, y) in a.data.iter_mut().zip(b.data.iter()) {
        *x = f16::from_f32(x.to_f32() + y.to_f32());
    }
}

pub(crate) fn lm_head(
    hidden: &GpuBuffer<f16>,
    weight: &GpuBuffer<f16>,
    num_tokens: usize,
    vocab_size: usize,
) -> Result<GpuBuffer<f32>> {
    let h = hidden.len() / num_tokens;
    let mut logits = Vec::with_capacity(num_tokens * vocab_size);

    for t in 0..num_tokens {
        let row_start = t * h;
        for v in 0..vocab_size {
            let w_start = v * h;
            let mut acc: f32 = 0.0;
            for k in 0..h {
                acc += hidden.data[row_start + k].to_f32() * weight.data[w_start + k].to_f32();
            }
            logits.push(acc);
        }
    }

    Ok(GpuBuffer::from_vec(logits, vec![num_tokens, vocab_size]))
}
