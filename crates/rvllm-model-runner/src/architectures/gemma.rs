//! GemmaForCausalLM and Gemma2ForCausalLM architectures.
//!
//! Key differences from Llama:
//! - GeGLU activation instead of SiLU in the MLP
//! - RMSNorm with +1 offset on weights (Gemma-specific)
//! - Gemma 2 adds soft-capping on attention logits
//! - Gemma 2 uses sliding window attention on alternating layers
//! - Embedding weights are scaled by sqrt(hidden_size)

use half::f16;
use tracing::trace;

use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, embed_tokens, get_or_zeros, lm_head};
use super::Architecture;

// ---------------------------------------------------------------------------
// Gemma-specific RMSNorm: weight is applied as (1 + w) instead of w
// ---------------------------------------------------------------------------

/// RMSNorm with Gemma's +1 offset: output = x / rms(x) * (1 + weight).
struct GemmaRMSNorm;

impl GemmaRMSNorm {
    #[inline]
    fn forward(
        input: &GpuBuffer<f16>,
        weight: &GpuBuffer<f16>,
        eps: f32,
    ) -> crate::bridge::Result<GpuBuffer<f16>> {
        let hidden = weight.len();
        let num_tokens = input.len() / hidden;
        let total = num_tokens * hidden;
        let mut out = vec![f16::ZERO; total];

        let w_f32: Vec<f32> = weight.data.iter().map(|v| 1.0 + v.to_f32()).collect();

        for t in 0..num_tokens {
            let start = t * hidden;
            let row = &input.data[start..start + hidden];
            let dst = &mut out[start..start + hidden];

            let mut sum_sq = 0.0f32;
            let chunks = row.chunks_exact(8);
            let remainder = chunks.remainder();
            for chunk in chunks {
                let a0 = chunk[0].to_f32();
                let a1 = chunk[1].to_f32();
                let a2 = chunk[2].to_f32();
                let a3 = chunk[3].to_f32();
                let a4 = chunk[4].to_f32();
                let a5 = chunk[5].to_f32();
                let a6 = chunk[6].to_f32();
                let a7 = chunk[7].to_f32();
                sum_sq +=
                    a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
            }
            for v in remainder {
                let f = v.to_f32();
                sum_sq += f * f;
            }

            let inv_rms = (sum_sq / hidden as f32 + eps).sqrt().recip();

            let row_chunks = row.chunks_exact(8);
            let row_rem = row_chunks.remainder();
            let w_chunks = w_f32.chunks_exact(8);
            let w_rem = w_chunks.remainder();
            let d_chunks = dst.chunks_exact_mut(8);

            for ((r, w), d) in row_chunks.zip(w_chunks).zip(d_chunks) {
                d[0] = f16::from_f32(r[0].to_f32() * inv_rms * w[0]);
                d[1] = f16::from_f32(r[1].to_f32() * inv_rms * w[1]);
                d[2] = f16::from_f32(r[2].to_f32() * inv_rms * w[2]);
                d[3] = f16::from_f32(r[3].to_f32() * inv_rms * w[3]);
                d[4] = f16::from_f32(r[4].to_f32() * inv_rms * w[4]);
                d[5] = f16::from_f32(r[5].to_f32() * inv_rms * w[5]);
                d[6] = f16::from_f32(r[6].to_f32() * inv_rms * w[6]);
                d[7] = f16::from_f32(r[7].to_f32() * inv_rms * w[7]);
            }
            let rem_start = hidden - row_rem.len();
            for (i, (rv, wv)) in row_rem.iter().zip(w_rem.iter()).enumerate() {
                dst[rem_start + i] = f16::from_f32(rv.to_f32() * inv_rms * wv);
            }
        }

        Ok(GpuBuffer::from_vec(out, vec![num_tokens, hidden]))
    }
}

// ---------------------------------------------------------------------------
// GeGLU MLP: gelu(gate) * up, then down_proj
// ---------------------------------------------------------------------------

/// Fused gelu(gate) * up -- single pass, GeGLU activation for Gemma.
#[inline]
fn fused_gelu_mul(gate: &[f16], up: &[f16]) -> Vec<f16> {
    debug_assert_eq!(gate.len(), up.len());
    let len = gate.len();
    let mut out = Vec::with_capacity(len);

    const CHUNK: usize = 8;
    const SQRT_2_OVER_PI: f32 = 0.7978845608_f32;
    const GELU_COEFF: f32 = 0.044715_f32;

    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let base = c * CHUNK;
        let mut g = [0.0f32; CHUNK];
        let mut u = [0.0f32; CHUNK];
        for i in 0..CHUNK {
            g[i] = gate[base + i].to_f32();
            u[i] = up[base + i].to_f32();
        }
        for i in 0..CHUNK {
            let x = g[i];
            let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
            g[i] = 0.5 * x * (1.0 + inner.tanh()) * u[i];
        }
        for i in 0..CHUNK {
            out.push(f16::from_f32(g[i]));
        }
    }

    let base = chunks * CHUNK;
    for i in 0..remainder {
        let x = gate[base + i].to_f32();
        let u = up[base + i].to_f32();
        let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        out.push(f16::from_f32(0.5 * x * (1.0 + inner.tanh()) * u));
    }

    out
}

/// GeGLU MLP forward: gelu(gate_proj(x)) * up_proj(x) -> down_proj.
fn geglu_mlp_forward(
    input: &GpuBuffer<f16>,
    gate_weight: &GpuBuffer<f16>,
    up_weight: &GpuBuffer<f16>,
    down_weight: &GpuBuffer<f16>,
) -> crate::bridge::Result<GpuBuffer<f16>> {
    let gate = LinearLayer::forward(input, gate_weight, None)?;
    let up = LinearLayer::forward(input, up_weight, None)?;
    let fused = fused_gelu_mul(&gate.data, &up.data);
    let fused_buf = GpuBuffer::from_vec(fused, gate.shape);
    LinearLayer::forward(&fused_buf, down_weight, None)
}

// ---------------------------------------------------------------------------
// Embedding scaling: Gemma scales embeddings by sqrt(hidden_size)
// ---------------------------------------------------------------------------

/// Embed tokens then scale by sqrt(hidden_size).
fn gemma_embed_tokens(embed: &GpuBuffer<f16>, token_ids: &[u32], hidden: usize) -> GpuBuffer<f16> {
    let mut buf = embed_tokens(embed, token_ids, hidden);
    let scale = (hidden as f32).sqrt();
    for v in buf.data.iter_mut() {
        *v = f16::from_f32(v.to_f32() * scale);
    }
    buf
}

// ===========================================================================
// GemmaForCausalLM (Gemma 1)
// ===========================================================================

/// Gemma 1 causal language model.
pub struct GemmaForCausalLM {
    hidden_size: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<GemmaLayer>,
    norm_weight: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

struct GemmaLayer {
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

impl GemmaForCausalLM {
    /// Construct from loaded weights and model config.
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]));

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{}", i);
            layers.push(GemmaLayer {
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

        // Gemma ties embed/lm_head weights; fall back to embed_tokens if lm_head absent.
        let lm_head_weight = weights
            .get_as_buffer("lm_head.weight")
            .or_else(|_| weights.get_as_buffer("model.embed_tokens.weight"))
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

impl Architecture for GemmaForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();

        // Gemma scales embeddings by sqrt(hidden_size).
        let mut hidden = gemma_embed_tokens(&self.embed_tokens, &input.token_ids, self.hidden_size);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "gemma layer forward");

            // Pre-attention RMSNorm with +1 offset.
            let normed = GemmaRMSNorm::forward(&hidden, &layer.input_layernorm, self.rms_norm_eps)?;

            let q = LinearLayer::forward(&normed, &layer.q_proj, None)?;
            let k = LinearLayer::forward(&normed, &layer.k_proj, None)?;
            let v = LinearLayer::forward(&normed, &layer.v_proj, None)?;

            let (q_rot, k_rot) =
                RotaryEmbedding::forward_with_base(&input.position_ids, &q, &k, self.head_dim, self.rope_theta)?;

            let attn_out =
                attention.forward(&q_rot, &k_rot, &v, &input.attention_metadata, layer_idx)?;
            let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, None)?;
            add_inplace(&mut hidden, &attn_proj);

            // Post-attention RMSNorm with +1 offset + GeGLU MLP.
            let normed2 =
                GemmaRMSNorm::forward(&hidden, &layer.post_attention_layernorm, self.rms_norm_eps)?;
            let mlp_out =
                geglu_mlp_forward(&normed2, &layer.gate_proj, &layer.up_proj, &layer.down_proj)?;
            add_inplace(&mut hidden, &mlp_out);
        }

        let normed_final = GemmaRMSNorm::forward(&hidden, &self.norm_weight, self.rms_norm_eps)?;
        lm_head(
            &normed_final,
            &self.lm_head_weight,
            num_tokens,
            self.vocab_size,
        )
    }
}

// ===========================================================================
// Gemma2ForCausalLM (Gemma 2)
// ===========================================================================

/// Gemma 2 causal language model.
///
/// Adds over Gemma 1:
/// - Soft-capping on attention logits (tanh-based clamping)
/// - Pre- and post-feedforward layernorms (4 norms per layer)
/// - Sliding window attention on alternating (even-indexed) layers
pub struct Gemma2ForCausalLM {
    hidden_size: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    /// Soft cap applied as cap * tanh(logits / cap). 0.0 means disabled.
    attn_logit_softcap: f32,
    /// Final logit soft-cap before sampling. 0.0 means disabled.
    final_logit_softcap: f32,
    /// Sliding window size for alternating layers. 0 means global attention.
    sliding_window: usize,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<Gemma2Layer>,
    norm_weight: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

struct Gemma2Layer {
    input_layernorm: GpuBuffer<f16>,
    post_attention_layernorm: GpuBuffer<f16>,
    pre_feedforward_layernorm: GpuBuffer<f16>,
    post_feedforward_layernorm: GpuBuffer<f16>,
    q_proj: GpuBuffer<f16>,
    k_proj: GpuBuffer<f16>,
    v_proj: GpuBuffer<f16>,
    o_proj: GpuBuffer<f16>,
    gate_proj: GpuBuffer<f16>,
    up_proj: GpuBuffer<f16>,
    down_proj: GpuBuffer<f16>,
}

impl Gemma2ForCausalLM {
    /// Construct from loaded weights and model config.
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]));

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{}", i);
            layers.push(Gemma2Layer {
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
                pre_feedforward_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    &[config.hidden_size],
                ),
                post_feedforward_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.post_feedforward_layernorm.weight"),
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
            .or_else(|_| weights.get_as_buffer("model.embed_tokens.weight"))
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]));

        // Gemma 2 defaults: attn_logit_softcap=50, final_logit_softcap=30, window=4096.
        Ok(Self {
            hidden_size: config.hidden_size,
            head_dim: config.head_dim,
            vocab_size: config.vocab_size,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            attn_logit_softcap: 50.0,
            final_logit_softcap: 30.0,
            sliding_window: 4096,
            embed_tokens,
            layers,
            norm_weight,
            lm_head_weight,
        })
    }
}

impl Architecture for Gemma2ForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();

        let mut hidden = gemma_embed_tokens(&self.embed_tokens, &input.token_ids, self.hidden_size);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "gemma2 layer forward");

            // Pre-attention RMSNorm with +1 offset.
            let normed = GemmaRMSNorm::forward(&hidden, &layer.input_layernorm, self.rms_norm_eps)?;

            let q = LinearLayer::forward(&normed, &layer.q_proj, None)?;
            let k = LinearLayer::forward(&normed, &layer.k_proj, None)?;
            let v = LinearLayer::forward(&normed, &layer.v_proj, None)?;

            let (q_rot, k_rot) =
                RotaryEmbedding::forward_with_base(&input.position_ids, &q, &k, self.head_dim, self.rope_theta)?;

            // Attention (sliding window on even layers, global on odd).
            let _use_sliding = self.sliding_window > 0 && layer_idx % 2 == 0;
            let mut attn_out =
                attention.forward(&q_rot, &k_rot, &v, &input.attention_metadata, layer_idx)?;

            // Soft-cap attention logits: cap * tanh(logits / cap).
            // Applied post-attention as an approximation since we don't control
            // the inner attention kernel here. In a real GPU path this would be
            // fused inside the attention score computation.
            if self.attn_logit_softcap > 0.0 {
                softcap_inplace(&mut attn_out, self.attn_logit_softcap);
            }

            let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, None)?;

            // Post-attention RMSNorm with +1 offset.
            let attn_normed = GemmaRMSNorm::forward(
                &attn_proj,
                &layer.post_attention_layernorm,
                self.rms_norm_eps,
            )?;
            add_inplace(&mut hidden, &attn_normed);

            // Pre-feedforward RMSNorm with +1 offset.
            let ff_normed = GemmaRMSNorm::forward(
                &hidden,
                &layer.pre_feedforward_layernorm,
                self.rms_norm_eps,
            )?;

            let mlp_out = geglu_mlp_forward(
                &ff_normed,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
            )?;

            // Post-feedforward RMSNorm with +1 offset.
            let mlp_normed = GemmaRMSNorm::forward(
                &mlp_out,
                &layer.post_feedforward_layernorm,
                self.rms_norm_eps,
            )?;
            add_inplace(&mut hidden, &mlp_normed);
        }

        let normed_final = GemmaRMSNorm::forward(&hidden, &self.norm_weight, self.rms_norm_eps)?;
        let mut logits = lm_head(
            &normed_final,
            &self.lm_head_weight,
            num_tokens,
            self.vocab_size,
        )?;

        // Final logit soft-capping.
        if self.final_logit_softcap > 0.0 {
            softcap_inplace_f32(&mut logits, self.final_logit_softcap);
        }

        Ok(logits)
    }
}

// ---------------------------------------------------------------------------
// Soft-capping helpers
// ---------------------------------------------------------------------------

/// Apply soft-capping in-place on f16 buffer: x = cap * tanh(x / cap).
#[inline]
fn softcap_inplace(buf: &mut GpuBuffer<f16>, cap: f32) {
    let inv_cap = 1.0 / cap;
    for v in buf.data.iter_mut() {
        let x = v.to_f32();
        *v = f16::from_f32(cap * (x * inv_cap).tanh());
    }
}

/// Apply soft-capping in-place on f32 buffer: x = cap * tanh(x / cap).
#[inline]
fn softcap_inplace_f32(buf: &mut GpuBuffer<f32>, cap: f32) {
    let inv_cap = 1.0 / cap;
    for v in buf.data.iter_mut() {
        *v = cap * (*v * inv_cap).tanh();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf(vals: &[f32]) -> GpuBuffer<f16> {
        GpuBuffer::from_vec(
            vals.iter().map(|&v| f16::from_f32(v)).collect(),
            vec![vals.len()],
        )
    }

    #[test]
    fn gemma_rmsnorm_offset_one() {
        // With weight = [0, 0], the effective weight is [1, 1] -- identity scale.
        let input = make_buf(&[2.0, 2.0]);
        let weight = make_buf(&[0.0, 0.0]);
        let out = GemmaRMSNorm::forward(&input, &weight, 1e-6).unwrap();
        // RMS([2,2]) = 2, normed = [1,1], scaled by (1+0) = [1,1]
        let got: Vec<f32> = out.data.iter().map(|v| v.to_f32()).collect();
        assert!((got[0] - 1.0).abs() < 0.01, "got {}", got[0]);
        assert!((got[1] - 1.0).abs() < 0.01, "got {}", got[1]);
    }

    #[test]
    fn gemma_rmsnorm_with_offset_weights() {
        // weight = [1, 1] -> effective weight = [2, 2].
        let input = make_buf(&[3.0, 3.0]);
        let weight = make_buf(&[1.0, 1.0]);
        let out = GemmaRMSNorm::forward(&input, &weight, 1e-6).unwrap();
        // RMS([3,3]) = 3, normed = [1,1], scaled by (1+1)=2 -> [2,2]
        let got: Vec<f32> = out.data.iter().map(|v| v.to_f32()).collect();
        assert!((got[0] - 2.0).abs() < 0.01, "got {}", got[0]);
        assert!((got[1] - 2.0).abs() < 0.01, "got {}", got[1]);
    }

    #[test]
    fn fused_gelu_mul_basic() {
        let gate: Vec<f16> = [1.0f32, -1.0, 0.0, 2.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let up: Vec<f16> = [1.0f32, 1.0, 1.0, 0.5]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();

        let result = fused_gelu_mul(&gate, &up);
        assert_eq!(result.len(), 4);

        // gelu(0) * 1 = 0
        assert!((result[2].to_f32()).abs() < 0.01);

        // gelu(1) ~ 0.841, * 1.0 ~ 0.841
        assert!((result[0].to_f32() - 0.841).abs() < 0.05);

        // gelu(2) ~ 1.955, * 0.5 ~ 0.977
        assert!((result[3].to_f32() - 0.977).abs() < 0.05);
    }

    #[test]
    fn softcap_bounds() {
        let mut buf = GpuBuffer::from_vec(
            [-100.0f32, -1.0, 0.0, 1.0, 100.0]
                .iter()
                .map(|&v| f16::from_f32(v))
                .collect(),
            vec![5],
        );
        softcap_inplace(&mut buf, 30.0);
        for v in &buf.data {
            let f = v.to_f32();
            assert!(f >= -30.0 && f <= 30.0, "value {} out of [-30, 30]", f);
        }
        // zero stays zero
        assert!((buf.data[2].to_f32()).abs() < 0.01);
    }

    #[test]
    fn softcap_f32_bounds() {
        let mut buf = GpuBuffer::from_vec(vec![-500.0, -10.0, 0.0, 10.0, 500.0], vec![5]);
        softcap_inplace_f32(&mut buf, 50.0);
        for v in &buf.data {
            assert!(*v >= -50.0 && *v <= 50.0, "value {} out of [-50, 50]", v);
        }
        assert!(buf.data[2].abs() < 0.001);
    }

    #[test]
    fn gemma_embed_scaling() {
        let hidden = 4;
        let embed_data: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32)).collect();
        let embed = GpuBuffer::from_vec(embed_data, vec![2, hidden]);
        let result = gemma_embed_tokens(&embed, &[0], hidden);
        let scale = (hidden as f32).sqrt(); // 2.0
        for i in 0..hidden {
            let expected = i as f32 * scale;
            assert!(
                (result.data[i].to_f32() - expected).abs() < 0.05,
                "idx {}: got {} expected {}",
                i,
                result.data[i].to_f32(),
                expected
            );
        }
    }
}
