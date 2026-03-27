//! PhiForCausalLM architecture (Phi-2, Phi-3, Phi-3.5).
//!
//! Key differences from Llama:
//! - Partial rotary embedding: only the first `partial_rotary_factor` fraction
//!   of head_dim gets RoPE; the rest passes through unchanged.
//! - QK layernorm variant (Phi-3): optional layernorm on Q and K after projection.
//! - Parallel attention + MLP (Phi-2 style): attention and MLP run on the same
//!   normed input, then both residuals are added simultaneously.
//! - Phi-3 uses sequential attention then MLP (like Llama).

use half::f16;
use tracing::trace;

use crate::bridge::{
    AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result,
};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::mlp::MLP;
use crate::layers::norm::{LayerNorm, RMSNorm};
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, embed_tokens, get_or_zeros, lm_head};
use super::Architecture;

/// Phi variant selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhiVariant {
    /// Phi-2: parallel attention + MLP, LayerNorm, partial RoPE.
    Phi2,
    /// Phi-3 / Phi-3.5: sequential attention then MLP, RMSNorm, QK layernorm,
    /// partial RoPE.
    Phi3,
}

/// Configuration specific to Phi models.
struct PhiConfig {
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    /// Fraction of head_dim that receives rotary embedding (e.g. 0.5 for Phi-2).
    #[allow(dead_code)]
    partial_rotary_factor: f32,
    /// Number of dimensions in head_dim that get RoPE.
    rotary_dim: usize,
    /// Norm epsilon.
    norm_eps: f32,
    #[allow(dead_code)]
    variant: PhiVariant,
}

/// Per-layer weights for Phi-2 (LayerNorm + optional QKV bias).
struct Phi2Layer {
    ln_weight: GpuBuffer<f16>,
    ln_bias: GpuBuffer<f16>,
    q_proj: GpuBuffer<f16>,
    q_bias: Option<GpuBuffer<f16>>,
    k_proj: GpuBuffer<f16>,
    k_bias: Option<GpuBuffer<f16>>,
    v_proj: GpuBuffer<f16>,
    v_bias: Option<GpuBuffer<f16>>,
    o_proj: GpuBuffer<f16>,
    o_bias: Option<GpuBuffer<f16>>,
    gate_proj: GpuBuffer<f16>,
    up_proj: GpuBuffer<f16>,
    down_proj: GpuBuffer<f16>,
}

/// Per-layer weights for Phi-3 (RMSNorm + QK layernorm).
struct Phi3Layer {
    input_layernorm: GpuBuffer<f16>,
    post_attention_layernorm: GpuBuffer<f16>,
    q_proj: GpuBuffer<f16>,
    k_proj: GpuBuffer<f16>,
    v_proj: GpuBuffer<f16>,
    o_proj: GpuBuffer<f16>,
    /// Optional QK layernorm weights + biases (Phi-3 specific).
    q_layernorm_weight: Option<GpuBuffer<f16>>,
    q_layernorm_bias: Option<GpuBuffer<f16>>,
    k_layernorm_weight: Option<GpuBuffer<f16>>,
    k_layernorm_bias: Option<GpuBuffer<f16>>,
    gate_up_proj: GpuBuffer<f16>,
    down_proj: GpuBuffer<f16>,
}

/// Unified layer enum.
enum PhiLayer {
    V2(Phi2Layer),
    V3(Phi3Layer),
}

/// Phi-family causal language model.
pub struct PhiForCausalLM {
    config: PhiConfig,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<PhiLayer>,
    /// Final norm weight.
    final_norm_weight: GpuBuffer<f16>,
    /// Final norm bias (Phi-2 uses LayerNorm with bias).
    final_norm_bias: Option<GpuBuffer<f16>>,
    lm_head_weight: GpuBuffer<f16>,
    lm_head_bias: Option<GpuBuffer<f16>>,
}

impl PhiForCausalLM {
    /// Construct a PhiForCausalLM from loaded weights.
    ///
    /// Detects Phi-2 vs Phi-3 based on weight name presence: if
    /// `model.layers.0.input_layernorm.weight` exists, it is Phi-3 style;
    /// otherwise it is Phi-2 style with `model.layers.0.ln.weight`.
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let is_phi3 = weights
            .get_as_buffer("model.layers.0.input_layernorm.weight")
            .is_ok();

        let variant = if is_phi3 {
            PhiVariant::Phi3
        } else {
            PhiVariant::Phi2
        };

        // Phi-2 default partial_rotary_factor = 0.5, Phi-3 = 0.5 (varies by
        // model size but 0.5 is the common default).
        let partial_rotary_factor = 0.5_f32;
        let rotary_dim = (config.head_dim as f32 * partial_rotary_factor) as usize;

        let norm_eps = match variant {
            PhiVariant::Phi2 => 1e-5,
            PhiVariant::Phi3 => 1e-5,
        };

        let cfg = PhiConfig {
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            partial_rotary_factor,
            rotary_dim,
            norm_eps,
            variant,
        };

        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, cfg.hidden_size]));

        let mut layers = Vec::with_capacity(cfg.num_layers);
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{}", i);
            match variant {
                PhiVariant::Phi2 => {
                    layers.push(PhiLayer::V2(Phi2Layer {
                        ln_weight: get_or_zeros(
                            &weights,
                            &format!("{p}.ln.weight"),
                            &[cfg.hidden_size],
                        ),
                        ln_bias: get_or_zeros(
                            &weights,
                            &format!("{p}.ln.bias"),
                            &[cfg.hidden_size],
                        ),
                        q_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.self_attn.q_proj.weight"),
                            &[q_dim, cfg.hidden_size],
                        ),
                        q_bias: weights
                            .get_as_buffer(&format!("{p}.self_attn.q_proj.bias"))
                            .ok(),
                        k_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.self_attn.k_proj.weight"),
                            &[kv_dim, cfg.hidden_size],
                        ),
                        k_bias: weights
                            .get_as_buffer(&format!("{p}.self_attn.k_proj.bias"))
                            .ok(),
                        v_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.self_attn.v_proj.weight"),
                            &[kv_dim, cfg.hidden_size],
                        ),
                        v_bias: weights
                            .get_as_buffer(&format!("{p}.self_attn.v_proj.bias"))
                            .ok(),
                        o_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.self_attn.o_proj.weight"),
                            &[cfg.hidden_size, q_dim],
                        ),
                        o_bias: weights
                            .get_as_buffer(&format!("{p}.self_attn.o_proj.bias"))
                            .ok(),
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
                    }));
                }
                PhiVariant::Phi3 => {
                    // Phi-3 uses a fused gate_up_proj (2 * intermediate_size, hidden_size).
                    let gate_up_size = 2 * cfg.intermediate_size;
                    layers.push(PhiLayer::V3(Phi3Layer {
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
                            &[q_dim, cfg.hidden_size],
                        ),
                        k_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.self_attn.k_proj.weight"),
                            &[kv_dim, cfg.hidden_size],
                        ),
                        v_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.self_attn.v_proj.weight"),
                            &[kv_dim, cfg.hidden_size],
                        ),
                        o_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.self_attn.o_proj.weight"),
                            &[cfg.hidden_size, q_dim],
                        ),
                        q_layernorm_weight: weights
                            .get_as_buffer(&format!("{p}.self_attn.q_layernorm.weight"))
                            .ok(),
                        q_layernorm_bias: weights
                            .get_as_buffer(&format!("{p}.self_attn.q_layernorm.bias"))
                            .ok(),
                        k_layernorm_weight: weights
                            .get_as_buffer(&format!("{p}.self_attn.k_layernorm.weight"))
                            .ok(),
                        k_layernorm_bias: weights
                            .get_as_buffer(&format!("{p}.self_attn.k_layernorm.bias"))
                            .ok(),
                        gate_up_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.mlp.gate_up_proj.weight"),
                            &[gate_up_size, cfg.hidden_size],
                        ),
                        down_proj: get_or_zeros(
                            &weights,
                            &format!("{p}.mlp.down_proj.weight"),
                            &[cfg.hidden_size, cfg.intermediate_size],
                        ),
                    }));
                }
            }
        }

        let final_norm_weight = weights
            .get_as_buffer("model.norm.weight")
            .or_else(|_| weights.get_as_buffer("model.final_layernorm.weight"))
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.hidden_size]));

        let final_norm_bias = weights
            .get_as_buffer("model.norm.bias")
            .or_else(|_| weights.get_as_buffer("model.final_layernorm.bias"))
            .ok();

        let lm_head_weight = weights
            .get_as_buffer("lm_head.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, cfg.hidden_size]));

        let lm_head_bias = weights.get_as_buffer("lm_head.bias").ok();

        Ok(Self {
            config: cfg,
            embed_tokens,
            layers,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            lm_head_bias,
        })
    }
}

impl Architecture for PhiForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let h = self.config.hidden_size;
        let mut hidden = embed_tokens(&self.embed_tokens, &input.token_ids, h);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match layer {
                PhiLayer::V2(l) => {
                    trace!(layer = layer_idx, "phi2 layer forward");
                    self.forward_phi2_layer(
                        &mut hidden,
                        l,
                        input,
                        attention,
                        layer_idx,
                    )?;
                }
                PhiLayer::V3(l) => {
                    trace!(layer = layer_idx, "phi3 layer forward");
                    self.forward_phi3_layer(
                        &mut hidden,
                        l,
                        input,
                        attention,
                        layer_idx,
                    )?;
                }
            }
        }

        // Final norm.
        let normed_final = if let Some(bias) = &self.final_norm_bias {
            LayerNorm::forward(&hidden, &self.final_norm_weight, bias, self.config.norm_eps)?
        } else {
            RMSNorm::forward(&hidden, &self.final_norm_weight, self.config.norm_eps)?
        };

        // LM head with optional bias.
        let mut logits = lm_head(
            &normed_final,
            &self.lm_head_weight,
            num_tokens,
            self.config.vocab_size,
        )?;

        if let Some(bias) = &self.lm_head_bias {
            add_lm_head_bias(&mut logits, bias, self.config.vocab_size);
        }

        Ok(logits)
    }
}

impl PhiForCausalLM {
    /// Phi-2 style: parallel attention + MLP on the same normed input.
    fn forward_phi2_layer(
        &self,
        hidden: &mut GpuBuffer<f16>,
        layer: &Phi2Layer,
        input: &ModelInput,
        attention: &dyn AttentionBackend,
        layer_idx: usize,
    ) -> Result<()> {
        // Single layernorm before both attention and MLP.
        let normed = LayerNorm::forward(
            hidden,
            &layer.ln_weight,
            &layer.ln_bias,
            self.config.norm_eps,
        )?;

        // --- Attention path ---
        let q = LinearLayer::forward(&normed, &layer.q_proj, layer.q_bias.as_ref())?;
        let k = LinearLayer::forward(&normed, &layer.k_proj, layer.k_bias.as_ref())?;
        let v = LinearLayer::forward(&normed, &layer.v_proj, layer.v_bias.as_ref())?;

        // Partial rotary embedding: only rotate first rotary_dim of each head.
        let (q_rot, k_rot) = partial_rotary_forward(
            &input.position_ids,
            &q,
            &k,
            self.config.head_dim,
            self.config.rotary_dim,
        )?;

        let attn_out = attention.forward(
            &q_rot,
            &k_rot,
            &v,
            &input.attention_metadata,
            layer_idx,
        )?;
        let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, layer.o_bias.as_ref())?;

        // --- MLP path (parallel with attention) ---
        let mlp_out = MLP::forward(
            &normed,
            &layer.gate_proj,
            &layer.up_proj,
            &layer.down_proj,
        )?;

        // Both residuals added simultaneously.
        add_inplace(hidden, &attn_proj);
        add_inplace(hidden, &mlp_out);

        Ok(())
    }

    /// Phi-3 style: sequential attention then MLP, with optional QK layernorm.
    fn forward_phi3_layer(
        &self,
        hidden: &mut GpuBuffer<f16>,
        layer: &Phi3Layer,
        input: &ModelInput,
        attention: &dyn AttentionBackend,
        layer_idx: usize,
    ) -> Result<()> {
        // Pre-attention RMSNorm.
        let normed = RMSNorm::forward(
            hidden,
            &layer.input_layernorm,
            self.config.norm_eps,
        )?;

        let mut q = LinearLayer::forward(&normed, &layer.q_proj, None)?;
        let mut k = LinearLayer::forward(&normed, &layer.k_proj, None)?;
        let v = LinearLayer::forward(&normed, &layer.v_proj, None)?;

        // Optional QK layernorm (Phi-3 specific).
        if let (Some(qw), Some(qb)) =
            (&layer.q_layernorm_weight, &layer.q_layernorm_bias)
        {
            q = LayerNorm::forward(&q, qw, qb, self.config.norm_eps)?;
        }
        if let (Some(kw), Some(kb)) =
            (&layer.k_layernorm_weight, &layer.k_layernorm_bias)
        {
            k = LayerNorm::forward(&k, kw, kb, self.config.norm_eps)?;
        }

        // Partial rotary embedding.
        let (q_rot, k_rot) = partial_rotary_forward(
            &input.position_ids,
            &q,
            &k,
            self.config.head_dim,
            self.config.rotary_dim,
        )?;

        let attn_out = attention.forward(
            &q_rot,
            &k_rot,
            &v,
            &input.attention_metadata,
            layer_idx,
        )?;
        let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, None)?;
        add_inplace(hidden, &attn_proj);

        // Post-attention RMSNorm + MLP.
        let normed2 = RMSNorm::forward(
            hidden,
            &layer.post_attention_layernorm,
            self.config.norm_eps,
        )?;

        // Phi-3 uses fused gate_up_proj: split output in half for gate and up.
        let fused = LinearLayer::forward(&normed2, &layer.gate_up_proj, None)?;
        let mlp_out = split_gate_up_and_silu_down(
            &fused,
            &layer.down_proj,
            self.config.intermediate_size,
        )?;
        add_inplace(hidden, &mlp_out);

        Ok(())
    }
}

/// Apply partial rotary embedding: rotate the first `rotary_dim` dimensions
/// of each head, pass the rest through unchanged.
fn partial_rotary_forward(
    positions: &[u32],
    query: &GpuBuffer<f16>,
    key: &GpuBuffer<f16>,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(GpuBuffer<f16>, GpuBuffer<f16>)> {
    if rotary_dim == head_dim {
        // Full rotation, use standard RoPE.
        return RotaryEmbedding::forward(positions, query, key, head_dim);
    }

    let num_tokens = positions.len();
    let q_total = query.len() / num_tokens;
    let k_total = key.len() / num_tokens;

    // Split Q into rotary part and pass-through part per head, apply RoPE to
    // rotary part, then concatenate back.
    let num_q_heads = q_total / head_dim;
    let num_k_heads = k_total / head_dim;

    // Build rotary-only and pass-through-only slices for Q.
    let mut q_rot_part = Vec::with_capacity(num_tokens * num_q_heads * rotary_dim);
    let mut q_pass_part = Vec::with_capacity(num_tokens * num_q_heads * (head_dim - rotary_dim));
    for t in 0..num_tokens {
        for h in 0..num_q_heads {
            let base = t * q_total + h * head_dim;
            q_rot_part.extend_from_slice(&query.data[base..base + rotary_dim]);
            q_pass_part.extend_from_slice(&query.data[base + rotary_dim..base + head_dim]);
        }
    }

    let mut k_rot_part = Vec::with_capacity(num_tokens * num_k_heads * rotary_dim);
    let mut k_pass_part = Vec::with_capacity(num_tokens * num_k_heads * (head_dim - rotary_dim));
    for t in 0..num_tokens {
        for h in 0..num_k_heads {
            let base = t * k_total + h * head_dim;
            k_rot_part.extend_from_slice(&key.data[base..base + rotary_dim]);
            k_pass_part.extend_from_slice(&key.data[base + rotary_dim..base + head_dim]);
        }
    }

    // Apply standard RoPE to the rotary portions (using rotary_dim as head_dim).
    let q_rot_buf = GpuBuffer::from_vec(
        q_rot_part,
        vec![num_tokens, num_q_heads * rotary_dim],
    );
    let k_rot_buf = GpuBuffer::from_vec(
        k_rot_part,
        vec![num_tokens, num_k_heads * rotary_dim],
    );
    let (q_rotated, k_rotated) =
        RotaryEmbedding::forward(positions, &q_rot_buf, &k_rot_buf, rotary_dim)?;

    // Reassemble: interleave rotary and pass-through per head.
    let pass_dim = head_dim - rotary_dim;
    let mut q_out = vec![f16::ZERO; num_tokens * q_total];
    let mut k_out = vec![f16::ZERO; num_tokens * k_total];

    for t in 0..num_tokens {
        for h in 0..num_q_heads {
            let dst = t * q_total + h * head_dim;
            let rot_src = t * (num_q_heads * rotary_dim) + h * rotary_dim;
            let pass_src = t * (num_q_heads * pass_dim) + h * pass_dim;
            q_out[dst..dst + rotary_dim]
                .copy_from_slice(&q_rotated.data[rot_src..rot_src + rotary_dim]);
            q_out[dst + rotary_dim..dst + head_dim]
                .copy_from_slice(&q_pass_part[pass_src..pass_src + pass_dim]);
        }
        for h in 0..num_k_heads {
            let dst = t * k_total + h * head_dim;
            let rot_src = t * (num_k_heads * rotary_dim) + h * rotary_dim;
            let pass_src = t * (num_k_heads * pass_dim) + h * pass_dim;
            k_out[dst..dst + rotary_dim]
                .copy_from_slice(&k_rotated.data[rot_src..rot_src + rotary_dim]);
            k_out[dst + rotary_dim..dst + head_dim]
                .copy_from_slice(&k_pass_part[pass_src..pass_src + pass_dim]);
        }
    }

    Ok((
        GpuBuffer::from_vec(q_out, query.shape.clone()),
        GpuBuffer::from_vec(k_out, key.shape.clone()),
    ))
}

/// Split a fused gate_up projection output, apply SiLU to the gate half,
/// multiply with up half, then run through down_proj.
fn split_gate_up_and_silu_down(
    fused: &GpuBuffer<f16>,
    down_weight: &GpuBuffer<f16>,
    intermediate_size: usize,
) -> Result<GpuBuffer<f16>> {
    use crate::layers::activation::fused_silu_mul;

    let num_tokens = fused.len() / (2 * intermediate_size);
    let mut gate_data = Vec::with_capacity(num_tokens * intermediate_size);
    let mut up_data = Vec::with_capacity(num_tokens * intermediate_size);

    for t in 0..num_tokens {
        let base = t * 2 * intermediate_size;
        gate_data.extend_from_slice(&fused.data[base..base + intermediate_size]);
        up_data.extend_from_slice(
            &fused.data[base + intermediate_size..base + 2 * intermediate_size],
        );
    }

    let activated = fused_silu_mul(&gate_data, &up_data);
    let activated_buf =
        GpuBuffer::from_vec(activated, vec![num_tokens, intermediate_size]);
    LinearLayer::forward(&activated_buf, down_weight, None)
}

/// Add LM head bias (f16) to f32 logits in-place.
fn add_lm_head_bias(logits: &mut GpuBuffer<f32>, bias: &GpuBuffer<f16>, vocab_size: usize) {
    let num_tokens = logits.len() / vocab_size;
    for t in 0..num_tokens {
        let base = t * vocab_size;
        for v in 0..vocab_size.min(bias.len()) {
            logits.data[base + v] += bias.data[v].to_f32();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf(vals: &[f32], shape: Vec<usize>) -> GpuBuffer<f16> {
        GpuBuffer::from_vec(
            vals.iter().map(|&v| f16::from_f32(v)).collect(),
            shape,
        )
    }

    #[test]
    fn partial_rotary_position_zero_identity() {
        // At position 0, cos=1, sin=0 so the rotary part is identity.
        let head_dim = 8;
        let rotary_dim = 4;
        let vals: Vec<f16> = (1..=8)
            .map(|i| f16::from_f32(i as f32))
            .collect();
        let q = GpuBuffer::from_vec(vals.clone(), vec![1, head_dim]);
        let k = GpuBuffer::from_vec(vals.clone(), vec![1, head_dim]);
        let (qr, kr) =
            partial_rotary_forward(&[0], &q, &k, head_dim, rotary_dim).unwrap();
        // All values should be unchanged at position 0.
        for i in 0..head_dim {
            assert!(
                (qr.data[i].to_f32() - q.data[i].to_f32()).abs() < 0.01,
                "q mismatch at {}: got {} expected {}",
                i,
                qr.data[i].to_f32(),
                q.data[i].to_f32(),
            );
            assert!(
                (kr.data[i].to_f32() - k.data[i].to_f32()).abs() < 0.01,
                "k mismatch at {}: got {} expected {}",
                i,
                kr.data[i].to_f32(),
                k.data[i].to_f32(),
            );
        }
    }

    #[test]
    fn partial_rotary_nonzero_position_preserves_passthrough() {
        // The pass-through portion (last half of head) must stay exactly the same.
        let head_dim = 8;
        let rotary_dim = 4;
        let vals: Vec<f16> = (1..=8)
            .map(|i| f16::from_f32(i as f32))
            .collect();
        let q = GpuBuffer::from_vec(vals.clone(), vec![1, head_dim]);
        let k = GpuBuffer::from_vec(vals.clone(), vec![1, head_dim]);
        let (qr, kr) =
            partial_rotary_forward(&[50], &q, &k, head_dim, rotary_dim).unwrap();
        // Pass-through dims (indices 4..8) should be unchanged.
        for i in rotary_dim..head_dim {
            assert!(
                (qr.data[i].to_f32() - q.data[i].to_f32()).abs() < 0.001,
                "q pass-through changed at {}: got {} expected {}",
                i,
                qr.data[i].to_f32(),
                q.data[i].to_f32(),
            );
            assert!(
                (kr.data[i].to_f32() - k.data[i].to_f32()).abs() < 0.001,
                "k pass-through changed at {}: got {} expected {}",
                i,
                kr.data[i].to_f32(),
                k.data[i].to_f32(),
            );
        }
        // Rotary dims (indices 0..4) should have changed at pos=50.
        let q_changed = (0..rotary_dim)
            .any(|i| (qr.data[i].to_f32() - q.data[i].to_f32()).abs() > 0.01);
        assert!(q_changed, "rotary dims should change at nonzero position");
    }

    #[test]
    fn full_rotary_dim_delegates_to_standard() {
        // When rotary_dim == head_dim, partial_rotary should match standard RoPE.
        let head_dim = 4;
        let vals: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let q = GpuBuffer::from_vec(vals.clone(), vec![1, head_dim]);
        let k = GpuBuffer::from_vec(vals.clone(), vec![1, head_dim]);
        let (qr_partial, kr_partial) =
            partial_rotary_forward(&[10], &q, &k, head_dim, head_dim).unwrap();
        let (qr_standard, kr_standard) =
            RotaryEmbedding::forward(&[10], &q, &k, head_dim).unwrap();
        for i in 0..head_dim {
            assert!(
                (qr_partial.data[i].to_f32() - qr_standard.data[i].to_f32()).abs()
                    < 0.01,
            );
            assert!(
                (kr_partial.data[i].to_f32() - kr_standard.data[i].to_f32()).abs()
                    < 0.01,
            );
        }
    }

    #[test]
    fn split_gate_up_silu_down_smoke() {
        // 1 token, intermediate_size=2, fused=[gate0, gate1, up0, up1]
        let fused = make_buf(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let down_w = make_buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let out = split_gate_up_and_silu_down(&fused, &down_w, 2).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        // silu(1) ~ 0.731, fused_silu_mul = 0.731 * 1.0, down = identity
        let v = out.data[0].to_f32();
        assert!(v > 0.5 && v < 1.0, "got {}", v);
    }

    #[test]
    fn add_lm_head_bias_works() {
        let mut logits = GpuBuffer::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![1, 4]);
        let bias = make_buf(&[10.0, 20.0, 30.0, 40.0], vec![4]);
        add_lm_head_bias(&mut logits, &bias, 4);
        assert!((logits.data[0] - 11.0).abs() < 0.1);
        assert!((logits.data[1] - 22.0).abs() < 0.1);
        assert!((logits.data[2] - 33.0).abs() < 0.1);
        assert!((logits.data[3] - 44.0).abs() < 0.1);
    }
}
