//! Mixture-of-Experts (MoE) layer.
//!
//! Generic top-k gated sparse MoE usable by DeepSeek-V2, Mixtral, and any
//! future MoE architecture. Supports an optional shared expert that runs
//! alongside the routed experts.

use half::f16;
use tracing::trace;

use crate::bridge::{GpuBuffer, Result};
use crate::layers::activation::fused_silu_mul;
use crate::layers::linear::LinearLayer;

/// A single expert FFN: gate_proj + up_proj -> fused SiLU*mul -> down_proj.
pub struct ExpertFFN {
    pub gate_proj: GpuBuffer<f16>,
    pub up_proj: GpuBuffer<f16>,
    pub down_proj: GpuBuffer<f16>,
}

impl ExpertFFN {
    /// Run the gated MLP for this expert.
    pub fn forward(&self, input: &GpuBuffer<f16>) -> Result<GpuBuffer<f16>> {
        let gate = LinearLayer::forward(input, &self.gate_proj, None)?;
        let up = LinearLayer::forward(input, &self.up_proj, None)?;
        let fused = fused_silu_mul(&gate.data, &up.data);
        let fused_buf = GpuBuffer::from_vec(fused, gate.shape);
        LinearLayer::forward(&fused_buf, &self.down_proj, None)
    }
}

/// Mixture-of-Experts layer with top-k routing and optional shared expert.
pub struct MoELayer {
    /// Router weight: [num_experts, hidden_size].
    pub gate: GpuBuffer<f16>,
    /// Routed expert FFNs.
    pub experts: Vec<ExpertFFN>,
    /// Number of experts to activate per token.
    pub top_k: usize,
    /// Whether to renormalize top-k weights to sum to 1.
    pub renormalize: bool,
    /// Optional shared expert that always runs (DeepSeek-V2 style).
    /// Its output is added to the routed experts' combined output.
    pub shared_expert: Option<ExpertFFN>,
}

impl MoELayer {
    /// Forward pass: route each token to top-k experts, combine by softmax
    /// weight, then add shared expert output if present.
    ///
    /// input shape: [num_tokens, hidden_size]
    /// output shape: [num_tokens, hidden_size]
    pub fn forward(&self, input: &GpuBuffer<f16>) -> Result<GpuBuffer<f16>> {
        let hidden_size = if input.shape.len() >= 2 {
            input.shape[1]
        } else {
            input.data.len()
        };
        let num_tokens = input.data.len() / hidden_size;
        let num_experts = self.experts.len();

        trace!(
            num_tokens = num_tokens,
            num_experts = num_experts,
            top_k = self.top_k,
            has_shared = self.shared_expert.is_some(),
            "moe forward"
        );

        // Router logits: [num_tokens, num_experts].
        let router_logits = LinearLayer::forward(input, &self.gate, None)?;

        let mut output = vec![f16::ZERO; num_tokens * hidden_size];

        for t in 0..num_tokens {
            let logit_offset = t * num_experts;
            let logits_f32: Vec<f32> = (0..num_experts)
                .map(|e| router_logits.data[logit_offset + e].to_f32())
                .collect();

            // Top-k expert selection.
            let top_indices = top_k_indices(&logits_f32, self.top_k);
            let top_logits: Vec<f32> = top_indices.iter().map(|&i| logits_f32[i]).collect();
            let weights = softmax(&top_logits);

            // Single-token slice.
            let tok_start = t * hidden_size;
            let tok_data = input.data[tok_start..tok_start + hidden_size].to_vec();
            let tok_buf = GpuBuffer::from_vec(tok_data, vec![1, hidden_size]);

            // Accumulate weighted expert outputs.
            let out_offset = t * hidden_size;
            for (rank, &expert_idx) in top_indices.iter().enumerate() {
                let expert_out = self.experts[expert_idx].forward(&tok_buf)?;
                let w = weights[rank];
                for h in 0..hidden_size {
                    let cur = output[out_offset + h].to_f32();
                    output[out_offset + h] =
                        f16::from_f32(cur + expert_out.data[h].to_f32() * w);
                }
            }

            // Add shared expert output (unweighted, always-on).
            if let Some(ref shared) = self.shared_expert {
                let shared_out = shared.forward(&tok_buf)?;
                for h in 0..hidden_size {
                    let cur = output[out_offset + h].to_f32();
                    output[out_offset + h] =
                        f16::from_f32(cur + shared_out.data[h].to_f32());
                }
            }
        }

        Ok(GpuBuffer::from_vec(output, vec![num_tokens, hidden_size]))
    }
}

/// Return indices of the top-k largest values.
fn top_k_indices(vals: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = vals.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|&(i, _)| i).collect()
}

/// Numerically stable softmax over a small slice.
fn softmax(vals: &[f32]) -> Vec<f32> {
    if vals.is_empty() {
        return vec![];
    }
    let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = vals.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(vals: &[f32], shape: Vec<usize>) -> GpuBuffer<f16> {
        GpuBuffer::from_vec(vals.iter().map(|&v| f16::from_f32(v)).collect(), shape)
    }

    fn identity_expert(size: usize) -> ExpertFFN {
        let mut id = vec![0.0f32; size * size];
        for i in 0..size {
            id[i * size + i] = 1.0;
        }
        ExpertFFN {
            gate_proj: buf(&id, vec![size, size]),
            up_proj: buf(&id, vec![size, size]),
            down_proj: buf(&id, vec![size, size]),
        }
    }

    #[test]
    fn top_k_indices_basic() {
        let vals = [0.1, 0.9, 0.5, 0.3];
        let top2 = top_k_indices(&vals, 2);
        assert_eq!(top2, vec![1, 2]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let vals = [1.0, 2.0, 3.0];
        let s = softmax(&vals);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_ordering() {
        let vals = [1.0, 3.0, 2.0];
        let s = softmax(&vals);
        assert!(s[1] > s[2]);
        assert!(s[2] > s[0]);
    }

    #[test]
    fn expert_ffn_smoke() {
        let expert = ExpertFFN {
            gate_proj: buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            up_proj: buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            down_proj: buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]),
        };
        let input = buf(&[1.0, 1.0], vec![1, 2]);
        let out = expert.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        let v = out.data[0].to_f32();
        assert!(v > 0.5 && v < 1.0, "got {}", v);
    }

    #[test]
    fn moe_layer_smoke() {
        let expert0 = identity_expert(2);
        let expert1 = identity_expert(2);
        let gate = buf(&[2.0, 0.0, 0.0, 1.0], vec![2, 2]);

        let moe = MoELayer {
            gate,
            experts: vec![expert0, expert1],
            top_k: 1,
            renormalize: true,
            shared_expert: None,
        };

        let input = buf(&[1.0, 1.0], vec![1, 2]);
        let out = moe.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        let v = out.data[0].to_f32();
        assert!(v.abs() > 0.01, "got {}", v);
    }

    #[test]
    fn moe_top2_combines() {
        let expert0 = identity_expert(2);
        let expert1 = identity_expert(2);
        let gate = buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);

        let moe = MoELayer {
            gate,
            experts: vec![expert0, expert1],
            top_k: 2,
            renormalize: true,
            shared_expert: None,
        };

        let input = buf(&[1.0, 1.0], vec![1, 2]);
        let out = moe.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        let v = out.data[0].to_f32();
        assert!(v > 0.5 && v < 1.0, "got {}", v);
    }

    #[test]
    fn moe_with_shared_expert() {
        let expert0 = identity_expert(2);
        let expert1 = identity_expert(2);
        let shared = identity_expert(2);
        let gate = buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);

        let with_shared = MoELayer {
            gate: gate.clone(),
            experts: vec![identity_expert(2), identity_expert(2)],
            top_k: 1,
            renormalize: true,
            shared_expert: Some(shared),
        };
        let without_shared = MoELayer {
            gate,
            experts: vec![expert0, expert1],
            top_k: 1,
            renormalize: true,
            shared_expert: None,
        };

        let input = buf(&[1.0, 1.0], vec![1, 2]);
        let out_with = with_shared.forward(&input).unwrap();
        let out_without = without_shared.forward(&input).unwrap();

        let mag_with: f32 = out_with.data.iter().map(|v| v.to_f32().abs()).sum();
        let mag_without: f32 = out_without.data.iter().map(|v| v.to_f32().abs()).sum();
        assert!(
            mag_with > mag_without,
            "shared expert should increase magnitude: {} vs {}",
            mag_with,
            mag_without
        );
    }

    #[test]
    fn moe_multi_token() {
        let gate = buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let moe = MoELayer {
            gate,
            experts: vec![identity_expert(2), identity_expert(2)],
            top_k: 1,
            renormalize: true,
            shared_expert: None,
        };

        let input = buf(&[1.0, 0.5, 0.5, 1.0, 0.0, 1.0], vec![3, 2]);
        let out = moe.forward(&input).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.data.len(), 6);
    }
}
