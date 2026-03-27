//! MLP (feed-forward) block: gate_proj + up_proj -> activation -> down_proj.

use half::f16;

use crate::bridge::{GpuBuffer, Result};
use crate::layers::activation::fused_silu_mul;
use crate::layers::linear::LinearLayer;

/// Standard gated MLP used by Llama/Mistral/Qwen2.
/// gate_proj and up_proj run in parallel, element-wise multiply with activation,
/// then down_proj.
pub struct MLP;

impl MLP {
    #[inline]
    pub fn forward(
        input: &GpuBuffer<f16>,
        gate_weight: &GpuBuffer<f16>,
        up_weight: &GpuBuffer<f16>,
        down_weight: &GpuBuffer<f16>,
    ) -> Result<GpuBuffer<f16>> {
        let gate = LinearLayer::forward(input, gate_weight, None)?;
        let up = LinearLayer::forward(input, up_weight, None)?;

        // Fused silu(gate) * up in a single pass -- saves one full traversal.
        let fused = fused_silu_mul(&gate.data, &up.data);
        let fused_buf = GpuBuffer::from_vec(fused, gate.shape);

        // Down projection.
        LinearLayer::forward(&fused_buf, down_weight, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(vals: &[f32], shape: Vec<usize>) -> GpuBuffer<f16> {
        GpuBuffer::from_vec(vals.iter().map(|&v| f16::from_f32(v)).collect(), shape)
    }

    #[test]
    fn mlp_smoke() {
        // in=2, intermediate=2, out=2
        let input = buf(&[1.0, 1.0], vec![1, 2]);
        let gate_w = buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let up_w = buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let down_w = buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let out = MLP::forward(&input, &gate_w, &up_w, &down_w).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        // gate=[1,1], silu(1)~0.731, up=[1,1], fused~[0.731,0.731], down=identity
        let v = out.data[0].to_f32();
        assert!(v > 0.5 && v < 1.0, "got {}", v);
    }
}
