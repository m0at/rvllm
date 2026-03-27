//! Linear (dense / GEMM) layer.

use half::f16;

use crate::bridge::{GpuBuffer, Result};

/// Dense linear projection: out = input @ weight^T [+ bias].
/// weight shape: [out_features, in_features] (row-major).
pub struct LinearLayer;

impl LinearLayer {
    #[inline]
    pub fn forward(
        input: &GpuBuffer<f16>,
        weight: &GpuBuffer<f16>,
        bias_opt: Option<&GpuBuffer<f16>>,
    ) -> Result<GpuBuffer<f16>> {
        let in_features = weight.shape.get(1).copied().unwrap_or(1);
        let out_features = weight.shape.get(0).copied().unwrap_or(1);
        let num_tokens = input.len() / in_features;

        let mut out = vec![f16::ZERO; num_tokens * out_features];

        for t in 0..num_tokens {
            let row_start = t * in_features;
            let input_row = &input.data[row_start..row_start + in_features];

            for o in 0..out_features {
                let w_start = o * in_features;
                let weight_row = &weight.data[w_start..w_start + in_features];

                let sum = dot_f16(input_row, weight_row);

                let val = if let Some(bias) = bias_opt {
                    sum + bias.data[o].to_f32()
                } else {
                    sum
                };
                out[t * out_features + o] = f16::from_f32(val);
            }
        }

        Ok(GpuBuffer::from_vec(out, vec![num_tokens, out_features]))
    }
}

/// 4-way accumulator dot product over f16 slices. Processes chunks of 4 to
/// break dependency chains and let the CPU pipeline multiple FMAs.
#[inline]
fn dot_f16(a: &[f16], b: &[f16]) -> f32 {
    let mut acc0: f32 = 0.0;
    let mut acc1: f32 = 0.0;
    let mut acc2: f32 = 0.0;
    let mut acc3: f32 = 0.0;

    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        acc0 += ca[0].to_f32() * cb[0].to_f32();
        acc1 += ca[1].to_f32() * cb[1].to_f32();
        acc2 += ca[2].to_f32() * cb[2].to_f32();
        acc3 += ca[3].to_f32() * cb[3].to_f32();
    }

    for (va, vb) in rem_a.iter().zip(rem_b.iter()) {
        acc0 += va.to_f32() * vb.to_f32();
    }

    (acc0 + acc1) + (acc2 + acc3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(vals: &[f32], shape: Vec<usize>) -> GpuBuffer<f16> {
        GpuBuffer::from_vec(vals.iter().map(|&v| f16::from_f32(v)).collect(), shape)
    }

    #[test]
    fn identity_projection() {
        // 2x2 identity weight, single token [3, 7]
        let input = buf(&[3.0, 7.0], vec![1, 2]);
        let weight = buf(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let out = LinearLayer::forward(&input, &weight, None).unwrap();
        let got: Vec<f32> = out.data.iter().map(|v| v.to_f32()).collect();
        assert!((got[0] - 3.0).abs() < 0.01);
        assert!((got[1] - 7.0).abs() < 0.01);
    }

    #[test]
    fn with_bias() {
        let input = buf(&[1.0, 1.0], vec![1, 2]);
        let weight = buf(&[2.0, 3.0], vec![1, 2]);
        let bias = buf(&[10.0], vec![1]);
        let out = LinearLayer::forward(&input, &weight, Some(&bias)).unwrap();
        // 2+3+10 = 15
        assert!((out.data[0].to_f32() - 15.0).abs() < 0.1);
    }
}
