//! Normalization layers: RMSNorm, LayerNorm.
//!
//! Optimized for autovectorization on Apple M5: chunk-of-8 accumulation,
//! fused normalize+scale passes, no intermediate allocations.

use half::f16;

use crate::bridge::{GpuBuffer, Result};

/// RMS normalization (used by Llama, Mistral, Qwen2).
pub struct RMSNorm;

impl RMSNorm {
    #[inline]
    pub fn forward(input: &GpuBuffer<f16>, weight: &GpuBuffer<f16>, eps: f32) -> Result<GpuBuffer<f16>> {
        let hidden = weight.len();
        let num_tokens = input.len() / hidden;
        let total = num_tokens * hidden;
        let mut out = vec![f16::ZERO; total];

        // Pre-convert weights to f32 once (amortized across all tokens).
        let w_f32: Vec<f32> = weight.data.iter().map(|v| v.to_f32()).collect();

        for t in 0..num_tokens {
            let start = t * hidden;
            let row = &input.data[start..start + hidden];
            let dst = &mut out[start..start + hidden];

            // Pass 1: sum of squares in chunks of 8 for autovectorization.
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
                sum_sq += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3
                        + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
            }
            for v in remainder {
                let f = v.to_f32();
                sum_sq += f * f;
            }

            let inv_rms = (sum_sq / hidden as f32 + eps).sqrt().recip();

            // Pass 2: fused normalize + scale, writing directly to output.
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

/// Standard LayerNorm with weight and bias.
pub struct LayerNorm;

impl LayerNorm {
    #[inline]
    pub fn forward(
        input: &GpuBuffer<f16>,
        weight: &GpuBuffer<f16>,
        bias: &GpuBuffer<f16>,
        eps: f32,
    ) -> Result<GpuBuffer<f16>> {
        let hidden = weight.len();
        let num_tokens = input.len() / hidden;
        let total = num_tokens * hidden;
        let mut out = vec![f16::ZERO; total];
        let inv_n = 1.0f32 / hidden as f32;

        // Pre-convert weights and biases to f32 once.
        let w_f32: Vec<f32> = weight.data.iter().map(|v| v.to_f32()).collect();
        let b_f32: Vec<f32> = bias.data.iter().map(|v| v.to_f32()).collect();

        for t in 0..num_tokens {
            let start = t * hidden;
            let row = &input.data[start..start + hidden];
            let dst = &mut out[start..start + hidden];

            // Pass 1: compute mean in chunks of 8.
            let mut sum = 0.0f32;
            let chunks = row.chunks_exact(8);
            let remainder = chunks.remainder();
            for chunk in chunks {
                sum += chunk[0].to_f32() + chunk[1].to_f32()
                     + chunk[2].to_f32() + chunk[3].to_f32()
                     + chunk[4].to_f32() + chunk[5].to_f32()
                     + chunk[6].to_f32() + chunk[7].to_f32();
            }
            for v in remainder {
                sum += v.to_f32();
            }
            let mean = sum * inv_n;

            // Pass 2: variance via sum of (x - mean)^2 in chunks of 8.
            let mut var_sum = 0.0f32;
            let chunks = row.chunks_exact(8);
            let remainder = chunks.remainder();
            for chunk in chunks {
                let d0 = chunk[0].to_f32() - mean;
                let d1 = chunk[1].to_f32() - mean;
                let d2 = chunk[2].to_f32() - mean;
                let d3 = chunk[3].to_f32() - mean;
                let d4 = chunk[4].to_f32() - mean;
                let d5 = chunk[5].to_f32() - mean;
                let d6 = chunk[6].to_f32() - mean;
                let d7 = chunk[7].to_f32() - mean;
                var_sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
                         + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
            }
            for v in remainder {
                let d = v.to_f32() - mean;
                var_sum += d * d;
            }
            let inv_std = (var_sum * inv_n + eps).sqrt().recip();

            // Pass 3: fused normalize + scale + bias, direct write.
            let row_chunks = row.chunks_exact(8);
            let row_rem = row_chunks.remainder();
            let w_chunks = w_f32.chunks_exact(8);
            let w_rem = w_chunks.remainder();
            let b_chunks = b_f32.chunks_exact(8);
            let b_rem = b_chunks.remainder();
            let d_chunks = dst.chunks_exact_mut(8);

            for (((r, w), b), d) in row_chunks.zip(w_chunks).zip(b_chunks).zip(d_chunks) {
                d[0] = f16::from_f32((r[0].to_f32() - mean) * inv_std * w[0] + b[0]);
                d[1] = f16::from_f32((r[1].to_f32() - mean) * inv_std * w[1] + b[1]);
                d[2] = f16::from_f32((r[2].to_f32() - mean) * inv_std * w[2] + b[2]);
                d[3] = f16::from_f32((r[3].to_f32() - mean) * inv_std * w[3] + b[3]);
                d[4] = f16::from_f32((r[4].to_f32() - mean) * inv_std * w[4] + b[4]);
                d[5] = f16::from_f32((r[5].to_f32() - mean) * inv_std * w[5] + b[5]);
                d[6] = f16::from_f32((r[6].to_f32() - mean) * inv_std * w[6] + b[6]);
                d[7] = f16::from_f32((r[7].to_f32() - mean) * inv_std * w[7] + b[7]);
            }
            let rem_start = hidden - row_rem.len();
            for (i, ((rv, wv), bv)) in row_rem.iter().zip(w_rem.iter()).zip(b_rem.iter()).enumerate() {
                dst[rem_start + i] = f16::from_f32((rv.to_f32() - mean) * inv_std * wv + bv);
            }
        }

        Ok(GpuBuffer::from_vec(out, vec![num_tokens, hidden]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf(vals: &[f32]) -> GpuBuffer<f16> {
        GpuBuffer::from_vec(vals.iter().map(|&v| f16::from_f32(v)).collect(), vec![vals.len()])
    }

    #[test]
    fn rms_norm_identity_weights() {
        let input = make_buf(&[1.0, 2.0, 3.0, 4.0]);
        let weight = make_buf(&[1.0, 1.0, 1.0, 1.0]);
        let out = RMSNorm::forward(&input, &weight, 1e-6).unwrap();
        assert_eq!(out.len(), 4);
        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ~ 2.7386
        let rms = (7.5_f32 + 1e-6).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0].iter().map(|v| v / rms).collect();
        for (i, &e) in expected.iter().enumerate() {
            let got = out.data[i].to_f32();
            assert!((got - e).abs() < 0.01, "idx {}: got {} expected {}", i, got, e);
        }
    }

    #[test]
    fn rms_norm_with_weights() {
        let input = make_buf(&[2.0, 2.0]);
        let weight = make_buf(&[0.5, 2.0]);
        let out = RMSNorm::forward(&input, &weight, 1e-6).unwrap();
        // RMS of [2,2] = sqrt(4) = 2, normed = [1,1], scaled = [0.5, 2.0]
        let got: Vec<f32> = out.data.iter().map(|v| v.to_f32()).collect();
        assert!((got[0] - 0.5).abs() < 0.01);
        assert!((got[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn layer_norm_zero_mean() {
        let input = make_buf(&[1.0, -1.0]);
        let weight = make_buf(&[1.0, 1.0]);
        let bias = make_buf(&[0.0, 0.0]);
        let out = LayerNorm::forward(&input, &weight, &bias, 1e-6).unwrap();
        // mean=0, var=1, normed=[1,-1]
        let got: Vec<f32> = out.data.iter().map(|v| v.to_f32()).collect();
        assert!((got[0] - 1.0).abs() < 0.01);
        assert!((got[1] + 1.0).abs() < 0.01);
    }

    #[test]
    fn layer_norm_with_bias() {
        let input = make_buf(&[3.0, 3.0]);
        let weight = make_buf(&[1.0, 1.0]);
        let bias = make_buf(&[5.0, 5.0]);
        let out = LayerNorm::forward(&input, &weight, &bias, 1e-6).unwrap();
        // mean=3, var=0, normed=[0,0], scaled=[5,5]
        let got: Vec<f32> = out.data.iter().map(|v| v.to_f32()).collect();
        assert!((got[0] - 5.0).abs() < 0.05);
        assert!((got[1] - 5.0).abs() < 0.05);
    }
}
