//! CPU-side FP8 E4M3 weight quantization for model load time.
//!
//! Converts f16 weight matrices to FP8 E4M3 with per-row scaling,
//! halving weight memory on GPU. Runs once at startup so CPU perf is fine.

use half::f16;

const FP8_E4M3_MAX: f32 = 448.0;

/// Convert a single f32 value to FP8 E4M3 byte representation.
fn float_to_fp8_e4m3(val: f32) -> u8 {
    let val = val.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
    let sign: u8 = if val < 0.0 { 0x80 } else { 0x00 };
    let abs_val = val.abs();

    if abs_val < 1.9531e-3 {
        // Subnormal or zero
        let mantissa = ((abs_val / 1.9531e-3) + 0.5) as u8;
        let mantissa = mantissa.min(7);
        return sign | mantissa;
    }

    let bits = abs_val.to_bits();
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp8_exp = fp32_exp + 7;

    if fp8_exp <= 0 {
        let subnormal_val = abs_val / 1.9531e-3;
        let mantissa = (subnormal_val + 0.5) as u8;
        let shift = (1 - fp8_exp) as u32;
        let mantissa = (mantissa >> shift).min(7);
        return sign | mantissa;
    }
    if fp8_exp > 15 {
        return sign | 0x7E; // max finite
    }

    let fp32_mantissa = bits & 0x7FFFFF;
    let mut mantissa = ((fp32_mantissa + (1 << 19)) >> 20) as i32;
    let mut fp8_exp = fp8_exp;

    if mantissa >= 8 {
        mantissa = 0;
        fp8_exp += 1;
        if fp8_exp > 15 {
            return sign | 0x7E;
        }
    }

    // Avoid NaN (exp=15, mantissa=7)
    if fp8_exp == 15 && mantissa > 6 {
        mantissa = 6;
    }

    sign | ((fp8_exp as u8 & 0xF) << 3) | (mantissa as u8 & 0x7)
}

/// Result of quantizing a weight matrix to FP8.
pub struct Fp8QuantizedWeight {
    /// FP8 E4M3 weight bytes, row-major [out_dim, in_dim]
    pub data: Vec<u8>,
    /// Per-row scale factors [out_dim], f16
    pub scales: Vec<f16>,
    pub out_dim: usize,
    pub in_dim: usize,
}

/// Quantize an f16 weight matrix [out_dim, in_dim] to FP8 E4M3 with per-row scales.
///
/// For each row: scale = max(|row|) / 448.0, then fp8_val = round(f16_val / scale).
/// Returns the quantized bytes and per-row scale factors.
pub fn quantize_weight_fp8(weights: &[f16], out_dim: usize, in_dim: usize) -> Fp8QuantizedWeight {
    assert_eq!(weights.len(), out_dim * in_dim);

    let mut data = vec![0u8; out_dim * in_dim];
    let mut scales = vec![f16::ZERO; out_dim];

    for row in 0..out_dim {
        let row_start = row * in_dim;
        let row_slice = &weights[row_start..row_start + in_dim];

        // Find absmax for this row
        let mut absmax: f32 = 0.0;
        for &v in row_slice {
            let fv = v.to_f32().abs();
            if fv > absmax {
                absmax = fv;
            }
        }

        // Compute scale (with epsilon to avoid div-by-zero)
        let scale = if absmax < 1e-12 {
            1e-12_f32
        } else {
            absmax / FP8_E4M3_MAX
        };
        let inv_scale = 1.0 / scale;
        scales[row] = f16::from_f32(scale);

        // Quantize each element
        let out_row = &mut data[row_start..row_start + in_dim];
        for (j, &v) in row_slice.iter().enumerate() {
            out_row[j] = float_to_fp8_e4m3(v.to_f32() * inv_scale);
        }
    }

    Fp8QuantizedWeight {
        data,
        scales,
        out_dim,
        in_dim,
    }
}

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream};

/// Quantize f16 weights and upload FP8 data + scales to GPU.
///
/// Takes host-side f16 weight data (already read from safetensors),
/// quantizes to FP8 on CPU, then uploads both buffers to the device.
#[cfg(feature = "cuda")]
pub fn quantize_and_upload_fp8(
    stream: &CudaStream,
    weights: &[f16],
    out_dim: usize,
    in_dim: usize,
) -> crate::Result<(CudaSlice<u8>, CudaSlice<f16>)> {
    let q = quantize_weight_fp8(weights, out_dim, in_dim);

    let fp8_dev = stream
        .clone_htod(&q.data)
        .map_err(|e| crate::LLMError::GpuError(format!("FP8 weight upload: {e}")))?;
    let scale_dev = stream
        .clone_htod(&q.scales)
        .map_err(|e| crate::LLMError::GpuError(format!("FP8 scale upload: {e}")))?;

    Ok((fp8_dev, scale_dev))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_fp8() {
        // Verify quantize -> dequantize preserves values within FP8 precision
        let test_vals: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -200.0, 448.0, -448.0];
        for &v in &test_vals {
            let fp8 = float_to_fp8_e4m3(v);
            let back = fp8_e4m3_to_float(fp8);
            let err = (back - v).abs();
            // FP8 E4M3 has ~0.1% relative error for normal range
            if v.abs() > 0.01 {
                assert!(
                    err / v.abs() < 0.15,
                    "roundtrip error too large: {v} -> fp8({fp8:#04x}) -> {back}, err={err}"
                );
            }
        }
    }

    #[test]
    fn test_quantize_weight_identity_scale() {
        // A row with max=448 should have scale=1.0
        let row: Vec<f16> = (0..128).map(|i| f16::from_f32(i as f32 * 3.5)).collect();
        let q = quantize_weight_fp8(&row, 1, 128);
        let scale = q.scales[0].to_f32();
        assert!(scale > 0.0);
        assert!(scale < 2.0); // max val is 127*3.5=444.5, scale ~= 444.5/448 ~= 0.99
    }

    fn fp8_e4m3_to_float(fp8: u8) -> f32 {
        let sign: f32 = if fp8 & 0x80 != 0 { -1.0 } else { 1.0 };
        let exp = ((fp8 >> 3) & 0xF) as i32;
        let mantissa = (fp8 & 0x7) as i32;

        if exp == 0 {
            if mantissa == 0 {
                return 0.0;
            }
            return sign * mantissa as f32 * 1.9531e-3;
        }
        if exp == 15 && mantissa == 7 {
            return 0.0;
        }

        let fmantissa = 1.0 + mantissa as f32 / 8.0;
        sign * fmantissa * 2.0_f32.powi(exp - 7)
    }
}
