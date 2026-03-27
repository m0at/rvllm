use rvllm_core::prelude::*;

use crate::dequant;
use crate::method::QuantMethod;
use crate::weight::QuantizedWeight;

/// Quantized GEMM: fused dequantize-accumulate matrix-vector multiply.
/// Dequantizes one row at a time into a reusable buffer and immediately
/// computes the dot product, avoiding a full-matrix allocation.
/// input shape: [1, cols] (flattened), weight shape: [rows, cols].
/// Output shape: [rows].
pub fn gemm_quantized(input: &[f32], weight: &QuantizedWeight) -> Result<Vec<f32>> {
    let (rows, cols) = weight.shape;
    if input.len() != cols {
        return Err(LLMError::ModelError(format!(
            "gemm_quantized: input len {} != weight cols {}",
            input.len(),
            cols
        )));
    }

    let empty_zeros;
    let zeros = match &weight.zeros {
        Some(z) => z.as_slice(),
        None => {
            empty_zeros = vec![0.0f32; weight.scales.len()];
            &empty_zeros
        }
    };

    let mut output = vec![0.0f32; rows];
    let mut row_buf = vec![0.0f32; cols];

    for r in 0..rows {
        dequantize_row(weight, zeros, r, &mut row_buf)?;
        output[r] = dot_product_chunked(&row_buf, input);
    }

    Ok(output)
}

/// Dequantize a single row of the weight matrix into `buf`.
/// `buf` must have length == weight.shape.1.
#[inline]
fn dequantize_row(
    weight: &QuantizedWeight,
    zeros: &[f32],
    row: usize,
    buf: &mut [f32],
) -> Result<()> {
    let cols = weight.shape.1;
    debug_assert_eq!(buf.len(), cols);

    match weight.quant_config.method {
        QuantMethod::None => {
            return Err(LLMError::ModelError(
                "gemm_quantized called on unquantized weight".into(),
            ));
        }
        QuantMethod::GgufQ4_0 => {
            let group_size = 32usize;
            let bytes_per_group = group_size / 2;
            let groups_per_row = cols / group_size;
            let data_offset = row * groups_per_row * bytes_per_group;
            let scale_offset = row * groups_per_row;

            for g in 0..groups_per_row {
                let scale = weight.scales[scale_offset + g];
                let d_off = data_offset + g * bytes_per_group;
                let b_off = g * group_size;
                let data_chunk = &weight.data[d_off..d_off + bytes_per_group];
                for (j, &byte) in data_chunk.iter().enumerate() {
                    let lo = (byte & 0x0F) as f32 - 8.0;
                    let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
                    buf[b_off + j * 2] = lo * scale;
                    buf[b_off + j * 2 + 1] = hi * scale;
                }
            }
        }
        QuantMethod::GgufQ4KM => {
            let group_size = 32usize;
            let bytes_per_group = group_size / 2;
            let groups_per_row = cols / group_size;
            let data_offset = row * groups_per_row * bytes_per_group;
            let scale_offset = row * groups_per_row;

            for g in 0..groups_per_row {
                let scale = weight.scales[scale_offset + g];
                let min = zeros[scale_offset + g];
                let d_off = data_offset + g * bytes_per_group;
                let b_off = g * group_size;
                let data_chunk = &weight.data[d_off..d_off + bytes_per_group];
                for (j, &byte) in data_chunk.iter().enumerate() {
                    let lo = (byte & 0x0F) as f32;
                    let hi = ((byte >> 4) & 0x0F) as f32;
                    buf[b_off + j * 2] = lo * scale + min;
                    buf[b_off + j * 2 + 1] = hi * scale + min;
                }
            }
        }
        QuantMethod::GPTQ | QuantMethod::SqueezeLLM => {
            let bits = weight.quant_config.bits;
            let group_size = weight.quant_config.group_size;
            let mask = (1u32 << bits) - 1;
            let groups_per_row = (cols + group_size - 1) / group_size;
            let group_base = row * groups_per_row;
            let row_offset = row * cols;

            for g in 0..groups_per_row {
                let scale = weight.scales[group_base + g];
                let zero = zeros[group_base + g];
                let col_start = g * group_size;
                let col_end = (col_start + group_size).min(cols);

                for c in col_start..col_end {
                    let idx = row_offset + c;
                    let bit_offset = (idx as u64) * (bits as u64);
                    let byte_offset = (bit_offset / 8) as usize;
                    let bit_shift = (bit_offset % 8) as u32;

                    let raw = if byte_offset + 3 < weight.data.len() {
                        u32::from_le_bytes([
                            weight.data[byte_offset],
                            weight.data[byte_offset + 1],
                            weight.data[byte_offset + 2],
                            weight.data[byte_offset + 3],
                        ])
                    } else {
                        let mut r = 0u32;
                        for b in 0..4 {
                            if byte_offset + b < weight.data.len() {
                                r |= (weight.data[byte_offset + b] as u32) << (b as u32 * 8);
                            }
                        }
                        r
                    };
                    let quantized = (raw >> bit_shift) & mask;
                    buf[c] = (quantized as f32 - zero) * scale;
                }
            }
        }
        QuantMethod::AWQ => {
            let group_size = weight.quant_config.group_size;
            let groups_per_row = (cols + group_size - 1) / group_size;
            let group_base = row * groups_per_row;
            let row_offset = row * cols;

            for g in 0..groups_per_row {
                let scale = weight.scales[group_base + g];
                let zero = zeros[group_base + g];
                let col_start = g * group_size;
                let col_end = (col_start + group_size).min(cols);

                let mut c = col_start;
                while c + 1 < col_end {
                    let byte_idx = (row_offset + c) / 2;
                    let byte = weight.data[byte_idx];
                    let lo = (byte & 0x0F) as f32;
                    let hi = ((byte >> 4) & 0x0F) as f32;
                    buf[c] = (lo - zero) * scale;
                    buf[c + 1] = (hi - zero) * scale;
                    c += 2;
                }
                if c < col_end {
                    let idx = row_offset + c;
                    let byte_idx = idx / 2;
                    let nibble = if idx % 2 == 0 {
                        weight.data[byte_idx] & 0x0F
                    } else {
                        (weight.data[byte_idx] >> 4) & 0x0F
                    };
                    buf[c] = (nibble as f32 - zero) * scale;
                }
            }
        }
        QuantMethod::FP8 => {
            let scale = if weight.scales.is_empty() {
                1.0
            } else {
                weight.scales[0]
            };
            let data_offset = row * cols;
            let src = &weight.data[data_offset..data_offset + cols];
            for (dst, &byte) in buf.iter_mut().zip(src.iter()) {
                *dst = dequant::fp8::FP8_E4M3_LUT[byte as usize] * scale;
            }
        }
        QuantMethod::GgufQ5_0 | QuantMethod::GgufQ5KM | QuantMethod::GgufQ8_0 => {
            return Err(LLMError::ModelError(format!(
                "dequantization not yet implemented for {}",
                weight.quant_config.method
            )));
        }
    }
    Ok(())
}

/// Chunked dot product for autovectorization.
/// Processes 8 elements at a time with 4 independent accumulators
/// to exploit ILP on wide SIMD (NEON 128-bit).
#[inline]
fn dot_product_chunked(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let base = i * 8;
        acc0 += a[base] * b[base] + a[base + 1] * b[base + 1];
        acc1 += a[base + 2] * b[base + 2] + a[base + 3] * b[base + 3];
        acc2 += a[base + 4] * b[base + 4] + a[base + 5] * b[base + 5];
        acc3 += a[base + 6] * b[base + 6] + a[base + 7] * b[base + 7];
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        acc0 += a[tail_start + i] * b[tail_start + i];
    }

    acc0 + acc1 + acc2 + acc3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QuantConfig;
    use crate::dequant::q4::quantize_q4_0;

    #[test]
    fn gemm_q4_0_identity_like() {
        // Create a simple 4x32 weight where each row is a scaled unit-ish pattern
        let rows = 4;
        let cols = 32;
        let mut weight_f32 = vec![0.0f32; rows * cols];
        // Row i has value 1.0 at column i, 0 elsewhere
        for r in 0..rows {
            weight_f32[r * cols + r] = 1.0;
        }

        let (data, scales) = quantize_q4_0(&weight_f32, 32);
        let cfg = QuantConfig::new(QuantMethod::GgufQ4_0, 32, 4, false);
        let w = QuantizedWeight::new(data, scales, None, (rows, cols), cfg);

        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let output = gemm_quantized(&input, &w).unwrap();

        assert_eq!(output.len(), 4);
        // output[r] should approximate input[r] since only weight[r][r] ~ 1
        for r in 0..4 {
            // Tolerance is generous due to quantization
            assert!(
                (output[r] - input[r]).abs() < 2.0,
                "row {r}: expected ~{}, got {}",
                input[r],
                output[r]
            );
        }
    }

    #[test]
    fn gemm_dimension_mismatch() {
        let cfg = QuantConfig::new(QuantMethod::GgufQ4_0, 32, 4, false);
        let w = QuantizedWeight::new(vec![0u8; 16], vec![1.0], None, (1, 32), cfg);
        let input = vec![1.0f32; 64]; // wrong size
        let result = gemm_quantized(&input, &w);
        assert!(result.is_err());
    }
}
