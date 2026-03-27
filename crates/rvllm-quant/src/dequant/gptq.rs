/// Dequantize GPTQ format: N-bit asymmetric quantization with per-group scales and zeros.
/// Data is packed with `bits` per weight, little-endian within each u32.
#[inline]
pub fn dequantize_gptq(
    data: &[u8],
    scales: &[f32],
    zeros: &[f32],
    group_size: usize,
    bits: u32,
    shape: (usize, usize),
) -> Vec<f32> {
    let (rows, cols) = shape;
    let total = rows * cols;
    let mut output = vec![0.0f32; total];
    let mask = (1u32 << bits) - 1;
    let groups_per_row = (cols + group_size - 1) / group_size;

    for row in 0..rows {
        let row_offset = row * cols;
        let group_base = row * groups_per_row;
        let out_row = &mut output[row_offset..row_offset + cols];

        // Process full groups via chunks_exact
        let full_groups = cols / group_size;
        let (full_part, remainder) = out_row.split_at_mut(full_groups * group_size);

        for (g, chunk) in full_part.chunks_exact_mut(group_size).enumerate() {
            let scale = scales[group_base + g];
            let zero = zeros[group_base + g];
            let col_start = g * group_size;

            for (j, dst) in chunk.iter_mut().enumerate() {
                let idx = row_offset + col_start + j;
                let bit_offset = (idx as u64) * (bits as u64);
                let byte_offset = (bit_offset / 8) as usize;
                let bit_shift = (bit_offset % 8) as u32;

                let raw = if byte_offset + 3 < data.len() {
                    u32::from_le_bytes([
                        data[byte_offset],
                        data[byte_offset + 1],
                        data[byte_offset + 2],
                        data[byte_offset + 3],
                    ])
                } else {
                    let mut r = 0u32;
                    for b in 0..4 {
                        if byte_offset + b < data.len() {
                            r |= (data[byte_offset + b] as u32) << (b as u32 * 8);
                        }
                    }
                    r
                };
                let quantized = (raw >> bit_shift) & mask;
                *dst = (quantized as f32 - zero) * scale;
            }
        }

        // Handle trailing partial group
        if !remainder.is_empty() {
            let g = full_groups;
            let scale = scales[group_base + g];
            let zero = zeros[group_base + g];
            let col_start = g * group_size;

            for (j, dst) in remainder.iter_mut().enumerate() {
                let idx = row_offset + col_start + j;
                let bit_offset = (idx as u64) * (bits as u64);
                let byte_offset = (bit_offset / 8) as usize;
                let bit_shift = (bit_offset % 8) as u32;

                let mut raw = 0u32;
                for b in 0..4 {
                    if byte_offset + b < data.len() {
                        raw |= (data[byte_offset + b] as u32) << (b as u32 * 8);
                    }
                }
                let quantized = (raw >> bit_shift) & mask;
                *dst = (quantized as f32 - zero) * scale;
            }
        }
    }
    output
}

/// Quantize f32 values to GPTQ-style format (for testing).
pub fn quantize_gptq(
    values: &[f32],
    group_size: usize,
    bits: u32,
    shape: (usize, usize),
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (rows, cols) = shape;
    let max_val = (1u32 << bits) - 1;
    let groups_per_row = (cols + group_size - 1) / group_size;
    let num_groups = rows * groups_per_row;
    let mut scales = vec![0.0f32; num_groups];
    let mut zeros = vec![0.0f32; num_groups];

    // Compute per-group scales and zeros
    for row in 0..rows {
        for g in 0..groups_per_row {
            let group_idx = row * groups_per_row + g;
            let col_start = g * group_size;
            let col_end = (col_start + group_size).min(cols);

            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for c in col_start..col_end {
                let v = values[row * cols + c];
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }

            let range = max_v - min_v;
            let scale = if range == 0.0 {
                1.0
            } else {
                range / max_val as f32
            };
            let zero = -min_v / scale;
            scales[group_idx] = scale;
            zeros[group_idx] = zero;
        }
    }

    // Pack quantized values bit-by-bit
    let total_bits = (rows * cols) as u64 * bits as u64;
    let total_bytes = ((total_bits + 7) / 8) as usize;
    let mut data = vec![0u8; total_bytes];

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let group_idx = row * groups_per_row + col / group_size;
            let scale = scales[group_idx];
            let zero = zeros[group_idx];

            let q = ((values[idx] / scale) + zero).round().clamp(0.0, max_val as f32) as u32;

            let bit_offset = idx as u64 * bits as u64;
            let byte_offset = (bit_offset / 8) as usize;
            let bit_shift = (bit_offset % 8) as u32;

            // Write the quantized value across bytes
            let shifted = (q as u64) << bit_shift;
            for b in 0..4 {
                if byte_offset + b < data.len() {
                    data[byte_offset + b] |= ((shifted >> (b as u64 * 8)) & 0xFF) as u8;
                }
            }
        }
    }

    (data, scales, zeros)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gptq_4bit_round_trip() {
        let shape = (2, 128);
        let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let (data, scales, zeros) = quantize_gptq(&original, 128, 4, shape);
        let restored = dequantize_gptq(&data, &scales, &zeros, 128, 4, shape);

        assert_eq!(restored.len(), 256);
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!(
                (o - r).abs() < 0.2,
                "original={o}, restored={r}, diff={}",
                (o - r).abs()
            );
        }
    }

    #[test]
    fn gptq_8bit_round_trip() {
        let shape = (1, 128);
        let original: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let (data, scales, zeros) = quantize_gptq(&original, 128, 8, shape);
        let restored = dequantize_gptq(&data, &scales, &zeros, 128, 8, shape);

        assert_eq!(restored.len(), 128);
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!(
                (o - r).abs() < 0.06,
                "original={o}, restored={r}, diff={}",
                (o - r).abs()
            );
        }
    }

    #[test]
    fn gptq_zeros_round_trip() {
        let shape = (1, 128);
        let original = vec![0.0f32; 128];
        let (data, scales, zeros) = quantize_gptq(&original, 128, 4, shape);
        let restored = dequantize_gptq(&data, &scales, &zeros, 128, 4, shape);

        for v in &restored {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }
}
