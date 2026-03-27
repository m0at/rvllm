/// Dequantize AWQ format: 4-bit asymmetric quantization with per-group scales and zeros.
/// AWQ uses the same bit-packing as GPTQ but always 4-bit.
#[inline]
pub fn dequantize_awq(
    data: &[u8],
    scales: &[f32],
    zeros: &[f32],
    group_size: usize,
    shape: (usize, usize),
) -> Vec<f32> {
    let (rows, cols) = shape;
    let total = rows * cols;
    let mut output = vec![0.0f32; total];
    let groups_per_row = (cols + group_size - 1) / group_size;

    for row in 0..rows {
        let row_offset = row * cols;
        let group_base = row * groups_per_row;
        let out_row = &mut output[row_offset..row_offset + cols];

        let full_groups = cols / group_size;
        let (full_part, remainder) = out_row.split_at_mut(full_groups * group_size);

        for (g, chunk) in full_part.chunks_exact_mut(group_size).enumerate() {
            let scale = scales[group_base + g];
            let zero = zeros[group_base + g];
            let base_idx = row_offset + g * group_size;

            // 4-bit: two nibbles per byte. Process pairs for tight autovectorizable loop.
            let pairs = chunk.chunks_exact_mut(2);
            for (j, pair) in pairs.enumerate() {
                let byte_idx = (base_idx + j * 2) / 2;
                let byte = data[byte_idx];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;
                pair[0] = (lo - zero) * scale;
                pair[1] = (hi - zero) * scale;
            }
            // If group_size is odd, handle the last element
            if group_size % 2 != 0 {
                let last = group_size - 1;
                let idx = base_idx + last;
                let byte_idx = idx / 2;
                let nibble = if idx % 2 == 0 {
                    data[byte_idx] & 0x0F
                } else {
                    (data[byte_idx] >> 4) & 0x0F
                };
                chunk[last] = (nibble as f32 - zero) * scale;
            }
        }

        // Handle trailing partial group
        if !remainder.is_empty() {
            let g = full_groups;
            let scale = scales[group_base + g];
            let zero = zeros[group_base + g];
            let base_idx = row_offset + g * group_size;

            for (j, dst) in remainder.iter_mut().enumerate() {
                let idx = base_idx + j;
                let byte_idx = idx / 2;
                let nibble = if idx % 2 == 0 {
                    data[byte_idx] & 0x0F
                } else {
                    (data[byte_idx] >> 4) & 0x0F
                };
                *dst = (nibble as f32 - zero) * scale;
            }
        }
    }
    output
}

/// Quantize f32 values to AWQ-style 4-bit format (for testing).
pub fn quantize_awq(
    values: &[f32],
    group_size: usize,
    shape: (usize, usize),
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (rows, cols) = shape;
    let groups_per_row = (cols + group_size - 1) / group_size;
    let num_groups = rows * groups_per_row;
    let mut scales = vec![0.0f32; num_groups];
    let mut zeros = vec![0.0f32; num_groups];

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
            let scale = if range == 0.0 { 1.0 } else { range / 15.0 };
            let zero = -min_v / scale;
            scales[group_idx] = scale;
            zeros[group_idx] = zero;
        }
    }

    // Pack 4-bit values, two per byte
    let total = rows * cols;
    let total_bytes = (total + 1) / 2;
    let mut data = vec![0u8; total_bytes];

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let group_idx = row * groups_per_row + col / group_size;
            let scale = scales[group_idx];
            let zero = zeros[group_idx];

            let q = ((values[idx] / scale) + zero).round().clamp(0.0, 15.0) as u8;

            let byte_idx = idx / 2;
            if idx % 2 == 0 {
                data[byte_idx] |= q;
            } else {
                data[byte_idx] |= q << 4;
            }
        }
    }

    (data, scales, zeros)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn awq_round_trip() {
        let shape = (2, 128);
        let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let (data, scales, zeros) = quantize_awq(&original, 128, shape);
        let restored = dequantize_awq(&data, &scales, &zeros, 128, shape);

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
    fn awq_zeros() {
        let shape = (1, 128);
        let original = vec![0.0f32; 128];
        let (data, scales, zeros) = quantize_awq(&original, 128, shape);
        let restored = dequantize_awq(&data, &scales, &zeros, 128, shape);

        for v in &restored {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }

    #[test]
    fn awq_small_group() {
        let shape = (1, 64);
        let original: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let (data, scales, zeros) = quantize_awq(&original, 32, shape);
        let restored = dequantize_awq(&data, &scales, &zeros, 32, shape);

        assert_eq!(restored.len(), 64);
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!(
                (o - r).abs() < 0.5,
                "original={o}, restored={r}",
            );
        }
    }
}
