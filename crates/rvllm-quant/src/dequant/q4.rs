/// Dequantize Q4_0 format: 4-bit symmetric quantization with per-group scales.
/// Each byte packs two 4-bit values (low nibble first).
/// Block size is 32 elements, one f32 scale per block.
#[inline]
pub fn dequantize_q4_0(data: &[u8], scales: &[f32], shape: (usize, usize)) -> Vec<f32> {
    let total = shape.0 * shape.1;
    let group_size = 32usize;
    let bytes_per_group = group_size / 2;
    let mut output = vec![0.0f32; total];

    for (g, (out_chunk, data_chunk)) in output
        .chunks_exact_mut(group_size)
        .zip(data.chunks_exact(bytes_per_group))
        .enumerate()
    {
        let scale = scales[g];
        for (j, &byte) in data_chunk.iter().enumerate() {
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
            out_chunk[j * 2] = lo * scale;
            out_chunk[j * 2 + 1] = hi * scale;
        }
    }
    output
}

/// Dequantize Q4_K_M format: 4-bit quantization with per-group scales and minimums.
/// Each byte packs two 4-bit values (low nibble first).
/// Block size is 32 elements.
#[inline]
pub fn dequantize_q4_k_m(
    data: &[u8],
    scales: &[f32],
    mins: &[f32],
    shape: (usize, usize),
) -> Vec<f32> {
    let total = shape.0 * shape.1;
    let group_size = 32usize;
    let bytes_per_group = group_size / 2;
    let mut output = vec![0.0f32; total];

    for (g, (out_chunk, data_chunk)) in output
        .chunks_exact_mut(group_size)
        .zip(data.chunks_exact(bytes_per_group))
        .enumerate()
    {
        let scale = scales[g];
        let min = mins[g];
        for (j, &byte) in data_chunk.iter().enumerate() {
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            out_chunk[j * 2] = lo * scale + min;
            out_chunk[j * 2 + 1] = hi * scale + min;
        }
    }
    output
}

/// Quantize f32 values to Q4_0 format (for testing round-trips).
pub fn quantize_q4_0(values: &[f32], group_size: usize) -> (Vec<u8>, Vec<f32>) {
    let num_groups = (values.len() + group_size - 1) / group_size;
    let mut data = Vec::with_capacity(num_groups * (group_size / 2));
    let mut scales = Vec::with_capacity(num_groups);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(values.len());
        let group = &values[start..end];

        // Find max absolute value after centering
        let max_abs = group
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);

        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
        scales.push(scale);

        // Pack pairs of 4-bit values
        let mut i = 0;
        while i < group.len() {
            let v0 = ((group[i] / scale) + 8.0).round().clamp(0.0, 15.0) as u8;
            let v1 = if i + 1 < group.len() {
                ((group[i + 1] / scale) + 8.0).round().clamp(0.0, 15.0) as u8
            } else {
                8 // zero point
            };
            data.push(v0 | (v1 << 4));
            i += 2;
        }
    }
    (data, scales)
}

/// Quantize f32 values to Q4_K_M format (for testing round-trips).
pub fn quantize_q4_k_m(values: &[f32], group_size: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let num_groups = (values.len() + group_size - 1) / group_size;
    let mut data = Vec::with_capacity(num_groups * (group_size / 2));
    let mut scales = Vec::with_capacity(num_groups);
    let mut mins = Vec::with_capacity(num_groups);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(values.len());
        let group = &values[start..end];

        let min_val = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range == 0.0 { 1.0 } else { range / 15.0 };
        scales.push(scale);
        mins.push(min_val);

        let mut i = 0;
        while i < group.len() {
            let v0 = ((group[i] - min_val) / scale).round().clamp(0.0, 15.0) as u8;
            let v1 = if i + 1 < group.len() {
                ((group[i + 1] - min_val) / scale).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };
            data.push(v0 | (v1 << 4));
            i += 2;
        }
    }
    (data, scales, mins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_round_trip() {
        let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let (data, scales) = quantize_q4_0(&original, 32);
        let restored = dequantize_q4_0(&data, &scales, (1, 32));
        assert_eq!(restored.len(), 32);
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!(
                (o - r).abs() < 0.3,
                "original={o}, restored={r}, diff={}",
                (o - r).abs()
            );
        }
    }

    #[test]
    fn q4_0_zeros() {
        let original = vec![0.0f32; 32];
        let (data, scales) = quantize_q4_0(&original, 32);
        let restored = dequantize_q4_0(&data, &scales, (1, 32));
        for v in &restored {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn q4_k_m_round_trip() {
        let original: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
        let (data, scales, mins) = quantize_q4_k_m(&original, 32);
        let restored = dequantize_q4_k_m(&data, &scales, &mins, (1, 32));
        assert_eq!(restored.len(), 32);
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!(
                (o - r).abs() < 1.2,
                "original={o}, restored={r}, diff={}",
                (o - r).abs()
            );
        }
    }

    #[test]
    fn q4_0_multiple_groups() {
        let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let (data, scales) = quantize_q4_0(&original, 32);
        assert_eq!(scales.len(), 2);
        let restored = dequantize_q4_0(&data, &scales, (2, 32));
        assert_eq!(restored.len(), 64);
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!(
                (o - r).abs() < 0.3,
                "original={o}, restored={r}",
            );
        }
    }
}
