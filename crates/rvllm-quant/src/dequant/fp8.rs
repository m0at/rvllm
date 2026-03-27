/// Precomputed lookup table: FP8 E4M3 byte -> f32.
/// 256 entries, built at compile time via const fn.
pub(crate) const FP8_E4M3_LUT: [f32; 256] = build_fp8_lut();

const fn build_fp8_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    let mut i: u16 = 0;
    while i < 256 {
        let byte = i as u8;
        let sign = (byte >> 7) & 1;
        let exp = (byte >> 3) & 0x0F;
        let mantissa = byte & 0x07;

        let val = if exp == 0 && mantissa == 0 {
            0.0f32
        } else if exp == 0 {
            // Subnormal: 2^(1-7) * (mantissa / 8) = 2^(-6) * mantissa / 8
            // 2^(-6) = 1/64 = 0.015625
            0.015625 * (mantissa as f32 / 8.0)
        } else {
            // Normal: 2^(exp-7) * (1 + mantissa/8)
            // Use bit manipulation: construct f32 from parts
            // 2^n for small integer n via repeated multiply
            let n = exp as i32 - 7;
            let pow = const_pow2(n);
            pow * (1.0 + mantissa as f32 / 8.0)
        };

        lut[i as usize] = if sign == 1 { -val } else { val };
        i += 1;
    }
    lut
}

/// const-compatible 2^n for n in [-6, 8]
const fn const_pow2(n: i32) -> f32 {
    match n {
        -6 => 1.0 / 64.0,
        -5 => 1.0 / 32.0,
        -4 => 1.0 / 16.0,
        -3 => 1.0 / 8.0,
        -2 => 1.0 / 4.0,
        -1 => 1.0 / 2.0,
        0 => 1.0,
        1 => 2.0,
        2 => 4.0,
        3 => 8.0,
        4 => 16.0,
        5 => 32.0,
        6 => 64.0,
        7 => 128.0,
        8 => 256.0,
        _ => 1.0,
    }
}

/// Dequantize FP8 (E4M3) format: each byte is an 8-bit float with per-tensor scale.
/// E4M3 layout: 1 sign, 4 exponent, 3 mantissa bits.
/// We use a single scale factor (scales[0]) for the entire tensor.
#[inline]
pub fn dequantize_fp8(data: &[u8], scales: &[f32], shape: (usize, usize)) -> Vec<f32> {
    let (rows, cols) = shape;
    let total = rows * cols;
    let scale = if scales.is_empty() { 1.0 } else { scales[0] };
    let mut output = vec![0.0f32; total];

    let src = &data[..total];
    let (chunks, tail) = src.split_at(total & !7);
    let (out_chunks, out_tail) = output.split_at_mut(chunks.len());

    // Process 8 bytes at a time for SIMD-friendly access
    for (in8, out8) in chunks.chunks_exact(8).zip(out_chunks.chunks_exact_mut(8)) {
        out8[0] = FP8_E4M3_LUT[in8[0] as usize] * scale;
        out8[1] = FP8_E4M3_LUT[in8[1] as usize] * scale;
        out8[2] = FP8_E4M3_LUT[in8[2] as usize] * scale;
        out8[3] = FP8_E4M3_LUT[in8[3] as usize] * scale;
        out8[4] = FP8_E4M3_LUT[in8[4] as usize] * scale;
        out8[5] = FP8_E4M3_LUT[in8[5] as usize] * scale;
        out8[6] = FP8_E4M3_LUT[in8[6] as usize] * scale;
        out8[7] = FP8_E4M3_LUT[in8[7] as usize] * scale;
    }

    // Handle remaining elements
    for (dst, &byte) in out_tail.iter_mut().zip(tail.iter()) {
        *dst = FP8_E4M3_LUT[byte as usize] * scale;
    }

    output
}

/// Convert a single FP8 E4M3 byte to f32 (uses LUT).
#[inline]
fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    FP8_E4M3_LUT[byte as usize]
}

/// Convert f32 to FP8 E4M3 byte.
fn f32_to_fp8_e4m3(val: f32) -> u8 {
    if val == 0.0 {
        return if val.is_sign_negative() { 0x80 } else { 0x00 };
    }

    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();
    let bias = 7i32;

    // Max representable: 2^8 * (1 + 7/8) = 448
    // Min normal: 2^(-6) = 0.015625
    // Min subnormal: 2^(-6) * (1/8) = 0.001953125
    let max_val = 448.0f32;
    let min_subnormal = 2.0f32.powi(1 - bias) / 8.0;

    if abs_val > max_val {
        // Clamp to max
        return (sign << 7) | 0x7F; // exp=15, mantissa=7
    }

    if abs_val < min_subnormal {
        // Too small, round to zero
        return sign << 7;
    }

    // Try subnormal first
    let min_normal = 2.0f32.powi(1 - bias);
    if abs_val < min_normal {
        let mantissa = (abs_val / min_normal * 8.0).round() as u8;
        let mantissa = mantissa.min(7);
        return (sign << 7) | mantissa;
    }

    // Normal number
    let log2 = abs_val.log2().floor() as i32;
    let exp = (log2 + bias).clamp(1, 15) as u8;
    let pow = 2.0f32.powi(exp as i32 - bias);
    let mantissa = ((abs_val / pow - 1.0) * 8.0).round().clamp(0.0, 7.0) as u8;

    (sign << 7) | (exp << 3) | mantissa
}

/// Quantize f32 values to FP8 E4M3 format (for testing).
pub fn quantize_fp8(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    // Compute per-tensor scale
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let fp8_max = 448.0f32;
    let scale = if max_abs == 0.0 {
        1.0
    } else {
        max_abs / fp8_max
    };
    let inv_scale = 1.0 / scale;

    let data: Vec<u8> = values
        .iter()
        .map(|&v| f32_to_fp8_e4m3(v * inv_scale))
        .collect();

    (data, vec![scale])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_zero() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(f32_to_fp8_e4m3(0.0), 0x00);
    }

    #[test]
    fn fp8_one() {
        // 1.0 = 2^0 * (1 + 0/8), exp=7, mantissa=0
        let byte = f32_to_fp8_e4m3(1.0);
        let back = fp8_e4m3_to_f32(byte);
        assert!((back - 1.0).abs() < 0.2, "expected ~1.0, got {back}");
    }

    #[test]
    fn fp8_negative() {
        let byte = f32_to_fp8_e4m3(-2.0);
        let back = fp8_e4m3_to_f32(byte);
        assert!((back - (-2.0)).abs() < 0.5, "expected ~-2.0, got {back}");
    }

    #[test]
    fn fp8_round_trip() {
        let shape = (1, 64);
        let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let (data, scales) = quantize_fp8(&original);
        let restored = dequantize_fp8(&data, &scales, shape);

        assert_eq!(restored.len(), 64);
        for (o, r) in original.iter().zip(restored.iter()) {
            // FP8 has limited precision, allow generous tolerance
            let tol = o.abs() * 0.25 + 0.5;
            assert!(
                (o - r).abs() < tol,
                "original={o}, restored={r}, diff={}, tol={tol}",
                (o - r).abs()
            );
        }
    }

    #[test]
    fn fp8_small_values() {
        let original = vec![0.001, 0.01, 0.1, 1.0, 10.0];
        let (data, scales) = quantize_fp8(&original);
        let restored = dequantize_fp8(&data, &scales, (1, 5));

        assert_eq!(restored.len(), 5);
        // Just verify they're in the right ballpark
        for (o, r) in original.iter().zip(restored.iter()) {
            let tol = o.abs() * 0.5 + 0.05;
            assert!(
                (o - r).abs() < tol,
                "original={o}, restored={r}",
            );
        }
    }
}
