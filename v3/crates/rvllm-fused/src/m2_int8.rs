use rayon::prelude::*;
use rvllm_core::{ConfigError, Result, RvllmError};

use crate::m2_nvfp4::{decode_fp4_e2m1, decode_fp8_e4m3, nvfp4_weight_at, M2Nvfp4MatmulShape};

pub fn nvfp4_to_int8_matrix(
    packed: &[u8],
    scales: &[u8],
    global_scale: f32,
    shape: M2Nvfp4MatmulShape,
    out_i8: &mut [i8],
    row_scales: &mut [f32],
) -> Result<()> {
    shape.validate()?;
    if packed.len() != shape.packed_len() {
        return Err(invalid("packed", "length does not match n*k/2"));
    }
    if scales.len() != shape.scale_len() {
        return Err(invalid("scales", "length does not match n*k/16"));
    }
    if out_i8.len() != shape.n * shape.k {
        return Err(invalid("out_i8", "length does not match n*k"));
    }
    if row_scales.len() != shape.n {
        return Err(invalid("row_scales", "length does not match n"));
    }

    out_i8
        .par_chunks_mut(shape.k)
        .zip(row_scales.par_iter_mut())
        .enumerate()
        .try_for_each(|(row, (out_row, row_scale_slot))| -> Result<()> {
            let mut max_abs = 0.0f32;
            for col in 0..shape.k {
                let v = nvfp4_weight_at(packed, scales, global_scale, shape, row, col);
                if !v.is_finite() {
                    return Err(invalid("nvfp4", "decoded weight is not finite"));
                }
                max_abs = max_abs.max(v.abs());
            }

            let row_scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            *row_scale_slot = row_scale;
            for (col, out) in out_row.iter_mut().enumerate() {
                let v = nvfp4_weight_at(packed, scales, global_scale, shape, row, col);
                *out = (v / row_scale).round().clamp(-127.0, 127.0) as i8;
            }
            Ok(())
        })?;
    Ok(())
}

pub fn int8_weight_at(
    weights: &[i8],
    row_scales: &[f32],
    shape: M2Nvfp4MatmulShape,
    row: usize,
    col: usize,
) -> f32 {
    weights[row * shape.k + col] as f32 * row_scales[row]
}

pub fn int8_matmul_ref(
    x: &[f32],
    weights: &[i8],
    row_scales: &[f32],
    shape: M2Nvfp4MatmulShape,
    out: &mut [f32],
) -> Result<()> {
    shape.validate()?;
    if x.len() != shape.x_len() {
        return Err(invalid("x", "length does not match m*k"));
    }
    if weights.len() != shape.n * shape.k {
        return Err(invalid("weights", "length does not match n*k"));
    }
    if row_scales.len() != shape.n {
        return Err(invalid("row_scales", "length does not match n"));
    }
    if out.len() != shape.out_len() {
        return Err(invalid("out", "length does not match m*n"));
    }

    for m in 0..shape.m {
        for n in 0..shape.n {
            let mut acc = 0.0f32;
            for k in 0..shape.k {
                acc += x[m * shape.k + k] * int8_weight_at(weights, row_scales, shape, n, k);
            }
            out[m * shape.n + n] = acc;
        }
    }
    Ok(())
}

pub fn nvfp4_scalar_weight(
    packed_byte: u8,
    high_nibble: bool,
    scale_bits: u8,
    global_scale: f32,
) -> f32 {
    let nibble = if high_nibble {
        packed_byte >> 4
    } else {
        packed_byte & 0x0f
    };
    decode_fp4_e2m1(nibble) * decode_fp8_e4m3(scale_bits) * global_scale
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_int8",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m2_nvfp4::nvfp4_matmul_ref;

    #[test]
    fn scalar_decode_bakes_global_scale() {
        let packed = 0x12;
        let scale_one = 0x38;
        assert_eq!(nvfp4_scalar_weight(packed, false, scale_one, 3.0), 3.0);
        assert_eq!(nvfp4_scalar_weight(packed, true, scale_one, 3.0), 1.5);
    }

    #[test]
    fn converts_nvfp4_rows_to_i8_with_row_scales() {
        let shape = M2Nvfp4MatmulShape { m: 1, n: 2, k: 16 };
        let packed = vec![0x11; shape.packed_len()];
        let scales = vec![0x38; shape.scale_len()];
        let mut out = vec![0i8; shape.n * shape.k];
        let mut row_scales = vec![0.0f32; shape.n];
        nvfp4_to_int8_matrix(&packed, &scales, 2.0, shape, &mut out, &mut row_scales).unwrap();
        assert_eq!(row_scales, vec![1.0 / 127.0; 2]);
        assert!(out.iter().all(|&x| x == 127));
    }

    #[test]
    fn int8_ref_tracks_nvfp4_ref_after_quantization() {
        let shape = M2Nvfp4MatmulShape { m: 2, n: 3, k: 16 };
        let packed = vec![
            0x12, 0x34, 0x56, 0x76, 0x21, 0x43, 0x65, 0x67, 0x9a, 0xbc, 0xde, 0xfe, 0xa9, 0xcb,
            0xed, 0xef, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        ];
        let scales = vec![0x38; shape.scale_len()];
        let x = vec![
            0.25, -0.5, 1.0, 0.75, -1.25, 0.5, 0.125, -0.25, 1.5, -0.75, 0.625, -0.375, 0.875,
            -1.0, 0.25, 0.5, -0.25, 0.5, -1.0, -0.75, 1.25, -0.5, -0.125, 0.25, -1.5, 0.75, -0.625,
            0.375, -0.875, 1.0, -0.25, -0.5,
        ];
        let mut i8w = vec![0i8; shape.n * shape.k];
        let mut row_scales = vec![0.0f32; shape.n];
        nvfp4_to_int8_matrix(&packed, &scales, 0.75, shape, &mut i8w, &mut row_scales).unwrap();

        let mut nvfp4_out = vec![0.0f32; shape.out_len()];
        let mut int8_out = vec![0.0f32; shape.out_len()];
        nvfp4_matmul_ref(&x, &packed, &scales, 0.75, shape, &mut nvfp4_out).unwrap();
        int8_matmul_ref(&x, &i8w, &row_scales, shape, &mut int8_out).unwrap();

        for (a, b) in nvfp4_out.iter().zip(int8_out.iter()) {
            assert!((a - b).abs() < 0.08, "{a} vs {b}");
        }
    }
}
