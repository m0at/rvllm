use rayon::prelude::*;
use rvllm_core::{ConfigError, Result, RvllmError};

use crate::m2_nvfp4::{decode_fp4_e2m1, decode_fp8_e4m3, nvfp4_weight_at, M2Nvfp4MatmulShape};

pub const M2_INT8_CUSTOM_CALL_TARGET: &str = "rvllm.m2.int8_bf16_matmul";
pub const M2_INT8_CUSTOM_CALL_ABI_VERSION: u32 = 1;
pub const M2_INT8_DESCRIPTOR_FORMAT: &str = "rvllm.m2.int8.custom_call.v1";

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct M2Int8KernelDescriptor {
    pub target: &'static str,
    pub shape: M2Nvfp4MatmulShape,
    pub x_dtype: &'static str,
    pub packed_dtype: &'static str,
    pub row_scale_dtype: &'static str,
    pub out_dtype: &'static str,
}

impl M2Int8KernelDescriptor {
    pub const fn new(shape: M2Nvfp4MatmulShape) -> Self {
        Self {
            target: M2_INT8_CUSTOM_CALL_TARGET,
            shape,
            x_dtype: "bf16",
            packed_dtype: "s8",
            row_scale_dtype: "f32",
            out_dtype: "bf16",
        }
    }

    pub fn validate(self) -> Result<()> {
        self.shape.validate()?;
        if self.target != M2_INT8_CUSTOM_CALL_TARGET {
            return Err(invalid("target", "must be the M2 int8 custom-call target"));
        }
        if self.x_dtype != "bf16" || self.packed_dtype != "s8" {
            return Err(invalid("dtype", "expected bf16 activations and s8 weights"));
        }
        if self.row_scale_dtype != "f32" || self.out_dtype != "bf16" {
            return Err(invalid("dtype", "expected f32 row scales and bf16 output"));
        }
        Ok(())
    }

    pub fn descriptor_inline(self) -> Result<String> {
        self.validate()?;
        Ok(format!(
            "format={format};abi_version={abi_version};target={target};x_dtype={x_dtype};packed_dtype={packed_dtype};row_scale_dtype={row_scale_dtype};out_dtype={out_dtype};m={m};n={n};k={k};x_dims={m}x{k};weight_dims={n}x{k};row_scale_dims={n};out_dims={m}x{n}",
            format = M2_INT8_DESCRIPTOR_FORMAT,
            abi_version = M2_INT8_CUSTOM_CALL_ABI_VERSION,
            target = self.target,
            x_dtype = self.x_dtype,
            packed_dtype = self.packed_dtype,
            row_scale_dtype = self.row_scale_dtype,
            out_dtype = self.out_dtype,
            m = self.shape.m,
            n = self.shape.n,
            k = self.shape.k,
        ))
    }
}

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

pub fn m2_int8_fixed_tile_parity_check() -> Result<()> {
    let shape = M2Nvfp4MatmulShape { m: 8, n: 16, k: 64 };
    let mut packed = vec![0u8; shape.packed_len()];
    for (idx, byte) in packed.iter_mut().enumerate() {
        *byte = (((idx * 13 + 7) as u8) << 4) | ((idx * 5 + 3) as u8 & 0x0f);
    }
    let scales = (0..shape.scale_len())
        .map(|idx| {
            const S: [u8; 7] = [0x08, 0x20, 0x30, 0x38, 0x40, 0x50, 0x70];
            S[idx % S.len()]
        })
        .collect::<Vec<_>>();
    let x = (0..shape.x_len())
        .map(|i| ((i.wrapping_mul(19) % 37) as f32 - 18.0) / 11.0)
        .collect::<Vec<_>>();
    let mut i8w = vec![0i8; shape.n * shape.k];
    let mut row_scales = vec![0.0f32; shape.n];
    nvfp4_to_int8_matrix(&packed, &scales, 1.0, shape, &mut i8w, &mut row_scales)?;

    let mut out = vec![0.0f32; shape.out_len()];
    int8_matmul_ref(&x, &i8w, &row_scales, shape, &mut out)?;

    let decoded = i8w
        .chunks_exact(shape.k)
        .zip(row_scales.iter())
        .flat_map(|(row, scale)| row.iter().map(move |&w| w as f32 * *scale))
        .collect::<Vec<_>>();
    for m in 0..shape.m {
        for n in 0..shape.n {
            let mut acc = 0.0f32;
            for k in 0..shape.k {
                acc += x[m * shape.k + k] * decoded[n * shape.k + k];
            }
            if out[m * shape.n + n].to_bits() != acc.to_bits() {
                return Err(invalid("parity", "fixed int8 tile f32 accumulators disagreed"));
            }
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
    use half::bf16;

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

        assert_int8_zero_ulp_bf16(
            "all-zero sub-BM/non-BK",
            M2Nvfp4MatmulShape {
                m: 4,
                n: 5,
                k: 1040,
            },
            EdgePattern::AllZero,
        );
        assert_int8_zero_ulp_bf16(
            "one-hot sub-BM/non-BK",
            M2Nvfp4MatmulShape {
                m: 4,
                n: 7,
                k: 1040,
            },
            EdgePattern::OneHot,
        );
        assert_int8_zero_ulp_bf16(
            "full-range scales sub-BM/non-BK",
            M2Nvfp4MatmulShape {
                m: 4,
                n: 11,
                k: 1040,
            },
            EdgePattern::FullRangeScales,
        );
    }

    #[test]
    fn int8_descriptor_pins_new_symbol_and_abi() {
        let shape = M2Nvfp4MatmulShape {
            m: 8,
            n: 512,
            k: 1040,
        };
        let desc = M2Int8KernelDescriptor::new(shape);
        desc.validate().unwrap();
        assert_eq!(desc.target, M2_INT8_CUSTOM_CALL_TARGET);
        assert_eq!(desc.packed_dtype, "s8");
        assert_eq!(
            desc.descriptor_inline().unwrap(),
            "format=rvllm.m2.int8.custom_call.v1;abi_version=1;target=rvllm.m2.int8_bf16_matmul;x_dtype=bf16;packed_dtype=s8;row_scale_dtype=f32;out_dtype=bf16;m=8;n=512;k=1040;x_dims=8x1040;weight_dims=512x1040;row_scale_dims=512;out_dims=8x512"
        );
    }

    #[test]
    fn fixed_tile_parity_gate_is_bit_exact() {
        m2_int8_fixed_tile_parity_check().unwrap();
    }

    enum EdgePattern {
        AllZero,
        OneHot,
        FullRangeScales,
    }

    fn assert_int8_zero_ulp_bf16(case: &str, shape: M2Nvfp4MatmulShape, pattern: EdgePattern) {
        let mut packed = vec![0u8; shape.packed_len()];
        let mut scales = vec![0x38u8; shape.scale_len()];
        match pattern {
            EdgePattern::AllZero => {
                fill_full_range_scales(&mut scales);
            }
            EdgePattern::OneHot => {
                for row in 0..shape.n {
                    let col = (row * 149 + 17) % shape.k;
                    set_nvfp4_nibble(&mut packed, shape, row, col, 0x7);
                }
            }
            EdgePattern::FullRangeScales => {
                fill_full_range_scales(&mut scales);
                for row in 0..shape.n {
                    let block = row % (shape.k / 16);
                    let col = block * 16 + row % 16;
                    set_nvfp4_nibble(&mut packed, shape, row, col, 0x7);
                }
            }
        }
        let x = deterministic_x(shape);
        let mut i8w = vec![0i8; shape.n * shape.k];
        let mut row_scales = vec![0.0f32; shape.n];
        nvfp4_to_int8_matrix(&packed, &scales, 1.0, shape, &mut i8w, &mut row_scales).unwrap();

        let mut int8_out = vec![0.0f32; shape.out_len()];
        let mut nvfp4_out = vec![0.0f32; shape.out_len()];
        let mut canonical_out = vec![0.0f32; shape.out_len()];
        int8_matmul_ref(&x, &i8w, &row_scales, shape, &mut int8_out).unwrap();
        nvfp4_matmul_ref(&x, &packed, &scales, 1.0, shape, &mut nvfp4_out).unwrap();
        canonical_int8_matmul_ref(&x, &i8w, &row_scales, shape, &mut canonical_out);

        let int8_bits = bf16_output_bits(&int8_out);
        let nvfp4_bits = bf16_output_bits(&nvfp4_out);
        let canonical_bits = bf16_output_bits(&canonical_out);
        assert_eq!(int8_bits, nvfp4_bits, "{case}: int8 vs nvfp4");
        assert_eq!(int8_bits, canonical_bits, "{case}");
    }

    fn deterministic_x(shape: M2Nvfp4MatmulShape) -> Vec<f32> {
        (0..shape.x_len())
            .map(|i| ((i.wrapping_mul(17) % 31) as f32 - 15.0) / 8.0)
            .collect()
    }

    fn canonical_int8_matmul_ref(
        x: &[f32],
        weights: &[i8],
        row_scales: &[f32],
        shape: M2Nvfp4MatmulShape,
        out: &mut [f32],
    ) {
        let decoded: Vec<f32> = weights
            .chunks_exact(shape.k)
            .zip(row_scales.iter())
            .flat_map(|(row, scale)| row.iter().map(move |&w| w as f32 * *scale))
            .collect();
        for m in 0..shape.m {
            for n in 0..shape.n {
                let mut acc = 0.0f32;
                for k in 0..shape.k {
                    acc += x[m * shape.k + k] * decoded[n * shape.k + k];
                }
                out[m * shape.n + n] = acc;
            }
        }
    }

    fn fill_full_range_scales(scales: &mut [u8]) {
        const FINITE_POSITIVE_E4M3: [u8; 11] = [
            0x01, 0x08, 0x10, 0x20, 0x30, 0x38, 0x40, 0x50, 0x60, 0x70, 0x76,
        ];
        for (idx, scale) in scales.iter_mut().enumerate() {
            *scale = FINITE_POSITIVE_E4M3[idx % FINITE_POSITIVE_E4M3.len()];
        }
    }

    fn set_nvfp4_nibble(
        packed: &mut [u8],
        shape: M2Nvfp4MatmulShape,
        row: usize,
        col: usize,
        nibble: u8,
    ) {
        let idx = row * (shape.k / 2) + col / 2;
        if col & 1 == 0 {
            packed[idx] = (packed[idx] & 0xf0) | (nibble & 0x0f);
        } else {
            packed[idx] = (packed[idx] & 0x0f) | ((nibble & 0x0f) << 4);
        }
    }

    fn bf16_output_bits(out: &[f32]) -> Vec<u16> {
        out.iter().map(|&v| bf16::from_f32(v).to_bits()).collect()
    }
}
