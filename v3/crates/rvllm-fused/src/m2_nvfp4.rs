//! MiniMax-M2 NVFP4 helpers for the Rust TPU path.
//!
//! This is the Rust-side ground truth for the lower-level Mosaic/custom-call
//! route: shape validation, exact NVFP4 decode semantics, and an MLIR scaffold
//! emitter with the real M2 matmul signature.

use rvllm_core::{ConfigError, Result, RvllmError};

pub const NVFP4_GROUP: usize = 16;

pub const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct M2Nvfp4MatmulShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl M2Nvfp4MatmulShape {
    pub fn validate(self) -> Result<()> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(invalid("shape", "m/n/k must be > 0"));
        }
        if self.k % NVFP4_GROUP != 0 {
            return Err(invalid("k", "must be a multiple of 16"));
        }
        Ok(())
    }

    pub const fn packed_len(self) -> usize {
        self.n * (self.k / 2)
    }

    pub const fn scale_len(self) -> usize {
        self.n * (self.k / NVFP4_GROUP)
    }

    pub const fn x_len(self) -> usize {
        self.m * self.k
    }

    pub const fn out_len(self) -> usize {
        self.m * self.n
    }

    pub fn mosaic_mlir(self, kernel_name: &str) -> Result<String> {
        self.validate()?;
        if !is_mlir_symbol(kernel_name) {
            return Err(invalid("kernel_name", "must be an MLIR symbol"));
        }
        Ok(format!(
            r#"module attributes {{rvllm.kind = "m2_nvfp4_mosaic"}} {{
  func.func @{kernel_name}(
      %x: memref<{m}x{k}xbf16>,
      %packed: memref<{n}x{k_half}xi8>,
      %scales: memref<{n}x{k_scale}xi8>,
      %global_scale: memref<f32>,
      %out: memref<{m}x{n}xbf16>) attributes {{
        rvllm.signature = "x_bf16,packed_u8,scale_fp8,global_f32,out_bf16",
        rvllm.layout = "row_major",
        rvllm.nvfp4_group = {group} : i64,
        rvllm.lowering = "mosaic_custom_call"
      }} {{
    // TODO: lower body to Mosaic TPU dialect:
    // 1. load packed uint8 and FP8 E4M3 scales from HBM
    // 2. decode FP4 E2M1 nibbles and FP8 scales in VMEM/registers
    // 3. feed bf16 RHS tiles to TPU matmul
    // 4. write bf16 output tile to %out
    return
  }}
}}
"#,
            m = self.m,
            n = self.n,
            k = self.k,
            k_half = self.k / 2,
            k_scale = self.k / NVFP4_GROUP,
            group = NVFP4_GROUP,
        ))
    }
}

pub fn decode_fp4_e2m1(nibble: u8) -> f32 {
    FP4_E2M1_LUT[(nibble & 0x0f) as usize]
}

pub fn decode_fp8_e4m3(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 3) & 0x0f;
    let mant = bits & 0x07;
    if exp == 0x0f && mant == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    }
}

pub fn nvfp4_weight_at(
    packed: &[u8],
    scales: &[u8],
    global_scale: f32,
    shape: M2Nvfp4MatmulShape,
    row: usize,
    col: usize,
) -> f32 {
    debug_assert!(row < shape.n);
    debug_assert!(col < shape.k);
    let byte = packed[row * (shape.k / 2) + col / 2];
    let nibble = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 };
    let block_scale = scales[row * (shape.k / NVFP4_GROUP) + col / NVFP4_GROUP];
    decode_fp4_e2m1(nibble) * decode_fp8_e4m3(block_scale) * global_scale
}

pub fn nvfp4_matmul_ref(
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    global_scale: f32,
    shape: M2Nvfp4MatmulShape,
    out: &mut [f32],
) -> Result<()> {
    shape.validate()?;
    if x.len() != shape.x_len() {
        return Err(invalid("x", "length does not match m*k"));
    }
    if packed.len() != shape.packed_len() {
        return Err(invalid("packed", "length does not match n*k/2"));
    }
    if scales.len() != shape.scale_len() {
        return Err(invalid("scales", "length does not match n*k/16"));
    }
    if out.len() != shape.out_len() {
        return Err(invalid("out", "length does not match m*n"));
    }

    for m in 0..shape.m {
        for n in 0..shape.n {
            let mut acc = 0.0f32;
            for k in 0..shape.k {
                let xv = x[m * shape.k + k];
                let wv = nvfp4_weight_at(packed, scales, global_scale, shape, n, k);
                acc += xv * wv;
            }
            out[m * shape.n + n] = acc;
        }
    }
    Ok(())
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_nvfp4",
    )
}

fn is_mlir_symbol(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_lut_matches_modelopt_order() {
        assert_eq!(decode_fp4_e2m1(0), 0.0);
        assert_eq!(decode_fp4_e2m1(1), 0.5);
        assert_eq!(decode_fp4_e2m1(7), 6.0);
        assert_eq!(decode_fp4_e2m1(8), -0.0);
        assert_eq!(decode_fp4_e2m1(15), -6.0);
    }

    #[test]
    fn fp8_e4m3_basic_values() {
        assert_eq!(decode_fp8_e4m3(0x38), 1.0);
        assert_eq!(decode_fp8_e4m3(0x40), 2.0);
        assert_eq!(decode_fp8_e4m3(0xb8), -1.0);
        assert_eq!(decode_fp8_e4m3(0x01), 2f32.powi(-9));
    }

    #[test]
    fn matmul_ref_decodes_low_then_high_nibbles() {
        let shape = M2Nvfp4MatmulShape { m: 1, n: 1, k: 16 };
        let x = vec![1.0f32; 16];
        let mut packed = vec![0u8; shape.packed_len()];
        for (i, b) in packed.iter_mut().enumerate() {
            *b = (((2 * i + 1) as u8) << 4) | (2 * i) as u8;
        }
        let scales = vec![0x38u8; shape.scale_len()];
        let mut out = vec![0.0f32; shape.out_len()];
        nvfp4_matmul_ref(&x, &packed, &scales, 1.0, shape, &mut out).unwrap();
        let expected: f32 = (0..16).map(|i| decode_fp4_e2m1(i)).sum();
        assert_eq!(out[0], expected);
    }

    #[test]
    fn emits_real_m2_signature() {
        let shape = M2Nvfp4MatmulShape { m: 8, n: 1536, k: 3072 };
        let mlir = shape.mosaic_mlir("rvllm_m2_nvfp4_matmul").unwrap();
        assert!(mlir.contains("memref<8x3072xbf16>"));
        assert!(mlir.contains("memref<1536x1536xi8>"));
        assert!(mlir.contains("memref<1536x192xi8>"));
        assert!(mlir.contains("rvllm.lowering = \"mosaic_custom_call\""));
    }
}
