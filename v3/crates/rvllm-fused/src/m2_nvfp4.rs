//! MiniMax-M2 NVFP4 helpers for the Rust TPU path.
//!
//! This is the Rust-side ground truth for the lower-level Mosaic/custom-call
//! route: shape validation, exact NVFP4 decode semantics, and an MLIR scaffold
//! emitter with the real M2 matmul signature.

use rvllm_core::{ConfigError, Result, RvllmError};

pub const NVFP4_GROUP: usize = 16;
pub const M2_MOSAIC_DEFAULT_BN: usize = 512;
pub const M2_MOSAIC_DEFAULT_BK: usize = 1024;
pub const M2_NVFP4_CUSTOM_CALL_TARGET: &str = "rvllm.m2.nvfp4_decode_bf16_matmul";

pub const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct M2Nvfp4MatmulShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct M2Nvfp4MosaicTilePlan {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct M2Nvfp4MosaicMemory {
    pub x_bf16_bytes: usize,
    pub rhs_packed_bytes: usize,
    pub rhs_scale_bytes: usize,
    pub rhs_decoded_bf16_bytes: usize,
    pub acc_f32_bytes: usize,
    pub out_bf16_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct M2Nvfp4CustomCallAbi {
    pub target: &'static str,
    pub shape: M2Nvfp4MatmulShape,
    pub tile: M2Nvfp4MosaicTilePlan,
    pub vmem_working_set_bytes: usize,
    pub packed_bytes: usize,
    pub scale_bytes: usize,
    pub out_bytes: usize,
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

    pub fn default_mosaic_plan(self) -> M2Nvfp4MosaicTilePlan {
        M2Nvfp4MosaicTilePlan {
            bm: self.m.min(8),
            bn: M2_MOSAIC_DEFAULT_BN.min(self.n),
            bk: M2_MOSAIC_DEFAULT_BK.min(self.k),
        }
    }

    pub fn mosaic_mlir(self, kernel_name: &str) -> Result<String> {
        self.mosaic_mlir_with_plan(kernel_name, self.default_mosaic_plan())
    }

    pub fn mosaic_mlir_with_plan(
        self,
        kernel_name: &str,
        plan: M2Nvfp4MosaicTilePlan,
    ) -> Result<String> {
        self.validate()?;
        plan.validate_for(self)?;
        if !is_mlir_symbol(kernel_name) {
            return Err(invalid("kernel_name", "must be an MLIR symbol"));
        }
        let abi = self.custom_call_abi(plan)?;
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
        rvllm.tile = "BM={bm},BN={bn},BK={bk}",
        rvllm.tiles = "M={tm},N={tn},K={tk}",
        rvllm.vmem_working_set_bytes = {working_set} : i64,
        rvllm.custom_call_target = "{target}",
        rvllm.lowering = "mosaic_custom_call",
        rvllm.lowering_plan = "tpu.load packed/scales -> unpack nibbles -> fp8/fp4 decode in VMEM/registers -> bf16 RHS tile -> tpu.matmul -> tpu.store"
      }} {{
    "{target}"(%x, %packed, %scales, %global_scale, %out) {{
      rvllm.tile_bm = {bm} : i64,
      rvllm.tile_bn = {bn} : i64,
      rvllm.tile_bk = {bk} : i64,
      rvllm.nvfp4_group = {group} : i64,
      rvllm.vmem_working_set_bytes = {working_set} : i64,
      rvllm.packed_bytes = {packed_bytes} : i64,
      rvllm.scale_bytes = {scale_bytes} : i64,
      rvllm.out_bytes = {out_bytes} : i64
    }} : (memref<{m}x{k}xbf16>, memref<{n}x{k_half}xi8>, memref<{n}x{k_scale}xi8>, memref<f32>, memref<{m}x{n}xbf16>) -> ()
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
            bm = plan.bm,
            bn = plan.bn,
            bk = plan.bk,
            tm = div_ceil(self.m, plan.bm),
            tn = div_ceil(self.n, plan.bn),
            tk = div_ceil(self.k, plan.bk),
            target = abi.target,
            working_set = abi.vmem_working_set_bytes,
            packed_bytes = abi.packed_bytes,
            scale_bytes = abi.scale_bytes,
            out_bytes = abi.out_bytes,
        ))
    }

    pub fn custom_call_abi(self, plan: M2Nvfp4MosaicTilePlan) -> Result<M2Nvfp4CustomCallAbi> {
        self.validate()?;
        plan.validate_for(self)?;
        let mem = plan.memory();
        Ok(M2Nvfp4CustomCallAbi {
            target: M2_NVFP4_CUSTOM_CALL_TARGET,
            shape: self,
            tile: plan,
            vmem_working_set_bytes: mem.total_bytes,
            packed_bytes: self.packed_len(),
            scale_bytes: self.scale_len(),
            out_bytes: self.out_len() * 2,
        })
    }
}

impl M2Nvfp4MosaicTilePlan {
    pub const fn new(bm: usize, bn: usize, bk: usize) -> Self {
        Self { bm, bn, bk }
    }

    pub fn validate_for(self, shape: M2Nvfp4MatmulShape) -> Result<()> {
        if self.bm == 0 || self.bn == 0 || self.bk == 0 {
            return Err(invalid("tile", "bm/bn/bk must be > 0"));
        }
        if self.bk % NVFP4_GROUP != 0 {
            return Err(invalid("bk", "must be a multiple of 16"));
        }
        if self.bm > shape.m || self.bn > shape.n || self.bk > shape.k {
            return Err(invalid("tile", "must not exceed shape"));
        }
        Ok(())
    }

    pub const fn rhs_packed_bytes(self) -> usize {
        self.bn * (self.bk / 2)
    }

    pub const fn rhs_scale_bytes(self) -> usize {
        self.bn * (self.bk / NVFP4_GROUP)
    }

    pub const fn memory(self) -> M2Nvfp4MosaicMemory {
        let x_bf16_bytes = self.bm * self.bk * 2;
        let rhs_packed_bytes = self.rhs_packed_bytes();
        let rhs_scale_bytes = self.rhs_scale_bytes();
        let rhs_decoded_bf16_bytes = self.bn * self.bk * 2;
        let acc_f32_bytes = self.bm * self.bn * 4;
        let out_bf16_bytes = self.bm * self.bn * 2;
        M2Nvfp4MosaicMemory {
            x_bf16_bytes,
            rhs_packed_bytes,
            rhs_scale_bytes,
            rhs_decoded_bf16_bytes,
            acc_f32_bytes,
            out_bf16_bytes,
            total_bytes: x_bf16_bytes
                + rhs_packed_bytes
                + rhs_scale_bytes
                + rhs_decoded_bf16_bytes
                + acc_f32_bytes
                + out_bf16_bytes,
        }
    }

    pub const fn fits_vmem(self, budget_bytes: usize) -> bool {
        self.memory().total_bytes <= budget_bytes
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

const fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
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
        let shape = M2Nvfp4MatmulShape {
            m: 8,
            n: 1536,
            k: 3072,
        };
        let mlir = shape.mosaic_mlir("rvllm_m2_nvfp4_matmul").unwrap();
        assert!(mlir.contains("memref<8x3072xbf16>"));
        assert!(mlir.contains("memref<1536x1536xi8>"));
        assert!(mlir.contains("memref<1536x192xi8>"));
        assert!(mlir.contains("rvllm.tile = \"BM=8,BN=512,BK=1024\""));
        assert!(mlir.contains("rvllm.vmem_working_set_bytes = 1384448 : i64"));
        assert!(mlir.contains("rvllm.custom_call_target = \"rvllm.m2.nvfp4_decode_bf16_matmul\""));
        assert!(mlir.contains("rvllm.lowering = \"mosaic_custom_call\""));
        assert!(mlir.contains(
            "\"rvllm.m2.nvfp4_decode_bf16_matmul\"(%x, %packed, %scales, %global_scale, %out)"
        ));
        assert!(mlir.contains("rvllm.packed_bytes = 2359296 : i64"));
        assert!(mlir.contains("rvllm.scale_bytes = 294912 : i64"));
        assert!(mlir.contains("rvllm.out_bytes = 24576 : i64"));
    }

    #[test]
    fn custom_call_abi_pins_b8_gate_up_contract() {
        let shape = M2Nvfp4MatmulShape {
            m: 8,
            n: 1536,
            k: 3072,
        };
        let plan = M2Nvfp4MosaicTilePlan::new(8, 512, 1024);
        let abi = shape.custom_call_abi(plan).unwrap();
        assert_eq!(abi.target, M2_NVFP4_CUSTOM_CALL_TARGET);
        assert_eq!(abi.shape, shape);
        assert_eq!(abi.tile, plan);
        assert_eq!(abi.vmem_working_set_bytes, 1_384_448);
        assert_eq!(abi.packed_bytes, 2_359_296);
        assert_eq!(abi.scale_bytes, 294_912);
        assert_eq!(abi.out_bytes, 24_576);
    }

    #[test]
    fn mosaic_tile_plan_fits_gate_up_and_down_shapes() {
        let gate_up = M2Nvfp4MatmulShape {
            m: 8,
            n: 1536,
            k: 3072,
        };
        let down = M2Nvfp4MatmulShape {
            m: 8,
            n: 3072,
            k: 1536,
        };
        let plan = M2Nvfp4MosaicTilePlan::new(8, 512, 1024);
        plan.validate_for(gate_up).unwrap();
        plan.validate_for(down).unwrap();
        let mem = plan.memory();
        assert_eq!(mem.x_bf16_bytes, 16_384);
        assert_eq!(mem.rhs_packed_bytes, 262_144);
        assert_eq!(mem.rhs_scale_bytes, 32_768);
        assert_eq!(mem.rhs_decoded_bf16_bytes, 1_048_576);
        assert_eq!(mem.acc_f32_bytes, 16_384);
        assert_eq!(mem.out_bf16_bytes, 8_192);
        assert_eq!(mem.total_bytes, 1_384_448);
        assert!(plan.fits_vmem(2 * 1024 * 1024));
    }

    #[test]
    fn matmul_ref_matches_expanded_dequant_on_deterministic_tensor() {
        let shape = M2Nvfp4MatmulShape { m: 3, n: 5, k: 32 };
        let x: Vec<f32> = (0..shape.x_len())
            .map(|i| ((i * 17 + 3) % 29) as f32 / 7.0 - 2.0)
            .collect();
        let packed: Vec<u8> = (0..shape.packed_len())
            .map(|i| ((i * 13 + 7) & 0xff) as u8)
            .collect();
        let fp8_scales = [0x30u8, 0x34, 0x38, 0x3c, 0x40, 0x44, 0xb8];
        let scales: Vec<u8> = (0..shape.scale_len())
            .map(|i| fp8_scales[i % fp8_scales.len()])
            .collect();
        let global_scale = 0.375;
        let mut got = vec![0.0f32; shape.out_len()];
        nvfp4_matmul_ref(&x, &packed, &scales, global_scale, shape, &mut got).unwrap();

        let mut expanded = vec![0.0f32; shape.n * shape.k];
        for n in 0..shape.n {
            for k in 0..shape.k {
                expanded[n * shape.k + k] =
                    nvfp4_weight_at(&packed, &scales, global_scale, shape, n, k);
            }
        }

        let mut want = vec![0.0f32; shape.out_len()];
        for m in 0..shape.m {
            for n in 0..shape.n {
                let mut acc = 0.0f32;
                for k in 0..shape.k {
                    acc += x[m * shape.k + k] * expanded[n * shape.k + k];
                }
                want[m * shape.n + n] = acc;
            }
        }
        assert_eq!(got, want);
    }
}
