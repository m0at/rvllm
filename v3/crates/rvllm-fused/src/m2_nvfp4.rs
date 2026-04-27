//! MiniMax-M2 NVFP4 helpers for the Rust TPU path.
//!
//! This is the Rust-side ground truth for the lower-level XLA custom-call route:
//! shape validation, exact NVFP4 decode semantics, and a deterministic MLIR
//! interface emitter with the real M2 matmul signature.

use rvllm_core::{ConfigError, Result, RvllmError};

pub const NVFP4_GROUP: usize = 16;
pub const M2_MOSAIC_DEFAULT_BN: usize = 512;
pub const M2_MOSAIC_DEFAULT_BK: usize = 1024;
pub const M2_NVFP4_CUSTOM_CALL_TARGET: &str = "rvllm.m2.nvfp4_decode_bf16_matmul";
pub const M2_NVFP4_CUSTOM_CALL_ABI_VERSION: u32 = 1;
pub const M2_NVFP4_DESCRIPTOR_FORMAT: &str = "rvllm.m2.nvfp4.custom_call.v1";

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
pub enum M2Nvfp4IoDType {
    Bf16,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum M2Nvfp4PackedDType {
    U8,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum M2Nvfp4ScaleDType {
    Fp8E4M3,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum M2Nvfp4ScalarDType {
    F32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct M2Nvfp4KernelDescriptor {
    pub target: &'static str,
    pub shape: M2Nvfp4MatmulShape,
    pub x_dtype: M2Nvfp4IoDType,
    pub packed_dtype: M2Nvfp4PackedDType,
    pub scale_dtype: M2Nvfp4ScaleDType,
    pub global_scale_dtype: M2Nvfp4ScalarDType,
    pub out_dtype: M2Nvfp4IoDType,
    pub block_size: usize,
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
    pub descriptor: M2Nvfp4KernelDescriptor,
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

    pub const fn kernel_descriptor(self) -> M2Nvfp4KernelDescriptor {
        M2Nvfp4KernelDescriptor::new(self)
    }

    pub fn default_mosaic_plan(self) -> M2Nvfp4MosaicTilePlan {
        M2Nvfp4MosaicTilePlan {
            bm: self.m.min(8),
            bn: M2_MOSAIC_DEFAULT_BN.min(self.n),
            bk: M2_MOSAIC_DEFAULT_BK.min(self.k),
        }
    }

    pub fn custom_call_mlir(self, kernel_name: &str) -> Result<String> {
        self.custom_call_mlir_with_plan(kernel_name, self.default_mosaic_plan())
    }

    pub fn custom_call_mlir_with_plan(
        self,
        kernel_name: &str,
        plan: M2Nvfp4MosaicTilePlan,
    ) -> Result<String> {
        self.kernel_descriptor()
            .custom_call_mlir_with_plan(kernel_name, plan)
    }

    pub fn mosaic_mlir(self, kernel_name: &str) -> Result<String> {
        self.custom_call_mlir(kernel_name)
    }

    pub fn mosaic_mlir_with_plan(
        self,
        kernel_name: &str,
        plan: M2Nvfp4MosaicTilePlan,
    ) -> Result<String> {
        self.custom_call_mlir_with_plan(kernel_name, plan)
    }

    pub fn custom_call_abi(self, plan: M2Nvfp4MosaicTilePlan) -> Result<M2Nvfp4CustomCallAbi> {
        self.kernel_descriptor().custom_call_abi(plan)
    }
}

impl M2Nvfp4IoDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
        }
    }

    pub const fn mlir_dtype(self) -> &'static str {
        self.as_str()
    }

    pub const fn elem_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
        }
    }
}

impl M2Nvfp4PackedDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
        }
    }

    pub const fn mlir_storage_dtype(self) -> &'static str {
        match self {
            Self::U8 => "i8",
        }
    }

    pub const fn elem_bytes(self) -> usize {
        match self {
            Self::U8 => 1,
        }
    }
}

impl M2Nvfp4ScaleDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp8E4M3 => "fp8_e4m3",
        }
    }

    pub const fn mlir_storage_dtype(self) -> &'static str {
        match self {
            Self::Fp8E4M3 => "i8",
        }
    }

    pub const fn elem_bytes(self) -> usize {
        match self {
            Self::Fp8E4M3 => 1,
        }
    }
}

impl M2Nvfp4ScalarDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
        }
    }

    pub const fn mlir_dtype(self) -> &'static str {
        self.as_str()
    }
}

impl M2Nvfp4KernelDescriptor {
    pub const fn new(shape: M2Nvfp4MatmulShape) -> Self {
        Self {
            target: M2_NVFP4_CUSTOM_CALL_TARGET,
            shape,
            x_dtype: M2Nvfp4IoDType::Bf16,
            packed_dtype: M2Nvfp4PackedDType::U8,
            scale_dtype: M2Nvfp4ScaleDType::Fp8E4M3,
            global_scale_dtype: M2Nvfp4ScalarDType::F32,
            out_dtype: M2Nvfp4IoDType::Bf16,
            block_size: NVFP4_GROUP,
        }
    }

    pub fn validate(self) -> Result<()> {
        self.shape.validate()?;
        if self.target != M2_NVFP4_CUSTOM_CALL_TARGET {
            return Err(invalid("target", "must be the M2 NVFP4 custom-call target"));
        }
        if self.block_size != NVFP4_GROUP {
            return Err(invalid("block_size", "must be 16"));
        }
        Ok(())
    }

    pub const fn x_dims(self) -> [usize; 2] {
        [self.shape.m, self.shape.k]
    }

    pub const fn packed_dims(self) -> [usize; 2] {
        [self.shape.n, self.shape.k / 2]
    }

    pub const fn scale_dims(self) -> [usize; 2] {
        [self.shape.n, self.shape.k / NVFP4_GROUP]
    }

    pub const fn out_dims(self) -> [usize; 2] {
        [self.shape.m, self.shape.n]
    }

    pub fn descriptor_text(self) -> Result<String> {
        self.validate()?;
        Ok(format!(
            "format={format}\nabi_version={abi_version}\ntarget={target}\nx_dtype={x_dtype}\npacked_dtype={packed_dtype}\nscale_dtype={scale_dtype}\nglobal_scale_dtype={global_scale_dtype}\nout_dtype={out_dtype}\nblock_size={block_size}\nm={m}\nn={n}\nk={k}\nx_dims={m}x{k}\npacked_dims={n}x{k_half}\nscale_dims={n}x{k_scale}\nout_dims={m}x{n}\n",
            format = M2_NVFP4_DESCRIPTOR_FORMAT,
            abi_version = M2_NVFP4_CUSTOM_CALL_ABI_VERSION,
            target = self.target,
            x_dtype = self.x_dtype.as_str(),
            packed_dtype = self.packed_dtype.as_str(),
            scale_dtype = self.scale_dtype.as_str(),
            global_scale_dtype = self.global_scale_dtype.as_str(),
            out_dtype = self.out_dtype.as_str(),
            block_size = self.block_size,
            m = self.shape.m,
            n = self.shape.n,
            k = self.shape.k,
            k_half = self.shape.k / 2,
            k_scale = self.shape.k / NVFP4_GROUP,
        ))
    }

    pub fn descriptor_inline(self) -> Result<String> {
        self.validate()?;
        Ok(format!(
            "format={format};abi_version={abi_version};target={target};x_dtype={x_dtype};packed_dtype={packed_dtype};scale_dtype={scale_dtype};global_scale_dtype={global_scale_dtype};out_dtype={out_dtype};block_size={block_size};m={m};n={n};k={k};x_dims={m}x{k};packed_dims={n}x{k_half};scale_dims={n}x{k_scale};out_dims={m}x{n}",
            format = M2_NVFP4_DESCRIPTOR_FORMAT,
            abi_version = M2_NVFP4_CUSTOM_CALL_ABI_VERSION,
            target = self.target,
            x_dtype = self.x_dtype.as_str(),
            packed_dtype = self.packed_dtype.as_str(),
            scale_dtype = self.scale_dtype.as_str(),
            global_scale_dtype = self.global_scale_dtype.as_str(),
            out_dtype = self.out_dtype.as_str(),
            block_size = self.block_size,
            m = self.shape.m,
            n = self.shape.n,
            k = self.shape.k,
            k_half = self.shape.k / 2,
            k_scale = self.shape.k / NVFP4_GROUP,
        ))
    }

    pub fn custom_call_abi(self, plan: M2Nvfp4MosaicTilePlan) -> Result<M2Nvfp4CustomCallAbi> {
        self.validate()?;
        plan.validate_for(self.shape)?;
        let mem = plan.memory();
        Ok(M2Nvfp4CustomCallAbi {
            target: self.target,
            descriptor: self,
            shape: self.shape,
            tile: plan,
            vmem_working_set_bytes: mem.total_bytes,
            packed_bytes: self.shape.packed_len() * self.packed_dtype.elem_bytes(),
            scale_bytes: self.shape.scale_len() * self.scale_dtype.elem_bytes(),
            out_bytes: self.shape.out_len() * self.out_dtype.elem_bytes(),
        })
    }

    pub fn custom_call_body(self, plan: M2Nvfp4MosaicTilePlan) -> Result<String> {
        let abi = self.custom_call_abi(plan)?;
        let descriptor = self.descriptor_inline()?;
        Ok(format!(
            r#"    "{target}"(%x, %packed, %scales, %global_scale, %out) {{
      rvllm.abi = "{format}",
      rvllm.abi_version = {abi_version} : i64,
      rvllm.descriptor = "{descriptor}",
      rvllm.target = "{target}",
      rvllm.x_dtype = "{x_dtype}",
      rvllm.packed_dtype = "{packed_dtype}",
      rvllm.scale_dtype = "{scale_dtype}",
      rvllm.global_scale_dtype = "{global_scale_dtype}",
      rvllm.out_dtype = "{out_dtype}",
      rvllm.block_size = {block_size} : i64,
      rvllm.x_dims = "{m}x{k}",
      rvllm.packed_dims = "{n}x{k_half}",
      rvllm.scale_dims = "{n}x{k_scale}",
      rvllm.out_dims = "{m}x{n}",
      rvllm.tile_bm = {bm} : i64,
      rvllm.tile_bn = {bn} : i64,
      rvllm.tile_bk = {bk} : i64,
      rvllm.tiles_m = {tm} : i64,
      rvllm.tiles_n = {tn} : i64,
      rvllm.tiles_k = {tk} : i64,
      rvllm.vmem_working_set_bytes = {working_set} : i64,
      rvllm.packed_bytes = {packed_bytes} : i64,
      rvllm.scale_bytes = {scale_bytes} : i64,
      rvllm.out_bytes = {out_bytes} : i64
    }} : ({x_ty}, {packed_ty}, {scales_ty}, {global_scale_ty}, {out_ty}) -> ()
"#,
            target = self.target,
            format = M2_NVFP4_DESCRIPTOR_FORMAT,
            abi_version = M2_NVFP4_CUSTOM_CALL_ABI_VERSION,
            descriptor = descriptor,
            x_dtype = self.x_dtype.as_str(),
            packed_dtype = self.packed_dtype.as_str(),
            scale_dtype = self.scale_dtype.as_str(),
            global_scale_dtype = self.global_scale_dtype.as_str(),
            out_dtype = self.out_dtype.as_str(),
            block_size = self.block_size,
            m = self.shape.m,
            n = self.shape.n,
            k = self.shape.k,
            k_half = self.shape.k / 2,
            k_scale = self.shape.k / NVFP4_GROUP,
            bm = abi.tile.bm,
            bn = abi.tile.bn,
            bk = abi.tile.bk,
            tm = div_ceil(self.shape.m, abi.tile.bm),
            tn = div_ceil(self.shape.n, abi.tile.bn),
            tk = div_ceil(self.shape.k, abi.tile.bk),
            working_set = abi.vmem_working_set_bytes,
            packed_bytes = abi.packed_bytes,
            scale_bytes = abi.scale_bytes,
            out_bytes = abi.out_bytes,
            x_ty = self.x_memref_ty(),
            packed_ty = self.packed_memref_ty(),
            scales_ty = self.scales_memref_ty(),
            global_scale_ty = self.global_scale_memref_ty(),
            out_ty = self.out_memref_ty(),
        ))
    }

    pub fn custom_call_mlir(self, kernel_name: &str) -> Result<String> {
        self.custom_call_mlir_with_plan(kernel_name, self.shape.default_mosaic_plan())
    }

    pub fn custom_call_mlir_with_plan(
        self,
        kernel_name: &str,
        plan: M2Nvfp4MosaicTilePlan,
    ) -> Result<String> {
        self.validate()?;
        if !is_mlir_symbol(kernel_name) {
            return Err(invalid("kernel_name", "must be an MLIR symbol"));
        }
        let abi = self.custom_call_abi(plan)?;
        let body = self.custom_call_body(plan)?;
        Ok(format!(
            r#"module attributes {{rvllm.kind = "m2_nvfp4_custom_call"}} {{
  func.func @{kernel_name}(
      %x: {x_ty},
      %packed: {packed_ty},
      %scales: {scales_ty},
      %global_scale: {global_scale_ty},
      %out: {out_ty}) attributes {{
        rvllm.signature = "x_bf16,packed_u8,scale_fp8_e4m3,global_f32,out_bf16",
        rvllm.abi = "{format}",
        rvllm.abi_version = {abi_version} : i64,
        rvllm.layout = "row_major",
        rvllm.block_size = {block_size} : i64,
        rvllm.tile = "BM={bm},BN={bn},BK={bk}",
        rvllm.tiles = "M={tm},N={tn},K={tk}",
        rvllm.vmem_working_set_bytes = {working_set} : i64,
        rvllm.custom_call_target = "{target}",
        rvllm.lowering = "rust_xla_custom_call",
        rvllm.lowering_plan = "XLA custom call receives bf16 activations, packed u8 NVFP4 weights, fp8_e4m3 scales, f32 global scale, and bf16 output buffer"
      }} {{
{body}    return
  }}
}}
"#,
            kernel_name = kernel_name,
            x_ty = self.x_memref_ty(),
            packed_ty = self.packed_memref_ty(),
            scales_ty = self.scales_memref_ty(),
            global_scale_ty = self.global_scale_memref_ty(),
            out_ty = self.out_memref_ty(),
            format = M2_NVFP4_DESCRIPTOR_FORMAT,
            abi_version = M2_NVFP4_CUSTOM_CALL_ABI_VERSION,
            block_size = self.block_size,
            bm = abi.tile.bm,
            bn = abi.tile.bn,
            bk = abi.tile.bk,
            tm = div_ceil(self.shape.m, abi.tile.bm),
            tn = div_ceil(self.shape.n, abi.tile.bn),
            tk = div_ceil(self.shape.k, abi.tile.bk),
            working_set = abi.vmem_working_set_bytes,
            target = self.target,
            body = body,
        ))
    }

    fn x_memref_ty(self) -> String {
        format!(
            "memref<{}x{}x{}>",
            self.shape.m,
            self.shape.k,
            self.x_dtype.mlir_dtype()
        )
    }

    fn packed_memref_ty(self) -> String {
        format!(
            "memref<{}x{}x{}>",
            self.shape.n,
            self.shape.k / 2,
            self.packed_dtype.mlir_storage_dtype()
        )
    }

    fn scales_memref_ty(self) -> String {
        format!(
            "memref<{}x{}x{}>",
            self.shape.n,
            self.shape.k / NVFP4_GROUP,
            self.scale_dtype.mlir_storage_dtype()
        )
    }

    fn global_scale_memref_ty(self) -> String {
        format!("memref<{}>", self.global_scale_dtype.mlir_dtype())
    }

    fn out_memref_ty(self) -> String {
        format!(
            "memref<{}x{}x{}>",
            self.shape.m,
            self.shape.n,
            self.out_dtype.mlir_dtype()
        )
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
    fn kernel_descriptor_pins_dtype_and_dims() {
        let shape = M2Nvfp4MatmulShape {
            m: 8,
            n: 1536,
            k: 3072,
        };
        let desc = shape.kernel_descriptor();
        desc.validate().unwrap();
        assert_eq!(desc.target, M2_NVFP4_CUSTOM_CALL_TARGET);
        assert_eq!(desc.x_dtype, M2Nvfp4IoDType::Bf16);
        assert_eq!(desc.packed_dtype, M2Nvfp4PackedDType::U8);
        assert_eq!(desc.scale_dtype, M2Nvfp4ScaleDType::Fp8E4M3);
        assert_eq!(desc.global_scale_dtype, M2Nvfp4ScalarDType::F32);
        assert_eq!(desc.out_dtype, M2Nvfp4IoDType::Bf16);
        assert_eq!(desc.block_size, 16);
        assert_eq!(desc.x_dims(), [8, 3072]);
        assert_eq!(desc.packed_dims(), [1536, 1536]);
        assert_eq!(desc.scale_dims(), [1536, 192]);
        assert_eq!(desc.out_dims(), [8, 1536]);
    }

    #[test]
    fn kernel_descriptor_rejects_non_contract_values() {
        let shape = M2Nvfp4MatmulShape { m: 1, n: 1, k: 16 };
        let mut desc = shape.kernel_descriptor();
        desc.block_size = 8;
        assert!(desc.validate().is_err());

        let mut desc = shape.kernel_descriptor();
        desc.target = "rvllm.m2.other";
        assert!(desc.validate().is_err());

        let bad_shape = M2Nvfp4MatmulShape { m: 1, n: 1, k: 15 }.kernel_descriptor();
        assert!(bad_shape.validate().is_err());
    }

    #[test]
    fn descriptor_text_is_deterministic() {
        let desc = M2Nvfp4MatmulShape { m: 2, n: 4, k: 16 }.kernel_descriptor();
        let expected = "format=rvllm.m2.nvfp4.custom_call.v1\nabi_version=1\ntarget=rvllm.m2.nvfp4_decode_bf16_matmul\nx_dtype=bf16\npacked_dtype=u8\nscale_dtype=fp8_e4m3\nglobal_scale_dtype=f32\nout_dtype=bf16\nblock_size=16\nm=2\nn=4\nk=16\nx_dims=2x16\npacked_dims=4x8\nscale_dims=4x1\nout_dims=2x4\n";
        assert_eq!(desc.descriptor_text().unwrap(), expected);
        assert_eq!(desc.descriptor_text().unwrap(), expected);
    }

    #[test]
    fn emits_real_m2_custom_call_signature() {
        let shape = M2Nvfp4MatmulShape {
            m: 8,
            n: 1536,
            k: 3072,
        };
        let mlir = shape.mosaic_mlir("rvllm_m2_nvfp4_matmul").unwrap();
        assert!(mlir.contains("rvllm.kind = \"m2_nvfp4_custom_call\""));
        assert!(mlir.contains("memref<8x3072xbf16>"));
        assert!(mlir.contains("memref<1536x1536xi8>"));
        assert!(mlir.contains("memref<1536x192xi8>"));
        assert!(mlir.contains("rvllm.abi = \"rvllm.m2.nvfp4.custom_call.v1\""));
        assert!(mlir.contains("rvllm.block_size = 16 : i64"));
        assert!(mlir.contains("rvllm.packed_dtype = \"u8\""));
        assert!(mlir.contains("rvllm.scale_dtype = \"fp8_e4m3\""));
        assert!(mlir.contains("rvllm.x_dims = \"8x3072\""));
        assert!(mlir.contains("rvllm.out_dims = \"8x1536\""));
        assert!(mlir.contains("rvllm.tile = \"BM=8,BN=512,BK=1024\""));
        assert!(mlir.contains("rvllm.vmem_working_set_bytes = 1384448 : i64"));
        assert!(mlir.contains("rvllm.custom_call_target = \"rvllm.m2.nvfp4_decode_bf16_matmul\""));
        assert!(mlir.contains("rvllm.lowering = \"rust_xla_custom_call\""));
        assert!(mlir.contains(
            "\"rvllm.m2.nvfp4_decode_bf16_matmul\"(%x, %packed, %scales, %global_scale, %out)"
        ));
        assert!(mlir.contains("rvllm.descriptor = \"format=rvllm.m2.nvfp4.custom_call.v1;abi_version=1;target=rvllm.m2.nvfp4_decode_bf16_matmul;x_dtype=bf16;packed_dtype=u8;scale_dtype=fp8_e4m3;global_scale_dtype=f32;out_dtype=bf16;block_size=16;m=8;n=1536;k=3072;x_dims=8x3072;packed_dims=1536x1536;scale_dims=1536x192;out_dims=8x1536\""));
        assert!(mlir.contains("rvllm.packed_bytes = 2359296 : i64"));
        assert!(mlir.contains("rvllm.scale_bytes = 294912 : i64"));
        assert!(mlir.contains("rvllm.out_bytes = 24576 : i64"));
        assert!(!mlir.contains("mosaic_custom_call"));
        assert!(!mlir.to_ascii_lowercase().contains("pallas"));
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
        assert_eq!(abi.descriptor, shape.kernel_descriptor());
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
