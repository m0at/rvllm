//! W4A8 GEMM wrapper around `libw4a8_gemm.so` (CUTLASS SM90, int4×fp8).
//!
//! Mirrors the `CublasLt::fp8_gemm` API surface so the dispatcher in
//! `layer_exec` can swap between FP8 and W4A8 paths by a per-linear
//! flag, without changing call-site shape.
//!
//! Weight format on disk: INT4 two's-complement, AWQ-re-encoded,
//! memory-reordered offline into the `LayoutB_Reordered` atom layout.
//! Scales: per-group (g=128) FP8 E4M3 LUT blocks (8 packed scale factors
//! per group × N). The encoder can synthesize symmetric scales or consume
//! calibrated per-group f32 scales before packing the CUTLASS LUT ABI.

use std::path::PathBuf;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

#[cfg(feature = "cuda")]
type W4a8GemmFn = unsafe extern "C" fn(
    a_fp8: *const std::ffi::c_void,
    b_int4_reordered: *const std::ffi::c_void,
    b_scales_packed: *const std::ffi::c_void,
    c_f16: *const std::ffi::c_void,
    d_f16: *mut std::ffi::c_void,
    m: i32,
    n: i32,
    k: i32,
    group_size: i32,
    alpha: f32,
    beta: f32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

#[cfg(feature = "cuda")]
type W4a8RowscaleGemmFn = unsafe extern "C" fn(
    a_fp8: *const std::ffi::c_void,
    a_scales: *const f32,
    b_int4_reordered: *const std::ffi::c_void,
    b_scales_packed: *const std::ffi::c_void,
    d_f16: *mut std::ffi::c_void,
    m: i32,
    n: i32,
    k: i32,
    group_size: i32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

#[cfg(feature = "cuda")]
type W4a8WorkspaceFn = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

#[cfg(feature = "cuda")]
type W4a8Int4BytesFn = unsafe extern "C" fn(n: i32, k: i32) -> usize;

// Weight encoder: FP16 [N,K] -> unified-encoded INT4 [N,K/2] bytes + LUT
// packed FP8 scales [N, K/g, 8] bytes.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
type W4a8EncodeFp16Fn = unsafe extern "C" fn(
    w_fp16: *const std::ffi::c_void,
    n: i32,
    k: i32,
    group_size: i32,
    w_int4_out: *mut std::ffi::c_void,
    scales_packed_out: *mut std::ffi::c_void,
    scales_f32_workspace: *mut std::ffi::c_void,
    shuffle: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
type W4a8EncodeFp16WithScalesFn = unsafe extern "C" fn(
    w_fp16: *const std::ffi::c_void,
    calibrated_scales_f32: *const f32,
    n: i32,
    k: i32,
    group_size: i32,
    w_int4_out: *mut std::ffi::c_void,
    scales_packed_out: *mut std::ffi::c_void,
    scales_f32_workspace: *mut std::ffi::c_void,
    shuffle: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

pub const W4A8_GROUP_SIZE: i32 = 128;
const W4A8_SCALE_LUT_BYTES: usize = 8;

pub struct W4a8Lib {
    pub so_path: PathBuf,
    #[cfg(feature = "cuda")]
    _lib: libloading::Library,
    #[cfg(feature = "cuda")]
    gemm_run: W4a8GemmFn,
    #[cfg(feature = "cuda")]
    gemm_rowscale: W4a8RowscaleGemmFn,
    #[cfg(feature = "cuda")]
    gemm_ws: W4a8WorkspaceFn,
    #[cfg(feature = "cuda")]
    int4_bytes: W4a8Int4BytesFn,
    #[cfg(feature = "cuda")]
    fn_encode_fp16: W4a8EncodeFp16Fn,
    #[cfg(feature = "cuda")]
    fn_encode_fp16_with_scales: Option<W4a8EncodeFp16WithScalesFn>,
}

fn w4a8_cuda_err(op: &'static str, kernel: &'static str, stream: u64) -> RvllmError {
    RvllmError::cuda(
        op,
        CudaErrorKind::LaunchFailed,
        CudaCtx {
            stream,
            kernel,
            launch: None,
            device: -1,
        },
    )
}

fn validate_w4a8_gemm_shape(m: i32, n: i32, k: i32, group_size: i32) -> Result<()> {
    if m <= 0 || n <= 0 || k <= 0 {
        return Err(w4a8_cuda_err("w4a8 invalid M/N/K", "rvllm_w4a8", 0));
    }
    validate_w4a8_weight_shape(n, k, group_size)
}

fn validate_w4a8_weight_shape(n: i32, k: i32, group_size: i32) -> Result<()> {
    if n <= 0 || k <= 0 {
        return Err(w4a8_cuda_err("w4a8 invalid N/K", "rvllm_w4a8", 0));
    }
    if group_size != W4A8_GROUP_SIZE {
        return Err(w4a8_cuda_err("w4a8 invalid group_size", "rvllm_w4a8", 0));
    }
    if k % group_size != 0 || k % 8 != 0 {
        return Err(w4a8_cuda_err("w4a8 invalid K", "rvllm_w4a8", 0));
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn validate_ptr(ptr: u64, op: &'static str, kernel: &'static str, stream: u64) -> Result<()> {
    if ptr == 0 {
        return Err(w4a8_cuda_err(op, kernel, stream));
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn validate_workspace(workspace: u64, workspace_bytes: usize, stream: u64) -> Result<()> {
    if workspace_bytes != 0 && workspace == 0 {
        return Err(w4a8_cuda_err("w4a8 null workspace", "rvllm_w4a8", stream));
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn validate_alpha_beta(alpha: f32, beta: f32, stream: u64) -> Result<()> {
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(w4a8_cuda_err(
            "w4a8 nonfinite alpha/beta",
            "rvllm_w4a8_gemm_run",
            stream,
        ));
    }
    Ok(())
}

fn scale_group_count(n: i32, k: i32, group_size: i32) -> Result<usize> {
    validate_w4a8_weight_shape(n, k, group_size)?;
    let n = usize::try_from(n).map_err(|_| w4a8_cuda_err("w4a8 invalid N", "rvllm_w4a8", 0))?;
    let groups = usize::try_from(k / group_size)
        .map_err(|_| w4a8_cuda_err("w4a8 invalid K", "rvllm_w4a8", 0))?;
    n.checked_mul(groups)
        .ok_or_else(|| w4a8_cuda_err("w4a8 scale size overflow", "rvllm_w4a8", 0))
}

#[cfg(any(feature = "cuda", test))]
fn rc_op(prefix: &'static str, rc: i32) -> &'static str {
    match rc {
        -9..=-1 => "w4a8 invalid abi arguments",
        -10 => "w4a8 can_implement",
        -11 => "w4a8 workspace too small",
        -12 => "w4a8 initialize",
        -13 => "w4a8 run",
        -29..=-20 => "w4a8 encode",
        -39..=-30 => "w4a8 rowscale",
        _ => prefix,
    }
}

impl W4a8Lib {
    pub fn load(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            return Err(RvllmError::cuda(
                "W4a8Lib::load missing .so",
                CudaErrorKind::LaunchFailed,
                CudaCtx {
                    stream: 0,
                    kernel: "w4a8",
                    launch: None,
                    device: -1,
                },
            ));
        }
        #[cfg(feature = "cuda")]
        unsafe {
            let _lib = libloading::Library::new(&path).map_err(|_| {
                RvllmError::cuda(
                    "W4a8Lib dlopen",
                    CudaErrorKind::LaunchFailed,
                    CudaCtx {
                        stream: 0,
                        kernel: "w4a8",
                        launch: None,
                        device: -1,
                    },
                )
            })?;
            let run_sym: libloading::Symbol<W4a8GemmFn> =
                _lib.get(b"rvllm_w4a8_gemm_run\0").map_err(|_| {
                    RvllmError::cuda(
                        "dlsym rvllm_w4a8_gemm_run",
                        CudaErrorKind::LaunchFailed,
                        CudaCtx {
                            stream: 0,
                            kernel: "w4a8",
                            launch: None,
                            device: -1,
                        },
                    )
                })?;
            let ws_sym: libloading::Symbol<W4a8WorkspaceFn> =
                _lib.get(b"rvllm_w4a8_gemm_workspace_size\0").map_err(|_| {
                    RvllmError::cuda(
                        "dlsym rvllm_w4a8_gemm_workspace_size",
                        CudaErrorKind::LaunchFailed,
                        CudaCtx {
                            stream: 0,
                            kernel: "w4a8",
                            launch: None,
                            device: -1,
                        },
                    )
                })?;
            let rowscale_sym: libloading::Symbol<W4a8RowscaleGemmFn> =
                _lib.get(b"rvllm_w4a8_gemm_run_rowscale\0").map_err(|_| {
                    RvllmError::cuda(
                        "dlsym rvllm_w4a8_gemm_run_rowscale",
                        CudaErrorKind::LaunchFailed,
                        CudaCtx {
                            stream: 0,
                            kernel: "w4a8",
                            launch: None,
                            device: -1,
                        },
                    )
                })?;
            let enc_sym: libloading::Symbol<W4a8EncodeFp16Fn> =
                _lib.get(b"rvllm_w4a8_encode_weight_fp16\0").map_err(|_| {
                    RvllmError::cuda(
                        "dlsym rvllm_w4a8_encode_weight_fp16",
                        CudaErrorKind::LaunchFailed,
                        CudaCtx {
                            stream: 0,
                            kernel: "w4a8",
                            launch: None,
                            device: -1,
                        },
                    )
                })?;
            let enc_with_scales_sym: Option<libloading::Symbol<W4a8EncodeFp16WithScalesFn>> = _lib
                .get(b"rvllm_w4a8_encode_weight_fp16_with_scales\0")
                .ok();
            let bytes_sym: libloading::Symbol<W4a8Int4BytesFn> = _lib
                .get(b"rvllm_w4a8_int4_reordered_bytes\0")
                .map_err(|_| {
                    RvllmError::cuda(
                        "dlsym rvllm_w4a8_int4_reordered_bytes",
                        CudaErrorKind::LaunchFailed,
                        CudaCtx {
                            stream: 0,
                            kernel: "w4a8",
                            launch: None,
                            device: -1,
                        },
                    )
                })?;
            let gemm_run = *run_sym;
            let gemm_rowscale = *rowscale_sym;
            let gemm_ws = *ws_sym;
            let int4_bytes = *bytes_sym;
            let fn_encode_fp16 = *enc_sym;
            let fn_encode_fp16_with_scales = enc_with_scales_sym.map(|sym| *sym);
            Ok(Self {
                so_path: path,
                _lib,
                gemm_run,
                gemm_rowscale,
                gemm_ws,
                int4_bytes,
                fn_encode_fp16,
                fn_encode_fp16_with_scales,
            })
        }
        #[cfg(not(feature = "cuda"))]
        Ok(Self { so_path: path })
    }

    /// Per-shape workspace size.
    pub fn workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        if validate_w4a8_gemm_shape(m, n, k, W4A8_GROUP_SIZE).is_err() {
            return 0;
        }
        #[cfg(feature = "cuda")]
        unsafe {
            (self.gemm_ws)(m, n, k)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (m, n, k);
            0
        }
    }

    /// Bytes required for a reordered INT4 weight tensor with shape [N, K].
    pub fn int4_reordered_bytes(&self, n: i32, k: i32) -> usize {
        if validate_w4a8_weight_shape(n, k, W4A8_GROUP_SIZE).is_err() {
            return 0;
        }
        #[cfg(feature = "cuda")]
        unsafe {
            (self.int4_bytes)(n, k)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (n, k);
            0
        }
    }

    /// Bytes required for `[N, K/group, 8]` packed FP8 scale LUT blocks.
    pub fn scale_packed_bytes(n: i32, k: i32, group_size: i32) -> Result<usize> {
        scale_group_count(n, k, group_size)?
            .checked_mul(W4A8_SCALE_LUT_BYTES)
            .ok_or_else(|| w4a8_cuda_err("w4a8 scale size overflow", "rvllm_w4a8", 0))
    }

    /// Bytes required for the temporary `[N, K/group]` f32 scale buffer.
    pub fn scale_workspace_bytes(n: i32, k: i32, group_size: i32) -> Result<usize> {
        scale_group_count(n, k, group_size)?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| w4a8_cuda_err("w4a8 scale workspace overflow", "rvllm_w4a8", 0))
    }

    /// D = alpha * A_fp8 * B_w4_unquant + beta * C.
    ///
    /// - `a_fp8`: `[m, k]` RowMajor E4M3 activations, device pointer.
    /// - `b_int4_reordered`: INT4 weights for logical `[n, k]`, already
    ///   offline-reordered to the LayoutB_Reordered atom layout expected by
    ///   the kernel. Allocate with `int4_reordered_bytes(n, k)`.
    /// - `b_scales_packed`: `[n, k/group_size]` packed FP8 LUT blocks
    ///   (each block is 8 packed E4M3 scales). Device pointer.
    /// - `c_f16`: optional `[m, n]` RowMajor residual. Pass 0 if `beta==0`.
    /// - `d_f16`: `[m, n]` RowMajor output. Device pointer.
    /// - `workspace`/`workspace_bytes`: scratch; size via `workspace_size`.
    ///
    /// # Safety
    /// All device pointers must be valid for the duration of the call on
    /// the given stream.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn w4a8_gemm(
        &self,
        a_fp8: u64,
        b_int4_reordered: u64,
        b_scales_packed: u64,
        c_f16: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        group_size: i32,
        alpha: f32,
        beta: f32,
        workspace: u64,
        workspace_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        validate_w4a8_gemm_shape(m, n, k, group_size)?;
        validate_alpha_beta(alpha, beta, stream)?;
        validate_ptr(a_fp8, "w4a8 null A", "rvllm_w4a8_gemm_run", stream)?;
        validate_ptr(
            b_int4_reordered,
            "w4a8 null B",
            "rvllm_w4a8_gemm_run",
            stream,
        )?;
        validate_ptr(
            b_scales_packed,
            "w4a8 null B scales",
            "rvllm_w4a8_gemm_run",
            stream,
        )?;
        validate_ptr(d_f16, "w4a8 null D", "rvllm_w4a8_gemm_run", stream)?;
        if beta != 0.0 {
            validate_ptr(c_f16, "w4a8 null C", "rvllm_w4a8_gemm_run", stream)?;
        }
        validate_workspace(workspace, workspace_bytes, stream)?;
        let needed = self.workspace_size(m, n, k);
        if needed > workspace_bytes {
            return Err(w4a8_cuda_err(
                "w4a8 workspace too small",
                "rvllm_w4a8_gemm_run",
                stream,
            ));
        }
        let rc = (self.gemm_run)(
            a_fp8 as *const _,
            b_int4_reordered as *const _,
            b_scales_packed as *const _,
            c_f16 as *const _,
            d_f16 as *mut _,
            m,
            n,
            k,
            group_size,
            alpha,
            beta,
            workspace as *mut _,
            workspace_bytes,
            stream as *mut _,
        );
        if rc != 0 {
            return Err(RvllmError::cuda(
                rc_op("w4a8_gemm_run", rc),
                CudaErrorKind::LaunchFailed,
                CudaCtx {
                    stream,
                    kernel: "rvllm_w4a8_gemm_run",
                    launch: None,
                    device: -1,
                },
            ));
        }
        Ok(())
    }

    /// D_f16 = a_scales[m] * (A_fp8 * B_w4). The row scale pointer is
    /// the per-token activation scale vector produced by rvLLM's FP8
    /// quantizers.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn w4a8_gemm_rowscale(
        &self,
        a_fp8: u64,
        a_scales: u64,
        b_int4_reordered: u64,
        b_scales_packed: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        group_size: i32,
        workspace: u64,
        workspace_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        validate_w4a8_gemm_shape(m, n, k, group_size)?;
        validate_ptr(a_fp8, "w4a8 null A", "rvllm_w4a8_gemm_run_rowscale", stream)?;
        validate_ptr(
            a_scales,
            "w4a8 null A row scales",
            "rvllm_w4a8_gemm_run_rowscale",
            stream,
        )?;
        validate_ptr(
            b_int4_reordered,
            "w4a8 null B",
            "rvllm_w4a8_gemm_run_rowscale",
            stream,
        )?;
        validate_ptr(
            b_scales_packed,
            "w4a8 null B scales",
            "rvllm_w4a8_gemm_run_rowscale",
            stream,
        )?;
        validate_ptr(d_f16, "w4a8 null D", "rvllm_w4a8_gemm_run_rowscale", stream)?;
        validate_workspace(workspace, workspace_bytes, stream)?;
        let needed = self.workspace_size(m, n, k);
        if needed > workspace_bytes {
            return Err(w4a8_cuda_err(
                "w4a8 workspace too small",
                "rvllm_w4a8_gemm_run_rowscale",
                stream,
            ));
        }
        let rc = (self.gemm_rowscale)(
            a_fp8 as *const _,
            a_scales as *const f32,
            b_int4_reordered as *const _,
            b_scales_packed as *const _,
            d_f16 as *mut _,
            m,
            n,
            k,
            group_size,
            workspace as *mut _,
            workspace_bytes,
            stream as *mut _,
        );
        if rc != 0 {
            return Err(RvllmError::cuda(
                rc_op("w4a8_gemm_run_rowscale", rc),
                CudaErrorKind::LaunchFailed,
                CudaCtx {
                    stream,
                    kernel: "rvllm_w4a8_gemm_run_rowscale",
                    launch: None,
                    device: -1,
                },
            ));
        }
        Ok(())
    }

    /// Encode FP16 weights [N, K] (row-major, device ptr) to reordered
    /// unified-encoded INT4 plus LUT packed FP8 scales [N, K/g, 8] bytes.
    /// `w_int4_out` must be allocated with `int4_reordered_bytes(n, k)`.
    /// Needs a scratch buffer `scales_f32_ws` of at least `N * K/g * 4`
    /// bytes.
    ///
    /// # Safety
    /// All device pointers must be valid for the duration of the call.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn encode_fp16(
        &self,
        w_fp16: u64,
        n: i32,
        k: i32,
        group_size: i32,
        w_int4_out: u64,
        scales_packed_out: u64,
        scales_f32_ws: u64,
        shuffle: bool,
        stream: u64,
    ) -> Result<()> {
        validate_w4a8_weight_shape(n, k, group_size)?;
        validate_ptr(
            w_fp16,
            "w4a8 null W fp16",
            "rvllm_w4a8_encode_weight_fp16",
            stream,
        )?;
        validate_ptr(
            w_int4_out,
            "w4a8 null W int4 output",
            "rvllm_w4a8_encode_weight_fp16",
            stream,
        )?;
        validate_ptr(
            scales_packed_out,
            "w4a8 null scales output",
            "rvllm_w4a8_encode_weight_fp16",
            stream,
        )?;
        validate_ptr(
            scales_f32_ws,
            "w4a8 null scale workspace",
            "rvllm_w4a8_encode_weight_fp16",
            stream,
        )?;
        let rc = (self.fn_encode_fp16)(
            w_fp16 as *const _,
            n,
            k,
            group_size,
            w_int4_out as *mut _,
            scales_packed_out as *mut _,
            scales_f32_ws as *mut _,
            if shuffle { 1 } else { 0 },
            stream as *mut _,
        );
        if rc != 0 {
            return Err(RvllmError::cuda(
                rc_op("w4a8_encode_fp16", rc),
                CudaErrorKind::LaunchFailed,
                CudaCtx {
                    stream,
                    kernel: "rvllm_w4a8_encode_weight_fp16",
                    launch: None,
                    device: -1,
                },
            ));
        }
        Ok(())
    }

    /// Encode FP16 weights using calibrated per-group f32 scales
    /// `[N, K/group]`, then pack those scales to the CUTLASS FP8 LUT ABI.
    /// The output layout and pointer requirements match `encode_fp16`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn encode_fp16_with_scales(
        &self,
        w_fp16: u64,
        calibrated_scales_f32: u64,
        n: i32,
        k: i32,
        group_size: i32,
        w_int4_out: u64,
        scales_packed_out: u64,
        scales_f32_ws: u64,
        shuffle: bool,
        stream: u64,
    ) -> Result<()> {
        let Some(encode_with_scales) = self.fn_encode_fp16_with_scales else {
            return Err(w4a8_cuda_err(
                "dlsym rvllm_w4a8_encode_weight_fp16_with_scales",
                "rvllm_w4a8_encode_weight_fp16_with_scales",
                stream,
            ));
        };
        validate_w4a8_weight_shape(n, k, group_size)?;
        validate_ptr(
            w_fp16,
            "w4a8 null W fp16",
            "rvllm_w4a8_encode_weight_fp16_with_scales",
            stream,
        )?;
        validate_ptr(
            calibrated_scales_f32,
            "w4a8 null calibrated scales",
            "rvllm_w4a8_encode_weight_fp16_with_scales",
            stream,
        )?;
        validate_ptr(
            w_int4_out,
            "w4a8 null W int4 output",
            "rvllm_w4a8_encode_weight_fp16_with_scales",
            stream,
        )?;
        validate_ptr(
            scales_packed_out,
            "w4a8 null scales output",
            "rvllm_w4a8_encode_weight_fp16_with_scales",
            stream,
        )?;
        validate_ptr(
            scales_f32_ws,
            "w4a8 null scale workspace",
            "rvllm_w4a8_encode_weight_fp16_with_scales",
            stream,
        )?;
        let rc = encode_with_scales(
            w_fp16 as *const _,
            calibrated_scales_f32 as *const f32,
            n,
            k,
            group_size,
            w_int4_out as *mut _,
            scales_packed_out as *mut _,
            scales_f32_ws as *mut _,
            if shuffle { 1 } else { 0 },
            stream as *mut _,
        );
        if rc != 0 {
            return Err(RvllmError::cuda(
                rc_op("w4a8_encode_fp16_with_scales", rc),
                CudaErrorKind::LaunchFailed,
                CudaCtx {
                    stream,
                    kernel: "rvllm_w4a8_encode_weight_fp16_with_scales",
                    launch: None,
                    device: -1,
                },
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_w4a8_shapes() {
        assert!(validate_w4a8_gemm_shape(128, 256, 128, W4A8_GROUP_SIZE).is_ok());
        assert!(validate_w4a8_gemm_shape(0, 256, 128, W4A8_GROUP_SIZE).is_err());
        assert!(validate_w4a8_gemm_shape(128, 256, 64, W4A8_GROUP_SIZE).is_err());
        assert!(validate_w4a8_gemm_shape(128, 256, 128, 64).is_err());
    }

    #[test]
    fn computes_scale_buffer_sizes() {
        assert_eq!(
            W4a8Lib::scale_packed_bytes(256, 128, W4A8_GROUP_SIZE).ok(),
            Some(256 * W4A8_SCALE_LUT_BYTES)
        );
        assert_eq!(
            W4a8Lib::scale_workspace_bytes(256, 256, W4A8_GROUP_SIZE).ok(),
            Some(256 * 2 * std::mem::size_of::<f32>())
        );
        assert!(W4a8Lib::scale_packed_bytes(256, 64, W4A8_GROUP_SIZE).is_err());
    }

    #[test]
    fn rejects_null_and_nonfinite_abi_inputs() {
        assert!(validate_ptr(0, "x", "kernel", 7).is_err());
        assert!(validate_ptr(1, "x", "kernel", 7).is_ok());
        assert!(validate_workspace(0, 16, 7).is_err());
        assert!(validate_workspace(0, 0, 7).is_ok());
        assert!(validate_alpha_beta(1.0, 0.0, 7).is_ok());
        assert!(validate_alpha_beta(f32::NAN, 0.0, 7).is_err());
        assert!(validate_alpha_beta(1.0, f32::INFINITY, 7).is_err());
    }

    #[test]
    fn maps_native_return_codes_to_stable_ops() {
        assert_eq!(rc_op("fallback", -11), "w4a8 workspace too small");
        assert_eq!(rc_op("fallback", -27), "w4a8 encode");
        assert_eq!(rc_op("fallback", -31), "w4a8 rowscale");
        assert_eq!(rc_op("fallback", 99), "fallback");
    }
}

#[cfg(feature = "cuda")]
unsafe impl Send for W4a8Lib {}
#[cfg(feature = "cuda")]
unsafe impl Sync for W4a8Lib {}
