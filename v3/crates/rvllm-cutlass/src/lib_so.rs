//! `libcutlass_kernels.so` dlopen + variant fn-pointer table.
//!
//! Opens the CUTLASS shared library once at engine init, resolves every
//! variant that appears in the autotune `Policy`, and caches the fn
//! pointers for zero-cost dispatch. A variant referenced by the policy
//! that's missing from the .so returns a typed error at load time — the
//! engine refuses to start rather than silently downgrade.

#[cfg(feature = "cuda")]
use std::ffi::c_void;
use std::path::PathBuf;

use rvllm_core::{CutlassCtx, CutlassError, Result, RvllmError};

use crate::variants::VariantId;

// Non-residual FP8 GEMM variant fn.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Fp8GemmFn = unsafe extern "C" fn(
    output: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    a_scales: *const c_void,
    b_scale: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

// Residual-fused FP8 GEMM variant fn (epilogue adds a host-provided C).
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Fp8GemmResidualFn = unsafe extern "C" fn(
    output: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    a_scales: *const c_void,
    b_scale: *const c_void,
    residual: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

#[cfg(feature = "cuda")]
pub type WorkspaceSizeFn = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Fp8GemmChannelscaleFn = unsafe extern "C" fn(
    output: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    row_scale: *const c_void,
    col_scale: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

#[cfg(feature = "cuda")]
pub type ChannelscaleWorkspaceFn = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

/// Signature of `cutlass_fp8_gemm_blockscale_sm120` in
/// `libcutlass_sm120.so`. The scale tensors carry 128×128 blockwise
/// semantics (Gemma 4 fp8-block format) — `a_scale` is SFA sized
/// `[ceil(M/128), K/128]`, `b_scale` is SFB sized `[N/128, K/128]`.
/// That's DIFFERENT from the per-vector SM90 channelscale ABI above;
/// we keep them as distinct fn-types so a miswiring fails at compile
/// time instead of silently passing the wrong pointer shape.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Fp8GemmBlockscaleSm120Fn = unsafe extern "C" fn(
    output: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    a_scale_sfa: *const c_void,
    b_scale_sfb: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

#[cfg(feature = "cuda")]
pub type BlockscaleSm120WorkspaceFn =
    unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

/// Scratch-sizing entry points for SFA / SFB staging tensors. Same
/// signature shape as `BlockscaleSm120WorkspaceFn` but taking only two
/// problem-shape ints (SFA depends on M,K; SFB depends on N,K).
#[cfg(feature = "cuda")]
pub type BlockscaleSm120SfBytesFn = unsafe extern "C" fn(a: i32, b: i32) -> usize;

/// Prep-kernel signature for the SFA broadcast + SFB transpose passes
/// that convert Gemma 4 fp8-block's per-token a_scale / row-major
/// b_chscale into the CUTLASS-layout SFA/SFB scratch tensors.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type BlockscaleSm120PrepFn = unsafe extern "C" fn(
    src: *const c_void,
    dst: *mut c_void,
    dim_outer: i32,
    dim_inner: i32,
    stream: *mut c_void,
) -> i32;

/// Resolved CUTLASS .so + variant fn-pointer table.
#[derive(Debug)]
pub struct CutlassLib {
    pub so_path: PathBuf,
    #[cfg(feature = "cuda")]
    _lib: libloading::Library,
    /// Keyed by VariantId; `None` if the variant is in the catalog but
    /// absent from the .so (the caller checks on load — missing for a
    /// policy-referenced variant is an error).
    #[cfg(feature = "cuda")]
    pub fp8_gemm: std::collections::BTreeMap<VariantId, Fp8GemmFn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_ws: std::collections::BTreeMap<VariantId, WorkspaceSizeFn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_residual: std::collections::BTreeMap<VariantId, Fp8GemmResidualFn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_residual_ws: std::collections::BTreeMap<VariantId, WorkspaceSizeFn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_channelscale: Option<Fp8GemmChannelscaleFn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_channelscale_ws: Option<ChannelscaleWorkspaceFn>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CutlassLib {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CutlassLib {}

impl CutlassLib {
    #[cfg(feature = "cuda")]
    pub fn load(path: PathBuf, policy_variants: &[VariantId]) -> Result<Self> {
        let lib = unsafe { libloading::Library::new(&path) }.map_err(|_| cutlass_miss(&path))?;
        let mut fp8_gemm = std::collections::BTreeMap::new();
        let mut fp8_gemm_ws = std::collections::BTreeMap::new();
        let mut fp8_gemm_residual = std::collections::BTreeMap::new();
        let mut fp8_gemm_residual_ws = std::collections::BTreeMap::new();

        // VariantId -> v2 symbol name (drop-in compat with v2's .so).
        //   0..=99  -> cutlass_fp8_gemm[*]  (+ _workspace_size)
        //  100..    -> cutlass_fp8_gemm_residual[*]
        for &vid in policy_variants {
            let (fn_name_str, ws_name_str) = v2_symbol_names(vid);
            let is_residual = vid.0 >= 100;
            if is_residual {
                let fn_c = format!("{fn_name_str}\0");
                let ws_c = format!("{ws_name_str}\0");
                unsafe {
                    let f: libloading::Symbol<Fp8GemmResidualFn> = lib
                        .get(fn_c.as_bytes())
                        .map_err(|_| variant_missing(&path, vid, "fp8_gemm_residual"))?;
                    let w: libloading::Symbol<WorkspaceSizeFn> = lib
                        .get(ws_c.as_bytes())
                        .map_err(|_| variant_missing(&path, vid, "fp8_gemm_residual_ws"))?;
                    fp8_gemm_residual.insert(vid, *f);
                    fp8_gemm_residual_ws.insert(vid, *w);
                }
            } else {
                let fn_c = format!("{fn_name_str}\0");
                let ws_c = format!("{ws_name_str}\0");
                unsafe {
                    let f: libloading::Symbol<Fp8GemmFn> = lib
                        .get(fn_c.as_bytes())
                        .map_err(|_| variant_missing(&path, vid, "fp8_gemm"))?;
                    let w: libloading::Symbol<WorkspaceSizeFn> = lib
                        .get(ws_c.as_bytes())
                        .map_err(|_| variant_missing(&path, vid, "fp8_gemm_ws"))?;
                    fp8_gemm.insert(vid, *f);
                    fp8_gemm_ws.insert(vid, *w);
                }
            }
        }

        let fp8_gemm_channelscale: Option<Fp8GemmChannelscaleFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_channelscale\0").ok().map(|s| *s)
        };
        let fp8_gemm_channelscale_ws: Option<ChannelscaleWorkspaceFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_channelscale_workspace\0").ok().map(|s| *s)
        };

        Ok(Self {
            so_path: path,
            _lib: lib,
            fp8_gemm,
            fp8_gemm_ws,
            fp8_gemm_residual,
            fp8_gemm_residual_ws,
            fp8_gemm_channelscale,
            fp8_gemm_channelscale_ws,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(path: PathBuf, _policy_variants: &[VariantId]) -> Result<Self> {
        if !path.exists() {
            return Err(cutlass_miss(&path));
        }
        Ok(Self { so_path: path })
    }

    /// Dispatch a non-residual FP8 GEMM. `workspace` may be null if the
    /// plan's `workspace_bytes == 0`; otherwise it must point at >=
    /// `plan.workspace_bytes` of device memory.
    ///
    /// # Safety
    /// All pointers must be valid device memory for the kernel's duration.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm(
        &self,
        plan: &crate::plan::Fp8GemmPlan,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        plan.check_workspace(workspace_size)?;
        let f = self.fp8_gemm.get(&plan.variant).ok_or_else(|| {
            variant_missing(&self.so_path, plan.variant, "fp8_gemm (runtime lookup)")
        })?;
        let rc = f(
            output as *mut c_void,
            a as *const c_void,
            b as *const c_void,
            a_scales as *const c_void,
            b_scale as *const c_void,
            plan.m as i32,
            plan.n as i32,
            plan.k as i32,
            workspace as *mut c_void,
            workspace_size,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: plan.variant.0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "fp8_gemm",
                    stream,
                },
            ));
        }
        Ok(())
    }

    /// Same, residual-fused variant. `residual` is the C-tensor the
    /// epilogue adds into `output`.
    ///
    /// # Safety
    /// All pointers valid for the call.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_residual(
        &self,
        plan: &crate::plan::Fp8GemmPlan,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        residual: u64,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        plan.check_workspace(workspace_size)?;
        let f = self.fp8_gemm_residual.get(&plan.variant).ok_or_else(|| {
            variant_missing(
                &self.so_path,
                plan.variant,
                "fp8_gemm_residual (runtime lookup)",
            )
        })?;
        let rc = f(
            output as *mut c_void,
            a as *const c_void,
            b as *const c_void,
            a_scales as *const c_void,
            b_scale as *const c_void,
            residual as *const c_void,
            plan.m as i32,
            plan.n as i32,
            plan.k as i32,
            workspace as *mut c_void,
            workspace_size,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: plan.variant.0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "fp8_gemm_residual",
                    stream,
                },
            ));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_channelscale(
        &self,
        output: u64,
        a: u64,
        b: u64,
        row_scale: u64,
        col_scale: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        let f = self.fp8_gemm_channelscale.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx { kernel: "fp8_gemm_channelscale (missing from .so)", stream },
            )
        })?;
        let rc = f(
            output as *mut c_void,
            a as *const c_void,
            b as *const c_void,
            row_scale as *const c_void,
            col_scale as *const c_void,
            m, n, k,
            workspace as *mut c_void,
            workspace_size,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx { kernel: "fp8_gemm_channelscale", stream },
            ));
        }
        Ok(())
    }
}

// ============================================================================
// CutlassBackend — unifies `So(CutlassLib)` (SM80/89/90) and `Absent` (sm_121)
// ============================================================================

/// Which CUTLASS backend the runtime is using on the live device.
///
///   * SM80 / SM89 / SM90 → `So(CutlassLib)` — dlopen
///     `libcutlass_kernels.so`, fn-ptr table keyed by `VariantId`.
///   * SM121 (Blackwell consumer) → `SoSm120(CutlassSm120Lib)` when a
///     `libcutlass_sm120.so` is found (built via
///     `kernels/build_cutlass_sm120_so.sh`), exposing the native
///     `cutlass_fp8_gemm_blockscale_sm120` kernel with correct
///     128×128 block-scale semantics for Gemma 4 fp8-block. Falls
///     back to `Absent` if the .so is missing — preserving the
///     previous "skip CUTLASS, use the PTX fallback" behaviour on
///     hosts that haven't built the library yet.
///
/// `#[non_exhaustive]` leaves room for more backends to slide in.
#[derive(Debug)]
#[non_exhaustive]
pub enum CutlassBackend {
    /// SM80/89/90 .so.
    So(CutlassLib),
    /// Blackwell-Geforce (SM120/SM121) .so.
    SoSm120(CutlassSm120Lib),
    /// No CUTLASS coverage — callers must route through the PTX /
    /// cuBLASLt fallback.
    Absent,
}

/// Find the sm_120 `.so` via env override or the default location
/// in the matching arch's kernels dir. Returns `None` if not found.
fn resolve_sm120_so_path(
    sm90_hint: &std::path::Path,
    arch_name: &str,
) -> Option<PathBuf> {
    if let Some(env) = std::env::var_os("RVLLM_CUTLASS_SM120_SO") {
        let p = PathBuf::from(env);
        if p.is_file() {
            return Some(p);
        }
    }
    // Codex27-1: only the matching arch dir. Codex25-1's
    // sm_121→sm_120→sm_122 fallback chain risked silently loading a
    // .so that was built for a different Blackwell sub-arch (e.g.
    // sm_120a on a sm_121a host); the cooperative kernel hits
    // `CUTE_INVALID_CONTROL_PATH` because the family-conditional
    // doesn't match. Operators with non-default layouts use the
    // env override.
    if let Some(grandparent) = sm90_hint.parent().and_then(|p| p.parent()) {
        let candidate = grandparent.join(arch_name).join("libcutlass_sm120.so");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

// ─── Mistral 3.5 NVFP4 ABI ──────────────────────────────────────────
//
// NVFP4 weight tensors carry a *per-row × per-16-K* scale grid
// (`[N, K/16]` E4M3) plus a single F32 global scale, NOT the
// `[N/128, K/128]` FP8 blockscale layout the Gemma 4 / Qwen 3.6
// path uses. Mixing the two shapes silently passes the wrong
// pointer to the kernel, so we keep these as DISTINCT fn-pointer
// types — the Rust type system refuses to call a blockscale wrapper
// with NVFP4 tensors and vice versa.
//
// Activations enter dynamically quantized to FP8 E4M3 with a
// per-token scale (matches the existing FP8 path). The kernel does
// FP8 × NVFP4 → BF16/F16 on the Blackwell tensor cores. Two entry
// points are distinguished:
//
//   * `cutlass_nvfp4_gemm_sm120`            — batched prefill (any M)
//   * `cutlass_nvfp4_gemm_sm120_decode_m1`  — optimised m=1 decode
//
// The companion preparation + sizing helpers are:
//
//   * `cutlass_nvfp4_gemm_sm120_workspace`     — CUTLASS workspace bytes
//   * `cutlass_nvfp4_gemm_sm120_sfa_bytes`     — activation-scale staging size
//   * `cutlass_nvfp4_gemm_sm120_prep_sfa`      — FP8-quant activations + stage SFA
//
// Until the `.cu` source lands and the Sm120 `.so` is rebuilt with
// these symbols (Step 4-CUDA), `CutlassSm120Lib::load` returns
// `None` for every NVFP4 fn pointer. `require_nvfp4()` then errors
// at startup so Mistral 3.5 refuses to bring up against a stale
// `.so`. The Gemma 4 / Qwen 3.6 paths keep working unchanged.

/// Main NVFP4 GEMM kernel. `b_packed` is the NVFP4 weight tensor
/// (`U8 [N, K/2]`, two E2M1 values per byte, low-nibble first).
/// `b_scale_e4m3` is the per-row × per-16-K scale grid
/// (`Fp8E4M3 [N, K/16]`). `b_global_scale_f32` is the F32 scalar.
/// Activation FP8 + per-token scale come in via `a_fp8` / `sfa`.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Nvfp4GemmSm120Fn = unsafe extern "C" fn(
    output: *mut c_void,
    a_fp8: *const c_void,
    b_packed: *const c_void,
    sfa: *const c_void,            // staged per-token activation scales
    b_scale_e4m3: *const c_void,   // [N, K/16] E4M3
    b_global_scale_f32: *const c_void, // [1] F32
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

#[cfg(feature = "cuda")]
pub type Nvfp4GemmSm120WorkspaceFn = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

/// SFA scratch size for the activation-scale staging (m × K/16
/// E4M3). Mirrors the FP8 blockscale `sf_bytes` helpers but with
/// the K/16 grid Mistral's NVFP4 layout uses.
#[cfg(feature = "cuda")]
pub type Nvfp4Sm120SfaBytesFn = unsafe extern "C" fn(m: i32, k: i32) -> usize;

/// Single-call activation prep: BF16/F16 -> NVFP4 packed + CUTLASS-
/// interleaved SFA. Internally chains [`Nvfp4Sm120PrepActFn`] and
/// [`Nvfp4Sm120SfaTransformFn`]; the caller still must provide a
/// scratch buffer for the natural-layout intermediate. The `.so`
/// today returns -100 for this entry pending the chained
/// implementation; callers should use the explicit prep_act +
/// sfa_natural_to_interleaved pair below until then.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Nvfp4Sm120PrepSfaFn = unsafe extern "C" fn(
    a_input_fp16_or_bf16: *const c_void,
    a_fp8_out: *mut c_void,
    sfa_out: *mut c_void,
    m: i32,
    k: i32,
    a_input_dtype: i32, // 0 = F16, 1 = BF16
    stream: *mut c_void,
) -> i32;

/// Activation prep step 1 — BF16/F16 -> NVFP4-packed bytes
/// `[m, k/2]` plus per-(row, K-block-of-16) E4M3 scales in the
/// natural row-major `[m, k/16]` layout.
///
/// Wired by `kernels/cutlass_nvfp4_prep_act_sm120.cu`. Output
/// `sfa_natural` must be `nvfp4_natural_sfa_bytes(m, k)` wide.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Nvfp4Sm120PrepActFn = unsafe extern "C" fn(
    a_input_fp16_or_bf16: *const c_void,
    a_packed_out: *mut c_void,
    sfa_natural_out: *mut c_void,
    m: i32,
    k: i32,
    a_input_dtype: i32, // 0 = F16, 1 = BF16
    stream: *mut c_void,
) -> i32;

/// Activation prep step 2 — natural `[m, k/16]` E4M3 -> CUTLASS
/// `Sm1xxBlkScaledConfig` interleaved SFA. Pure index permutation.
#[cfg(feature = "cuda")]
pub type Nvfp4Sm120SfaTransformFn = unsafe extern "C" fn(
    src_natural: *const c_void,
    dst_cutlass: *mut c_void,
    m: i32,
    k: i32,
    stream: *mut c_void,
) -> i32;

/// Natural-layout SFA scratch size in bytes (one E4M3 byte per
/// (row, K-block) pair = `m * k / 16`). Distinct from
/// [`Nvfp4Sm120SfaBytesFn`], which queries the CUTLASS-interleaved
/// destination size and includes layout-swizzle padding.
#[cfg(feature = "cuda")]
pub type Nvfp4Sm120NaturalSfaBytesFn = unsafe extern "C" fn(m: i32, k: i32) -> usize;

/// Resolved `libcutlass_sm120.so` + fn pointer. Intentionally simpler
/// than `CutlassLib`: only one entry point today
/// (`cutlass_fp8_gemm_blockscale_sm120` + its workspace helper),
/// no policy-driven variant table, no residual fusions.
#[derive(Debug)]
pub struct CutlassSm120Lib {
    pub so_path: PathBuf,
    #[cfg(feature = "cuda")]
    _lib: libloading::Library,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_blockscale: Option<Fp8GemmBlockscaleSm120Fn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_blockscale_ws: Option<BlockscaleSm120WorkspaceFn>,
    #[cfg(feature = "cuda")]
    pub sfa_bytes: Option<BlockscaleSm120SfBytesFn>,
    #[cfg(feature = "cuda")]
    pub sfb_bytes: Option<BlockscaleSm120SfBytesFn>,
    #[cfg(feature = "cuda")]
    pub prep_sfa: Option<BlockscaleSm120PrepFn>,
    #[cfg(feature = "cuda")]
    pub prep_sfb: Option<BlockscaleSm120PrepFn>,
    /// Mistral 3.5 NVFP4 entry point — batched prefill (any M).
    /// `None` when the `.so` was built before the NVFP4 source landed.
    #[cfg(feature = "cuda")]
    pub nvfp4_gemm: Option<Nvfp4GemmSm120Fn>,
    /// Mistral 3.5 NVFP4 — optimised m=1 decode kernel.
    #[cfg(feature = "cuda")]
    pub nvfp4_gemm_decode_m1: Option<Nvfp4GemmSm120Fn>,
    #[cfg(feature = "cuda")]
    pub nvfp4_workspace: Option<Nvfp4GemmSm120WorkspaceFn>,
    #[cfg(feature = "cuda")]
    pub nvfp4_sfa_bytes: Option<Nvfp4Sm120SfaBytesFn>,
    #[cfg(feature = "cuda")]
    pub nvfp4_prep_sfa: Option<Nvfp4Sm120PrepSfaFn>,
    /// Step-1 activation prep — BF16/F16 -> NVFP4-packed +
    /// natural-layout SFA. Required member of the chain.
    #[cfg(feature = "cuda")]
    pub nvfp4_prep_act: Option<Nvfp4Sm120PrepActFn>,
    /// Step-2 SFA layout transform — natural -> CUTLASS interleaved.
    /// Required member of the chain.
    #[cfg(feature = "cuda")]
    pub nvfp4_sfa_transform: Option<Nvfp4Sm120SfaTransformFn>,
    /// Natural-layout SFA size query.
    #[cfg(feature = "cuda")]
    pub nvfp4_natural_sfa_bytes: Option<Nvfp4Sm120NaturalSfaBytesFn>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CutlassSm120Lib {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CutlassSm120Lib {}

impl CutlassSm120Lib {
    /// Load `libcutlass_sm120.so` from `path`. Returns a loaded lib
    /// even if the symbols are missing — caller decides whether that
    /// is a hard error or graceful fall-through.
    #[cfg(feature = "cuda")]
    pub fn load(path: PathBuf) -> Result<Self> {
        let lib =
            unsafe { libloading::Library::new(&path) }.map_err(|_| cutlass_miss(&path))?;
        let fp8_gemm_blockscale: Option<Fp8GemmBlockscaleSm120Fn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120\0")
                .ok()
                .map(|s| *s)
        };
        let fp8_gemm_blockscale_ws: Option<BlockscaleSm120WorkspaceFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120_workspace\0")
                .ok()
                .map(|s| *s)
        };
        let sfa_bytes: Option<BlockscaleSm120SfBytesFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120_sfa_bytes\0")
                .ok()
                .map(|s| *s)
        };
        let sfb_bytes: Option<BlockscaleSm120SfBytesFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120_sfb_bytes\0")
                .ok()
                .map(|s| *s)
        };
        let prep_sfa: Option<BlockscaleSm120PrepFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120_prep_sfa\0")
                .ok()
                .map(|s| *s)
        };
        let prep_sfb: Option<BlockscaleSm120PrepFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120_prep_sfb\0")
                .ok()
                .map(|s| *s)
        };
        // Mistral 3.5 NVFP4 entry points (optional — `None` on a
        // pre-NVFP4 `.so`; required at startup for Mistral via
        // `require_nvfp4`). The five symbols form one set; either
        // all resolve or the path is treated as unavailable.
        let nvfp4_gemm: Option<Nvfp4GemmSm120Fn> = unsafe {
            lib.get(b"cutlass_nvfp4_gemm_sm120\0").ok().map(|s| *s)
        };
        let nvfp4_gemm_decode_m1: Option<Nvfp4GemmSm120Fn> = unsafe {
            lib.get(b"cutlass_nvfp4_gemm_sm120_decode_m1\0")
                .ok()
                .map(|s| *s)
        };
        let nvfp4_workspace: Option<Nvfp4GemmSm120WorkspaceFn> = unsafe {
            lib.get(b"cutlass_nvfp4_gemm_sm120_workspace\0")
                .ok()
                .map(|s| *s)
        };
        let nvfp4_sfa_bytes: Option<Nvfp4Sm120SfaBytesFn> = unsafe {
            lib.get(b"cutlass_nvfp4_gemm_sm120_sfa_bytes\0")
                .ok()
                .map(|s| *s)
        };
        let nvfp4_prep_sfa: Option<Nvfp4Sm120PrepSfaFn> = unsafe {
            lib.get(b"cutlass_nvfp4_gemm_sm120_prep_sfa\0")
                .ok()
                .map(|s| *s)
        };
        let nvfp4_prep_act: Option<Nvfp4Sm120PrepActFn> = unsafe {
            lib.get(b"cutlass_nvfp4_gemm_sm120_prep_act\0")
                .ok()
                .map(|s| *s)
        };
        let nvfp4_sfa_transform: Option<Nvfp4Sm120SfaTransformFn> = unsafe {
            lib.get(b"cutlass_nvfp4_gemm_sm120_sfa_natural_to_interleaved\0")
                .ok()
                .map(|s| *s)
        };
        let nvfp4_natural_sfa_bytes: Option<Nvfp4Sm120NaturalSfaBytesFn> = unsafe {
            lib.get(b"cutlass_nvfp4_gemm_sm120_prep_act_sfa_bytes\0")
                .ok()
                .map(|s| *s)
        };
        Ok(Self {
            so_path: path,
            _lib: lib,
            fp8_gemm_blockscale,
            fp8_gemm_blockscale_ws,
            sfa_bytes,
            sfb_bytes,
            prep_sfa,
            prep_sfb,
            nvfp4_gemm,
            nvfp4_gemm_decode_m1,
            nvfp4_workspace,
            nvfp4_sfa_bytes,
            nvfp4_prep_sfa,
            nvfp4_prep_act,
            nvfp4_sfa_transform,
            nvfp4_natural_sfa_bytes,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            return Err(cutlass_miss(&path));
        }
        Ok(Self { so_path: path })
    }

    /// Workspace size CUTLASS asks for at this problem shape. `0`
    /// when the symbol is missing — the caller's workspace-size
    /// check will trip if a non-zero requirement was silently ignored.
    #[cfg(feature = "cuda")]
    pub fn workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        self.fp8_gemm_blockscale_ws
            .map(|f| unsafe { f(m, n, k) })
            .unwrap_or(0)
    }

    /// SFA / SFB scratch sizes in bytes. `0` when the `.so` is missing
    /// the helper (legacy build) — caller must treat 0 as "CUTLASS
    /// path unavailable" and fall back.
    #[cfg(feature = "cuda")]
    pub fn sfa_bytes(&self, m: i32, k: i32) -> usize {
        self.sfa_bytes.map(|f| unsafe { f(m, k) }).unwrap_or(0)
    }
    #[cfg(feature = "cuda")]
    pub fn sfb_bytes(&self, n: i32, k: i32) -> usize {
        self.sfb_bytes.map(|f| unsafe { f(n, k) }).unwrap_or(0)
    }

    /// Populate SFA scratch from the per-token `a_scale[M]` vector.
    /// Kernel max-reduces each 128-row chunk into one scalar and
    /// replicates it across K/128 entries in the CUTLASS SFA layout.
    ///
    /// # Safety
    /// `a_scale` and `sfa` must be valid device pointers for the
    /// kernel's duration; `sfa` must be at least `sfa_bytes(m, k)` wide.
    #[cfg(feature = "cuda")]
    pub unsafe fn launch_prep_sfa(
        &self,
        a_scale: u64,
        sfa: u64,
        m: i32,
        k: i32,
        stream: u64,
    ) -> Result<()> {
        let f = self.prep_sfa.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx {
                    kernel: "prep_sfa_sm120 (missing from .so)",
                    stream,
                },
            )
        })?;
        let rc = f(
            a_scale as *const c_void,
            sfa as *mut c_void,
            m,
            k,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "prep_sfa_sm120",
                    stream,
                },
            ));
        }
        Ok(())
    }

    /// Populate SFB scratch by transposing row-major `[N/128, K/128]`
    /// `b_chscale` into the CUTLASS SFB layout (N-tile fastest).
    ///
    /// # Safety
    /// `b_chscale` and `sfb` must be valid device pointers; `sfb`
    /// must be at least `sfb_bytes(n, k)` wide.
    #[cfg(feature = "cuda")]
    pub unsafe fn launch_prep_sfb(
        &self,
        b_chscale: u64,
        sfb: u64,
        n: i32,
        k: i32,
        stream: u64,
    ) -> Result<()> {
        let f = self.prep_sfb.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx {
                    kernel: "prep_sfb_sm120 (missing from .so)",
                    stream,
                },
            )
        })?;
        let rc = f(
            b_chscale as *const c_void,
            sfb as *mut c_void,
            n,
            k,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "prep_sfb_sm120",
                    stream,
                },
            ));
        }
        Ok(())
    }

    /// # Safety
    /// All device pointers must be valid for the kernel's duration.
    /// `a_scale_sfa`, `b_scale_sfb` carry 128×128 block-scale
    /// semantics per `cutlass::detail::sm120_trivial_blockwise_scale_config`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_blockscale(
        &self,
        output: u64,
        a: u64,
        b: u64,
        a_scale_sfa: u64,
        b_scale_sfb: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        let f = self.fp8_gemm_blockscale.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx {
                    kernel: "fp8_gemm_blockscale_sm120 (missing from .so)",
                    stream,
                },
            )
        })?;
        let rc = f(
            output as *mut c_void,
            a as *const c_void,
            b as *const c_void,
            a_scale_sfa as *const c_void,
            b_scale_sfb as *const c_void,
            m,
            n,
            k,
            workspace as *mut c_void,
            workspace_size,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "fp8_gemm_blockscale_sm120",
                    stream,
                },
            ));
        }
        Ok(())
    }

    // ─── NVFP4 (Mistral 3.5) ─────────────────────────────────────────

    /// True iff the resolved `.so` exposes the full NVFP4 entry point
    /// set. Either every required symbol resolves or this returns
    /// false — partial binding is treated as missing.
    ///
    /// `nvfp4_prep_sfa` is the future single-call wrapper (today
    /// stubs to -100 even when bound); the runtime relies on the
    /// explicit chain `nvfp4_prep_act` -> `nvfp4_sfa_transform` ->
    /// `nvfp4_gemm`, so those three are the gating set.
    /// `nvfp4_gemm_decode_m1` is optional (the prefill kernel can
    /// serve m=1 too, just slower); treat it as a hint, not gate.
    #[cfg(feature = "cuda")]
    pub fn nvfp4_active(&self) -> bool {
        self.nvfp4_gemm.is_some()
            && self.nvfp4_workspace.is_some()
            && self.nvfp4_sfa_bytes.is_some()
            && self.nvfp4_prep_act.is_some()
            && self.nvfp4_sfa_transform.is_some()
            && self.nvfp4_natural_sfa_bytes.is_some()
    }

    /// `Ok(())` iff `nvfp4_active`. Used at Mistral 3.5 startup —
    /// missing symbols are a hard error per the integration spec
    /// ("missing required CUTLASS symbols are startup errors").
    #[cfg(feature = "cuda")]
    pub fn require_nvfp4(&self) -> Result<()> {
        if self.nvfp4_active() {
            return Ok(());
        }
        let mut missing: Vec<&'static str> = Vec::new();
        if self.nvfp4_gemm.is_none() {
            missing.push("cutlass_nvfp4_gemm_sm120");
        }
        if self.nvfp4_workspace.is_none() {
            missing.push("cutlass_nvfp4_gemm_sm120_workspace");
        }
        if self.nvfp4_sfa_bytes.is_none() {
            missing.push("cutlass_nvfp4_gemm_sm120_sfa_bytes");
        }
        if self.nvfp4_prep_act.is_none() {
            missing.push("cutlass_nvfp4_gemm_sm120_prep_act");
        }
        if self.nvfp4_sfa_transform.is_none() {
            missing.push("cutlass_nvfp4_gemm_sm120_sfa_natural_to_interleaved");
        }
        if self.nvfp4_natural_sfa_bytes.is_none() {
            missing.push("cutlass_nvfp4_gemm_sm120_prep_act_sfa_bytes");
        }
        // Diagnostic stderr line so the operator sees exactly which
        // symbols failed to bind without grepping a missing-feature
        // generic message. rvllm-cutlass deliberately avoids a
        // `tracing` dep so the lower crates stay ABI-light.
        eprintln!(
            "[rvllm-cutlass] Mistral 3.5 NVFP4 backend not loaded; \
             missing symbols: {missing:?} (path={})",
            self.so_path.display()
        );
        Err(RvllmError::cutlass(
            CutlassError::KernelLaunchFailed {
                variant: 0,
                cuda: rvllm_core::CudaErrorKind::Other,
            },
            CutlassCtx {
                kernel: "nvfp4 sm_120 symbol set incomplete: \
                         rebuild libcutlass_sm120.so with the NVFP4 \
                         source on top of build_cutlass_sm120_so.sh",
                stream: 0,
            },
        ))
    }

    /// CUTLASS workspace bytes for an NVFP4 GEMM at this problem
    /// shape. Returns 0 when the symbol is missing — the caller
    /// must already have failed via `require_nvfp4` for Mistral.
    #[cfg(feature = "cuda")]
    pub fn nvfp4_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        self.nvfp4_workspace
            .map(|f| unsafe { f(m, n, k) })
            .unwrap_or(0)
    }

    /// Activation-scale staging bytes for an NVFP4 GEMM (m × K/16
    /// E4M3, plus alignment slack). 0 when the symbol is missing.
    #[cfg(feature = "cuda")]
    pub fn nvfp4_sfa_bytes(&self, m: i32, k: i32) -> usize {
        self.nvfp4_sfa_bytes
            .map(|f| unsafe { f(m, k) })
            .unwrap_or(0)
    }

    /// Natural-layout SFA scratch size in bytes. Distinct from
    /// `nvfp4_sfa_bytes` — the natural intermediate is plain
    /// row-major `[m, k/16]` E4M3 (one byte per (row, K-block));
    /// the CUTLASS query reports the swizzled-and-padded final
    /// destination size.
    #[cfg(feature = "cuda")]
    pub fn nvfp4_natural_sfa_bytes(&self, m: i32, k: i32) -> usize {
        self.nvfp4_natural_sfa_bytes
            .map(|f| unsafe { f(m, k) })
            .unwrap_or(0)
    }

    /// Activation prep step 1: BF16/F16 -> NVFP4-packed bytes +
    /// natural-layout SFA. The natural SFA must be transformed via
    /// `launch_nvfp4_sfa_transform` before it can feed the GEMM.
    ///
    /// # Safety
    /// `a_input` must be `m * k * sizeof(elem_dtype)` bytes wide.
    /// `a_packed_out` must be at least `m * k / 2` bytes;
    /// `sfa_natural_out` must be at least `nvfp4_natural_sfa_bytes`
    /// bytes.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4_prep_act(
        &self,
        a_input: u64,
        a_packed_out: u64,
        sfa_natural_out: u64,
        m: i32,
        k: i32,
        a_input_dtype_f16_eq_0_bf16_eq_1: i32,
        stream: u64,
    ) -> Result<()> {
        let f = self.nvfp4_prep_act.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx { kernel: "nvfp4_prep_act_sm120 (missing from .so)", stream },
            )
        })?;
        let rc = f(
            a_input as *const c_void,
            a_packed_out as *mut c_void,
            sfa_natural_out as *mut c_void,
            m, k,
            a_input_dtype_f16_eq_0_bf16_eq_1,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx { kernel: "nvfp4_prep_act_sm120", stream },
            ));
        }
        Ok(())
    }

    /// Activation prep step 2: natural-layout `[m, k/16]` E4M3 ->
    /// CUTLASS interleaved SFA. Pure index permutation.
    ///
    /// # Safety
    /// `src_natural` must be `nvfp4_natural_sfa_bytes(m, k)` bytes;
    /// `dst_cutlass` must be `nvfp4_sfa_bytes(m, k)` bytes.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4_sfa_transform(
        &self,
        src_natural: u64,
        dst_cutlass: u64,
        m: i32,
        k: i32,
        stream: u64,
    ) -> Result<()> {
        let f = self.nvfp4_sfa_transform.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx { kernel: "nvfp4_sfa_transform_sm120 (missing from .so)", stream },
            )
        })?;
        let rc = f(
            src_natural as *const c_void,
            dst_cutlass as *mut c_void,
            m, k,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx { kernel: "nvfp4_sfa_transform_sm120", stream },
            ));
        }
        Ok(())
    }

    /// High-level convenience: prep_act + sfa_transform back-to-back.
    /// `a_packed_out` and `sfa_cutlass_out` are the GEMM-ready
    /// outputs; `sfa_natural_scratch` is consumed and overwritten,
    /// the caller can reuse it across layers.
    ///
    /// # Safety
    /// All four buffers must outlive the launch. `sfa_natural_scratch`
    /// must be at least `nvfp4_natural_sfa_bytes(m, k)`,
    /// `sfa_cutlass_out` must be at least `nvfp4_sfa_bytes(m, k)`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4_prep_sfa_chain(
        &self,
        a_input: u64,
        a_packed_out: u64,
        sfa_natural_scratch: u64,
        sfa_cutlass_out: u64,
        m: i32,
        k: i32,
        a_input_dtype_f16_eq_0_bf16_eq_1: i32,
        stream: u64,
    ) -> Result<()> {
        self.launch_nvfp4_prep_act(
            a_input,
            a_packed_out,
            sfa_natural_scratch,
            m, k,
            a_input_dtype_f16_eq_0_bf16_eq_1,
            stream,
        )?;
        self.launch_nvfp4_sfa_transform(
            sfa_natural_scratch,
            sfa_cutlass_out,
            m, k,
            stream,
        )
    }

    /// Quantize activations to FP8 + stage SFA scratch. Single launch
    /// per layer.
    ///
    /// # Safety
    /// All device pointers must be valid for the kernel's duration.
    /// `a_fp8_out` must have at least `m * k` bytes; `sfa_out` must
    /// have at least `nvfp4_sfa_bytes(m, k)` bytes.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4_prep_sfa(
        &self,
        a_input: u64,
        a_fp8_out: u64,
        sfa_out: u64,
        m: i32,
        k: i32,
        a_input_dtype_f16_eq_0_bf16_eq_1: i32,
        stream: u64,
    ) -> Result<()> {
        let f = self.nvfp4_prep_sfa.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx {
                    kernel: "nvfp4_prep_sfa_sm120 (missing from .so)",
                    stream,
                },
            )
        })?;
        let rc = f(
            a_input as *const c_void,
            a_fp8_out as *mut c_void,
            sfa_out as *mut c_void,
            m,
            k,
            a_input_dtype_f16_eq_0_bf16_eq_1,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "nvfp4_prep_sfa_sm120",
                    stream,
                },
            ));
        }
        Ok(())
    }

    /// Launch the NVFP4 GEMM. Selects the m=1 specialised kernel
    /// when one is loaded and `m == 1`; otherwise routes to the
    /// general batched-prefill kernel.
    ///
    /// # Safety
    /// All device pointers must be valid for the kernel's duration.
    /// `b_packed` is `[N, K/2]` U8; `b_scale_e4m3` is `[N, K/16]`
    /// E4M3; `b_global_scale_f32` is `[1]` F32. `a_fp8` is the
    /// activations FP8E4M3-quantized by `launch_nvfp4_prep_sfa`;
    /// `sfa` is the staged per-token scales.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4_gemm(
        &self,
        output: u64,
        a_fp8: u64,
        b_packed: u64,
        sfa: u64,
        b_scale_e4m3: u64,
        b_global_scale_f32: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        // Decode-fast-path selection: only when m==1 AND the
        // specialised symbol is present. Otherwise the general
        // kernel handles m=1 correctly, just at a small perf cost.
        let chosen = if m == 1 {
            self.nvfp4_gemm_decode_m1.or(self.nvfp4_gemm)
        } else {
            self.nvfp4_gemm
        };
        let kernel_name: &'static str = if m == 1 && self.nvfp4_gemm_decode_m1.is_some() {
            "nvfp4_gemm_sm120_decode_m1"
        } else {
            "nvfp4_gemm_sm120"
        };
        let f = chosen.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx {
                    kernel: "nvfp4_gemm_sm120 (missing from .so)",
                    stream,
                },
            )
        })?;
        let rc = f(
            output as *mut c_void,
            a_fp8 as *const c_void,
            b_packed as *const c_void,
            sfa as *const c_void,
            b_scale_e4m3 as *const c_void,
            b_global_scale_f32 as *const c_void,
            m,
            n,
            k,
            workspace as *mut c_void,
            workspace_size,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: kernel_name,
                    stream,
                },
            ));
        }
        Ok(())
    }
}

impl CutlassBackend {
    /// Construct a backend per device `CompileTarget`. `path` and
    /// `policy_variants` are only consulted for the `So` path — when
    /// the live device is sm_121 we skip `.so` loading entirely.
    /// On sm_121 we look for the matching `libcutlass_sm120.so` in
    /// the runtime arch's directory (Codex27-1 tightened earlier
    /// sm_120/sm_122 fallbacks to a single arch). The file is
    /// resolved in this order:
    ///   1. env `RVLLM_CUTLASS_SM120_SO` (absolute path).
    ///   2. `<path.parent().parent()>/sm_121/libcutlass_sm120.so`
    ///      — the layout produced by `kernels/build_cutlass_sm120_so.sh`
    ///      (auto-detect picks `sm_121a` on GB10 hosts so the .so
    ///      lands in `kernels/sm_121/`). Operators on RTX 5090 /
    ///      RTX 6000 Blackwell (sm_120) or sm_122 use the env override
    ///      since the resolver is locked to sm_121 here.
    /// If neither is present, fall through to `Absent` — previous
    /// behaviour, so a host without the library built keeps working
    /// via the PTX fallback.
    #[cfg(feature = "cuda")]
    pub fn load_for(
        target: Option<rvllm_core::CompileTarget>,
        path: PathBuf,
        policy_variants: &[VariantId],
    ) -> Result<Self> {
        if matches!(target, Some(rvllm_core::CompileTarget::Sm121)) {
            if let Some(sm120_path) = resolve_sm120_so_path(&path, "sm_121") {
                return Ok(CutlassBackend::SoSm120(CutlassSm120Lib::load(sm120_path)?));
            }
            return Ok(CutlassBackend::Absent);
        }
        Ok(CutlassBackend::So(CutlassLib::load(path, policy_variants)?))
    }

    /// Without the `cuda` feature there is no runtime to dlopen
    /// against, so every backend collapses to `Absent`. Callers
    /// already have to route through the non-CUDA code paths; this
    /// keeps the signature shape identical to the cuda build.
    #[cfg(not(feature = "cuda"))]
    pub fn load_for(
        _target: Option<rvllm_core::CompileTarget>,
        _path: PathBuf,
        _policy_variants: &[VariantId],
    ) -> Result<Self> {
        Ok(CutlassBackend::Absent)
    }

    /// Path the `.so` was (or would be) loaded from — exposed for
    /// probe / diagnostic output. Empty `PathBuf` on the `Absent`
    /// variant.
    #[must_use]
    pub fn so_path(&self) -> std::path::PathBuf {
        match self {
            CutlassBackend::So(lib) => lib.so_path.clone(),
            CutlassBackend::SoSm120(lib) => lib.so_path.clone(),
            CutlassBackend::Absent => PathBuf::new(),
        }
    }

    /// Dispatch `launch_fp8_gemm` to the underlying backend.
    ///
    /// # Safety
    /// Same as `CutlassLib::launch_fp8_gemm`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm(
        &self,
        plan: &crate::plan::Fp8GemmPlan,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::So(lib) => lib.launch_fp8_gemm(
                plan,
                output,
                a,
                b,
                a_scales,
                b_scale,
                workspace,
                workspace_size,
                stream,
            ),
            CutlassBackend::Absent => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm (CutlassBackend::Absent — sm_121 has no CUTLASS .so)",
                },
                CutlassCtx {
                    kernel: "fp8_gemm",
                    stream,
                },
            )),
            CutlassBackend::SoSm120(_) => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm (non-channelscale) — SoSm120 only ships the blockwise entry",
                },
                CutlassCtx {
                    kernel: "fp8_gemm",
                    stream,
                },
            )),
        }
    }

    /// Dispatch `launch_fp8_gemm_residual` to the underlying backend.
    ///
    /// # Safety
    /// Same as `CutlassLib::launch_fp8_gemm_residual`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_residual(
        &self,
        plan: &crate::plan::Fp8GemmPlan,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        residual: u64,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::So(lib) => lib.launch_fp8_gemm_residual(
                plan,
                output,
                a,
                b,
                a_scales,
                b_scale,
                residual,
                workspace,
                workspace_size,
                stream,
            ),
            CutlassBackend::Absent => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_residual (CutlassBackend::Absent — sm_121 has no CUTLASS .so)",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_residual",
                    stream,
                },
            )),
            CutlassBackend::SoSm120(_) => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_residual — SoSm120 has no residual-fused entry yet",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_residual",
                    stream,
                },
            )),
        }
    }
}

impl CutlassBackend {
    /// Dispatch `launch_fp8_gemm_channelscale` — upstream added this
    /// as a row×col-scale epilogue variant. `Absent` has no
    /// equivalent kernel; it returns `FeatureNotAvailable`.
    ///
    /// # Safety
    /// Same as `CutlassLib::launch_fp8_gemm_channelscale`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_channelscale(
        &self,
        output: u64,
        a: u64,
        b: u64,
        row_scale: u64,
        col_scale: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::So(lib) => lib.launch_fp8_gemm_channelscale(
                output,
                a,
                b,
                row_scale,
                col_scale,
                m,
                n,
                k,
                workspace,
                workspace_size,
                stream,
            ),
            CutlassBackend::Absent => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_channelscale (CutlassBackend::Absent — sm_121 has no CUTLASS .so)",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_channelscale",
                    stream,
                },
            )),
            CutlassBackend::SoSm120(_) => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_channelscale — SoSm120 uses the blockscale ABI; call launch_fp8_gemm_blockscale_sm120 instead",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_channelscale",
                    stream,
                },
            )),
        }
    }

    /// Dispatch `launch_fp8_gemm_blockscale_sm120` — native Blackwell-
    /// Geforce 128×128 blockwise FP8 GEMM. Only the `SoSm120` variant
    /// has this kernel; `So` (SM90) and `Absent` return
    /// `FeatureNotAvailable` so the caller falls back to the PTX /
    /// channelscale path.
    ///
    /// # Safety
    /// Same as `CutlassSm120Lib::launch_fp8_gemm_blockscale`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_blockscale_sm120(
        &self,
        output: u64,
        a: u64,
        b: u64,
        a_scale_sfa: u64,
        b_scale_sfb: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::SoSm120(lib) => lib.launch_fp8_gemm_blockscale(
                output,
                a,
                b,
                a_scale_sfa,
                b_scale_sfb,
                m,
                n,
                k,
                workspace,
                workspace_size,
                stream,
            ),
            CutlassBackend::So(_) => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_blockscale_sm120 — SM90 .so has no Blackwell blockwise kernel",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_blockscale_sm120",
                    stream,
                },
            )),
            CutlassBackend::Absent => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_blockscale_sm120 (CutlassBackend::Absent — libcutlass_sm120.so not built)",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_blockscale_sm120",
                    stream,
                },
            )),
        }
    }

    /// Workspace size for the SM120 blockwise kernel. Returns `0` for
    /// non-`SoSm120` variants so the caller can uniformly allocate
    /// `max(workspace_size(...), other_requirements)`.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn fp8_gemm_blockscale_sm120_workspace(&self, m: i32, n: i32, k: i32) -> usize {
        match self {
            CutlassBackend::SoSm120(lib) => lib.workspace_size(m, n, k),
            _ => 0,
        }
    }

    // ─── NVFP4 dispatchers (Mistral 3.5) ─────────────────────────

    /// Whether the NVFP4 entry-point set is fully resolved on this
    /// backend. Always false for `So` (SM90) and `Absent`.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn nvfp4_active(&self) -> bool {
        matches!(self, CutlassBackend::SoSm120(lib) if lib.nvfp4_active())
    }

    /// Hard-fail at startup if the active backend doesn't expose
    /// the NVFP4 symbols. Mistral 3.5 calls this in
    /// `Mistral35Bringup::load`; per the integration spec missing
    /// symbols are a startup error rather than a silent fall-through.
    #[cfg(feature = "cuda")]
    pub fn require_nvfp4(&self) -> Result<()> {
        match self {
            CutlassBackend::SoSm120(lib) => lib.require_nvfp4(),
            CutlassBackend::So(_) => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "nvfp4_gemm_sm120 requires the Blackwell .so backend; \
                         the active backend is the SM90 fp8 .so. \
                         Mistral 3.5 only runs on sm_121.",
                },
                CutlassCtx { kernel: "nvfp4 backend", stream: 0 },
            )),
            CutlassBackend::Absent => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "nvfp4_gemm_sm120: libcutlass_sm120.so not built or not found. \
                         Build via kernels/build_cutlass_sm120_so.sh sm_121a and \
                         either set RVLLM_CUTLASS_SM120_SO or place the .so under \
                         <kernels>/sm_121/libcutlass_sm120.so.",
                },
                CutlassCtx { kernel: "nvfp4 backend", stream: 0 },
            )),
        }
    }

    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn nvfp4_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        match self {
            CutlassBackend::SoSm120(lib) => lib.nvfp4_workspace_size(m, n, k),
            _ => 0,
        }
    }

    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn nvfp4_sfa_bytes(&self, m: i32, k: i32) -> usize {
        match self {
            CutlassBackend::SoSm120(lib) => lib.nvfp4_sfa_bytes(m, k),
            _ => 0,
        }
    }

    /// Natural-layout SFA scratch size — see
    /// `CutlassSm120Lib::nvfp4_natural_sfa_bytes`. Returns 0 for
    /// non-Sm120 backends.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn nvfp4_natural_sfa_bytes(&self, m: i32, k: i32) -> usize {
        match self {
            CutlassBackend::SoSm120(lib) => lib.nvfp4_natural_sfa_bytes(m, k),
            _ => 0,
        }
    }

    /// # Safety
    /// See `CutlassSm120Lib::launch_nvfp4_prep_act` for buffer
    /// requirements.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4_prep_act(
        &self,
        a_input: u64,
        a_packed_out: u64,
        sfa_natural_out: u64,
        m: i32,
        k: i32,
        a_input_dtype_f16_eq_0_bf16_eq_1: i32,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::SoSm120(lib) => lib.launch_nvfp4_prep_act(
                a_input, a_packed_out, sfa_natural_out, m, k,
                a_input_dtype_f16_eq_0_bf16_eq_1, stream,
            ),
            _ => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "nvfp4_prep_act requires CutlassBackend::SoSm120",
                },
                CutlassCtx { kernel: "nvfp4_prep_act", stream },
            )),
        }
    }

    /// # Safety
    /// See `CutlassSm120Lib::launch_nvfp4_sfa_transform` for buffer
    /// requirements.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4_sfa_transform(
        &self,
        src_natural: u64,
        dst_cutlass: u64,
        m: i32,
        k: i32,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::SoSm120(lib) => lib.launch_nvfp4_sfa_transform(
                src_natural, dst_cutlass, m, k, stream,
            ),
            _ => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "nvfp4_sfa_transform requires CutlassBackend::SoSm120",
                },
                CutlassCtx { kernel: "nvfp4_sfa_transform", stream },
            )),
        }
    }

    /// High-level prep: prep_act + sfa_transform back-to-back. The
    /// caller-supplied `sfa_natural_scratch` is consumed across the
    /// chain (overwritten then immediately read by the transform);
    /// reuse across layers is fine.
    ///
    /// # Safety
    /// All four buffers must outlive the launch and be sized per
    /// the underlying backend's queries.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4_prep_sfa_chain(
        &self,
        a_input: u64,
        a_packed_out: u64,
        sfa_natural_scratch: u64,
        sfa_cutlass_out: u64,
        m: i32,
        k: i32,
        a_input_dtype_f16_eq_0_bf16_eq_1: i32,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::SoSm120(lib) => lib.launch_nvfp4_prep_sfa_chain(
                a_input, a_packed_out, sfa_natural_scratch,
                sfa_cutlass_out, m, k,
                a_input_dtype_f16_eq_0_bf16_eq_1, stream,
            ),
            _ => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "nvfp4_prep_sfa_chain requires CutlassBackend::SoSm120",
                },
                CutlassCtx { kernel: "nvfp4_prep_sfa_chain", stream },
            )),
        }
    }

    /// # Safety
    /// All device pointers must be valid for the kernel's duration.
    /// See `CutlassSm120Lib::launch_nvfp4_prep_sfa` for layout
    /// requirements.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4_prep_sfa(
        &self,
        a_input: u64,
        a_fp8_out: u64,
        sfa_out: u64,
        m: i32,
        k: i32,
        a_input_dtype_f16_eq_0_bf16_eq_1: i32,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::SoSm120(lib) => lib.launch_nvfp4_prep_sfa(
                a_input,
                a_fp8_out,
                sfa_out,
                m,
                k,
                a_input_dtype_f16_eq_0_bf16_eq_1,
                stream,
            ),
            _ => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "nvfp4_prep_sfa requires CutlassBackend::SoSm120",
                },
                CutlassCtx { kernel: "nvfp4_prep_sfa", stream },
            )),
        }
    }

    /// # Safety
    /// All device pointers must be valid for the kernel's duration.
    /// See `CutlassSm120Lib::launch_nvfp4_gemm` for layout requirements.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4_gemm(
        &self,
        output: u64,
        a_fp8: u64,
        b_packed: u64,
        sfa: u64,
        b_scale_e4m3: u64,
        b_global_scale_f32: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::SoSm120(lib) => lib.launch_nvfp4_gemm(
                output,
                a_fp8,
                b_packed,
                sfa,
                b_scale_e4m3,
                b_global_scale_f32,
                m,
                n,
                k,
                workspace,
                workspace_size,
                stream,
            ),
            _ => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "nvfp4_gemm requires CutlassBackend::SoSm120",
                },
                CutlassCtx { kernel: "nvfp4_gemm_sm120", stream },
            )),
        }
    }
}

impl From<CutlassLib> for CutlassBackend {
    fn from(lib: CutlassLib) -> Self {
        CutlassBackend::So(lib)
    }
}

/// Map a policy `VariantId` to the C-linkage symbol names in
/// `libcutlass_kernels.so`. id <100 uses the `cutlass_fp8_gemm_v{id}`
/// autotune suite; id >=100 uses `cutlass_fp8_gemm_residual_v{id-100}`.
/// Returns a heap-allocated pair (empty when out-of-range).
fn v2_symbol_names(vid: VariantId) -> (String, String) {
    if vid.0 >= 100 {
        let i = vid.0 - 100;
        (
            format!("cutlass_fp8_gemm_residual_v{i}"),
            format!("cutlass_fp8_gemm_residual_v{i}_workspace_size"),
        )
    } else {
        let i = vid.0;
        (
            format!("cutlass_fp8_gemm_v{i}"),
            format!("cutlass_fp8_gemm_v{i}_workspace_size"),
        )
    }
}

fn cutlass_miss(path: &std::path::Path) -> RvllmError {
    RvllmError::cutlass(
        CutlassError::AutotuneCacheMiss {
            m: 0,
            n: 0,
            k: 0,
            dtype: rvllm_core::DType::Fp8E4M3,
        },
        CutlassCtx {
            kernel: "libcutlass_kernels.so",
            stream: 0,
        },
    )
    // note: the actual error classifies as SoMissing; we overload
    // AutotuneCacheMiss here until the core error enum adds CutlassSoMissing.
    .into_cutlass_so_missing(path.to_path_buf())
}

fn variant_missing(path: &std::path::Path, vid: VariantId, kind: &'static str) -> RvllmError {
    RvllmError::cutlass(
        CutlassError::KernelLaunchFailed {
            variant: vid.0,
            cuda: rvllm_core::CudaErrorKind::ModuleLoadFailed,
        },
        CutlassCtx {
            kernel: kind,
            stream: 0,
        },
    )
    .into_cutlass_variant_missing(path.to_path_buf(), vid)
}

// Small extension to chain on an existing error. Avoids adding new
// variants to rvllm_core::RvllmError for this one case.
trait CutlassErrExt {
    fn into_cutlass_so_missing(self, path: PathBuf) -> RvllmError;
    fn into_cutlass_variant_missing(self, path: PathBuf, vid: VariantId) -> RvllmError;
}

impl CutlassErrExt for RvllmError {
    fn into_cutlass_so_missing(self, path: PathBuf) -> RvllmError {
        // Repackage with a loader-style path context.
        RvllmError::Loader {
            err: rvllm_core::LoaderError::Corrupt {
                detail: format!("libcutlass_kernels.so not at {}", path.display()),
            },
            ctx: rvllm_core::LoaderCtx {
                path,
                tensor: None,
            },
            bt: std::backtrace::Backtrace::capture(),
        }
    }
    fn into_cutlass_variant_missing(self, path: PathBuf, vid: VariantId) -> RvllmError {
        RvllmError::Loader {
            err: rvllm_core::LoaderError::Corrupt {
                detail: format!(
                    "libcutlass_kernels.so at {} missing variant {}",
                    path.display(),
                    vid.0,
                ),
            },
            ctx: rvllm_core::LoaderCtx {
                path,
                tensor: None,
            },
            bt: std::backtrace::Backtrace::capture(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_so_rejected() {
        let err = CutlassLib::load("/nonexistent/libcutlass.so".into(), &[]).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("libcutlass_kernels.so not at"));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn nvfp4_absent_backend_rejects() {
        // Default-feature build has no `.so` mechanism; require_nvfp4
        // / nvfp4_active are still callable on `Absent` and must
        // refuse cleanly so a Mistral 3.5 startup on a no-cuda build
        // gets a typed error rather than a confusing crash.
        let backend = CutlassBackend::Absent;
        // The require_nvfp4 method is gated on `cuda`; this branch
        // just sanity-checks that the no-cuda build still compiles.
        let _ = backend;
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_require_on_absent_errors() {
        let backend = CutlassBackend::Absent;
        assert!(!backend.nvfp4_active());
        let err = backend.require_nvfp4().unwrap_err();
        assert!(format!("{err}").contains("nvfp4"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_workspace_zero_when_inactive() {
        let backend = CutlassBackend::Absent;
        assert_eq!(backend.nvfp4_workspace_size(1, 12288, 12288), 0);
        assert_eq!(backend.nvfp4_sfa_bytes(1, 12288), 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_natural_sfa_bytes_zero_when_inactive() {
        let backend = CutlassBackend::Absent;
        assert_eq!(backend.nvfp4_natural_sfa_bytes(1, 12288), 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_chain_dispatch_rejects_absent_backend() {
        let backend = CutlassBackend::Absent;
        // SAFETY: the dispatch fails before touching the dummy
        // pointers, so passing zeros is fine.
        let r = unsafe {
            backend.launch_nvfp4_prep_sfa_chain(0, 0, 0, 0, 1, 16, 0, 0)
        };
        assert!(r.is_err());
        let r = unsafe {
            backend.launch_nvfp4_prep_act(0, 0, 0, 1, 16, 0, 0)
        };
        assert!(r.is_err());
        let r = unsafe {
            backend.launch_nvfp4_sfa_transform(0, 0, 1, 16, 0)
        };
        assert!(r.is_err());
    }
}
