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
        Ok(Self {
            so_path: path,
            _lib: lib,
            fp8_gemm_blockscale,
            fp8_gemm_blockscale_ws,
            sfa_bytes,
            sfb_bytes,
            prep_sfa,
            prep_sfb,
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
}

impl CutlassBackend {
    /// Construct a backend per device `CompileTarget`. `path` and
    /// `policy_variants` are only consulted for the `So` path — when
    /// the live device is sm_121 we skip `.so` loading entirely.
    /// On sm_121 we first look for a sm_120 `libcutlass_sm120.so`.
    /// The file is resolved in this order:
    ///   1. env `RVLLM_CUTLASS_SM120_SO` (absolute path).
    ///   2. `<path.parent()>/sm_120/libcutlass_sm120.so` — the layout
    ///      produced by `kernels/build_cutlass_sm120_so.sh`, which
    ///      makes "kernels_dir" a natural resolution root for
    ///      `probe-gemma4-load` / `rvllm-server`.
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
}
