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
                    let f: libloading::Symbol<Fp8GemmResidualFn> =
                        lib.get(fn_c.as_bytes()).map_err(|_| {
                            variant_missing(&path, vid, "fp8_gemm_residual")
                        })?;
                    let w: libloading::Symbol<WorkspaceSizeFn> =
                        lib.get(ws_c.as_bytes()).map_err(|_| {
                            variant_missing(&path, vid, "fp8_gemm_residual_ws")
                        })?;
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

        Ok(Self {
            so_path: path,
            _lib: lib,
            fp8_gemm,
            fp8_gemm_ws,
            fp8_gemm_residual,
            fp8_gemm_residual_ws,
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
