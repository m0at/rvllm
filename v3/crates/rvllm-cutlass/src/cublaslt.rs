//! cuBLASLt FP8 GEMM wrappers.
//!
//! Three entry points share one dispatcher (`fp8_gemm_inner`):
//!   - `fp8_gemm`          : D = A * B^T
//!   - `fp8_gemm_bias`     : D = A * B^T + bias   (CUBLASLT_EPILOGUE_BIAS)
//!   - `fp8_gemm_residual` : D = A * B^T + C      (alpha=1, beta=1; C is residual)
//!
//! FP8 E4M3 TN on Hopper: we pass row-major inputs and let cuBLASLt
//! (column-major) see their transposes, swapping A/B arguments and
//! setting transa=T, transb=N.

#[cfg(feature = "cuda")]
use cudarc::cublaslt::sys as lt;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

/// Key for the per-shape algorithm cache. Distinguishes plain / bias /
/// residual dispatch because the matmul descriptor differs and cuBLASLt's
/// heuristic returns different algos.
#[cfg(feature = "cuda")]
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct AlgoKey {
    m: i32,
    n: i32,
    k: i32,
    kind: u8, // 0 = plain, 1 = bias, 2 = residual (beta=1)
}

pub struct CublasLt {
    #[cfg(feature = "cuda")]
    handle: lt::cublasLtHandle_t,
    workspace: u64,
    workspace_bytes: usize,
    /// Per-(M,N,K,kind) cache of the heuristic-picked algorithm. cuBLASLt's
    /// algo struct is opaque but `Copy+Hash+Eq` — we reuse it on subsequent
    /// calls with the same shape instead of re-running the heuristic.
    #[cfg(feature = "cuda")]
    algo_cache: Mutex<HashMap<AlgoKey, lt::cublasLtMatmulAlgo_t>>,
}

unsafe impl Send for CublasLt {}
unsafe impl Sync for CublasLt {}

impl CublasLt {
    #[cfg(feature = "cuda")]
    pub fn new(workspace: u64, workspace_bytes: usize) -> Result<Self> {
        let mut handle: lt::cublasLtHandle_t = std::ptr::null_mut();
        let r = unsafe { lt::cublasLtCreate(&mut handle) };
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(RvllmError::cuda(
                "cublasLtCreate",
                CudaErrorKind::Other,
                CudaCtx::setup(),
            ));
        }
        Ok(Self {
            handle,
            workspace,
            workspace_bytes,
            algo_cache: Mutex::new(HashMap::new()),
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(workspace: u64, workspace_bytes: usize) -> Result<Self> {
        Ok(Self {
            workspace,
            workspace_bytes,
        })
    }

    /// Plain FP8 E4M3 matmul: D = A * B^T.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(a_fp8, b_fp8, 0, 0, d_f16, m, n, k, a_scale, b_scale, stream, false)
    }

    /// FP8 matmul with row-broadcast f16 bias epilogue.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm_bias(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        bias_f16: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(
            a_fp8, b_fp8, bias_f16, 0, d_f16, m, n, k, a_scale, b_scale, stream, false,
        )
    }

    /// FP8 matmul with residual-add epilogue: D = A*B^T + residual (C).
    /// `residual_f16` and `d_f16` may alias to do the add in place.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm_residual(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        residual_f16: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(
            a_fp8, b_fp8, 0, residual_f16, d_f16, m, n, k, a_scale, b_scale, stream, true,
        )
    }

    /// Shared body. `bias_f16=0` means no bias epilogue. `beta_one=true`
    /// enables the residual path with C = `c_residual` (or d_f16 if
    /// c_residual is 0).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn fp8_gemm_inner(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        bias_f16: u64,
        c_residual: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
        beta_one: bool,
    ) -> Result<()> {
        let mut desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
        let rc = lt::cublasLtMatmulDescCreate(
            &mut desc,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType_t::CUDA_R_32F,
        );
        if rc != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("cublasLtMatmulDescCreate"));
        }

        let transa: i32 = 1; // T
        let transb: i32 = 0; // N
        set_attr(
            desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &transa as *const _ as *const _,
            std::mem::size_of_val(&transa),
        )?;
        set_attr(
            desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &transb as *const _ as *const _,
            std::mem::size_of_val(&transb),
        )?;

        if bias_f16 != 0 {
            let epi = lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
            set_attr(
                desc,
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epi as *const _ as *const _,
                std::mem::size_of_val(&epi),
            )?;
            set_attr(
                desc,
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_f16 as *const _ as *const _,
                std::mem::size_of_val(&bias_f16),
            )?;
        }

        set_attr(
            desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale as *const _ as *const _,
            std::mem::size_of_val(&a_scale),
        )?;
        set_attr(
            desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale as *const _ as *const _,
            std::mem::size_of_val(&b_scale),
        )?;

        // Layouts: col-major view of our row-major buffers, TN.
        let mut layout_a: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut layout_b: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut layout_d: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut layout_a,
            lt::cudaDataType_t::CUDA_R_8F_E4M3,
            k as u64,
            n as u64,
            k as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout A"));
        }
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut layout_b,
            lt::cudaDataType_t::CUDA_R_8F_E4M3,
            k as u64,
            m as u64,
            k as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout B"));
        }
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut layout_d,
            lt::cudaDataType_t::CUDA_R_16F,
            n as u64,
            m as u64,
            n as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout D"));
        }

        // Heuristic path: check the per-shape cache first; on miss, run
        // cublasLtMatmulAlgoGetHeuristic once and save the algo for
        // future calls. At N=128 we hit this for 5 unique (M,N,K,kind)
        // tuples per step — 4 repeated across 28 layers — so reuse is
        // essentially 100% after the first step.
        let key = AlgoKey {
            m,
            n,
            k,
            kind: if bias_f16 != 0 {
                1
            } else if beta_one {
                2
            } else {
                0
            },
        };
        let cached_algo = self
            .algo_cache
            .lock()
            .ok()
            .and_then(|c| c.get(&key).copied());
        let algo = if let Some(a) = cached_algo {
            a
        } else {
            // Top-K heuristic + on-device timing. cuBLASLt's heuristic
            // estimator is often wrong at small M for FP8 (picks a
            // 128x128x128 tile even when most of M is padding). Pulling
            // the top-16 candidates and timing each with CUDA events
            // closes the small-batch gap measurably. One-time cost per
            // unique (M,N,K,kind); cached thereafter.
            const CANDIDATES: i32 = 16;
            const WARMUP_ITERS: u32 = 3;
            const TIMED_ITERS: u32 = 10;

            let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
            let r = lt::cublasLtMatmulPreferenceCreate(&mut pref);
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(cublaslt_err("preference create"));
            }
            let ws_bytes = self.workspace_bytes;
            let r = lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const _ as *const _,
                std::mem::size_of::<usize>(),
            );
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(cublaslt_err("pref set workspace"));
            }

            let mut heur: [lt::cublasLtMatmulHeuristicResult_t; CANDIDATES as usize] =
                std::mem::zeroed();
            let mut ret: i32 = 0;
            let r = lt::cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                desc,
                layout_a,
                layout_b,
                layout_d,
                layout_d,
                pref,
                CANDIDATES,
                heur.as_mut_ptr(),
                &mut ret,
            );
            lt::cublasLtMatmulPreferenceDestroy(pref);
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
                return Err(cublaslt_err("heuristic"));
            }

            // Time each candidate. We reuse the caller's A/B/D device
            // pointers — safe because the caller's buffers are already
            // valid for the intended matmul on this stream.
            let mut best_algo = heur[0].algo;
            let mut best_ns: f64 = f64::MAX;
            let one_f32: f32 = 1.0;
            let zero_f32: f32 = 0.0;
            let c_ptr_probe = if beta_one && c_residual != 0 {
                c_residual as *const _
            } else {
                d_f16 as *const _
            };
            let beta_probe = if beta_one { &one_f32 } else { &zero_f32 };

            let mut ev_start: cudarc::driver::sys::CUevent = std::ptr::null_mut();
            let mut ev_stop: cudarc::driver::sys::CUevent = std::ptr::null_mut();
            cudarc::driver::sys::cuEventCreate(&mut ev_start, 0);
            cudarc::driver::sys::cuEventCreate(&mut ev_stop, 0);

            for i in 0..ret as usize {
                // warmup
                for _ in 0..WARMUP_ITERS {
                    let _ = lt::cublasLtMatmul(
                        self.handle, desc,
                        &one_f32 as *const _ as *const _,
                        b_fp8 as *const _, layout_a,
                        a_fp8 as *const _, layout_b,
                        beta_probe as *const _ as *const _,
                        c_ptr_probe, layout_d,
                        d_f16 as *mut _, layout_d,
                        &heur[i].algo,
                        self.workspace as *mut _, self.workspace_bytes,
                        stream as _,
                    );
                }
                cudarc::driver::sys::cuEventRecord(ev_start, stream as _);
                for _ in 0..TIMED_ITERS {
                    let r2 = lt::cublasLtMatmul(
                        self.handle, desc,
                        &one_f32 as *const _ as *const _,
                        b_fp8 as *const _, layout_a,
                        a_fp8 as *const _, layout_b,
                        beta_probe as *const _ as *const _,
                        c_ptr_probe, layout_d,
                        d_f16 as *mut _, layout_d,
                        &heur[i].algo,
                        self.workspace as *mut _, self.workspace_bytes,
                        stream as _,
                    );
                    if r2 != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        break;
                    }
                }
                cudarc::driver::sys::cuEventRecord(ev_stop, stream as _);
                cudarc::driver::sys::cuEventSynchronize(ev_stop);
                let mut ms: f32 = 0.0;
                cudarc::driver::sys::cuEventElapsedTime(&mut ms, ev_start, ev_stop);
                let per_iter_ns = (ms as f64) * 1.0e6 / (TIMED_ITERS as f64);
                if per_iter_ns < best_ns && per_iter_ns > 0.0 {
                    best_ns = per_iter_ns;
                    best_algo = heur[i].algo;
                }
            }
            cudarc::driver::sys::cuEventDestroy_v2(ev_start);
            cudarc::driver::sys::cuEventDestroy_v2(ev_stop);

            if let Ok(mut c) = self.algo_cache.lock() {
                c.insert(key, best_algo);
            }
            best_algo
        };

        let one_f32: f32 = 1.0;
        let zero_f32: f32 = 0.0;
        let c_ptr = if beta_one && c_residual != 0 {
            c_residual as *const _
        } else {
            d_f16 as *const _
        };
        let r = lt::cublasLtMatmul(
            self.handle,
            desc,
            &one_f32 as *const _ as *const _,
            b_fp8 as *const _, // cublas "A" := our weight (transa=T)
            layout_a,
            a_fp8 as *const _, // cublas "B" := our activation (transb=N)
            layout_b,
            if beta_one { &one_f32 } else { &zero_f32 } as *const _ as *const _,
            c_ptr,
            layout_d,
            d_f16 as *mut _,
            layout_d,
            &algo,
            self.workspace as *mut _,
            self.workspace_bytes,
            stream as _,
        );

        lt::cublasLtMatrixLayoutDestroy(layout_d);
        lt::cublasLtMatrixLayoutDestroy(layout_b);
        lt::cublasLtMatrixLayoutDestroy(layout_a);
        lt::cublasLtMatmulDescDestroy(desc);

        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("cublasLtMatmul"));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn cublaslt_err(op: &'static str) -> RvllmError {
    RvllmError::cuda(op, CudaErrorKind::LaunchFailed, CudaCtx::setup())
}

#[cfg(feature = "cuda")]
unsafe fn set_attr(
    desc: lt::cublasLtMatmulDesc_t,
    attr: lt::cublasLtMatmulDescAttributes_t,
    buf: *const core::ffi::c_void,
    size: usize,
) -> Result<()> {
    let r = lt::cublasLtMatmulDescSetAttribute(desc, attr, buf, size);
    if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(cublaslt_err("cublasLtMatmulDescSetAttribute"));
    }
    Ok(())
}

impl Drop for CublasLt {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            if !self.handle.is_null() {
                let _ = lt::cublasLtDestroy(self.handle);
            }
        }
    }
}
