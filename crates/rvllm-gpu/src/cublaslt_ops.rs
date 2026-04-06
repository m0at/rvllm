//! cublasLt GEMM operations with automatic algorithm selection and split-K.
//!
//! cublasLt provides better performance than cublasGemmEx for tall-skinny
//! shapes (small M, large N/K) common in the decode path, thanks to automatic
//! split-K heuristics and a larger algorithm search space.

use crate::cublaslt_raw as lt_sys;
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig, MatmulShared};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::f16;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use crate::{LLMError, Result};

/// Threshold: use cublasLt for decode-sized and mid-batch f16 GEMMs
/// (M <= this value). Above this we fall back to standard cuBLAS which
/// has less overhead for larger prefill shapes.
pub const CUBLASLT_M_THRESHOLD: usize = 128;

/// Runtime cublasLt workspace.
///
/// Keep this aligned with the autotuner's search budget so cached plans are
/// replayed under the same workspace constraint they were selected for.
const CUBLASLT_WORKSPACE_SIZE: usize = 32 * 1024 * 1024;
const FP8_WORKSPACE_SIZE: usize = CUBLASLT_WORKSPACE_SIZE;

fn is_retryable_cached_algo_failure(status: lt_sys::cublasStatus_t) -> bool {
    matches!(
        status,
        lt_sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
            | lt_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE
            | lt_sys::cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED
    )
}

/// Cached FP8 matmul plan (descriptors + algo) for a specific (M, N, K) shape.
struct Fp8Plan {
    desc: lt_sys::cublasLtMatmulDesc_t,
    layout_a: lt_sys::cublasLtMatrixLayout_t,
    layout_b: lt_sys::cublasLtMatrixLayout_t,
    layout_c: lt_sys::cublasLtMatrixLayout_t,
    algo: lt_sys::cublasLtMatmulAlgo_t,
}

// cublasLt handles are thread-safe (used on a single GPU stream)
unsafe impl Send for Fp8Plan {}
unsafe impl Sync for Fp8Plan {}

impl Drop for Fp8Plan {
    fn drop(&mut self) {
        unsafe {
            lt_sys::cublasLtMatrixLayoutDestroy(self.layout_a);
            lt_sys::cublasLtMatrixLayoutDestroy(self.layout_b);
            lt_sys::cublasLtMatrixLayoutDestroy(self.layout_c);
            lt_sys::cublasLtMatmulDescDestroy(self.desc);
        }
    }
}

/// Cached plan for an autotuned f16 GEMM shape.
struct F16Plan {
    desc: lt_sys::cublasLtMatmulDesc_t,
    layout_a: lt_sys::cublasLtMatrixLayout_t,
    layout_b: lt_sys::cublasLtMatrixLayout_t,
    layout_c: lt_sys::cublasLtMatrixLayout_t,
    algo: lt_sys::cublasLtMatmulAlgo_t,
}

unsafe impl Send for F16Plan {}
unsafe impl Sync for F16Plan {}

impl Drop for F16Plan {
    fn drop(&mut self) {
        unsafe {
            lt_sys::cublasLtMatrixLayoutDestroy(self.layout_a);
            lt_sys::cublasLtMatrixLayoutDestroy(self.layout_b);
            lt_sys::cublasLtMatrixLayoutDestroy(self.layout_c);
            lt_sys::cublasLtMatmulDescDestroy(self.desc);
        }
    }
}

/// Wrapper around cudarc's `CudaBlasLT` with workspace for heuristic algo selection.
pub struct CublasLtOps {
    handle: CudaBlasLT,
    stream: Arc<CudaStream>,
    /// Persistent workspace for cublasLtMatmul (split-K, etc).
    workspace: CudaSlice<u8>,
    /// Cached FP8 plans keyed by (m, n, k).
    fp8_cache: std::cell::RefCell<std::collections::HashMap<(usize, usize, usize), Fp8Plan>>,
    /// Cached autotuned f16 plans keyed by (m, n, k).
    f16_cache: std::cell::RefCell<std::collections::HashMap<(usize, usize, usize), F16Plan>>,
}

impl CublasLtOps {
    fn m_threshold(&self) -> usize {
        std::env::var("RVLLM_CUBLASLT_M_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(CUBLASLT_M_THRESHOLD)
    }

    fn log_shape_choice_once(&self, m: usize, n: usize, k: usize, cached: bool, use_lt: bool) {
        if std::env::var("RVLLM_LOG_GEMM_SHAPES").map_or(true, |v| v != "1") {
            return;
        }
        static SEEN: OnceLock<Mutex<HashSet<(usize, usize, usize, bool, bool)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut lock = seen.lock().unwrap();
        if lock.insert((m, n, k, cached, use_lt)) {
            tracing::info!(m, n, k, cached, use_lt, "f16 gemm shape decision");
        }
    }

    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let handle = CudaBlasLT::new(stream.clone())
            .map_err(|e| LLMError::GpuError(format!("CudaBlasLT init failed: {e}")))?;
        let workspace = unsafe { stream.alloc::<u8>(CUBLASLT_WORKSPACE_SIZE) }
            .map_err(|e| LLMError::GpuError(format!("cublasLt workspace alloc: {e}")))?;
        Ok(Self {
            handle,
            stream,
            workspace,
            fp8_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            f16_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    /// Install autotuned algorithms. Call after `CublasAutotuner::autotune_model`.
    /// Creates cached descriptors + layouts for each shape so the hot path
    /// skips heuristic lookup entirely.
    pub fn install_autotuned(&self, tuner: &crate::cublas_autotune::CublasAutotuner) {
        use std::ffi::c_void;

        let handle = unsafe { *self.handle.handle() };
        let mut cache = self.f16_cache.borrow_mut();

        for (&(m, n, k), result) in tuner.iter() {
            unsafe {
                let mut desc: lt_sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
                let s = lt_sys::cublasLtMatmulDescCreate(
                    &mut desc,
                    lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    lt_sys::cudaDataType_t::CUDA_R_32F,
                );
                if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    tracing::warn!(m, n, k, "install_autotuned: desc create failed");
                    continue;
                }
                let trans_a = lt_sys::cublasOperation_t::CUBLAS_OP_T;
                let trans_b = lt_sys::cublasOperation_t::CUBLAS_OP_N;
                lt_sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &trans_a as *const _ as *const c_void,
                    std::mem::size_of_val(&trans_a),
                );
                lt_sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &trans_b as *const _ as *const c_void,
                    std::mem::size_of_val(&trans_b),
                );

                let f16_type = lt_sys::cudaDataType_t::CUDA_R_16F;
                let mut la: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut lb: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut lc: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                lt_sys::cublasLtMatrixLayoutCreate(&mut la, f16_type, k as u64, n as u64, k as i64);
                lt_sys::cublasLtMatrixLayoutCreate(&mut lb, f16_type, k as u64, m as u64, k as i64);
                lt_sys::cublasLtMatrixLayoutCreate(&mut lc, f16_type, n as u64, m as u64, n as i64);

                cache.insert(
                    (m, n, k),
                    F16Plan {
                        desc,
                        layout_a: la,
                        layout_b: lb,
                        layout_c: lc,
                        algo: result.algo,
                    },
                );
                let _ = handle; // suppress unused warning
            }
            tracing::debug!(
                m,
                n,
                k,
                time_us = result.time_us,
                "installed autotuned algo"
            );
        }
        tracing::info!(
            count = cache.len(),
            "autotuned algos installed into cublasLt cache"
        );
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn has_autotuned_f16(&self, m: usize, n: usize, k: usize) -> bool {
        self.f16_cache.borrow().contains_key(&(m, n, k))
    }

    pub fn should_use_f16_for_shape(&self, m: usize, n: usize, k: usize) -> bool {
        let cached = self.has_autotuned_f16(m, n, k);
        let use_lt = cached || m <= self.m_threshold();
        self.log_shape_choice_once(m, n, k, cached, use_lt);
        use_lt
    }

    /// Raw cublasLt handle for autotuning.
    pub fn handle(&self) -> &lt_sys::cublasLtHandle_t {
        self.handle.handle()
    }

    /// Row-major HGEMM via cublasLt: `C[m,n] = alpha * A[m,k] @ B^T[k,n] + beta * C[m,n]`
    ///
    /// Same layout as `CublasOps::hgemm_a_bt` but uses cublasLt's heuristic
    /// algorithm selection with workspace. Better for small M (decode path)
    /// due to automatic split-K.
    pub fn hgemm_a_bt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<f16>,
        b: &CudaSlice<f16>,
        beta: f32,
        c: &mut CudaSlice<f16>,
    ) -> Result<()> {
        self.hgemm_a_bt_into(m, n, k, alpha, a, b, beta, c)
    }

    /// Row-major HGEMM into a view via cublasLt. Accepts any DevicePtr/DevicePtrMut
    /// so callers can pass CudaViewMut (sub-slices of a larger buffer).
    ///
    /// If an autotuned algo exists for this (m,n,k) shape, uses the cached plan
    /// (zero descriptor creation, zero heuristic lookup). Otherwise falls back
    /// to top-1 heuristic.
    pub fn hgemm_a_bt_into(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &impl DevicePtr<f16>,
        b: &impl DevicePtr<f16>,
        beta: f32,
        c: &mut impl DevicePtrMut<f16>,
    ) -> Result<()> {
        use std::ffi::c_void;

        let (a_ptr, _ga) = a.device_ptr(&self.stream);
        let (b_ptr, _gb) = b.device_ptr(&self.stream);
        let (c_ptr, _gc) = c.device_ptr_mut(&self.stream);
        let (ws_ptr, _gw) = DevicePtr::device_ptr(&self.workspace, &self.stream);
        let cu_stream = self.stream.cu_stream();

        // Fast path: use cached autotuned plan (no descriptor/layout creation)
        let mut retry_without_cache = false;
        let cache = self.f16_cache.borrow();
        if let Some(plan) = cache.get(&(m, n, k)) {
            let alpha_f32 = alpha;
            let beta_f32 = beta;
            unsafe {
                let s = lt_sys::cublasLtMatmul(
                    *self.handle.handle(),
                    plan.desc,
                    &alpha_f32 as *const f32 as *const c_void,
                    b_ptr as *const c_void,
                    plan.layout_a,
                    a_ptr as *const c_void,
                    plan.layout_b,
                    &beta_f32 as *const f32 as *const c_void,
                    c_ptr as *mut c_void,
                    plan.layout_c,
                    c_ptr as *mut c_void,
                    plan.layout_c,
                    &plan.algo,
                    ws_ptr as *mut c_void,
                    CUBLASLT_WORKSPACE_SIZE,
                    lt_sys::cu_stream_to_cuda_stream(cu_stream),
                );
                if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    if is_retryable_cached_algo_failure(s) {
                        retry_without_cache = true;
                    } else {
                        return Err(LLMError::GpuError(format!(
                            "AUTOTUNED cublasLt GEMM FAILED: status={s:?} shape=({m},{n},{k})"
                        )));
                    }
                }
            }
            if !retry_without_cache {
                return Ok(());
            }
        }
        drop(cache);

        if retry_without_cache {
            self.f16_cache.borrow_mut().remove(&(m, n, k));
            tracing::warn!(
                m,
                n,
                k,
                "cached cublasLt algo rejected at runtime, evicting in-memory plan and retrying"
            );
        }

        // Slow path: create descriptors, get top-1 heuristic
        unsafe {
            let handle = *self.handle.handle();
            let mut desc: lt_sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
            let s = lt_sys::cublasLtMatmulDescCreate(
                &mut desc,
                lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                lt_sys::cudaDataType_t::CUDA_R_32F,
            );
            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(LLMError::GpuError(format!("hgemm_a_bt_into desc: {s:?}")));
            }

            let trans_a = lt_sys::cublasOperation_t::CUBLAS_OP_T;
            let trans_b = lt_sys::cublasOperation_t::CUBLAS_OP_N;
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &trans_a as *const _ as *const c_void,
                std::mem::size_of_val(&trans_a),
            );
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &trans_b as *const _ as *const c_void,
                std::mem::size_of_val(&trans_b),
            );

            let f16_type = lt_sys::cudaDataType_t::CUDA_R_16F;
            let mut la: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut lb: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut lc: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            lt_sys::cublasLtMatrixLayoutCreate(&mut la, f16_type, k as u64, n as u64, k as i64);
            lt_sys::cublasLtMatrixLayoutCreate(&mut lb, f16_type, k as u64, m as u64, k as i64);
            lt_sys::cublasLtMatrixLayoutCreate(&mut lc, f16_type, n as u64, m as u64, n as i64);

            let mut pref: lt_sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
            lt_sys::cublasLtMatmulPreferenceCreate(&mut pref);
            let ws_size: usize = CUBLASLT_WORKSPACE_SIZE;
            lt_sys::cublasLtMatmulPreferenceSetAttribute(
                pref,
                lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const _ as *const c_void,
                std::mem::size_of_val(&ws_size),
            );

            let mut heur = std::mem::zeroed::<lt_sys::cublasLtMatmulHeuristicResult_t>();
            let mut ret: i32 = 0;
            let s = lt_sys::cublasLtMatmulAlgoGetHeuristic(
                handle, desc, la, lb, lc, lc, pref, 1, &mut heur, &mut ret,
            );
            lt_sys::cublasLtMatmulPreferenceDestroy(pref);

            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
                lt_sys::cublasLtMatrixLayoutDestroy(la);
                lt_sys::cublasLtMatrixLayoutDestroy(lb);
                lt_sys::cublasLtMatrixLayoutDestroy(lc);
                lt_sys::cublasLtMatmulDescDestroy(desc);
                return Err(LLMError::GpuError(format!(
                    "hgemm_a_bt_into no algo: {s:?}"
                )));
            }

            let alpha_f32 = alpha;
            let beta_f32 = beta;
            let s = lt_sys::cublasLtMatmul(
                handle,
                desc,
                &alpha_f32 as *const f32 as *const c_void,
                b_ptr as *const c_void,
                la,
                a_ptr as *const c_void,
                lb,
                &beta_f32 as *const f32 as *const c_void,
                c_ptr as *mut c_void,
                lc,
                c_ptr as *mut c_void,
                lc,
                &heur.algo,
                ws_ptr as *mut c_void,
                ws_size,
                lt_sys::cu_stream_to_cuda_stream(cu_stream),
            );

            lt_sys::cublasLtMatrixLayoutDestroy(la);
            lt_sys::cublasLtMatrixLayoutDestroy(lb);
            lt_sys::cublasLtMatrixLayoutDestroy(lc);
            lt_sys::cublasLtMatmulDescDestroy(desc);

            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(LLMError::GpuError(format!("hgemm_a_bt_into matmul: {s:?}")));
            }
        }
        Ok(())
    }

    /// Row-major SGEMM via cublasLt: `C[m,n] = alpha * A[m,k] @ B^T[k,n] + beta * C[m,n]`
    pub fn sgemm_a_bt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        beta: f32,
        c: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: n as u64,
            n: m as u64,
            k: k as u64,
            alpha,
            lda: k as i64,
            ldb: k as i64,
            beta,
            ldc: n as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };

        unsafe {
            self.handle
                .matmul(cfg, b, a, c, None, None)
                .map_err(|e| LLMError::GpuError(format!("cublasLt sgemm_a_bt failed: {e}")))?;
        }
        Ok(())
    }

    /// FP8 E4M3 GEMM via raw device pointers with cached plan.
    /// C_f16[m,n] = A_fp8[m,k] @ B_fp8[n,k]^T
    /// First call for a given (m,n,k) creates descriptors + selects algo.
    /// Subsequent calls reuse the cached plan (just the matmul dispatch).
    pub fn fp8_gemm_a_bt_raw(
        &self,
        m: usize,
        n: usize,
        k: usize,
        input_fp8_ptr: u64,
        weight_fp8_ptr: u64,
        output_f16_ptr: u64,
    ) -> Result<()> {
        use std::ffi::c_void;

        let key = (m, n, k);
        if !self.fp8_cache.borrow().contains_key(&key) {
            unsafe {
                let handle = *self.handle.handle();
                let mut desc: lt_sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
                let s = lt_sys::cublasLtMatmulDescCreate(
                    &mut desc,
                    lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    lt_sys::cudaDataType_t::CUDA_R_32F,
                );
                if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(LLMError::GpuError(format!("fp8 desc create: {s:?}")));
                }
                let trans_a = lt_sys::cublasOperation_t::CUBLAS_OP_T;
                let trans_b = lt_sys::cublasOperation_t::CUBLAS_OP_N;
                lt_sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &trans_a as *const _ as *const c_void,
                    std::mem::size_of_val(&trans_a),
                );
                lt_sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &trans_b as *const _ as *const c_void,
                    std::mem::size_of_val(&trans_b),
                );

                let mut layout_a: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut layout_b: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut layout_c: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                lt_sys::cublasLtMatrixLayoutCreate(
                    &mut layout_a,
                    lt_sys::cudaDataType_t::CUDA_R_8F_E4M3,
                    k as u64,
                    n as u64,
                    k as i64,
                );
                lt_sys::cublasLtMatrixLayoutCreate(
                    &mut layout_b,
                    lt_sys::cudaDataType_t::CUDA_R_8F_E4M3,
                    k as u64,
                    m as u64,
                    k as i64,
                );
                lt_sys::cublasLtMatrixLayoutCreate(
                    &mut layout_c,
                    lt_sys::cudaDataType_t::CUDA_R_16F,
                    n as u64,
                    m as u64,
                    n as i64,
                );

                let mut pref: lt_sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
                lt_sys::cublasLtMatmulPreferenceCreate(&mut pref);
                let ws_size: usize = FP8_WORKSPACE_SIZE;
                lt_sys::cublasLtMatmulPreferenceSetAttribute(pref, lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size as *const _ as *const c_void, std::mem::size_of_val(&ws_size));

                let mut heur = std::mem::zeroed::<lt_sys::cublasLtMatmulHeuristicResult_t>();
                let mut ret: i32 = 0;
                let s = lt_sys::cublasLtMatmulAlgoGetHeuristic(
                    handle, desc, layout_a, layout_b, layout_c, layout_c, pref, 1, &mut heur,
                    &mut ret,
                );
                lt_sys::cublasLtMatmulPreferenceDestroy(pref);
                if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
                    lt_sys::cublasLtMatrixLayoutDestroy(layout_a);
                    lt_sys::cublasLtMatrixLayoutDestroy(layout_b);
                    lt_sys::cublasLtMatrixLayoutDestroy(layout_c);
                    lt_sys::cublasLtMatmulDescDestroy(desc);
                    return Err(LLMError::GpuError(format!("fp8 no algo: {s:?} ret={ret}")));
                }
                self.fp8_cache.borrow_mut().insert(
                    key,
                    Fp8Plan {
                        desc,
                        layout_a,
                        layout_b,
                        layout_c,
                        algo: heur.algo,
                    },
                );
            }
        }

        let cache = self.fp8_cache.borrow();
        let plan = cache.get(&key).unwrap();
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            let handle = *self.handle.handle();
            let (ws_ptr, _ws_guard) = DevicePtr::device_ptr(&self.workspace, &self.stream);
            let s = lt_sys::cublasLtMatmul(
                handle,
                plan.desc,
                &alpha as *const f32 as *const c_void,
                weight_fp8_ptr as *const c_void,
                plan.layout_a,
                input_fp8_ptr as *const c_void,
                plan.layout_b,
                &beta as *const f32 as *const c_void,
                output_f16_ptr as *mut c_void,
                plan.layout_c,
                output_f16_ptr as *mut c_void,
                plan.layout_c,
                &plan.algo,
                ws_ptr as *mut c_void,
                FP8_WORKSPACE_SIZE,
                lt_sys::cu_stream_to_cuda_stream(self.stream.cu_stream()),
            );
            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(LLMError::GpuError(format!("fp8 matmul: {s:?}")));
            }
        }
        Ok(())
    }

    /// Row-major HGEMM with fused bias epilogue:
    /// `C[m,n] = alpha * A[m,k] @ B^T[k,n] + bias[n]`
    /// Bias vector [n] is broadcast across all M rows.
    pub fn hgemm_a_bt_bias_into(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &impl DevicePtr<f16>,
        b: &impl DevicePtr<f16>,
        bias_ptr: u64,
        c: &mut impl DevicePtrMut<f16>,
    ) -> Result<()> {
        use std::ffi::c_void;

        let (a_ptr, _ga) = a.device_ptr(&self.stream);
        let (b_ptr, _gb) = b.device_ptr(&self.stream);
        let (c_ptr, _gc) = c.device_ptr_mut(&self.stream);
        let (ws_ptr, _gw) = DevicePtr::device_ptr(&self.workspace, &self.stream);
        let cu_stream = self.stream.cu_stream();

        unsafe {
            let handle = *self.handle.handle();
            let mut desc: lt_sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
            let s = lt_sys::cublasLtMatmulDescCreate(
                &mut desc,
                lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                lt_sys::cudaDataType_t::CUDA_R_32F,
            );
            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(LLMError::GpuError(format!("hgemm_bias desc: {s:?}")));
            }

            let trans_a = lt_sys::cublasOperation_t::CUBLAS_OP_T;
            let trans_b = lt_sys::cublasOperation_t::CUBLAS_OP_N;
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &trans_a as *const _ as *const c_void,
                std::mem::size_of_val(&trans_a),
            );
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &trans_b as *const _ as *const c_void,
                std::mem::size_of_val(&trans_b),
            );

            // Set BIAS epilogue
            let epilogue = lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue as *const _ as *const c_void,
                std::mem::size_of_val(&epilogue),
            );
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_ptr as *const u64 as *const c_void,
                std::mem::size_of::<u64>(),
            );

            let f16_type = lt_sys::cudaDataType_t::CUDA_R_16F;
            let mut la: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut lb: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut lc: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            lt_sys::cublasLtMatrixLayoutCreate(&mut la, f16_type, k as u64, n as u64, k as i64);
            lt_sys::cublasLtMatrixLayoutCreate(&mut lb, f16_type, k as u64, m as u64, k as i64);
            lt_sys::cublasLtMatrixLayoutCreate(&mut lc, f16_type, n as u64, m as u64, n as i64);

            let mut pref: lt_sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
            lt_sys::cublasLtMatmulPreferenceCreate(&mut pref);
            let ws_size: usize = FP8_WORKSPACE_SIZE;
            lt_sys::cublasLtMatmulPreferenceSetAttribute(
                pref,
                lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const _ as *const c_void,
                std::mem::size_of_val(&ws_size),
            );

            let mut heur = std::mem::zeroed::<lt_sys::cublasLtMatmulHeuristicResult_t>();
            let mut ret: i32 = 0;
            let s = lt_sys::cublasLtMatmulAlgoGetHeuristic(
                handle, desc, la, lb, lc, lc, pref, 1, &mut heur, &mut ret,
            );
            lt_sys::cublasLtMatmulPreferenceDestroy(pref);

            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
                lt_sys::cublasLtMatrixLayoutDestroy(la);
                lt_sys::cublasLtMatrixLayoutDestroy(lb);
                lt_sys::cublasLtMatrixLayoutDestroy(lc);
                lt_sys::cublasLtMatmulDescDestroy(desc);
                return Err(LLMError::GpuError(format!("hgemm_bias no algo: {s:?}")));
            }

            let alpha_f32 = alpha;
            let beta_f32: f32 = 0.0;
            let s = lt_sys::cublasLtMatmul(
                handle,
                desc,
                &alpha_f32 as *const f32 as *const c_void,
                b_ptr as *const c_void,
                la,
                a_ptr as *const c_void,
                lb,
                &beta_f32 as *const f32 as *const c_void,
                c_ptr as *mut c_void,
                lc,
                c_ptr as *mut c_void,
                lc,
                &heur.algo,
                ws_ptr as *mut c_void,
                ws_size,
                lt_sys::cu_stream_to_cuda_stream(cu_stream),
            );

            lt_sys::cublasLtMatrixLayoutDestroy(la);
            lt_sys::cublasLtMatrixLayoutDestroy(lb);
            lt_sys::cublasLtMatrixLayoutDestroy(lc);
            lt_sys::cublasLtMatmulDescDestroy(desc);

            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(LLMError::GpuError(format!("hgemm_bias matmul: {s:?}")));
            }
        }
        Ok(())
    }

    /// FP8 GEMM with fused bias epilogue via raw device pointers.
    /// C_f16[m,n] = A_fp8[m,k] @ B_fp8[n,k]^T + bias_f16[n]
    pub fn fp8_gemm_a_bt_bias_raw(
        &self,
        m: usize,
        n: usize,
        k: usize,
        input_fp8_ptr: u64,
        weight_fp8_ptr: u64,
        bias_f16_ptr: u64,
        output_f16_ptr: u64,
    ) -> Result<()> {
        use std::ffi::c_void;

        unsafe {
            let handle = *self.handle.handle();
            let mut desc: lt_sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
            let s = lt_sys::cublasLtMatmulDescCreate(
                &mut desc,
                lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                lt_sys::cudaDataType_t::CUDA_R_32F,
            );
            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(LLMError::GpuError(format!("fp8_bias desc: {s:?}")));
            }

            let trans_a = lt_sys::cublasOperation_t::CUBLAS_OP_T;
            let trans_b = lt_sys::cublasOperation_t::CUBLAS_OP_N;
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &trans_a as *const _ as *const c_void,
                std::mem::size_of_val(&trans_a),
            );
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &trans_b as *const _ as *const c_void,
                std::mem::size_of_val(&trans_b),
            );

            // BIAS epilogue
            let epilogue = lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue as *const _ as *const c_void,
                std::mem::size_of_val(&epilogue),
            );
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_f16_ptr as *const u64 as *const c_void,
                std::mem::size_of::<u64>(),
            );

            let mut layout_a: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut layout_b: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut layout_c: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            lt_sys::cublasLtMatrixLayoutCreate(
                &mut layout_a,
                lt_sys::cudaDataType_t::CUDA_R_8F_E4M3,
                k as u64,
                n as u64,
                k as i64,
            );
            lt_sys::cublasLtMatrixLayoutCreate(
                &mut layout_b,
                lt_sys::cudaDataType_t::CUDA_R_8F_E4M3,
                k as u64,
                m as u64,
                k as i64,
            );
            lt_sys::cublasLtMatrixLayoutCreate(
                &mut layout_c,
                lt_sys::cudaDataType_t::CUDA_R_16F,
                n as u64,
                m as u64,
                n as i64,
            );

            let mut pref: lt_sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
            lt_sys::cublasLtMatmulPreferenceCreate(&mut pref);
            let ws_size: usize = FP8_WORKSPACE_SIZE;
            lt_sys::cublasLtMatmulPreferenceSetAttribute(pref, lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size as *const _ as *const c_void, std::mem::size_of_val(&ws_size));

            let mut heur = std::mem::zeroed::<lt_sys::cublasLtMatmulHeuristicResult_t>();
            let mut ret: i32 = 0;
            let s = lt_sys::cublasLtMatmulAlgoGetHeuristic(
                handle, desc, layout_a, layout_b, layout_c, layout_c, pref, 1, &mut heur, &mut ret,
            );
            lt_sys::cublasLtMatmulPreferenceDestroy(pref);
            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
                lt_sys::cublasLtMatrixLayoutDestroy(layout_a);
                lt_sys::cublasLtMatrixLayoutDestroy(layout_b);
                lt_sys::cublasLtMatrixLayoutDestroy(layout_c);
                lt_sys::cublasLtMatmulDescDestroy(desc);
                return Err(LLMError::GpuError(format!(
                    "fp8_bias no algo: {s:?} ret={ret}"
                )));
            }

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            let (ws_ptr, _ws_guard) = DevicePtr::device_ptr(&self.workspace, &self.stream);
            let s = lt_sys::cublasLtMatmul(
                handle,
                desc,
                &alpha as *const f32 as *const c_void,
                weight_fp8_ptr as *const c_void,
                layout_a,
                input_fp8_ptr as *const c_void,
                layout_b,
                &beta as *const f32 as *const c_void,
                output_f16_ptr as *mut c_void,
                layout_c,
                output_f16_ptr as *mut c_void,
                layout_c,
                &heur.algo,
                ws_ptr as *mut c_void,
                FP8_WORKSPACE_SIZE,
                lt_sys::cu_stream_to_cuda_stream(self.stream.cu_stream()),
            );

            lt_sys::cublasLtMatrixLayoutDestroy(layout_a);
            lt_sys::cublasLtMatrixLayoutDestroy(layout_b);
            lt_sys::cublasLtMatrixLayoutDestroy(layout_c);
            lt_sys::cublasLtMatmulDescDestroy(desc);

            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(LLMError::GpuError(format!("fp8_bias matmul: {s:?}")));
            }
        }
        Ok(())
    }
}
