//! cublasLt GEMM operations with automatic algorithm selection and split-K.
//!
//! cublasLt provides better performance than cublasGemmEx for tall-skinny
//! shapes (small M, large N/K) common in the decode path, thanks to automatic
//! split-K heuristics and a larger algorithm search space.

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig, MatmulShared};
use cudarc::cublaslt::sys as lt_sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::f16;
use std::sync::Arc;

use crate::{LLMError, Result};

/// Threshold: use cublasLt for decode-sized GEMMs (M <= this value).
/// Above this we fall back to standard cuBLAS which has less overhead
/// for large batch prefill shapes.
pub const CUBLASLT_M_THRESHOLD: usize = 256;

/// 4 MiB workspace for split-K heuristics in cublasLt.
const FP8_WORKSPACE_SIZE: usize = 4 * 1024 * 1024;

/// Wrapper around cudarc's `CudaBlasLT` with workspace for heuristic algo selection.
pub struct CublasLtOps {
    handle: CudaBlasLT,
    stream: Arc<CudaStream>,
    /// Persistent workspace for cublasLtMatmul (split-K, etc).
    workspace: CudaSlice<u8>,
}

impl CublasLtOps {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let handle = CudaBlasLT::new(stream.clone())
            .map_err(|e| LLMError::GpuError(format!("CudaBlasLT init failed: {e}")))?;
        let workspace = unsafe { stream.alloc::<u8>(FP8_WORKSPACE_SIZE) }
            .map_err(|e| LLMError::GpuError(format!("cublasLt workspace alloc: {e}")))?;
        Ok(Self { handle, stream, workspace })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
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
        // Row-major C[m,n] = A[m,k] @ B[n,k]^T
        // cuBLAS col-major: C_col[n,m] = B_col[k,n]^T @ A_col[k,m]
        //   B row[n,k] = col[k,n]. transa=true -> transpose to [n,k]. lda=k.
        //   A row[m,k] = col[k,m]. transb=false -> [k,m]. ldb=k.
        //   C_col[n,m]. ldc=n.
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
                .map_err(|e| LLMError::GpuError(format!("cublasLt hgemm_a_bt failed: {e}")))?;
        }
        Ok(())
    }

    /// Row-major HGEMM into a view via cublasLt. Accepts any DevicePtr/DevicePtrMut
    /// so callers can pass CudaViewMut (sub-slices of a larger buffer).
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
                .map_err(|e| LLMError::GpuError(format!("cublasLt hgemm_a_bt_into failed: {e}")))?;
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

    /// FP8 E4M3 GEMM via raw device pointers.
    /// C_f16[m,n] = A_fp8[m,k] @ B_fp8[n,k]^T
    /// All pointers are raw u64 device addresses to avoid borrow issues.
    pub fn fp8_gemm_a_bt_raw(
        &self,
        m: usize,
        n: usize,
        k: usize,
        input_fp8_ptr: u64,   // [m, k] FP8
        weight_fp8_ptr: u64,  // [n, k] FP8
        output_f16_ptr: u64,  // [m, n] FP16
    ) -> Result<()> {
        use std::ffi::c_void;

        unsafe {
            let handle = self.handle.handle();

            let mut desc: lt_sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
            let s = lt_sys::cublasLtMatmulDescCreate(&mut desc, lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, lt_sys::cudaDataType_t::CUDA_R_32F);
            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(LLMError::GpuError(format!("fp8 desc create: {s:?}")));
            }

            let trans_a = lt_sys::cublasOperation_t::CUBLAS_OP_T;
            let trans_b = lt_sys::cublasOperation_t::CUBLAS_OP_N;
            lt_sys::cublasLtMatmulDescSetAttribute(desc, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &trans_a as *const _ as *const c_void, std::mem::size_of_val(&trans_a));
            lt_sys::cublasLtMatmulDescSetAttribute(desc, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &trans_b as *const _ as *const c_void, std::mem::size_of_val(&trans_b));

            let mut layout_a: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut layout_b: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut layout_c: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            // A = weight[n,k] transposed, B = input[m,k], C = output[m,n]
            // col-major: A_col[k,n], B_col[k,m], C_col[n,m]
            lt_sys::cublasLtMatrixLayoutCreate(&mut layout_a, lt_sys::cudaDataType_t::CUDA_R_8F_E4M3, k as u64, n as u64, k as i64);
            lt_sys::cublasLtMatrixLayoutCreate(&mut layout_b, lt_sys::cudaDataType_t::CUDA_R_8F_E4M3, k as u64, m as u64, k as i64);
            lt_sys::cublasLtMatrixLayoutCreate(&mut layout_c, lt_sys::cudaDataType_t::CUDA_R_16F, n as u64, m as u64, n as i64);

            let mut pref: lt_sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
            lt_sys::cublasLtMatmulPreferenceCreate(&mut pref);
            let ws_size: usize = FP8_WORKSPACE_SIZE;
            lt_sys::cublasLtMatmulPreferenceSetAttribute(pref, lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size as *const _ as *const c_void, std::mem::size_of_val(&ws_size));

            let mut heur = std::mem::zeroed::<lt_sys::cublasLtMatmulHeuristicResult_t>();
            let mut ret: i32 = 0;
            let s = lt_sys::cublasLtMatmulAlgoGetHeuristic(handle, desc, layout_a, layout_b, layout_c, layout_c, pref, 1, &mut heur, &mut ret);
            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
                lt_sys::cublasLtMatmulPreferenceDestroy(pref);
                lt_sys::cublasLtMatrixLayoutDestroy(layout_a);
                lt_sys::cublasLtMatrixLayoutDestroy(layout_b);
                lt_sys::cublasLtMatrixLayoutDestroy(layout_c);
                lt_sys::cublasLtMatmulDescDestroy(desc);
                return Err(LLMError::GpuError(format!("fp8 no algo: {s:?} ret={ret}")));
            }

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            let (ws_ptr, _ws_guard) = DevicePtr::device_ptr(&self.workspace, &self.stream);
            let s = lt_sys::cublasLtMatmul(
                handle, desc,
                &alpha as *const f32 as *const c_void,
                weight_fp8_ptr as *const c_void, layout_a,   // A = weight
                input_fp8_ptr as *const c_void, layout_b,    // B = input
                &beta as *const f32 as *const c_void,
                output_f16_ptr as *mut c_void, layout_c,
                output_f16_ptr as *mut c_void, layout_c,
                &heur.algo,
                ws_ptr as *mut c_void, ws_size,
                self.stream.cu_stream(),
            );

            lt_sys::cublasLtMatmulPreferenceDestroy(pref);
            lt_sys::cublasLtMatrixLayoutDestroy(layout_a);
            lt_sys::cublasLtMatrixLayoutDestroy(layout_b);
            lt_sys::cublasLtMatrixLayoutDestroy(layout_c);
            lt_sys::cublasLtMatmulDescDestroy(desc);

            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(LLMError::GpuError(format!("fp8 matmul: {s:?}")));
            }
        }
        Ok(())
    }
}
