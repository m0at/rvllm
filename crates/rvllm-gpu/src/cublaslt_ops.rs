//! cublasLt GEMM operations with automatic algorithm selection and split-K.
//!
//! cublasLt provides better performance than cublasGemmEx for tall-skinny
//! shapes (small M, large N/K) common in the decode path, thanks to automatic
//! split-K heuristics and a larger algorithm search space.

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::f16;
use std::sync::Arc;

use crate::{LLMError, Result};

/// Threshold: use cublasLt for decode-sized GEMMs (M <= this value).
/// Above this we fall back to standard cuBLAS which has less overhead
/// for large batch prefill shapes.
pub const CUBLASLT_M_THRESHOLD: usize = 32;

/// Wrapper around cudarc's `CudaBlasLT` with workspace for heuristic algo selection.
pub struct CublasLtOps {
    handle: CudaBlasLT,
    stream: Arc<CudaStream>,
}

impl CublasLtOps {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let handle = CudaBlasLT::new(stream.clone())
            .map_err(|e| LLMError::GpuError(format!("CudaBlasLT init failed: {e}")))?;
        Ok(Self { handle, stream })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
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
}
