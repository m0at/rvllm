//! cuBLAS GEMM operations for linear algebra.

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

use crate::Result;

/// Wrapper around cuBLAS for matrix operations.
pub struct CublasHandle {
    blas: CudaBlas,
    device: Arc<CudaDevice>,
}

impl CublasHandle {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let blas = CudaBlas::new(device.clone())
            .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS init failed: {e}")))?;
        Ok(Self { blas, device })
    }

    /// Returns a reference to the underlying device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// SGEMM: C = alpha * A * B + beta * C
    ///
    /// A: [m, k], B: [k, n], C: [m, n] in row-major layout.
    ///
    /// cuBLAS expects column-major, so we compute C^T = B^T * A^T which
    /// yields the correct row-major result without explicit transposes.
    ///
    /// # Safety
    /// Caller must ensure slices have the correct lengths:
    /// a >= m*k, b >= k*n, c >= m*n.
    pub fn sgemm(
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
        // SAFETY: cuBLAS reads/writes device memory through valid CudaSlice handles.
        // Row-major trick: C^T = B^T * A^T  =>  call gemm(N, N, n, m, k, B, A, C)
        // but since the *data* is row-major and cuBLAS sees it as column-major-transposed,
        // we pass Op_T for both and swap A/B.
        unsafe {
            self.blas
                .gemm(
                    cublasOperation_t::CUBLAS_OP_T, // B transposed (row->col)
                    cublasOperation_t::CUBLAS_OP_N, // A as-is in cuBLAS view
                    n as i32,
                    m as i32,
                    k as i32,
                    alpha,
                    b,
                    n as i32, // ldb = n (row stride of B in row-major)
                    a,
                    k as i32, // lda = k (row stride of A in row-major)
                    beta,
                    c,
                    n as i32, // ldc = n (row stride of C in row-major)
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS sgemm failed: {e}")))?;
        }
        Ok(())
    }

    /// HGEMM: half-precision GEMM for f16.
    ///
    /// Same layout conventions as [`sgemm`](Self::sgemm) but operates on f16
    /// tensors. Internally uses f32 accumulation for numerical stability
    /// (matching cuBLAS CUBLAS_COMPUTE_32F behavior on Ampere+).
    ///
    /// This halves memory bandwidth for weight-bound operations (all linear
    /// projections in the transformer), which is the primary bottleneck for
    /// inference at moderate batch sizes.
    pub fn hgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: half::f16,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        beta: half::f16,
        c: &mut CudaSlice<half::f16>,
    ) -> Result<()> {
        // Row-major trick: C^T = B^T * A^T
        // cuBLAS sees column-major, so we swap A<->B and adjust dims.
        //
        // For f16, cudarc's Gemm trait implementation handles the GemmEx
        // dispatch automatically when T=f16.
        unsafe {
            self.blas
                .gemm(
                    cublasOperation_t::CUBLAS_OP_T,
                    cublasOperation_t::CUBLAS_OP_N,
                    n as i32,
                    m as i32,
                    k as i32,
                    alpha,
                    b,
                    n as i32,
                    a,
                    k as i32,
                    beta,
                    c,
                    n as i32,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS hgemm failed: {e}")))?;
        }
        Ok(())
    }

    /// Batched SGEMM for multiple independent matrix multiplications (e.g. multi-head attention).
    ///
    /// Each triple (a_batch[i], b_batch[i], c_batch[i]) is an independent GEMM with
    /// the same m/n/k dimensions.
    pub fn sgemm_batched(
        &self,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a_batch: &[&CudaSlice<f32>],
        _b_batch: &[&CudaSlice<f32>],
        _beta: f32,
        _c_batch: &mut [&mut CudaSlice<f32>],
    ) -> Result<()> {
        // TODO: implement via cublasSgemmBatched or cublasSgemmStridedBatched
        Err(crate::LLMError::GpuError(
            "sgemm_batched not yet implemented".into(),
        ))
    }

    /// SGEMV: y = alpha * A * x + beta * y
    ///
    /// A: [m, n] row-major, x: [n], y: [m].
    ///
    /// For row-major A, cuBLAS (column-major) sees A^T, so we pass CUBLAS_OP_T
    /// to get the correct row-major matrix-vector product.
    pub fn sgemv(
        &self,
        m: usize,
        n: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        beta: f32,
        y: &mut CudaSlice<f32>,
    ) -> Result<()> {
        // SAFETY: cuBLAS reads/writes device memory through valid CudaSlice handles.
        // Row-major A stored contiguously is column-major A^T with dims (n, m).
        // We want y = A * x  =>  cublas: y = Op(A_col) * x  where A_col is (n,m).
        // Op = CUBLAS_OP_T gives us A^T_col = A_row which is what we want.
        unsafe {
            self.blas
                .gemv(
                    cublasOperation_t::CUBLAS_OP_T,
                    n as i32, // rows of A in column-major = cols of A in row-major
                    m as i32, // cols of A in column-major = rows of A in row-major
                    alpha,
                    a,
                    n as i32, // lda = n (row stride in row-major = leading dim in col-major)
                    x,
                    1, // incx
                    beta,
                    y,
                    1, // incy
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS sgemv failed: {e}")))?;
        }
        Ok(())
    }
}
