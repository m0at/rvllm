//! Higher-level cuBLAS operations built on top of cudarc's Gemm trait.
//!
//! These functions encapsulate the `unsafe` cuBLAS calls so that downstream
//! crates (like rvllm-model-runner, which `#![forbid(unsafe_code)]`) can use
//! GPU linear algebra without unsafe blocks.

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaSlice};
use half::f16;
use std::sync::Arc;

use crate::{LLMError, Result};

/// Wrapper that owns a `CudaBlas` handle and exposes safe GEMM operations.
pub struct CublasOps {
    cublas: CudaBlas,
    device: Arc<CudaDevice>,
}

impl CublasOps {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let cublas = CudaBlas::new(device.clone())
            .map_err(|e| LLMError::GpuError(format!("CudaBlas init failed: {e}")))?;
        Ok(Self { cublas, device })
    }

    /// Reference to the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Row-major SGEMM: `C[m,n] = alpha * A[m,k] @ B^T[k,n] + beta * C[m,n]`
    ///
    /// A is `[m, k]` row-major, B is `[n, k]` row-major (transposed relative to
    /// standard GEMM), C is `[m, n]` row-major.
    ///
    /// This is the operation needed for linear layers where weight is stored as
    /// `[out_features, in_features]`.
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
        // Row-major to column-major mapping:
        //   A[m,k] row  = [k,m] col, ld=k
        //   B[n,k] row  = [k,n] col, ld=k
        //   C[m,n] row  = [n,m] col, ld=n
        //
        // Want: C_col[n,m] = op(A_cublas)[n,k] @ op(B_cublas)[k,m]
        //   A_cublas_ptr = B data [k,n] col. OP_T -> [n,k]. lda=k.
        //   B_cublas_ptr = A data [k,m] col. OP_N -> [k,m]. ldb=k.
        //   C_cublas = [n,m] col. ldc=n.
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: k as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,
        };

        // SAFETY: CudaSlice pointers are valid device memory on the same device
        // as the CudaBlas handle. Caller is responsible for ensuring buffer lengths
        // are >= m*k, n*k, m*n respectively.
        unsafe {
            self.cublas
                .gemm(cfg, b, a, c)
                .map_err(|e| LLMError::GpuError(format!("cuBLAS sgemm_a_bt failed: {e}")))?;
        }
        Ok(())
    }

    /// Row-major SGEMM: `C[m,n] = alpha * A[m,k] @ B[k,n] + beta * C[m,n]`
    ///
    /// Standard (non-transposed) GEMM. All matrices row-major.
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
        // Same derivation as CublasHandle::sgemm.
        // C[m,n] = A[m,k] @ B[k,n]  all row-major.
        // cuBLAS col-major: C_col[n,m] = B_col[n,k]^T @ A_col[k,m]
        //   actually using OP_N on both with swapped pointers:
        //     A_cublas_ptr = B [k,n] row = [n,k] col. With OP_T: [k,n] col = [n,k]^T. Hmm.
        //
        // Standard trick: C^T = B^T * A^T. All stored row-major.
        // cuBLAS sees B^T_col = B_row memory, A^T_col = A_row memory.
        // C^T_col = C_row memory.
        // cuBLAS: C^T_col[n,m] = B_row_as_col[n,k] * A_row_as_col[k,m]
        //   B is [k,n] row = [n,k] col. We need [n,k] for the multiply. OP_N: as-is [n,k].
        //   Wait, B[k,n] row-major = [n,k] col-major, OP_N gives [n,k] col.
        //   A[m,k] row-major = [k,m] col-major, OP_N gives [k,m] col.
        //   C[m,n] row-major = [n,m] col-major.
        //   [n,m] = [n,k] @ [k,m]. Checks out.
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: n as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,
        };

        // SAFETY: Same invariants as sgemm_a_bt.
        unsafe {
            self.cublas
                .gemm(cfg, b, a, c)
                .map_err(|e| LLMError::GpuError(format!("cuBLAS sgemm failed: {e}")))?;
        }
        Ok(())
    }

    /// Row-major HGEMM: `C[m,n] = alpha * A[m,k] @ B^T[k,n] + beta * C[m,n]`
    ///
    /// Half-precision variant of [`sgemm_a_bt`]. All matrices are f16.
    /// Uses f32 accumulation internally for numerical stability.
    pub fn hgemm_a_bt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f16,
        a: &CudaSlice<f16>,
        b: &CudaSlice<f16>,
        beta: f16,
        c: &mut CudaSlice<f16>,
    ) -> Result<()> {
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: k as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,
        };

        unsafe {
            self.cublas
                .gemm(cfg, b, a, c)
                .map_err(|e| LLMError::GpuError(format!("cuBLAS hgemm_a_bt failed: {e}")))?;
        }
        Ok(())
    }
}
