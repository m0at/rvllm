//! cuBLAS GEMM operations for linear algebra.

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm as _, GemmConfig, Gemv as _, GemvConfig};
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

    /// SGEMM: C[m,n] = A[m,k] @ B[n,k]^T
    ///
    /// A is activations in row-major [m, k].
    /// B is weights in PyTorch layout row-major [n, k].
    /// C is output row-major [m, n].
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
        // b row[n,k] = col[k,n], OP_T -> [n,k]. lda=k.
        // a row[m,k] = col[k,m], OP_N -> [k,m]. ldb=k.
        // C_col[n,m] = row C[m,n]. ldc=n.
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
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
                    },
                    b,
                    a,
                    c,
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
        // Same mapping as sgemm: C[m,n] = A[m,k] @ B[n,k]^T
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
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
                    },
                    b,
                    a,
                    c,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS hgemm failed: {e}")))?;
        }
        Ok(())
    }

    /// SGEMM (no transpose): C[m,n] = A[m,k] @ B[k,n]
    ///
    /// Both A and B are row-major. No transpose on either operand.
    /// Used for attention: probs[tokens, kv_len] @ V[kv_len, head_dim].
    pub fn sgemm_nn(
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
        // Row-major C[m,n] = A[m,k] @ B[k,n]
        // cuBLAS col-major: C_col[n,m] = B_col[n,k] @ A_col[k,m]
        // B row[k,n] = col[n,k], OP_N -> [n,k]. lda=n.
        // A row[m,k] = col[k,m], OP_N -> [k,m]. ldb=k.
        // C_col[n,m], ldc=n.
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
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
                    },
                    b,
                    a,
                    c,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS sgemm_nn failed: {e}")))?;
        }
        Ok(())
    }

    /// BGEMM: BF16 GEMM for bfloat16.
    ///
    /// Same layout conventions as [`sgemm`](Self::sgemm) but operates on bf16
    /// tensors. Uses CUDA_R_16BF data type with f32 accumulation.
    /// This matches HuggingFace/PyTorch BF16 inference numerics.
    pub fn bgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: half::bf16,
        a: &CudaSlice<half::bf16>,
        b: &CudaSlice<half::bf16>,
        beta: half::bf16,
        c: &mut CudaSlice<half::bf16>,
    ) -> Result<()> {
        // Same mapping as sgemm: C[m,n] = A[m,k] @ B[n,k]^T
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
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
                    },
                    b,
                    a,
                    c,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS bgemm failed: {e}")))?;
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
                    GemvConfig {
                        trans: cublasOperation_t::CUBLAS_OP_T,
                        m: n as i32,
                        n: m as i32,
                        alpha,
                        lda: n as i32,
                        incx: 1,
                        beta,
                        incy: 1,
                    },
                    a,
                    x,
                    y,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS sgemv failed: {e}")))?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    #[test]
    fn sgemm_a_times_bt() {
        let dev = CudaDevice::new(0).unwrap();
        let handle = CublasHandle::new(dev.clone()).unwrap();

        // A[2,3] row-major (activations)
        let a_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B[4,3] row-major (weights in PyTorch [out, in] layout)
        let b_host: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        let a_gpu = dev.htod_sync_copy(&a_host).unwrap();
        let b_gpu = dev.htod_sync_copy(&b_host).unwrap();
        let mut c_gpu = dev.alloc_zeros::<f32>(2 * 4).unwrap();

        // sgemm(m=2, n=4, k=3): C[2,4] = A[2,3] @ B[4,3]^T
        handle
            .sgemm(2, 4, 3, 1.0, &a_gpu, &b_gpu, 0.0, &mut c_gpu)
            .unwrap();

        let c_host = dev.dtoh_sync_copy(&c_gpu).unwrap();

        // CPU reference: C[i,j] = sum_k A[i,k] * B[j,k]
        let mut expected = vec![0.0f32; 8];
        for i in 0..2 {
            for j in 0..4 {
                for kk in 0..3 {
                    expected[i * 4 + j] += a_host[i * 3 + kk] * b_host[j * 3 + kk];
                }
            }
        }

        for (idx, (got, exp)) in c_host.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "mismatch at index {idx}: got {got}, expected {exp}"
            );
        }
    }
}
