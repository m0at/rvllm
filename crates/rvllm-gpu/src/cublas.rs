//! cuBLAS GEMM operations for linear algebra.

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm as _, GemmConfig, Gemv as _, GemvConfig};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
use std::sync::Arc;

use crate::Result;

/// Default cuBLAS workspace size for graph capture (4 MiB).
/// NVIDIA recommends at least 4 KiB; 4 MiB covers all GEMM tile configs.
const CUBLAS_GRAPH_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;

/// Wrapper around cuBLAS for matrix operations.
pub struct CublasHandle {
    blas: CudaBlas,
    stream: Arc<CudaStream>,
    /// Pre-allocated workspace buffer for CUDA graph capture.
    /// cuBLAS requires an explicit workspace via `cublasSetWorkspace_v2`
    /// before any GEMM call inside a graph capture region, otherwise it
    /// tries to allocate internally with `cudaMalloc` which is forbidden.
    graph_workspace: Option<CudaSlice<u8>>,
}

impl CublasHandle {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS init failed: {e}")))?;
        Ok(Self { blas, stream, graph_workspace: None })
    }

    /// Returns a reference to the underlying stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Pre-allocate and register a cuBLAS workspace for CUDA graph capture.
    ///
    /// Must be called BEFORE `cuStreamBeginCapture`. The workspace stays
    /// registered for the lifetime of this handle; subsequent captures
    /// reuse the same buffer.
    #[cfg(feature = "cuda")]
    pub fn prepare_for_graph_capture(&mut self) -> Result<()> {
        if self.graph_workspace.is_some() {
            return Ok(()); // already prepared
        }

        tracing::info!(
            bytes = CUBLAS_GRAPH_WORKSPACE_BYTES,
            "allocating cuBLAS graph workspace"
        );

        let mut ws = self.stream
            .alloc_zeros::<u8>(CUBLAS_GRAPH_WORKSPACE_BYTES)
            .map_err(|e| crate::LLMError::GpuError(
                format!("cuBLAS workspace alloc: {e}")
            ))?;

        // Get raw device pointer and call cublasSetWorkspace_v2 in a scoped
        // borrow so we can move `ws` into self.graph_workspace afterwards.
        {
            let (raw_ptr, _guard) = DevicePtrMut::device_ptr_mut(&mut ws, &self.stream);
            unsafe {
                let status = cudarc::cublas::sys::cublasSetWorkspace_v2(
                    *self.blas.handle(),
                    raw_ptr as *mut std::ffi::c_void,
                    CUBLAS_GRAPH_WORKSPACE_BYTES,
                );
                if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(crate::LLMError::GpuError(
                        format!("cublasSetWorkspace_v2 failed: {status:?}")
                    ));
                }
            }
        }

        self.graph_workspace = Some(ws);
        tracing::info!("cuBLAS graph workspace registered");
        Ok(())
    }

    /// No-op when cuda feature is off.
    #[cfg(not(feature = "cuda"))]
    pub fn prepare_for_graph_capture(&mut self) -> Result<()> {
        Ok(())
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
    use cudarc::driver::CudaContext;

    #[test]
    fn sgemm_a_times_bt() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let handle = CublasHandle::new(stream.clone()).unwrap();

        // A[2,3] row-major (activations)
        let a_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B[4,3] row-major (weights in PyTorch [out, in] layout)
        let b_host: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        let a_gpu = stream.clone_htod(&a_host).unwrap();
        let b_gpu = stream.clone_htod(&b_host).unwrap();
        let mut c_gpu = stream.alloc_zeros::<f32>(2 * 4).unwrap();

        // sgemm(m=2, n=4, k=3): C[2,4] = A[2,3] @ B[4,3]^T
        handle
            .sgemm(2, 4, 3, 1.0, &a_gpu, &b_gpu, 0.0, &mut c_gpu)
            .unwrap();

        let c_host = stream.clone_dtoh(&c_gpu).unwrap();

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
