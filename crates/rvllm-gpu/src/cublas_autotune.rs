//! cublasLt algorithm autotuning.
//!
//! At model load time, benchmark all candidate cublasLt algorithms for each
//! GEMM shape and cache the fastest one. This is what vLLM/torch.compile does
//! under the hood.

use crate::cublaslt_raw as lt_sys;
use cudarc::driver::sys as cu_sys;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use std::collections::HashMap;
use std::ffi::c_void;

use crate::cublaslt_ops::CublasLtOps;
use crate::{LLMError, Result};

const MAX_ALGOS: usize = 32;
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmDtype {
    F16,
    Fp8E4M3,
}

#[derive(Debug, Clone)]
pub struct AutotunedAlgo {
    pub algo: lt_sys::cublasLtMatmulAlgo_t,
    pub workspace_size: usize,
    pub time_us: f64,
}

pub struct CublasAutotuner {
    results: HashMap<(usize, usize, usize), AutotunedAlgo>,
}

/// RAII guard for cublasLt matmul descriptor.
struct MatmulDesc(lt_sys::cublasLtMatmulDesc_t);
impl Drop for MatmulDesc {
    fn drop(&mut self) {
        unsafe { lt_sys::cublasLtMatmulDescDestroy(self.0); }
    }
}

/// RAII guard for cublasLt matrix layout.
struct MatLayout(lt_sys::cublasLtMatrixLayout_t);
impl Drop for MatLayout {
    fn drop(&mut self) {
        unsafe { lt_sys::cublasLtMatrixLayoutDestroy(self.0); }
    }
}

/// RAII guard for cublasLt matmul preference.
struct MatmulPref(lt_sys::cublasLtMatmulPreference_t);
impl Drop for MatmulPref {
    fn drop(&mut self) {
        unsafe { lt_sys::cublasLtMatmulPreferenceDestroy(self.0); }
    }
}

/// RAII guard for a CUDA event.
struct CuEvent(cu_sys::CUevent);
impl Drop for CuEvent {
    fn drop(&mut self) {
        unsafe { cu_sys::cuEventDestroy_v2(self.0); }
    }
}

impl CuEvent {
    fn new() -> Result<Self> {
        let mut ev: cu_sys::CUevent = std::ptr::null_mut();
        let r = unsafe { cu_sys::cuEventCreate(&mut ev, 0) };
        if r != cu_sys::CUresult::CUDA_SUCCESS {
            return Err(LLMError::GpuError(format!("cuEventCreate: {r:?}")));
        }
        Ok(Self(ev))
    }

    fn record(&self, stream: cu_sys::CUstream) -> Result<()> {
        let r = unsafe { cu_sys::cuEventRecord(self.0, stream) };
        if r != cu_sys::CUresult::CUDA_SUCCESS {
            return Err(LLMError::GpuError(format!("cuEventRecord: {r:?}")));
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        let r = unsafe { cu_sys::cuEventSynchronize(self.0) };
        if r != cu_sys::CUresult::CUDA_SUCCESS {
            return Err(LLMError::GpuError(format!("cuEventSynchronize: {r:?}")));
        }
        Ok(())
    }

    fn elapsed_ms(&self, start: &CuEvent) -> Result<f32> {
        let mut ms: f32 = 0.0;
        let r = unsafe { cu_sys::cuEventElapsedTime(&mut ms, start.0, self.0) };
        if r != cu_sys::CUresult::CUDA_SUCCESS {
            return Err(LLMError::GpuError(format!("cuEventElapsedTime: {r:?}")));
        }
        Ok(ms)
    }
}

fn check_lt(s: lt_sys::cublasStatus_t, ctx: &str) -> Result<()> {
    if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(LLMError::GpuError(format!("{ctx}: {s:?}")));
    }
    Ok(())
}

impl CublasAutotuner {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Benchmark all cublasLt algorithms for a given (m, n, k) f16 GEMM shape.
    /// Layout: row-major C[m,n] = A[m,k] @ B[n,k]^T (same as hgemm_a_bt).
    pub fn autotune_shape(
        lt_ops: &CublasLtOps,
        m: usize,
        n: usize,
        k: usize,
        dtype: GemmDtype,
    ) -> Result<AutotunedAlgo> {
        let stream = lt_ops.stream();

        // Allocate scratch buffers (input element size depends on dtype)
        let in_bytes = match dtype {
            GemmDtype::F16 => 2,
            GemmDtype::Fp8E4M3 => 1,
        };
        let a_buf = unsafe { stream.alloc::<u8>(m * k * in_bytes) }
            .map_err(|e| LLMError::GpuError(format!("autotune alloc A: {e}")))?;
        let b_buf = unsafe { stream.alloc::<u8>(n * k * in_bytes) }
            .map_err(|e| LLMError::GpuError(format!("autotune alloc B: {e}")))?;
        let mut c_buf = unsafe { stream.alloc::<u8>(m * n * 2) } // output always f16
            .map_err(|e| LLMError::GpuError(format!("autotune alloc C: {e}")))?;

        // Workspace (reuse the cublasLt internal workspace size: 32MB Hopper, 4MB else)
        let ws_size: usize = 4 * 1024 * 1024; // 4 MiB fallback
        let ws_buf = unsafe { stream.alloc::<u8>(ws_size) }
            .map_err(|e| LLMError::GpuError(format!("autotune alloc ws: {e}")))?;

        let (a_ptr, _ra) = a_buf.device_ptr(stream);
        let (b_ptr, _rb) = b_buf.device_ptr(stream);
        let (c_ptr, _rc) = c_buf.device_ptr_mut(stream);
        let (ws_ptr, _rw) = ws_buf.device_ptr(stream);
        let cu_stream = stream.cu_stream();

        let (a_type, b_type, c_type, compute_type) = match dtype {
            GemmDtype::F16 => (
                lt_sys::cudaDataType_t::CUDA_R_16F,
                lt_sys::cudaDataType_t::CUDA_R_16F,
                lt_sys::cudaDataType_t::CUDA_R_16F,
                lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            ),
            GemmDtype::Fp8E4M3 => (
                lt_sys::cudaDataType_t::CUDA_R_8F_E4M3,
                lt_sys::cudaDataType_t::CUDA_R_8F_E4M3,
                lt_sys::cudaDataType_t::CUDA_R_16F,
                lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            ),
        };

        let scale_type = lt_sys::cudaDataType_t::CUDA_R_32F;

        unsafe {
            // Create matmul descriptor
            let mut desc: lt_sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
            check_lt(
                lt_sys::cublasLtMatmulDescCreate(&mut desc, compute_type, scale_type),
                "autotune desc create",
            )?;
            let desc = MatmulDesc(desc);

            // Row-major: C[m,n] = A[m,k] @ B[n,k]^T
            // cublasLt col-major: C_col[n,m] = B_col[k,n]^T @ A_col[k,m]
            let trans_a = lt_sys::cublasOperation_t::CUBLAS_OP_T;
            let trans_b = lt_sys::cublasOperation_t::CUBLAS_OP_N;
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc.0,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &trans_a as *const _ as *const c_void,
                std::mem::size_of_val(&trans_a),
            );
            lt_sys::cublasLtMatmulDescSetAttribute(
                desc.0,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &trans_b as *const _ as *const c_void,
                std::mem::size_of_val(&trans_b),
            );

            // Matrix layouts: A_col = weight[k,n], B_col = input[k,m], C_col = output[n,m]
            let mut la: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut lb: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut lc: lt_sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
            check_lt(
                lt_sys::cublasLtMatrixLayoutCreate(&mut la, a_type, k as u64, n as u64, k as i64),
                "layout A",
            )?;
            let la = MatLayout(la);
            check_lt(
                lt_sys::cublasLtMatrixLayoutCreate(&mut lb, b_type, k as u64, m as u64, k as i64),
                "layout B",
            )?;
            let lb = MatLayout(lb);
            check_lt(
                lt_sys::cublasLtMatrixLayoutCreate(&mut lc, c_type, n as u64, m as u64, n as i64),
                "layout C",
            )?;
            let lc = MatLayout(lc);

            // Preference with workspace
            let mut pref: lt_sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
            check_lt(
                lt_sys::cublasLtMatmulPreferenceCreate(&mut pref),
                "pref create",
            )?;
            let pref = MatmulPref(pref);
            lt_sys::cublasLtMatmulPreferenceSetAttribute(
                pref.0,
                lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const _ as *const c_void,
                std::mem::size_of_val(&ws_size),
            );

            // Get candidate algorithms
            let mut heuristics = vec![std::mem::zeroed::<lt_sys::cublasLtMatmulHeuristicResult_t>(); MAX_ALGOS];
            let mut returned: i32 = 0;

            let handle = *lt_ops.handle();

            let s = lt_sys::cublasLtMatmulAlgoGetHeuristic(
                handle,
                desc.0,
                la.0,
                lb.0,
                lc.0,
                lc.0,
                pref.0,
                MAX_ALGOS as i32,
                heuristics.as_mut_ptr(),
                &mut returned,
            );
            if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || returned == 0 {
                return Err(LLMError::GpuError(format!(
                    "autotune no algos for ({m},{n},{k}): {s:?} returned={returned}"
                )));
            }

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            let alpha_ptr = &alpha as *const f32 as *const c_void;
            let beta_ptr = &beta as *const f32 as *const c_void;

            let mut best: Option<AutotunedAlgo> = None;

            for i in 0..returned as usize {
                let h = &heuristics[i];
                if h.state != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    continue;
                }
                if h.workspaceSize > ws_size {
                    continue;
                }

                // Warmup
                let mut ok = true;
                for _ in 0..WARMUP_ITERS {
                    let s = lt_sys::cublasLtMatmul(
                        handle,
                        desc.0,
                        alpha_ptr,
                        b_ptr as *const c_void,
                        la.0,
                        a_ptr as *const c_void,
                        lb.0,
                        beta_ptr,
                        c_ptr as *mut c_void,
                        lc.0,
                        c_ptr as *mut c_void,
                        lc.0,
                        &h.algo,
                        ws_ptr as *mut c_void,
                        ws_size,
                        lt_sys::cu_stream_to_cuda_stream(cu_stream),
                    );
                    if s != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }

                // Benchmark
                let start_ev = CuEvent::new()?;
                let stop_ev = CuEvent::new()?;

                start_ev.record(cu_stream)?;
                for _ in 0..BENCH_ITERS {
                    lt_sys::cublasLtMatmul(
                        handle,
                        desc.0,
                        alpha_ptr,
                        b_ptr as *const c_void,
                        la.0,
                        a_ptr as *const c_void,
                        lb.0,
                        beta_ptr,
                        c_ptr as *mut c_void,
                        lc.0,
                        c_ptr as *mut c_void,
                        lc.0,
                        &h.algo,
                        ws_ptr as *mut c_void,
                        ws_size,
                        lt_sys::cu_stream_to_cuda_stream(cu_stream),
                    );
                }
                stop_ev.record(cu_stream)?;
                stop_ev.sync()?;

                let total_ms = stop_ev.elapsed_ms(&start_ev)?;
                let avg_us = (total_ms as f64 / BENCH_ITERS as f64) * 1000.0;

                let dominated = best.as_ref().map_or(false, |b| avg_us >= b.time_us);
                if !dominated {
                    best = Some(AutotunedAlgo {
                        algo: h.algo,
                        workspace_size: h.workspaceSize,
                        time_us: avg_us,
                    });
                }
            }

            best.ok_or_else(|| {
                LLMError::GpuError(format!("autotune: all {returned} algos failed for ({m},{n},{k})"))
            })
        }
    }

    /// Autotune all GEMM shapes for the model across decode + prefill batch sizes.
    /// Covers M=1,2,4,8,16,32,64,128 x all projection shapes.
    pub fn autotune_model(
        lt_ops: &CublasLtOps,
        dtype: GemmDtype,
        hidden: usize,
        q_dim: usize,
        qkv_dim: usize,
        intermediate: usize,
        gate_up_dim: usize,
    ) -> Result<Self> {
        let m_values: &[usize] = &[1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128];
        let nk_shapes: &[(usize, usize)] = &[
            (qkv_dim, hidden),        // QKV projection
            (hidden, q_dim),           // O-proj
            (gate_up_dim, hidden),     // GateUp
            (hidden, intermediate),    // Down
        ];

        let mut shapes = Vec::new();
        for &m in m_values {
            for &(n, k) in nk_shapes {
                shapes.push((m, n, k));
            }
        }

        let mut tuner = Self::new();
        tracing::info!(
            num_shapes = shapes.len(),
            m_values = ?m_values,
            "autotuning cublasLt algos"
        );
        for &(m, n, k) in &shapes {
            tracing::info!(m, n, k, ?dtype, "autotune start");
            match Self::autotune_shape(lt_ops, m, n, k, dtype) {
                Ok(result) => {
                    tracing::info!(
                        m, n, k,
                        time_us = result.time_us,
                        ws = result.workspace_size,
                        "autotune best algo"
                    );
                    tuner.results.insert((m, n, k), result);
                }
                Err(e) => {
                    tracing::warn!(m, n, k, %e, "autotune failed");
                }
            }
        }
        Ok(tuner)
    }

    /// Get the best algo for a shape (returns None if not autotuned).
    pub fn get(&self, m: usize, n: usize, k: usize) -> Option<&AutotunedAlgo> {
        self.results.get(&(m, n, k))
    }

    /// Number of shapes autotuned.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Iterate over all autotuned (m, n, k) -> algo mappings.
    pub fn iter(&self) -> impl Iterator<Item = (&(usize, usize, usize), &AutotunedAlgo)> {
        self.results.iter()
    }

    /// Maximum workspace size across all autotuned algorithms.
    pub fn max_workspace_size(&self) -> usize {
        self.results.values().map(|a| a.workspace_size).max().unwrap_or(0)
    }
}
