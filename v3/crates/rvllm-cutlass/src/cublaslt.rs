//! cuBLASLt FP8 GEMM with bias epilogue.
//!
//! One call per QKV: the matmul multiplies FP8 E4M3 A*B, adds an f16
//! row-broadcast bias, and writes an f16 D. One launch replaces the
//! (cutlass_fp8_gemm → add_bias_f16) pair — 1 kernel per layer x 28
//! layers = 28 fewer kernels per decode step.

#[cfg(feature = "cuda")]
use cudarc::cublaslt::sys as lt;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

/// Handle to a cuBLASLt library instance. Created once at engine init.
pub struct CublasLt {
    #[cfg(feature = "cuda")]
    handle: lt::cublasLtHandle_t,
    // Workspace for cuBLASLt (separate from CUTLASS workspace).
    workspace: u64,
    workspace_bytes: usize,
}

// cuBLASLt handles are not Send/Sync by default across threads.
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
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(workspace: u64, workspace_bytes: usize) -> Result<Self> {
        Ok(Self {
            workspace,
            workspace_bytes,
        })
    }

    /// FP8 E4M3 matmul with per-tensor scales and f16 row-broadcast bias.
    ///
    /// Layout convention (matches cuBLASLt FP8 requirement on H100):
    /// - A is `[M, K]` row-major FP8 (the "activation" hidden_fp8)
    /// - B is `[N, K]` row-major FP8 (the "weight" qkv_fp8), which
    ///   cuBLASLt treats as B^T with transa=N, transb=T.
    /// - D is `[M, N]` row-major f16 (output)
    /// - bias is `[N]` f16 (broadcast across M rows)
    /// - a_scale / b_scale are per-tensor f32 scalar device pointers
    ///
    /// # Safety
    /// Every pointer must point at valid device memory for the call.
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
        // Descriptors.
        let mut desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
        let rc = lt::cublasLtMatmulDescCreate(
            &mut desc,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType_t::CUDA_R_32F,
        );
        if rc != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("cublasLtMatmulDescCreate"));
        }

        // Set transa = N (no transpose on A), transb = T (transpose on B)
        // which is required for FP8 TN layout on H100.
        let transa = lt::cublasOperation_t::CUBLAS_OP_N;
        let transb = lt::cublasOperation_t::CUBLAS_OP_T;
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

        // Epilogue = bias.
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
        // FP8 requires A and B scale pointers on the descriptor.
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

        // Matrix layouts. A = [M, K] FP8, B = [N, K] FP8 (cuBLASLt
        // interprets via transb=T so logical op is A * B^T = [M, N]).
        let mut layout_a: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut layout_b: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut layout_d: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut layout_a,
            lt::cudaDataType_t::CUDA_R_8F_E4M3,
            m as u64,
            k as u64,
            k as i64, // row-major, ld = K
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout A"));
        }
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut layout_b,
            lt::cudaDataType_t::CUDA_R_8F_E4M3,
            n as u64,
            k as u64,
            k as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout B"));
        }
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut layout_d,
            lt::cudaDataType_t::CUDA_R_16F,
            m as u64,
            n as u64,
            n as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout D"));
        }

        // Heuristic.
        let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
        let r = lt::cublasLtMatmulPreferenceCreate(&mut pref);
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("preference create"));
        }
        set_attr(
            pref as *mut _,
            // CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 0
            std::mem::transmute::<u32, lt::cublasLtMatmulDescAttributes_t>(0),
            &self.workspace_bytes as *const _ as *const _,
            std::mem::size_of::<usize>(),
        )
        .ok();
        let mut heur: lt::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
        let mut ret: i32 = 0;
        let r = lt::cublasLtMatmulAlgoGetHeuristic(
            self.handle,
            desc,
            layout_a,
            layout_b,
            layout_d,
            layout_d,
            pref,
            1,
            &mut heur,
            &mut ret,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
            return Err(cublaslt_err("heuristic"));
        }

        let one_f32: f32 = 1.0;
        let zero_f32: f32 = 0.0;
        let r = lt::cublasLtMatmul(
            self.handle,
            desc,
            &one_f32 as *const _ as *const _,
            a_fp8 as *const _,
            layout_a,
            b_fp8 as *const _,
            layout_b,
            &zero_f32 as *const _ as *const _,
            std::ptr::null(), // C layout unused when beta=0
            std::ptr::null_mut(),
            d_f16 as *mut _,
            layout_d,
            &heur.algo,
            self.workspace as *mut _,
            self.workspace_bytes,
            stream as _,
        );

        lt::cublasLtMatmulPreferenceDestroy(pref);
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
