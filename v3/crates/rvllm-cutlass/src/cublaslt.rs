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
    kind: u8,
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
        self.fp8_gemm_inner(a_fp8, b_fp8, 0, 0, d_f16, m, n, k, a_scale, b_scale, stream, false, 0, None)
    }

    /// Plain FP8 E4M3 matmul with bf16 output: D_bf16 = A * B^T.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm_bf16(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        d_bf16: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(a_fp8, b_fp8, 0, 0, d_bf16, m, n, k, a_scale, b_scale, stream, false, 1, None)
    }

    /// Plain FP8 E4M3 matmul with f32 output: D_f32 = A * B^T.
    /// Guaranteed supported on all architectures.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm_f32(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        d_f32: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(a_fp8, b_fp8, 0, 0, d_f32, m, n, k, a_scale, b_scale, stream, false, 2, None)
    }

    /// FP8 matmul with f32 output and per-channel weight scales (OUTER_VEC_32F).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm_f32_channelscale(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        d_f32: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_channelscale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(a_fp8, b_fp8, 0, 0, d_f32, m, n, k, a_scale, 0, stream, false, 2, Some(b_channelscale))
    }

    /// F16 x F16 matmul with F32 output: D_f32 = A_f16 * B_f16^T.
    /// No FP8, no scale pointers. Used for lm_head where FP8 quantization
    /// destroys the weight distribution.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f16_gemm_f32(
        &self,
        a_f16: u64,
        b_f16: u64,
        d_f32: u64,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> Result<()> {
        let mut desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
        let rc = lt::cublasLtMatmulDescCreate(
            &mut desc,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType_t::CUDA_R_32F,
        );
        if rc != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("cublasLtMatmulDescCreate(f16)"));
        }
        let transa: i32 = 1;
        let transb: i32 = 0;
        set_attr(desc, lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &transa as *const _ as *const _, std::mem::size_of_val(&transa))?;
        set_attr(desc, lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &transb as *const _ as *const _, std::mem::size_of_val(&transb))?;

        let mut layout_a: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut layout_b: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut layout_d: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let r = lt::cublasLtMatrixLayoutCreate(&mut layout_a,
            lt::cudaDataType_t::CUDA_R_16F, k as u64, n as u64, k as i64);
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS { return Err(cublaslt_err("layout A(f16)")); }
        let r = lt::cublasLtMatrixLayoutCreate(&mut layout_b,
            lt::cudaDataType_t::CUDA_R_16F, k as u64, m as u64, k as i64);
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS { return Err(cublaslt_err("layout B(f16)")); }
        let r = lt::cublasLtMatrixLayoutCreate(&mut layout_d,
            lt::cudaDataType_t::CUDA_R_32F, n as u64, m as u64, n as i64);
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS { return Err(cublaslt_err("layout D(f16)")); }

        let key = AlgoKey { m, n, k, kind: 20 };
        let cached_algo = self.algo_cache.lock().ok().and_then(|c| c.get(&key).copied());
        let algo = if let Some(a) = cached_algo { a } else {
            let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
            lt::cublasLtMatmulPreferenceCreate(&mut pref);
            let ws = self.workspace_bytes;
            lt::cublasLtMatmulPreferenceSetAttribute(pref,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws as *const _ as *const _, std::mem::size_of::<usize>());
            let mut heur: [lt::cublasLtMatmulHeuristicResult_t; 1] = std::mem::zeroed();
            let mut ret: i32 = 0;
            let r = lt::cublasLtMatmulAlgoGetHeuristic(
                self.handle, desc, layout_a, layout_b, layout_d, layout_d,
                pref, 1, heur.as_mut_ptr(), &mut ret);
            lt::cublasLtMatmulPreferenceDestroy(pref);
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
                return Err(cublaslt_err("heuristic(f16)"));
            }
            let best = heur[0].algo;
            if let Ok(mut c) = self.algo_cache.lock() { c.insert(key, best); }
            best
        };

        let one: f32 = 1.0;
        let zero: f32 = 0.0;
        let r = lt::cublasLtMatmul(
            self.handle, desc,
            &one as *const _ as *const _,
            b_f16 as *const _, layout_a,
            a_f16 as *const _, layout_b,
            &zero as *const _ as *const _,
            d_f32 as *const _, layout_d,
            d_f32 as *mut _, layout_d,
            &algo, self.workspace as *mut _, self.workspace_bytes, stream as _,
        );
        lt::cublasLtMatrixLayoutDestroy(layout_d);
        lt::cublasLtMatrixLayoutDestroy(layout_b);
        lt::cublasLtMatrixLayoutDestroy(layout_a);
        lt::cublasLtMatmulDescDestroy(desc);
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("cublasLtMatmul(f16)"));
        }
        Ok(())
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
            a_fp8, b_fp8, bias_f16, 0, d_f16, m, n, k, a_scale, b_scale, stream, false, 0, None,
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
            a_fp8, b_fp8, 0, residual_f16, d_f16, m, n, k, a_scale, b_scale, stream, true, 0, None,
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
        d_out_type: u8, // 0=f16, 1=bf16, 2=f32
        b_channelscale: Option<u64>, // per-channel weight scale (OUTER_VEC_32F)
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

        // TN swap: cuBLAS A = our weight (b_fp8), cuBLAS B = our activation (a_fp8).
        // Scale pointers must match: A_SCALE = weight scale, B_SCALE = activation scale.
        let cublas_a_scale = if let Some(cs) = b_channelscale { cs } else { b_scale };
        let cublas_b_scale = a_scale;
        set_attr(
            desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &cublas_a_scale as *const _ as *const _,
            std::mem::size_of_val(&cublas_a_scale),
        )?;
        set_attr(
            desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &cublas_b_scale as *const _ as *const _,
            std::mem::size_of_val(&cublas_b_scale),
        )?;

        if b_channelscale.is_some() {
            let scale_mode: u32 = 3; // OUTER_VEC_32F
            let attr_a_scale_mode: u32 = 31; // CUBLASLT_MATMUL_DESC_A_SCALE_MODE
            set_attr(
                desc,
                unsafe { std::mem::transmute::<u32, lt::cublasLtMatmulDescAttributes_t>(attr_a_scale_mode) },
                &scale_mode as *const _ as *const _,
                std::mem::size_of_val(&scale_mode),
            )?;
        }


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
        let d_type = match d_out_type {
            1 => lt::cudaDataType_t::CUDA_R_16BF,
            2 => lt::cudaDataType_t::CUDA_R_32F,
            _ => lt::cudaDataType_t::CUDA_R_16F,
        };
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut layout_d,
            d_type,
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
            kind: match (bias_f16 != 0, beta_one, d_out_type, b_channelscale.is_some()) {
                (true, _, _, true) => 31u8,
                (true, _, _, false) => 1u8,
                (_, true, _, true) => 32 + d_out_type,
                (_, true, _, false) => 2 + d_out_type,
                (_, false, _, true) => 40 + d_out_type,
                (_, false, _, false) => 10 + d_out_type,
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
            let mut heur: [lt::cublasLtMatmulHeuristicResult_t; 1] =
                std::mem::zeroed();
            let mut ret: i32 = 0;
            let r = lt::cublasLtMatmulAlgoGetHeuristic(
                self.handle, desc, layout_a, layout_b, layout_d, layout_d,
                pref, 1, heur.as_mut_ptr(), &mut ret,
            );
            lt::cublasLtMatmulPreferenceDestroy(pref);
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
                return Err(cublaslt_err("heuristic"));
            }
            let best_algo = heur[0].algo;
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

        // One-shot diagnostic: compare cuBLASLt D[0][0] against manual FP8 dot product.
        // Fires once per (M,N,K,kind) shape, controlled by env RVLLM_GEMM_DIAG=1.
        if !beta_one && bias_f16 == 0 {
            use std::sync::atomic::{AtomicBool, Ordering};
            static DIAG_DONE: AtomicBool = AtomicBool::new(false);
            if std::env::var("RVLLM_GEMM_DIAG").map_or(false, |v| v == "1")
                && !DIAG_DONE.swap(true, Ordering::Relaxed)
            {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                // Read scales
                let mut sa = [0.0f32; 1];
                let mut sb = [0.0f32; 1];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    sa.as_mut_ptr() as *mut _, a_scale, 4);
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    sb.as_mut_ptr() as *mut _, b_scale, 4);
                // Read first K bytes of cuBLAS-A (our weight row 0) and cuBLAS-B (our act row 0)
                let kk = k as usize;
                let mut wa = vec![0u8; kk];
                let mut ab = vec![0u8; kk];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    wa.as_mut_ptr() as *mut _, b_fp8, kk);
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    ab.as_mut_ptr() as *mut _, a_fp8, kk);
                // Manual FP8 dot product for D[0][0]:
                // cuBLAS computes D = A_SCALE * B_SCALE * op(A)^T * op(B)
                // D[0][0] = sa * sb * sum_c( fp8(weight[0][c]) * fp8(act[0][c]) )
                let mut raw_dot: f64 = 0.0;
                for c in 0..kk {
                    let wv = fp8_e4m3_to_f32_inline(wa[c]) as f64;
                    let av = fp8_e4m3_to_f32_inline(ab[c]) as f64;
                    raw_dot += wv * av;
                }
                let expected = sa[0] as f64 * sb[0] as f64 * raw_dot;
                // Read cuBLASLt result D[0][0]
                let cublas_val: f64 = match d_out_type {
                    2 => {
                        let mut v = [0.0f32; 1];
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            v.as_mut_ptr() as *mut _, d_f16, 4);
                        v[0] as f64
                    }
                    1 => {
                        let mut v = [0u16; 1];
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            v.as_mut_ptr() as *mut _, d_f16, 2);
                        let bits = v[0] as u32;
                        let sign = (bits >> 15) & 1;
                        let exp = (bits >> 7) & 0xff;
                        let mant = bits & 0x7f;
                        let f = if exp == 0 { (mant as f32) * (1.0 / 16384.0 / 128.0) }
                                else { f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 16)) };
                        if sign == 1 { -f.abs() as f64 } else { f as f64 }
                    }
                    _ => {
                        let mut v = [0u16; 1];
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            v.as_mut_ptr() as *mut _, d_f16, 2);
                        let bits = v[0] as u32;
                        let sign = (bits >> 15) & 1;
                        let exp = (bits >> 10) & 0x1f;
                        let mant = bits & 0x3ff;
                        let f = if exp == 0 { (mant as f32) * (1.0 / 16384.0 / 1024.0) }
                                else { f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13)) };
                        if sign == 1 { -f.abs() as f64 } else { f as f64 }
                    }
                };
                let ratio = if expected.abs() > 1e-20 { cublas_val / expected } else { f64::NAN };
                eprintln!("[GEMM_DIAG] M={} N={} K={} d_out_type={}", m, n, k, d_out_type);
                eprintln!("[GEMM_DIAG] a_scale(act)={:.6e} b_scale(wt)={:.6e}", sa[0], sb[0]);
                eprintln!("[GEMM_DIAG] weight_fp8[0..8]={:?} act_fp8[0..8]={:?}",
                    &wa[..8.min(kk)], &ab[..8.min(kk)]);
                eprintln!("[GEMM_DIAG] raw_dot(f64)={:.6e} expected(sa*sb*dot)={:.6e}", raw_dot, expected);
                eprintln!("[GEMM_DIAG] cuBLASLt_D[0][0]={:.6e} ratio(cublas/expected)={:.6}", cublas_val, ratio);
                if (ratio - 1.0).abs() > 0.05 {
                    eprintln!("[GEMM_DIAG] *** MISMATCH *** ratio={:.6} -- cuBLASLt config is wrong!", ratio);
                } else {
                    eprintln!("[GEMM_DIAG] PASS -- cuBLASLt matches manual computation");
                    eprintln!("[GEMM_DIAG] Bug is upstream of fp8_gemm (wrong bytes or wrong scales)");
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn fp8_e4m3_to_f32_inline(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = (b >> 3) & 0xF;
    let m = b & 0x7;
    let val = if e == 0 {
        if m == 0 { 0.0f32 } else { (m as f32) * (1.0 / 512.0) }
    } else if e == 15 && m == 7 {
        return f32::NAN;
    } else {
        f32::from_bits(((e as u32 + 120) << 23) | ((m as u32) << 20))
    };
    if s != 0 { -val } else { val }
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
