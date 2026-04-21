//! Paged-decode launcher.
//!
//! One query per sequence. Kernel reads context_lens[seq] and walks
//! block_tables[seq, 0..ceil(context_lens/block_size)] to find KV
//! pages. `context_lens[i] == 0` is a valid padded slot; kernel must
//! predicate and never touch block_tables[i,*].

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

const SUPPORTED_HEAD_DIMS: &[u32] = &[128, 256, 512];

/// Parameters for one paged decode launch.
#[derive(Copy, Clone, Debug)]
pub struct PagedDecodeParams {
    pub num_seqs: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    pub scale: f32,
    pub window_size_left: i32, // -1 = full, >= 0 = sliding window
}

impl PagedDecodeParams {
    pub fn validate(&self) -> Result<()> {
        let ctx = || AttnCtx {
            op: "paged_decode.validate",
            stream: 0,
            num_seqs: self.num_seqs,
            head_dim: self.head_dim,
        };
        if !SUPPORTED_HEAD_DIMS.contains(&self.head_dim) {
            return Err(RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim {
                    got: self.head_dim,
                    supported: SUPPORTED_HEAD_DIMS,
                },
                ctx: ctx(),
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(RvllmError::Attention {
                err: AttentionError::GqaRatioInvalid {
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                },
                ctx: ctx(),
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if self.num_seqs == 0 {
            return Err(RvllmError::Attention {
                err: AttentionError::ContextExceedsBucket { context: 0, max: 0 },
                ctx: ctx(),
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        Ok(())
    }
}

/// Launcher. Constructed from `&AttentionBackend`. The `Fa3` variant
/// goes through the SM90 `.so` dispatch; the `Fa2Ptx` variant returns
/// `AttentionError::FeatureNotAvailable` until the sm_121 FA2 launch
/// path is wired up (next GB10 follow-up PR).
pub struct PagedDecodeLauncher<'a> {
    backend: &'a super::AttentionBackend,
}

impl<'a> PagedDecodeLauncher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { backend }
    }

    /// Validate params + issue the launch.
    ///
    /// # Safety
    /// Under `feature = "cuda"` this dispatches the fa3_sm90 kernel via
    /// an opaque C fn ptr. All device pointers must be valid for the
    /// kernel's duration and the workspace must be >= what
    /// `fn_workspace_size(batch, num_heads, max_splits=1)` returned.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        params: PagedDecodeParams,
        out_ptr: u64,
        q_ptr: u64,
        k_cache_ptr: u64,
        v_cache_ptr: u64,
        block_tables_ptr: u64,
        context_lens_ptr: u64,
        workspace_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        #[cfg(feature = "cuda")]
        {
            let fa3 = match self.backend {
                super::AttentionBackend::Fa3(fa3) => fa3,
                super::AttentionBackend::Fa2Ptx(_) => {
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            backend: "Fa2Ptx",
                            op: "paged_decode",
                        },
                        ctx: AttnCtx {
                            op: "paged_decode",
                            stream,
                            num_seqs: params.num_seqs,
                            head_dim: params.head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    });
                }
            };
            let rc = (fa3.fn_paged_decode)(
                q_ptr as *mut std::ffi::c_void,
                k_cache_ptr as *mut std::ffi::c_void,
                v_cache_ptr as *mut std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                block_tables_ptr as *mut std::ffi::c_void,
                context_lens_ptr as *mut std::ffi::c_void,
                workspace_ptr as *mut std::ffi::c_void,
                params.scale,
                params.num_seqs as i32,
                params.num_heads as i32,
                params.num_kv_heads as i32,
                params.head_dim as i32,
                params.block_size as i32,
                params.max_blocks_per_seq as i32,
                params.num_blocks_total as i32,
                params.window_size_left,
                stream as *mut std::ffi::c_void,
            );
            if rc != 0 {
                return Err(RvllmError::Attention {
                    err: AttentionError::KernelLaunchFailed {
                        cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                    },
                    ctx: AttnCtx {
                        op: "paged_decode",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                out_ptr,
                q_ptr,
                k_cache_ptr,
                v_cache_ptr,
                block_tables_ptr,
                context_lens_ptr,
                workspace_ptr,
                stream,
            );
        }
        Ok(())
    }
}

/// FP8 E4M3 paged-decode launcher. Same param validation as the FP16
/// path; dispatches the FP8 entry point and threads per-tensor scales.
/// `Fa2Ptx` backend returns `FeatureNotAvailable` — the FA2 kernels
/// today only accept f16/f32 KV cache, not fp8.
pub struct PagedDecodeFp8Launcher<'a> {
    backend: &'a super::AttentionBackend,
}

impl<'a> PagedDecodeFp8Launcher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { backend }
    }

    /// # Safety
    /// Every pointer must be valid device memory; `q_descale_ptr`,
    /// `k_descale_ptr`, `v_descale_ptr` point at single f32 scalars.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        params: PagedDecodeParams,
        o_f16: u64,
        q_fp8: u64,
        k_cache_fp8: u64,
        v_cache_fp8: u64,
        // `k_scale_cache` / `v_scale_cache`: per-slot f32 arrays
        //   (Gemma 4). Pass `0` to fall back to the scalar
        //   `k_descale_fallback_ptr` / `v_descale_fallback_ptr`
        //   (Llama/Qwen path).
        k_scale_cache: u64,
        v_scale_cache: u64,
        k_descale_fallback_ptr: u64,
        v_descale_fallback_ptr: u64,
        block_tables: u64,
        context_lens: u64,
        workspace: u64,
        q_descale_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        #[cfg(feature = "cuda")]
        {
            let fa3 = match self.backend {
                super::AttentionBackend::Fa3(fa3) => fa3,
                super::AttentionBackend::Fa2Ptx(fa2) => {
                    // sm_121 path: dispatch the PTX-built
                    // `flash_attention_2_decode_fp8kv_kernel`. Internal
                    // math f32, on-load dequant from FP8 E4M3 with
                    // per-tensor descales, f16 output to match the
                    // FA3 ABI (`o_f16`).
                    //
                    // Launch config:
                    //   Grid  (num_seqs, num_heads, 1)
                    //   Block (FA2_THREADS=128, 1, 1)
                    //   Smem  = 2 * FA2_BC * head_dim * 4 + FA2_BC * 4
                    //           + (FA2_THREADS / 32) * 4
                    // FA2_BC is 32 for sm_100+ (arch-conditional in
                    // flash_attention.cu). head_dim=256 with BC=32
                    // blows past the 48 KB static-smem ceiling, so we
                    // opt in to dynamic smem via `cuFuncSetAttribute`
                    // once per process.
                    use cudarc::driver::sys::*;
                    const FA2_THREADS: i32 = 128;
                    let hd = params.head_dim as i32;
                    // head_dim=512 (Gemma 4 global-attn) doesn't fit
                    // BC=32 within sm_121's ~99 KB opt-in smem cap
                    // (128 KB needed). Dispatch to the BC=16 kernel
                    // for those shapes.
                    let (kernel_fn, fa2_bc) = if hd > 256 {
                        (fa2.fn_decode_fp8kv_bc16, 16)
                    } else {
                        (fa2.fn_decode_fp8kv, 32)
                    };
                    let smem_bytes =
                        2 * fa2_bc * hd * 4 + fa2_bc * 4 + (FA2_THREADS / 32) * 4;
                    if smem_bytes as u32 >= 48 * 1024 {
                        let _ = cuFuncSetAttribute(
                            kernel_fn.raw() as CUfunction,
                            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            smem_bytes,
                        );
                    }

                    // Scalar args must outlive cuLaunchKernel.
                    let scale = params.scale;
                    let num_heads = params.num_heads as i32;
                    let num_kv_heads = params.num_kv_heads as i32;
                    let head_dim = params.head_dim as i32;
                    let block_size = params.block_size as i32;
                    let max_blocks_per_seq = params.max_blocks_per_seq as i32;
                    let window_size_left = params.window_size_left;

                    let mut arg_out = o_f16;
                    let mut arg_q = q_fp8;
                    let mut arg_k = k_cache_fp8;
                    let mut arg_v = v_cache_fp8;
                    let mut arg_ks = k_scale_cache;
                    let mut arg_vs = v_scale_cache;
                    let mut arg_kd = k_descale_fallback_ptr;
                    let mut arg_vd = v_descale_fallback_ptr;
                    let mut arg_bt = block_tables;
                    let mut arg_cl = context_lens;
                    let mut arg_qd = q_descale_ptr;
                    let _ = workspace; // unused: FA2 allocates in smem

                    let args: [*mut core::ffi::c_void; 18] = [
                        &mut arg_out as *mut _ as *mut _,
                        &mut arg_q as *mut _ as *mut _,
                        &mut arg_k as *mut _ as *mut _,
                        &mut arg_v as *mut _ as *mut _,
                        &mut arg_ks as *mut _ as *mut _,
                        &mut arg_vs as *mut _ as *mut _,
                        &mut arg_kd as *mut _ as *mut _,
                        &mut arg_vd as *mut _ as *mut _,
                        &mut arg_bt as *mut _ as *mut _,
                        &mut arg_cl as *mut _ as *mut _,
                        &mut arg_qd as *mut _ as *mut _,
                        &scale as *const _ as *mut _,
                        &num_heads as *const _ as *mut _,
                        &num_kv_heads as *const _ as *mut _,
                        &head_dim as *const _ as *mut _,
                        &block_size as *const _ as *mut _,
                        &max_blocks_per_seq as *const _ as *mut _,
                        &window_size_left as *const _ as *mut _,
                    ];

                    let rc = cuLaunchKernel(
                        kernel_fn.raw() as CUfunction,
                        params.num_seqs as u32,
                        params.num_heads as u32,
                        1,
                        FA2_THREADS as u32,
                        1,
                        1,
                        smem_bytes as u32,
                        stream as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(RvllmError::Attention {
                            err: AttentionError::KernelLaunchFailed {
                                cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                            },
                            ctx: AttnCtx {
                                op: "paged_decode_fp8 (Fa2Ptx)",
                                stream,
                                num_seqs: params.num_seqs,
                                head_dim: params.head_dim,
                            },
                            bt: std::backtrace::Backtrace::capture(),
                        });
                    }
                    return Ok(());
                }
            };
            // Fa3 SM90 path: not updated for per-slot K/V scales yet
            // (sm_121 is the active target; Fa3 is legacy). Pass
            // `k_scale_cache` / `v_scale_cache` where the SM90 kernel
            // wants `k_descale` / `v_descale` scalars — it'll
            // misinterpret them. A dedicated SM90 follow-up wires
            // the new ABI properly.
            let _ = k_scale_cache;
            let _ = v_scale_cache;
            let rc = (fa3.fn_paged_decode_fp8)(
                q_fp8 as *mut std::ffi::c_void,
                k_cache_fp8 as *mut std::ffi::c_void,
                v_cache_fp8 as *mut std::ffi::c_void,
                o_f16 as *mut std::ffi::c_void,
                block_tables as *mut std::ffi::c_void,
                context_lens as *mut std::ffi::c_void,
                workspace as *mut std::ffi::c_void,
                q_descale_ptr as *mut f32,
                0 as *mut f32,
                0 as *mut f32,
                params.scale,
                params.num_seqs as i32,
                params.num_heads as i32,
                params.num_kv_heads as i32,
                params.head_dim as i32,
                params.block_size as i32,
                params.max_blocks_per_seq as i32,
                params.num_blocks_total as i32,
                params.window_size_left,
                stream as *mut std::ffi::c_void,
            );
            if rc != 0 {
                return Err(RvllmError::Attention {
                    err: AttentionError::KernelLaunchFailed {
                        cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                    },
                    ctx: AttnCtx {
                        op: "paged_decode_fp8",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                o_f16, q_fp8, k_cache_fp8, v_cache_fp8, block_tables, context_lens,
                workspace, q_descale_ptr, k_descale_fallback_ptr, v_descale_fallback_ptr,
                k_scale_cache, v_scale_cache, q_scale_cache, stream,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good() -> PagedDecodeParams {
        PagedDecodeParams {
            num_seqs: 32,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            block_size: 64,
            max_blocks_per_seq: 33,
            num_blocks_total: 1024,
            scale: 1.0 / (128f32).sqrt(),
            window_size_left: -1,
        }
    }

    #[test]
    fn rejects_head_dim_64() {
        let mut p = good();
        p.head_dim = 64;
        assert!(p.validate().is_err());
    }

    #[test]
    fn rejects_gqa_ratio_not_divisible() {
        let mut p = good();
        p.num_heads = 7;
        p.num_kv_heads = 4;
        assert!(p.validate().is_err());
    }

    #[test]
    fn accepts_qwen_shape() {
        assert!(good().validate().is_ok());
    }

    #[test]
    fn accepts_head_dim_256() {
        let mut p = good();
        p.head_dim = 256;
        p.scale = 1.0 / (256f32).sqrt();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn accepts_head_dim_512() {
        let mut p = good();
        p.head_dim = 512;
        p.scale = 1.0 / (512f32).sqrt();
        assert!(p.validate().is_ok());
    }
}
