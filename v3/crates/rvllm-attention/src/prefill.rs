//! Paged-prefill launcher. Same .so as decode; different entry point.
//!
//! Prefill runs on `num_tokens` query tokens (not one-per-seq). The
//! kernel uses `cu_seqlens_q` / `cu_seqlens_k` to find each request's
//! span in the concatenated tensor.

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

const SUPPORTED_HEAD_DIMS: &[u32] = &[128, 256, 512];

#[derive(Copy, Clone, Debug)]
pub struct PagedPrefillParams {
    pub num_tokens: u32,
    pub num_seqs: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    pub scale: f32,
    pub window_size_left: i32,
}

impl PagedPrefillParams {
    pub fn validate(&self) -> Result<()> {
        let ctx = || AttnCtx {
            op: "paged_prefill.validate",
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
        Ok(())
    }
}

pub struct PagedPrefillLauncher<'a> {
    _backend: &'a super::AttentionBackend,
}

impl<'a> PagedPrefillLauncher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { _backend: backend }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        params: PagedPrefillParams,
        _out_ptr: u64,
        _q_ptr: u64,
        _k_cache_ptr: u64,
        _v_cache_ptr: u64,
        _block_tables_ptr: u64,
        _context_lens_ptr: u64,
        _cu_seqlens_q_ptr: u64,
        _cu_seqlens_k_ptr: u64,
        _workspace_ptr: u64,
        _stream: u64,
    ) -> Result<()> {
        params.validate()?;
        Ok(())
    }
}

/// FP8 E4M3 paged-prefill launcher. Q / K / V are FP8 with per-tensor
/// descales. Multi-query self-attention with a per-seq causal mask.
/// `Fa2Ptx` backend returns `FeatureNotAvailable` — see decode.rs.
pub struct PagedPrefillFp8Launcher<'a> {
    backend: &'a super::AttentionBackend,
}

impl<'a> PagedPrefillFp8Launcher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { backend }
    }

    /// # Safety
    /// Caller owns all device pointers. `cu_seqlens_q` is a
    /// [batch+1]-len i32 prefix-sum device buffer; `max_seqlen_q` is the
    /// longest per-seq Q length; `total_q` is the sum (= Q tensor's
    /// leading dim).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        params: PagedPrefillParams,
        o_f16: u64,
        q_fp8: u64,
        k_cache_fp8: u64,
        v_cache_fp8: u64,
        block_tables: u64,
        context_lens: u64,
        cu_seqlens_q: u64,
        workspace: u64,
        q_descale_ptr: u64,
        k_descale_ptr: u64,
        v_descale_ptr: u64,
        max_seqlen_q: u32,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        #[cfg(feature = "cuda")]
        {
            let fa3 = match self.backend {
                super::AttentionBackend::Fa3(fa3) => fa3,
                super::AttentionBackend::Fa2Ptx(fa2) => {
                    // sm_121 path: dispatch
                    // `flash_attention_2_prefill_fp8kv{_bc16}_kernel`
                    // directly. Same ABI story as the decode Fa2Ptx
                    // arm: f32 math, FP8→f32 on-load dequant with
                    // per-tensor descales, f16 output, opt-in dynamic
                    // smem for the large BC×head_dim tile.
                    use cudarc::driver::sys::*;
                    const FA2_THREADS: i32 = 128;
                    let hd = params.head_dim as i32;
                    let (kernel_fn, fa2_bc) = if hd > 256 {
                        (fa2.fn_prefill_fp8kv_bc16, 16)
                    } else {
                        (fa2.fn_prefill_fp8kv, 32)
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
                    let mut arg_bt = block_tables;
                    let mut arg_cl = context_lens;
                    let mut arg_cu = cu_seqlens_q;
                    let mut arg_qd = q_descale_ptr;
                    let mut arg_kd = k_descale_ptr;
                    let mut arg_vd = v_descale_ptr;
                    let _ = (workspace, max_seqlen_q); // FA2 allocates in smem; qi-loop reads cu_seqlens_q

                    let args: [*mut core::ffi::c_void; 17] = [
                        &mut arg_out as *mut _ as *mut _,
                        &mut arg_q as *mut _ as *mut _,
                        &mut arg_k as *mut _ as *mut _,
                        &mut arg_v as *mut _ as *mut _,
                        &mut arg_bt as *mut _ as *mut _,
                        &mut arg_cl as *mut _ as *mut _,
                        &mut arg_cu as *mut _ as *mut _,
                        &mut arg_qd as *mut _ as *mut _,
                        &mut arg_kd as *mut _ as *mut _,
                        &mut arg_vd as *mut _ as *mut _,
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
                                op: "paged_prefill_fp8 (Fa2Ptx)",
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
            let Some(f) = fa3.fn_paged_prefill_fp8 else {
                return Err(RvllmError::Attention {
                    err: AttentionError::Fa3SoMissing {
                        path: fa3.so_path.clone(),
                    },
                    ctx: AttnCtx {
                        op: "paged_prefill_fp8 symbol missing from .so (rebuild fa3)",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            };
            let rc = f(
                q_fp8 as *mut std::ffi::c_void,
                k_cache_fp8 as *mut std::ffi::c_void,
                v_cache_fp8 as *mut std::ffi::c_void,
                o_f16 as *mut std::ffi::c_void,
                block_tables as *mut std::ffi::c_void,
                context_lens as *mut std::ffi::c_void,
                cu_seqlens_q as *mut std::ffi::c_void,
                workspace as *mut std::ffi::c_void,
                q_descale_ptr as *mut f32,
                k_descale_ptr as *mut f32,
                v_descale_ptr as *mut f32,
                params.scale,
                params.num_tokens as i32,
                max_seqlen_q as i32,
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
                        op: "paged_prefill_fp8",
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
                cu_seqlens_q, workspace, q_descale_ptr, k_descale_ptr, v_descale_ptr,
                max_seqlen_q, stream,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_validates_head_dim() {
        let p = PagedPrefillParams {
            num_tokens: 256,
            num_seqs: 4,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 64, // bad
            block_size: 64,
            max_blocks_per_seq: 33,
            num_blocks_total: 1024,
            scale: 1.0,
            window_size_left: -1,
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn prefill_accepts_head_dim_256() {
        let p = PagedPrefillParams {
            num_tokens: 256,
            num_seqs: 4,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 256,
            block_size: 64,
            max_blocks_per_seq: 33,
            num_blocks_total: 1024,
            scale: 1.0 / (256f32).sqrt(),
            window_size_left: -1,
        };
        assert!(p.validate().is_ok());
    }
}
