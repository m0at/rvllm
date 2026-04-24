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

/// Launcher. Construction from `&Fa3Kernels` guarantees the .so is
/// loaded and head_dim is supported by the backend.
pub struct PagedDecodeLauncher<'a> {
    fa3: &'a super::Fa3Kernels,
}

impl<'a> PagedDecodeLauncher<'a> {
    pub fn new(fa3: &'a super::Fa3Kernels) -> Self {
        Self { fa3 }
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
            let rc = (self.fa3.fn_paged_decode)(
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
pub struct PagedDecodeFp8Launcher<'a> {
    fa3: &'a super::Fa3Kernels,
}

impl<'a> PagedDecodeFp8Launcher<'a> {
    pub fn new(fa3: &'a super::Fa3Kernels) -> Self {
        Self { fa3 }
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
        block_tables: u64,
        context_lens: u64,
        workspace: u64,
        q_descale_ptr: u64,
        k_descale_ptr: u64,
        v_descale_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        #[cfg(feature = "cuda")]
        {
            let rc = (self.fa3.fn_paged_decode_fp8)(
                q_fp8 as *mut std::ffi::c_void,
                k_cache_fp8 as *mut std::ffi::c_void,
                v_cache_fp8 as *mut std::ffi::c_void,
                o_f16 as *mut std::ffi::c_void,
                block_tables as *mut std::ffi::c_void,
                context_lens as *mut std::ffi::c_void,
                workspace as *mut std::ffi::c_void,
                q_descale_ptr as *mut f32,
                k_descale_ptr as *mut f32,
                v_descale_ptr as *mut f32,
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
                o_f16,
                q_fp8,
                k_cache_fp8,
                v_cache_fp8,
                block_tables,
                context_lens,
                workspace,
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                stream,
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
