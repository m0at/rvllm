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

/// sm_121 unified prefill: BLOCK_M is fixed at 16 in the kernel (every
/// Gemma 4 layer lands on `max(16, next_pow2(num_queries_per_kv)) = 16`),
/// so callers don't need to pass it.
pub const UNIFIED_PREFILL_BLOCK_M: u32 = 16;

/// sm_121 unified prefill extra knobs. Separate struct so the Fa3
/// launch path doesn't drag per-slot scale caches + tile config it
/// doesn't need.
#[derive(Copy, Clone, Debug)]
pub struct UnifiedPrefillParams {
    pub num_queries_per_kv: u32,
    pub tile_size: u32,
    pub block_q: u32,
    /// Route Q·Kᵀ through the sm_121a FP8 tensor-core MMA when `true`.
    /// Scalar FMA otherwise (the correctness-validated baseline). The
    /// engine selects this per-launch from `RVLLM_UNIFIED_PREFILL_MMA`
    /// so A/B bisects stay trivial.
    pub use_mma: bool,
}

impl<'a> PagedPrefillFp8Launcher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { backend }
    }

    /// sm_121-only unified multi-Q FP8-KV prefill. Replaces the
    /// `PagedDecodeFp8Launcher` per-qi loop currently used by
    /// `gemma4_layer_exec::Gemma4Phase::Prefill`.
    ///
    /// Grid sizing, per the Triton `unified_attention` host code:
    ///     total_num_q_blocks = sum_i(query_len_i / block_q) + num_seqs
    /// We use the upper bound here (programs past the per-seq limit
    /// early-return inside the kernel).
    ///
    /// Other backends return `FeatureNotAvailable` — the Fa3 / SM90
    /// code continues to go through `launch()` above.
    ///
    /// # Safety
    /// Caller owns all device pointers. `cu_seqlens_q` is a
    /// `[num_seqs + 1]` i32 prefix-sum device buffer. The per-slot
    /// scale caches may be null, in which case the `*_descale_fallback`
    /// scalar pointers are used; they must outlive the launch.
    #[cfg_attr(feature = "cuda", allow(clippy::too_many_arguments))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8kv_unified_sm121(
        &self,
        params: PagedPrefillParams,
        unified: UnifiedPrefillParams,
        o_f16: u64,
        q_fp8: u64,
        k_cache_fp8: u64,
        v_cache_fp8: u64,
        k_scale_cache: u64,    // nullable
        v_scale_cache: u64,    // nullable
        q_scale_cache: u64,    // nullable
        k_descale_fallback: u64,
        v_descale_fallback: u64,
        block_tables: u64,
        cu_seqlens_q: u64,
        context_lens: u64,
        q_descale_fallback: u64,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        #[cfg(feature = "cuda")]
        {
            let fa2 = match self.backend {
                super::AttentionBackend::Fa2Ptx(f) => f,
                _ => {
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            op: "paged_prefill_fp8_unified (sm_121 only)",
                            backend: "non-Fa2Ptx",
                        },
                        ctx: AttnCtx {
                            op: "launch_fp8kv_unified_sm121",
                            stream,
                            num_seqs: params.num_seqs,
                            head_dim: params.head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    });
                }
            };
            let Some(kernel_fn) = fa2.fn_prefill_fp8kv_unified else {
                return Err(RvllmError::Attention {
                    err: AttentionError::FeatureNotAvailable {
                        op: "paged_prefill_fp8_unified (PTX module not loaded)",
                        backend: "Fa2Ptx",
                    },
                    ctx: AttnCtx {
                        op: "launch_fp8kv_unified_sm121",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            };

            use cudarc::driver::sys::*;
            const FA2_THREADS: u32 = 128;
            const BLOCK_M: u32 = UNIFIED_PREFILL_BLOCK_M;
            let hd = params.head_dim;
            let ts = unified.tile_size;

            // Smem budget — must match the kernel's layout. 128 B
            // cushion covers alignment slop without masking bugs.
            // Phase F3: Q is stored as FP8 + per-row f32 scale (was
            // pre-decoded f32). Phase F4: adds V-transpose buffer
            // + re-quantised-P region so the P·V MMA has the
            // operand layouts the tensor core expects.
            // Phase F7: s_v_fp8_T, s_s, s_p_fp8 are sized off MMA_K
            // (=32) rather than tile_size so the same kernel handles
            // sliding (tile_size=32) and global (tile_size=16) layers.
            // For sliding tile_size == MMA_K so no change; for global
            // these three regions grow by (MMA_K - tile_size)/ts
            // ratio. Full budget stays inside the 99 KB cap even at
            // head_dim=512 (~77 KB total, empirically).
            const MMA_K: u32 = 32;
            let smem_bytes: u32 = BLOCK_M * hd        // s_q_fp8
                + BLOCK_M * 4                          // s_q_scale
                + hd * ts                              // s_k_fp8
                + ts * hd                              // s_v_fp8
                + MMA_K * hd                           // s_v_fp8_T (F4+F7)
                + ts * 4                               // s_k_scale
                + ts * 4                               // s_v_scale
                + BLOCK_M * MMA_K * 4                  // s_s (F7: MMA_K, not ts)
                + BLOCK_M * 4 * 3                      // s_m + s_l + s_alpha
                + BLOCK_M * MMA_K                      // s_p_fp8 (F4+F7)
                + BLOCK_M * 4                          // s_p_scale (F4)
                + BLOCK_M * hd * 4                     // s_acc
                + (FA2_THREADS / 32) * 4               // reduce tail
                + 128;

            if smem_bytes >= 48 * 1024 {
                let _ = cuFuncSetAttribute(
                    kernel_fn.raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes as i32,
                );
            }

            // Scalars must outlive cuLaunchKernel.
            let scale = params.scale;
            let num_heads = params.num_heads as i32;
            let num_kv_heads = params.num_kv_heads as i32;
            let head_dim = params.head_dim as i32;
            let block_size = params.block_size as i32;
            let max_blocks_per_seq = params.max_blocks_per_seq as i32;
            let tile_size = unified.tile_size as i32;
            let num_queries_per_kv = unified.num_queries_per_kv as i32;
            let block_q = unified.block_q as i32;
            let num_seqs = params.num_seqs as i32;
            let window_size_left = params.window_size_left;
            let use_mma: i32 = if unified.use_mma { 1 } else { 0 };

            let mut arg_out = o_f16;
            let mut arg_q = q_fp8;
            let mut arg_k = k_cache_fp8;
            let mut arg_v = v_cache_fp8;
            let mut arg_ks = k_scale_cache;
            let mut arg_vs = v_scale_cache;
            let mut arg_qs = q_scale_cache;
            let mut arg_kdf = k_descale_fallback;
            let mut arg_vdf = v_descale_fallback;
            let mut arg_bt = block_tables;
            let mut arg_cu = cu_seqlens_q;
            let mut arg_cl = context_lens;
            let mut arg_qdf = q_descale_fallback;

            let args: [*mut core::ffi::c_void; 25] = [
                &mut arg_out as *mut _ as *mut _,
                &mut arg_q as *mut _ as *mut _,
                &mut arg_k as *mut _ as *mut _,
                &mut arg_v as *mut _ as *mut _,
                &mut arg_ks as *mut _ as *mut _,
                &mut arg_vs as *mut _ as *mut _,
                &mut arg_qs as *mut _ as *mut _,
                &mut arg_kdf as *mut _ as *mut _,
                &mut arg_vdf as *mut _ as *mut _,
                &mut arg_bt as *mut _ as *mut _,
                &mut arg_cu as *mut _ as *mut _,
                &mut arg_cl as *mut _ as *mut _,
                &mut arg_qdf as *mut _ as *mut _,
                &scale as *const _ as *mut _,
                &num_heads as *const _ as *mut _,
                &num_kv_heads as *const _ as *mut _,
                &head_dim as *const _ as *mut _,
                &block_size as *const _ as *mut _,
                &max_blocks_per_seq as *const _ as *mut _,
                &tile_size as *const _ as *mut _,
                &num_queries_per_kv as *const _ as *mut _,
                &block_q as *const _ as *mut _,
                &num_seqs as *const _ as *mut _,
                &window_size_left as *const _ as *mut _,
                &use_mma as *const _ as *mut _,
            ];

            // Grid: (total_num_q_blocks, num_kv_heads). params.num_tokens
            // is the sum of per-seq query lengths.
            let total_num_q_blocks =
                params.num_tokens.div_ceil(unified.block_q) + params.num_seqs;

            let rc = cuLaunchKernel(
                kernel_fn.raw() as CUfunction,
                total_num_q_blocks,
                params.num_kv_heads,
                1,
                FA2_THREADS,
                1,
                1,
                smem_bytes,
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
                        op: "paged_prefill_fp8_unified (Fa2Ptx)",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                params, unified, o_f16, q_fp8, k_cache_fp8, v_cache_fp8,
                k_scale_cache, v_scale_cache, q_scale_cache,
                k_descale_fallback, v_descale_fallback, block_tables,
                cu_seqlens_q, context_lens, q_descale_fallback, stream,
            );
            Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    op: "paged_prefill_fp8_unified (non-cuda build)",
                    backend: "mock",
                },
                ctx: AttnCtx {
                    op: "launch_fp8kv_unified_sm121",
                    stream,
                    num_seqs: params.num_seqs,
                    head_dim: params.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            })
        }
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
                super::AttentionBackend::Fa2Ptx(_) => {
                    // sm_121 no longer ships a dedicated FA2 prefill
                    // kernel — Gemma 4's unified attention in
                    // gemma4_layer_exec.rs replaces batch prefill with
                    // a loop of single-query decode launches, which is
                    // numerically identical to the per-token decode
                    // path rvllm-ppl validates. Callers on sm_121
                    // should route through `PagedDecodeFp8Launcher`
                    // directly; keeping this arm would just tempt them
                    // back into the less-accurate FA2 prefill.
                    let _ = (o_f16, q_fp8, k_cache_fp8, v_cache_fp8,
                        block_tables, context_lens, cu_seqlens_q,
                        workspace, q_descale_ptr, k_descale_ptr,
                        v_descale_ptr, max_seqlen_q, stream);
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            op: "paged_prefill_fp8 Fa2Ptx (use decode-per-qi loop)",
                            backend: "Fa2Ptx",
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

/// NVFP4 KV-cache paged-prefill launcher. Same shape as
/// `PagedPrefillFp8Launcher` but threads per-block E4M3 microscale
/// pointers and a packed 4-bit cache layout. Only `Fa2Ptx` implements
/// it; FA3 returns `FeatureNotAvailable`.
pub struct PagedPrefillNvfp4Launcher<'a> {
    backend: &'a super::AttentionBackend,
}

impl<'a> PagedPrefillNvfp4Launcher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { backend }
    }

    /// # Safety
    /// Same invariants as the FP8 prefill launcher, plus:
    ///   * `k_cache_packed` / `v_cache_packed`: 4-bit bytes.
    ///   * `k_cache_scale` / `v_cache_scale`: per-16-element E4M3.
    ///   * `q_descale_ptr`: single f32 scalar (Q stays FP8 per-tensor).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        params: PagedPrefillParams,
        o_f16: u64,
        q_fp8: u64,
        k_cache_packed: u64,
        v_cache_packed: u64,
        k_cache_scale: u64,
        v_cache_scale: u64,
        // === DYNAMIC NVFP4 Q SCALE ===
        q_scale_cache: u64,
        // === END DYNAMIC NVFP4 Q SCALE ===
        block_tables: u64,
        context_lens: u64,
        cu_seqlens_q: u64,
        q_descale_ptr: u64,
        max_seqlen_q: u32,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        #[cfg(feature = "cuda")]
        {
            let fa2 = match self.backend {
                super::AttentionBackend::Fa2Ptx(fa2) => fa2,
                _ => {
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            backend: "non-Fa2Ptx",
                            op: "paged_prefill_nvfp4",
                        },
                        ctx: AttnCtx {
                            op: "paged_prefill_nvfp4",
                            stream,
                            num_seqs: params.num_seqs,
                            head_dim: params.head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    });
                }
            };
            use cudarc::driver::sys::*;
            const FA2_THREADS: i32 = 128;
            let hd = params.head_dim as i32;
            let (kernel_opt, fa2_bc) = if hd > 256 {
                (fa2.fn_prefill_nvfp4kv_bc16, 16)
            } else {
                (fa2.fn_prefill_nvfp4kv, 32)
            };
            let kernel_fn = kernel_opt.ok_or_else(|| RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "Fa2Ptx",
                    op: "paged_prefill_nvfp4 (kernel missing from PTX)",
                },
                ctx: AttnCtx {
                    op: "paged_prefill_nvfp4",
                    stream,
                    num_seqs: params.num_seqs,
                    head_dim: params.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            })?;

            // Phase 2a (aa01001nvf4f16mma): K/V dequant target is
            // f16 smem (2 bytes/elem) — see the matching comment in
            // decode.rs's PagedDecodeNvfp4Launcher.
            let smem_bytes =
                2 * fa2_bc * hd * 2 + fa2_bc * 4 + (FA2_THREADS / 32) * 4;
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
            let mut arg_kp = k_cache_packed;
            let mut arg_vp = v_cache_packed;
            let mut arg_ks = k_cache_scale;
            let mut arg_vs = v_cache_scale;
            // === DYNAMIC NVFP4 Q SCALE ===
            let mut arg_qsc = q_scale_cache;
            // === END DYNAMIC NVFP4 Q SCALE ===
            let mut arg_bt = block_tables;
            let mut arg_cl = context_lens;
            let mut arg_cu = cu_seqlens_q;
            let mut arg_qd = q_descale_ptr;
            let _ = max_seqlen_q; // qi-loop drives over cu_seqlens_q

            let args: [*mut core::ffi::c_void; 18] = [
                &mut arg_out as *mut _ as *mut _,
                &mut arg_q   as *mut _ as *mut _,
                &mut arg_kp  as *mut _ as *mut _,
                &mut arg_vp  as *mut _ as *mut _,
                &mut arg_ks  as *mut _ as *mut _,
                &mut arg_vs  as *mut _ as *mut _,
                // === DYNAMIC NVFP4 Q SCALE ===
                &mut arg_qsc as *mut _ as *mut _,
                // === END DYNAMIC NVFP4 Q SCALE ===
                &mut arg_bt  as *mut _ as *mut _,
                &mut arg_cl  as *mut _ as *mut _,
                &mut arg_cu  as *mut _ as *mut _,
                &mut arg_qd  as *mut _ as *mut _,
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
                        op: "paged_prefill_nvfp4 (Fa2Ptx)",
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
            let _ = (o_f16, q_fp8, k_cache_packed, v_cache_packed,
                     k_cache_scale, v_cache_scale, q_scale_cache,
                     block_tables,
                     context_lens, cu_seqlens_q, q_descale_ptr,
                     max_seqlen_q, stream);
        }
        Ok(())
    }
}

impl<'a> PagedPrefillNvfp4Launcher<'a> {
    /// Multi-query NVFP4 prefill with `m16n8k16` f16 MMA inner loop
    /// (task aa01001nvf4f16mma Phase 2b). Grid + smem layout mirrors
    /// `launch_fp8kv_unified_sm121` but reads the KV cache as NVFP4
    /// packed bytes + per-16-element E4M3 microscale, dequants to
    /// f16 smem via `cvt.rn.f16x2.e2m1x2`, and drives the tensor
    /// core through `f16_mma_frag_pack.cuh`.
    ///
    /// # Safety
    /// Same invariants as the FP8 unified launcher. Additionally:
    ///   * `k_cache_scale` / `v_cache_scale` must be
    ///     `[num_blocks * block_size * num_kv_heads * head_dim/16]`
    ///     E4M3 microscales.
    ///   * `q_scale_cache` may be null; the kernel falls back to
    ///     `*q_descale_fallback` per row in that case.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_nvfp4kv_unified_sm121(
        &self,
        params: PagedPrefillParams,
        unified: UnifiedPrefillParams,
        o_f16: u64,
        q_fp8: u64,
        k_cache_packed: u64,
        v_cache_packed: u64,
        k_cache_scale: u64,
        v_cache_scale: u64,
        q_scale_cache: u64,
        block_tables: u64,
        cu_seqlens_q: u64,
        context_lens: u64,
        q_descale_fallback: u64,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        #[cfg(feature = "cuda")]
        {
            let fa2 = match self.backend {
                super::AttentionBackend::Fa2Ptx(fa2) => fa2,
                _ => {
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            backend: "non-Fa2Ptx",
                            op: "paged_prefill_nvfp4_unified",
                        },
                        ctx: AttnCtx {
                            op: "paged_prefill_nvfp4_unified",
                            stream,
                            num_seqs: params.num_seqs,
                            head_dim: params.head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    });
                }
            };
            use cudarc::driver::sys::*;
            const FA2_THREADS: i32 = 128;
            let kernel_fn = fa2.fn_prefill_nvfp4kv_unified.ok_or_else(|| {
                RvllmError::Attention {
                    err: AttentionError::FeatureNotAvailable {
                        backend: "Fa2Ptx",
                        op: "paged_prefill_nvfp4_unified (kernel missing from PTX)",
                    },
                    ctx: AttnCtx {
                        op: "paged_prefill_nvfp4_unified",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                }
            })?;

            // Smem layout must match `flash_attention_unified_prefill_nvfp4kv.cu`.
            const MMA_K: u32 = 16;
            let block_m: u32 = UNIFIED_PREFILL_BLOCK_M;
            let hd = params.head_dim;
            let ts = unified.tile_size;
            let s_s_stride = ts.max(MMA_K);
            let smem_bytes: u32 = block_m * hd * 2        // s_q_f16
                + block_m * 4                              // s_q_scale
                + ts * hd * 2                              // s_k_f16
                + ts * hd * 2                              // s_v_f16
                + MMA_K * hd * 2                           // s_v_f16_T
                + block_m * s_s_stride * 4                 // s_s
                + block_m * 4 * 3                          // s_m + s_l + s_alpha
                + block_m * MMA_K * 2                      // s_p_f16
                + block_m * 4                              // s_p_scale
                + block_m * hd * 4                         // s_acc
                + 256;                                     // alignment cushion

            if smem_bytes >= 48 * 1024 {
                let _ = cuFuncSetAttribute(
                    kernel_fn.raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes as i32,
                );
            }

            let scale = params.scale;
            let num_heads = params.num_heads as i32;
            let num_kv_heads = params.num_kv_heads as i32;
            let head_dim = params.head_dim as i32;
            let block_size = params.block_size as i32;
            let max_blocks_per_seq = params.max_blocks_per_seq as i32;
            let tile_size = unified.tile_size as i32;
            let num_queries_per_kv = unified.num_queries_per_kv as i32;
            let block_q = unified.block_q as i32;
            let num_seqs = params.num_seqs as i32;
            let window_size_left = params.window_size_left;

            let mut arg_out  = o_f16;
            let mut arg_q    = q_fp8;
            let mut arg_kp   = k_cache_packed;
            let mut arg_vp   = v_cache_packed;
            let mut arg_ks   = k_cache_scale;
            let mut arg_vs   = v_cache_scale;
            let mut arg_qs   = q_scale_cache;
            let mut arg_bt   = block_tables;
            let mut arg_cu   = cu_seqlens_q;
            let mut arg_cl   = context_lens;
            let mut arg_qd   = q_descale_fallback;

            let args: [*mut core::ffi::c_void; 22] = [
                &mut arg_out as *mut _ as *mut _,
                &mut arg_q   as *mut _ as *mut _,
                &mut arg_kp  as *mut _ as *mut _,
                &mut arg_vp  as *mut _ as *mut _,
                &mut arg_ks  as *mut _ as *mut _,
                &mut arg_vs  as *mut _ as *mut _,
                &mut arg_qs  as *mut _ as *mut _,
                &mut arg_bt  as *mut _ as *mut _,
                &mut arg_cu  as *mut _ as *mut _,
                &mut arg_cl  as *mut _ as *mut _,
                &mut arg_qd  as *mut _ as *mut _,
                &scale as *const _ as *mut _,
                &num_heads as *const _ as *mut _,
                &num_kv_heads as *const _ as *mut _,
                &head_dim as *const _ as *mut _,
                &block_size as *const _ as *mut _,
                &max_blocks_per_seq as *const _ as *mut _,
                &tile_size as *const _ as *mut _,
                &num_queries_per_kv as *const _ as *mut _,
                &block_q as *const _ as *mut _,
                &num_seqs as *const _ as *mut _,
                &window_size_left as *const _ as *mut _,
            ];

            // Grid: (total_num_q_blocks, num_kv_heads) — same shape as
            // the FP8 unified launcher.
            let total_num_q_blocks =
                params.num_tokens.div_ceil(unified.block_q) + params.num_seqs;

            let rc = cuLaunchKernel(
                kernel_fn.raw() as CUfunction,
                total_num_q_blocks,
                params.num_kv_heads,
                1,
                FA2_THREADS as u32,
                1,
                1,
                smem_bytes,
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
                        op: "paged_prefill_nvfp4_unified (Fa2Ptx)",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (params, unified, o_f16, q_fp8, k_cache_packed, v_cache_packed,
                     k_cache_scale, v_cache_scale, q_scale_cache,
                     block_tables, cu_seqlens_q, context_lens,
                     q_descale_fallback, stream);
            Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    op: "paged_prefill_nvfp4_unified (non-cuda build)",
                    backend: "mock",
                },
                ctx: AttnCtx {
                    op: "launch_nvfp4kv_unified_sm121",
                    stream,
                    num_seqs: params.num_seqs,
                    head_dim: params.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            })
        }
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
