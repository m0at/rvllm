//! Paged-decode launcher.
//!
//! One query per sequence. Kernel reads context_lens[seq] and walks
//! block_tables[seq, 0..ceil(context_lens/block_size)] to find KV
//! pages. `context_lens[i] == 0` is a valid padded slot; kernel must
//! predicate and never touch block_tables[i,*].

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

const SUPPORTED_HEAD_DIMS: &[u32] = &[128, 256, 512];

/// Cycle 52 step 11c (codex audit finding #3): assert that the launch
/// `params.head_dim` matches the backend's load-time head_dim. Gemma 4
/// 31B has heterogeneous attention (sliding head_dim=256, global=512)
/// with separate Fa2PtxKernels instances loaded for each. A wrong
/// route — e.g. dispatching a global-layer launch through the
/// sliding backend — silently picks the wrong BC variant + mismatched
/// smem layout. Fa3 returns head_dim=0 (validated at load time);
/// skip the assert in that case.
fn assert_head_dim_matches_backend(
    backend: &super::AttentionBackend,
    params_head_dim: u32,
    op: &'static str,
    stream: u64,
    num_seqs: u32,
) -> Result<()> {
    let backend_hd = backend.head_dim();
    if backend_hd != 0 && backend_hd != params_head_dim {
        eprintln!(
            "[attn] head_dim mismatch at {op}: params={params_head_dim}, backend={backend_hd}"
        );
        return Err(RvllmError::Attention {
            err: AttentionError::FeatureNotAvailable {
                backend: "head_dim-cross-check",
                op,
            },
            ctx: AttnCtx { op, stream, num_seqs, head_dim: params_head_dim },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    Ok(())
}

/// Cycle 52 step 11b (codex audit finding #1): host-side null-pointer
/// guard. Pre-launch, every required device pointer must be non-zero;
/// silent zero pointers manifest as kernel-side invalid reads or
/// silent garbage (e.g. a zero `context_lens` causes the kernel to
/// read seq lens from device address 0 → SIGSEGV on first attention
/// dispatch).
///
/// `op` and `head_dim`/`num_seqs` go into the AttnCtx so the error
/// message points at the right launch site. Nullable pointers
/// (e.g. some scale caches that are documented Option) MUST NOT be
/// passed here — only required ones.
fn require_nonnull(
    ptrs: &[(&'static str, u64)],
    op: &'static str,
    stream: u64,
    num_seqs: u32,
    head_dim: u32,
) -> Result<()> {
    for (name, p) in ptrs {
        if *p == 0 {
            return Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "host-validation",
                    op,
                },
                ctx: AttnCtx {
                    op,
                    stream,
                    num_seqs,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            }).map_err(|e| {
                eprintln!("[attn] required device ptr {name:?} == 0 at {op}");
                e
            });
        }
    }
    Ok(())
}

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
        // Cycle 52 attention hardening (codex finding #2): reject more
        // launch-shape misuse before we hit a kernel-side invalid read
        // or a divide-by-zero in the grid sizer.
        if self.num_heads == 0 || self.block_size == 0
            || self.num_blocks_total == 0 || self.max_blocks_per_seq == 0
        {
            return Err(RvllmError::Attention {
                err: AttentionError::ContextExceedsBucket { context: 0, max: 0 },
                ctx: ctx(),
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
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
        require_nonnull(
            &[
                ("out_ptr",          out_ptr),
                ("q_ptr",            q_ptr),
                ("k_cache_ptr",      k_cache_ptr),
                ("v_cache_ptr",      v_cache_ptr),
                ("block_tables_ptr", block_tables_ptr),
                ("context_lens_ptr", context_lens_ptr),
                ("workspace_ptr",    workspace_ptr),
            ],
            "paged_decode (Fa3/SM90)",
            stream, params.num_seqs, params.head_dim,
        )?;
        assert_head_dim_matches_backend(
            self.backend, params.head_dim,
            "paged_decode (Fa3/SM90)", stream, params.num_seqs,
        )?;
        #[cfg(feature = "cuda")]
        {
            let fa3 = match self.backend {
                super::AttentionBackend::Fa3(fa3) => fa3,
                super::AttentionBackend::Fa2Ptx(fa2) => {
                    // sm_121 F16 KV path: dispatch
                    // `flash_attention_2_decode_f16io_kernel` (f16 I/O
                    // against the paged f16 cache). head_dim=512 does
                    // NOT fit the BC=32 smem budget on sm_121's 99 KB
                    // opt-in cap — gate it off. Gemma 4 global layers
                    // must use FP8 KV; F16 KV on sm_121 today is a
                    // head_dim ≤ 256 path (Llama/Qwen or Gemma 4
                    // sliding-only).
                    if params.head_dim > 256 {
                        return Err(RvllmError::Attention {
                            err: AttentionError::FeatureNotAvailable {
                                backend: "Fa2Ptx",
                                op: "paged_decode (F16 KV, head_dim>256 — use FP8 KV)",
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
                    use cudarc::driver::sys::*;
                    const FA2_THREADS: i32 = 128;
                    const FA2_BC: i32 = 32;
                    let hd = params.head_dim as i32;
                    let smem_bytes = 2 * FA2_BC * hd * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;
                    if smem_bytes as u32 >= 48 * 1024 {
                        let _ = cuFuncSetAttribute(
                            fa2.fn_decode_f16io.raw() as CUfunction,
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

                    let mut arg_out = out_ptr;
                    let mut arg_q = q_ptr;
                    let mut arg_k = k_cache_ptr;
                    let mut arg_v = v_cache_ptr;
                    let mut arg_bt = block_tables_ptr;
                    let mut arg_cl = context_lens_ptr;
                    let _ = workspace_ptr; // unused: FA2 allocates smem dynamically

                    let args: [*mut core::ffi::c_void; 12] = [
                        &mut arg_out as *mut _ as *mut _,
                        &mut arg_q as *mut _ as *mut _,
                        &mut arg_k as *mut _ as *mut _,
                        &mut arg_v as *mut _ as *mut _,
                        &mut arg_bt as *mut _ as *mut _,
                        &mut arg_cl as *mut _ as *mut _,
                        &scale as *const _ as *mut _,
                        &num_heads as *const _ as *mut _,
                        &num_kv_heads as *const _ as *mut _,
                        &head_dim as *const _ as *mut _,
                        &block_size as *const _ as *mut _,
                        &max_blocks_per_seq as *const _ as *mut _,
                    ];

                    let rc = cuLaunchKernel(
                        fa2.fn_decode_f16io.raw() as CUfunction,
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
                                op: "paged_decode (Fa2Ptx F16 KV)",
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
        // `q_scale_cache`: `[num_seqs * num_heads]` f32 array of
        //   per-(seq, head) Q scales. Pass `0` to fall back to the
        //   scalar `q_descale_ptr`.
        q_scale_cache: u64,
        k_descale_fallback_ptr: u64,
        v_descale_fallback_ptr: u64,
        block_tables: u64,
        context_lens: u64,
        workspace: u64,
        q_descale_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        require_nonnull(
            &[
                ("o_f16",         o_f16),
                ("q_fp8",         q_fp8),
                ("k_cache_fp8",   k_cache_fp8),
                ("v_cache_fp8",   v_cache_fp8),
                ("block_tables",  block_tables),
                ("context_lens",  context_lens),
                ("q_descale_ptr", q_descale_ptr),
            ],
            "paged_decode (FP8 KV)",
            stream, params.num_seqs, params.head_dim,
        )?;
        // Codex review of bc358cf: K/V need at least one scale source.
        // For Fa2Ptx: k_scale_cache (per-slot) OR k_descale_fallback_ptr
        // (scalar) must be non-zero. For Fa3: scale_cache is ignored,
        // fallback ptr is required. Either way one path must be set.
        if k_scale_cache == 0 && k_descale_fallback_ptr == 0 {
            eprintln!("[attn] FP8 decode: both k_scale_cache and k_descale_fallback_ptr are 0");
            return Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "host-validation",
                    op: "paged_decode (FP8 KV) — K scale source missing",
                },
                ctx: AttnCtx { op: "paged_decode (FP8 KV)", stream,
                    num_seqs: params.num_seqs, head_dim: params.head_dim },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if v_scale_cache == 0 && v_descale_fallback_ptr == 0 {
            eprintln!("[attn] FP8 decode: both v_scale_cache and v_descale_fallback_ptr are 0");
            return Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "host-validation",
                    op: "paged_decode (FP8 KV) — V scale source missing",
                },
                ctx: AttnCtx { op: "paged_decode (FP8 KV)", stream,
                    num_seqs: params.num_seqs, head_dim: params.head_dim },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        // FA3-specific gate: the SM90 ABI (`fn_paged_decode_fp8`)
        // takes scalar K/V descales only and cannot consume per-slot
        // caches — the launch site at the bottom of this fn drops
        // `k_scale_cache` / `v_scale_cache` and forwards just the
        // fallback pointers. If the caller supplied ONLY a cache
        // (no fallback), the kernel would launch with null descale
        // pointers and silently produce garbage. Reject the combo
        // up front rather than passing validation under false
        // pretences. Fa2Ptx (sm_121 path) consumes per-slot caches
        // natively, so this restriction does not apply there.
        #[cfg(feature = "cuda")]
        if matches!(self.backend, super::AttentionBackend::Fa3(_)) {
            if k_descale_fallback_ptr == 0 {
                return Err(RvllmError::Attention {
                    err: AttentionError::FeatureNotAvailable {
                        backend: "fa3-sm90",
                        op: "paged_decode (FP8 KV) — Fa3 ABI requires \
                             scalar K descale via k_descale_fallback_ptr; \
                             per-slot k_scale_cache is not consumed",
                    },
                    ctx: AttnCtx { op: "paged_decode (FP8 KV)", stream,
                        num_seqs: params.num_seqs, head_dim: params.head_dim },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
            if v_descale_fallback_ptr == 0 {
                return Err(RvllmError::Attention {
                    err: AttentionError::FeatureNotAvailable {
                        backend: "fa3-sm90",
                        op: "paged_decode (FP8 KV) — Fa3 ABI requires \
                             scalar V descale via v_descale_fallback_ptr; \
                             per-slot v_scale_cache is not consumed",
                    },
                    ctx: AttnCtx { op: "paged_decode (FP8 KV)", stream,
                        num_seqs: params.num_seqs, head_dim: params.head_dim },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
        }
        assert_head_dim_matches_backend(
            self.backend, params.head_dim,
            "paged_decode (FP8 KV)", stream, params.num_seqs,
        )?;
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
                    // sm_121 FP8-KV decode is BC=16 only. head_dim=512
                    // never fit BC=32 within the 99 KB opt-in smem cap,
                    // and head_dim=256 measurably favours BC=16 too
                    // (+2.5%/+5.5% at batch=128/256 — halving the tile
                    // lets 2+ blocks live per SM and hides per-tile
                    // __syncthreads latency). The BC=32 kernel was
                    // removed from flash_attention.cu in the cleanup
                    // that followed the ncu profile.
                    use cudarc::driver::sys::*;
                    const FA2_THREADS: i32 = 128;
                    const FA2_BC: i32 = 16;
                    const MAX_GQA_DECODE: i32 = 4;
                    let hd = params.head_dim as i32;
                    // GQA-grouped dispatch: one CTA per (seq, kv_head)
                    // amortizes KV dequant across the GQA query group.
                    // Transparent fallback to per-head kernel when the
                    // GQA symbol is absent (older PTX) or ratio is out
                    // of range.
                    let gqa_ratio = if params.num_kv_heads > 0 {
                        params.num_heads / params.num_kv_heads
                    } else {
                        1
                    };
                    // FP8 GQA decode is opt-in via env gate until the
                    // fp64 harness is ported to FP8. NVFP4 GQA has
                    // harness coverage and defaults on. Set
                    // `RVLLM_FP8_DECODE_GQA=1` to enable.
                    let gqa_env_on = std::env::var("RVLLM_FP8_DECODE_GQA")
                        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                        .unwrap_or(false);
                    let use_gqa = gqa_env_on
                        && gqa_ratio > 1
                        && gqa_ratio <= MAX_GQA_DECODE as u32
                        && fa2.fn_decode_fp8kv_gqa.is_some();
                    let kernel_fn = if use_gqa {
                        fa2.fn_decode_fp8kv_gqa.unwrap()
                    } else {
                        fa2.fn_decode_fp8kv
                    };
                    let score_rows = if use_gqa { MAX_GQA_DECODE } else { 1 };
                    let smem_bytes =
                        2 * FA2_BC * hd * 4 + score_rows * FA2_BC * 4 + (FA2_THREADS / 32) * 4;
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
                    let mut arg_qs = q_scale_cache;
                    let mut arg_kd = k_descale_fallback_ptr;
                    let mut arg_vd = v_descale_fallback_ptr;
                    let mut arg_bt = block_tables;
                    let mut arg_cl = context_lens;
                    let mut arg_qd = q_descale_ptr;
                    let _ = workspace; // unused: FA2 allocates in smem

                    let args: [*mut core::ffi::c_void; 19] = [
                        &mut arg_out as *mut _ as *mut _,
                        &mut arg_q as *mut _ as *mut _,
                        &mut arg_k as *mut _ as *mut _,
                        &mut arg_v as *mut _ as *mut _,
                        &mut arg_ks as *mut _ as *mut _,
                        &mut arg_vs as *mut _ as *mut _,
                        &mut arg_qs as *mut _ as *mut _,
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

                    let grid_y = if use_gqa {
                        params.num_kv_heads as u32
                    } else {
                        params.num_heads as u32
                    };
                    let rc = cuLaunchKernel(
                        kernel_fn.raw() as CUfunction,
                        params.num_seqs as u32,
                        grid_y,
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
            // Fa3 SM90 path: takes scalar K/V descales (its ABI
            // predates per-slot scales). Llama/Qwen callers pass
            // their per-tensor scale via `k_descale_fallback_ptr` /
            // `v_descale_fallback_ptr`; Gemma 4 sm_121 never reaches
            // this arm (it goes through the Fa2Ptx branch above).
            // If per-slot scales are populated AND an Fa3 caller
            // somehow routes through here, that caller needs to
            // materialize a representative scalar into the fallback
            // pointer — the SM90 kernel can't consume per-slot.
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
                k_descale_fallback_ptr as *mut f32,
                v_descale_fallback_ptr as *mut f32,
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

/// NVFP4 KV-cache paged-decode launcher. Same shape as
/// `PagedDecodeFp8Launcher` but threads per-block E4M3 microscale
/// pointers instead of per-tensor descales, and the K/V cache
/// pointers reference the packed 4-bit byte layout.
///
/// Only the `Fa2Ptx` backend implements this — FA3 (SM90) has no
/// NVFP4 path and returns `FeatureNotAvailable`.
pub struct PagedDecodeNvfp4Launcher<'a> {
    backend: &'a super::AttentionBackend,
}

impl<'a> PagedDecodeNvfp4Launcher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { backend }
    }

    /// # Safety
    /// Every pointer must be valid device memory.
    ///   * `k_cache_packed` / `v_cache_packed`: packed 4-bit bytes
    ///     `[num_blocks * block_size * num_kv_heads * head_dim/2]`.
    ///   * `k_cache_scale` / `v_cache_scale`: `__nv_fp8_e4m3`
    ///     `[num_blocks * block_size * num_kv_heads * head_dim/16]`.
    ///   * `q_descale_ptr`: single f32 scalar (Q stays FP8 per-tensor).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        params: PagedDecodeParams,
        o_f16: u64,
        q_fp8: u64,
        k_cache_packed: u64,
        v_cache_packed: u64,
        k_cache_scale: u64,
        v_cache_scale: u64,
        // === DYNAMIC NVFP4 Q SCALE ===
        // Optional `[num_seqs, num_heads]` f32 per-(seq, head) Q
        // descale. When non-zero, kernel reads from this; when 0,
        // falls back to the scalar `q_descale_ptr`.
        q_scale_cache: u64,
        // === END DYNAMIC NVFP4 Q SCALE ===
        block_tables: u64,
        context_lens: u64,
        q_descale_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        require_nonnull(
            &[
                ("o_f16",          o_f16),
                ("q_fp8",          q_fp8),
                ("k_cache_packed", k_cache_packed),
                ("v_cache_packed", v_cache_packed),
                ("k_cache_scale",  k_cache_scale),
                ("v_cache_scale",  v_cache_scale),
                ("block_tables",   block_tables),
                ("context_lens",   context_lens),
                ("q_descale_ptr",  q_descale_ptr),
            ],
            "paged_decode (NVFP4 KV)",
            stream, params.num_seqs, params.head_dim,
        )?;
        assert_head_dim_matches_backend(
            self.backend, params.head_dim,
            "paged_decode (NVFP4 KV)", stream, params.num_seqs,
        )?;
        #[cfg(feature = "cuda")]
        {
            let fa2 = match self.backend {
                super::AttentionBackend::Fa2Ptx(fa2) => fa2,
                _ => {
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            backend: "non-Fa2Ptx",
                            op: "paged_decode_nvfp4",
                        },
                        ctx: AttnCtx {
                            op: "paged_decode_nvfp4",
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
            const MAX_GQA_DECODE: u32 = 4;
            let hd = params.head_dim as i32;
            // GQA-grouped path: one CTA per (seq, kv_head) shares a
            // single KV dequant across all queries in the group. Only
            // valid when ratio > 1 and ≤ MAX_GQA_DECODE (matches the
            // compile-time cap on per-thread register arrays in the
            // kernel). Transparent fallback to per-head kernel when
            // the GQA symbol is absent (older PTX tree).
            let gqa_ratio = if params.num_kv_heads > 0 {
                params.num_heads / params.num_kv_heads
            } else {
                1
            };
            // NVFP4 GQA decode — harness-validated in isolation but
            // gated off by default in production until stability is
            // confirmed end-to-end on 15k-ctx workloads. Set
            // `RVLLM_NVFP4_DECODE_GQA=1` to opt in.
            let gqa_env_on = std::env::var("RVLLM_NVFP4_DECODE_GQA")
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false);
            let use_gqa = gqa_env_on && gqa_ratio > 1 && gqa_ratio <= MAX_GQA_DECODE;
            let (kernel_opt, fa2_bc) = if hd > 256 {
                let k = if use_gqa {
                    fa2.fn_decode_nvfp4kv_gqa_bc16.or(fa2.fn_decode_nvfp4kv_bc16)
                } else {
                    fa2.fn_decode_nvfp4kv_bc16
                };
                (k, 16)
            } else {
                let k = if use_gqa {
                    fa2.fn_decode_nvfp4kv_gqa.or(fa2.fn_decode_nvfp4kv)
                } else {
                    fa2.fn_decode_nvfp4kv
                };
                (k, 32)
            };
            let gqa_dispatched = use_gqa
                && if hd > 256 {
                    fa2.fn_decode_nvfp4kv_gqa_bc16.is_some()
                } else {
                    fa2.fn_decode_nvfp4kv_gqa.is_some()
                };
            let kernel_fn = kernel_opt.ok_or_else(|| RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "Fa2Ptx",
                    op: "paged_decode_nvfp4 (kernel missing from PTX — rebuild kernels/ with rusty_sm121_nvfp4)",
                },
                ctx: AttnCtx {
                    op: "paged_decode_nvfp4",
                    stream,
                    num_seqs: params.num_seqs,
                    head_dim: params.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            })?;

            // Phase 2a (aa01001nvf4f16mma): K/V dequant target is
            // f16 smem (2 bytes/elem) instead of f32 (4 bytes/elem),
            // halving the K/V tile footprint. s_score and s_reduce
            // stay f32 for softmax precision. GQA path widens s_score
            // by MAX_GQA_DECODE (one scores row per query in group).
            let score_rows = if gqa_dispatched { MAX_GQA_DECODE as i32 } else { 1 };
            let smem_bytes =
                2 * fa2_bc * hd * 2 + score_rows * fa2_bc * 4 + (FA2_THREADS / 32) * 4;
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
            let mut arg_qd = q_descale_ptr;

            let args: [*mut core::ffi::c_void; 17] = [
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
                &mut arg_qd  as *mut _ as *mut _,
                &scale as *const _ as *mut _,
                &num_heads as *const _ as *mut _,
                &num_kv_heads as *const _ as *mut _,
                &head_dim as *const _ as *mut _,
                &block_size as *const _ as *mut _,
                &max_blocks_per_seq as *const _ as *mut _,
                &window_size_left as *const _ as *mut _,
            ];

            let grid_y = if gqa_dispatched {
                params.num_kv_heads as u32
            } else {
                params.num_heads as u32
            };
            let rc = cuLaunchKernel(
                kernel_fn.raw() as CUfunction,
                params.num_seqs as u32,
                grid_y,
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
                        op: "paged_decode_nvfp4 (Fa2Ptx)",
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
                     context_lens, q_descale_ptr, stream);
        }
        Ok(())
    }

    /// Split-KV decode (paged_attention_v2-style, phase-1 + phase-2).
    /// Launches:
    ///   Phase 1: `(num_seqs, num_heads, max_num_partitions)` CTAs
    ///            into `workspace_ptr` (tmp_out / max_logits / exp_sums).
    ///   Phase 2: `(num_seqs, num_heads)` CTAs combining partitions.
    ///
    /// Target: 3-5× speedup at bs=1 long context where the single-CTA
    /// per-head kernel under-fills GB10's 108 SMs (32 CTAs → ~30%).
    ///
    /// Caller invariants:
    ///   * `workspace_ptr` must have at least
    ///     `num_seqs * num_heads * max_num_partitions *
    ///     (head_dim * 4 + 8)` bytes (f32 tmp_out + 2×f32 metadata).
    ///     (was f16 tmp_out — codex cycle21 fix: NVFP4 split-decode
    ///     was losing precision via f16 round-trip on many independently
    ///     normalized partial outputs; bumped to f32 to match single-CTA's
    ///     "one final f16 cast" behavior.)
    ///   * `max_num_partitions >= ceil(max_ctx_len / partition_size)`.
    ///   * `partition_size` must be a multiple of 16 (our FA2 block size).
    ///
    /// Returns `FeatureNotAvailable` if the split kernels aren't in the
    /// PTX module (older kernel tree). Caller should fall back to
    /// `launch`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_split(
        &self,
        params: PagedDecodeParams,
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
        q_descale_ptr: u64,
        workspace_ptr: u64,
        partition_size: u32,
        max_num_partitions: u32,
        // When `true`, dispatch the bf16-output split kernel pair
        // (`fn_decode_nvfp4kv_split{,_bc16}_bf16out` + the bf16
        // reducer). When `false`, the original f16-output pair runs.
        // Pair must stay in lock-step: f16 reducer reading bf16
        // partial tiles (or vice versa) is silent layout corruption.
        // Today's gemma4 caller passes `false`; the parameter exists
        // to make the bf16 path actually reachable when the cycle 55
        // bf16-everywhere work activates it.
        output_bf16: bool,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        // Enforce the documented caller invariants up front. Caller
        // code in `gemma4_layer_exec` already gets these right today,
        // but the launcher is an exported boundary and the docs
        // promise checks that weren't actually performed. The reduce
        // kernel's scratch indexing assumes `max_num_partitions` is
        // an upper bound on the per-seq partition count derived from
        // the device-side `context_lens`; a mismatched value silently
        // OOB-reads / -writes the partition scratch arena
        // (kernels/flash_attention_split_decode_nvfp4kv.cu:654).
        if partition_size == 0 || partition_size % 16 != 0 {
            return Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "host-validation",
                    op: "paged_decode_nvfp4_split — partition_size must be a non-zero multiple of 16",
                },
                ctx: AttnCtx { op: "paged_decode_nvfp4_split", stream,
                    num_seqs: params.num_seqs, head_dim: params.head_dim },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if max_num_partitions == 0 {
            return Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "host-validation",
                    op: "paged_decode_nvfp4_split — max_num_partitions must be >= 1",
                },
                ctx: AttnCtx { op: "paged_decode_nvfp4_split", stream,
                    num_seqs: params.num_seqs, head_dim: params.head_dim },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        // Bucket-context-len ceiling check: max_num_partitions must
        // cover ceil(max_ctx_len / partition_size). max_ctx_len is
        // bounded by the page table: `max_blocks_per_seq *
        // block_size`. If the caller's `max_num_partitions` is
        // smaller than this ceiling, the reduce-kernel will read
        // partition slots that were never written.
        let bucket_ctx = (params.max_blocks_per_seq as u64)
            .saturating_mul(params.block_size as u64);
        let needed_parts = bucket_ctx
            .div_ceil(partition_size as u64)
            .max(1) as u32;
        if max_num_partitions < needed_parts {
            return Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "host-validation",
                    op: "paged_decode_nvfp4_split — max_num_partitions too small \
                         for bucket_ctx / partition_size",
                },
                ctx: AttnCtx { op: "paged_decode_nvfp4_split", stream,
                    num_seqs: params.num_seqs, head_dim: params.head_dim },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        require_nonnull(
            &[
                ("o_f16",          o_f16),
                ("q_fp8",          q_fp8),
                ("k_cache_packed", k_cache_packed),
                ("v_cache_packed", v_cache_packed),
                ("k_cache_scale",  k_cache_scale),
                ("v_cache_scale",  v_cache_scale),
                ("block_tables",   block_tables),
                ("context_lens",   context_lens),
                ("q_descale_ptr",  q_descale_ptr),
                ("workspace_ptr",  workspace_ptr),
            ],
            "paged_decode_nvfp4_split",
            stream, params.num_seqs, params.head_dim,
        )?;
        assert_head_dim_matches_backend(
            self.backend, params.head_dim,
            "paged_decode_nvfp4_split", stream, params.num_seqs,
        )?;
        #[cfg(feature = "cuda")]
        {
            let fa2 = match self.backend {
                super::AttentionBackend::Fa2Ptx(fa2) => fa2,
                _ => {
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            backend: "non-Fa2Ptx",
                            op: "paged_decode_nvfp4_split",
                        },
                        ctx: AttnCtx {
                            op: "paged_decode_nvfp4_split",
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
            const REDUCE_THREADS: i32 = 128;
            let hd = params.head_dim as i32;

            // Dispatch split-attention + reducer pair coherently
            // based on the requested output dtype. Previously the
            // bf16 reducer was loaded into Fa2PtxKernels but never
            // wired here — every call used the f16 reducer, even
            // when a future caller would invoke `launch_split` with
            // bf16-out. Once the bf16 split path is exercised this
            // would mean the bf16 split kernel writes bf16 partial
            // tiles and the f16 reducer reads them as f16 → silent
            // numeric drift (or worse, a layout mismatch). The
            // `output_bf16` flag now keeps the pair in lock-step.
            let split_fn = if output_bf16 {
                if hd > 256 {
                    fa2.fn_decode_nvfp4kv_split_bc16_bf16out
                } else {
                    fa2.fn_decode_nvfp4kv_split_bf16out
                }
            } else if hd > 256 {
                fa2.fn_decode_nvfp4kv_split_bc16
            } else {
                fa2.fn_decode_nvfp4kv_split
            }
            .ok_or_else(|| RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "Fa2Ptx",
                    op: if output_bf16 {
                        "paged_decode_nvfp4_split (bf16-out split kernel missing — rebuild kernels/)"
                    } else {
                        "paged_decode_nvfp4_split (split kernel missing — rebuild kernels/)"
                    },
                },
                ctx: AttnCtx {
                    op: "paged_decode_nvfp4_split",
                    stream,
                    num_seqs: params.num_seqs,
                    head_dim: params.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            })?;
            let reduce_fn = if output_bf16 {
                fa2.fn_paged_attn_reduce_bf16
            } else {
                fa2.fn_paged_attn_reduce_f16
            }
            .ok_or_else(|| RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "Fa2Ptx",
                    op: if output_bf16 {
                        "paged_decode_nvfp4_split (bf16 reduce kernel missing)"
                    } else {
                        "paged_decode_nvfp4_split (reduce kernel missing)"
                    },
                },
                ctx: AttnCtx {
                    op: "paged_decode_nvfp4_split",
                    stream,
                    num_seqs: params.num_seqs,
                    head_dim: params.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            })?;

            let fa2_bc: i32 = if hd > 256 { 16 } else { 32 };
            // Phase-1 smem: 2 * BC * D (f16 K+V) + BC * 4 (scores) +
            // (FA2_THREADS/32) * 4 (reduce scratch).
            let split_smem =
                2 * fa2_bc * hd * 2 + fa2_bc * 4 + (FA2_THREADS / 32) * 4;
            if split_smem as u32 >= 48 * 1024 {
                let _ = cuFuncSetAttribute(
                    split_fn.raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    split_smem,
                );
            }
            // Phase-2 smem: 2 * max_num_partitions * 4 (max_logits +
            // rescaled exp_sums) + (REDUCE_THREADS/32) * 4 (reduce scratch).
            let reduce_smem =
                (2 * max_num_partitions as i32) * 4 + (REDUCE_THREADS / 32) * 4;
            if reduce_smem as u32 >= 48 * 1024 {
                let _ = cuFuncSetAttribute(
                    reduce_fn.raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    reduce_smem,
                );
            }

            // Workspace layout inside `workspace_ptr`:
            //   [0,                       tmp_bytes)    — tmp_out f32 [S,H,P,D]
            //   [tmp_bytes,               +meta_bytes)  — max_logits f32 [S,H,P]
            //   [tmp_bytes+meta_bytes,    +meta_bytes)  — exp_sums   f32 [S,H,P]
            // tmp_out widened from f16 to f32 in cycle21 (codex diag): the
            // f16 round-trip between phase-1 normalize and phase-2 reduce
            // was the main precision regression vs single-CTA decode — see
            // kernels/flash_attention_split_decode_nvfp4kv.cu header.
            let slots = (params.num_seqs as u64)
                * (params.num_heads as u64)
                * (max_num_partitions as u64);
            let tmp_bytes  = slots * (params.head_dim as u64) * 4;
            let meta_bytes = slots * 4;
            let d_tmp_out    = workspace_ptr;
            let d_max_logits = workspace_ptr + tmp_bytes;
            let d_exp_sums   = workspace_ptr + tmp_bytes + meta_bytes;

            let scale = params.scale;
            let num_heads         = params.num_heads as i32;
            let num_kv_heads      = params.num_kv_heads as i32;
            let head_dim          = params.head_dim as i32;
            let block_size        = params.block_size as i32;
            let max_blocks_per_seq = params.max_blocks_per_seq as i32;
            let window_size_left  = params.window_size_left;
            let partition_size_i  = partition_size as i32;
            let max_parts_i       = max_num_partitions as i32;

            // --- Phase 1: split kernel ---
            let mut a_tmp   = d_tmp_out;
            let mut a_maxl  = d_max_logits;
            let mut a_esum  = d_exp_sums;
            let mut a_q     = q_fp8;
            let mut a_kp    = k_cache_packed;
            let mut a_vp    = v_cache_packed;
            let mut a_ks    = k_cache_scale;
            let mut a_vs    = v_cache_scale;
            // === DYNAMIC NVFP4 Q SCALE ===
            let mut a_qsc   = q_scale_cache;
            // === END DYNAMIC NVFP4 Q SCALE ===
            let mut a_bt    = block_tables;
            let mut a_cl    = context_lens;
            let mut a_qd    = q_descale_ptr;

            let split_args: [*mut core::ffi::c_void; 21] = [
                &mut a_tmp  as *mut _ as *mut _,
                &mut a_maxl as *mut _ as *mut _,
                &mut a_esum as *mut _ as *mut _,
                &mut a_q    as *mut _ as *mut _,
                &mut a_kp   as *mut _ as *mut _,
                &mut a_vp   as *mut _ as *mut _,
                &mut a_ks   as *mut _ as *mut _,
                &mut a_vs   as *mut _ as *mut _,
                // === DYNAMIC NVFP4 Q SCALE ===
                &mut a_qsc  as *mut _ as *mut _,
                // === END DYNAMIC NVFP4 Q SCALE ===
                &mut a_bt   as *mut _ as *mut _,
                &mut a_cl   as *mut _ as *mut _,
                &mut a_qd   as *mut _ as *mut _,
                &scale              as *const _ as *mut _,
                &num_heads          as *const _ as *mut _,
                &num_kv_heads       as *const _ as *mut _,
                &head_dim           as *const _ as *mut _,
                &block_size         as *const _ as *mut _,
                &max_blocks_per_seq as *const _ as *mut _,
                &window_size_left   as *const _ as *mut _,
                &partition_size_i   as *const _ as *mut _,
                &max_parts_i        as *const _ as *mut _,
            ];

            let rc = cuLaunchKernel(
                split_fn.raw() as CUfunction,
                params.num_seqs as u32,
                params.num_heads as u32,
                max_num_partitions,
                FA2_THREADS as u32,
                1,
                1,
                split_smem as u32,
                stream as CUstream,
                split_args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::Attention {
                    err: AttentionError::KernelLaunchFailed {
                        cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                    },
                    ctx: AttnCtx {
                        op: "paged_decode_nvfp4_split phase-1",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }

            // --- Phase 2: reduce ---
            let mut r_out = o_f16;
            let mut r_tmp = d_tmp_out;
            let mut r_mxl = d_max_logits;
            let mut r_esm = d_exp_sums;
            let mut r_cl  = context_lens;
            let reduce_args: [*mut core::ffi::c_void; 9] = [
                &mut r_out as *mut _ as *mut _,
                &mut r_tmp as *mut _ as *mut _,
                &mut r_mxl as *mut _ as *mut _,
                &mut r_esm as *mut _ as *mut _,
                &mut r_cl  as *mut _ as *mut _,
                &num_heads          as *const _ as *mut _,
                &head_dim           as *const _ as *mut _,
                &max_parts_i        as *const _ as *mut _,
                &partition_size_i   as *const _ as *mut _,
            ];
            let rc = cuLaunchKernel(
                reduce_fn.raw() as CUfunction,
                params.num_seqs as u32,
                params.num_heads as u32,
                1,
                REDUCE_THREADS as u32,
                1,
                1,
                reduce_smem as u32,
                stream as CUstream,
                reduce_args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::Attention {
                    err: AttentionError::KernelLaunchFailed {
                        cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                    },
                    ctx: AttnCtx {
                        op: "paged_decode_nvfp4_split phase-2 reduce",
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
                     context_lens, q_descale_ptr, workspace_ptr,
                     partition_size, max_num_partitions, stream);
        }
        Ok(())
    }

    /// Returns true iff the PTX module exposes the split + reduce
    /// kernels. Callers use this to decide whether to attempt the
    /// split path vs the single-CTA fallback.
    #[cfg(feature = "cuda")]
    pub fn has_split_kernels(&self) -> bool {
        match self.backend {
            super::AttentionBackend::Fa2Ptx(fa2) => {
                fa2.fn_decode_nvfp4kv_split.is_some()
                    && fa2.fn_decode_nvfp4kv_split_bc16.is_some()
                    && fa2.fn_paged_attn_reduce_f16.is_some()
            }
            _ => false,
        }
    }
    #[cfg(not(feature = "cuda"))]
    pub fn has_split_kernels(&self) -> bool { false }
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
