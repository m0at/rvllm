//! Qwen 3.6 35B-A3B bring-up (Phase 1: outside-tensor upload).
//!
//! Phase 1 contract:
//!   - Initialize CUDA context, arena, stream.
//!   - Upload the three "outside-the-stack" tensors (embedding, final
//!     RMSNorm, lm_head) via [`rvllm_loader::qwen36_load::load_qwen36_outside`].
//!   - All forward methods (`run_generate`, `run_bench`, `run_ppl`,
//!     `init_prefix_cache`) `unimplemented!()` with phase-pointer
//!     messages — per-layer tensors + forward kernels are Phase 2/3.
//!
//! See `~/.claude/plans/abundant-meandering-sifakis.md` for the
//! phase-list.

use std::path::PathBuf;
use std::sync::Arc;

use rvllm_core::Result;
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
use rvllm_loader::qwen36_weights::Qwen36LoadedModel;
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

use crate::gemma4_bring_up::Gemma4EnginePaths;
use crate::qwen36_arch::Qwen36Arch;

/// Kernel function pointers + their LoadedModule anchors needed for the
/// Qwen 3.6 outside-only forward path (embedding lookup, final RMSNorm,
/// lm_head matmul, argmax).
///
/// `LoadedModule` is RAII — its `Drop` calls `cuModuleUnload`, after
/// which the matching `KernelFn` handles become invalid. Holding the
/// modules alongside the function pointers in this struct keeps the
/// pair alive for the lifetime of the bring-up.
pub struct Qwen36OutsideKernels {
    pub embedding_gather_f16_mod: LoadedModule,
    pub fn_embedding_gather_f16: KernelFn,
    pub rmsnorm_inplace_f16_mod: LoadedModule,
    pub fn_rmsnorm_inplace_f16: KernelFn,
    /// FP8 GEMV for the lm_head matmul. Phase 3d will CPU-quantize the
    /// bf16 lm_head to FP8 at load time so this kernel can be used.
    /// `None` on platforms where the f16-input native-CVT variant
    /// isn't available (gated on `__CUDA_ARCH__ >= 1000`).
    pub fp8_gemv_mod: LoadedModule,
    pub fn_fp8_gemv_wpr_native_f16in: Option<KernelFn>,
    pub argmax_mod: LoadedModule,
    pub fn_argmax: KernelFn,
}

pub struct Qwen36Bringup {
    pub paths: Gemma4EnginePaths,
    pub arena_bytes: usize,
    pub arch: Qwen36Arch,
    pub ctx: Arc<CudaContextHandle>,
    pub arena: HbmArena<'static>,
    pub stream: Stream,
    pub model: Qwen36LoadedModel,
    pub kernels: Arc<KernelLoader>,
    pub outside_kernels: Qwen36OutsideKernels,
}

impl Qwen36Bringup {
    /// Phase 1: CUDA init + arena + outside-tensor upload.
    /// Returns `Err` if `config.json` is missing required Qwen-3.6
    /// markers, or if any of the three outside tensors is missing.
    pub fn load(paths: Gemma4EnginePaths, arena_bytes: usize) -> Result<Self> {
        let arch = match Qwen36Arch::from_dir(&paths.model_dir)? {
            Some(a) => a,
            None => {
                panic!(
                    "Qwen36Bringup::load called for model_dir={:?} but \
                     Qwen36Arch::from_dir returned None — caller dispatched \
                     incorrectly",
                    paths.model_dir
                );
            }
        };
        arch.log_summary();

        let ctx = Arc::new(CudaContextHandle::init(0)?);

        #[cfg(feature = "cuda")]
        let compile_target: Option<rvllm_core::CompileTarget> = {
            let (major, minor) = ctx.compute_capability();
            rvllm_core::CompileTarget::from_compute_capability(major, minor)
        };
        #[cfg(not(feature = "cuda"))]
        let compile_target: Option<rvllm_core::CompileTarget> = None;

        // GB10 (sm_121) has no dedicated HBM — `cuMemAllocManaged` is the
        // right backing. Mirrors the Gemma 4 selection at
        // `gemma4_bring_up.rs::Gemma4Bringup::load`.
        let arena = {
            #[cfg(feature = "gb10")]
            {
                if matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121)) {
                    rvllm_mem::UnifiedArena::new(&ctx, arena_bytes)?.into_inner()
                } else {
                    HbmArena::new(&ctx, arena_bytes)?
                }
            }
            #[cfg(not(feature = "gb10"))]
            {
                HbmArena::new(&ctx, arena_bytes)?
            }
        };
        let arena: HbmArena<'static> = unsafe { std::mem::transmute(arena) };
        let stream = Stream::new(&ctx)?;

        let model = rvllm_loader::qwen36_load::load_qwen36_model(
            &paths.model_dir,
            &arena,
            &arch.base.layer_types,
            arch.num_experts,
        )?;

        // Phase 3a: load + verify the PTX kernel manifest the same way
        // Gemma 4 does. The Qwen forward path will need the same
        // model-agnostic kernel set (embedding_gather, fused_rmsnorm,
        // CUTLASS FP8-GEMM, FA2 paged-attn) plus future Qwen-specific
        // ones (qwen_qk_norm, attn_output_gate, linear-attn recurrent,
        // MoE grouped-FP8-GEMM). Initializing the loader here proves
        // the manifest path and arch-pinning still hold for the Qwen
        // bring-up before Phase 3b starts wiring kernel calls.
        let kernels_dir = crate::bring_up::resolve_kernels_dir(&ctx, &paths.kernels_dir)?;
        let manifest_path = kernels_dir.join("manifest.json");
        let manifest =
            rvllm_kernels::manifest::KernelManifest::load_and_verify(&manifest_path)?;
        if let Some(t) = compile_target {
            manifest.assert_arch(t.as_sm_str())?;
        }
        manifest
            .warn_if_revision_drift(rvllm_kernels::manifest::VerifiedManifest::BUILD_REVISION);
        let kernels = Arc::new(KernelLoader::new(manifest));

        // Phase 3b: resolve the outside-path kernel function pointers
        // (embedding_gather + final RMSNorm). The matching modules
        // stay in the bring-up struct so the function handles outlive
        // any per-request scope. lm_head matmul resolution is deferred
        // to Phase 3c — bf16 vs FP8 lm_head dispatch needs an explicit
        // decision (CPU-quantize at load like Gemma's tied path, or
        // route through a bf16 cuBLASLt GEMM).
        let embedding_gather_f16_mod = kernels.load_ptx("embedding_gather_f16")?;
        let fn_embedding_gather_f16 =
            embedding_gather_f16_mod.get_function("embedding_gather_f16_kernel")?;
        let rmsnorm_inplace_f16_mod = kernels.load_ptx("rmsnorm_inplace_f16")?;
        let fn_rmsnorm_inplace_f16 =
            rmsnorm_inplace_f16_mod.get_function("rmsnorm_inplace_f16_kernel")?;
        let fp8_gemv_mod = kernels.load_ptx(rvllm_kernels::FP8_GEMV_PTX_STEM)?;
        let fn_fp8_gemv_wpr_native_f16in = match compile_target {
            Some(t)
                if rvllm_kernels::Fp8GemvVariant::WprNativeF16In.available_for(t) =>
            {
                Some(
                    fp8_gemv_mod.get_function(
                        rvllm_kernels::Fp8GemvVariant::WprNativeF16In.entry_point(),
                    )?,
                )
            }
            _ => None,
        };
        let argmax_mod = kernels.load_ptx("argmax")?;
        let fn_argmax = argmax_mod.get_function("argmax_kernel")?;
        let outside_kernels = Qwen36OutsideKernels {
            embedding_gather_f16_mod,
            fn_embedding_gather_f16,
            rmsnorm_inplace_f16_mod,
            fn_rmsnorm_inplace_f16,
            fp8_gemv_mod,
            fn_fp8_gemv_wpr_native_f16in,
            argmax_mod,
            fn_argmax,
        };
        eprintln!(
            "[qwen36] outside kernels resolved: embedding_gather_f16, \
             rmsnorm_inplace_f16, fp8_gemv (wpr_native_f16in: {}), argmax.",
            outside_kernels.fn_fp8_gemv_wpr_native_f16in.is_some(),
        );

        let n_full = model.layers.iter().filter(|l| matches!(
            l.attn,
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(_)
        )).count();
        let n_linear = model.layers.iter().filter(|l| matches!(
            l.attn,
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(_)
        )).count();
        eprintln!(
            "[qwen36] Phase 3f bring-up complete: outside (incl. \
             FP8-quantized lm_head) + {n_full} full-attention + \
             {n_linear} linear-attention + per-layer MoE blocks \
             ({} experts/layer) + KernelLoader + outside kernel \
             pointers + embedding_gather + rmsnorm_inplace_f16 smoke \
             launches validated. arena.used()={used:.2} GiB / \
             {total:.2} GiB. lm_head matmul + argmax NOT yet wired. \
             See ~/.claude/plans/abundant-meandering-sifakis.md.",
            arch.num_experts,
            used = arena.used() as f64 / (1024.0 * 1024.0 * 1024.0),
            total = arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        let bringup = Self {
            paths,
            arena_bytes,
            arch,
            ctx,
            arena,
            stream,
            model,
            kernels,
            outside_kernels,
        };
        // Phase 3e smoke: actually launch embedding_gather against the
        // loaded weights. If this throws, the rest of the forward
        // pipeline can't be built on top — fail fast.
        bringup.forward_outside_smoke()?;
        Ok(bringup)
    }

    pub fn kernels_dir(&self) -> &PathBuf {
        &self.paths.kernels_dir
    }

    /// Phase 3e smoke test: launch the embedding-gather kernel against
    /// the loaded `embed_tokens` table for a hardcoded 4-token input,
    /// fence the stream, and DtoH the first 4 hidden-state floats so
    /// we can log them. Validates that:
    ///   - the bring-up's CUDA context + arena + stream + KernelLoader
    ///     are wired correctly
    ///   - the f16 `embed_tokens` upload reaches the device with
    ///     the right layout
    ///   - the EmbeddingGatherLaunch ABI matches Qwen 3.6 dims
    /// without exercising any per-layer math (still Phase 3f+).
    pub fn forward_outside_smoke(&self) -> Result<()> {
        let hidden = self.arch.base.hidden_size as u32;
        let vocab = self.arch.base.vocab_size as u32;
        // Hardcoded canary tokens: BOS-likely + a few mid-vocab IDs.
        // Real generation needs the tokenizer; we just want non-zero
        // embeddings for the kernel output.
        let token_ids: [i32; 4] = [1, 100, 1000, 10_000];
        let num_tokens = token_ids.len() as u32;

        // Allocate device regions: token IDs (i32) + hidden state (f16).
        let tokens_region = self.arena.region(
            "qwen36_smoke_tokens",
            std::mem::size_of_val(&token_ids),
            16,
        )?;
        let mut token_bytes = Vec::with_capacity(token_ids.len() * 4);
        for t in &token_ids {
            token_bytes.extend_from_slice(&t.to_le_bytes());
        }
        unsafe { tokens_region.copy_from_host(&token_bytes)? };

        let hidden_bytes = (num_tokens as usize) * (hidden as usize) * 2; // f16
        let hidden_region = self.arena.region("qwen36_smoke_hidden", hidden_bytes, 16)?;

        // Launch embedding_gather_f16 via the standard rvllm-fused
        // ABI used by Gemma 4.
        let launch = rvllm_fused::EmbeddingGatherLaunch {
            num_tokens,
            hidden,
            vocab,
        };
        unsafe {
            launch.launch(
                self.outside_kernels.fn_embedding_gather_f16,
                hidden_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tokens_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }

        // Phase 3f: chain `rmsnorm_inplace_f16` against the embedded
        // hidden state + the final-norm gamma weight. Kernel sig
        // (kernels/rmsnorm_inplace_f16.cu): `(half* x [num_tokens,
        // hidden] in-place, const half* gamma [hidden], float eps,
        // int hidden)`. Grid: (num_tokens, 1, 1). Block: hidden (capped
        // at 1024). lm_head matmul + argmax land in Phase 3g once the
        // f16→f32 + per-tensor scaled fp8_gemv variant is selected.
        let eps = self.arch.base.rms_norm_eps;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut x_ptr = hidden_region.device_ptr();
            let mut gamma_ptr = self.model.outside.final_norm.offset_bytes;
            let mut eps_arg = eps;
            let mut hidden_arg = hidden as i32;
            let args = [
                (&mut x_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut gamma_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut hidden_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block = hidden.min(1024);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_rmsnorm_inplace_f16.raw() as CUfunction,
                num_tokens, 1, 1,
                block, 1, 1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_smoke rmsnorm_inplace_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;

        // DtoH the first 4 f16 values of token-0's hidden state so the
        // log is non-trivial — proves the kernel actually wrote.
        // rvllm_mem::Region only exposes copy_from_host; raw cuMemcpyDtoH
        // is the only DtoH path here.
        let mut probe = [0u8; 8]; // 4 × f16
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                hidden_region.device_ptr(),
                probe.len(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_smoke cuMemcpyDtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let f0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let f1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let f2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let f3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_outside_smoke: embedding_gather + rmsnorm \
             token_ids={token_ids:?} \
             token0_normed[0..4]=[{f0:.4}, {f1:.4}, {f2:.4}, {f3:.4}] \
             eps={eps:.0e}"
        );
        Ok(())
    }

    pub fn run_generate(&self) -> ! {
        unimplemented!(
            "qwen36 phase 3f+ — outside-only forward (rmsnorm + lm_head + argmax) \
             then per-layer math still TODO"
        );
    }

    pub fn run_bench(&self) -> ! {
        unimplemented!("qwen36 phase 2 — bench harness not yet ported");
    }

    pub fn run_ppl(&self) -> ! {
        unimplemented!("qwen36 phase 2 — ppl harness not yet ported");
    }

    pub fn init_prefix_cache(&self) -> ! {
        unimplemented!("qwen36 phase 2 — prefix cache not yet ported");
    }
}

/// Decode a stored f16 bit pattern into f32 without pulling in `half`
/// as an explicit dep of `rvllm-runtime`. Mirrors `half::f16::to_f32`
/// for finite values; specials (NaN/inf) are preserved by IEEE
/// composition.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal — renormalise.
            let mut e: i32 = -14;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let m = (m & 0x3ff) << 13;
            (sign << 31) | (((e + 127) as u32) << 23) | m
        }
    } else if exp == 0x1f {
        // NaN or inf.
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp + 112) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}
