//! CUDA-backed worker. Feature-gated (`cuda` / `gb10`).
//!
//! Phase 2 scope: wrap the existing monolithic
//! `Gemma4Bringup::run_generate` synchronously. All generated tokens
//! are emitted to the `events_tx` channel **after** generation
//! completes — not token-by-token on the GPU's pace. True per-token
//! streaming is phase 5 (requires breaking `run_generate` apart into
//! `prefill()` + `decode_one()`).
//!
//! ## RAII gotcha fixed here
//!
//! `LoadedModule` is RAII: its `Drop` calls `cuModuleUnload`. A
//! `KernelFn` is only an opaque handle into that module. If the
//! module drops while the handle lives, the next `cuLaunchKernel`
//! on the handle fails with `LaunchFailed`. [`GenerateKernels`]
//! holds the module alongside the fn-handles to anchor its lifetime
//! — the first Gemma 4 chat request on GB10 died with exactly that
//! error before this fix.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use tokio::sync::mpsc;
use tokio::sync::oneshot;

use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};

use crate::config::ModelFamily;
use crate::error::ApiError;
use crate::openai::types::FinishReason;
use crate::worker::{GenerateEvent, GenerateRequest, WorkerHandle};

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

/// Minimum config needed to bring up the CUDA worker. Mirrors the
/// `probe-gemma4-load` flags so operators can move directly from
/// probe to serve.
///
/// `Gemma4EnginePaths` is neither `Debug` nor `Clone` upstream, so
/// this wrapper is not either.
pub struct CudaWorkerConfig {
    pub paths: Gemma4EnginePaths,
    pub arena_bytes: usize,
    pub queue_depth: usize,
    /// Resolved model family. Drives explicit dispatch in
    /// [`spawn_cuda_worker`]. With `Auto` the worker still does
    /// per-family marker probing as before, but `Mistral35` /
    /// `Qwen36` / `Gemma4` short-circuit to the matching bring-up
    /// without re-probing.
    pub family: ModelFamily,
}

/// Spawn the CUDA worker on a dedicated OS thread.
///
/// Returns once `Gemma4Bringup::load` has completed — the returned
/// `WorkerHandle` is immediately usable. `load` takes ~25 s for
/// Gemma 4 31B; the caller should render a "loading" log line before
/// awaiting.
pub async fn spawn_cuda_worker(
    cfg: CudaWorkerConfig,
) -> Result<(WorkerHandle, std::thread::JoinHandle<()>), ApiError> {
    // Channel buffer must equal admission-permit count. The earlier
    // arithmetic (`queue_depth - 1` plus an "in-flight slot on the
    // worker") was wrong: the worker only frees a buffer slot when
    // it pulls from the channel, so until that recv returns, the
    // buffer IS the cap. With `queue_depth` permits but a
    // `queue_depth - 1` buffer, a cold-burst of `queue_depth`
    // concurrent admissions could see one handler's `try_send` fail
    // with Busy after fetch+tokenize work — violating the "permit
    // reserves the lifecycle slot" contract. Clamp to >= 1 because
    // `mpsc::channel(0)` panics.
    let channel_buf = cfg.queue_depth.max(1);
    let (req_tx, mut req_rx) = mpsc::channel::<GenerateRequest>(channel_buf);
    let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();

    let CudaWorkerConfig { paths, arena_bytes, family, .. } = cfg;
    let join = std::thread::Builder::new()
        .name("rvllm-serve-cuda-worker".into())
        .spawn(move || {
            // Mistral 3.5: parse arch + validate inventory + assert
            // CUTLASS NVFP4 symbols present, then refuse per-request
            // generation cleanly until the GPU forward path is wired.
            // This covers steps 1-4 + 7 partial — the operator gets
            // concrete startup diagnostics on a real Mistral
            // checkpoint, and per-request errors carry the same
            // "kernel not implemented" reason instead of corrupting
            // arena state.
            if matches!(family, ModelFamily::Mistral35) {
                let bringup = match rvllm_runtime::mistral35_bring_up::Mistral35Bringup::load(
                    paths,
                    arena_bytes,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!(
                            "Mistral35Bringup::load: {e:?}"
                        )));
                        return;
                    }
                };
                tracing::info!(
                    nvfp4_active = bringup.nvfp4_active,
                    "Mistral 3.5 bring-up validated; forward path not yet implemented"
                );
                let _ = ready_tx.send(Ok(()));

                while let Some(req) = req_rx.blocking_recv() {
                    // Smoke-forward path (Step 5 GPU half, iteration A):
                    // run embed → first-layer RMSNorm → first-layer
                    // Q-projection on token id 42 against the resident
                    // model. Logs first 16 BF16-as-F32 channels +
                    // basic plausibility stats, then returns a typed
                    // "smoke-only" error so the per-request pipeline
                    // reports concrete numerics instead of garbage.
                    let token_id: u32 = 42;
                    let smoke = unsafe { bringup.forward_smoke_q_proj(token_id) };
                    let summary = match smoke {
                        Ok(q) => {
                            let n = q.len();
                            let any_nan = q.iter().any(|v| v.is_nan());
                            let any_inf = q.iter().any(|v| v.is_infinite());
                            let mut min = f32::INFINITY;
                            let mut max = f32::NEG_INFINITY;
                            let mut nz = 0usize;
                            let mut sumsq = 0.0f64;
                            for &v in &q {
                                if v < min { min = v; }
                                if v > max { max = v; }
                                if v != 0.0 { nz += 1; }
                                sumsq += (v as f64) * (v as f64);
                            }
                            let rms = (sumsq / (n as f64)).sqrt();
                            let head: Vec<String> = q.iter().take(16)
                                .map(|v| format!("{:+.4}", v)).collect();
                            tracing::info!(
                                token_id, n, any_nan, any_inf, nz,
                                min, max, rms,
                                "mistral35 smoke Q-proj head[16]={}",
                                head.join(",")
                            );
                            format!(
                                "smoke Q-proj OK: token={token_id} \
                                 n={n} nan={any_nan} inf={any_inf} \
                                 nz={nz} min={min:+.4} max={max:+.4} \
                                 rms={rms:.4} head=[{}]",
                                head.join(",")
                            )
                        }
                        Err(e) => format!("smoke Q-proj FAILED: {e:?}"),
                    };
                    let _ = req.events_tx.send(GenerateEvent::Error(format!(
                        "Mistral 3.5 forward path: smoke-only iteration A \
                         (embed + first-layer RMSNorm + Q-projection NVFP4 \
                         GEMM); full layer + sampling pending. \
                         nvfp4_active={}, {}",
                        bringup.nvfp4_active, summary,
                    )));
                }
                tracing::info!("Mistral 3.5 cuda worker queue closed, exiting");
                return;
            }

            // Qwen 3.6 detection — under explicit `--model-family`,
            // skip the probe; under Auto fall back to the marker
            // probe. `Gemma4` selection short-circuits straight to the
            // Gemma 4 loader.
            let qwen_probe = match family {
                ModelFamily::Gemma4 => Ok(None),
                ModelFamily::Qwen36 => {
                    rvllm_runtime::qwen36_arch::Qwen36Arch::from_dir(&paths.model_dir)
                        .and_then(|opt| match opt {
                            Some(a) => Ok(Some(a)),
                            None => {
                                use rvllm_core::{LoaderCtx, LoaderError, RvllmError};
                                Err(RvllmError::Loader {
                                    err: LoaderError::Corrupt {
                                        detail:
                                            "operator forced --model-family=qwen36 but config.json \
                                             does not match the Qwen 3.6 markers"
                                                .into(),
                                    },
                                    ctx: LoaderCtx {
                                        path: paths.model_dir.join("config.json"),
                                        tensor: None,
                                    },
                                    bt: std::backtrace::Backtrace::capture(),
                                })
                            }
                        })
                }
                ModelFamily::Auto | ModelFamily::Mistral35 => {
                    rvllm_runtime::qwen36_arch::Qwen36Arch::from_dir(&paths.model_dir)
                }
            };
            match qwen_probe {
                Ok(Some(_)) => {
                    let qwen = match rvllm_runtime::qwen36_bring_up::Qwen36Bringup::load(
                        paths,
                        arena_bytes,
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            let _ = ready_tx.send(Err(format!(
                                "Qwen36Bringup::load: {e:?}"
                            )));
                            return;
                        }
                    };
                    let scratch_ck = qwen.arena.checkpoint();
                    tracing::info!(
                        "qwen36 cuda worker ready (Phase 5d: \
                         linear-attn layout rewritten per vLLM \
                         qwen3_next reference — Q[16,128]+K[16,128]+ \
                         V[32,128] split, L2-norm on Q/K, in_proj_a/b \
                         for α/β, GQA expansion for state update)"
                    );
                    let _ = ready_tx.send(Ok(()));

                    // Phase 5c request loop: prefill the prompt
                    // (state + KV accumulate causally), then decode
                    // step-by-step up to max_new_tokens, with EOS /
                    // stop-token early-exit. arena.restore between
                    // requests keeps memory bounded; the persistent
                    // linear-state + KV cache regions live ABOVE the
                    // scratch checkpoint so they survive that restore
                    // and only get reset by the explicit calls below.
                    while let Some(req) = req_rx.blocking_recv() {
                        // GB10 belt-and-suspenders: rebind the retained
                        // primary CUDA context to this worker thread once
                        // per request. `Qwen36Bringup::load` already
                        // bound it during init, but a context binding that
                        // sat idle across `blocking_recv` has been
                        // observed to drop on GB10, producing
                        // `cuLaunchKernel` failures on the next launch.
                        // One `cuCtxSetCurrent` per request is free next
                        // to decode cost. See
                        // rvllm-mem/src/context.rs:bind_to_current_thread.
                        if let Err(e) = qwen.ctx.bind_to_current_thread() {
                            let _ = req.events_tx.send(GenerateEvent::Error(
                                format!("qwen36 ctx rebind: {e:?}"),
                            ));
                            continue;
                        }
                        let prompt_len = req.prompt_ids.len() as u32;
                        // Qwen 3.6 path is greedy-only today: the
                        // forward_qwen36_decode runtime samples internally
                        // (returns the picked token, not logits). We reject
                        // non-greedy sampling explicitly so callers can't
                        // believe their `temperature`/`top_p`/`top_k`/`seed`
                        // were honoured. Lift the rejection when Qwen's
                        // bring-up grows a logits-out variant + sampler.
                        if !req.sampling.is_greedy() {
                            // Note: the absent-temperature default now
                            // resolves to 0.0 (greedy) globally; if the
                            // client lands here they explicitly asked
                            // for stochastic. The old hint mentioned
                            // "omit sampling params" — that's exactly
                            // the path that USED to fail; corrected.
                            let _ = req.events_tx.send(GenerateEvent::Error(
                                "qwen36 path: non-greedy sampling \
                                 (temperature>0 / top_p<1 / top_k / seed) \
                                 is not yet supported on Qwen 3.6. Set \
                                 temperature=0 explicitly, or omit it to \
                                 take the (now greedy) default."
                                    .to_string(),
                            ));
                            continue;
                        }
                        // Reset per-request transient state. cuMemsetD8Async
                        // can return real CUDA errors (e.g. context lost
                        // after a kernel fault); silently swallowing them
                        // would make the next request run on stale state.
                        let reset_ok = qwen.reset_linear_state()
                            .and_then(|_| qwen.reset_kv_cache())
                            .and_then(|_| qwen.reset_conv_state());
                        if let Err(e) = reset_ok {
                            let _ = req.events_tx.send(GenerateEvent::Error(
                                format!("qwen36 per-request reset: {e:?}"),
                            ));
                            unsafe { qwen.arena.restore(scratch_ck); }
                            continue;
                        }
                        if req.cancelled.load(Ordering::Relaxed) {
                            let _ = req.events_tx.send(GenerateEvent::Done {
                                finish: FinishReason::Cancelled,
                                prompt_tokens: prompt_len,
                                completion_tokens: 0,
                            });
                            unsafe { qwen.arena.restore(scratch_ck); }
                            continue;
                        }
                        let prompt_i32: Vec<i32> = req
                            .prompt_ids
                            .iter()
                            .map(|&t| t as i32)
                            .collect();

                        // Vision pre-pass: run native ViT forward on
                        // each image. Outputs accumulate per slot for
                        // splicing during the prefill embed step.
                        let mut vision_outputs: Vec<rvllm_runtime::qwen36_bring_up::VisionForwardOutput> =
                            Vec::with_capacity(req.vision_items.len());
                        let mut vision_failed = false;
                        for (i, item) in req.vision_items.iter().enumerate() {
                            if req.cancelled.load(Ordering::Relaxed) {
                                vision_failed = true;
                                break;
                            }
                            match qwen.forward_qwen_vision(&item.bytes) {
                                Ok(out) => {
                                    if out.num_tokens != item.num_tokens {
                                        let _ = req.events_tx.send(
                                            GenerateEvent::Error(format!(
                                                "vision tokens mismatch: predicted {} got {}",
                                                item.num_tokens, out.num_tokens
                                            )),
                                        );
                                        vision_failed = true;
                                        break;
                                    }
                                    tracing::info!(
                                        idx = i,
                                        tokens = out.num_tokens,
                                        hidden = out.hidden_dim,
                                        "vision: ViT forward done"
                                    );
                                    vision_outputs.push(out);
                                }
                                Err(e) => {
                                    let _ = req.events_tx.send(GenerateEvent::Error(
                                        format!("vision forward: {e:?}"),
                                    ));
                                    vision_failed = true;
                                    break;
                                }
                            }
                        }
                        if vision_failed {
                            unsafe { qwen.arena.restore(scratch_ck); }
                            continue;
                        }

                        // Prefill: feed full prompt at start_position=0.
                        // For each vision slot, build (token_start,
                        // embedding bytes) tuple — the splice happens
                        // inside forward_qwen36_decode after embed_gather.
                        let vision_splice: Vec<(usize, &[u8])> = req
                            .vision_slots
                            .iter()
                            .map(|s| {
                                (s.token_start, vision_outputs[s.vision_item_idx].data.as_slice())
                            })
                            .collect();
                        let timing_on = std::env::var("RVLLM_QWEN36_TIMING")
                            .map(|s| matches!(s.as_str(), "1"|"true"|"TRUE"|"yes"))
                            .unwrap_or(false);
                        let prefill_start = if timing_on {
                            Some(std::time::Instant::now())
                        } else { None };
                        let mut next_token = match qwen
                            .forward_qwen36_decode_cancellable(
                                &prompt_i32, 0, &vision_splice,
                                Some(&*req.cancelled),
                            )
                        {
                            Ok(t) => t,
                            Err(e) => {
                                let _ = req.events_tx.send(GenerateEvent::Error(
                                    format!("qwen36 prefill: {e:?}"),
                                ));
                                unsafe { qwen.arena.restore(scratch_ck); }
                                continue;
                            }
                        };
                        if let Some(t0) = prefill_start {
                            let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            // Round-27d: gates are default-ON post-audit;
                            // log the RESOLVED state (env-var=0/false → off,
                            // anything else including "unset" → on) so the
                            // log reflects what the runtime actually
                            // dispatched, not just the literal env string.
                            let resolved = |name: &str| -> &'static str {
                                match std::env::var(name) {
                                    Ok(s) if matches!(s.as_str(),
                                        "0"|"false"|"FALSE"|"no") => "off",
                                    _ => "on",
                                }
                            };
                            tracing::info!(
                                prompt_tokens = prompt_len,
                                prefill_ms = format!("{dt_ms:.2}"),
                                batch_linear = resolved("RVLLM_QWEN36_BATCH_LINEAR_PREFILL"),
                                batch_full = resolved("RVLLM_QWEN36_BATCH_FULL_PREFILL"),
                                batch_moe = resolved("RVLLM_QWEN36_BATCH_MOE_PREFILL"),
                                batch_moe_routed_ffn = resolved("RVLLM_QWEN36_BATCH_MOE_ROUTED_FFN"),
                                batch_moe_shared = resolved("RVLLM_QWEN36_BATCH_MOE_SHARED"),
                                "[qwen36-timing] prefill done",
                            );
                        }
                        let mut completion_tokens: u32 = 0;
                        let mut finish = FinishReason::Length;
                        let max_new = req.max_new_tokens.max(1);
                        for step in 0..max_new {
                            let id = if next_token < 0 { 0u32 } else { next_token as u32 };
                            // Round-20 finding #2: stop-token check BEFORE
                            // emit + counter bump, so the EOS / stop
                            // token never reaches the SSE consumer and
                            // `completion_tokens` doesn't include it
                            // (mirrors mock-worker semantics; previously
                            // qwen leaked the stop token both ways).
                            if req.stop_token_ids.contains(&id) {
                                finish = FinishReason::Stop;
                                break;
                            }
                            let _ = req.events_tx.send(GenerateEvent::Token {
                                id,
                                position: prompt_len + step,
                            });
                            completion_tokens += 1;
                            if req.cancelled.load(Ordering::Relaxed) {
                                finish = FinishReason::Cancelled;
                                break;
                            }
                            if step + 1 >= max_new {
                                break;
                            }
                            // Decode step: feed just this token at
                            // start_position = prompt_len + step.
                            let pos = prompt_len + step;
                            match qwen.forward_qwen36_decode(&[next_token], pos, &[]) {
                                Ok(t) => next_token = t,
                                Err(e) => {
                                    let _ = req.events_tx.send(GenerateEvent::Error(
                                        format!("qwen36 decode step {step}: {e:?}"),
                                    ));
                                    finish = FinishReason::Stop;
                                    break;
                                }
                            }
                        }
                        let _ = req.events_tx.send(GenerateEvent::Done {
                            finish,
                            prompt_tokens: prompt_len,
                            completion_tokens,
                        });
                        unsafe { qwen.arena.restore(scratch_ck); }
                    }
                    tracing::info!("qwen36 cuda worker queue closed, exiting");
                    return;
                }
                Ok(None) => {
                    // Not Qwen 3.6 — fall through to Gemma 4 path.
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(format!(
                        "Qwen36Arch::from_dir: {e:?}"
                    )));
                    return;
                }
            }

            // Bring-up on the worker thread so Gemma4Bringup (which
            // contains !Send CUDA state like streams) never crosses
            // a thread boundary.
            let bringup = match Gemma4Bringup::load(paths, arena_bytes) {
                Ok(b) => b,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("Gemma4Bringup::load: {e:?}")));
                    return;
                }
            };

            // Resolve kernel function pointers once. See the struct
            // doc — the LoadedModule MUST outlive the KernelFn.
            let kernels_ctx = match resolve_generate_kernels(&bringup) {
                Ok(k) => k,
                Err(msg) => {
                    let _ = ready_tx.send(Err(msg));
                    return;
                }
            };

            // Snapshot the arena's bump pointer here — everything
            // after this point (scratch regions allocated inside
            // `run_generate`) will be released back to this mark at
            // the end of each request. Without this the bump pointer
            // grows monotonically across requests and we'd hit
            // `HbmArena::region AllocFailed` after a handful of
            // calls (each `run_generate` allocates ~a dozen named
            // scratch regions sized for max_tokens + KV cache).
            //
            // Initialise the session-level prefix cache BEFORE the
            // scratch checkpoint so the persistent KV region sits
            // below the checkpoint and survives every
            // `arena.restore(scratch_ck)` between requests. This is
            // what makes the MVP vLLM-style prefix reuse work on
            // zeroclaw's "identical 15k-token persona every request"
            // pattern.
            // === DIAGNOSTIC: RVLLM_DISABLE_PREFIX_CACHE gate ===
            // When set, skip prefix-cache initialization entirely so
            // every request goes through the per-call fallback KV
            // allocation path (full prefill, no cross-request reuse).
            // Used to discriminate "prefix-cache reuse causes the
            // R1≠R2 divergence" from "deeper non-determinism." Remove
            // after the prefix-cache hypothesis is settled.
            let disable_prefix_cache = env_truthy("RVLLM_DISABLE_PREFIX_CACHE");
            if disable_prefix_cache {
                tracing::warn!("RVLLM_DISABLE_PREFIX_CACHE=1 — skipping init_prefix_cache (diagnostic mode)");
            } else {
                if let Err(e) = bringup.init_prefix_cache() {
                    let _ = ready_tx.send(Err(format!("init_prefix_cache: {e:?}")));
                    return;
                }
            }
            // === END DIAGNOSTIC ===

            // Pre-allocate persistent NVFP4 helper buffers BEFORE the
            // scratch checkpoint. The lazy allocation paths inside
            // `run_generate` had a fatal lifetime bug: they ran
            // AFTER the checkpoint, so `arena.restore(scratch_ck)`
            // at the end of every request marked the buffers' bytes
            // as free, and the SAME pointer in
            // `Gemma4Bringup::nvfp4_hadamard` (or `_shadow`) on the
            // next request silently aliased fresh scratch
            // allocations. With `RVLLM_NVFP4_HADAMARD=1` enabled in
            // production this could corrupt every Nth request's
            // attention math without crashing — the worst kind of
            // failure mode in an inference server.
            //
            // `build_nvfp4_hadamard_signs` is a no-op (returns
            // `Ok(None)`) when the env gate is off, so adding the
            // pre-allocation step here is free for the default
            // configuration.
            {
                let max_hd = bringup.arch.max_head_dim() as u32;
                let nl = bringup.arch.num_hidden_layers as u32;
                match rvllm_runtime::gemma4_bring_up::build_nvfp4_hadamard_signs(
                    nl, max_hd, &bringup.arena,
                ) {
                    Ok(alloc) => {
                        if alloc.is_some() {
                            tracing::info!(
                                "nvfp4_hadamard signs pre-allocated above scratch checkpoint"
                            );
                        }
                        *bringup.nvfp4_hadamard.lock().unwrap() = alloc;
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!(
                            "build_nvfp4_hadamard_signs: {e:?}"
                        )));
                        return;
                    }
                }
            }

            // Pre-allocate the NVFP4 shadow KV / Q / throwaway regions
            // BEFORE the scratch checkpoint, mirroring the hadamard
            // pre-alloc above. Same lifetime contract — the lazy alloc
            // inside `run_generate` had a known silent-corruption bug
            // from request 2 onward (regions allocated below the
            // checkpoint were marked free by `arena.restore(scratch_ck)`
            // and aliased into fresh scratch on the next request). The
            // helper returns `None` when `RVLLM_NVFP4_SHADOW_LAYERS` is
            // unset, so this is a no-op on production where the
            // diagnostic is disabled.
            {
                const BLOCK_SIZE: u32 = 32;
                let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
                    .ok().and_then(|s| s.parse().ok())
                    .unwrap_or(1024);
                // Sliding layers share the same block budget on the
                // current code path; if that ever diverges, mirror
                // the runtime's calculation here.
                let sliding_blocks: u32 = num_blocks_total;
                match rvllm_runtime::gemma4_bring_up::build_nvfp4_shadow_alloc(
                    &bringup.arch,
                    num_blocks_total,
                    sliding_blocks,
                    BLOCK_SIZE,
                    &bringup.arena,
                ) {
                    Ok(alloc) => {
                        if alloc.is_some() {
                            tracing::info!(
                                "nvfp4_shadow regions pre-allocated above scratch checkpoint"
                            );
                        }
                        *bringup.nvfp4_shadow.lock().unwrap() = alloc;
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!(
                            "build_nvfp4_shadow_alloc: {e:?}"
                        )));
                        return;
                    }
                }
            }
            let scratch_ck = bringup.arena.checkpoint();
            tracing::info!(
                compute_cap = ?bringup.ctx.compute_capability(),
                arena_mib = bringup.arena.capacity() / (1024 * 1024),
                scratch_checkpoint = scratch_ck,
                "cuda worker ready",
            );
            let _ = ready_tx.send(Ok(()));

            // Main serve loop. One request at a time (single-seq).
            while let Some(req) = req_rx.blocking_recv() {
                // Same GB10 ctx-rebind guard as the qwen36 branch
                // (see rvllm-mem/src/context.rs::bind_to_current_thread).
                // One `cuCtxSetCurrent` per request is free; a dropped
                // binding after a long blocking_recv idle can produce
                // cuLaunchKernel failures on the next decode.
                if let Err(e) = bringup.ctx.bind_to_current_thread() {
                    let _ = req.events_tx.send(GenerateEvent::Error(
                        format!("gemma4 ctx rebind: {e:?}"),
                    ));
                    continue;
                }
                run_one(&bringup, &kernels_ctx, req);
                // SAFETY: `run_one` fully consumes the `Region`s it
                // allocated inside `bringup.run_generate` — they're
                // function-local there and drop before we return.
                // No region reference above the checkpoint survives,
                // so rewinding the bump pointer is safe.
                unsafe { bringup.arena.restore(scratch_ck); }
            }

            tracing::info!("cuda worker queue closed, exiting");
        })
        .map_err(|e| ApiError::Internal(format!("spawn cuda worker: {e}")))?;

    match ready_rx.await {
        Ok(Ok(())) => Ok((WorkerHandle::new(req_tx, cfg.queue_depth.max(1)), join)),
        Ok(Err(msg)) => Err(ApiError::Internal(msg)),
        Err(_) => Err(ApiError::Internal(
            "cuda worker thread exited before signalling ready".into(),
        )),
    }
}

/// Held together intentionally: `LoadedModule` is RAII, its `Drop`
/// calls `cuModuleUnload` — if we drop the module but keep the
/// `KernelFn` around, the next `cuLaunchKernel` dereferences a
/// freed handle and fails with `LaunchFailed` (how this bug first
/// surfaced on the GB10 live-test smoke). Struct owns the module
/// to keep it alive alongside the function handles.
struct GenerateKernels {
    fn_embed: rvllm_kernels::KernelFn,
    fn_argmax: rvllm_kernels::KernelFn,
    // Never read — its lifetime anchors the module so `fn_embed`
    // stays valid across run_generate calls.
    _embed_mod: rvllm_kernels::LoadedModule,
}

fn resolve_generate_kernels(
    bringup: &Gemma4Bringup,
) -> Result<GenerateKernels, String> {
    let embed_mod = bringup
        .kernels
        .load_ptx("embedding_gather_f16")
        .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
    let fn_embed = embed_mod
        .get_function("embedding_gather_f16_kernel")
        .map_err(|e| format!("resolve embedding_gather_f16_kernel: {e}"))?;
    let fn_argmax = bringup.fused.fn_argmax;
    Ok(GenerateKernels { fn_embed, fn_argmax, _embed_mod: embed_mod })
}

fn run_one(bringup: &Gemma4Bringup, kernels: &GenerateKernels, req: GenerateRequest) {
    let prompt_len = req.prompt_ids.len() as u32;

    if req.cancelled.load(Ordering::Relaxed) {
        let _ = req.events_tx.send(GenerateEvent::Done {
            finish: FinishReason::Cancelled,
            prompt_tokens: prompt_len,
            completion_tokens: 0,
        });
        return;
    }

    tracing::debug!(
        request_id = %req.request_id,
        prompt_tokens = req.prompt_ids.len(),
        max_new = req.max_new_tokens,
        "calling run_generate",
    );
    let sampling_cfg = match req.sampling {
        crate::sampling::SamplingDecision::Greedy => {
            rvllm_runtime::gemma4_bring_up::SamplingConfig::Greedy
        }
        crate::sampling::SamplingDecision::Stochastic(s) => {
            rvllm_runtime::gemma4_bring_up::SamplingConfig::Stochastic {
                temperature: s.temperature,
                top_p: s.top_p,
                top_k: s.top_k,
                seed: s.seed,
            }
        }
    };
    // Per-token streaming sink. Fires on every token the runtime
    // emits (prefill's first + every decode-loop token, in order).
    // Returning `false` stops generation — used for closed-channel
    // (client disconnect / handler exit) and for the existing
    // cancellation flag (timeout, stop-string match in handler,
    // SSE Drop). Counter `emitted` runs from 0 so SSE position
    // matches.
    //
    // This is the change that makes `stream=true` actually deliver
    // tokens incrementally instead of in a burst at the end of
    // run_generate, AND lets handler-side stop-string detection
    // cancel the worker within ~one decode step instead of after
    // all max_tokens are produced.
    let mut emitted: u32 = 0;
    let cancelled_ref = req.cancelled.clone();
    let events_tx_ref = req.events_tx.clone();
    let mut on_token = |id: u32| -> bool {
        if cancelled_ref.load(Ordering::Relaxed) {
            return false;
        }
        let pos = emitted;
        match events_tx_ref.send(GenerateEvent::Token { id, position: pos }) {
            Ok(()) => {
                emitted += 1;
                true
            }
            Err(_) => false,
        }
    };

    // Phase 3b: Gemma vision pre-pass. Run the ViT forward for each
    // image; collect device-side embeddings as raw f16 bytes for
    // splice-during-prefill inside `run_generate`.
    let mut vision_outputs: Vec<rvllm_runtime::qwen36_bring_up::VisionForwardOutput> =
        Vec::with_capacity(req.vision_items.len());
    let mut vision_failed_msg: Option<String> = None;
    for (i, item) in req.vision_items.iter().enumerate() {
        if req.cancelled.load(Ordering::Relaxed) {
            vision_failed_msg = Some("cancelled".to_string());
            break;
        }
        match bringup.forward_gemma_vision(&item.bytes) {
            Ok(out) => {
                tracing::info!(
                    idx = i,
                    tokens = out.num_tokens,
                    hidden = out.hidden_dim,
                    "vision: Gemma ViT forward done"
                );
                if out.num_tokens != item.num_tokens {
                    vision_failed_msg = Some(format!(
                        "vision tokens mismatch: predicted {} got {}",
                        item.num_tokens, out.num_tokens
                    ));
                    break;
                }
                vision_outputs.push(out);
            }
            Err(e) => {
                vision_failed_msg = Some(format!("gemma vision forward: {e:?}"));
                break;
            }
        }
    }
    if let Some(msg) = vision_failed_msg {
        let _ = req.events_tx.send(GenerateEvent::Error(msg));
        let _ = req.events_tx.send(GenerateEvent::Done {
            finish: FinishReason::Stop,
            prompt_tokens: prompt_len,
            completion_tokens: 0,
        });
        return;
    }
    let vision_splice: Vec<(usize, &[u8])> = req
        .vision_slots
        .iter()
        .map(|s| (s.token_start, vision_outputs[s.vision_item_idx].data.as_slice()))
        .collect();

    let result = unsafe {
        bringup.run_generate(
            kernels.fn_embed,
            kernels.fn_argmax,
            &req.prompt_ids,
            req.max_new_tokens as usize,
            &req.stop_token_ids,
            // shadow_requested: per-request header was removed; gate on
            // the env var directly so cycle-26 ShadowDumper analysis can
            // fire without needing to restore the header path.
            env_truthy("RVLLM_NVFP4_SHADOW_F16"),
            sampling_cfg,
            // Forward the request-level cancellation flag. The HTTP
            // handler sets this on client disconnect / wall-clock
            // timeout; the runtime breaks out of the decode loop on
            // the next step so the worker thread does not stay
            // blocked rendering tokens nobody will read.
            Some(req.cancelled.as_ref()),
            Some(&mut on_token),
            &vision_splice,
        )
    };

    match result {
        Ok(generated_ids) => {
            // Tokens were already emitted to the events channel via
            // the on_token callback during run_generate. We just
            // need the final Done event with the right finish_reason
            // and completion_tokens count.
            let final_emitted = emitted;
            let finish = if req.cancelled.load(Ordering::Relaxed) {
                FinishReason::Cancelled
            } else if generated_ids
                .last()
                .is_some_and(|id| req.stop_token_ids.contains(id))
            {
                FinishReason::Stop
            } else if final_emitted >= req.max_new_tokens {
                FinishReason::Length
            } else {
                FinishReason::Stop
            };
            let _ = req.events_tx.send(GenerateEvent::Done {
                finish,
                prompt_tokens: prompt_len,
                completion_tokens: final_emitted,
            });
        }
        Err(e) => {
            let msg = format!("run_generate: {e}");
            tracing::error!(request_id = %req.request_id, error = %msg, "generation failed");
            let _ = req.events_tx.send(GenerateEvent::Error(msg));
        }
    }
}

/// Resolve `Gemma4EnginePaths` from env vars + sensible fallbacks,
/// matching what `probe-gemma4-load` accepts (so operators can move
/// straight from probe to serve). On sm_121 the cutlass_so / fa3_so
/// are never opened; the policy file is parsed but its entries are
/// not consulted, so a minimal placeholder is acceptable.
pub fn resolve_paths(
    model_dir: PathBuf,
    kernels_dir: PathBuf,
    cutlass_so: Option<PathBuf>,
    fa3_so: Option<PathBuf>,
    policy_json: Option<PathBuf>,
) -> Result<Gemma4EnginePaths, ApiError> {
    const UNUSED: &str = "<unused-on-sm121>";

    // Codex31/34: detect sm_121 host so we can (a) point cutlass_so at
    // the real libcutlass_sm120.so (lib_so.rs's resolver anchors at
    // sm90_hint.parent().parent() — placeholder breaks the search),
    // and (b) skip the minimal-policy placeholder write on read-only
    // /tmp / restricted containers. Two signals, OR'd: nvidia-smi
    // (preferred — matches the runtime's actual device) and the
    // kernels_dir layout (deterministic fallback for containers
    // without nvidia-smi or whose first-line cap doesn't match the
    // CUDA device). Either is sufficient.
    fn is_sm121_via_nvidia_smi() -> bool {
        // Authoritative device-property probe via nvidia-smi
        // compute_cap. The earlier code OR'd this with a
        // kernels_dir-based detection that flipped to sm_121 if the
        // operator's kernel root happened to ship `sm_121/` alongside
        // other arch dirs — an H100 host with a shared multi-arch
        // kernel install would then be misclassified, the
        // sm_121-only `libcutlass_sm120.so` resolver would fire, and
        // startup would either ENOENT or load the wrong .so.
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output();
        match out {
            Ok(o) if o.status.success() => {
                let s = String::from_utf8_lossy(&o.stdout);
                s.lines().next().map(|l| l.trim() == "12.1").unwrap_or(false)
            }
            _ => false,
        }
    }
    fn is_sm121_via_env_override() -> bool {
        // Operator escape hatch: `RVLLM_FORCE_SM121=1` if nvidia-smi
        // is unavailable in the container but the device IS sm_121.
        // Replaces the implicit "directory exists" signal with an
        // explicit one.
        std::env::var("RVLLM_FORCE_SM121")
            .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
            .unwrap_or(false)
    }
    let sm121 = is_sm121_via_nvidia_smi() || is_sm121_via_env_override();

    // Resolve the policy-json path. We never write into `kernels_dir`
    // here — that path can be a system-wide read-only location
    // (read-only container layer, immutable Nix store, root-owned
    // shared install) and a resolver should not depend on being able
    // to write there.
    //
    // Order:
    //   1. Caller-supplied `policy_json` — used as-is.
    //   2. `RVLLM_MINIMAL_POLICY_PATH` env var — explicit operator
    //      override; useful for read-only filesystems where the
    //      operator wants the file in a known location.
    //   3. `std::env::temp_dir()` fallback — written once per process
    //      with the placeholder body. The file is small (~80 B) and
    //      idempotent, so a stale copy from a prior process is
    //      harmless.
    // The runtime loader at `Gemma4Bringup::load` reads
    // `paths.policy_json` strictly via `std::fs::read`, so the file
    // must exist on disk before bring-up. Both fallback branches
    // therefore write the placeholder body if the file is not
    // already there. A previous iteration only wrote the temp_dir
    // path and returned the env-override path raw — operators
    // following the "set RVLLM_MINIMAL_POLICY_PATH" hint then hit
    // ENOENT despite obeying the error message.
    let policy_json = match policy_json {
        Some(p) => p,
        None => {
            // On sm_121 the runtime (Codex30-2 / Codex31-2) skips
            // policy.json entirely — neither the generic nor the
            // gemma4 bring-up reads it. Synthesise a path so anything
            // that prints `policy_json` for diagnostics shows a
            // sensible location, but do NOT touch disk: a read-only
            // /tmp or a restricted container shouldn't abort startup
            // for a file that won't be read.
            //
            // On non-sm_121 (RTX 5090, RTX 6000 Blackwell, …) the
            // FP8 plan loader DOES read this file — and the legacy
            // fallback wrote a tiny placeholder body that
            // `Fp8GemmPlan::from_policy(...)` then rejects later, deep
            // inside autotune. That produced a confusing "plan
            // missing entry for shape X" instead of the truthful
            // "operator forgot to pass --policy-json". Refuse here
            // with a clear startup error.
            let p = std::env::var_os("RVLLM_MINIMAL_POLICY_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    std::env::temp_dir().join("rvllm-serve-minimal-policy.json")
                });
            if !sm121 {
                return Err(ApiError::Internal(format!(
                    "policy_json required on non-sm_121 hosts: pass an \
                     explicit FP8 policy via --policy-json (a real one \
                     produced by the autotune sweep, not a placeholder). \
                     Resolved would-be path was {} but no policy was \
                     supplied and synthesising a stub here only delays \
                     the failure to Fp8GemmPlan::from_policy(...)",
                    p.display()
                )));
            }
            p
        }
    };

    // Codex31-1: when the operator hasn't supplied an explicit
    // cutlass_so on a sm_121 host, point at the per-arch dir so
    // CutlassBackend::resolve_sm120_so_path (which anchors at
    // sm90_hint.parent().parent().join(arch)) actually finds the
    // shipped libcutlass_sm120.so. The legacy "<unused-on-sm121>"
    // placeholder broke that search and silently fell to
    // CutlassBackend::Absent — production was running without the
    // CUTLASS blockwise FP8 path unless RVLLM_CUTLASS_SM120_SO was
    // set explicitly. The file may still be absent (operators
    // without CUTLASS), in which case the resolver keeps falling
    // through to Absent — same outcome as before, just no longer
    // hidden behind a string-shaped tripwire.
    let cutlass_so = cutlass_so.unwrap_or_else(|| {
        if sm121 {
            // Codex34-2: when the operator already pointed kernels_dir
            // at the per-arch sm_121 subdir (Codex32-2 lets that work
            // for resolve_kernels_dir), don't append sm_121 again —
            // that would land at .../sm_121/sm_121/libcutlass_sm120.so
            // and the resolver's parent().parent() walk would still
            // land in the wrong dir, falling back to Absent.
            let arch_dir = if kernels_dir.file_name().and_then(|s| s.to_str()) == Some("sm_121") {
                kernels_dir.clone()
            } else {
                kernels_dir.join("sm_121")
            };
            arch_dir.join("libcutlass_sm120.so")
        } else {
            PathBuf::from(UNUSED)
        }
    });
    Ok(Gemma4EnginePaths {
        model_dir,
        kernels_dir,
        cutlass_so,
        fa3_so: fa3_so.unwrap_or_else(|| PathBuf::from(UNUSED)),
        policy_json,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    /// Both tests in this module manipulate the
    /// `RVLLM_MINIMAL_POLICY_PATH` env var. cargo runs lib tests in
    /// parallel by default, so without serialisation they race and
    /// one observes the other's env state. A process-wide lock keeps
    /// the env-var critical section ordered without needing
    /// `--test-threads=1` from the user.
    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Detect sm_121 the same way `resolve_paths` does, so tests
    /// stay portable between an x86 dev box and a GB10 CI host
    /// without pulling in a cudarc dep.
    fn detect_sm121_for_test() -> bool {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok()
                } else { None }
            })
            .map(|s| s.lines().next().map(|l| l.trim() == "12.1").unwrap_or(false))
            .unwrap_or(false)
            || std::env::var("RVLLM_FORCE_SM121")
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false)
    }

    /// `resolve_paths` must NOT write into `kernels_dir`. A read-only
    /// kernel install (Nix store, container layer, root-owned shared
    /// dir) is a legitimate deploy shape; the resolver writing a
    /// `.serve-minimal-policy.json` there used to fail at startup
    /// with a confusing IO error.
    ///
    /// On sm_121 the runtime skips policy_json entirely so the
    /// resolver succeeds without touching disk. On non-sm_121 the
    /// resolver now refuses (Codex60 / round-16 finding #2): the
    /// loader needs a real autotune-produced policy, and the old
    /// behaviour of synthesising a placeholder only delayed the
    /// failure to `Fp8GemmPlan::from_policy(...)`.
    #[test]
    fn resolve_paths_does_not_write_into_kernels_dir() {
        let _g = env_lock().lock().expect("env lock");
        let tmp = std::env::temp_dir().join(format!(
            "rvllm-serve-resolve-paths-test-{}", std::process::id()
        ));
        std::fs::create_dir_all(&tmp).expect("setup tmp");
        let kernels_dir = tmp.join("kernels-readonly");
        std::fs::create_dir_all(&kernels_dir).expect("setup kernels");
        let model_dir = tmp.join("model");
        std::fs::create_dir_all(&model_dir).expect("setup model");

        std::env::remove_var("RVLLM_MINIMAL_POLICY_PATH");
        let result = resolve_paths(model_dir, kernels_dir.clone(), None, None, None);

        if detect_sm121_for_test() {
            let paths = result.expect("resolve on sm_121");
            assert!(!paths.policy_json.starts_with(&kernels_dir),
                "policy_json {:?} is inside kernels_dir {:?}",
                paths.policy_json, kernels_dir);
        } else {
            assert!(result.is_err(),
                "resolve_paths must refuse on non-sm_121 without --policy-json: {:?}",
                result.map(|p| p.policy_json));
        }

        // Postcondition: kernels_dir untouched in either branch.
        let after: Vec<_> = std::fs::read_dir(&kernels_dir)
            .expect("read kernels_dir")
            .collect();
        assert!(after.is_empty(),
            "resolve_paths wrote into kernels_dir: {:?}",
            after.into_iter().filter_map(|e| e.ok().map(|e| e.path())).collect::<Vec<_>>()
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Operator can override the policy path via env var, e.g. on a
    /// read-only host where /tmp is also read-only. The override path
    /// On sm_121 the env override decides where the (unread)
    /// policy_json path is recorded; the resolver doesn't touch
    /// disk. On non-sm_121 the resolver refuses regardless of the
    /// env override — the env var only steers the placeholder
    /// location, not a real policy.
    #[test]
    fn resolve_paths_honours_env_override() {
        let _g = env_lock().lock().expect("env lock");
        let tmp = std::env::temp_dir().join(format!(
            "rvllm-serve-resolve-env-test-{}", std::process::id()
        ));
        std::fs::create_dir_all(&tmp).expect("setup tmp");
        let kernels_dir = tmp.join("kernels");
        std::fs::create_dir_all(&kernels_dir).expect("setup kernels");
        let custom_policy = tmp.join("my-policy.json");

        std::env::set_var("RVLLM_MINIMAL_POLICY_PATH", &custom_policy);
        let result = resolve_paths(
            tmp.join("model"),
            kernels_dir.clone(),
            None,
            None,
            None,
        );
        std::env::remove_var("RVLLM_MINIMAL_POLICY_PATH");

        if detect_sm121_for_test() {
            let paths = result.expect("resolve on sm_121 with env override");
            assert_eq!(paths.policy_json, custom_policy,
                "env override not honoured: got {:?}", paths.policy_json);
        } else {
            assert!(result.is_err(),
                "non-sm_121 must refuse even with env override: {:?}",
                result.map(|p| p.policy_json));
        }

        let after: Vec<_> = std::fs::read_dir(&kernels_dir)
            .expect("read kernels_dir")
            .collect();
        assert!(after.is_empty(), "kernels_dir was written to despite env override");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
