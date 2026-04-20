//! CUDA-backed worker. Feature-gated (`cuda` / `gb10`).
//!
//! Phase 2 scope: wrap the existing monolithic
//! `Gemma4Bringup::run_generate` synchronously. All generated tokens
//! are emitted to the `events_tx` channel **after** generation
//! completes — not token-by-token on the GPU's pace. True per-token
//! streaming is phase 5 (requires breaking `run_generate` apart into
//! `prefill()` + `decode_one()`).
//!
//! This is a deliberate trade-off: end-to-end path lands today, SSE
//! delivers tokens (just all at once, not as they emerge), and the
//! API surface doesn't change when phase 5 swaps in real streaming.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use tokio::sync::mpsc;
use tokio::sync::oneshot;

use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};

use crate::error::ApiError;
use crate::openai::types::FinishReason;
use crate::worker::{GenerateEvent, GenerateRequest, WorkerHandle};

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
}

/// Spawn the CUDA worker on a dedicated OS thread.
///
/// Returns once `Gemma4Bringup::load` has completed — the returned
/// `WorkerHandle` is immediately usable. `load` takes ~20 s for
/// Gemma 4 31B; the caller should render a "loading" log line before
/// awaiting.
///
/// The `JoinHandle` must be held by the caller for the lifetime of
/// the server; dropping it + all `WorkerHandle` clones closes the
/// worker's queue, causing the thread to exit cleanly.
pub async fn spawn_cuda_worker(
    cfg: CudaWorkerConfig,
) -> Result<(WorkerHandle, std::thread::JoinHandle<()>), ApiError> {
    let (req_tx, mut req_rx) = mpsc::channel::<GenerateRequest>(cfg.queue_depth.max(1));
    let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();

    let CudaWorkerConfig { paths, arena_bytes, .. } = cfg;
    let join = std::thread::Builder::new()
        .name("rvllm-serve-cuda-worker".into())
        .spawn(move || {
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

            // Resolve kernel function pointers once.
            let kernels_ctx = match resolve_generate_kernels(&bringup) {
                Ok(k) => k,
                Err(msg) => {
                    let _ = ready_tx.send(Err(msg));
                    return;
                }
            };

            tracing::info!(
                compute_cap = ?bringup.ctx.compute_capability(),
                arena_mib = bringup.arena.capacity() / (1024 * 1024),
                "cuda worker ready",
            );
            let _ = ready_tx.send(Ok(()));

            // Main serve loop. One request at a time (single-seq).
            while let Some(req) = req_rx.blocking_recv() {
                run_one(&bringup, &kernels_ctx, req);
            }

            tracing::info!("cuda worker queue closed, exiting");
        })
        .map_err(|e| ApiError::Internal(format!("spawn cuda worker: {e}")))?;

    match ready_rx.await {
        Ok(Ok(())) => Ok((WorkerHandle::new(req_tx), join)),
        Ok(Err(msg)) => Err(ApiError::Internal(msg)),
        Err(_) => Err(ApiError::Internal(
            "cuda worker thread exited before signalling ready".into(),
        )),
    }
}

struct GenerateKernels {
    fn_embed: rvllm_kernels::KernelFn,
    fn_argmax: rvllm_kernels::KernelFn,
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
    Ok(GenerateKernels { fn_embed, fn_argmax })
}

fn run_one(bringup: &Gemma4Bringup, kernels: &GenerateKernels, req: GenerateRequest) {
    let prompt_len = req.prompt_ids.len() as u32;

    // Cancellation check before we even start (client might have
    // given up while we were dequeuing).
    if req.cancelled.load(Ordering::Relaxed) {
        let _ = req.events_tx.blocking_send(GenerateEvent::Done {
            finish: FinishReason::Cancelled,
            prompt_tokens: prompt_len,
            completion_tokens: 0,
        });
        return;
    }

    let result = unsafe {
        bringup.run_generate(
            kernels.fn_embed,
            kernels.fn_argmax,
            &req.prompt_ids,
            req.max_new_tokens as usize,
            &req.stop_token_ids,
        )
    };

    match result {
        Ok(all_token_ids) => {
            // `run_generate` returns prompt + generated ids. Trim the
            // prompt prefix so the handler gets completion tokens only.
            let generated = all_token_ids
                .get(req.prompt_ids.len()..)
                .unwrap_or(&[]);

            let mut emitted: u32 = 0;
            for (i, &id) in generated.iter().enumerate() {
                // Honour cancellation at the boundary, even though
                // run_generate can't be interrupted mid-call today.
                if req.cancelled.load(Ordering::Relaxed) {
                    let _ = req.events_tx.blocking_send(GenerateEvent::Done {
                        finish: FinishReason::Cancelled,
                        prompt_tokens: prompt_len,
                        completion_tokens: emitted,
                    });
                    return;
                }
                if req
                    .events_tx
                    .blocking_send(GenerateEvent::Token { id, position: i as u32 })
                    .is_err()
                {
                    // Handler dropped the receiver — nothing more to do.
                    return;
                }
                emitted += 1;
            }

            // Distinguish "hit EOS" from "hit max_new_tokens":
            //   run_generate stops early on EOS OR max_new, but doesn't
            //   tell us which. Infer from the emitted count + last-id
            //   against the stop set.
            let finish = if generated
                .last()
                .is_some_and(|id| req.stop_token_ids.contains(id))
            {
                FinishReason::Stop
            } else if emitted >= req.max_new_tokens {
                FinishReason::Length
            } else {
                // Stopped short without an EOS match — unexpected but
                // treat as stop.
                FinishReason::Stop
            };

            let _ = req.events_tx.blocking_send(GenerateEvent::Done {
                finish,
                prompt_tokens: prompt_len,
                completion_tokens: emitted,
            });
        }
        Err(e) => {
            let msg = format!("run_generate: {e}");
            tracing::error!(request_id = %req.request_id, error = %msg, "generation failed");
            let _ = req.events_tx.blocking_send(GenerateEvent::Error(msg));
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
    const MINIMAL_POLICY: &str =
        r#"{"revision":"serve-sm121","arch":"sm_121","variants":[],"entries":{}}"#;

    let policy_json = match policy_json {
        Some(p) => p,
        None => {
            let p = kernels_dir.join(".serve-minimal-policy.json");
            if !p.exists() {
                std::fs::write(&p, MINIMAL_POLICY).map_err(|e| {
                    ApiError::Internal(format!(
                        "write minimal policy {}: {e}",
                        p.display()
                    ))
                })?;
            }
            p
        }
    };

    Ok(Gemma4EnginePaths {
        model_dir,
        kernels_dir,
        cutlass_so: cutlass_so.unwrap_or_else(|| PathBuf::from(UNUSED)),
        fa3_so: fa3_so.unwrap_or_else(|| PathBuf::from(UNUSED)),
        policy_json,
    })
}
