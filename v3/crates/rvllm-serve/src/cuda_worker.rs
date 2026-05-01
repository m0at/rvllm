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
        Ok(Ok(())) => Ok((WorkerHandle::new(req_tx), join)),
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
        let _ = req.events_tx.blocking_send(GenerateEvent::Done {
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
        )
    };

    match result {
        Ok(generated_ids) => {
            // `run_generate` returns ONLY the generated tokens (not
            // prompt + generated). See gemma4_bring_up.rs:1634 —
            // `output_ids` starts empty and only decode samples push.
            let generated: &[u32] = &generated_ids;

            let mut emitted: u32 = 0;
            for (i, &id) in generated.iter().enumerate() {
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
                    return;
                }
                emitted += 1;
            }

            let finish = if generated
                .last()
                .is_some_and(|id| req.stop_token_ids.contains(id))
            {
                FinishReason::Stop
            } else if emitted >= req.max_new_tokens {
                FinishReason::Length
            } else {
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
            let p = std::env::var_os("RVLLM_MINIMAL_POLICY_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    std::env::temp_dir().join("rvllm-serve-minimal-policy.json")
                });
            if !p.exists() {
                std::fs::write(&p, MINIMAL_POLICY).map_err(|e| {
                    ApiError::Internal(format!(
                        "write minimal policy {}: {e} \
                         (set RVLLM_MINIMAL_POLICY_PATH to a writeable path or \
                         pass an explicit policy via --policy-json)",
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

    /// `resolve_paths` must NOT write into `kernels_dir`. A read-only
    /// kernel install (Nix store, container layer, root-owned shared
    /// dir) is a legitimate deploy shape; the resolver writing a
    /// `.serve-minimal-policy.json` there used to fail at startup
    /// with a confusing IO error. The fallback now lives in
    /// `std::env::temp_dir()`.
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

        // Sanity: the kernels_dir starts empty.
        let before: Vec<_> = std::fs::read_dir(&kernels_dir)
            .expect("read kernels_dir")
            .collect();
        assert!(before.is_empty(), "test setup not empty");

        // Strip env override so we exercise the temp_dir fallback path.
        std::env::remove_var("RVLLM_MINIMAL_POLICY_PATH");
        let paths = resolve_paths(model_dir, kernels_dir.clone(), None, None, None)
            .expect("resolve");

        // Postcondition #1: kernels_dir untouched.
        let after: Vec<_> = std::fs::read_dir(&kernels_dir)
            .expect("read kernels_dir")
            .collect();
        assert!(after.is_empty(),
            "resolve_paths wrote into kernels_dir: {:?}",
            after.into_iter().filter_map(|e| e.ok().map(|e| e.path())).collect::<Vec<_>>()
        );

        // Postcondition #2: policy_json points somewhere outside
        // kernels_dir AND the file exists.
        assert!(!paths.policy_json.starts_with(&kernels_dir),
            "policy_json {:?} is inside kernels_dir {:?}",
            paths.policy_json, kernels_dir);
        assert!(paths.policy_json.exists(),
            "policy_json {:?} not created", paths.policy_json);

        // Cleanup.
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Operator can override the policy path via env var, e.g. on a
    /// read-only host where /tmp is also read-only. The override path
    /// MUST be materialised on disk by `resolve_paths` — the runtime
    /// loader reads it strictly, and a path that "exists in the env
    /// var" but not on the filesystem produces an ENOENT that
    /// contradicts the error message we hand the operator.
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
        let paths = resolve_paths(
            tmp.join("model"),
            kernels_dir.clone(),
            None,
            None,
            None,
        )
        .expect("resolve");
        std::env::remove_var("RVLLM_MINIMAL_POLICY_PATH");

        assert_eq!(paths.policy_json, custom_policy,
            "env override not honoured: got {:?}", paths.policy_json);
        assert!(paths.policy_json.exists(),
            "env-override path {:?} not materialised on disk — runtime loader will ENOENT",
            paths.policy_json);
        let body = std::fs::read_to_string(&paths.policy_json)
            .expect("read materialised policy");
        assert!(body.contains("serve-sm121"), "minimal-policy body missing: {body:?}");
        let after: Vec<_> = std::fs::read_dir(&kernels_dir)
            .expect("read kernels_dir")
            .collect();
        assert!(after.is_empty(), "kernels_dir was written to despite env override");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
