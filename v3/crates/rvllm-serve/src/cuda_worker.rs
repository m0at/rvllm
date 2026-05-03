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
    // `max_queue_depth` is documented as "in-flight + queued"; the
    // worker pulls one request out of the channel and processes it
    // synchronously, so the channel buffer should be sized for
    // `queue_depth - 1` (queued) and the in-flight slot lives on the
    // worker thread. Clamp to >= 1 because `mpsc::channel(0)`
    // panics; depth=1 then accepts one in-flight + zero queued
    // (effectively channel-of-1, matching the doc within ±0).
    let channel_buf = cfg.queue_depth.saturating_sub(1).max(1);
    let (req_tx, mut req_rx) = mpsc::channel::<GenerateRequest>(channel_buf);
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
        match events_tx_ref.blocking_send(GenerateEvent::Token { id, position: pos }) {
            Ok(()) => {
                emitted += 1;
                true
            }
            Err(_) => false,
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
            Some(&mut on_token),
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
            let _ = req.events_tx.blocking_send(GenerateEvent::Done {
                finish,
                prompt_tokens: prompt_len,
                completion_tokens: final_emitted,
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
    fn is_sm121_via_kernels_dir(p: &std::path::Path) -> bool {
        // Codex34-3: kernels_dir already pointed at the per-arch
        // sm_121 dir (allowed by Codex32-2) OR the kernels root has
        // a sm_121/ subdir on disk. Either implies the operator
        // expects sm_121.
        if p.file_name().and_then(|s| s.to_str()) == Some("sm_121") {
            return true;
        }
        p.join("sm_121").is_dir()
    }
    let sm121 = is_sm121_via_nvidia_smi() || is_sm121_via_kernels_dir(&kernels_dir);

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
            // Codex31-2: on sm_121 the runtime (Codex30-2) skips
            // policy.json entirely — neither the generic nor the
            // gemma4 bring-up reads it. Avoid touching disk when
            // unnecessary so a read-only /tmp or restricted container
            // doesn't abort startup. Still record the path so any
            // diagnostic that prints `policy_json` shows the would-be
            // location.
            if !sm121 && !p.exists() {
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
