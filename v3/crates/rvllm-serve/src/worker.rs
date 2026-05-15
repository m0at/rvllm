//! Engine worker thread.
//!
//! The Gemma 4 bringup binds CUDA streams, modules, and a context to
//! the thread that calls `Gemma4Bringup::load`. We park it on a
//! dedicated `std::thread` and feed it `EngineReq`s through a
//! `tokio::sync::mpsc::UnboundedReceiver`. Each request carries its
//! own per-request response channel.
//!
//! Generation is currently single-sequence: each request is processed
//! to completion before the next one starts. The HTTP-side semaphore
//! (size = `RVLLM_MAX_NUM_SEQS`) bounds queue depth into this worker
//! so that backpressure shows up on the client rather than as an
//! unbounded buffer.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::info;
#[cfg(feature = "cuda")]
use tracing::error;

#[derive(Debug)]
pub struct EngineReq {
    pub req_id: uuid::Uuid,
    pub prompt_ids: Vec<u32>,
    pub max_tokens: u32,
    /// Stop sequences expressed as token id sequences.
    pub stop: Vec<Vec<u32>>,
    /// EOS token ids; generation halts when one is sampled.
    pub eos: Vec<u32>,
    #[allow(dead_code)]
    pub temperature: f32,
    #[allow(dead_code)]
    pub top_p: f32,
    pub tx: mpsc::UnboundedSender<TokenEvent>,
}

#[derive(Debug, Clone)]
pub enum TokenEvent {
    /// One sampled token. `id` is the token id; `text` is the
    /// incremental decoded string. Empty `text` is valid for
    /// tokenizers that emit byte-level continuations.
    Token { id: u32, text: String },
    Finish {
        reason: &'static str,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    Error(String),
}

#[derive(Debug, Clone)]
pub struct WorkerHandle {
    /// Send chat-completion requests here. Worker reads them on its
    /// own thread.
    pub tx: mpsc::UnboundedSender<EngineReq>,
    pub tokenizer: crate::openai::SharedTokenizer,
    pub chat_template: Arc<crate::openai::ChatTemplate>,
}

/// What the worker needs to bring the engine up. We pass paths +
/// arena sizing rather than a populated `Gemma4Bringup` so that the
/// blocking CUDA init happens on the worker thread, not the caller.
pub struct WorkerPaths {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
}

/// Spawn the worker. Blocks until the engine has finished loading
/// (or returns an error if init failed) so that callers can return
/// startup errors before binding the HTTP port.
///
/// `dry_run = true` skips engine init entirely — used for compile/CI
/// tests off-GPU.
pub fn spawn(
    paths: WorkerPaths,
    tokenizer: crate::openai::SharedTokenizer,
    chat_template: Arc<crate::openai::ChatTemplate>,
    dry_run: bool,
) -> Result<WorkerHandle, String> {
    let (req_tx, req_rx) = mpsc::unbounded_channel::<EngineReq>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    std::thread::Builder::new()
        .name("rvllm-engine".into())
        .spawn(move || worker_main(paths, req_rx, ready_tx, dry_run))
        .map_err(|e| format!("spawn engine thread: {e}"))?;

    match ready_rx.recv() {
        Ok(Ok(())) => Ok(WorkerHandle {
            tx: req_tx,
            tokenizer,
            chat_template,
        }),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("engine thread died before signalling ready".into()),
    }
}

#[cfg(feature = "cuda")]
fn worker_main(
    paths: WorkerPaths,
    mut rx: mpsc::UnboundedReceiver<EngineReq>,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
    dry_run: bool,
) {
    use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};

    if dry_run {
        info!("RVLLM_DRY_RUN: skipping engine init");
        let _ = ready.send(Ok(()));
        // Drain forever, replying with an error.
        while let Some(req) = rx.blocking_recv() {
            let _ = req
                .tx
                .send(TokenEvent::Error("RVLLM_DRY_RUN is set — engine init was skipped".into()));
        }
        return;
    }

    info!(model_dir = %paths.model_dir.display(), "loading Gemma4Bringup");

    let arena_bytes = compute_arena_bytes();
    let g4_paths = Gemma4EnginePaths {
        model_dir: paths.model_dir,
        kernels_dir: paths.kernels_dir,
        cutlass_so: paths.cutlass_so,
        fa3_so: paths.fa3_so,
        policy_json: paths.policy_json,
    };
    let g4 = match Gemma4Bringup::load(g4_paths, arena_bytes) {
        Ok(g) => g,
        Err(e) => {
            let _ = ready.send(Err(format!("gemma4 bringup: {e}")));
            return;
        }
    };

    let embed_mod = match g4.kernels.load_ptx("embedding_gather_f16") {
        Ok(m) => m,
        Err(e) => {
            let _ = ready.send(Err(format!("load embedding_gather_f16: {e}")));
            return;
        }
    };
    let fn_embed = match embed_mod.get_function("embedding_gather_f16_kernel") {
        Ok(f) => f,
        Err(e) => {
            let _ = ready.send(Err(format!("get embedding_gather_f16_kernel: {e}")));
            return;
        }
    };
    let argmax_mod = match g4.kernels.load_ptx("argmax") {
        Ok(m) => m,
        Err(e) => {
            let _ = ready.send(Err(format!("load argmax: {e}")));
            return;
        }
    };
    let fn_argmax = match argmax_mod.get_function("argmax_kernel") {
        Ok(f) => f,
        Err(e) => {
            let _ = ready.send(Err(format!("get argmax_kernel: {e}")));
            return;
        }
    };

    info!("Gemma4Bringup ready");
    if ready.send(Ok(())).is_err() {
        error!("caller dropped ready channel before engine signalled ready");
        return;
    }

    while let Some(req) = rx.blocking_recv() {
        process_request(&g4, fn_embed, fn_argmax, req);
    }
    info!("rvllm-engine worker shutting down");
}

#[cfg(feature = "cuda")]
fn compute_arena_bytes() -> usize {
    if let Some(gb) = std::env::var("RVLLM_ARENA_GB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        return gb * 1024 * 1024 * 1024;
    }
    let free = rvllm_runtime::gpu_helpers::probe_free_bytes()
        .expect("probe free HBM bytes for arena sizing");
    let reserve = 512 * 1024 * 1024;
    let arena = if free > reserve { free - reserve } else { free };
    info!(
        free_gb = (free as f64 / 1e9),
        arena_gb = (arena as f64 / 1e9),
        "auto-sized arena"
    );
    arena
}

#[cfg(feature = "cuda")]
fn process_request(
    g4: &rvllm_runtime::gemma4_bring_up::Gemma4Bringup,
    fn_embed: rvllm_runtime::KernelFn,
    fn_argmax: rvllm_runtime::KernelFn,
    req: EngineReq,
) {
    let EngineReq {
        req_id,
        prompt_ids,
        max_tokens,
        stop,
        eos,
        temperature: _,
        top_p: _,
        tx,
    } = req;

    let prompt_tokens = prompt_ids.len() as u32;
    let mut completion_tokens: u32 = 0;
    let mut emitted_text: String = String::new();
    let mut stop_hit = false;

    // Greedy match: track suffix of emitted token-id sequence against
    // each stop sequence. Decoded-string-based stop matching is
    // delegated to the HTTP side (it has the tokenizer cached).
    let mut emitted_ids: Vec<u32> = Vec::with_capacity(max_tokens as usize);

    let result = unsafe {
        g4.run_generate_streaming(
            fn_embed,
            fn_argmax,
            &prompt_ids,
            max_tokens as usize,
            &eos,
            &mut |tok_id| {
                completion_tokens += 1;
                emitted_ids.push(tok_id);
                // Stop on any token-id suffix match.
                for s in &stop {
                    if !s.is_empty() && emitted_ids.ends_with(s) {
                        stop_hit = true;
                        break;
                    }
                }
                // Detokenization happens on the HTTP side; here we
                // just emit the id and an empty incremental string.
                // The HTTP task decodes the running id sequence each
                // step and diff'es against `emitted_text`.
                let _ = &mut emitted_text;
                let _ = tx.send(TokenEvent::Token {
                    id: tok_id,
                    text: String::new(),
                });
                !stop_hit
            },
        )
    };

    let reason: &'static str = match (result, stop_hit) {
        (Err(e), _) => {
            let _ = tx.send(TokenEvent::Error(format!("req {req_id}: {e}")));
            return;
        }
        (Ok(()), true) => "stop",
        (Ok(()), false) => {
            if completion_tokens >= max_tokens {
                "length"
            } else {
                "stop"
            }
        }
    };
    let _ = tx.send(TokenEvent::Finish {
        reason,
        prompt_tokens,
        completion_tokens,
    });
}

#[cfg(not(feature = "cuda"))]
fn worker_main(
    _paths: WorkerPaths,
    mut rx: mpsc::UnboundedReceiver<EngineReq>,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
    dry_run: bool,
) {
    if !dry_run {
        let _ = ready.send(Err(
            "rvllm-server built without the `cuda` feature; set RVLLM_DRY_RUN=1 to bypass engine init".into(),
        ));
        return;
    }
    info!("RVLLM_DRY_RUN: skipping engine init (no cuda feature)");
    let _ = ready.send(Ok(()));
    while let Some(req) = rx.blocking_recv() {
        let _ = req
            .tx
            .send(TokenEvent::Error("dry-run build (no cuda feature)".into()));
    }
}

// Dropping the request mpsc tx is the signal to the worker loop to
// exit on next recv. No explicit shutdown handler needed; the tokio
// `serve` future's graceful-shutdown future is wired in `http::serve`.
