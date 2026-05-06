//! Worker bridge between tokio (frontend) and the generate loop
//! (`Worker: !Send`, dedicated OS thread).
//!
//! Design: see `v3/INFERENCE_SERVER_PLAN.md` § Architecture.
//!
//! This module defines the **shape** of the bridge. The real worker
//! (feature `cuda`) is implemented in a follow-up phase; phase 1
//! ships a mock worker used by `cargo test` on laptops without CUDA.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tokio::sync::mpsc;
use uuid::Uuid;

use crate::error::ApiError;
use crate::openai::types::FinishReason;
use crate::sampling::SamplingDecision;

/// Sent from a handler to the worker.
pub struct GenerateRequest {
    pub request_id: Uuid,
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingDecision,
    pub max_new_tokens: u32,
    pub stop_token_ids: Vec<u32>,
    // Round-20 finding #1: unbounded so a slow / non-polling SSE
    // client cannot back-pressure the single CUDA worker via the
    // 64-event blocking_send. Memory is bounded by `max_new_tokens`
    // (server-capped); worst case ~4 K events × small payload ≈ 64
    // KiB per request, far below any realistic admission concern.
    pub events_tx: mpsc::UnboundedSender<GenerateEvent>,
    pub cancelled: Arc<AtomicBool>,
    /// Vision attachments (one per `image_url` content part). Empty
    /// for text-only requests. The worker runs the native vision
    /// tower forward on each image's bytes, then splices the
    /// resulting embeddings into the post-embed hidden buffer.
    pub vision_items: Vec<VisionItem>,
    /// Per-image (token_start, num_tokens) slots in `prompt_ids`,
    /// aligned to `vision_items` by index. The worker overwrites
    /// `hidden_region[slot.token_start .. slot.token_start + slot.num_tokens]`
    /// with the vision-tower output for `vision_items[slot.vision_item_idx]`.
    pub vision_slots: Vec<crate::tokenize::VisionSlot>,
    /// Admission permit "owned" by this request for as long as the
    /// worker is processing it. The handler passes a clone of its
    /// `Arc<OwnedSemaphorePermit>` here; the permit is released
    /// only when both the handler's clone and this one drop. The
    /// worker owns this clone until it removes the request from
    /// the channel + finishes (or cancels) it. Without this, a
    /// handler-side timeout (chat_collect/completion_collect) used
    /// to release the permit while the request was still queued
    /// in the mpsc channel and/or in flight on the GPU; new
    /// requests then passed admission, did the expensive
    /// preprocessing, and bounced on a `try_send` "queue full".
    pub _admission: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
}

/// One image attached to a chat request.
pub struct VisionItem {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Number of placeholder tokens this image takes in `prompt_ids`
    /// (= post-PatchMerger embedding count, predicted host-side from
    /// image dims). Actual splice slots live in `vision_slots`.
    pub num_tokens: usize,
}

/// Events produced by the worker, consumed by the handler for
/// streaming (SSE) or aggregated for non-streaming responses.
#[derive(Debug)]
pub enum GenerateEvent {
    /// One newly generated token.
    Token { id: u32, position: u32 },
    /// Stream ended normally. Carries the reason + usage counts.
    Done { finish: FinishReason, prompt_tokens: u32, completion_tokens: u32 },
    /// Stream ended with an error. Handler maps to `ApiError`.
    Error(String),
}

/// Handle held by the tokio side. Cheap to clone — just wraps an
/// `mpsc::Sender` and an admission semaphore.
#[derive(Clone)]
pub struct WorkerHandle {
    submit: mpsc::Sender<GenerateRequest>,
    /// Bounds total in-flight requests across the entire request
    /// lifecycle (image fetch + tokenize + queued + running). The
    /// permit count equals the worker's queue depth. Handlers call
    /// [`Self::try_admit`] BEFORE doing any expensive per-request
    /// work and hold the returned permit for the rest of the
    /// handler scope; the permit is released on Drop. Without this
    /// reservation, a saturating client could pass a capacity-
    /// only `check_admission` in parallel and burn the blocking
    /// pool producing 429-bound requests before `submit` rejects.
    admission: Arc<tokio::sync::Semaphore>,
}

impl WorkerHandle {
    /// Construct from an already-created `mpsc::Sender` plus the
    /// admission permit count. Used by the worker-spawn helpers
    /// (mock + cuda).
    pub(crate) fn new(
        submit: mpsc::Sender<GenerateRequest>,
        admission_permits: usize,
    ) -> Self {
        Self {
            submit,
            admission: Arc::new(tokio::sync::Semaphore::new(admission_permits)),
        }
    }

    /// True if the worker thread is still alive (the receive side of
    /// the submit channel hasn't been dropped). Used by /health to
    /// turn ok-vs-503 on the worker's actual liveness instead of
    /// reporting ok regardless. After a CUDA worker thread crashes
    /// or exits, the channel closes and this flips to false.
    pub fn is_alive(&self) -> bool {
        !self.submit.is_closed()
    }

    /// Reserve one in-flight slot before doing any per-request
    /// work. Returns an `OwnedSemaphorePermit` that releases the
    /// slot on Drop, so the caller just keeps it in scope for the
    /// handler's lifetime. Burst arrivals that would exceed the
    /// admission permit count get `Err(Busy)` immediately,
    /// without consuming the blocking pool or network.
    ///
    /// Replaces the older capacity-only `check_admission`, which
    /// did not actually reserve anything — multiple concurrent
    /// requests could pass the check and only race for failure
    /// at `submit` time, after the expensive fetch/tokenize work.
    pub fn try_admit(&self) -> Result<tokio::sync::OwnedSemaphorePermit, ApiError> {
        if self.submit.is_closed() {
            return Err(ApiError::Unavailable("worker shut down".into()));
        }
        self.admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| ApiError::Busy("worker queue is full".into()))
    }

    /// Try to enqueue a request. Returns [`ApiError::Busy`] if the
    /// worker's queue is full (bounded `mpsc::channel`).
    pub async fn submit(&self, req: GenerateRequest) -> Result<(), ApiError> {
        match self.submit.try_send(req) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                Err(ApiError::Busy("worker queue is full".into()))
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(ApiError::Unavailable("worker shut down".into()))
            }
        }
    }
}

/// Spawn a **mock** worker on a dedicated OS thread. The mock emits
/// fake token ids 1..=N where N = `req.max_new_tokens`, one every
/// 5 ms, and respects the cancellation flag. Used by integration
/// tests and local dev without CUDA.
///
/// Returns the [`WorkerHandle`] and a [`std::thread::JoinHandle`]
/// that the caller must keep alive (the thread exits when the
/// handle + all clones are dropped).
pub fn spawn_mock_worker(
    queue_depth: usize,
) -> (WorkerHandle, std::thread::JoinHandle<()>) {
    // Match the cuda_worker's queue arithmetic: channel buffer is
    // `queue_depth - 1` queued slots + 1 in-flight on the worker
    // thread = `queue_depth` total. Asymmetry between mock and CUDA
    // would let admission tests pass against the mock while failing
    // in production (one extra request slips through).
    let (tx, mut rx) = mpsc::channel::<GenerateRequest>(queue_depth.max(1));
    let join = std::thread::Builder::new()
        .name("rvllm-serve-mock-worker".into())
        .spawn(move || {
            // Minimal blocking recv loop — no tokio inside the worker.
            while let Some(req) = rx.blocking_recv() {
                mock_run(req);
            }
        })
        .unwrap_or_else(|e| {
            panic!("failed to spawn mock worker thread: {e}");
        });
    (WorkerHandle::new(tx, queue_depth.max(1)), join)
}

/// Test-only: spawn a worker that emits a single `GenerateEvent::Error`
/// for every request and then drops the events channel.
///
/// Used by SSE error-path integration tests to assert that worker
/// errors surface as OpenAI-shaped error events on the wire (not as
/// successful-looking `finish_reason="cancelled"` chunks).
pub fn spawn_erroring_mock_worker(
    queue_depth: usize,
    error_msg: impl Into<String>,
) -> (WorkerHandle, std::thread::JoinHandle<()>) {
    // Match the cuda_worker's queue arithmetic: channel buffer is
    // `queue_depth - 1` queued slots + 1 in-flight on the worker
    // thread = `queue_depth` total. Asymmetry between mock and CUDA
    // would let admission tests pass against the mock while failing
    // in production (one extra request slips through).
    let (tx, mut rx) = mpsc::channel::<GenerateRequest>(queue_depth.max(1));
    let msg = error_msg.into();
    let join = std::thread::Builder::new()
        .name("rvllm-serve-erroring-mock-worker".into())
        .spawn(move || {
            while let Some(req) = rx.blocking_recv() {
                let _ = req
                    .events_tx
                    .send(GenerateEvent::Error(msg.clone()));
                // Drop events_tx implicitly — the SSE handler treats
                // a dropped channel as the "channel closed before Done"
                // failure mode, but it has already received the
                // Error event above so EmitError fires from that path.
            }
        })
        .unwrap_or_else(|e| {
            panic!("failed to spawn erroring-mock worker thread: {e}");
        });
    (WorkerHandle::new(tx, queue_depth.max(1)), join)
}

fn mock_run(req: GenerateRequest) {
    use std::sync::atomic::Ordering;
    let prompt_len = req.prompt_ids.len() as u32;
    let mut produced: u32 = 0;
    for tok in 1..=req.max_new_tokens {
        if req.cancelled.load(Ordering::Relaxed) {
            let _ = req.events_tx.send(GenerateEvent::Done {
                finish: FinishReason::Cancelled,
                prompt_tokens: prompt_len,
                completion_tokens: produced,
            });
            return;
        }
        if req.stop_token_ids.contains(&tok) {
            let _ = req.events_tx.send(GenerateEvent::Done {
                finish: FinishReason::Stop,
                prompt_tokens: prompt_len,
                completion_tokens: produced,
            });
            return;
        }
        if req
            .events_tx
            .send(GenerateEvent::Token { id: tok, position: produced })
            .is_err()
        {
            // Receiver dropped — handler is gone, stop.
            return;
        }
        produced += 1;
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let _ = req.events_tx.send(GenerateEvent::Done {
        finish: FinishReason::Length,
        prompt_tokens: prompt_len,
        completion_tokens: produced,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplingParams;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mock_produces_tokens() {
        let (handle, _join) = spawn_mock_worker(4);
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sampling = SamplingParams { temperature: 0.0, ..Default::default() }
            .ensure_supported()
            .expect("greedy");
        handle
            .submit(GenerateRequest {
                request_id: Uuid::new_v4(),
                prompt_ids: vec![1, 2, 3],
                sampling,
                max_new_tokens: 3,
                stop_token_ids: vec![],
                events_tx: tx,
                cancelled: Arc::new(AtomicBool::new(false)),
                vision_items: Vec::new(),
                vision_slots: Vec::new(),
                _admission: None,
            })
            .await
            .expect("submit");

        let mut ids = vec![];
        while let Some(ev) = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .ok()
            .flatten()
        {
            match ev {
                GenerateEvent::Token { id, .. } => ids.push(id),
                GenerateEvent::Done { finish, completion_tokens, .. } => {
                    assert_eq!(finish, FinishReason::Length);
                    assert_eq!(completion_tokens, 3);
                    break;
                }
                GenerateEvent::Error(e) => panic!("mock worker errored: {e}"),
            }
        }
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mock_honours_cancellation() {
        let (handle, _join) = spawn_mock_worker(4);
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sampling = SamplingParams { temperature: 0.0, ..Default::default() }
            .ensure_supported()
            .expect("greedy");
        let cancel = Arc::new(AtomicBool::new(false));
        handle
            .submit(GenerateRequest {
                request_id: Uuid::new_v4(),
                prompt_ids: vec![],
                sampling,
                max_new_tokens: 100,
                stop_token_ids: vec![],
                events_tx: tx,
                cancelled: cancel.clone(),
                vision_items: Vec::new(),
                vision_slots: Vec::new(),
                _admission: None,
            })
            .await
            .expect("submit");

        // Read a few tokens, then cancel.
        for _ in 0..3 {
            let _ = rx.recv().await;
        }
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        // Drain until Done.
        loop {
            match tokio::time::timeout(Duration::from_secs(2), rx.recv())
                .await
                .ok()
                .flatten()
            {
                Some(GenerateEvent::Done { finish, .. }) => {
                    assert_eq!(finish, FinishReason::Cancelled);
                    return;
                }
                Some(_) => continue,
                None => panic!("worker did not honour cancel"),
            }
        }
    }
}
