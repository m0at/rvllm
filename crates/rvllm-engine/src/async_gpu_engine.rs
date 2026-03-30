//! Async GPU inference engine wrapping [`GpuLLMEngine`] with tokio.
//!
//! Runs `GpuLLMEngine` on a dedicated OS thread so the tokio async loop
//! stays responsive during the ~4ms GPU compute wait. New HTTP requests
//! are drained from channels and pushed to the shared `RequestQueue`
//! while the GPU thread is busy.
//!
//! Gated behind `#[cfg(feature = "cuda")]` -- only compiled when the CUDA
//! feature is active.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc as std_mpsc;
    use std::sync::Arc;

    use tokio::sync::{mpsc, oneshot};
    use tokio_stream::wrappers::ReceiverStream;
    use tokio_util::sync::CancellationToken;
    use tracing::{debug, error, info, warn};

    use rvllm_config::EngineConfig;
    use rvllm_core::prelude::{
        LLMError, RequestId, RequestOutput, Result, SamplingParams, TokenId,
    };

    use crate::gpu_engine::{AbortQueue, GpuLLMEngine, PendingRequest, RequestQueue};

    // -------------------------------------------------------------------
    // Channel message types
    // -------------------------------------------------------------------

    /// Commands sent from the public API to the background engine task.
    enum GpuEngineCommand {
        /// Add a request with an explicit RequestId (non-streaming path).
        AddRequest {
            request_id: RequestId,
            prompt: String,
            prompt_token_ids: Option<Vec<TokenId>>,
            sampling_params: SamplingParams,
            response_tx: oneshot::Sender<Result<()>>,
        },
        /// Abort a running request.
        AbortRequest { request_id: RequestId },
        /// Graceful shutdown.
        Shutdown,
    }

    /// A streaming generation request with its output channel.
    struct GpuEngineRequest {
        prompt: String,
        prompt_token_ids: Option<Vec<TokenId>>,
        sampling_params: SamplingParams,
        output_tx: mpsc::Sender<RequestOutput>,
        /// Sends back the assigned RequestId once the engine accepts the request.
        id_tx: oneshot::Sender<Result<RequestId>>,
    }

    // -------------------------------------------------------------------
    // GPU thread protocol
    // -------------------------------------------------------------------

    enum GpuWork {
        Step,
        Shutdown,
    }

    struct GpuStepResult {
        outputs: Result<Vec<RequestOutput>>,
        has_unfinished: bool,
    }

    // -------------------------------------------------------------------
    // AsyncGpuLLMEngine
    // -------------------------------------------------------------------

    /// Async GPU inference engine that runs [`GpuLLMEngine`] on a dedicated
    /// OS thread and exposes an async streaming interface.
    ///
    /// Architecture:
    /// - A dedicated `std::thread` owns the `GpuLLMEngine` and blocks on
    ///   `engine.step()` calls (~4ms GPU compute).
    /// - The tokio async loop drains HTTP requests from mpsc channels and
    ///   pushes them to the shared `RequestQueue` / `AbortQueue` while the
    ///   GPU thread is busy computing.
    /// - Communication between async loop and GPU thread uses `std::sync::mpsc`.
    ///
    /// This keeps the tokio executor responsive during GPU waits.
    pub struct AsyncGpuLLMEngine {
        cmd_tx: mpsc::Sender<GpuEngineCommand>,
        gen_tx: mpsc::Sender<GpuEngineRequest>,
        cancel: CancellationToken,
    }

    impl Clone for AsyncGpuLLMEngine {
        fn clone(&self) -> Self {
            Self {
                cmd_tx: self.cmd_tx.clone(),
                gen_tx: self.gen_tx.clone(),
                // Clone the token for sending, but DON'T cancel on drop of clones.
                // Only the original (via explicit shutdown()) should cancel.
                cancel: self.cancel.clone(),
            }
        }
    }

    impl AsyncGpuLLMEngine {
        /// Create a new async GPU engine.
        ///
        /// Constructs a [`GpuLLMEngine`] from the provided config, spawns
        /// a dedicated OS thread for GPU work, and a tokio task to bridge
        /// the async channels.
        pub async fn new(config: EngineConfig) -> Result<Self> {
            let mut engine = GpuLLMEngine::new(config)?;
            let cancel = CancellationToken::new();
            let (cmd_tx, cmd_rx) = mpsc::channel::<GpuEngineCommand>(256);
            let (gen_tx, gen_rx) = mpsc::channel::<GpuEngineRequest>(256);

            let cancel_bg = cancel.clone();
            tokio::spawn(async move {
                Self::background_loop(engine, cmd_rx, gen_rx, cancel_bg).await;
            });

            info!("AsyncGpuLLMEngine started background task + GPU thread");
            Ok(Self {
                cmd_tx,
                gen_tx,
                cancel,
            })
        }

        /// Submit a generation request and receive a stream of incremental outputs.
        pub async fn generate(
            &self,
            prompt: String,
            params: SamplingParams,
        ) -> Result<(RequestId, ReceiverStream<RequestOutput>)> {
            let (output_tx, output_rx) = mpsc::channel(64);
            let (id_tx, id_rx) = oneshot::channel();

            self.gen_tx
                .send(GpuEngineRequest {
                    prompt,
                    prompt_token_ids: None,
                    sampling_params: params,
                    output_tx,
                    id_tx,
                })
                .await
                .map_err(|_| LLMError::GpuError("GPU engine background task stopped".into()))?;

            let request_id = id_rx
                .await
                .map_err(|_| LLMError::GpuError("request ID channel dropped".into()))??;

            Ok((request_id, ReceiverStream::new(output_rx)))
        }

        /// Add a request with an explicit ID, without streaming.
        pub async fn add_request(
            &self,
            request_id: RequestId,
            prompt: String,
            params: SamplingParams,
        ) -> Result<()> {
            let (resp_tx, resp_rx) = oneshot::channel();
            self.cmd_tx
                .send(GpuEngineCommand::AddRequest {
                    request_id,
                    prompt,
                    prompt_token_ids: None,
                    sampling_params: params,
                    response_tx: resp_tx,
                })
                .await
                .map_err(|_| LLMError::GpuError("GPU engine background task stopped".into()))?;

            resp_rx
                .await
                .map_err(|_| LLMError::GpuError("response channel dropped".into()))?
        }

        pub async fn generate_token_ids(
            &self,
            prompt: String,
            prompt_token_ids: Vec<TokenId>,
            params: SamplingParams,
        ) -> Result<(RequestId, ReceiverStream<RequestOutput>)> {
            let (output_tx, output_rx) = mpsc::channel(64);
            let (id_tx, id_rx) = oneshot::channel();

            self.gen_tx
                .send(GpuEngineRequest {
                    prompt,
                    prompt_token_ids: Some(prompt_token_ids),
                    sampling_params: params,
                    output_tx,
                    id_tx,
                })
                .await
                .map_err(|_| LLMError::GpuError("GPU engine background task stopped".into()))?;

            let request_id = id_rx
                .await
                .map_err(|_| LLMError::GpuError("request ID channel dropped".into()))??;

            Ok((request_id, ReceiverStream::new(output_rx)))
        }

        /// Abort a running request. No-op if the request has already finished.
        pub async fn abort_request(&self, request_id: &RequestId) {
            let _ = self
                .cmd_tx
                .send(GpuEngineCommand::AbortRequest {
                    request_id: *request_id,
                })
                .await;
        }

        /// Graceful shutdown: cancel the background loop.
        pub fn shutdown(&self) -> Result<()> {
            info!("AsyncGpuLLMEngine shutting down");
            self.cancel.cancel();
            let _ = self.cmd_tx.try_send(GpuEngineCommand::Shutdown);
            Ok(())
        }

        // -------------------------------------------------------------------
        // Background loop (runs on tokio task)
        // -------------------------------------------------------------------

        async fn background_loop(
            mut engine: GpuLLMEngine,
            mut cmd_rx: mpsc::Receiver<GpuEngineCommand>,
            mut gen_rx: mpsc::Receiver<GpuEngineRequest>,
            cancel: CancellationToken,
        ) {
            let mut output_channels: HashMap<RequestId, mpsc::Sender<RequestOutput>> =
                HashMap::new();
            let next_request_id: Arc<AtomicU64> = engine.request_id_counter();

            // Shared queues: async loop pushes, GPU engine drains during step().
            let request_queue: RequestQueue =
                std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let abort_queue: AbortQueue =
                std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            engine.set_request_queue(request_queue.clone());
            engine.set_abort_queue(abort_queue.clone());

            // Spawn dedicated GPU thread.
            // Communication: async loop sends GpuWork, GPU thread sends back GpuStepResult.
            let (gpu_tx, gpu_rx) = std_mpsc::channel::<GpuWork>();
            // Use tokio::sync::mpsc for results so we can .await instead of spin+yield
            let (result_tx, mut result_rx) = mpsc::channel::<GpuStepResult>(4);

            let gpu_thread = std::thread::Builder::new()
                .name("gpu-step".into())
                .spawn(move || {
                    Self::gpu_thread_loop(engine, gpu_rx, result_tx);
                })
                .expect("failed to spawn GPU thread");

            let mut has_unfinished = false;

            loop {
                if cancel.is_cancelled() {
                    info!("GPU background loop cancelled, draining");
                    break;
                }

                // -- Drain pending commands into shared queues (non-blocking) --
                Self::drain_commands_to_queue(
                    &mut cmd_rx,
                    &request_queue,
                    &abort_queue,
                    &mut output_channels,
                );

                // -- Drain pending generate requests into shared queue (non-blocking) --
                Self::drain_generate_requests_to_queue(
                    &mut gen_rx,
                    &request_queue,
                    &mut output_channels,
                    &next_request_id,
                );

                // Check if shared queues have pending work (pushed during GPU compute)
                let queues_have_work = !request_queue.lock().unwrap().is_empty();

                // -- If idle, block until new work arrives --
                if !has_unfinished && !queues_have_work {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            info!("GPU background loop cancelled while idle");
                            break;
                        }
                        cmd = cmd_rx.recv() => {
                            match cmd {
                                Some(cmd) => Self::handle_command_to_queue(
                                    cmd,
                                    &request_queue,
                                    &abort_queue,
                                    &mut output_channels,
                                ),
                                None => break,
                            }
                        }
                        gen = gen_rx.recv() => {
                            match gen {
                                Some(req) => Self::handle_generate_to_queue(
                                    req,
                                    &request_queue,
                                    &mut output_channels,
                                    &next_request_id,
                                ),
                                None => break,
                            }
                        }
                    }
                    // We just queued work, so there should be unfinished work
                    // after the GPU thread drains queues in step().
                    has_unfinished = true;
                    // Fall through to send Step to GPU thread.
                }

                // -- Send step to GPU thread (non-blocking send) --
                if gpu_tx.send(GpuWork::Step).is_err() {
                    error!("GPU thread died unexpectedly");
                    break;
                }

                // -- Wait for GPU result. Drain requests that arrive during the wait. --
                let result = result_rx.recv().await;
                // GPU done. Drain any requests that arrived during compute.
                Self::drain_commands_to_queue(
                    &mut cmd_rx, &request_queue, &abort_queue, &mut output_channels,
                );
                Self::drain_generate_requests_to_queue(
                    &mut gen_rx, &request_queue, &mut output_channels, &next_request_id,
                );
                match result {
                    Some(result) => {
                        has_unfinished = result.has_unfinished;
                        Self::send_outputs(
                            result.outputs, &mut output_channels, &abort_queue,
                        ).await;
                    }
                    None => {
                        error!("GPU thread result channel disconnected");
                        has_unfinished = false;
                    }
                }
            }

            // Shutdown the GPU thread
            let _ = gpu_tx.send(GpuWork::Shutdown);
            if let Err(e) = gpu_thread.join() {
                error!("GPU thread panicked: {:?}", e);
            }

            // Notify remaining streaming clients
            for (rid, tx) in output_channels.drain() {
                warn!(%rid, "GPU engine shutting down with in-flight request");
                drop(tx);
            }

            info!("GPU background loop exited");
        }

        // -------------------------------------------------------------------
        // GPU thread (dedicated OS thread, NOT on tokio)
        // -------------------------------------------------------------------

        fn gpu_thread_loop(
            mut engine: GpuLLMEngine,
            work_rx: std_mpsc::Receiver<GpuWork>,
            result_tx: mpsc::Sender<GpuStepResult>,
        ) {
            info!("GPU thread started");
            while let Ok(work) = work_rx.recv() {
                match work {
                    GpuWork::Step => {
                        let outputs = engine.step();
                        let unfinished = engine.has_unfinished();
                        if result_tx.blocking_send(GpuStepResult {
                            outputs,
                            has_unfinished: unfinished,
                        }).is_err() {
                            break;
                        }
                    }
                    GpuWork::Shutdown => break,
                }
            }
            info!("GPU thread exiting");
        }

        // -------------------------------------------------------------------
        // Helpers: drain channels into shared queues
        // -------------------------------------------------------------------

        /// Drain all pending commands from the tokio channel, pushing
        /// requests/aborts to the shared queues (no engine access needed).
        fn drain_commands_to_queue(
            cmd_rx: &mut mpsc::Receiver<GpuEngineCommand>,
            request_queue: &RequestQueue,
            abort_queue: &AbortQueue,
            output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
        ) {
            loop {
                match cmd_rx.try_recv() {
                    Ok(cmd) => Self::handle_command_to_queue(
                        cmd,
                        request_queue,
                        abort_queue,
                        output_channels,
                    ),
                    Err(_) => break,
                }
            }
        }

        /// Handle a single command by pushing to shared queues.
        fn handle_command_to_queue(
            cmd: GpuEngineCommand,
            request_queue: &RequestQueue,
            abort_queue: &AbortQueue,
            output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
        ) {
            match cmd {
                GpuEngineCommand::AddRequest {
                    request_id,
                    prompt,
                    prompt_token_ids,
                    sampling_params,
                    response_tx,
                } => {
                    request_queue.lock().unwrap().push(PendingRequest {
                        request_id,
                        prompt,
                        prompt_token_ids,
                        params: sampling_params,
                    });
                    // Ack immediately -- the request is queued and will be
                    // picked up by the GPU thread at the next step().
                    let _ = response_tx.send(Ok(()));
                }
                GpuEngineCommand::AbortRequest { request_id } => {
                    abort_queue.lock().unwrap().push(request_id);
                    output_channels.remove(&request_id);
                }
                GpuEngineCommand::Shutdown => {
                    info!("GPU background loop received shutdown via drain");
                }
            }
        }

        /// Drain all pending generate requests, assigning IDs and pushing
        /// to the shared request queue.
        fn drain_generate_requests_to_queue(
            gen_rx: &mut mpsc::Receiver<GpuEngineRequest>,
            request_queue: &RequestQueue,
            output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
            next_id: &Arc<AtomicU64>,
        ) {
            loop {
                match gen_rx.try_recv() {
                    Ok(req) => Self::handle_generate_to_queue(
                        req,
                        request_queue,
                        output_channels,
                        next_id,
                    ),
                    Err(_) => break,
                }
            }
        }

        /// Process a single generate request: assign ID, push to queue, wire up channel.
        fn handle_generate_to_queue(
            req: GpuEngineRequest,
            request_queue: &RequestQueue,
            output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
            next_id: &Arc<AtomicU64>,
        ) {
            let rid = RequestId(next_id.fetch_add(1, Ordering::Relaxed));

            request_queue.lock().unwrap().push(PendingRequest {
                request_id: rid,
                prompt: req.prompt,
                prompt_token_ids: req.prompt_token_ids,
                params: req.sampling_params,
            });

            let _ = req.id_tx.send(Ok(rid));
            output_channels.insert(rid, req.output_tx);
        }

        /// Process step outputs: fan out to per-request streaming channels.
        async fn send_outputs(
            outputs: Result<Vec<RequestOutput>>,
            output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
            abort_queue: &AbortQueue,
        ) {
            match outputs {
                Ok(outputs) => {
                    for output in outputs {
                        let rid = output.request_id;
                        let finished = output.finished;

                        if let Some(tx) = output_channels.get(&rid) {
                            if tx.send(output).await.is_err() {
                                debug!(%rid, "GPU output receiver dropped, aborting");
                                abort_queue.lock().unwrap().push(rid);
                                output_channels.remove(&rid);
                                continue;
                            }
                        }

                        if finished {
                            output_channels.remove(&rid);
                            debug!(%rid, "GPU request finished, channel removed");
                        }
                    }
                }
                Err(e) => {
                    error!(%e, "GPU engine step failed");
                }
            }
        }
    }

    // No Drop impl -- clones share the CancellationToken. Dropping a clone
    // must NOT cancel the background loop. Call shutdown() explicitly, or let
    // the last Sender drop naturally close the channels.
}

#[cfg(feature = "cuda")]
pub use inner::AsyncGpuLLMEngine;
