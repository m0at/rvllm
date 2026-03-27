//! Async GPU inference engine wrapping [`GpuLLMEngine`] with tokio.
//!
//! Runs a background step loop, accepts requests via mpsc, and returns
//! incremental [`RequestOutput`] via a tokio Stream per request.
//!
//! Gated behind `#[cfg(feature = "cuda")]` -- only compiled when the CUDA
//! feature is active.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;

    use tokio::sync::{mpsc, oneshot};
    use tokio_stream::wrappers::ReceiverStream;
    use tokio_util::sync::CancellationToken;
    use tracing::{debug, error, info, warn};

    use rvllm_config::EngineConfig;
    use rvllm_core::prelude::{LLMError, RequestId, RequestOutput, Result, SamplingParams};

    use crate::gpu_engine::GpuLLMEngine;

    // -------------------------------------------------------------------
    // Channel message types
    // -------------------------------------------------------------------

    /// Commands sent from the public API to the background engine task.
    enum GpuEngineCommand {
        /// Add a request with an explicit RequestId (non-streaming path).
        AddRequest {
            request_id: RequestId,
            prompt: String,
            sampling_params: SamplingParams,
            response_tx: oneshot::Sender<Result<()>>,
        },
        /// Abort a running request.
        AbortRequest {
            request_id: RequestId,
        },
        /// Graceful shutdown.
        Shutdown,
    }

    /// A streaming generation request with its output channel.
    struct GpuEngineRequest {
        prompt: String,
        sampling_params: SamplingParams,
        output_tx: mpsc::Sender<RequestOutput>,
        /// Sends back the assigned RequestId once the engine accepts the request.
        id_tx: oneshot::Sender<Result<RequestId>>,
    }

    // -------------------------------------------------------------------
    // AsyncGpuLLMEngine
    // -------------------------------------------------------------------

    /// Async GPU inference engine that runs [`GpuLLMEngine`] on a background
    /// tokio task and exposes an async streaming interface.
    ///
    /// The background task runs a continuous step loop:
    /// 1. Drain pending commands and generation requests from mpsc channels
    /// 2. If the engine has unfinished work, call `step()`
    /// 3. Fan out `RequestOutput`s to per-request streaming channels
    /// 4. If idle, block on the channels until new work arrives
    ///
    /// Thread safety: the public methods (`generate`, `add_request`,
    /// `abort_request`, `shutdown`) are all `&self` and safe to call from
    /// any tokio task. The `GpuLLMEngine` itself lives exclusively on the
    /// background task and is never shared.
    pub struct AsyncGpuLLMEngine {
        cmd_tx: mpsc::Sender<GpuEngineCommand>,
        gen_tx: mpsc::Sender<GpuEngineRequest>,
        cancel: CancellationToken,
    }

    impl AsyncGpuLLMEngine {
        /// Create a new async GPU engine.
        ///
        /// Constructs a [`GpuLLMEngine`] from the provided config and spawns
        /// a background tokio task to drive the step loop.
        pub async fn new(config: EngineConfig) -> Result<Self> {
            let engine = GpuLLMEngine::new(config)?;
            let cancel = CancellationToken::new();
            let (cmd_tx, cmd_rx) = mpsc::channel::<GpuEngineCommand>(256);
            let (gen_tx, gen_rx) = mpsc::channel::<GpuEngineRequest>(256);

            let cancel_bg = cancel.clone();
            tokio::spawn(async move {
                Self::background_loop(engine, cmd_rx, gen_rx, cancel_bg).await;
            });

            info!("AsyncGpuLLMEngine started background task");
            Ok(Self {
                cmd_tx,
                gen_tx,
                cancel,
            })
        }

        /// Submit a generation request and receive a stream of incremental outputs.
        ///
        /// Returns `(RequestId, Stream)` where the stream yields `RequestOutput`
        /// as the engine produces tokens. The stream closes when the request
        /// finishes (hit max tokens, stop string, or EOS).
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
                    sampling_params: params,
                    output_tx,
                    id_tx,
                })
                .await
                .map_err(|_| {
                    LLMError::GpuError("GPU engine background task stopped".into())
                })?;

            let request_id = id_rx
                .await
                .map_err(|_| LLMError::GpuError("request ID channel dropped".into()))??;

            Ok((request_id, ReceiverStream::new(output_rx)))
        }

        /// Add a request with an explicit ID, without streaming.
        ///
        /// Outputs accumulate inside the engine; use `step()` indirectly
        /// through the background loop. Primarily useful for batch / non-streaming
        /// callers that already track their own request IDs.
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
                    sampling_params: params,
                    response_tx: resp_tx,
                })
                .await
                .map_err(|_| {
                    LLMError::GpuError("GPU engine background task stopped".into())
                })?;

            resp_rx
                .await
                .map_err(|_| LLMError::GpuError("response channel dropped".into()))?
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
        // Background loop
        // -------------------------------------------------------------------

        async fn background_loop(
            mut engine: GpuLLMEngine,
            mut cmd_rx: mpsc::Receiver<GpuEngineCommand>,
            mut gen_rx: mpsc::Receiver<GpuEngineRequest>,
            cancel: CancellationToken,
        ) {
            let mut output_channels: HashMap<RequestId, mpsc::Sender<RequestOutput>> =
                HashMap::new();

            loop {
                if cancel.is_cancelled() {
                    info!("GPU background loop cancelled, draining");
                    break;
                }

                // -- Drain pending commands (non-blocking) --
                Self::drain_commands(&mut engine, &mut cmd_rx, &mut output_channels);

                // -- Drain pending generate requests (non-blocking) --
                Self::drain_generate_requests(&mut engine, &mut gen_rx, &mut output_channels);

                // -- If idle, block until new work arrives --
                if !engine.has_unfinished() {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            info!("GPU background loop cancelled while idle");
                            break;
                        }
                        cmd = cmd_rx.recv() => {
                            if !Self::handle_command(&mut engine, cmd, &mut output_channels) {
                                break;
                            }
                        }
                        gen = gen_rx.recv() => {
                            if let Some(req) = gen {
                                Self::handle_generate_request(&mut engine, req, &mut output_channels);
                            } else {
                                break;
                            }
                        }
                    }
                    continue;
                }

                // -- Run one engine step --
                match engine.step() {
                    Ok(outputs) => {
                        for output in outputs {
                            let rid = output.request_id;
                            let finished = output.finished;

                            if let Some(tx) = output_channels.get(&rid) {
                                if tx.send(output).await.is_err() {
                                    debug!(%rid, "GPU output receiver dropped, aborting");
                                    engine.abort_request(&rid);
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

                // Yield to let other tokio tasks run
                tokio::task::yield_now().await;
            }

            // Notify remaining streaming clients that we're shutting down
            for (rid, tx) in output_channels.drain() {
                warn!(%rid, "GPU engine shutting down with in-flight request");
                drop(tx);
            }

            info!("GPU background loop exited");
        }

        // -------------------------------------------------------------------
        // Helpers to reduce duplication in the loop
        // -------------------------------------------------------------------

        /// Drain all pending commands from the channel without blocking.
        fn drain_commands(
            engine: &mut GpuLLMEngine,
            cmd_rx: &mut mpsc::Receiver<GpuEngineCommand>,
            output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
        ) {
            loop {
                match cmd_rx.try_recv() {
                    Ok(GpuEngineCommand::AddRequest {
                        request_id,
                        prompt,
                        sampling_params,
                        response_tx,
                    }) => {
                        let result =
                            engine.add_request(request_id, prompt, sampling_params);
                        let _ = response_tx.send(result);
                    }
                    Ok(GpuEngineCommand::AbortRequest { request_id }) => {
                        engine.abort_request(&request_id);
                        output_channels.remove(&request_id);
                    }
                    Ok(GpuEngineCommand::Shutdown) => {
                        info!("GPU background loop received shutdown via drain");
                        return;
                    }
                    Err(_) => break,
                }
            }
        }

        /// Drain all pending generate requests from the channel without blocking.
        fn drain_generate_requests(
            engine: &mut GpuLLMEngine,
            gen_rx: &mut mpsc::Receiver<GpuEngineRequest>,
            output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
        ) {
            loop {
                match gen_rx.try_recv() {
                    Ok(req) => {
                        Self::handle_generate_request(engine, req, output_channels);
                    }
                    Err(_) => break,
                }
            }
        }

        /// Process a single generate request: add to engine, send back the ID.
        fn handle_generate_request(
            engine: &mut GpuLLMEngine,
            req: GpuEngineRequest,
            output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
        ) {
            match engine.add_request_auto_id(req.prompt, req.sampling_params) {
                Ok(rid) => {
                    let _ = req.id_tx.send(Ok(rid));
                    output_channels.insert(rid, req.output_tx);
                }
                Err(e) => {
                    error!(%e, "failed to add GPU generate request");
                    let _ = req.id_tx.send(Err(e));
                }
            }
        }

        /// Handle a single received command. Returns false if the loop should exit.
        fn handle_command(
            engine: &mut GpuLLMEngine,
            cmd: Option<GpuEngineCommand>,
            output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
        ) -> bool {
            match cmd {
                Some(GpuEngineCommand::AddRequest {
                    request_id,
                    prompt,
                    sampling_params,
                    response_tx,
                }) => {
                    let result =
                        engine.add_request(request_id, prompt, sampling_params);
                    let _ = response_tx.send(result);
                    true
                }
                Some(GpuEngineCommand::AbortRequest { request_id }) => {
                    engine.abort_request(&request_id);
                    output_channels.remove(&request_id);
                    true
                }
                Some(GpuEngineCommand::Shutdown) | None => false,
            }
        }
    }

    impl Drop for AsyncGpuLLMEngine {
        fn drop(&mut self) {
            self.cancel.cancel();
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::AsyncGpuLLMEngine;
