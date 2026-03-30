//! Async inference engine wrapping [`LLMEngine`] with tokio.

use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use rvllm_config::EngineConfig;
use rvllm_core::prelude::{LLMError, RequestId, RequestOutput, Result, SamplingParams, TokenId};
use rvllm_tokenizer::Tokenizer;

use crate::engine::{Executor, LLMEngine, Scheduler};

/// Commands sent from the public async API to the background engine task.
enum EngineCommand {
    AddRequest {
        request_id: RequestId,
        prompt: String,
        prompt_token_ids: Option<Vec<TokenId>>,
        sampling_params: SamplingParams,
        response_tx: oneshot::Sender<Result<()>>,
    },
    AbortRequest {
        request_id: RequestId,
    },
}

/// Internal struct wrapping a generation request with its streaming channel.
struct EngineRequest {
    request_id: RequestId,
    prompt: String,
    prompt_token_ids: Option<Vec<TokenId>>,
    sampling_params: SamplingParams,
    output_tx: mpsc::Sender<RequestOutput>,
}

/// Async inference engine that runs the synchronous [`LLMEngine`] on a
/// background tokio task and exposes an async streaming interface.
pub struct AsyncLLMEngine {
    cmd_tx: mpsc::Sender<EngineCommand>,
    gen_tx: mpsc::Sender<EngineRequest>,
    cancel: CancellationToken,
}

impl AsyncLLMEngine {
    /// Create a new async engine.
    ///
    /// Spawns a background task that runs the step loop.
    pub fn new(
        config: EngineConfig,
        executor: Box<dyn Executor>,
        scheduler: Box<dyn Scheduler>,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let engine = LLMEngine::new(config, executor, scheduler, tokenizer)?;
        let cancel = CancellationToken::new();
        let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>(256);
        let (gen_tx, gen_rx) = mpsc::channel::<EngineRequest>(256);

        let cancel_bg = cancel.clone();
        tokio::spawn(async move {
            Self::background_loop(engine, cmd_rx, gen_rx, cancel_bg).await;
        });

        info!("AsyncLLMEngine started background task");
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
        // Allocate a unique request id based on channel position (simple monotonic)
        let request_id = RequestId(rand_id());
        let (output_tx, output_rx) = mpsc::channel(64);

        self.gen_tx
            .send(EngineRequest {
                request_id,
                prompt,
                prompt_token_ids: None,
                sampling_params: params,
                output_tx,
            })
            .await
            .map_err(|_| LLMError::SchedulerError("engine background task stopped".into()))?;

        Ok((request_id, ReceiverStream::new(output_rx)))
    }

    /// Add a request without streaming -- outputs are accumulated internally.
    pub async fn add_request(
        &self,
        request_id: RequestId,
        prompt: String,
        params: SamplingParams,
    ) -> Result<()> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::AddRequest {
                request_id,
                prompt,
                prompt_token_ids: None,
                sampling_params: params,
                response_tx: resp_tx,
            })
            .await
            .map_err(|_| LLMError::SchedulerError("engine background task stopped".into()))?;

        resp_rx
            .await
            .map_err(|_| LLMError::SchedulerError("response channel dropped".into()))?
    }

    pub async fn generate_token_ids(
        &self,
        prompt: String,
        prompt_token_ids: Vec<TokenId>,
        params: SamplingParams,
    ) -> Result<(RequestId, ReceiverStream<RequestOutput>)> {
        let request_id = RequestId(rand_id());
        let (output_tx, output_rx) = mpsc::channel(64);

        self.gen_tx
            .send(EngineRequest {
                request_id,
                prompt,
                prompt_token_ids: Some(prompt_token_ids),
                sampling_params: params,
                output_tx,
            })
            .await
            .map_err(|_| LLMError::SchedulerError("engine background task stopped".into()))?;

        Ok((request_id, ReceiverStream::new(output_rx)))
    }

    /// Abort a request.
    pub async fn abort_request(&self, request_id: &RequestId) {
        let _ = self
            .cmd_tx
            .send(EngineCommand::AbortRequest {
                request_id: *request_id,
            })
            .await;
    }

    /// Graceful shutdown: cancel the background loop and wait for it to drain.
    pub fn shutdown(&self) -> Result<()> {
        info!("AsyncLLMEngine shutting down");
        self.cancel.cancel();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Background loop
    // -----------------------------------------------------------------------

    async fn background_loop(
        mut engine: LLMEngine,
        mut cmd_rx: mpsc::Receiver<EngineCommand>,
        mut gen_rx: mpsc::Receiver<EngineRequest>,
        cancel: CancellationToken,
    ) {
        use std::collections::HashMap;

        // Streaming output channels keyed by request id
        let mut output_channels: HashMap<RequestId, mpsc::Sender<RequestOutput>> = HashMap::new();

        loop {
            if cancel.is_cancelled() {
                info!("background loop cancelled, draining");
                break;
            }

            // Drain all pending commands without blocking
            loop {
                match cmd_rx.try_recv() {
                    Ok(EngineCommand::AddRequest {
                        request_id,
                        prompt,
                        prompt_token_ids,
                        sampling_params,
                        response_tx,
                    }) => {
                        let result = if let Some(prompt_token_ids) = prompt_token_ids {
                            engine.add_request_with_token_ids(
                                request_id,
                                prompt,
                                prompt_token_ids,
                                sampling_params,
                            )
                        } else {
                            engine.add_request(request_id, prompt, sampling_params)
                        };
                        let _ = response_tx.send(result);
                    }
                    Ok(EngineCommand::AbortRequest { request_id }) => {
                        engine.abort_request(&request_id);
                        output_channels.remove(&request_id);
                    }
                    Err(_) => break,
                }
            }

            // Drain all pending generate requests
            loop {
                match gen_rx.try_recv() {
                    Ok(req) => {
                        let rid = req.request_id;
                        let result = if let Some(prompt_token_ids) = req.prompt_token_ids {
                            engine.add_request_with_token_ids(
                                rid,
                                req.prompt,
                                prompt_token_ids,
                                req.sampling_params,
                            )
                        } else {
                            engine.add_request(rid, req.prompt, req.sampling_params)
                        };
                        if let Err(e) = result {
                            error!(%rid, %e, "failed to add generate request");
                            continue;
                        }
                        output_channels.insert(rid, req.output_tx);
                    }
                    Err(_) => break,
                }
            }

            // If nothing to do, yield and wait briefly for new work
            if !engine.has_unfinished() {
                // Wait for a new command or generate request, or cancellation
                tokio::select! {
                    _ = cancel.cancelled() => {
                        info!("background loop cancelled while idle");
                        break;
                    }
                    cmd = cmd_rx.recv() => {
                        if let Some(cmd) = cmd {
                            match cmd {
                                EngineCommand::AddRequest {
                                    request_id,
                                    prompt,
                                    prompt_token_ids,
                                    sampling_params,
                                    response_tx,
                                } => {
                                    let result = if let Some(prompt_token_ids) = prompt_token_ids {
                                        engine.add_request_with_token_ids(
                                            request_id,
                                            prompt,
                                            prompt_token_ids,
                                            sampling_params,
                                        )
                                    } else {
                                        engine.add_request(request_id, prompt, sampling_params)
                                    };
                                    let _ = response_tx.send(result);
                                }
                                EngineCommand::AbortRequest { request_id } => {
                                    engine.abort_request(&request_id);
                                    output_channels.remove(&request_id);
                                }
                            }
                        } else {
                            // All senders dropped
                            break;
                        }
                    }
                    gen = gen_rx.recv() => {
                        if let Some(req) = gen {
                            let rid = req.request_id;
                            let result = if let Some(prompt_token_ids) = req.prompt_token_ids {
                                engine.add_request_with_token_ids(
                                    rid,
                                    req.prompt,
                                    prompt_token_ids,
                                    req.sampling_params,
                                )
                            } else {
                                engine.add_request(rid, req.prompt, req.sampling_params)
                            };
                            if let Err(e) = result {
                                error!(%rid, %e, "failed to add generate request");
                            } else {
                                output_channels.insert(rid, req.output_tx);
                            }
                        } else {
                            break;
                        }
                    }
                }
                continue;
            }

            // `engine.step()` bridges into async executor code via `Handle::block_on`.
            // Run it inside `block_in_place` so the background task can safely call
            // that sync wrapper on Tokio's multi-thread runtime.
            match tokio::task::block_in_place(|| engine.step()) {
                Ok(outputs) => {
                    for output in outputs {
                        let rid = output.request_id;
                        let finished = output.finished;

                        if let Some(tx) = output_channels.get(&rid) {
                            if tx.send(output).await.is_err() {
                                debug!(%rid, "output receiver dropped, aborting");
                                engine.abort_request(&rid);
                                output_channels.remove(&rid);
                                continue;
                            }
                        }

                        if finished {
                            output_channels.remove(&rid);
                        }
                    }
                }
                Err(e) => {
                    error!(%e, "engine step failed");
                    // Continue running -- transient errors shouldn't kill the loop
                }
            }

            // Yield to let other tasks run
            tokio::task::yield_now().await;
        }

        info!("background loop exited");
    }
}

/// Simple random-ish ID generator (no external rand dependency needed at runtime).
fn rand_id() -> u64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    // Mix in thread id for uniqueness across concurrent callers
    let tid = std::thread::current().id();
    let tid_hash = format!("{:?}", tid).len() as u64;
    (nanos as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(tid_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ExecutorAdapter, ExecutorInput, SamplerOutput, SchedulerOutputs};
    use rvllm_executor::ExecutorConfig;
    use rvllm_sequence::SequenceGroup;
    use tokio_stream::StreamExt;

    // Reuse mock types from engine tests

    struct MockScheduler {
        groups: Vec<SequenceGroup>,
    }

    impl MockScheduler {
        fn new() -> Self {
            Self { groups: Vec::new() }
        }
    }

    impl Scheduler for MockScheduler {
        fn add_seq_group(&mut self, seq_group: SequenceGroup) {
            self.groups.push(seq_group);
        }

        fn abort_seq_group(&mut self, request_id: &RequestId) {
            self.groups.retain(|g| g.request_id != *request_id);
        }

        fn schedule(&mut self) -> SchedulerOutputs {
            let groups = self.groups.clone();
            self.groups.retain(|g| !g.is_finished());
            let num_tokens = groups
                .iter()
                .flat_map(|g| g.get_seqs())
                .map(|s| s.num_new_tokens().max(1))
                .sum();
            SchedulerOutputs {
                scheduled_seq_groups: groups,
                num_batched_tokens: num_tokens,
                preempted: false,
            }
        }

        fn has_unfinished_seqs(&self) -> bool {
            !self.groups.is_empty()
        }

        fn get_num_unfinished_seq_groups(&self) -> usize {
            self.groups.len()
        }
    }

    struct MockExecutor {
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockExecutor {
        fn new() -> Self {
            Self {
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl Executor for MockExecutor {
        fn execute_model(&mut self, input: ExecutorInput) -> Result<Vec<SamplerOutput>> {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut outputs = Vec::new();
            for meta in &input.seq_group_metadata {
                for &seq_id in meta.seq_data.keys() {
                    outputs.push(SamplerOutput {
                        seq_id,
                        token_id: 1,
                        logprob: -0.5,
                        top_logprobs: None,
                    });
                }
            }
            Ok(outputs)
        }
    }

    fn make_test_tokenizer() -> Tokenizer {
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::Tokenizer as HfTokenizer;

        let mut vocab = std::collections::HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert(" ".to_string(), 2);
        vocab.insert("!".to_string(), 3);
        vocab.insert("[UNK]".to_string(), 4);

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();

        let mut hf = HfTokenizer::new(bpe);
        hf.with_pre_tokenizer(Some(Whitespace {}));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        hf.save(&path, false).unwrap();
        Tokenizer::from_file(&path).unwrap()
    }

    #[tokio::test]
    async fn async_engine_shutdown() {
        let config = EngineConfig::default();
        let tokenizer = make_test_tokenizer();
        let scheduler = Box::new(MockScheduler::new());
        let executor = Box::new(MockExecutor::new());

        let engine = AsyncLLMEngine::new(config, executor, scheduler, tokenizer).unwrap();
        assert!(engine.shutdown().is_ok());
    }

    #[tokio::test]
    async fn async_engine_add_request() {
        let config = EngineConfig::default();
        let tokenizer = make_test_tokenizer();
        let scheduler = Box::new(MockScheduler::new());
        let executor = Box::new(MockExecutor::new());

        let engine = AsyncLLMEngine::new(config, executor, scheduler, tokenizer).unwrap();

        let mut params = SamplingParams::default();
        params.max_tokens = 2;

        let result = engine
            .add_request(RequestId(1), "hello".to_string(), params)
            .await;
        assert!(result.is_ok());

        engine.shutdown().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn async_engine_generate_with_executor_adapter_does_not_panic_in_tokio_runtime() {
        let config = EngineConfig::default();
        let tokenizer = make_test_tokenizer();
        let scheduler = Box::new(MockScheduler::new());
        let rt = tokio::runtime::Handle::current();
        let executor = Box::new(
            ExecutorAdapter::from_config(
                ExecutorConfig {
                    num_gpus: 1,
                    model_name: "test-model".into(),
                    ..ExecutorConfig::default()
                },
                rt,
            )
            .unwrap(),
        );

        let engine = AsyncLLMEngine::new(config, executor, scheduler, tokenizer).unwrap();

        let mut params = SamplingParams::default();
        params.max_tokens = 1;

        let (_request_id, mut stream) = engine.generate("hello".to_string(), params).await.unwrap();
        let output = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
            .await
            .expect("timed out waiting for generated output")
            .expect("output stream ended unexpectedly");

        assert!(!output.outputs.is_empty());
        engine.shutdown().unwrap();
    }
}
