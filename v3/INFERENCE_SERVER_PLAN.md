# rvLLM Inference Server — design + phase plan

**Target:** `rvllm-serve` crate ships a binary `rvllm-server` exposing an
OpenAI-compatible HTTP API backed by the sm_121 GB10 decode path
(landed on `rusty_sm121`, ~5.2 tok/s on Gemma 4 31B fp8-block). Robust
construction, idiomatic Rust + Tokio + axum, streaming SSE, typed
errors end-to-end, one worker thread, no unwraps in the request path.

## Non-goals (v1)

- **Continuous batching.** `Gemma4Bringup::run_generate` is single-seq.
  True continuous batching requires a scheduler + paged attention
  across live sequences + per-seq KV block tables — runtime refactor
  explicitly out of scope. v1 serialises requests through one worker.
- **Paged KV eviction / preemption.** One request at a time means no
  preemption; a long request blocks the queue. Admission control
  returns 429 when queue depth exceeds a bound (configurable).
- **Multi-model hosting.** One engine instance, one model.
- **Prefix caching, speculative decoding, LoRA.** Follow-ups.
- **Authentication / rate limiting beyond depth.** Expected to live
  behind a reverse proxy (nginx, caddy).

## API coverage (v1)

| Endpoint                        | Method | Streaming | Status |
|---------------------------------|--------|-----------|--------|
| `/v1/models`                    | GET    | no        | full   |
| `/v1/chat/completions`          | POST   | SSE       | full   |
| `/v1/completions`               | POST   | SSE       | full   |
| `/health`                       | GET    | no        | full   |
| `/v1/embeddings`                | POST   | no        | **out-of-scope** (embedding model is a separate service) |
| tool-calling, function-calling  | —      | —         | **out-of-scope** v1 |

OpenAI request/response schemas are typed via `serde` structs. Unknown
fields on request: `#[serde(default)]` + ignored. Unknown fields on
response: never — we only emit spec'd fields.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  tokio runtime (axum)                                            │
│                                                                  │
│   HTTP handler ──┐                                               │
│                  │ 1. parse request                              │
│                  │ 2. validate (Result<_, HttpError>)            │
│                  │ 3. tokenize prompt (tokenizers crate, CPU)    │
│                  │ 4. render chat template (minijinja)          │
│                  │ 5. tx.send(GenerateRequest)  ◀──── admission │
│                  │                                               │
│   SSE response  ◀┤ 6. stream tokens from rx as they land         │
└──────────────────┼───────────────────────────────────────────────┘
                   │ tokio::sync::mpsc
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  dedicated OS thread (Worker: !Send)                             │
│                                                                  │
│   loop {                                                         │
│     req = rx.recv();                        // blocking          │
│     for token in bringup.run_generate(...): {                    │
│         req.tokens_tx.send(token);          // backpressure-aware│
│         if req.cancelled.load(): break;     // client disconnect │
│     }                                                            │
│     req.done.send(Ok/Err);                                       │
│   }                                                              │
└──────────────────────────────────────────────────────────────────┘
```

### Why an OS thread, not a tokio task?

`Worker` owns `Gemma4Bringup` which owns CUDA context + stream. The
concurrency spec (`v3/specs/05-concurrency.md`) declares
`Worker: !Send`; moving it across `.await` points is a bug. A dedicated
thread + blocking mpsc recv is the correct bridge pattern.

### Cancellation

`GenerateRequest` carries an `Arc<AtomicBool>` cancellation flag. The
axum handler flips it on client disconnect (via `axum::extract::Request`
drop / SSE channel close). The worker polls the flag between tokens
and breaks cleanly. KV cache is per-request today (single-seq); a
cancelled request frees its KV on break, no state leak.

### Error mapping

Every error crosses the HTTP boundary as a typed `ApiError` that
converts to an OpenAI-style JSON body:

```json
{ "error": { "message": "...", "type": "invalid_request_error",
             "param": null, "code": "invalid_prompt" } }
```

| RvllmError / internal        | HTTP  | OpenAI `error.type`         |
|------------------------------|-------|-----------------------------|
| bad request (missing field)  | 400   | `invalid_request_error`     |
| unknown model                | 404   | `invalid_request_error`     |
| context too long             | 400   | `invalid_request_error`     |
| worker queue full            | 429   | `rate_limit_exceeded`       |
| cancelled mid-stream         | 499   | (client-side; no body)      |
| CUDA / kernel launch failure | 500   | `server_error`              |
| model not loaded             | 503   | `service_unavailable`       |

**No unwraps in the request path.** Library code already enforces
`Result<T, RvllmError>` end-to-end; the HTTP layer adds `ApiError`
with `From<RvllmError>`.

## Crate layout (`v3/crates/rvllm-serve/src/`)

```
main.rs           binary entrypoint + CLI (clap)
lib.rs            pub re-exports for integration tests
config.rs         ServerConfig (bind addr, model dir, queue depth, limits)
error.rs          ApiError + IntoResponse + From<RvllmError>
worker.rs         WorkerHandle (tokio side) + worker_loop (thread side)
router.rs         axum Router construction, middleware, state
tokenize.rs       TokenizerHandle (load tokenizer.json + chat template)
sampling.rs       SamplingParams (temperature, top_k, top_p, seed)
openai/
  mod.rs
  types.rs        shared primitives (Role, Usage, FinishReason)
  models.rs       /v1/models — static list from loaded model
  chat.rs         ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionChunk (SSE). Handler + stream builder.
  completions.rs  CompletionRequest, CompletionResponse,
                  CompletionChunk. Same pattern.
```

## Types (sketch)

```rust
// Across-thread boundary payload.
pub struct GenerateRequest {
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub max_new_tokens: u32,
    pub stop_token_ids: Vec<u32>,
    pub tokens_tx: mpsc::Sender<GenerateEvent>,
    pub cancelled: Arc<AtomicBool>,
    pub request_id: Uuid,
}

pub enum GenerateEvent {
    Token { id: u32, pos: u32 },
    Done { finish: FinishReason, usage: Usage },
    Error(RvllmError),
}
```

`SamplingParams` v1 = greedy only (argmax) because that's what
`run_generate` exposes today. Stub temperature/top_k fields parsed
from the request but return 501 if non-default values are set —
honest about the limitation. Sampling upgrade is its own follow-up
once a sampling kernel lands.

## Phase plan

| Phase | Scope                                                      | Deliverable                               |
|-------|------------------------------------------------------------|-------------------------------------------|
| 0     | **this document**                                          | `v3/INFERENCE_SERVER_PLAN.md`             |
| 1     | Cargo deps + module skeleton + typed OpenAI schemas        | `cargo check -p rvllm-serve` clean, no endpoint logic yet |
| 2     | Worker thread + tokio bridge (echo test, no real model)    | Unit test: send prompt, receive back canned tokens |
| 3     | Tokenizer + Gemma 4 chat template                          | `encode_chat(messages)` + `decode_stream(ids) -> Utf8Stream`, round-trip test against a known prompt |
| 4     | `/v1/models` + `/v1/chat/completions` non-streaming        | `curl` against local server returns a completion from Gemma 4 |
| 5     | SSE streaming on `/v1/chat/completions`                    | `curl -N` shows token-by-token output, first token within decode latency |
| 6     | `/v1/completions` (text path)                              | OpenAI SDK's `Completion.create` works |
| 7     | Admission control + cancellation + graceful shutdown       | 429 on queue full, `Ctrl-C` drains in-flight, client disconnect cancels mid-stream |
| 8     | Integration tests (`tests/` dir)                           | `cargo test -p rvllm-serve` against a mock worker, no CUDA |
| 9     | Sampling (temperature, top_p, top_k)                       | **deferred** — depends on a sampling kernel upgrade |

Each phase ends with a green `cargo check` + a commit. Phases 4-6
need a GB10 machine to verify end-to-end; mock-worker tests cover
everything below on CI.

## Dependencies to add

```toml
axum = { version = "0.8", features = ["macros", "http2"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "cors", "request-id"] }
tokio = { workspace = true, features = ["rt-multi-thread", "macros", "signal", "sync"] }
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
tokenizers = { version = "0.21", default-features = false, features = ["onig"] }
minijinja = { version = "2.5", features = ["loader"] }
uuid = { version = "1.11", features = ["v4", "serde"] }
clap = { version = "4", features = ["derive"] }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
futures = "0.3"
thiserror = { workspace = true }
```

## Invariants we'll enforce

1. **One worker thread owns Gemma4Bringup.** Compiler-enforced via
   `!Send`; the spawn helper returns a `WorkerHandle` that only
   holds `mpsc::Sender`s.
2. **Request admission is bounded.** `mpsc::channel(queue_depth)` —
   full channel → 429 response. Never unbounded.
3. **Streaming respects backpressure.** `tokens_tx` is bounded; the
   worker blocks on send, which naturally throttles generation to
   client read speed.
4. **Cancellation is prompt (≤ 1 token).** Poll `cancelled` flag
   between `bringup.forward_one_token()` calls inside the worker.
5. **No unwraps in `src/`.** `#![deny(clippy::unwrap_used)]` +
   allow-list for `main.rs` startup paths only.
6. **Tokenizer lives in Arc — shared across handlers.** No re-load per
   request.
7. **All OpenAI response JSON round-trips through the official schema
   test fixtures** (captured from ChatGPT API responses).

## Open questions deferred for Phase 1

- Do we want a `/metrics` Prometheus endpoint? (prob yes, `axum_prometheus` crate)
- Do we expose a `/v1/internal/probe` health check with full model info?
- Logging: JSON structured (prod-friendly) or pretty (dev)?
  `tracing-subscriber` supports both via env var.

These are easy additions once the main path lands.
