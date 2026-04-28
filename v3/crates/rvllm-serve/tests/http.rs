//! HTTP integration tests.
//!
//! Drives the router with `tower::ServiceExt::oneshot` — no TCP
//! socket, no real CUDA worker, but exercises the full axum layer
//! stack (TraceLayer, state injection, handler, `IntoResponse`,
//! error envelope).
//!
//! The fixture uses the real `TokenizerHandle::load` path for
//! everything that actually tokenizes (chat/completions), so each
//! test that needs it skips gracefully when `RVLLM_TEST_MODEL_DIR`
//! isn't pointed at an HF model directory. Tests that don't need a
//! tokenizer (health, /v1/models, unknown-model 404) build state via
//! the same fixture but still need the model dir for
//! `TokenizerHandle::load` — there isn't a built-in null tokenizer
//! upstream, and the tests only cost an extra load at startup.
//!
//! Run locally against Gemma 4:
//!
//! ```bash
//! RVLLM_TEST_MODEL_DIR=/home/r00t/.vllm/models/gemma-4-31b-it-fp8-block \
//!   cargo test -p rvllm-serve --test http
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::to_bytes;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use rvllm_serve::{build_router, worker::spawn_mock_worker, AppState, ServerConfig};
use tower::ServiceExt;

/// 1 MiB response-body cap for tests. Mock worker produces tiny
/// payloads; anything over this signals a bug (infinite stream,
/// runaway SSE loop).
const BODY_CAP: usize = 1024 * 1024;

fn model_dir() -> Option<PathBuf> {
    std::env::var_os("RVLLM_TEST_MODEL_DIR").map(PathBuf::from)
}

/// Build a full AppState + Router backed by the mock worker. Returns
/// `None` (tests print skip + early-return) when there is no test
/// model directory on this machine.
fn try_build_state() -> Option<(AppState, std::thread::JoinHandle<()>)> {
    let dir = model_dir()?;
    let model_id = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "rvllm-test".into());
    let config = ServerConfig {
        bind: ([127, 0, 0, 1], 0).into(),
        model_dir: dir.clone(),
        model_id,
        max_queue_depth: 2,
        max_new_tokens_cap: 32,
        request_timeout: Duration::from_secs(10),
        sse_keepalive: Duration::from_secs(15),
        shutdown_drain_timeout: Duration::from_secs(5),
    };
    config.validate().expect("test config valid");

    let tokenizer = rvllm_serve::tokenize::TokenizerHandle::load(&dir).expect("tokenizer");
    let (worker, join) = spawn_mock_worker(config.max_queue_depth);
    let state = AppState {
        config: Arc::new(config),
        tokenizer,
        worker,
        started_at: 0,
    };
    Some((state, join))
}

macro_rules! need_model {
    () => {
        match try_build_state() {
            Some(v) => v,
            None => {
                eprintln!("skip: RVLLM_TEST_MODEL_DIR not set");
                return;
            }
        }
    };
}

async fn send<B: Into<axum::body::Body>>(
    router: axum::Router,
    method: Method,
    path: &str,
    body: B,
) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(body.into())
        .expect("request builds");
    let resp = router.oneshot(req).await.expect("router responds");
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = to_bytes(resp.into_body(), BODY_CAP)
        .await
        .expect("body collect")
        .to_vec();
    (status, headers, bytes)
}

// ─── /health ────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn health_endpoint() {
    let (state, _join) = need_model!();
    let router = build_router(state);
    let (status, _h, body) = send(router, Method::GET, "/health", axum::body::Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["status"], "ok");
}

// ─── /v1/models ─────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_models_returns_configured_model() {
    let (state, _join) = need_model!();
    let expected = state.config.model_id.clone();
    let router = build_router(state);
    let (status, _h, body) = send(router, Method::GET, "/v1/models", axum::body::Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], expected.as_str());
    assert_eq!(v["data"][0]["object"], "model");
}

// ─── Error paths ────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_unknown_model_returns_404() {
    let (state, _join) = need_model!();
    let router = build_router(state);
    let body = r#"{"model":"wrong","messages":[{"role":"user","content":"hi"}],"temperature":0}"#;
    let (status, _h, body) = send(router, Method::POST, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap_or_default().contains("wrong"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_empty_messages_returns_400() {
    let (state, _join) = need_model!();
    let model = state.config.model_id.clone();
    let router = build_router(state);
    let body = format!(r#"{{"model":"{model}","messages":[],"temperature":0}}"#);
    let (status, _h, body) = send(router, Method::POST, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["error"]["code"], "empty_messages");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_non_greedy_sampling_coerces_silently() {
    // v1 runtime is greedy-only, but OpenAI-default sampling params
    // (temperature=1.0 / top_p=1.0) must not 400. They coerce to
    // greedy with a one-shot WARN log. Negative / out-of-range
    // values still 400; those are covered by sampling.rs unit tests.
    let (state, _join) = need_model!();
    let model = state.config.model_id.clone();
    let router = build_router(state);
    let body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}],"temperature":0.7}}"#,
    );
    let (status, _h, _body) = send(router, Method::POST, "/v1/chat/completions", body).await;
    // Not 400. The worker may fail for a different reason in the
    // test harness (no real model) but the sampling check itself
    // must pass — i.e. the status must NOT be 400 with
    // sampling_unsupported. We assert status != 400 to cover both
    // the happy path and test-harness errors that surface as 500.
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "temperature=0.7 should coerce to greedy, not 400"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_max_tokens_over_cap_returns_400() {
    let (state, _join) = need_model!();
    let model = state.config.model_id.clone();
    let cap = state.config.max_new_tokens_cap;
    let router = build_router(state);
    let body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}],"temperature":0,"max_tokens":{}}}"#,
        cap + 1
    );
    let (status, _h, body) = send(router, Method::POST, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["error"]["code"], "max_tokens_too_large");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_tool_choice_rejected() {
    let (state, _join) = need_model!();
    let model = state.config.model_id.clone();
    let router = build_router(state);
    let body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}],"temperature":0,"tools":[{{"type":"function"}}]}}"#,
    );
    let (status, _h, _body) = send(router, Method::POST, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ─── Happy path against mock worker ─────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_non_stream_happy_path() {
    let (state, _join) = need_model!();
    let model = state.config.model_id.clone();
    let router = build_router(state);
    let body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}],"temperature":0,"max_tokens":4}}"#,
    );
    let (status, _h, body) = send(router, Method::POST, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["object"], "chat.completion");
    assert!(v["id"].as_str().unwrap_or_default().starts_with("chatcmpl-"));
    assert_eq!(v["choices"][0]["index"], 0);
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert!(v["choices"][0]["finish_reason"].is_string());
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completions_non_stream_happy_path() {
    let (state, _join) = need_model!();
    let model = state.config.model_id.clone();
    let router = build_router(state);
    let body = format!(
        r#"{{"model":"{model}","prompt":"once upon","temperature":0,"max_tokens":3}}"#,
    );
    let (status, _h, body) = send(router, Method::POST, "/v1/completions", body).await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["object"], "text_completion");
    assert!(v["id"].as_str().unwrap_or_default().starts_with("cmpl-"));
    assert_eq!(v["choices"][0]["index"], 0);
    assert!(v["choices"][0]["text"].is_string());
}

// ─── SSE streaming ──────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_emits_sse_frames_and_done() {
    let (state, _join) = need_model!();
    let model = state.config.model_id.clone();
    let router = build_router(state);
    let req_body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}],"temperature":0,"max_tokens":2,"stream":true}}"#,
    );
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(req_body))
        .expect("request builds");
    let resp = router.oneshot(req).await.expect("router responds");
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(""),
        "text/event-stream",
    );

    // Collect the SSE body. With max_tokens=2 + mock eos-on-id=1
    // emission, we expect at minimum the terminal finish chunk +
    // `data: [DONE]`.
    let raw = resp
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let s = std::str::from_utf8(&raw).expect("utf8");
    assert!(s.contains("data: "), "no data: prefix in SSE output:\n{s}");
    assert!(s.trim_end().ends_with("data: [DONE]"), "no [DONE] terminator:\n{s}");

    // Every data: chunk up to [DONE] is JSON-parseable.
    for frame in s.split("\n\n").filter(|f| !f.trim().is_empty()) {
        let data = frame
            .lines()
            .find_map(|l| l.strip_prefix("data: "))
            .unwrap_or_default();
        if data == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(data).unwrap_or_else(|e| {
            panic!("chunk is not JSON: {data:?} err={e}")
        });
        assert_eq!(v["object"], "chat.completion.chunk");
    }
}

// ─── Malformed JSON ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_garbage_json_returns_400() {
    let (state, _join) = need_model!();
    let router = build_router(state);
    let (status, _h, _b) = send(
        router,
        Method::POST,
        "/v1/chat/completions",
        "{not json".to_string(),
    )
    .await;
    // axum::Json extractor returns 400 on parse failure by default.
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
