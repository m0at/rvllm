use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use serde_json::Value;

use crate::openai::{
    completion_id, completion_response, created_unix, error_response, models_response,
    prepare_chat_request, stream_content_chunk, stream_finish_chunk, stream_role_chunk,
    stream_text_chunks, ApiError, ChatCompletionRequest,
};
use crate::worker::{GenerateError, GenerateRequest, WorkerHandle};
use crate::ServeConfig;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone)]
struct State {
    config: ServeConfig,
    worker: WorkerHandle,
}

struct Request {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

pub fn serve(config: ServeConfig, worker: WorkerHandle) -> Result<(), String> {
    let addr = config.addr();
    let listener = TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
    tracing::info!("rvllm-server listening on http://{addr}");
    let state = Arc::new(State { config, worker });

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = Arc::clone(&state);
                thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, state) {
                        tracing::debug!("connection error: {e}");
                    }
                });
            }
            Err(e) => tracing::warn!("accept failed: {e}"),
        }
    }
    Ok(())
}

fn handle_connection(mut stream: TcpStream, state: Arc<State>) -> Result<(), String> {
    stream
        .set_nodelay(true)
        .map_err(|e| format!("set TCP_NODELAY: {e}"))?;
    let req = read_request(&mut stream)?;
    let path = req.path.split('?').next().unwrap_or(req.path.as_str());

    match (req.method.as_str(), path) {
        ("OPTIONS", _) => write_empty(&mut stream, 204),
        ("GET", "/health") => write_text(&mut stream, 200, "ok\n"),
        ("GET", "/status") => write_json(&mut stream, 200, &status_response(&state)),
        ("GET", "/v1/models") => write_json(
            &mut stream,
            200,
            &models_response(&state.config.served_model_name),
        ),
        ("POST", "/v1/chat/completions") => handle_chat(stream, state, req),
        _ => write_json(
            &mut stream,
            404,
            &error_response(&ApiError::not_found("not found")),
        ),
    }
}

fn handle_chat(mut stream: TcpStream, state: Arc<State>, req: Request) -> Result<(), String> {
    let content_type = req
        .headers
        .get("content-type")
        .map(String::as_str)
        .unwrap_or("");
    if !content_type.is_empty() && !content_type.contains("application/json") {
        return write_json(
            &mut stream,
            415,
            &error_response(&ApiError::invalid("content-type must be application/json")),
        );
    }

    let chat_req: ChatCompletionRequest = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => {
            return write_json(
                &mut stream,
                400,
                &error_response(&ApiError::invalid(format!("invalid JSON body: {e}"))),
            )
        }
    };

    let prepared = match prepare_chat_request(
        chat_req,
        &state.config.served_model_name,
        256,
        state.config.default_system_prompt.as_deref(),
    ) {
        Ok(p) => p,
        Err(e) => return write_json(&mut stream, e.status, &error_response(&e)),
    };

    if prepared.stream {
        handle_chat_stream(stream, state, prepared)
    } else {
        let created = created_unix();
        let id = completion_id(created);
        match state.worker.generate(GenerateRequest {
            prompt: prepared.prompt,
            max_tokens: prepared.max_tokens,
            stop: prepared.stop,
        }) {
            Ok(out) => write_json(
                &mut stream,
                200,
                &completion_response(
                    &id,
                    &state.config.served_model_name,
                    created,
                    &out.text,
                    out.prompt_tokens,
                    out.completion_tokens,
                ),
            ),
            Err(e) => write_generate_error(&mut stream, e),
        }
    }
}

fn handle_chat_stream(
    mut stream: TcpStream,
    state: Arc<State>,
    prepared: crate::openai::PreparedChat,
) -> Result<(), String> {
    let created = created_unix();
    let id = completion_id(created);
    write_stream_headers(&mut stream)?;
    write_sse_json(
        &mut stream,
        &stream_role_chunk(&id, &state.config.served_model_name, created),
    )?;

    match state.worker.generate(GenerateRequest {
        prompt: prepared.prompt,
        max_tokens: prepared.max_tokens,
        stop: prepared.stop,
    }) {
        Ok(out) => {
            for chunk in stream_text_chunks(&out.text) {
                write_sse_json(
                    &mut stream,
                    &stream_content_chunk(&id, &state.config.served_model_name, created, &chunk),
                )?;
            }
            write_sse_json(
                &mut stream,
                &stream_finish_chunk(&id, &state.config.served_model_name, created),
            )?;
            write_sse_done(&mut stream)
        }
        Err(e) => {
            let value = error_response(&api_error_for_generate(e));
            write_sse_json(&mut stream, &value)?;
            write_sse_done(&mut stream)
        }
    }
}

fn write_generate_error(stream: &mut TcpStream, err: GenerateError) -> Result<(), String> {
    let err = api_error_for_generate(err);
    write_json(stream, err.status, &error_response(&err))
}

fn api_error_for_generate(err: GenerateError) -> ApiError {
    match err {
        GenerateError::Busy { max_inflight } => ApiError::busy(format!(
            "waiting for available inference slot; max_inflight_requests={max_inflight}"
        )),
        GenerateError::Engine(e) => ApiError::internal(e),
    }
}

fn status_response(state: &State) -> Value {
    let stats = state.worker.stats();
    serde_json::json!({
        "object": "rvllm.status",
        "model": state.config.served_model_name,
        "max_model_len": state.config.max_model_len,
        "max_num_seqs": state.config.max_num_seqs,
        "max_inflight_requests": stats.max_inflight,
        "in_flight_requests": stats.in_flight
    })
}

fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        if let Some(idx) = find_header_end(&buf) {
            break idx;
        }
        if buf.len() > MAX_HEADER_BYTES {
            return Err("request headers too large".into());
        }
        let n = stream.read(&mut tmp).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            return Err("connection closed before headers".into());
        }
        buf.extend_from_slice(&tmp[..n]);
    };

    let head = std::str::from_utf8(&buf[..header_end])
        .map_err(|e| format!("request headers are not UTF-8: {e}"))?;
    let mut lines = head.split("\r\n");
    let request_line = lines.next().ok_or("empty request")?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().ok_or("missing method")?.to_string();
    let path = parts.next().ok_or("missing path")?.to_string();
    let version = parts.next().ok_or("missing HTTP version")?;
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(format!("unsupported HTTP version: {version}"));
    }

    let mut headers = HashMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let content_len = headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    if content_len > MAX_BODY_BYTES {
        return Err("request body too large".into());
    }

    let body_start = header_end + 4;
    while buf.len().saturating_sub(body_start) < content_len {
        let n = stream
            .read(&mut tmp)
            .map_err(|e| format!("read body: {e}"))?;
        if n == 0 {
            return Err("connection closed before body".into());
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = buf[body_start..body_start + content_len].to_vec();

    Ok(Request {
        method,
        path,
        headers,
        body,
    })
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn write_empty(stream: &mut TcpStream, status: u16) -> Result<(), String> {
    let head = format!(
        "HTTP/1.1 {}\r\n{}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        status_text(status),
        cors_headers(),
    );
    stream
        .write_all(head.as_bytes())
        .map_err(|e| format!("write response: {e}"))
}

fn write_text(stream: &mut TcpStream, status: u16, body: &str) -> Result<(), String> {
    write_response(stream, status, "text/plain; charset=utf-8", body.as_bytes())
}

fn write_json(stream: &mut TcpStream, status: u16, value: &Value) -> Result<(), String> {
    let body = serde_json::to_vec(value).map_err(|e| format!("json serialize: {e}"))?;
    write_response(stream, status, "application/json", &body)
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let head = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\n{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_text(status),
        content_type,
        cors_headers(),
        body.len(),
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(body))
        .map_err(|e| format!("write response: {e}"))
}

fn write_stream_headers(stream: &mut TcpStream) -> Result<(), String> {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{}\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
        cors_headers(),
    );
    stream
        .write_all(head.as_bytes())
        .map_err(|e| format!("write stream headers: {e}"))
}

fn write_sse_json(stream: &mut TcpStream, value: &Value) -> Result<(), String> {
    let mut frame = String::from("data: ");
    frame.push_str(&serde_json::to_string(value).map_err(|e| format!("json serialize: {e}"))?);
    frame.push_str("\n\n");
    write_chunk(stream, frame.as_bytes())
}

fn write_sse_done(stream: &mut TcpStream) -> Result<(), String> {
    write_chunk(stream, b"data: [DONE]\n\n")?;
    stream
        .write_all(b"0\r\n\r\n")
        .and_then(|_| stream.flush())
        .map_err(|e| format!("finish stream: {e}"))
}

fn write_chunk(stream: &mut TcpStream, bytes: &[u8]) -> Result<(), String> {
    let head = format!("{:X}\r\n", bytes.len());
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(bytes))
        .and_then(|_| stream.write_all(b"\r\n"))
        .and_then(|_| stream.flush())
        .map_err(|e| format!("write stream chunk: {e}"))
}

fn cors_headers() -> &'static str {
    "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: authorization, content-type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS"
}

fn status_text(status: u16) -> String {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        415 => "Unsupported Media Type",
        500 => "Internal Server Error",
        _ => "OK",
    };
    format!("{status} {reason}")
}
