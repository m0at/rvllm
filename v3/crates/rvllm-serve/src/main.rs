use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rvllm_fused::M2PrefillKvDType;
use rvllm_xla::{plan_m2_rust_prefill_decode, M2RustPrefillDecodeConfig, M2RustPrefillDecodePlan};
use serde::Deserialize;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    let plan = plan_m2_rust_prefill_decode(&M2RustPrefillDecodeConfig {
        model_dir: args.model_dir.clone(),
        batch: args.batch,
        prompt_len: args.prompt_len,
        decode_steps: args.decode_steps,
        ctx: args.max_ctx,
        block_size: args.block_size,
        kv_dtype: M2PrefillKvDType::Int8,
    })?;
    let listener = TcpListener::bind((args.host.as_str(), args.port))?;
    eprintln!(
        "rvllm-server m2: listening on {}:{} model={} batch={} ctx={} decode_steps={}",
        args.host, args.port, args.model_name, args.batch, args.max_ctx, args.decode_steps
    );
    let state = ServerState {
        model_name: args.model_name,
        max_ctx: args.max_ctx,
        plan,
    };
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(err) = handle_stream(stream, &state) {
                    eprintln!("request error: {err}");
                }
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }
    Ok(())
}

struct ServerState {
    model_name: String,
    max_ctx: usize,
    plan: M2RustPrefillDecodePlan,
}

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    prompt_token_ids: Option<Vec<i32>>,
    max_tokens: Option<usize>,
    stream: Option<bool>,
    temperature: Option<f64>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

struct Args {
    model_dir: PathBuf,
    host: String,
    port: u16,
    model_name: String,
    batch: usize,
    prompt_len: usize,
    decode_steps: usize,
    max_ctx: usize,
    block_size: u32,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            model_dir: PathBuf::from("/dev/shm/m2-nvfp4"),
            host: "0.0.0.0".to_string(),
            port: 8080,
            model_name: "MiniMax-M2.7-NVFP4".to_string(),
            batch: 8,
            prompt_len: 20,
            decode_steps: 256,
            max_ctx: 2048,
            block_size: 32,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--model-dir" => {
                    i += 1;
                    out.model_dir = PathBuf::from(value(&args, i, "--model-dir")?);
                }
                "--host" => {
                    i += 1;
                    out.host = value(&args, i, "--host")?.to_string();
                }
                "--port" => {
                    i += 1;
                    out.port = parse(value(&args, i, "--port")?, "--port")?;
                }
                "--model-name" => {
                    i += 1;
                    out.model_name = value(&args, i, "--model-name")?.to_string();
                }
                "--batch" | "--batch-size" => {
                    i += 1;
                    out.batch = parse(value(&args, i, "--batch")?, "--batch")?;
                }
                "--prompt-len" => {
                    i += 1;
                    out.prompt_len = parse(value(&args, i, "--prompt-len")?, "--prompt-len")?;
                }
                "--decode-steps" => {
                    i += 1;
                    out.decode_steps = parse(value(&args, i, "--decode-steps")?, "--decode-steps")?;
                }
                "--max-ctx" | "--ctx" => {
                    i += 1;
                    out.max_ctx = parse(value(&args, i, "--max-ctx")?, "--max-ctx")?;
                }
                "--block-size" => {
                    i += 1;
                    out.block_size = parse(value(&args, i, "--block-size")?, "--block-size")?;
                }
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown arg {other:?}\n{}", usage())),
            }
            i += 1;
        }
        Ok(out)
    }
}

fn handle_stream(
    mut stream: TcpStream,
    state: &ServerState,
) -> Result<(), Box<dyn std::error::Error>> {
    let req = read_request(&mut stream)?;
    let (status, body) = match (req.method.as_str(), req.path.as_str()) {
        ("OPTIONS", _) => (204, json!({})),
        ("GET", "/health") => (
            200,
            json!({
                "status": "ok",
                "model": state.model_name,
                "runtime": "rust-m2-pjrt",
                "decode_graph_bytes": state.plan.decode_mlir.len(),
                "weight_arena_bytes": state.plan.weight_arena_bytes,
            }),
        ),
        ("GET", "/v1/models") => (
            200,
            json!({
                "object": "list",
                "data": [{
                    "id": state.model_name,
                    "object": "model",
                    "created": now_unix(),
                    "owned_by": "rvllm",
                }]
            }),
        ),
        ("POST", "/v1/chat/completions") => handle_chat(&req.body, state),
        _ => (404, error_json(404, "not found")),
    };
    write_response(&mut stream, status, &body)?;
    Ok(())
}

fn handle_chat(body: &[u8], state: &ServerState) -> (u16, serde_json::Value) {
    let req: ChatRequest = match serde_json::from_slice(body) {
        Ok(req) => req,
        Err(err) => return (400, error_json(400, &format!("bad json: {err}"))),
    };
    if req.stream.unwrap_or(false) {
        return (
            400,
            error_json(400, "streaming is not enabled in the Rust M2 server yet"),
        );
    }
    if req.temperature.unwrap_or(0.0) != 0.0 {
        return (
            400,
            error_json(
                400,
                "MiniMax M2 Rust server currently supports temperature=0 only",
            ),
        );
    }
    let prompt_ids = match req.prompt_token_ids {
        Some(ids) if !ids.is_empty() => ids,
        Some(_) => return (400, error_json(400, "prompt_token_ids must not be empty")),
        None => {
            let msg_count = req.messages.as_ref().map(Vec::len).unwrap_or(0);
            let msg_bytes = req
                .messages
                .as_ref()
                .map(|msgs| {
                    msgs.iter()
                        .map(|msg| msg.role.len() + msg.content.len())
                        .sum::<usize>()
                })
                .unwrap_or(0);
            return (
                400,
                error_json(
                    400,
                    &format!(
                        "Rust M2 server needs prompt_token_ids until tokenizer/chat-template is ported; messages={msg_count} bytes={msg_bytes}"
                    ),
                ),
            );
        }
    };
    let max_tokens = req.max_tokens.unwrap_or(state.plan.decode_steps);
    if prompt_ids.len() + max_tokens >= state.max_ctx {
        return (
            400,
            error_json(400, "prompt_token_ids + max_tokens exceeds max_ctx"),
        );
    }
    (
        503,
        json!({
            "error": {
                "message": "Rust M2 HTTP surface is live, but decode custom calls are not executable yet",
                "type": "backend_unavailable",
                "code": 503
            },
            "rvllm": {
                "model": req.model.unwrap_or_else(|| state.model_name.clone()),
                "prompt_tokens": prompt_ids.len(),
                "max_tokens": max_tokens,
                "batch": state.plan.decode_shape.batch,
                "weight_arena_bytes": state.plan.weight_arena_bytes
            }
        }),
    )
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, Box<dyn std::error::Error>> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end;
    loop {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Err("connection closed before headers".into());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_header_end(&buf) {
            header_end = pos;
            break;
        }
        if buf.len() > 64 * 1024 {
            return Err("headers too large".into());
        }
    }
    let header = std::str::from_utf8(&buf[..header_end])?;
    let mut lines = header.split("\r\n");
    let request_line = lines.next().ok_or("missing request line")?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().ok_or("missing method")?.to_string();
    let path = parts.next().ok_or("missing path")?.to_string();
    let mut content_len = 0usize;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_len = value.trim().parse()?;
            }
        }
    }
    let body_start = header_end + 4;
    let mut body = buf[body_start..].to_vec();
    while body.len() < content_len {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Err("connection closed before body".into());
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(content_len);
    Ok(HttpRequest { method, path, body })
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    body: &serde_json::Value,
) -> std::io::Result<()> {
    let body_bytes = if status == 204 {
        Vec::new()
    } else {
        serde_json::to_vec(body).unwrap()
    };
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        503 => "Service Unavailable",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body_bytes.len()
    )?;
    stream.write_all(&body_bytes)
}

fn error_json(code: u16, message: &str) -> serde_json::Value {
    json!({ "error": { "message": message, "type": "invalid_request_error", "code": code } })
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn value<'a>(args: &'a [String], i: usize, name: &str) -> Result<&'a str, String> {
    args.get(i)
        .map(String::as_str)
        .ok_or_else(|| format!("{name}: missing value"))
}

fn parse<T: std::str::FromStr>(s: &str, name: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    s.parse::<T>()
        .map_err(|e| format!("{name}: expected number, got {s:?}: {e}"))
}

fn usage() -> String {
    "usage: rvllm-server --model-dir DIR [--host HOST] [--port PORT] [--model-name NAME] [--batch-size N] [--prompt-len N] [--decode-steps N] [--max-ctx N] [--block-size N]".to_string()
}
