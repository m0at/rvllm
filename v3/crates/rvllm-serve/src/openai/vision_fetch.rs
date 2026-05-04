//! Vision-content fetch + sidecar call.
//!
//! Two stages:
//!   1. `fetch_image_bytes(url)` resolves an OpenAI `image_url.url` —
//!      either a `data:` URI or `http(s)://` URL — into raw bytes +
//!      mime hint.
//!   2. `embed_via_sidecar(bytes, mime)` POSTs the bytes (base64) to
//!      the rvllm-vision-sidecar service and decodes the
//!      `[num_tokens, hidden_dim]` f16 tensor it returns.
//!
//! The sidecar URL comes from env `RVLLM_VISION_SIDECAR_URL`
//! (default `http://127.0.0.1:8765`). The model selection is
//! sidecar-side (it loads only one model per process).

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("fetch timeout")]
    FetchTimeout,
    #[error("fetch failed: {0}")]
    FetchFailed(String),
    #[error("image too large ({0} bytes, max {1})")]
    TooLarge(usize, usize),
    #[error("malformed data URI: {0}")]
    BadDataUri(String),
    #[error("sidecar unreachable: {0}")]
    SidecarUnreachable(String),
    #[error("sidecar error {status}: {body}")]
    SidecarError { status: u16, body: String },
    #[error("sidecar returned malformed embedding: {0}")]
    BadEmbedding(String),
}

/// Maximum image-bytes payload accepted by the fetcher (defensive).
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
/// HTTP fetch timeout for image URLs.
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);
/// Sidecar embed-call timeout.
const SIDECAR_TIMEOUT: Duration = Duration::from_secs(60);

/// Vision embeddings for one image, ready to splice into hidden_region.
#[derive(Debug, Clone)]
pub struct VisionEmbedding {
    pub num_tokens: usize,
    pub hidden_dim: usize,
    /// Raw little-endian f16 bytes, layout `[num_tokens, hidden_dim]`.
    pub data: Vec<u8>,
    /// Qwen3-VL grid_thw (used for MRoPE wiring later); None for Gemma.
    pub grid_thw: Option<[u32; 3]>,
}

impl VisionEmbedding {
    pub fn expected_byte_len(&self) -> usize {
        self.num_tokens * self.hidden_dim * 2
    }
}

/// Resolve an `image_url.url` field to (bytes, mime).
pub fn fetch_image_bytes(url: &str) -> Result<(Vec<u8>, String), VisionError> {
    if let Some(rest) = url.strip_prefix("data:") {
        return parse_data_uri(rest);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return fetch_http(url);
    }
    Err(VisionError::FetchFailed(format!(
        "url scheme not supported (only data: and http(s)://): {url}"
    )))
}

fn parse_data_uri(rest: &str) -> Result<(Vec<u8>, String), VisionError> {
    // RFC 2397: data:[<mediatype>][;base64],<data>
    // mediatype itself may contain `;` (charset etc.) — split on the
    // FIRST comma (delimits header from data), then parse the header.
    let comma = rest
        .find(',')
        .ok_or_else(|| VisionError::BadDataUri("missing comma".into()))?;
    let (header, data) = rest.split_at(comma);
    let data = &data[1..]; // skip the comma
    let mut mime = "image/png".to_string();
    let mut is_base64 = false;
    for token in header.split(';') {
        if token == "base64" {
            is_base64 = true;
        } else if token.contains('/') {
            mime = token.to_string();
        }
    }
    let bytes = if is_base64 {
        B64.decode(data.as_bytes())
            .map_err(|e| VisionError::BadDataUri(format!("base64: {e}")))?
    } else {
        data.as_bytes().to_vec()
    };
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(VisionError::TooLarge(bytes.len(), MAX_IMAGE_BYTES));
    }
    Ok((bytes, mime))
}

fn fetch_http(url: &str) -> Result<(Vec<u8>, String), VisionError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .build()
        .map_err(|e| VisionError::FetchFailed(format!("client: {e}")))?;
    let resp = client.get(url).send().map_err(|e| {
        if e.is_timeout() {
            VisionError::FetchTimeout
        } else {
            VisionError::FetchFailed(format!("{e}"))
        }
    })?;
    let mime = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .split(';')
        .next()
        .unwrap_or("application/octet-stream")
        .to_string();
    let bytes = resp
        .bytes()
        .map_err(|e| VisionError::FetchFailed(format!("body: {e}")))?
        .to_vec();
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(VisionError::TooLarge(bytes.len(), MAX_IMAGE_BYTES));
    }
    Ok((bytes, mime))
}

#[derive(Serialize)]
struct SidecarReq<'a> {
    image_b64: String,
    mime: &'a str,
}

#[derive(Deserialize)]
struct SidecarResp {
    num_tokens: usize,
    hidden_dim: usize,
    dtype: String,
    embeddings_b64: String,
    grid_thw: Option<[u32; 3]>,
}

fn sidecar_url() -> String {
    std::env::var("RVLLM_VISION_SIDECAR_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8765".to_string())
}

/// POST one image to the sidecar, decode the f16 tensor it returns.
pub fn embed_via_sidecar(bytes: &[u8], mime: &str) -> Result<VisionEmbedding, VisionError> {
    let url = format!("{}/embed", sidecar_url());
    let req = SidecarReq {
        image_b64: B64.encode(bytes),
        mime,
    };
    let client = reqwest::blocking::Client::builder()
        .timeout(SIDECAR_TIMEOUT)
        .build()
        .map_err(|e| VisionError::SidecarUnreachable(format!("client: {e}")))?;
    let resp = client.post(&url).json(&req).send().map_err(|e| {
        VisionError::SidecarUnreachable(format!("{e}"))
    })?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp
            .text()
            .unwrap_or_else(|e| format!("(unreadable body: {e})"));
        return Err(VisionError::SidecarError {
            status: status.as_u16(),
            body,
        });
    }
    let parsed: SidecarResp = resp
        .json()
        .map_err(|e| VisionError::BadEmbedding(format!("json: {e}")))?;
    if parsed.dtype != "float16" {
        return Err(VisionError::BadEmbedding(format!(
            "expected dtype=float16, got {}",
            parsed.dtype
        )));
    }
    let data = B64
        .decode(parsed.embeddings_b64.as_bytes())
        .map_err(|e| VisionError::BadEmbedding(format!("base64: {e}")))?;
    let expected = parsed.num_tokens * parsed.hidden_dim * 2;
    if data.len() != expected {
        return Err(VisionError::BadEmbedding(format!(
            "byte-len mismatch: got {}, expected {} ({} tokens × {} dim × 2 bytes)",
            data.len(),
            expected,
            parsed.num_tokens,
            parsed.hidden_dim
        )));
    }
    Ok(VisionEmbedding {
        num_tokens: parsed.num_tokens,
        hidden_dim: parsed.hidden_dim,
        data,
        grid_thw: parsed.grid_thw,
    })
}
