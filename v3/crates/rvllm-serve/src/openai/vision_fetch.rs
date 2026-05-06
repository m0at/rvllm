//! Vision-content fetch + num-tokens prediction for OpenAI multimodal
//! chat-completions requests.
//!
//! Handles two URL schemes:
//!   - `data:` URI (RFC 2397) with optional base64 + media type
//!   - `http(s)://` (5s timeout, 20 MiB cap)
//!
//! Returns raw image bytes (not decoded) so the cuda-worker thread can
//! run the native vision-tower forward without re-blocking on tokio.
//!
//! Also predicts `num_tokens` per image from header-decoded dims, so
//! the chat template knows how many placeholder tokens to emit.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use std::time::Duration;

/// Decode `%xx` percent-encoding into raw bytes. Used for
/// non-base64 data: URIs per RFC 2397 §3. Pulls in no external
/// crate — the format is trivially small (two hex digits after
/// every `%`).
fn percent_decode(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let b = input[i];
        if b == b'%' {
            if i + 2 >= input.len() {
                return Err(format!("truncated %xx escape at offset {i}"));
            }
            let hi = (input[i + 1] as char).to_digit(16)
                .ok_or_else(|| format!("bad hex {:?} at offset {}", input[i + 1] as char, i + 1))?;
            let lo = (input[i + 2] as char).to_digit(16)
                .ok_or_else(|| format!("bad hex {:?} at offset {}", input[i + 2] as char, i + 2))?;
            out.push(((hi << 4) | lo) as u8);
            i += 3;
        } else {
            out.push(b);
            i += 1;
        }
    }
    Ok(out)
}

#[derive(Debug)]
pub enum VisionError {
    FetchTimeout,
    FetchFailed(String),
    TooLarge(usize, usize),
    BadDataUri(String),
    Decode(String),
    Predict(String),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FetchTimeout => write!(f, "fetch timeout"),
            Self::FetchFailed(s) => write!(f, "fetch failed: {s}"),
            Self::TooLarge(g, m) => write!(f, "image too large: {g} bytes (max {m})"),
            Self::BadDataUri(s) => write!(f, "bad data uri: {s}"),
            Self::Decode(s) => write!(f, "image header decode failed: {s}"),
            Self::Predict(s) => write!(f, "vision-token prediction failed: {s}"),
        }
    }
}
impl std::error::Error for VisionError {}

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// One image attached to a chat request, ready for the cuda_worker.
pub struct VisionItem {
    /// Raw image bytes (PNG/JPEG/WebP). cuda_worker re-decodes via
    /// `vision_preprocess::decode_image`.
    pub bytes: Vec<u8>,
    /// Image dims (width, height) read from the header. Used for the
    /// num-tokens prediction below; also useful for diagnostics.
    pub width: u32,
    pub height: u32,
    /// Predicted number of vision-output tokens (post-PatchMerger).
    /// Caller uses this to emit the right number of placeholder
    /// tokens in the chat template render.
    pub predicted_num_tokens: usize,
    /// Index of the chat-message that carried this image; logged.
    pub msg_idx: usize,
}

/// Resolve an `image_url.url` field to (raw bytes, width, height).
/// Decodes header only — not the full image — for cheap dim lookup.
pub fn fetch_image(url: &str) -> Result<(Vec<u8>, u32, u32), VisionError> {
    let bytes = if let Some(rest) = url.strip_prefix("data:") {
        parse_data_uri(rest)?
    } else if url.starts_with("http://") || url.starts_with("https://") {
        fetch_http(url)?
    } else {
        return Err(VisionError::FetchFailed(format!(
            "unsupported URL scheme: {url}"
        )));
    };
    // Header-only dim decode via image::ImageReader.
    let cursor = std::io::Cursor::new(&bytes);
    let reader = image::ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| VisionError::Decode(format!("guess: {e}")))?;
    let (w, h) = reader
        .into_dimensions()
        .map_err(|e| VisionError::Decode(format!("dims: {e}")))?;
    Ok((bytes, w, h))
}

fn parse_data_uri(rest: &str) -> Result<Vec<u8>, VisionError> {
    let comma = rest
        .find(',')
        .ok_or_else(|| VisionError::BadDataUri("missing comma".into()))?;
    let (header, data) = rest.split_at(comma);
    let data = &data[1..];
    let mut is_base64 = false;
    for token in header.split(';') {
        if token == "base64" {
            is_base64 = true;
        }
    }
    // Pre-check the encoded size before doing the actual decode so a
    // 60 MiB base64 payload doesn't get fully decoded into a fresh
    // ~45 MiB Vec just to be rejected at line below. base64 expands
    // 3 → 4 bytes, so a decoded payload of `MAX_IMAGE_BYTES` cannot
    // arrive in fewer than `MAX_IMAGE_BYTES * 4 / 3 + 4` encoded
    // bytes (the +4 covers the optional padding pair). Percent-
    // encoding is per-byte 1:1 (or 3:1 for escapes) so the encoded
    // length is already a strict upper bound on the decoded length.
    let max_encoded = if is_base64 {
        // saturating_mul keeps the math safe for a hypothetical
        // bumped MAX_IMAGE_BYTES; saturating_add adds the padding.
        MAX_IMAGE_BYTES.saturating_mul(4) / 3 + 4
    } else {
        MAX_IMAGE_BYTES
    };
    if data.len() > max_encoded {
        return Err(VisionError::TooLarge(data.len(), max_encoded));
    }

    let bytes = if is_base64 {
        B64.decode(data.as_bytes())
            .map_err(|e| VisionError::BadDataUri(format!("base64: {e}")))?
    } else {
        // RFC 2397 §3: non-base64 data: URIs carry their payload
        // percent-encoded (URI character set). Decode `%xx` triples
        // back to bytes; pass other bytes through untouched. Without
        // this, a spec-compliant `data:image/png,%89PNG…` payload
        // arrives at image::load with literal '%','8','9','P',… and
        // fails header-decode.
        percent_decode(data.as_bytes())
            .map_err(|e| VisionError::BadDataUri(format!("percent-decode: {e}")))?
    };
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(VisionError::TooLarge(bytes.len(), MAX_IMAGE_BYTES));
    }
    Ok(bytes)
}

/// Process-wide blocking HTTP client for vision-image fetches.
/// One-time build at first use; reused for every image URL across
/// every request. Without this each `fetch_http` call paid for a
/// fresh DNS+TLS handshake — a request with N images and the next
/// request with the same hosts both lost connection-pool benefits.
fn http_client() -> &'static reqwest::blocking::Client {
    use std::sync::OnceLock;
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(FETCH_TIMEOUT)
            // Sensible defaults for short-lived image fetches.
            .pool_idle_timeout(std::time::Duration::from_secs(60))
            .pool_max_idle_per_host(8)
            .build()
            .expect("build vision-fetch reqwest client")
    })
}

/// Returns true if `ip` is one we refuse to fetch by default —
/// loopback, private (RFC 1918, ULA), link-local (incl. cloud-metadata
/// 169.254.169.254), multicast, reserved, and the IPv4 broadcast.
/// Operators running rvllm-serve in trusted internal contexts can
/// bypass this with `RVLLM_VISION_FETCH_ALLOW_PRIVATE=1`.
fn is_disallowed_target(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_multicast()
                || v4.is_unspecified()
                || v4.is_broadcast()
                // RFC 6598 carrier-grade NAT, not flagged by std today.
                || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xC0) == 64)
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_multicast()
                || v6.is_unspecified()
                // Unique local fc00::/7 — std lacks `is_unique_local`
                // on stable, so check the leading byte directly.
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                // Link-local fe80::/10.
                || (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

fn vet_url_target(url: &str) -> Result<(), VisionError> {
    use std::net::ToSocketAddrs;
    if std::env::var("RVLLM_VISION_FETCH_ALLOW_PRIVATE")
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(false)
    {
        return Ok(());
    }
    let parsed = url::Url::parse(url)
        .map_err(|e| VisionError::FetchFailed(format!("invalid url: {e}")))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| VisionError::FetchFailed("url has no host".into()))?;
    let port = parsed.port_or_known_default().unwrap_or(80);
    // Resolve and check EVERY answer. Note: this is one DNS lookup;
    // reqwest will do its own when it actually connects. A determined
    // attacker can DNS-rebind between the two — closing that gap
    // properly needs a custom reqwest connector that pins the IP we
    // verified here. Filed for later; this still catches the obvious
    // SSRF cases (literal-IP URLs, `metadata.google.internal`, etc).
    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|e| VisionError::FetchFailed(format!("dns: {e}")))?;
    for addr in addrs {
        if is_disallowed_target(addr.ip()) {
            return Err(VisionError::FetchFailed(format!(
                "refusing to fetch from non-public address {} (set \
                 RVLLM_VISION_FETCH_ALLOW_PRIVATE=1 if intentional)",
                addr.ip()
            )));
        }
    }
    Ok(())
}

fn fetch_http(url: &str) -> Result<Vec<u8>, VisionError> {
    use std::io::Read;
    vet_url_target(url)?;
    let client = http_client();
    let mut resp = client.get(url).send().map_err(|e| {
        if e.is_timeout() {
            VisionError::FetchTimeout
        } else {
            VisionError::FetchFailed(format!("{e}"))
        }
    })?;
    // Early-reject by Content-Length BEFORE buffering any body
    // (Codex review #1, round 4). The previous `resp.bytes()` call
    // pulled the whole body into RAM and only then checked the cap,
    // letting attackers / careless clients allocate hundreds of MB
    // before getting a 400 back.
    if let Some(len) = resp.content_length() {
        if (len as usize) > MAX_IMAGE_BYTES {
            return Err(VisionError::TooLarge(len as usize, MAX_IMAGE_BYTES));
        }
    }
    // Stream-read with a hard cap. Servers omitting Content-Length
    // (chunked transfer) still hit the limit at 1 byte over cap and
    // return an error instead of growing the buffer further.
    let mut buf = Vec::with_capacity(64 * 1024);
    let mut chunk = [0u8; 16 * 1024];
    loop {
        let n = match resp.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => return Err(VisionError::FetchFailed(format!("body: {e}"))),
        };
        if buf.len() + n > MAX_IMAGE_BYTES {
            return Err(VisionError::TooLarge(buf.len() + n, MAX_IMAGE_BYTES));
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok(buf)
}

/// Predict vision-token count for Qwen 3.6 from image dims.
/// Mirrors the `vision_preprocess::qwen_smart_resize` formula but
/// without doing the actual resize.
pub fn predict_qwen_num_tokens(width: u32, height: u32) -> Result<usize, VisionError> {
    use rvllm_runtime::vision_preprocess::{qwen_smart_resize, QwenPreprocessConfig};
    let cfg = QwenPreprocessConfig::default();
    let factor = cfg.patch_size * cfg.merge_size; // 32
    let (h_bar, w_bar) = qwen_smart_resize(
        height,
        width,
        factor,
        cfg.min_pixels,
        cfg.max_pixels,
    )
    .map_err(|e| VisionError::Predict(format!("qwen_smart_resize({width}x{height}): {e:?}")))?;
    let grid_h = h_bar / cfg.patch_size;
    let grid_w = w_bar / cfg.patch_size;
    let merge_sq = (cfg.merge_size * cfg.merge_size) as u32;
    Ok(((grid_h * grid_w) / merge_sq) as usize)
}

/// Predict vision-token count for Gemma 4 from image dims.
/// Mirrors `vision_preprocess::gemma_aspect_resize_dims`.
pub fn predict_gemma_num_tokens(width: u32, height: u32) -> Result<usize, VisionError> {
    use rvllm_runtime::vision_preprocess::{gemma_aspect_resize_dims, GemmaPreprocessConfig};
    let cfg = GemmaPreprocessConfig::default();
    let (target_h, target_w) = gemma_aspect_resize_dims(height, width, &cfg).map_err(|e| {
        VisionError::Predict(format!("gemma_aspect_resize_dims({width}x{height}): {e:?}"))
    })?;
    let p = cfg.patch_size;
    let num_h = target_h / p;
    let num_w = target_w / p;
    let n_patches = (num_h * num_w) as usize;
    let k2 = (cfg.pooling_kernel_size * cfg.pooling_kernel_size) as usize;
    Ok(n_patches / k2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_decode_passes_through_plain_ascii() {
        assert_eq!(percent_decode(b"hello world").unwrap(), b"hello world");
    }

    #[test]
    fn percent_decode_handles_png_signature() {
        // RFC 2397 spec example: PNG header bytes 89 50 4E 47 …
        // arrive in a non-base64 data URI as "%89PNG%0D%0A%1A%0A".
        let got = percent_decode(b"%89PNG%0D%0A%1A%0A").unwrap();
        assert_eq!(got, [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    }

    #[test]
    fn percent_decode_rejects_truncated_escape() {
        assert!(percent_decode(b"abc%9").is_err());
        assert!(percent_decode(b"abc%").is_err());
    }

    #[test]
    fn percent_decode_rejects_bad_hex() {
        assert!(percent_decode(b"abc%ZZ").is_err());
    }

    #[test]
    fn parse_data_uri_base64_path() {
        // base64('hello') = aGVsbG8=
        let bytes = parse_data_uri("text/plain;base64,aGVsbG8=").unwrap();
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn parse_data_uri_percent_encoded_path() {
        let bytes = parse_data_uri("image/png,%89PNG%0D%0A%1A%0A").unwrap();
        assert_eq!(bytes, [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    }
}
