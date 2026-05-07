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
            // Refuse redirects: every redirect would otherwise need
            // its own `vet_url_target` call to keep the SSRF guard
            // honest, and an attacker-controlled redirect chain can
            // do that arbitrarily many times. The simpler fix is to
            // turn redirects off — clients that need the redirected
            // image must resolve and supply the final URL themselves.
            .redirect(reqwest::redirect::Policy::none())
            // Round-21 finding #3: reqwest reads HTTP_PROXY /
            // HTTPS_PROXY / ALL_PROXY from the env by default. If
            // any of those are set in production the client routes
            // via the proxy, which then resolves and connects on its
            // own — defeating the IP-vetting + DNS-rebind protection
            // we did upfront in `resolve_to_addrs`. Disable proxy use
            // entirely so the SSRF guard is the actual network path.
            .no_proxy()
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
    // Collapse IPv4-mapped IPv6 (`::ffff:127.0.0.1` etc.) to its
    // underlying v4 first — otherwise an attacker can reach loopback
    // by spelling it as a v6 literal and bypass the v4 checks.
    let canonical = match ip {
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => IpAddr::V4(v4),
            None => IpAddr::V6(v6),
        },
        v => v,
    };
    match canonical {
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

/// Result of a successful URL vet — `host:port` parsed out and a
/// VERIFIED set of public IPs that the host resolved to. The caller
/// pins reqwest to these exact addresses so a DNS rebind between vet
/// and connect can't swing onto a private IP.
struct VettedTarget {
    host: String,
    // The `addrs` already carry the port via `SocketAddr`; we don't
    // need to keep it separately. Used only for `resolve_to_addrs`.
    addrs: Vec<std::net::SocketAddr>,
}

fn vet_url_target(url: &str) -> Result<Option<VettedTarget>, VisionError> {
    use std::net::ToSocketAddrs;
    if std::env::var("RVLLM_VISION_FETCH_ALLOW_PRIVATE")
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(false)
    {
        return Ok(None);
    }
    let parsed = url::Url::parse(url)
        .map_err(|e| VisionError::FetchFailed(format!("invalid url: {e}")))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| VisionError::FetchFailed("url has no host".into()))?;
    let port = parsed.port_or_known_default().unwrap_or(80);
    // Resolve once; check EVERY answer; pin reqwest to those IPs so
    // its connect-time resolution can't drift onto a different (and
    // possibly private) IP via DNS rebinding.
    let addrs: Vec<_> = (host, port)
        .to_socket_addrs()
        .map_err(|e| VisionError::FetchFailed(format!("dns: {e}")))?
        .collect();
    if addrs.is_empty() {
        return Err(VisionError::FetchFailed(format!(
            "dns: no addresses for host {host}"
        )));
    }
    for addr in &addrs {
        if is_disallowed_target(addr.ip()) {
            return Err(VisionError::FetchFailed(format!(
                "refusing to fetch from non-public address {} (set \
                 RVLLM_VISION_FETCH_ALLOW_PRIVATE=1 if intentional)",
                addr.ip()
            )));
        }
    }
    Ok(Some(VettedTarget { host: host.to_string(), addrs }))
}

/// Cache of `(host, sorted vetted-addr set)` → pinned reqwest
/// `Client`. Builders are cheap on cache hits; a single bounded
/// number of unique vision-image hosts × addr-sets per process
/// keeps it tiny in practice. Bounded growth is enforced by a soft
/// eviction at 256 entries (FIFO via the rebuild + clear pattern;
/// we don't strictly need LRU here).
fn pinned_client_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<reqwest::blocking::Client>>>
{
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<reqwest::blocking::Client>>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn pinned_client_for(
    t: &VettedTarget,
) -> Result<std::sync::Arc<reqwest::blocking::Client>, VisionError> {
    // Build a stable cache key from host + sorted IPs. Sorting the
    // address strings means (v4, v6) and (v6, v4) collapse to the
    // same client.
    let mut ip_strs: Vec<String> = t.addrs.iter().map(|a| a.to_string()).collect();
    ip_strs.sort();
    let key = format!("{}|{}", t.host, ip_strs.join(","));

    let cache = pinned_client_cache();
    {
        let map = cache.lock().expect("pinned client cache poisoned");
        if let Some(c) = map.get(&key) {
            return Ok(c.clone());
        }
    }

    let mut b = reqwest::blocking::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .pool_idle_timeout(std::time::Duration::from_secs(60))
        .pool_max_idle_per_host(8)
        .redirect(reqwest::redirect::Policy::none())
        // Round-21 finding #3: same as the unpinned client above —
        // ignore HTTP(S)_PROXY / ALL_PROXY env vars so the
        // `resolve_to_addrs` pin below is the actual network path
        // rather than a hint a configured proxy can override.
        .no_proxy();
    b = b.resolve_to_addrs(&t.host, &t.addrs);
    let c = std::sync::Arc::new(
        b.build()
            .map_err(|e| VisionError::FetchFailed(format!("client build: {e}")))?,
    );

    let mut map = cache.lock().expect("pinned client cache poisoned");
    if map.len() >= 256 {
        // Defensive cap: drop everything rather than implement an
        // LRU. A vision-image deployment with >256 distinct hosts is
        // already past this trade-off's design point.
        map.clear();
    }
    map.insert(key, c.clone());
    Ok(c)
}

fn fetch_http(url: &str) -> Result<Vec<u8>, VisionError> {
    use std::io::Read;
    let vet = vet_url_target(url)?;
    // Cache pinned clients per (host, vetted-addr-set) so we keep
    // reqwest's connection pool, TLS session reuse, and DNS pin
    // across multiple images on the same host. Building a fresh
    // client every fetch — like the previous version — meant a
    // full TLS handshake per image, swamping common cases (a chat
    // request with several images on the same CDN).
    let pinned_client = match &vet {
        Some(t) => Some(pinned_client_for(t)?),
        None => None,
    };
    let client = match pinned_client.as_deref() {
        Some(c) => c,
        None => http_client(),
    };
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

/// Pixtral preprocessing parameters used for token-count prediction
/// (Mistral 3.5). Match the public checkpoint:
/// `image_size = 1540` (longest edge), `patch_size = 14`,
/// `spatial_merge_size = 2`. The product `patch_size *
/// spatial_merge_size = 28` is the rounding factor for the resized
/// dims so the merged grid stays integer.
#[derive(Clone, Copy, Debug)]
pub struct Mistral35PixtralPredictConfig {
    pub longest_edge: u32,
    pub patch_size: u32,
    pub spatial_merge_size: u32,
}

impl Default for Mistral35PixtralPredictConfig {
    fn default() -> Self {
        Self {
            longest_edge: 1540,
            patch_size: 14,
            spatial_merge_size: 2,
        }
    }
}

/// Predict the number of soft tokens Mistral 3.5's Pixtral tower
/// emits for an image of `width × height`. Must match the actual
/// preprocess (`vision_preprocess::preprocess_mistral35_pixtral`,
/// landed in Step 7); the worker compares predicted vs measured
/// token counts and rejects on mismatch.
///
/// Algorithm:
///   1. Reject zero-pixel images up-front.
///   2. Scale by `s = min(1, longest_edge / max(w, h))` so the
///      longest edge fits within `longest_edge` px while preserving
///      aspect ratio.
///   3. Round each dim to a multiple of `patch_size *
///      spatial_merge_size` so `merged_h × merged_w` stays integer.
///   4. `merged_h = resized_h / (patch_size * spatial_merge_size)`,
///      `merged_w = resized_w / (patch_size * spatial_merge_size)`.
///   5. `num_tokens = merged_h * merged_w`.
pub fn predict_mistral35_num_tokens(width: u32, height: u32) -> Result<usize, VisionError> {
    predict_mistral35_num_tokens_with(
        width,
        height,
        Mistral35PixtralPredictConfig::default(),
    )
}

pub fn predict_mistral35_num_tokens_with(
    width: u32,
    height: u32,
    cfg: Mistral35PixtralPredictConfig,
) -> Result<usize, VisionError> {
    if width == 0 || height == 0 {
        return Err(VisionError::Predict(format!(
            "mistral35: zero-pixel image ({width}x{height})"
        )));
    }
    if cfg.patch_size == 0 || cfg.spatial_merge_size == 0 || cfg.longest_edge == 0 {
        return Err(VisionError::Predict(
            "mistral35: pixtral predict config has zero field".into(),
        ));
    }

    let factor = cfg.patch_size * cfg.spatial_merge_size; // 14 * 2 = 28
    let max_side = width.max(height);

    // Aspect-preserving downscale (no upscale: preserve images
    // already within the longest-edge cap as-is, then round).
    let (mut resized_h, mut resized_w) = if max_side <= cfg.longest_edge {
        (height, width)
    } else {
        let scale = cfg.longest_edge as f64 / max_side as f64;
        let h = ((height as f64) * scale).floor().max(1.0) as u32;
        let w = ((width as f64) * scale).floor().max(1.0) as u32;
        (h, w)
    };

    // Round each dim DOWN to a multiple of `factor`. If the rounded
    // value would be zero, snap up to one factor unit so the image
    // contributes at least 1 merged token (matches HF Pixtral's
    // ceil-min behaviour for tiny images).
    resized_h = (resized_h / factor) * factor;
    resized_w = (resized_w / factor) * factor;
    if resized_h == 0 {
        resized_h = factor;
    }
    if resized_w == 0 {
        resized_w = factor;
    }

    let merged_h = resized_h / factor;
    let merged_w = resized_w / factor;
    let num_tokens = (merged_h as usize) * (merged_w as usize);
    if num_tokens == 0 {
        return Err(VisionError::Predict(format!(
            "mistral35: predicted zero tokens for {width}x{height}"
        )));
    }
    Ok(num_tokens)
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

    #[test]
    fn mistral35_predict_square_at_longest_edge() {
        // 1540x1540 fits exactly. factor=28, merged grid 55x55 = 3025.
        let n = predict_mistral35_num_tokens(1540, 1540).unwrap();
        assert_eq!(n, 55 * 55);
    }

    #[test]
    fn mistral35_predict_wide_image_scales_down() {
        // 3080x1540 → scale 0.5 → 1540x770 → factor-28 round →
        // 1540x756 → merged 55x27 = 1485.
        let n = predict_mistral35_num_tokens(3080, 1540).unwrap();
        assert_eq!(n, 55 * 27);
    }

    #[test]
    fn mistral35_predict_tall_image_scales_down() {
        // Symmetric to wide.
        let n = predict_mistral35_num_tokens(1540, 3080).unwrap();
        assert_eq!(n, 27 * 55);
    }

    #[test]
    fn mistral35_predict_tiny_image_snaps_to_one_factor() {
        // 10x10 < factor=28, rounds down to 0 then snaps up to 28
        // so merged = 1x1 = 1 token.
        let n = predict_mistral35_num_tokens(10, 10).unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn mistral35_predict_rejects_zero_pixels() {
        assert!(predict_mistral35_num_tokens(0, 100).is_err());
        assert!(predict_mistral35_num_tokens(100, 0).is_err());
    }

    #[test]
    fn mistral35_predict_under_longest_edge_no_upscale() {
        // 100x200 stays at 100x200 (no upscale) → factor-28 →
        // 84x196 → merged 3x7 = 21.
        let n = predict_mistral35_num_tokens(100, 200).unwrap();
        assert_eq!(n, 3 * 7);
    }
}
