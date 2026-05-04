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

#[derive(Debug)]
pub enum VisionError {
    FetchTimeout,
    FetchFailed(String),
    TooLarge(usize, usize),
    BadDataUri(String),
    Decode(String),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FetchTimeout => write!(f, "fetch timeout"),
            Self::FetchFailed(s) => write!(f, "fetch failed: {s}"),
            Self::TooLarge(g, m) => write!(f, "image too large: {g} bytes (max {m})"),
            Self::BadDataUri(s) => write!(f, "bad data uri: {s}"),
            Self::Decode(s) => write!(f, "image header decode failed: {s}"),
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
    let bytes = if is_base64 {
        B64.decode(data.as_bytes())
            .map_err(|e| VisionError::BadDataUri(format!("base64: {e}")))?
    } else {
        data.as_bytes().to_vec()
    };
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(VisionError::TooLarge(bytes.len(), MAX_IMAGE_BYTES));
    }
    Ok(bytes)
}

fn fetch_http(url: &str) -> Result<Vec<u8>, VisionError> {
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
    let bytes = resp
        .bytes()
        .map_err(|e| VisionError::FetchFailed(format!("body: {e}")))?
        .to_vec();
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(VisionError::TooLarge(bytes.len(), MAX_IMAGE_BYTES));
    }
    Ok(bytes)
}

/// Predict vision-token count for Qwen 3.6 from image dims.
/// Mirrors the `vision_preprocess::qwen_smart_resize` formula but
/// without doing the actual resize.
pub fn predict_qwen_num_tokens(width: u32, height: u32) -> usize {
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
    .unwrap_or((factor, factor)); // tiny fallback
    let grid_h = h_bar / cfg.patch_size;
    let grid_w = w_bar / cfg.patch_size;
    let merge_sq = (cfg.merge_size * cfg.merge_size) as u32;
    ((grid_h * grid_w) / merge_sq) as usize
}

/// Predict vision-token count for Gemma 4 from image dims.
/// Mirrors `vision_preprocess::gemma_aspect_resize_dims`.
pub fn predict_gemma_num_tokens(width: u32, height: u32) -> usize {
    use rvllm_runtime::vision_preprocess::{gemma_aspect_resize_dims, GemmaPreprocessConfig};
    let cfg = GemmaPreprocessConfig::default();
    let (target_h, target_w) =
        gemma_aspect_resize_dims(height, width, &cfg).unwrap_or((cfg.patch_size, cfg.patch_size));
    let p = cfg.patch_size;
    let num_h = target_h / p;
    let num_w = target_w / p;
    let n_patches = (num_h * num_w) as usize;
    let k2 = (cfg.pooling_kernel_size * cfg.pooling_kernel_size) as usize;
    n_patches / k2
}
