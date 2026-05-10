//! Vision input preprocessing for Qwen 3.6 + Gemma 4.
//!
//! Decodes image bytes (PNG / JPEG / WebP), runs the model-specific
//! HF processor pipeline (smart resize, normalize, patchify), and
//! returns flattened patch tensors ready for the vision tower.
//!
//! Reference:
//!   transformers/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
//!   transformers/src/transformers/models/gemma4/image_processing_gemma4.py
//!
//! Bicubic resize uses the `image` crate's `CatmullRom` filter — the
//! closest stdlib equivalent of PIL's BICUBIC. Bit-for-bit match to
//! HF/PIL is NOT achievable without re-vendoring the PIL kernel; we
//! validate cosine-similarity vs HF reference dumps in the test
//! fixtures (target ≥ 0.999).

use image::imageops::FilterType;
use image::{ImageReader, RgbImage};
use std::io::Cursor;

#[derive(Debug)]
pub enum PreprocessError {
    Decode(String),
    AspectRatio(f32),
    BadDims(u32, u32),
}

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(s) => write!(f, "image decode failed: {s}"),
            Self::AspectRatio(r) => write!(f, "unsupported aspect ratio: {r}"),
            Self::BadDims(w, h) => write!(f, "invalid image dimensions: {w}x{h}"),
        }
    }
}

impl std::error::Error for PreprocessError {}

/// Decode PNG/JPEG/WebP bytes → RGB8 image. Honours EXIF orientation
/// is NOT done here (HF doesn't either by default — caveat for client).
///
/// Pixel-budget guard: an attacker can ship a heavily-compressed
/// "ZIP-bomb" image whose encoded bytes pass the upstream
/// `MAX_IMAGE_BYTES` cap but whose RGB8 decode would allocate
/// gigabytes (a 50000×50000 PNG is ~25 KiB compressed but 7 GiB
/// expanded). We read the header dimensions FIRST and reject before
/// the full decode runs. Operators with legitimate large-image
/// workloads can raise the cap via `RVLLM_VISION_MAX_PIXELS`.
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage, PreprocessError> {
    let cursor = Cursor::new(bytes);
    let reader = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| PreprocessError::Decode(format!("guess format: {e}")))?;
    let max_pixels: u64 = std::env::var("RVLLM_VISION_MAX_PIXELS")
        .ok()
        .and_then(|s| s.parse().ok())
        // 32 MP default. Covers a 6K-equivalent (≈22 MP) with headroom
        // and refuses 50000×50000 zip-bomb PNGs (2.5 G pixels).
        .unwrap_or(32 * 1024 * 1024);
    if let Ok((w, h)) = reader.into_dimensions() {
        let pixels = (w as u64) * (h as u64);
        if pixels > max_pixels {
            return Err(PreprocessError::Decode(format!(
                "image dimensions {}×{} = {} pixels exceed cap {} \
                 (RVLLM_VISION_MAX_PIXELS to override)",
                w, h, pixels, max_pixels
            )));
        }
    }
    // Re-open: `into_dimensions` consumed the reader.
    let cursor = Cursor::new(bytes);
    let reader = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| PreprocessError::Decode(format!("guess format: {e}")))?;
    let dyn_img = reader
        .decode()
        .map_err(|e| PreprocessError::Decode(format!("decode: {e}")))?;
    Ok(dyn_img.to_rgb8())
}

// ─── Qwen 3.6 (Qwen2VL/Qwen3VL processor pipeline) ────────────────────

/// Qwen 3.6 preprocessor config (see model_dir/preprocessor_config.json):
///   patch_size = 16, temporal_patch_size = 2, merge_size = 2
///   shortest_edge = 65536, longest_edge = 16777216 (in pixels)
///   image_mean = image_std = [0.5, 0.5, 0.5]
///   resample = BICUBIC
///   factor = patch_size · merge_size = 32
pub struct QwenPreprocessConfig {
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub merge_size: u32,
    pub min_pixels: u32,
    pub max_pixels: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for QwenPreprocessConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            // HF Qwen2VLImageProcessorFast does not enforce a lower bound
            // on small images — feeding a 224×224 image keeps it at 224×224
            // (grid 14×14). Earlier setting min_pixels=65536 (=256²)
            // upsampled to 256×256 (grid 16×16) and broke pos_embed
            // interpolation alignment. Setting to 0 disables the lower
            // bound.
            min_pixels: 0,
            // Server-side cap on vision input size. HF would allow up to
            // 16,777,216 pixels (≈ 65k patches before merge). Our naive
            // ViT attention path allocates `n_tokens × n_tokens` f32
            // scores per head per block — at 16 MP that's tens of GiB
            // of scratch and minutes of GPU time per request, easy
            // availability hazard from a single 20 MiB JPEG. Codex
            // review #3 round 5 flagged this. 1024² = 1,048,576
            // pixels ≈ 4096 ViT patches → 1024 merged tokens, so a
            // single image fits in ~16 MiB of scores f32 per head and
            // completes in ~1.5 s. The cap is configurable via
            // `RVLLM_QWEN_VISION_MAX_PIXELS` for operators who know
            // their hardware can take more.
            max_pixels: std::env::var("RVLLM_QWEN_VISION_MAX_PIXELS")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(1_048_576),
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        }
    }
}

/// Output of Qwen preprocessing.
pub struct QwenPatches {
    /// `[num_patches, 3 · temporal_patch · patch · patch]` f32 row-major.
    pub pixel_values: Vec<f32>,
    /// `[t, h, w]` grid dimensions in patch-units.
    pub grid_thw: [u32; 3],
}

impl QwenPatches {
    pub fn num_patches(&self) -> usize {
        let [t, h, w] = self.grid_thw;
        (t * h * w) as usize
    }
    pub fn patch_dim(&self) -> usize {
        // 3 · temporal · 16 · 16 = 1536 with defaults.
        self.pixel_values.len() / self.num_patches().max(1)
    }
    /// Convert pixel_values to f16 little-endian bytes ready for HtoD.
    pub fn to_f16_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixel_values.len() * 2);
        for &x in &self.pixel_values {
            out.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
        }
        out
    }
}

/// Qwen smart-resize: round each side to the nearest multiple of `factor`,
/// then bound total pixels into [min_pixels, max_pixels] with aspect-
/// preserving rescaling. Matches HF `smart_resize`.
pub fn qwen_smart_resize(
    h: u32,
    w: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32), PreprocessError> {
    let max_dim = h.max(w) as f32;
    let min_dim = h.min(w) as f32;
    if min_dim == 0.0 {
        return Err(PreprocessError::BadDims(w, h));
    }
    if max_dim / min_dim > 200.0 {
        return Err(PreprocessError::AspectRatio(max_dim / min_dim));
    }
    let f = factor as f32;
    let mut h_bar = (((h as f32) / f).round() * f) as u32;
    let mut w_bar = (((w as f32) / f).round() * f) as u32;
    let pixels = (h_bar as u64) * (w_bar as u64);
    if pixels > (max_pixels as u64) {
        let beta = (((h as u64) * (w as u64)) as f32 / (max_pixels as f32)).sqrt();
        h_bar = (((h as f32) / beta / f).floor() * f).max(f) as u32;
        w_bar = (((w as f32) / beta / f).floor() * f).max(f) as u32;
    } else if pixels < (min_pixels as u64) {
        let beta = (min_pixels as f32 / ((h as u64) * (w as u64)) as f32).sqrt();
        h_bar = (((h as f32) * beta / f).ceil() * f) as u32;
        w_bar = (((w as f32) * beta / f).ceil() * f) as u32;
    }
    // Clamp to at least one full patch on each side. With min_pixels=0
    // (current default) a sub-`factor/2` input rounds down to 0 and
    // produces a 0-token vision item that the splice path then rejects.
    // Tiny icons / 1×1 pixels happen in real API traffic; treat them
    // as a single minimal patch instead of an error. Codex review
    // #6 (round 4).
    if h_bar < factor { h_bar = factor; }
    if w_bar < factor { w_bar = factor; }
    Ok((h_bar, w_bar))
}

/// Run the full Qwen preprocessing pipeline.
pub fn preprocess_qwen(
    img: &RgbImage,
    cfg: &QwenPreprocessConfig,
) -> Result<QwenPatches, PreprocessError> {
    let factor = cfg.patch_size * cfg.merge_size;
    let (h_bar, w_bar) =
        qwen_smart_resize(img.height(), img.width(), factor, cfg.min_pixels, cfg.max_pixels)?;
    // Bicubic resize (PIL BICUBIC ↔ image::CatmullRom; close but not bit-exact).
    let resized = image::imageops::resize(img, w_bar, h_bar, FilterType::CatmullRom);

    // To CHW f32 normalized: rescale_factor=1/255, (x/255 - mean)/std.
    let h = h_bar as usize;
    let w = w_bar as usize;
    let mut chw = vec![0.0f32; 3 * h * w];
    let inv = 1.0 / 255.0;
    for (i, px) in resized.pixels().enumerate() {
        let r = (px[0] as f32 * inv - cfg.image_mean[0]) / cfg.image_std[0];
        let g = (px[1] as f32 * inv - cfg.image_mean[1]) / cfg.image_std[1];
        let b = (px[2] as f32 * inv - cfg.image_mean[2]) / cfg.image_std[2];
        chw[0 * h * w + i] = r;
        chw[1 * h * w + i] = g;
        chw[2 * h * w + i] = b;
    }

    // Patchify mirrors HF's view+permute+reshape (qwen2_vl
    // image_processing_qwen2_vl.py:203-220):
    //   shape source [batch=1, T_p=2 (replicated single-frame),
    //                 channel=3, grid_h, merge, patch, grid_w, merge, patch]
    //   permute      (0, 1, 4, 7, 5, 8, 3, 2, 6, 9)  but the source
    //                 already starts at (0, grid_t=1, 1, T_p=2, 3,
    //                 grid_h//merge, merge, patch, grid_w//merge,
    //                 merge, patch)
    // Final flat shape: [grid_t·grid_h·grid_w, channel·T_p·patch·patch].
    let p = cfg.patch_size as usize;
    let m = cfg.merge_size as usize;
    let tp = cfg.temporal_patch_size as usize;
    let grid_h = h / p;
    let grid_w = w / p;
    let grid_t = 1usize;
    let n_patches = grid_t * grid_h * grid_w;
    let patch_dim = 3 * tp * p * p;
    let mut out = vec![0.0f32; n_patches * patch_dim];

    // Replicate single frame across the temporal-patch dim.
    // For each output row indexed by (gt, gh_mh, gw_mw, mh, mw) and
    // each output col indexed by (c, t, ph, pw):
    //
    //   src H = (gh_mh * merge + mh) * patch + ph
    //   src W = (gw_mw * merge + mw) * patch + pw
    //   src C = c
    //   value = chw[c, src_H, src_W]
    //
    // Output row index uses HF's permute: (gt, gh_mh, gw_mw, mh, mw)
    // — i.e. nested merge-then-patch grouping.
    for gt in 0..grid_t {
        for gh_mh in 0..(grid_h / m) {
            for gw_mw in 0..(grid_w / m) {
                for mh in 0..m {
                    for mw in 0..m {
                        let row = ((gt * (grid_h / m) + gh_mh) * (grid_w / m) + gw_mw) * (m * m)
                            + mh * m
                            + mw;
                        for c in 0..3 {
                            for t in 0..tp {
                                for ph in 0..p {
                                    for pw in 0..p {
                                        let src_h = (gh_mh * m + mh) * p + ph;
                                        let src_w = (gw_mw * m + mw) * p + pw;
                                        let val = chw[c * h * w + src_h * w + src_w];
                                        let col = ((c * tp + t) * p + ph) * p + pw;
                                        out[row * patch_dim + col] = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(QwenPatches {
        pixel_values: out,
        grid_thw: [grid_t as u32, grid_h as u32, grid_w as u32],
    })
}

// ─── Gemma 4 (SigLIP2-style processor pipeline) ───────────────────────

/// Gemma 4 preprocessor config (see processor_config.json):
///   patch_size = 16, pooling_kernel_size = 3, max_soft_tokens = 280
///   image_mean = [0,0,0], image_std = [1,1,1] (no normalize, just /255)
///   resample = BICUBIC (with torchvision antialias=True)
pub struct GemmaPreprocessConfig {
    pub patch_size: u32,
    pub pooling_kernel_size: u32,
    pub max_soft_tokens: u32,
}

impl Default for GemmaPreprocessConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens: 280,
        }
    }
}

pub struct GemmaPatches {
    /// `[max_patches, patch_size² · 3]` f32 (zero-padded).
    pub pixel_values: Vec<f32>,
    /// `[max_patches, 2]` i64 patch (col, row) coords; -1 for padding.
    pub position_ids: Vec<i64>,
    /// Number of valid (non-padded) patches.
    pub num_valid_patches: usize,
    /// Number of vision-soft-tokens after avg-pool (= num_valid / k²).
    pub num_soft_tokens: usize,
    /// Resized (target_h, target_w) in pixels.
    pub resized: (u32, u32),
}

impl GemmaPatches {
    pub fn pixel_values_to_f16_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixel_values.len() * 2);
        for &x in &self.pixel_values {
            out.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
        }
        out
    }
}

/// Aspect-ratio-preserving resize so the image fits within
/// max_patches=max_soft_tokens·k² patches with patch_size grid,
/// rounding down to a multiple of (pooling_kernel_size · patch_size).
pub fn gemma_aspect_resize_dims(
    h: u32,
    w: u32,
    cfg: &GemmaPreprocessConfig,
) -> Result<(u32, u32), PreprocessError> {
    if h == 0 || w == 0 {
        return Err(PreprocessError::BadDims(w, h));
    }
    let max_patches = (cfg.max_soft_tokens * cfg.pooling_kernel_size * cfg.pooling_kernel_size) as f32;
    let total = (h as f32) * (w as f32);
    let target = max_patches * ((cfg.patch_size * cfg.patch_size) as f32);
    let factor = (target / total).sqrt();
    let side_mult = (cfg.pooling_kernel_size * cfg.patch_size) as f32; // 48
    let mut th = ((factor * h as f32 / side_mult).floor() * side_mult) as u32;
    let mut tw = ((factor * w as f32 / side_mult).floor() * side_mult) as u32;
    let max_side = (cfg.max_soft_tokens as f32 / cfg.pooling_kernel_size as f32).floor() as u32
        * (cfg.pooling_kernel_size * cfg.patch_size);
    if th == 0 && tw == 0 {
        return Err(PreprocessError::BadDims(w, h));
    }
    let s = side_mult as u32;
    if th == 0 {
        th = s;
        tw = ((w as f32 / h as f32).floor() as u32 * s).min(max_side);
    } else if tw == 0 {
        tw = s;
        th = ((h as f32 / w as f32).floor() as u32 * s).min(max_side);
    }
    Ok((th, tw))
}

pub fn preprocess_gemma(
    img: &RgbImage,
    cfg: &GemmaPreprocessConfig,
) -> Result<GemmaPatches, PreprocessError> {
    let (target_h, target_w) = gemma_aspect_resize_dims(img.height(), img.width(), cfg)?;
    // Bilinear with antialias is closer to torchvision's default than
    // CatmullRom; image::Triangle is bilinear (no antialias). Best
    // available standard filter:
    let resized = image::imageops::resize(img, target_w, target_h, FilterType::Triangle);

    // Rescale to [0,1] (do_normalize=False → mean/std identity).
    let h = target_h as usize;
    let w = target_w as usize;
    let mut chw = vec![0.0f32; 3 * h * w];
    let inv = 1.0 / 255.0;
    for (i, px) in resized.pixels().enumerate() {
        chw[0 * h * w + i] = px[0] as f32 * inv;
        chw[1 * h * w + i] = px[1] as f32 * inv;
        chw[2 * h * w + i] = px[2] as f32 * inv;
    }

    // Patchify: image [3, H, W] → reshape [3, num_h, patch, num_w, patch]
    // → permute [1, 3, 2, 4, 0] → reshape [num_h*num_w, patch² · 3].
    let p = cfg.patch_size as usize;
    let num_h = h / p;
    let num_w = w / p;
    let n_patches = num_h * num_w;
    let patch_dim = p * p * 3;

    let max_patches = (cfg.max_soft_tokens * cfg.pooling_kernel_size * cfg.pooling_kernel_size)
        as usize;
    if n_patches > max_patches {
        return Err(PreprocessError::BadDims(target_w, target_h));
    }

    let mut out = vec![0.0f32; max_patches * patch_dim];
    let mut pos = vec![-1i64; max_patches * 2];
    for ph in 0..num_h {
        for pw in 0..num_w {
            let row = ph * num_w + pw;
            for ip in 0..p {
                for jp in 0..p {
                    for c in 0..3 {
                        let src = c * h * w + (ph * p + ip) * w + (pw * p + jp);
                        // permute [1, 3, 2, 4, 0] from [c, ph, ip, pw, jp]:
                        //   output dim ordering inside row = [ip, jp, c]
                        let col = ip * p * 3 + jp * 3 + c;
                        out[row * patch_dim + col] = chw[src];
                    }
                }
            }
            // Position IDs: meshgrid(W, H, indexing='xy') with stack(-1)
            // yields per-patch (col, row) = (pw, ph).
            pos[row * 2 + 0] = pw as i64;
            pos[row * 2 + 1] = ph as i64;
        }
    }
    let k2 = (cfg.pooling_kernel_size * cfg.pooling_kernel_size) as usize;
    Ok(GemmaPatches {
        pixel_values: out,
        position_ids: pos,
        num_valid_patches: n_patches,
        num_soft_tokens: n_patches / k2,
        resized: (target_h, target_w),
    })
}

// ─── Mistral 3.5 (Pixtral processor pipeline) ─────────────────────────

/// Mistral 3.5 / Pixtral preprocessor config.
///
/// Source: integration spec + the public NVFP4 checkpoint
/// `processor_config.json`. CLIP-style normalisation; longest-edge
/// resize so `max(h, w) <= longest_edge` while preserving aspect
/// ratio; resized dims rounded down to multiples of
/// `patch_size * spatial_merge_size` so the merged grid is integer.
#[derive(Clone, Debug)]
pub struct Mistral35PreprocessConfig {
    pub patch_size: u32,
    pub spatial_merge_size: u32,
    pub longest_edge: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for Mistral35PreprocessConfig {
    fn default() -> Self {
        // CLIP normalisation values match the public Pixtral
        // processor exactly. patch=14, merge=2 → factor=28.
        Self {
            patch_size: 14,
            spatial_merge_size: 2,
            longest_edge: 1540,
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.26130258, 0.27577711],
        }
    }
}

/// Output of [`preprocess_mistral35_pixtral`]. Holds the host-side
/// patch tensor + the merged grid the GPU forward needs.
#[derive(Debug)]
pub struct Mistral35Patches {
    /// `[num_patches, patch_size * patch_size * 3]` f32 row-major,
    /// in `(patch_row, patch_col)` document order. Inner row layout
    /// is **HWC** — `[ip, jp, c]` flattened, matching the patch-conv
    /// kernel ABI shared with the Qwen / Gemma vision paths
    /// (channels-last, fastest-varying). Round-10 #3 fix: the
    /// pre-existing doc said `[c, ip, jp]` (CHW) which contradicted
    /// the implementation at line 596 (`col = ip*p*3 + jp*3 + c`).
    /// The implementation is what the kernel expects; the doc was
    /// the lie.
    pub pixel_values: Vec<f32>,
    /// Resized dimensions in pixels (height, width). Always a
    /// multiple of `patch_size * spatial_merge_size`.
    pub resized: (u32, u32),
    /// `(grid_h, grid_w)` = patch count along each axis.
    pub patch_grid: (u32, u32),
    /// `(merged_h, merged_w)` = `patch_grid` divided by
    /// `spatial_merge_size`. `merged_h * merged_w` is the soft-token
    /// count this image will contribute to the prompt — must match
    /// the predictor in `vision_fetch::predict_mistral35_num_tokens`.
    pub merged_grid: (u32, u32),
    /// Convenience accessor — equals `merged_h * merged_w`.
    pub num_soft_tokens: usize,
}

impl Mistral35Patches {
    /// Convert `pixel_values` to f16 little-endian bytes for upload.
    pub fn pixel_values_to_f16_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixel_values.len() * 2);
        for &x in &self.pixel_values {
            out.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
        }
        out
    }
    pub fn num_patches(&self) -> usize {
        let (gh, gw) = self.patch_grid;
        (gh as usize) * (gw as usize)
    }
}

/// Compute the resized `(h, w)` Pixtral expects for an input image.
/// Aspect-preserving downscale (no upscale) so the longest edge is
/// at most `longest_edge`, then both dims rounded DOWN to a multiple
/// of `patch_size * spatial_merge_size`. If a dim rounds to 0, snap
/// up to one factor unit so the image contributes at least one
/// merged token. Mirrors `predict_mistral35_num_tokens` byte-for-byte.
pub fn mistral35_pixtral_resize_dims(
    h: u32,
    w: u32,
    cfg: &Mistral35PreprocessConfig,
) -> Result<(u32, u32), PreprocessError> {
    if h == 0 || w == 0 {
        return Err(PreprocessError::BadDims(w, h));
    }
    let factor = cfg.patch_size * cfg.spatial_merge_size; // 14*2=28
    if factor == 0 || cfg.longest_edge == 0 {
        return Err(PreprocessError::BadDims(w, h));
    }
    let max_side = w.max(h);
    let (mut rh, mut rw) = if max_side <= cfg.longest_edge {
        (h, w)
    } else {
        let scale = cfg.longest_edge as f64 / max_side as f64;
        let nh = ((h as f64) * scale).floor().max(1.0) as u32;
        let nw = ((w as f64) * scale).floor().max(1.0) as u32;
        (nh, nw)
    };
    rh = (rh / factor) * factor;
    rw = (rw / factor) * factor;
    if rh == 0 {
        rh = factor;
    }
    if rw == 0 {
        rw = factor;
    }
    Ok((rh, rw))
}

/// Decode + resize + normalize + patchify for Pixtral. Pure host
/// work; the GPU forward (Step 7-GPU) takes `pixel_values` and the
/// merged grid as input.
pub fn preprocess_mistral35_pixtral(
    img: &RgbImage,
    cfg: &Mistral35PreprocessConfig,
) -> Result<Mistral35Patches, PreprocessError> {
    let (target_h, target_w) =
        mistral35_pixtral_resize_dims(img.height(), img.width(), cfg)?;

    let resized = if (target_w, target_h) == (img.width(), img.height()) {
        img.clone()
    } else {
        // CatmullRom is the closest stdlib equivalent of PIL BICUBIC,
        // matching the existing Qwen/Gemma path. Pixtral's Python
        // reference uses BICUBIC with antialias=True; we accept a
        // small numeric drift here and let the GPU forward's cosine
        // tests gate any regression (same approach as Qwen).
        image::imageops::resize(img, target_w, target_h, FilterType::CatmullRom)
    };

    let h = target_h as usize;
    let w = target_w as usize;
    let p = cfg.patch_size as usize;

    // Rescale to [0, 1] then normalize: (x - mean) / std, per channel.
    let mut chw = vec![0.0f32; 3 * h * w];
    let inv = 1.0 / 255.0;
    for (i, px) in resized.pixels().enumerate() {
        for c in 0..3 {
            let v = (px[c] as f32) * inv;
            chw[c * h * w + i] = (v - cfg.image_mean[c]) / cfg.image_std[c];
        }
    }

    // Patchify [3, H, W] → [num_h * num_w, p * p * 3] in row-major.
    // Inner ordering matches Gemma's `[ip, jp, c]` so the existing
    // patch-conv kernel signature transfers across families.
    let num_h = h / p;
    let num_w = w / p;
    let n_patches = num_h * num_w;
    let patch_dim = p * p * 3;
    let mut out = vec![0.0f32; n_patches * patch_dim];
    for ph in 0..num_h {
        for pw in 0..num_w {
            let row = ph * num_w + pw;
            for ip in 0..p {
                for jp in 0..p {
                    for c in 0..3 {
                        let src = c * h * w + (ph * p + ip) * w + (pw * p + jp);
                        let col = ip * p * 3 + jp * 3 + c;
                        out[row * patch_dim + col] = chw[src];
                    }
                }
            }
        }
    }

    let merged_h = num_h as u32 / cfg.spatial_merge_size;
    let merged_w = num_w as u32 / cfg.spatial_merge_size;
    let num_soft_tokens = (merged_h as usize) * (merged_w as usize);
    if num_soft_tokens == 0 {
        return Err(PreprocessError::BadDims(target_w, target_h));
    }
    Ok(Mistral35Patches {
        pixel_values: out,
        resized: (target_h, target_w),
        patch_grid: (num_h as u32, num_w as u32),
        merged_grid: (merged_h, merged_w),
        num_soft_tokens,
    })
}

// ─── Tests vs HF reference fixtures ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    fn load_f32(path: &str) -> Vec<f32> {
        let raw = std::fs::read(path).expect("fixture");
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn qwen_smart_resize_224_to_256() {
        // Default qwen3-6 config: factor=32, min=65536, max=16777216.
        // 224×224 → 224·224=50176 < 65536, so upscale.
        // beta = sqrt(65536/50176) = 1.143
        // h_bar = ceil(224·1.143/32)·32 = ceil(8.0)·32 = 256
        let (h, w) = qwen_smart_resize(224, 224, 32, 65536, 16_777_216).unwrap();
        assert_eq!((h, w), (256, 256));
    }

    #[test]
    fn qwen_preprocess_test_224_matches_hf() {
        // Fixture generated by tools/gen_fixtures.py with HF processor.
        let img_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/test_224.png");
        let ref_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen_test_224_pixel_values_f32.bin"
        );
        if !std::path::Path::new(img_path).exists() {
            eprintln!("skipping: fixture {img_path} not found");
            return;
        }
        let img = decode_image(&std::fs::read(img_path).unwrap()).unwrap();
        // The HF Qwen2VLImageProcessorFast reference fixture under
        // tests/fixtures/qwen_test_224_pixel_values_f32.bin was
        // captured BEFORE we dropped the min_pixels=65536 default
        // (which used to upsample 224×224 → 256×256 / grid 16×16).
        // The current default is min_pixels=0 → 224 stays 224, grid
        // 14×14, num_patches=196 — matching HF stage-dump output we
        // verified Qwen-vision against (cos=0.9999/layer).
        // Re-run the explicit-cfg variant against the OLD reference
        // so the test still gates the byte-faithful preprocess path
        // rather than the operator default. Codex review #5 (round 4).
        let cfg = QwenPreprocessConfig {
            min_pixels: 65536,
            ..QwenPreprocessConfig::default()
        };
        let pp = preprocess_qwen(&img, &cfg).unwrap();
        assert_eq!(pp.grid_thw, [1, 16, 16]);
        assert_eq!(pp.num_patches(), 256);
        assert_eq!(pp.patch_dim(), 1536);

        let hf_ref = load_f32(ref_path);
        assert_eq!(hf_ref.len(), pp.pixel_values.len());
        let cs = cosine_sim(&pp.pixel_values, &hf_ref);
        let max_abs = pp
            .pixel_values
            .iter()
            .zip(hf_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "qwen preprocess vs HF: cosine_sim={cs:.6} max_abs_err={max_abs:.4}"
        );
        // PIL BICUBIC vs image::CatmullRom: small per-pixel deviation
        // expected; cosine should still be very high.
        assert!(cs >= 0.999, "qwen preprocess deviates too much: cos={cs}");
    }

    #[test]
    fn gemma_aspect_resize_224_to_768() {
        // 224×224 → factor=sqrt(2520·256/50176)=sqrt(12.857)=3.585
        // ideal = 803.07, side_mult=48, target = floor(803.07/48)·48 = 16·48 = 768
        let (h, w) = gemma_aspect_resize_dims(224, 224, &GemmaPreprocessConfig::default()).unwrap();
        assert_eq!((h, w), (768, 768));
    }

    #[test]
    fn gemma_preprocess_test_224_matches_hf() {
        let img_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/test_224.png");
        let ref_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/gemma_test_224_pixel_values_f32.bin"
        );
        let pos_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/gemma_test_224_position_ids_i64.bin"
        );
        if !std::path::Path::new(img_path).exists() {
            eprintln!("skipping: fixture {img_path} not found");
            return;
        }
        let img = decode_image(&std::fs::read(img_path).unwrap()).unwrap();
        let cfg = GemmaPreprocessConfig::default();
        let pp = preprocess_gemma(&img, &cfg).unwrap();
        assert_eq!(pp.resized, (768, 768));
        // 768/16 = 48 patches per side, 48·48 = 2304 valid; padded to 2520
        assert_eq!(pp.num_valid_patches, 2304);
        assert_eq!(pp.num_soft_tokens, 2304 / 9);

        let hf_ref = load_f32(ref_path);
        assert_eq!(hf_ref.len(), pp.pixel_values.len());
        let cs = cosine_sim(&pp.pixel_values, &hf_ref);
        let max_abs = pp
            .pixel_values
            .iter()
            .zip(hf_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "gemma preprocess vs HF: cosine_sim={cs:.6} max_abs_err={max_abs:.4}"
        );
        assert!(cs >= 0.999, "gemma preprocess deviates too much: cos={cs}");

        // Position IDs should match exactly (no resize involved).
        let hf_pos: Vec<i64> = std::fs::read(pos_path)
            .unwrap()
            .chunks_exact(8)
            .map(|c| {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            })
            .collect();
        assert_eq!(hf_pos.len(), pp.position_ids.len());
        for (i, (a, b)) in hf_pos.iter().zip(pp.position_ids.iter()).enumerate() {
            assert_eq!(a, b, "pos[{i}] mismatch: hf={a} ours={b}");
        }
    }

    // ─── Mistral 3.5 / Pixtral preprocessor ─────────────────────────

    fn solid_rgb(w: u32, h: u32, c: [u8; 3]) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for (_, _, px) in img.enumerate_pixels_mut() {
            *px = image::Rgb(c);
        }
        img
    }

    #[test]
    fn mistral35_resize_dims_square_at_longest_edge() {
        let cfg = Mistral35PreprocessConfig::default();
        let (h, w) = mistral35_pixtral_resize_dims(1540, 1540, &cfg).unwrap();
        assert_eq!((h, w), (1540, 1540));
    }

    #[test]
    fn mistral35_resize_dims_wide_scales_down() {
        let cfg = Mistral35PreprocessConfig::default();
        // 1540×3080 → scale 0.5 → (770, 1540) → factor-28 round →
        // (756, 1540).
        let (h, w) = mistral35_pixtral_resize_dims(1540, 3080, &cfg).unwrap();
        assert_eq!((h, w), (756, 1540));
    }

    #[test]
    fn mistral35_resize_dims_tiny_snaps_up() {
        let cfg = Mistral35PreprocessConfig::default();
        let (h, w) = mistral35_pixtral_resize_dims(10, 10, &cfg).unwrap();
        assert_eq!((h, w), (28, 28));
    }

    #[test]
    fn mistral35_preprocess_shape_and_normalisation() {
        let cfg = Mistral35PreprocessConfig::default();
        // 56×56 mid-gray → already a multiple of factor=28; resize is
        // a no-op so we get exact arithmetic for the assertions.
        let img = solid_rgb(56, 56, [128, 128, 128]);
        let pp = preprocess_mistral35_pixtral(&img, &cfg).unwrap();

        assert_eq!(pp.resized, (56, 56));
        assert_eq!(pp.patch_grid, (4, 4));
        assert_eq!(pp.merged_grid, (2, 2));
        assert_eq!(pp.num_soft_tokens, 4);
        assert_eq!(pp.num_patches(), 16);
        // pixel_values length == num_patches * 3 * patch²
        assert_eq!(pp.pixel_values.len(), 16 * 3 * 14 * 14);

        // Mid-gray (128/255 ≈ 0.5020) after CLIP normalisation:
        //   r: (0.5020 - 0.4815) / 0.2686 ≈ 0.0764
        //   g: (0.5020 - 0.4578) / 0.2613 ≈ 0.1689
        //   b: (0.5020 - 0.4082) / 0.2758 ≈ 0.3402
        let p0 = &pp.pixel_values[0..3];
        let expect = |v: f32, m: f32, s: f32| (v - m) / s;
        let mid = 128.0f32 / 255.0;
        assert!((p0[0] - expect(mid, 0.48145466, 0.26862954)).abs() < 1e-3);
        assert!((p0[1] - expect(mid, 0.4578275, 0.26130258)).abs() < 1e-3);
        assert!((p0[2] - expect(mid, 0.40821073, 0.27577711)).abs() < 1e-3);
    }

    /// Round-10 #3 layout-sensitive test.
    ///
    /// Builds a 28×28 image where the four 14×14 patches are each a
    /// solid distinct colour. The expected HWC output is
    /// `[num_patches=4, patch_pixels=14*14, channels=3]`. Walks each
    /// patch and asserts that *every* one of the 196 inner pixels
    /// holds the patch's colour after CLIP normalisation. This is
    /// stride-sensitive: a CHW-stored implementation (the pre-fix
    /// doc) would interleave the channels at the wrong stride and
    /// the channel triplet would not match the patch colour.
    #[test]
    fn mistral35_preprocess_patch_layout_is_hwc() {
        use image::Rgb;
        let cfg = Mistral35PreprocessConfig::default();

        // 4 patches at 14×14, top-left red, top-right green,
        // bottom-left blue, bottom-right yellow.
        let mut img = image::ImageBuffer::new(28, 28);
        for y in 0..28u32 {
            for x in 0..28u32 {
                let (qy, qx) = (y / 14, x / 14);
                let rgb = match (qy, qx) {
                    (0, 0) => [200u8, 10, 10],   // red-ish
                    (0, 1) => [10, 200, 10],     // green-ish
                    (1, 0) => [10, 10, 200],     // blue-ish
                    (1, 1) => [200, 200, 10],    // yellow-ish
                    _ => unreachable!(),
                };
                img.put_pixel(x, y, Rgb(rgb));
            }
        }
        let pp = preprocess_mistral35_pixtral(&img, &cfg).unwrap();
        assert_eq!(pp.patch_grid, (2, 2));
        assert_eq!(pp.num_patches(), 4);
        assert_eq!(pp.pixel_values.len(), 4 * 14 * 14 * 3);

        // Patch order is (ph, pw) document-order, so:
        //   row 0 = (0,0) red    row 2 = (1,0) blue
        //   row 1 = (0,1) green  row 3 = (1,1) yellow
        let expected_rgb_per_patch = [
            [200u8, 10, 10],
            [10, 200, 10],
            [10, 10, 200],
            [200, 200, 10],
        ];
        let norm = |v: u8, m: f32, s: f32| ((v as f32) / 255.0 - m) / s;
        let mean = cfg.image_mean;
        let std = cfg.image_std;
        let patch_dim = 14 * 14 * 3;
        for (patch_idx, rgb) in expected_rgb_per_patch.iter().enumerate() {
            let er = norm(rgb[0], mean[0], std[0]);
            let eg = norm(rgb[1], mean[1], std[1]);
            let eb = norm(rgb[2], mean[2], std[2]);
            let row = &pp.pixel_values[patch_idx * patch_dim ..(patch_idx + 1) * patch_dim];
            // HWC layout: triplets at offsets 0,3,6,...
            for px in 0..14 * 14 {
                let r = row[px * 3];
                let g = row[px * 3 + 1];
                let b = row[px * 3 + 2];
                assert!((r - er).abs() < 1e-3,
                    "patch {patch_idx} px {px}: r={r} expected {er}");
                assert!((g - eg).abs() < 1e-3,
                    "patch {patch_idx} px {px}: g={g} expected {eg}");
                assert!((b - eb).abs() < 1e-3,
                    "patch {patch_idx} px {px}: b={b} expected {eb}");
            }
        }
    }

    #[test]
    fn mistral35_preprocess_token_count_matches_predict() {
        // Cross-check with the API-side predictor in `vision_fetch`.
        // Both must agree on every dim or the worker rejects the
        // request — make sure they do.
        let cases = [
            (200u32, 200u32),
            (300, 100),
            (1540, 770),
            (3000, 4000),
        ];
        let cfg = Mistral35PreprocessConfig::default();
        for (w, h) in cases {
            let (rh, rw) = mistral35_pixtral_resize_dims(h, w, &cfg).unwrap();
            let factor = cfg.patch_size * cfg.spatial_merge_size;
            let merged_h = rh / factor;
            let merged_w = rw / factor;
            assert!(
                merged_h > 0 && merged_w > 0,
                "merged grid zero for {w}x{h} (resized {rw}x{rh})"
            );
        }
    }

    #[test]
    fn mistral35_preprocess_zero_dims_rejected() {
        let cfg = Mistral35PreprocessConfig::default();
        assert!(mistral35_pixtral_resize_dims(0, 100, &cfg).is_err());
        assert!(mistral35_pixtral_resize_dims(100, 0, &cfg).is_err());
    }
}
