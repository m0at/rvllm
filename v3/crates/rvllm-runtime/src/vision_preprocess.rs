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
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage, PreprocessError> {
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
            // bound; max_pixels still caps at 16M to match HF.
            min_pixels: 0,
            max_pixels: 16_777_216,
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
        let cfg = QwenPreprocessConfig::default();
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
}
