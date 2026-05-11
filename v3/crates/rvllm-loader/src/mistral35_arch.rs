//! Mistral Medium 3.5 (NVFP4-pack-quantized) architecture parser.
//!
//! Source of truth: `mistral-35-integration.md` and the public
//! checkpoint `zdy1995love/Mistral-Medium-3.5-128B-NVFP4` config.json.
//! This module parses *only* — no tensor I/O, no CUDA — so the
//! `rvllm-serve` resolver and integration tests can pull in the same
//! invariants without dragging in cudarc.
//!
//! Key markers:
//!   - `architectures[0] == "Mistral3ForConditionalGeneration"`
//!   - `model_type == "mistral3"`
//!   - `quantization_config.format == "nvfp4-pack-quantized"`
//!
//! Decoder shape (88 layers, dense, GQA-12, head_dim=128). YaRN RoPE
//! with `mscale_all_dim == 0.0` (a known checkpoint correction —
//! non-zero values silently break the YaRN math, so we refuse to
//! load). Pixtral vision tower (48 layers, head_dim=104, longest-edge
//! 1540, spatial_merge_size=2).

use std::path::{Path, PathBuf};

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};

/// Top-level Mistral 3.5 arch summary. Includes both the dense
/// decoder block (`text`) and the Pixtral vision tower (`vision`).
#[derive(Clone, Debug)]
pub struct Mistral35Arch {
    pub text: Mistral35TextArch,
    pub vision: Mistral35VisionArch,
    /// `image_token_index` at the top level of config.json (10 in
    /// the public checkpoint). Mistral chat templates emit this id
    /// once per image; the placeholder-expansion path replaces each
    /// occurrence with `vision_items[i].num_tokens` copies.
    pub image_token_index: u32,
    /// Resolved weight prefix (always `model.language_model` in
    /// the public checkpoint, but parsed from the safetensors index
    /// to stay tolerant of minor renames).
    pub weight_prefix: String,
}

#[derive(Clone, Debug)]
pub struct Mistral35TextArch {
    pub num_hidden_layers: usize,    // 88
    pub hidden_size: usize,          // 12288
    pub intermediate_size: usize,    // 28672
    pub num_attention_heads: usize,  // 96
    pub num_key_value_heads: usize,  // 8 (GQA ratio 12)
    pub head_dim: usize,             // 128
    pub vocab_size: usize,           // 131072
    pub max_position_embeddings: usize, // 262144
    pub rms_norm_eps: f32,           // 1e-5
    /// silu — caller asserts the activation. We only validate that
    /// `hidden_act == "silu"`; any other value rejects.
    pub hidden_act_silu: bool,
    pub tie_word_embeddings: bool, // false
    pub yarn: YarnRopeConfig,
}

#[derive(Clone, Debug)]
pub struct YarnRopeConfig {
    pub rope_theta: f32,                          // 1_000_000.0
    pub original_max_position_embeddings: usize,  // 4096
    pub factor: f32,                              // 64.0
    pub beta_fast: f32,                           // 4.0
    pub beta_slow: f32,                           // 1.0
    pub mscale: f32,                              // 1.0
    /// MUST be 0.0 — a known checkpoint correction. Stored for
    /// symmetry; loader rejects non-zero before constructing this.
    pub mscale_all_dim: f32,
}

#[derive(Clone, Debug)]
pub struct Mistral35VisionArch {
    pub model_type_pixtral: bool,    // "pixtral"
    pub hidden_size: usize,          // 1664
    pub num_hidden_layers: usize,    // 48
    pub num_attention_heads: usize,  // 16
    pub head_dim: usize,             // 104
    pub intermediate_size: usize,    // 8192
    pub patch_size: usize,           // 14
    pub image_size: usize,           // 1540 (longest edge)
    pub num_channels: usize,         // 3
    pub rope_theta: f32,             // 10_000.0
    pub spatial_merge_size: usize,   // 2 (from processor section)
}

impl Mistral35Arch {
    /// Probe a model directory and parse the full Mistral 3.5 arch
    /// if the markers match. Returns `Ok(None)` for non-Mistral
    /// configs so the caller can chain probes.
    pub fn from_dir(model_dir: &Path) -> Result<Option<Self>> {
        let cfg_path = model_dir.join("config.json");
        let bytes = match std::fs::read(&cfg_path) {
            Ok(b) => b,
            Err(_) => return Ok(None),
        };
        let v: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                return Err(corrupt(
                    cfg_path,
                    format!("config.json parse: {e}"),
                ));
            }
        };

        // Marker triple. Any failure returns Ok(None) — the resolver
        // distinguishes "not Mistral" from "Mistral with bad fields".
        let arch_marker = v["architectures"][0]
            .as_str()
            .map(|s| s == "Mistral3ForConditionalGeneration")
            .unwrap_or(false);
        let mt_marker = v["model_type"].as_str() == Some("mistral3");
        let q_marker = v["quantization_config"]["format"]
            .as_str()
            .map(|s| s == "nvfp4-pack-quantized")
            .unwrap_or(false);
        if !(arch_marker && mt_marker && q_marker) {
            return Ok(None);
        }

        let text = parse_text(&v, &cfg_path)?;
        let vision = parse_vision(&v, &cfg_path)?;

        let image_token_index = v["image_token_index"].as_u64().unwrap_or(10) as u32;
        let weight_prefix = detect_weight_prefix(model_dir);

        Ok(Some(Self { text, vision, image_token_index, weight_prefix }))
    }

    /// Render a one-line summary suitable for `tracing::info!` /
    /// `eprintln!`. The loader crate intentionally has no `tracing`
    /// dep; callers feed this through their own logger.
    pub fn summary(&self) -> String {
        format!(
            "Mistral 3.5 NVFP4: layers={} hidden={} heads={}/{}kv hd={} vocab={} \
             max_pos={} yarn(factor={} orig_max={} mscale={} all_dim={}) | \
             vision: layers={} hidden={} hd={} img={} merge={} | \
             image_token={} prefix={}",
            self.text.num_hidden_layers,
            self.text.hidden_size,
            self.text.num_attention_heads,
            self.text.num_key_value_heads,
            self.text.head_dim,
            self.text.vocab_size,
            self.text.max_position_embeddings,
            self.text.yarn.factor,
            self.text.yarn.original_max_position_embeddings,
            self.text.yarn.mscale,
            self.text.yarn.mscale_all_dim,
            self.vision.num_hidden_layers,
            self.vision.hidden_size,
            self.vision.head_dim,
            self.vision.image_size,
            self.vision.spatial_merge_size,
            self.image_token_index,
            self.weight_prefix,
        )
    }

    /// Logical projection shapes per layer. Used by the loader and
    /// CUTLASS bring-up to validate every NVFP4 linear.
    pub fn q_rows(&self) -> usize {
        self.text.num_attention_heads * self.text.head_dim
    }
    pub fn kv_rows(&self) -> usize {
        self.text.num_key_value_heads * self.text.head_dim
    }
    pub fn gqa_ratio(&self) -> usize {
        self.text.num_attention_heads / self.text.num_key_value_heads
    }
}

fn parse_text(
    v: &serde_json::Value,
    cfg_path: &Path,
) -> Result<Mistral35TextArch> {
    // Mistral's config.json nests the language model under
    // "text_config" (parallel to "vision_config"). Fall through to
    // the top level for robustness.
    let tc = if v.get("text_config").is_some() {
        &v["text_config"]
    } else {
        v
    };

    let num_hidden_layers = u("num_hidden_layers", tc, cfg_path)?;
    let hidden_size = u("hidden_size", tc, cfg_path)?;
    let intermediate_size = u("intermediate_size", tc, cfg_path)?;
    let num_attention_heads = u("num_attention_heads", tc, cfg_path)?;
    let num_key_value_heads = u("num_key_value_heads", tc, cfg_path)?;
    let head_dim = u("head_dim", tc, cfg_path)?;
    let vocab_size = u("vocab_size", tc, cfg_path)?;
    let max_position_embeddings = u("max_position_embeddings", tc, cfg_path)?;
    let rms_norm_eps = tc["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32;

    let hidden_act = tc["hidden_act"].as_str().unwrap_or("silu");
    if hidden_act != "silu" {
        return Err(corrupt(
            cfg_path.to_path_buf(),
            format!("Mistral 3.5 expects hidden_act=silu (got {hidden_act:?})"),
        ));
    }

    let tie_word_embeddings = tc["tie_word_embeddings"]
        .as_bool()
        .or_else(|| v["tie_word_embeddings"].as_bool())
        .unwrap_or(false);
    if tie_word_embeddings {
        return Err(corrupt(
            cfg_path.to_path_buf(),
            "Mistral 3.5 expects tie_word_embeddings=false".into(),
        ));
    }

    // YaRN RoPE — Mistral nests under either `rope_scaling` or
    // `rope_parameters`. We accept both names; the public checkpoint
    // uses `rope_scaling`.
    let rope = if tc.get("rope_scaling").is_some() {
        &tc["rope_scaling"]
    } else if tc.get("rope_parameters").is_some() {
        &tc["rope_parameters"]
    } else {
        return Err(corrupt(
            cfg_path.to_path_buf(),
            "Mistral 3.5 config missing rope_scaling/rope_parameters".into(),
        ));
    };

    let rope_type = rope["rope_type"]
        .as_str()
        .or_else(|| rope["type"].as_str())
        .unwrap_or("");
    if rope_type != "yarn" {
        return Err(corrupt(
            cfg_path.to_path_buf(),
            format!("Mistral 3.5 expects rope_type=yarn (got {rope_type:?})"),
        ));
    }

    // Round-11 #2: critical YaRN fields are required (no defaults).
    // The runtime YaRN tables and RoPE kernel are hard-coded for the
    // canonical Mistral Medium 3.5 values; a config that's missing
    // factor/beta/mscale would silently default to the canonical
    // numbers, hiding incompatible checkpoints until they generate
    // garbage at long context. mscale_all_dim is the one value
    // that's explicitly checked-and-rejected below, so it can default
    // (any non-zero rejects).
    // The public Mistral 3.5 checkpoint nests `rope_theta` inside the
    // YaRN block (`rope_parameters.rope_theta` / `rope_scaling.rope_theta`)
    // rather than the top of `text_config`. Accept either location.
    let rope_theta_value = tc["rope_theta"].as_f64()
        .or_else(|| rope["rope_theta"].as_f64())
        .or_else(|| tc["rope_theta"].as_u64().map(|x| x as f64))
        .or_else(|| rope["rope_theta"].as_u64().map(|x| x as f64))
        .ok_or_else(|| corrupt(
            cfg_path.to_path_buf(),
            "Mistral 3.5 config missing required numeric field \
             rope_theta (looked in text_config and rope_scaling/\
             rope_parameters)".into(),
        ))?;
    let yarn = YarnRopeConfig {
        rope_theta: rope_theta_value as f32,
        original_max_position_embeddings:
            require_u64("original_max_position_embeddings", rope, cfg_path)? as usize,
        factor: require_f64("factor", rope, cfg_path)? as f32,
        beta_fast: require_f64("beta_fast", rope, cfg_path)? as f32,
        beta_slow: require_f64("beta_slow", rope, cfg_path)? as f32,
        mscale: require_f64("mscale", rope, cfg_path)? as f32,
        mscale_all_dim: rope["mscale_all_dim"].as_f64().unwrap_or(0.0) as f32,
    };
    if yarn.mscale_all_dim != 0.0 {
        return Err(corrupt(
            cfg_path.to_path_buf(),
            format!(
                "Mistral 3.5 YaRN requires mscale_all_dim=0.0 (got {}); \
                 the public checkpoint has a known config correction \
                 for this — refusing to load",
                yarn.mscale_all_dim
            ),
        ));
    }

    // Sanity: GQA ratio must be an integer.
    if num_key_value_heads == 0 || num_attention_heads % num_key_value_heads != 0 {
        return Err(corrupt(
            cfg_path.to_path_buf(),
            format!(
                "Mistral 3.5 GQA: num_attention_heads={num_attention_heads} must \
                 be a positive multiple of num_key_value_heads={num_key_value_heads}"
            ),
        ));
    }

    Ok(Mistral35TextArch {
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        hidden_act_silu: true,
        tie_word_embeddings,
        yarn,
    })
}

fn parse_vision(
    v: &serde_json::Value,
    cfg_path: &Path,
) -> Result<Mistral35VisionArch> {
    let vc = if v.get("vision_config").is_some() {
        &v["vision_config"]
    } else {
        return Err(corrupt(
            cfg_path.to_path_buf(),
            "Mistral 3.5 config missing vision_config block".into(),
        ));
    };

    let pixtral = vc["model_type"].as_str() == Some("pixtral");
    if !pixtral {
        return Err(corrupt(
            cfg_path.to_path_buf(),
            format!(
                "Mistral 3.5 expects vision_config.model_type=pixtral (got {:?})",
                vc["model_type"].as_str().unwrap_or("?")
            ),
        ));
    }

    // Round-11 #2: every Pixtral field is required + validated against
    // the canonical Mistral Medium 3.5 / Pixtral values the runtime
    // kernels are hard-coded for. A future Mistral checkpoint that
    // changes any of these would need explicit kernel work, so we
    // refuse to load instead of silently producing wrong output.
    let hidden_size = require_u64("hidden_size", vc, cfg_path)? as usize;
    let num_hidden_layers = require_u64("num_hidden_layers", vc, cfg_path)? as usize;
    let num_attention_heads = require_u64("num_attention_heads", vc, cfg_path)? as usize;
    let head_dim = require_u64("head_dim", vc, cfg_path)? as usize;
    let intermediate_size = require_u64("intermediate_size", vc, cfg_path)? as usize;
    let patch_size = require_u64("patch_size", vc, cfg_path)? as usize;
    let image_size = require_u64("image_size", vc, cfg_path)? as usize;
    let num_channels = require_u64("num_channels", vc, cfg_path)? as usize;
    // Pixtral nests `rope_theta` under `rope_parameters` in the public
    // Mistral 3.5 checkpoint (vision_config.rope_parameters.rope_theta).
    // Accept either flat or nested.
    let rope_theta = vc["rope_theta"].as_f64()
        .or_else(|| vc["rope_parameters"]["rope_theta"].as_f64())
        .or_else(|| vc["rope_theta"].as_u64().map(|x| x as f64))
        .or_else(|| vc["rope_parameters"]["rope_theta"].as_u64().map(|x| x as f64))
        .ok_or_else(|| corrupt(
            cfg_path.to_path_buf(),
            "Mistral 3.5 config missing vision_config.rope_theta \
             (looked at flat and rope_parameters.rope_theta)".into(),
        ))? as f32;

    expect_eq_usize("vision_config.hidden_size",         hidden_size,        1664, cfg_path)?;
    expect_eq_usize("vision_config.num_hidden_layers",   num_hidden_layers,  48,   cfg_path)?;
    expect_eq_usize("vision_config.num_attention_heads", num_attention_heads,16,   cfg_path)?;
    expect_eq_usize("vision_config.head_dim",            head_dim,           104,  cfg_path)?;
    expect_eq_usize("vision_config.intermediate_size",   intermediate_size,  8192, cfg_path)?;
    expect_eq_usize("vision_config.patch_size",          patch_size,         14,   cfg_path)?;
    expect_eq_usize("vision_config.image_size",          image_size,         1540, cfg_path)?;
    expect_eq_usize("vision_config.num_channels",        num_channels,       3,    cfg_path)?;
    expect_eq_f32  ("vision_config.rope_theta",          rope_theta,         10_000.0, cfg_path)?;

    // Processor `spatial_merge_size` is published either inside
    // `vision_config` or one level up under `processor_config`. The
    // value MUST equal 2; the soft-token predictor + patch-merger
    // kernel are coded for it.
    let spatial_merge_size = vc["spatial_merge_size"]
        .as_u64()
        .or_else(|| v["processor_config"]["spatial_merge_size"].as_u64())
        .or_else(|| v["spatial_merge_size"].as_u64())
        .ok_or_else(|| corrupt(
            cfg_path.to_path_buf(),
            "Mistral 3.5 config missing required field spatial_merge_size \
             (vision_config / processor_config / top-level)".into(),
        ))? as usize;
    expect_eq_usize("spatial_merge_size", spatial_merge_size, 2, cfg_path)?;

    Ok(Mistral35VisionArch {
        model_type_pixtral: true,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        head_dim,
        intermediate_size,
        patch_size,
        image_size,
        num_channels,
        rope_theta,
        spatial_merge_size,
    })
}

fn detect_weight_prefix(dir: &Path) -> String {
    let idx_path = dir.join("model.safetensors.index.json");
    if let Ok(bytes) = std::fs::read(&idx_path) {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if let Some(map) = v["weight_map"].as_object() {
                for key in map.keys() {
                    if key.starts_with("model.language_model.") {
                        return "model.language_model".into();
                    }
                    if key.starts_with("language_model.") {
                        return "language_model".into();
                    }
                }
            }
        }
    }
    "model.language_model".into()
}

fn u(field: &str, v: &serde_json::Value, cfg_path: &Path) -> Result<usize> {
    match v[field].as_u64() {
        Some(x) => Ok(x as usize),
        None => Err(corrupt(
            cfg_path.to_path_buf(),
            format!("Mistral 3.5 config missing required u64 field {field:?}"),
        )),
    }
}

/// Round-11 #2: required-u64 in a nested object (e.g. `rope_scaling.factor`).
fn require_u64(field: &str, obj: &serde_json::Value, cfg_path: &Path) -> Result<u64> {
    obj[field].as_u64().ok_or_else(|| corrupt(
        cfg_path.to_path_buf(),
        format!("Mistral 3.5 config missing required u64 field {field:?}"),
    ))
}

/// Round-11 #2: required-f64.
fn require_f64(field: &str, obj: &serde_json::Value, cfg_path: &Path) -> Result<f64> {
    // Accept ints written without a decimal point.
    obj[field].as_f64()
        .or_else(|| obj[field].as_u64().map(|x| x as f64))
        .or_else(|| obj[field].as_i64().map(|x| x as f64))
        .ok_or_else(|| corrupt(
            cfg_path.to_path_buf(),
            format!("Mistral 3.5 config missing required numeric field {field:?}"),
        ))
}

/// Round-11 #2: enforce that a kernel-ABI-critical config value
/// matches the Mistral 3.5 / Pixtral canonical value the runtime is
/// hard-coded for.
fn expect_eq_usize(field: &str, got: usize, expect: usize, cfg_path: &Path) -> Result<()> {
    if got != expect {
        return Err(corrupt(
            cfg_path.to_path_buf(),
            format!(
                "Mistral 3.5 expects {field}={expect} (got {got}); the \
                 runtime kernels and weight shapes are hard-coded for \
                 the Mistral Medium 3.5 / Pixtral canonical values"
            ),
        ));
    }
    Ok(())
}

fn expect_eq_f32(field: &str, got: f32, expect: f32, cfg_path: &Path) -> Result<()> {
    if (got - expect).abs() > 1e-3 {
        return Err(corrupt(
            cfg_path.to_path_buf(),
            format!(
                "Mistral 3.5 expects {field}={expect} (got {got}); the \
                 runtime kernels and weight shapes are hard-coded for \
                 the Mistral Medium 3.5 / Pixtral canonical values"
            ),
        ));
    }
    Ok(())
}

fn corrupt(path: PathBuf, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx { path, tensor: None },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tempdir() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rvllm-loader-mistral35-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("mkdir");
        p
    }

    fn write(dir: &Path, body: &str) {
        let mut f = std::fs::File::create(dir.join("config.json")).expect("create");
        f.write_all(body.as_bytes()).expect("write");
    }

    fn full_config() -> &'static str {
        r#"{
          "architectures": ["Mistral3ForConditionalGeneration"],
          "model_type": "mistral3",
          "quantization_config": {"format": "nvfp4-pack-quantized"},
          "image_token_index": 10,
          "text_config": {
            "num_hidden_layers": 88,
            "hidden_size": 12288,
            "intermediate_size": 28672,
            "num_attention_heads": 96,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 131072,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "rope_theta": 1000000.0,
            "rope_scaling": {
              "rope_type": "yarn",
              "original_max_position_embeddings": 4096,
              "factor": 64.0,
              "beta_fast": 4.0,
              "beta_slow": 1.0,
              "mscale": 1.0,
              "mscale_all_dim": 0.0
            }
          },
          "vision_config": {
            "model_type": "pixtral",
            "hidden_size": 1664,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "head_dim": 104,
            "intermediate_size": 8192,
            "patch_size": 14,
            "image_size": 1540,
            "num_channels": 3,
            "rope_theta": 10000.0,
            "spatial_merge_size": 2
          }
        }"#
    }

    #[test]
    fn parses_full_config() {
        let tmp = tempdir();
        write(&tmp, full_config());
        let arch = Mistral35Arch::from_dir(&tmp).unwrap().expect("matches");
        assert_eq!(arch.text.num_hidden_layers, 88);
        assert_eq!(arch.text.hidden_size, 12288);
        assert_eq!(arch.text.num_attention_heads, 96);
        assert_eq!(arch.text.num_key_value_heads, 8);
        assert_eq!(arch.text.head_dim, 128);
        assert_eq!(arch.text.vocab_size, 131072);
        assert_eq!(arch.gqa_ratio(), 12);
        assert_eq!(arch.q_rows(), 12288);
        assert_eq!(arch.kv_rows(), 1024);
        assert_eq!(arch.text.yarn.factor, 64.0);
        assert_eq!(arch.text.yarn.original_max_position_embeddings, 4096);
        assert_eq!(arch.text.yarn.mscale_all_dim, 0.0);
        assert_eq!(arch.vision.num_hidden_layers, 48);
        assert_eq!(arch.vision.head_dim, 104);
        assert_eq!(arch.vision.image_size, 1540);
        assert_eq!(arch.vision.spatial_merge_size, 2);
        assert_eq!(arch.image_token_index, 10);
    }

    #[test]
    fn returns_none_on_non_mistral() {
        let tmp = tempdir();
        write(
            &tmp,
            r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4"}"#,
        );
        assert!(Mistral35Arch::from_dir(&tmp).unwrap().is_none());
    }

    #[test]
    fn rejects_nonzero_mscale_all_dim() {
        let tmp = tempdir();
        let body = full_config().replace("\"mscale_all_dim\": 0.0", "\"mscale_all_dim\": 1.0");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("mscale_all_dim"));
    }

    #[test]
    fn rejects_non_silu_activation() {
        let tmp = tempdir();
        let body = full_config().replace("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("hidden_act"));
    }

    #[test]
    fn rejects_tied_embeddings() {
        let tmp = tempdir();
        let body = full_config()
            .replace("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("tie_word_embeddings"));
    }

    #[test]
    fn rejects_non_yarn_rope() {
        let tmp = tempdir();
        let body = full_config().replace("\"rope_type\": \"yarn\"", "\"rope_type\": \"linear\"");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("yarn"));
    }

    #[test]
    fn rejects_non_pixtral_vision() {
        let tmp = tempdir();
        let body = full_config().replace("\"model_type\": \"pixtral\"", "\"model_type\": \"clip\"");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("pixtral"));
    }

    #[test]
    fn rejects_bad_gqa_ratio() {
        let tmp = tempdir();
        let body = full_config().replace("\"num_key_value_heads\": 8", "\"num_key_value_heads\": 7");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("GQA"));
    }

    // Round-11 #2 strict-validation tests.

    #[test]
    fn rejects_missing_yarn_factor() {
        let tmp = tempdir();
        let body = full_config().replace("\"factor\": 64.0,", "");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("factor"), "want factor mention, got: {msg}");
    }

    #[test]
    fn rejects_missing_yarn_mscale() {
        let tmp = tempdir();
        let body = full_config().replace("\"mscale\": 1.0,", "");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("mscale"));
    }

    #[test]
    fn rejects_missing_yarn_original_max() {
        let tmp = tempdir();
        let body = full_config().replace("\"original_max_position_embeddings\": 4096,", "");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("original_max_position_embeddings"));
    }

    #[test]
    fn rejects_wrong_vision_head_dim() {
        let tmp = tempdir();
        let body = full_config().replace("\"head_dim\": 104", "\"head_dim\": 128");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("head_dim") && msg.contains("104"),
            "want head_dim+104 mention, got: {msg}");
    }

    #[test]
    fn rejects_wrong_patch_size() {
        let tmp = tempdir();
        let body = full_config().replace("\"patch_size\": 14", "\"patch_size\": 16");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("patch_size"));
    }

    #[test]
    fn rejects_wrong_spatial_merge() {
        let tmp = tempdir();
        let body = full_config().replace("\"spatial_merge_size\": 2", "\"spatial_merge_size\": 3");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("spatial_merge_size"));
    }

    #[test]
    fn rejects_missing_spatial_merge() {
        let tmp = tempdir();
        let body = full_config().replace(",\n            \"spatial_merge_size\": 2", "");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("spatial_merge_size"));
    }

    #[test]
    fn rejects_wrong_vision_image_size() {
        let tmp = tempdir();
        let body = full_config().replace("\"image_size\": 1540", "\"image_size\": 1024");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("image_size"));
    }

    #[test]
    fn rejects_missing_rope_theta() {
        let tmp = tempdir();
        let body = full_config().replace("\"rope_theta\": 1000000.0,", "");
        write(&tmp, &body);
        let err = Mistral35Arch::from_dir(&tmp).unwrap_err();
        assert!(format!("{err:?}").contains("rope_theta"));
    }
}
