//! Single source of truth for resolving the loaded model's family
//! (Qwen 3.6 / Gemma 4 / Mistral 3.5) from a model directory.
//!
//! Until this module landed, vision-arch detection in `main.rs` and
//! decoder dispatch in `cuda_worker.rs::spawn_cuda_worker` each
//! hand-rolled their own probe. The two could disagree if a future
//! family was wired into one site but not the other; the chat
//! template, image-token predictor, and worker would then disagree on
//! what was loaded. Both sites now call `resolve_model_family`.
//!
//! Markers used (see `mistral-35-integration.md`, the Qwen 3.6
//! `Qwen36Arch::from_dir` probe, and `Gemma4Arch::from_dir`):
//!
//! - **Mistral 3.5**: `architectures[0] == "Mistral3ForConditionalGeneration"`
//!   AND `model_type == "mistral3"` AND
//!   `quantization_config.format == "nvfp4-pack-quantized"`.
//! - **Qwen 3.6**: `Qwen36Arch::from_dir` (MoE marker count + linear
//!   layer present + `attn_output_gate=true`).
//! - **Gemma 4**: fall-through default.
//!
//! Explicit `--model-family` overrides assert that the requested
//! family matches the markers and fail with a typed error on mismatch
//! — never silently fall through.

use std::path::Path;

use crate::config::ModelFamily;
use crate::router::VisionArch;

#[derive(Debug, thiserror::Error)]
pub enum FamilyResolveError {
    #[error("config.json missing or unreadable at {path}: {source}")]
    ConfigUnreadable {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("config.json at {path} is not valid JSON: {source}")]
    ConfigJson {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "operator selected --model-family={requested} but config.json at {path} does not match \
         (markers seen: {markers}); refusing to fall through to a different family"
    )]
    Mismatch {
        requested: &'static str,
        path: std::path::PathBuf,
        markers: String,
    },
    #[error("Mistral 3.5 config requires mscale_all_dim=0.0 (got {got}); refusing to load")]
    MistralBadMscaleAllDim { got: f64 },
    #[error("internal: {0}")]
    Other(String),
}

/// Selected family + matching `VisionArch` for the HTTP layer.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedFamily {
    pub family: ModelFamily,
    pub vision_arch: VisionArch,
}

impl ResolvedFamily {
    pub fn log_summary(&self, model_dir: &Path) {
        tracing::info!(
            family = %self.family.as_str(),
            vision_arch = ?self.vision_arch,
            model_dir = %model_dir.display(),
            "model family resolved",
        );
    }
}

/// Inspect `<model_dir>/config.json` and decide which decoder/vision
/// stack to load. `selected` is the operator preference (`Auto` =
/// detect; explicit value = assert match).
pub fn resolve_model_family(
    model_dir: &Path,
    selected: ModelFamily,
) -> Result<ResolvedFamily, FamilyResolveError> {
    // Cheap probes first (don't fail just because there's no
    // config.json — Qwen probe handles its own absence gracefully).
    let mistral_image_token = is_mistral35(model_dir)?;
    let mistral_match = mistral_image_token.is_some();
    let qwen_match = is_qwen36(model_dir);

    match selected {
        ModelFamily::Auto => {
            if let Some(image_token_id) = mistral_image_token {
                Ok(ResolvedFamily {
                    family: ModelFamily::Mistral35,
                    vision_arch: VisionArch::Mistral35 { image_token_id },
                })
            } else if qwen_match {
                Ok(ResolvedFamily {
                    family: ModelFamily::Qwen36,
                    vision_arch: VisionArch::Qwen36,
                })
            } else {
                Ok(ResolvedFamily {
                    family: ModelFamily::Gemma4,
                    vision_arch: VisionArch::Gemma4,
                })
            }
        }
        ModelFamily::Mistral35 => {
            if let Some(image_token_id) = mistral_image_token {
                Ok(ResolvedFamily {
                    family: ModelFamily::Mistral35,
                    vision_arch: VisionArch::Mistral35 { image_token_id },
                })
            } else {
                Err(FamilyResolveError::Mismatch {
                    requested: "mistral35",
                    path: model_dir.join("config.json"),
                    markers: collect_markers(model_dir),
                })
            }
        }
        ModelFamily::Qwen36 => {
            if qwen_match {
                Ok(ResolvedFamily {
                    family: ModelFamily::Qwen36,
                    vision_arch: VisionArch::Qwen36,
                })
            } else {
                Err(FamilyResolveError::Mismatch {
                    requested: "qwen36",
                    path: model_dir.join("config.json"),
                    markers: collect_markers(model_dir),
                })
            }
        }
        ModelFamily::Gemma4 => {
            // Gemma 4 has no single-line config marker; we assert by
            // exclusion. If the dir matches Mistral or Qwen markers,
            // refusing here is the right thing — the user explicitly
            // asked for Gemma 4 and silently loading another family
            // would defeat the purpose of the explicit flag.
            if mistral_match || qwen_match {
                Err(FamilyResolveError::Mismatch {
                    requested: "gemma4",
                    path: model_dir.join("config.json"),
                    markers: collect_markers(model_dir),
                })
            } else {
                Ok(ResolvedFamily {
                    family: ModelFamily::Gemma4,
                    vision_arch: VisionArch::Gemma4,
                })
            }
        }
    }
}

fn read_config(model_dir: &Path) -> Result<Option<serde_json::Value>, FamilyResolveError> {
    let path = model_dir.join("config.json");
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(FamilyResolveError::ConfigUnreadable { path, source }),
    };
    let v: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|source| FamilyResolveError::ConfigJson { path: path.clone(), source })?;
    Ok(Some(v))
}

/// Round-10 #2: returns `Some(image_token_id)` when this is a Mistral
/// 3.5 NVFP4 checkpoint, `None` otherwise. `image_token_id` is read
/// from `config.json::image_token_index` so the renderer + handlers
/// can no longer rely on a hard-coded `[IMG] = 10`.
fn is_mistral35(model_dir: &Path) -> Result<Option<u32>, FamilyResolveError> {
    // Cheap marker probe first so a Gemma / Qwen dir doesn't pull in
    // the full Mistral arch parser (which validates YaRN, GQA ratio,
    // pixtral vision, etc.). Only when the three Mistral markers
    // match do we run the full parser — and at that point any field
    // failure surfaces as `FamilyResolveError::Mistral{...}`.
    let v = match read_config(model_dir)? {
        Some(v) => v,
        None => return Ok(None),
    };
    let arch_match = v["architectures"][0]
        .as_str()
        .map(|s| s == "Mistral3ForConditionalGeneration")
        .unwrap_or(false);
    let model_type_match = v["model_type"].as_str() == Some("mistral3");
    let quant_match = v["quantization_config"]["format"]
        .as_str()
        .map(|s| s == "nvfp4-pack-quantized")
        .unwrap_or(false);
    if !(arch_match && model_type_match && quant_match) {
        return Ok(None);
    }

    // Markers say Mistral 3.5 — run the full arch parser so YaRN /
    // GQA / pixtral invariants get validated *before* the worker
    // even starts. Any error here propagates so the operator sees
    // the field-level reason rather than a generic "wrong family".
    match rvllm_runtime::mistral35_arch::Mistral35Arch::from_dir(model_dir) {
        Ok(Some(arch)) => Ok(Some(arch.image_token_index)),
        Ok(None) => {
            // Markers matched but the parser returned None — should
            // not happen, treat as non-Mistral so the caller falls
            // through gracefully.
            Ok(None)
        }
        Err(e) => {
            // Surface the YaRN-correction case with its dedicated
            // error variant so the test suite + operator log keep
            // their existing message; everything else flows through
            // the generic `Other` variant.
            let msg = format!("{e:?}");
            if msg.contains("mscale_all_dim") {
                let got = v["text_config"]["rope_scaling"]["mscale_all_dim"]
                    .as_f64()
                    .or_else(|| v["rope_scaling"]["mscale_all_dim"].as_f64())
                    .unwrap_or(0.0);
                Err(FamilyResolveError::MistralBadMscaleAllDim { got })
            } else {
                Err(FamilyResolveError::Other(msg))
            }
        }
    }
}

fn is_qwen36(model_dir: &Path) -> bool {
    matches!(
        rvllm_runtime::qwen36_arch::Qwen36Arch::from_dir(model_dir),
        Ok(Some(_))
    )
}

fn collect_markers(model_dir: &Path) -> String {
    let v = match std::fs::read(model_dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
    {
        Some(v) => v,
        None => return "no config.json".into(),
    };
    let arch = v["architectures"][0].as_str().unwrap_or("?");
    let mt = v["model_type"].as_str().unwrap_or("?");
    let q = v["quantization_config"]["format"].as_str().unwrap_or("?");
    format!("architectures[0]={arch:?} model_type={mt:?} quant.format={q:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tempdir() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rvllm-serve-family-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("create_dir_all");
        p
    }

    fn write_config(dir: &Path, body: &str) {
        let mut f = std::fs::File::create(dir.join("config.json")).expect("create");
        f.write_all(body.as_bytes()).expect("write");
    }

    fn mistral_full() -> &'static str {
        r#"{
          "architectures": ["Mistral3ForConditionalGeneration"],
          "model_type": "mistral3",
          "quantization_config": {"format": "nvfp4-pack-quantized"},
          "image_token_index": 10,
          "text_config": {
            "num_hidden_layers": 88, "hidden_size": 12288,
            "intermediate_size": 28672, "num_attention_heads": 96,
            "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 131072, "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-5, "hidden_act": "silu",
            "tie_word_embeddings": false, "rope_theta": 1000000.0,
            "rope_scaling": {"rope_type":"yarn","original_max_position_embeddings":4096,
              "factor":64.0,"beta_fast":4.0,"beta_slow":1.0,"mscale":1.0,"mscale_all_dim":0.0}
          },
          "vision_config": {
            "model_type":"pixtral","hidden_size":1664,"num_hidden_layers":48,
            "num_attention_heads":16,"head_dim":104,"intermediate_size":8192,
            "patch_size":14,"image_size":1540,"num_channels":3,
            "rope_theta":10000.0,"spatial_merge_size":2
          }
        }"#
    }

    #[test]
    fn auto_detects_mistral_marker() {
        let tmp = tempdir();
        write_config(&tmp, mistral_full());
        let r = resolve_model_family(&tmp, ModelFamily::Auto).expect("ok");
        assert_eq!(r.family, ModelFamily::Mistral35);
        assert_eq!(r.vision_arch, VisionArch::Mistral35 { image_token_id: 10 });
    }

    #[test]
    fn explicit_mistral_rejects_gemma_dir() {
        let tmp = tempdir();
        write_config(
            &tmp,
            r#"{
              "architectures": ["Gemma4ForConditionalGeneration"],
              "model_type": "gemma4"
            }"#,
        );
        let err = resolve_model_family(&tmp, ModelFamily::Mistral35).unwrap_err();
        assert!(matches!(err, FamilyResolveError::Mismatch { .. }));
    }

    #[test]
    fn explicit_gemma_rejects_mistral_dir() {
        let tmp = tempdir();
        write_config(&tmp, mistral_full());
        let err = resolve_model_family(&tmp, ModelFamily::Gemma4).unwrap_err();
        assert!(matches!(err, FamilyResolveError::Mismatch { .. }));
    }

    #[test]
    fn mistral_bad_mscale_all_dim_rejected() {
        let tmp = tempdir();
        let body = mistral_full().replace("\"mscale_all_dim\":0.0", "\"mscale_all_dim\":1.0");
        write_config(&tmp, &body);
        let err = resolve_model_family(&tmp, ModelFamily::Auto).unwrap_err();
        assert!(matches!(err, FamilyResolveError::MistralBadMscaleAllDim { .. }));
    }

    #[test]
    fn auto_falls_through_to_gemma() {
        let tmp = tempdir();
        write_config(
            &tmp,
            r#"{ "architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4" }"#,
        );
        let r = resolve_model_family(&tmp, ModelFamily::Auto).expect("ok");
        assert_eq!(r.family, ModelFamily::Gemma4);
    }

    #[test]
    fn parse_modelfamily_aliases() {
        assert_eq!(ModelFamily::parse("auto").unwrap(), ModelFamily::Auto);
        assert_eq!(ModelFamily::parse("mistral35").unwrap(), ModelFamily::Mistral35);
        assert_eq!(ModelFamily::parse("Mistral-3.5").unwrap(), ModelFamily::Mistral35);
        assert_eq!(ModelFamily::parse("gemma4").unwrap(), ModelFamily::Gemma4);
        assert_eq!(ModelFamily::parse("qwen36").unwrap(), ModelFamily::Qwen36);
        assert!(ModelFamily::parse("llama").is_err());
    }
}
