use std::path::{Path, PathBuf};

use rvllm_core::{ConfigError, Result, RvllmError};
use serde::Serialize;

use crate::{M2GraphAbi, M2GraphShape, M2WeightUploadPlan, M2_NUM_LAYERS};

pub const M2_RUST_DECODE_BENCH_SCHEMA: &str = "rvllm.m2.rust_xla_bench_plan.v1";
pub const M2_DEFAULT_DECODE_BENCH_BATCHES: &[usize] = &[1, 8, 16, 32];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RustDecodeBenchConfig {
    pub model_dir: PathBuf,
    pub batches: Vec<usize>,
    pub ctx: usize,
    pub iters: usize,
    pub warmup: usize,
    pub kv_cache: String,
    pub moe_impl: String,
    pub artifact_dir: PathBuf,
    pub report_path: Option<PathBuf>,
    pub ppl_text_path: Option<PathBuf>,
    pub prompt: String,
    pub gen_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeBenchReport {
    pub schema: &'static str,
    pub bench_kind: &'static str,
    pub runtime: &'static str,
    pub arch: &'static str,
    pub slice: &'static str,
    pub nl: usize,
    pub ctx: usize,
    pub kv_cache: String,
    pub moe_impl: String,
    pub load_seconds: Option<f64>,
    pub batches: Vec<usize>,
    pub artifacts: M2RustDecodeBenchArtifacts,
    pub sweep: Vec<M2RustDecodeSweepItem>,
    pub ppl: M2RustDecodePplPlan,
    pub generation: M2RustDecodeGenerationPlan,
    pub rust_runtime: M2RustDecodeRuntimeReport,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeBenchArtifacts {
    pub model_dir: String,
    pub artifact_dir: String,
    pub report_json: Option<String>,
    pub sweep_json: String,
    pub ppl_json: String,
    pub generation_json: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeSweepItem {
    pub batch: usize,
    pub ctx: usize,
    pub iters: usize,
    pub warmup: usize,
    pub status: &'static str,
    pub error: Option<String>,
    pub artifact_json: String,
    pub decode_mlir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<M2RustDecodeExecutedTiming>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeExecutedTiming {
    timing_source: M2RustDecodeTimingSource,
    ms_min: f64,
    ms_mean: f64,
    ms_max: f64,
    ms_p50: f64,
    tok_per_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeTimingSource {
    pub runtime: &'static str,
    pub executed: bool,
    pub device: String,
    pub executable: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodePplPlan {
    pub status: &'static str,
    pub input_text_path: Option<String>,
    pub artifact_json: String,
    pub result: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeGenerationPlan {
    pub status: &'static str,
    pub prompt: String,
    pub gen_tokens: usize,
    pub artifact_json: String,
    pub result: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeRuntimeReport {
    pub graph_phase: &'static str,
    pub weight_tensors: usize,
    pub weight_upload_bytes: usize,
    pub batch_shapes: Vec<M2RustDecodeRuntimeShape>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeRuntimeShape {
    pub batch: usize,
    pub runtime_inputs: usize,
    pub runtime_outputs: usize,
    pub kv_cache_bytes: usize,
}

pub fn plan_m2_rust_decode_bench(cfg: &M2RustDecodeBenchConfig) -> Result<M2RustDecodeBenchReport> {
    cfg.validate()?;
    let kv_bytes_per_elem = match cfg.kv_cache.as_str() {
        "int8" => 1,
        "bf16" => 2,
        _ => return Err(invalid("kv_cache", "must be int8 or bf16")),
    };
    let batches = cfg.batches.clone();
    let abi = M2GraphAbi::new(M2GraphShape::decode(batches[0], cfg.ctx, kv_bytes_per_elem))?;
    let weights = M2WeightUploadPlan::from_index_dir(&cfg.model_dir, &abi)?;
    let mut sweep = Vec::with_capacity(batches.len());
    let mut batch_shapes = Vec::with_capacity(batches.len());
    for &batch in &batches {
        let batch_abi = M2GraphAbi::new(M2GraphShape::decode(batch, cfg.ctx, kv_bytes_per_elem))?;
        batch_shapes.push(M2RustDecodeRuntimeShape {
            batch,
            runtime_inputs: batch_abi.runtime_inputs.len(),
            runtime_outputs: batch_abi.runtime_outputs.len(),
            kv_cache_bytes: batch_abi.shape.kv_cache_bytes(),
        });
        sweep.push(M2RustDecodeSweepItem {
            batch,
            ctx: cfg.ctx,
            iters: cfg.iters,
            warmup: cfg.warmup,
            status: "planned",
            error: Some("Rust XLA decode timing has not executed".to_string()),
            artifact_json: artifact_file(&cfg.artifact_dir, &format!("m2_decode_b{batch}.json")),
            decode_mlir: artifact_file(&cfg.artifact_dir, &format!("m2_decode_b{batch}.mlir")),
            timing: None,
        });
    }
    Ok(M2RustDecodeBenchReport {
        schema: M2_RUST_DECODE_BENCH_SCHEMA,
        bench_kind: "m2_full_equivalent_decode_plan",
        runtime: "rust+xla",
        arch: "MiniMax-M2.7-NVFP4",
        slice: "v6e-8",
        nl: M2_NUM_LAYERS,
        ctx: cfg.ctx,
        kv_cache: cfg.kv_cache.clone(),
        moe_impl: cfg.moe_impl.clone(),
        load_seconds: None,
        batches,
        artifacts: M2RustDecodeBenchArtifacts {
            model_dir: path_string(&cfg.model_dir),
            artifact_dir: path_string(&cfg.artifact_dir),
            report_json: cfg.report_path.as_ref().map(|p| path_string(p)),
            sweep_json: artifact_file(&cfg.artifact_dir, "m2_decode_sweep.json"),
            ppl_json: artifact_file(&cfg.artifact_dir, "m2_ppl.json"),
            generation_json: artifact_file(&cfg.artifact_dir, "m2_generation.json"),
        },
        sweep,
        ppl: M2RustDecodePplPlan {
            status: "planned",
            input_text_path: cfg.ppl_text_path.as_ref().map(|p| path_string(p)),
            artifact_json: artifact_file(&cfg.artifact_dir, "m2_ppl.json"),
            result: None,
        },
        generation: M2RustDecodeGenerationPlan {
            status: "planned",
            prompt: cfg.prompt.clone(),
            gen_tokens: cfg.gen_tokens,
            artifact_json: artifact_file(&cfg.artifact_dir, "m2_generation.json"),
            result: None,
        },
        rust_runtime: M2RustDecodeRuntimeReport {
            graph_phase: "decode",
            weight_tensors: weights.specs.len(),
            weight_upload_bytes: weights.total_device_bytes(),
            batch_shapes,
        },
    })
}

pub fn check_m2_rust_decode_bench_json(json: &serde_json::Value) -> Result<()> {
    check_tok_per_s(json, "$")
}

pub fn m2_rust_decode_timing(
    timing_source: M2RustDecodeTimingSource,
    batch: usize,
    ms_min: f64,
    ms_mean: f64,
    ms_max: f64,
    ms_p50: f64,
) -> Result<M2RustDecodeExecutedTiming> {
    if timing_source.runtime != "rust_xla" || !timing_source.executed {
        return Err(invalid(
            "timing_source",
            "tok_per_s requires executed rust_xla timing",
        ));
    }
    if batch == 0 {
        return Err(invalid("batch", "must be > 0"));
    }
    for (name, value) in [
        ("ms_min", ms_min),
        ("ms_mean", ms_mean),
        ("ms_max", ms_max),
        ("ms_p50", ms_p50),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid(name, "must be finite and > 0"));
        }
    }
    Ok(M2RustDecodeExecutedTiming {
        timing_source,
        ms_min,
        ms_mean,
        ms_max,
        ms_p50,
        tok_per_s: 1000.0 * batch as f64 / ms_mean,
    })
}

impl M2RustDecodeBenchConfig {
    fn validate(&self) -> Result<()> {
        if self.batches.is_empty() {
            return Err(invalid("batches", "must not be empty"));
        }
        for &batch in &self.batches {
            if batch == 0 {
                return Err(invalid("batches", "must contain only values > 0"));
            }
        }
        if self.ctx == 0 {
            return Err(invalid("ctx", "must be > 0"));
        }
        if self.iters == 0 {
            return Err(invalid("iters", "must be > 0"));
        }
        if self.gen_tokens == 0 {
            return Err(invalid("gen_tokens", "must be > 0"));
        }
        Ok(())
    }
}

fn check_tok_per_s(json: &serde_json::Value, path: &str) -> Result<()> {
    match json {
        serde_json::Value::Object(map) => {
            if map.contains_key("tok_per_s") && !has_executed_rust_xla_source(map) {
                return Err(invalid_owned(
                    "tok_per_s",
                    format!("{path}: tok_per_s requires timing_source.runtime=rust_xla and timing_source.executed=true"),
                ));
            }
            for (key, value) in map {
                let child = format!("{path}.{key}");
                check_tok_per_s(value, &child)?;
            }
        }
        serde_json::Value::Array(values) => {
            for (i, value) in values.iter().enumerate() {
                let child = format!("{path}[{i}]");
                check_tok_per_s(value, &child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn has_executed_rust_xla_source(map: &serde_json::Map<String, serde_json::Value>) -> bool {
    let Some(source) = map.get("timing_source").and_then(|v| v.as_object()) else {
        return false;
    };
    source.get("runtime").and_then(|v| v.as_str()) == Some("rust_xla")
        && source.get("executed").and_then(|v| v.as_bool()) == Some(true)
}

fn artifact_file(dir: &Path, name: &str) -> String {
    path_string(&dir.join(name))
}

fn path_string(path: &Path) -> String {
    path.display().to_string()
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_rust_decode_bench",
    )
}

fn invalid_owned(field: &'static str, reason: String) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason,
        },
        "m2_rust_decode_bench",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tpu/harness/m2_checkpoint_schema")
    }

    fn test_cfg() -> M2RustDecodeBenchConfig {
        M2RustDecodeBenchConfig {
            model_dir: schema_dir(),
            batches: M2_DEFAULT_DECODE_BENCH_BATCHES.to_vec(),
            ctx: 2048,
            iters: 10,
            warmup: 3,
            kv_cache: "int8".to_string(),
            moe_impl: "auto".to_string(),
            artifact_dir: PathBuf::from("tpu/out/m2/rust_xla_plan"),
            report_path: Some(PathBuf::from("tpu/out/m2/rust_xla_plan/full.json")),
            ppl_text_path: Some(PathBuf::from("tpu/harness/ppl.txt")),
            prompt: "Explain angular momentum.".to_string(),
            gen_tokens: 256,
        }
    }

    #[test]
    fn rust_decode_report_has_full_equivalent_plan_schema() {
        let report = plan_m2_rust_decode_bench(&test_cfg()).unwrap();
        let json = serde_json::to_value(&report).unwrap();
        for key in [
            "schema",
            "bench_kind",
            "runtime",
            "arch",
            "slice",
            "nl",
            "ctx",
            "kv_cache",
            "moe_impl",
            "load_seconds",
            "batches",
            "artifacts",
            "sweep",
            "ppl",
            "generation",
            "rust_runtime",
        ] {
            assert!(json.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(json["schema"], M2_RUST_DECODE_BENCH_SCHEMA);
        assert_eq!(json["arch"], "MiniMax-M2.7-NVFP4");
        assert_eq!(json["runtime"], "rust+xla");
        assert_eq!(json["batches"], json!([1, 8, 16, 32]));
        assert_eq!(json["sweep"].as_array().unwrap().len(), 4);
        assert_eq!(json["sweep"][0]["batch"], 1);
        assert_eq!(json["sweep"][1]["batch"], 8);
        assert_eq!(json["sweep"][2]["batch"], 16);
        assert_eq!(json["sweep"][3]["batch"], 32);
        assert_eq!(json["sweep"][0]["status"], "planned");
        assert_eq!(json["load_seconds"], serde_json::Value::Null);
        assert_eq!(json["ppl"]["status"], "planned");
        assert_eq!(json["ppl"]["result"], serde_json::Value::Null);
        assert_eq!(json["generation"]["status"], "planned");
        assert_eq!(json["generation"]["result"], serde_json::Value::Null);
        assert_eq!(
            json["artifacts"]["sweep_json"],
            "tpu/out/m2/rust_xla_plan/m2_decode_sweep.json"
        );
        assert_eq!(json["rust_runtime"]["weight_tensors"], 191_069);
    }

    #[test]
    fn decode_report_tracks_int8_kv_bytes() {
        let report = plan_m2_rust_decode_bench(&test_cfg()).unwrap();
        let b8 = report
            .rust_runtime
            .batch_shapes
            .iter()
            .find(|shape| shape.batch == 8)
            .unwrap();
        assert_eq!(b8.runtime_inputs, 3);
        assert_eq!(b8.runtime_outputs, 3);
        assert_eq!(b8.kv_cache_bytes, 2_080_374_784);
    }

    #[test]
    fn planned_report_does_not_report_tok_per_s() {
        let report = plan_m2_rust_decode_bench(&test_cfg()).unwrap();
        let json = serde_json::to_value(&report).unwrap();
        check_m2_rust_decode_bench_json(&json).unwrap();
        assert!(!serde_json::to_string(&json).unwrap().contains("tok_per_s"));
    }

    #[test]
    fn tok_per_s_requires_executed_rust_xla_source() {
        let fake = json!({
            "sweep": [{
                "batch": 8,
                "tok_per_s": 55.0
            }]
        });
        assert!(check_m2_rust_decode_bench_json(&fake).is_err());

        let timing = m2_rust_decode_timing(
            M2RustDecodeTimingSource {
                runtime: "rust_xla",
                executed: true,
                device: "v6e-8".to_string(),
                executable: "rvllm_m2_decode".to_string(),
            },
            8,
            140.0,
            145.0,
            150.0,
            145.0,
        )
        .unwrap();
        let trusted = serde_json::to_value(&timing).unwrap();
        check_m2_rust_decode_bench_json(&trusted).unwrap();
    }
}
