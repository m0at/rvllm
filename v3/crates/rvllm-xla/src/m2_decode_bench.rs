use std::path::PathBuf;

use rvllm_core::{ConfigError, Result, RvllmError};
use serde::Serialize;

use crate::{M2GraphAbi, M2GraphShape, M2WeightUploadPlan, M2_NUM_LAYERS};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RustDecodeBenchConfig {
    pub model_dir: PathBuf,
    pub batch: usize,
    pub ctx: usize,
    pub iters: usize,
    pub warmup: usize,
    pub kv_cache: String,
    pub moe_impl: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeBenchReport {
    pub arch: &'static str,
    pub slice: &'static str,
    pub nl: usize,
    pub ctx: usize,
    pub kv_cache: String,
    pub moe_impl: String,
    pub load_seconds: f64,
    pub sweep: Vec<M2RustDecodeSweepItem>,
    pub ppl: Option<serde_json::Value>,
    pub generation: Option<serde_json::Value>,
    pub rust_runtime: M2RustDecodeRuntimeReport,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeSweepItem {
    pub batch: usize,
    pub ctx: usize,
    pub iters: usize,
    pub warmup: usize,
    pub status: &'static str,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct M2RustDecodeRuntimeReport {
    pub graph_phase: &'static str,
    pub runtime_inputs: usize,
    pub runtime_outputs: usize,
    pub weight_tensors: usize,
    pub weight_upload_bytes: usize,
    pub kv_cache_bytes: usize,
}

pub fn plan_m2_rust_decode_bench(cfg: &M2RustDecodeBenchConfig) -> Result<M2RustDecodeBenchReport> {
    cfg.validate()?;
    let kv_bytes_per_elem = match cfg.kv_cache.as_str() {
        "int8" => 1,
        "bf16" => 2,
        _ => return Err(invalid("kv_cache", "must be int8 or bf16")),
    };
    let abi = M2GraphAbi::new(M2GraphShape::decode(cfg.batch, cfg.ctx, kv_bytes_per_elem))?;
    let weights = M2WeightUploadPlan::from_index_dir(&cfg.model_dir, &abi)?;
    Ok(M2RustDecodeBenchReport {
        arch: "MiniMax-M2.7-NVFP4",
        slice: "v6e-8",
        nl: M2_NUM_LAYERS,
        ctx: cfg.ctx,
        kv_cache: cfg.kv_cache.clone(),
        moe_impl: cfg.moe_impl.clone(),
        load_seconds: 0.0,
        sweep: vec![M2RustDecodeSweepItem {
            batch: cfg.batch,
            ctx: cfg.ctx,
            iters: cfg.iters,
            warmup: cfg.warmup,
            status: "planned",
            error: Some("Rust PJRT graph execution is not wired yet".to_string()),
        }],
        ppl: None,
        generation: None,
        rust_runtime: M2RustDecodeRuntimeReport {
            graph_phase: "decode",
            runtime_inputs: abi.runtime_inputs.len(),
            runtime_outputs: abi.runtime_outputs.len(),
            weight_tensors: weights.specs.len(),
            weight_upload_bytes: weights.total_device_bytes(),
            kv_cache_bytes: abi.shape.kv_cache_bytes(),
        },
    })
}

impl M2RustDecodeBenchConfig {
    fn validate(&self) -> Result<()> {
        if self.batch == 0 {
            return Err(invalid("batch", "must be > 0"));
        }
        if self.ctx == 0 {
            return Err(invalid("ctx", "must be > 0"));
        }
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tpu/harness/m2_checkpoint_schema")
    }

    #[test]
    fn rust_decode_report_matches_python_full_bench_top_level_shape() {
        let report = plan_m2_rust_decode_bench(&M2RustDecodeBenchConfig {
            model_dir: schema_dir(),
            batch: 8,
            ctx: 2048,
            iters: 10,
            warmup: 3,
            kv_cache: "int8".to_string(),
            moe_impl: "auto".to_string(),
        })
        .unwrap();
        let json = serde_json::to_value(&report).unwrap();
        for key in [
            "arch",
            "slice",
            "nl",
            "ctx",
            "kv_cache",
            "moe_impl",
            "load_seconds",
            "sweep",
            "ppl",
            "generation",
        ] {
            assert!(json.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(json["arch"], "MiniMax-M2.7-NVFP4");
        assert_eq!(json["sweep"][0]["batch"], 8);
        assert_eq!(json["sweep"][0]["status"], "planned");
        assert_eq!(json["rust_runtime"]["weight_tensors"], 191_069);
    }

    #[test]
    fn decode_report_tracks_int8_kv_bytes() {
        let report = plan_m2_rust_decode_bench(&M2RustDecodeBenchConfig {
            model_dir: schema_dir(),
            batch: 8,
            ctx: 2048,
            iters: 10,
            warmup: 3,
            kv_cache: "int8".to_string(),
            moe_impl: "auto".to_string(),
        })
        .unwrap();
        assert_eq!(report.rust_runtime.runtime_inputs, 3);
        assert_eq!(report.rust_runtime.runtime_outputs, 3);
        assert_eq!(report.rust_runtime.kv_cache_bytes, 2_080_374_784);
    }
}
