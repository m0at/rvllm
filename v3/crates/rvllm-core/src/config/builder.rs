//! `RuntimeConfigBuilder` — the only path to construct a `RuntimeConfig`.
//! Accumulates every invalid field into `ConfigError::Inconsistent` so the
//! caller sees all problems at once, not the first one.

use std::path::PathBuf;

use crate::error::{ConfigError, Result, RvllmError};

use super::model::ModelConfig;
use super::runtime::{GraphMode, LogLevel, PreemptionMode, RuntimeConfig};

#[derive(Default)]
pub struct RuntimeConfigBuilder {
    device_id: Option<u32>,
    max_batch: Option<u32>,
    max_context: Option<u32>,
    kv_block_size: Option<u32>,
    num_gpu_blocks: Option<u32>,
    num_cpu_blocks: Option<u32>,
    gpu_memory_utilization: Option<f32>,
    fp8_weights: Option<bool>,
    fp8_kv_cache: Option<bool>,
    graph_capture: Option<GraphMode>,
    preemption: Option<PreemptionMode>,
    log_level: Option<LogLevel>,
    kernel_dir: Option<PathBuf>,
}

macro_rules! setter {
    ($name:ident, $ty:ty) => {
        pub fn $name(mut self, v: $ty) -> Self {
            self.$name = Some(v);
            self
        }
    };
}

impl RuntimeConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    setter!(device_id, u32);
    setter!(max_batch, u32);
    setter!(max_context, u32);
    setter!(kv_block_size, u32);
    setter!(num_gpu_blocks, u32);
    setter!(num_cpu_blocks, u32);
    setter!(gpu_memory_utilization, f32);
    setter!(fp8_weights, bool);
    setter!(fp8_kv_cache, bool);
    setter!(graph_capture, GraphMode);
    setter!(preemption, PreemptionMode);
    setter!(log_level, LogLevel);

    pub fn kernel_dir(mut self, p: PathBuf) -> Self {
        self.kernel_dir = Some(p);
        self
    }

    pub fn build(self, model: &ModelConfig) -> Result<RuntimeConfig> {
        let mut reasons: Vec<String> = Vec::new();
        macro_rules! req {
            ($field:ident) => {
                match self.$field {
                    Some(v) => Some(v),
                    None => {
                        reasons.push(concat!(stringify!($field), " is required").into());
                        None
                    }
                }
            };
        }
        let device_id = req!(device_id);
        let max_batch = req!(max_batch);
        let max_context = req!(max_context);
        let kv_block_size = req!(kv_block_size);
        let num_gpu_blocks = req!(num_gpu_blocks);
        let num_cpu_blocks = req!(num_cpu_blocks);
        let gpu_memory_utilization = req!(gpu_memory_utilization);
        let fp8_weights = req!(fp8_weights);
        let fp8_kv_cache = req!(fp8_kv_cache);
        let graph_capture = req!(graph_capture);
        let preemption = req!(preemption);

        if let Some(v) = kv_block_size {
            if ![16u32, 32, 64].contains(&v) {
                reasons.push(format!("kv_block_size must be 16|32|64, got {v}"));
            }
        }
        if let Some(v) = max_batch {
            if !(1..=256).contains(&v) {
                reasons.push(format!("max_batch must be in 1..=256, got {v}"));
            }
        }
        if let Some(ctx) = max_context {
            if ctx as usize > model.max_position_embeddings {
                reasons.push(format!(
                    "max_context {ctx} > model.max_position_embeddings {}",
                    model.max_position_embeddings
                ));
            }
        }
        if let Some(u) = gpu_memory_utilization {
            if !(u > 0.0 && u <= 0.95) {
                reasons.push(format!(
                    "gpu_memory_utilization must be in (0.0, 0.95], got {u}"
                ));
            }
        }
        if let (Some(true), Some(bs)) = (fp8_kv_cache, kv_block_size) {
            if bs < 32 {
                reasons.push("fp8_kv_cache requires kv_block_size >= 32".into());
            }
        }

        if !reasons.is_empty() {
            return Err(RvllmError::config(
                ConfigError::Inconsistent { reasons },
                "RuntimeConfig::build",
            ));
        }

        // All required present and validated — safe to unwrap.
        #[allow(clippy::unwrap_used)]
        Ok(RuntimeConfig {
            device_id: device_id.unwrap(),
            max_batch: max_batch.unwrap(),
            max_context: max_context.unwrap(),
            kv_block_size: kv_block_size.unwrap(),
            num_gpu_blocks: num_gpu_blocks.unwrap(),
            num_cpu_blocks: num_cpu_blocks.unwrap(),
            gpu_memory_utilization: gpu_memory_utilization.unwrap(),
            fp8_weights: fp8_weights.unwrap(),
            fp8_kv_cache: fp8_kv_cache.unwrap(),
            graph_capture: graph_capture.unwrap(),
            preemption: preemption.unwrap(),
            log_level: self.log_level.unwrap_or(LogLevel::Info),
            kernel_dir: self.kernel_dir,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{ModelArch, ModelConfig};
    use crate::dtype::DType;

    fn qwen() -> ModelConfig {
        ModelConfig {
            architecture: ModelArch::Qwen2,
            hidden_size: 3584,
            num_layers: 28,
            num_attention_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            intermediate_size: 18944,
            vocab_size: 152064,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
            torch_dtype: DType::Bf16,
        }
    }

    #[test]
    fn rejects_missing_fields() {
        let err = RuntimeConfigBuilder::new()
            .device_id(0)
            .max_batch(128)
            .build(&qwen())
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("max_context is required"));
        assert!(s.contains("kv_block_size is required"));
    }

    #[test]
    fn rejects_bad_block_size() {
        let err = RuntimeConfigBuilder::new()
            .device_id(0)
            .max_batch(128)
            .max_context(2048)
            .kv_block_size(48)
            .num_gpu_blocks(1024)
            .num_cpu_blocks(0)
            .gpu_memory_utilization(0.9)
            .fp8_weights(true)
            .fp8_kv_cache(false)
            .graph_capture(GraphMode::Off)
            .preemption(PreemptionMode::Recompute)
            .build(&qwen())
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("kv_block_size must be 16|32|64"));
    }

    #[test]
    fn happy_path() {
        let rt = RuntimeConfigBuilder::new()
            .device_id(0)
            .max_batch(128)
            .max_context(2048)
            .kv_block_size(64)
            .num_gpu_blocks(1024)
            .num_cpu_blocks(0)
            .gpu_memory_utilization(0.9)
            .fp8_weights(true)
            .fp8_kv_cache(false)
            .graph_capture(GraphMode::Buckets(vec![1, 2, 4, 8, 16, 32, 64, 128]))
            .preemption(PreemptionMode::Recompute)
            .build(&qwen())
            .unwrap();
        assert_eq!(rt.max_batch(), 128);
        assert_eq!(rt.kv_block_size(), 64);
    }
}
