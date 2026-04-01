#![forbid(unsafe_code)]
//! Configuration loading, validation, and CLI argument parsing for vllm-rs.
//!
//! This crate provides:
//! - Strongly-typed config structs for every engine subsystem
//! - TOML file and environment variable deserialization
//! - CLI argument parsing via `clap`
//! - Validation that catches contradictions early
//! - Builder pattern for ergonomic test construction

pub mod cache;
pub mod cli;
pub mod device;
pub mod engine;
pub mod model;
pub mod parallel;
pub mod scheduler;
pub mod telemetry;
pub mod validation;

pub use cache::CacheConfigImpl;
pub use cli::CliArgs;
pub use device::DeviceConfig;
pub use engine::EngineConfig;
pub use model::{
    resolve_runtime_max_model_len, ModelConfigImpl, AUTO_LLAMA_MAX_MODEL_LEN_CAP,
    DEFAULT_MAX_MODEL_LEN,
};
pub use parallel::ParallelConfigImpl;
pub use scheduler::{PreemptionMode, SchedulerConfigImpl};
pub use telemetry::TelemetryConfig;
pub use validation::validate;

use thiserror::Error;

/// Configuration-specific error type.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Validation failed.
    #[error("validation error: {0}")]
    Validation(String),
    /// Failed to read or parse a config file.
    #[error("file error: {0}")]
    FileError(String),
    /// TOML deserialization failed.
    #[error("toml parse error: {0}")]
    TomlError(#[from] toml::de::Error),
    /// I/O error.
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Convenience result alias for this crate.
pub type Result<T> = std::result::Result<T, ConfigError>;

fn config_file_sets_max_model_len(contents: &str) -> bool {
    toml::from_str::<toml::Value>(contents)
        .ok()
        .and_then(|value| value.get("model").and_then(|model| model.get("max_model_len")).cloned())
        .is_some()
}

/// Load a fully-validated [`EngineConfig`] from CLI args.
///
/// Resolution order (later wins):
/// 1. Struct defaults
/// 2. TOML config file (if `--config-file` provided)
/// 3. CLI arguments / environment variables
pub fn load_config(args: &CliArgs) -> Result<EngineConfig> {
    // Start from defaults
    let mut config = if let Some(ref path) = args.config_file {
        tracing::info!(path = %path, "loading config from file");
        let contents = std::fs::read_to_string(path)?;
        let mut config = toml::from_str::<EngineConfig>(&contents)?;
        if config_file_sets_max_model_len(&contents) {
            config.model.max_model_len_explicit = true;
        }
        config
    } else {
        EngineConfig::default()
    };

    // CLI / env overrides on top
    apply_cli_overrides(&mut config, args);

    // Validate the merged config
    validate(&config).map_err(ConfigError::Validation)?;

    tracing::info!(model = %config.model.model_path, "config loaded successfully");
    Ok(config)
}

/// Apply CLI argument values on top of an existing config.
fn apply_cli_overrides(config: &mut EngineConfig, args: &CliArgs) {
    // Model
    config.model.model_path = args.model.clone();
    if let Some(ref tok) = args.tokenizer {
        config.model.tokenizer_path = Some(tok.clone());
    }
    config.model.dtype = args.dtype;
    if let Some(max_model_len) = args.max_model_len {
        config.model.max_model_len = max_model_len;
        config.model.max_model_len_explicit = true;
    }
    config.model.trust_remote_code = args.trust_remote_code;

    // Cache
    config.cache.block_size = args.block_size;
    config.cache.gpu_memory_utilization = args.gpu_memory_utilization;
    config.cache.gpu_memory_reserve_gb = args.gpu_memory_reserve_gb;
    config.cache.swap_space_gb = args.swap_space_gb;
    config.cache.num_gpu_blocks = args.num_gpu_blocks;
    config.cache.num_cpu_blocks = args.num_cpu_blocks;
    config.cache.enable_prefix_caching = args.enable_prefix_caching;
    config.cache.kv_cache_dtype = args.kv_cache_dtype.clone();

    // Scheduler
    config.scheduler.max_num_seqs = args.max_num_seqs;
    config.scheduler.max_num_batched_tokens = args.max_num_batched_tokens;
    config.scheduler.max_prefill_chunk = args.max_prefill_chunk;
    config.scheduler.max_paddings = args.max_paddings;
    if let Ok(mode) = args.preemption_mode.parse::<PreemptionMode>() {
        config.scheduler.preemption_mode = mode;
    }

    // Parallel
    config.parallel.tensor_parallel_size = args.tensor_parallel_size;
    config.parallel.pipeline_parallel_size = args.pipeline_parallel_size;

    // Device
    config.device.device = args.device.clone();

    // Telemetry
    config.telemetry.enabled = !args.disable_telemetry;
    config.telemetry.prometheus_port = args.prometheus_port;
    config.telemetry.otlp_endpoint = args.otlp_endpoint.clone();
    config.telemetry.log_level = args.log_level.clone();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_from_cli_args() {
        use rvllm_core::types::Dtype;
        let args = CliArgs {
            model: "test/model".into(),
            tokenizer: None,
            dtype: Dtype::Float16,
            max_model_len: Some(4096),
            trust_remote_code: false,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            gpu_memory_reserve_gb: 0.0,
            swap_space_gb: 4.0,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
            enable_prefix_caching: false,
            kv_cache_dtype: "auto".into(),
            max_num_seqs: 256,
            max_num_batched_tokens: 4096,
            max_prefill_chunk: 128,
            max_paddings: 256,
            preemption_mode: "recompute".into(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            device: "cuda".into(),
            disable_telemetry: false,
            prometheus_port: Some(9090),
            otlp_endpoint: None,
            log_level: "debug".into(),
            config_file: None,
        };

        let cfg = load_config(&args).unwrap();
        assert_eq!(cfg.model.model_path, "test/model");
        assert_eq!(cfg.model.dtype, Dtype::Float16);
        assert_eq!(cfg.model.max_model_len, 4096);
        assert!(cfg.model.max_model_len_explicit);
        assert_eq!(cfg.telemetry.prometheus_port, Some(9090));
        assert_eq!(cfg.telemetry.log_level, "debug");
    }

    #[test]
    fn load_from_toml_file() {
        use rvllm_core::types::Dtype;
        let toml_content = r#"
[model]
model_path = "bigscience/bloom-560m"
dtype = "bfloat16"
max_model_len = 1024
trust_remote_code = false

[cache]
block_size = 8
gpu_memory_utilization = 0.8
swap_space_gb = 2.0

[scheduler]
max_num_seqs = 128
max_num_batched_tokens = 1024
max_prefill_chunk = 128
max_paddings = 64
preemption_mode = "Swap"

[parallel]
tensor_parallel_size = 1
pipeline_parallel_size = 1

[device]
device = "cuda"

[telemetry]
enabled = true
log_level = "warn"
"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, toml_content).unwrap();

        let args = CliArgs {
            model: "bigscience/bloom-560m".into(),
            tokenizer: None,
            dtype: Dtype::BFloat16,
            max_model_len: Some(1024),
            trust_remote_code: false,
            block_size: 8,
            gpu_memory_utilization: 0.8,
            gpu_memory_reserve_gb: 0.0,
            swap_space_gb: 2.0,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
            enable_prefix_caching: false,
            kv_cache_dtype: "auto".into(),
            max_num_seqs: 128,
            max_num_batched_tokens: 1024,
            max_prefill_chunk: 128,
            max_paddings: 64,
            preemption_mode: "swap".into(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            device: "cuda".into(),
            disable_telemetry: false,
            prometheus_port: None,
            otlp_endpoint: None,
            log_level: "warn".into(),
            config_file: Some(path.to_string_lossy().into()),
        };

        let cfg = load_config(&args).unwrap();
        assert_eq!(cfg.model.model_path, "bigscience/bloom-560m");
        assert!(cfg.model.max_model_len_explicit);
        assert_eq!(cfg.cache.block_size, 8);
        assert_eq!(cfg.scheduler.preemption_mode, PreemptionMode::Swap);
    }

    #[test]
    fn default_cli_max_model_len_is_not_marked_explicit() {
        use rvllm_core::types::Dtype;
        let args = CliArgs {
            model: "test/model".into(),
            tokenizer: None,
            dtype: Dtype::Float16,
            max_model_len: None,
            trust_remote_code: false,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            swap_space_gb: 4.0,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
            enable_prefix_caching: false,
            kv_cache_dtype: "auto".into(),
            max_num_seqs: 256,
            max_num_batched_tokens: 4096,
            max_paddings: 256,
            preemption_mode: "recompute".into(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            device: "cuda".into(),
            disable_telemetry: false,
            prometheus_port: None,
            otlp_endpoint: None,
            log_level: "info".into(),
            config_file: None,
        };

        let cfg = load_config(&args).unwrap();
        assert_eq!(cfg.model.max_model_len, DEFAULT_MAX_MODEL_LEN);
        assert!(!cfg.model.max_model_len_explicit);
    }

    #[test]
    fn load_rejects_invalid() {
        use rvllm_core::types::Dtype;
        let args = CliArgs {
            model: "".into(), // empty = invalid
            tokenizer: None,
            dtype: Dtype::Auto,
            max_model_len: Some(2048),
            trust_remote_code: false,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            gpu_memory_reserve_gb: 0.0,
            swap_space_gb: 4.0,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
            enable_prefix_caching: false,
            kv_cache_dtype: "auto".into(),
            max_num_seqs: 256,
            max_num_batched_tokens: 2048,
            max_prefill_chunk: 128,
            max_paddings: 256,
            preemption_mode: "recompute".into(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            device: "cuda".into(),
            disable_telemetry: false,
            prometheus_port: None,
            otlp_endpoint: None,
            log_level: "info".into(),
            config_file: None,
        };

        assert!(load_config(&args).is_err());
    }

    #[test]
    fn serde_round_trip() {
        use rvllm_core::types::Dtype;
        let cfg = EngineConfig::builder()
            .model(
                ModelConfigImpl::builder()
                    .model_path("test/model")
                    .dtype(Dtype::Float16)
                    .max_model_len(8192)
                    .build(),
            )
            .build();

        let json = serde_json::to_string(&cfg).unwrap();
        let back: EngineConfig = serde_json::from_str(&json).unwrap();
        assert!(!back.model.max_model_len_explicit);

        let mut expected = cfg;
        expected.model.max_model_len_explicit = false;
        assert_eq!(expected, back);
    }

    #[test]
    fn toml_round_trip() {
        let cfg = EngineConfig::builder()
            .model(ModelConfigImpl::builder().model_path("test/model").build())
            .cache(CacheConfigImpl::builder().block_size(32).build())
            .build();

        let toml_str = toml::to_string_pretty(&cfg).unwrap();
        let back: EngineConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(cfg, back);
    }
}
