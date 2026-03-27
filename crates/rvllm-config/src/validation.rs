//! Validation logic for [`EngineConfig`].

use crate::engine::EngineConfig;

/// Validate an [`EngineConfig`], returning an error string on the first
/// contradiction found.
///
/// Checks include:
/// - `model_path` is non-empty
/// - `gpu_memory_utilization` is in (0.0, 1.0]
/// - `block_size` is a power of two
/// - `tensor_parallel_size` and `pipeline_parallel_size` are >= 1
/// - `max_num_batched_tokens` >= `max_num_seqs`
/// - CPU device cannot have `tensor_parallel_size > 1`
/// - `swap_space_gb` is non-negative
/// - `max_model_len` > 0
pub fn validate(config: &EngineConfig) -> Result<(), String> {
    if config.model.model_path.is_empty() {
        return Err("model_path must not be empty".into());
    }

    if config.model.max_model_len == 0 {
        return Err("max_model_len must be > 0".into());
    }

    let u = config.cache.gpu_memory_utilization;
    if u <= 0.0 || u > 1.0 {
        return Err(format!(
            "gpu_memory_utilization must be in (0.0, 1.0], got {u}"
        ));
    }

    if !config.cache.block_size.is_power_of_two() {
        return Err(format!(
            "block_size must be a power of two, got {}",
            config.cache.block_size
        ));
    }

    if config.cache.swap_space_gb < 0.0 {
        return Err(format!(
            "swap_space_gb must be >= 0, got {}",
            config.cache.swap_space_gb
        ));
    }

    if config.parallel.tensor_parallel_size == 0 {
        return Err("tensor_parallel_size must be >= 1".into());
    }

    if config.parallel.pipeline_parallel_size == 0 {
        return Err("pipeline_parallel_size must be >= 1".into());
    }

    if config.device.device == "cpu" && config.parallel.tensor_parallel_size > 1 {
        return Err(format!(
            "tensor_parallel_size={} not supported on CPU device",
            config.parallel.tensor_parallel_size
        ));
    }

    if config.scheduler.max_num_batched_tokens < config.scheduler.max_num_seqs {
        return Err(format!(
            "max_num_batched_tokens ({}) must be >= max_num_seqs ({})",
            config.scheduler.max_num_batched_tokens, config.scheduler.max_num_seqs
        ));
    }

    if config.scheduler.max_num_seqs == 0 {
        return Err("max_num_seqs must be > 0".into());
    }

    tracing::debug!("engine config validation passed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheConfigImpl;
    use crate::device::DeviceConfig;
    use crate::model::ModelConfigImpl;
    use crate::parallel::ParallelConfigImpl;

    fn valid_config() -> EngineConfig {
        EngineConfig {
            model: ModelConfigImpl {
                model_path: "meta-llama/Llama-2-7b".into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn valid_default_passes() {
        let cfg = valid_config();
        assert!(validate(&cfg).is_ok());
    }

    #[test]
    fn empty_model_path_rejected() {
        let cfg = EngineConfig::default();
        let err = validate(&cfg).unwrap_err();
        assert!(err.contains("model_path"), "got: {err}");
    }

    #[test]
    fn bad_gpu_mem_utilization() {
        let mut cfg = valid_config();
        cfg.cache.gpu_memory_utilization = 0.0;
        assert!(validate(&cfg).unwrap_err().contains("gpu_memory_utilization"));

        cfg.cache.gpu_memory_utilization = 1.5;
        assert!(validate(&cfg).unwrap_err().contains("gpu_memory_utilization"));
    }

    #[test]
    fn non_power_of_two_block_size() {
        let mut cfg = valid_config();
        cfg.cache.block_size = 15;
        assert!(validate(&cfg).unwrap_err().contains("block_size"));
    }

    #[test]
    fn tp_on_cpu_rejected() {
        let mut cfg = valid_config();
        cfg.device.device = "cpu".into();
        cfg.parallel.tensor_parallel_size = 2;
        assert!(validate(&cfg).unwrap_err().contains("tensor_parallel_size"));
    }

    #[test]
    fn batched_tokens_less_than_seqs() {
        let mut cfg = valid_config();
        cfg.scheduler.max_num_seqs = 512;
        cfg.scheduler.max_num_batched_tokens = 256;
        assert!(validate(&cfg)
            .unwrap_err()
            .contains("max_num_batched_tokens"));
    }

    #[test]
    fn zero_tp_rejected() {
        let mut cfg = valid_config();
        cfg.parallel.tensor_parallel_size = 0;
        assert!(validate(&cfg).unwrap_err().contains("tensor_parallel_size"));
    }

    #[test]
    fn zero_pp_rejected() {
        let mut cfg = valid_config();
        cfg.parallel.pipeline_parallel_size = 0;
        assert!(validate(&cfg)
            .unwrap_err()
            .contains("pipeline_parallel_size"));
    }

    #[test]
    fn zero_max_model_len_rejected() {
        let mut cfg = valid_config();
        cfg.model.max_model_len = 0;
        assert!(validate(&cfg).unwrap_err().contains("max_model_len"));
    }

    #[test]
    fn negative_swap_space_rejected() {
        let mut cfg = valid_config();
        cfg.cache.swap_space_gb = -1.0;
        assert!(validate(&cfg).unwrap_err().contains("swap_space_gb"));
    }

    #[test]
    fn zero_max_num_seqs_rejected() {
        let mut cfg = valid_config();
        cfg.scheduler.max_num_seqs = 0;
        assert!(validate(&cfg).unwrap_err().contains("max_num_seqs"));
    }

    #[test]
    fn builder_round_trip() {
        let cfg = EngineConfig::builder()
            .model(
                ModelConfigImpl::builder()
                    .model_path("test/model")
                    .max_model_len(4096)
                    .build(),
            )
            .cache(
                CacheConfigImpl::builder()
                    .block_size(32)
                    .gpu_memory_utilization(0.85)
                    .build(),
            )
            .parallel(
                ParallelConfigImpl::builder()
                    .tensor_parallel_size(2)
                    .build(),
            )
            .device(DeviceConfig::builder().device("cuda").build())
            .build();

        assert!(validate(&cfg).is_ok());
        assert_eq!(cfg.model.max_model_len, 4096);
        assert_eq!(cfg.cache.block_size, 32);
        assert_eq!(cfg.parallel.tensor_parallel_size, 2);
    }
}
