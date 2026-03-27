//! Executor configuration.

/// Configuration for constructing an executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Number of GPUs to use (1 = single-GPU path).
    pub num_gpus: usize,
    /// Model name or path.
    pub model_name: String,
    /// Block size for KV-cache.
    pub block_size: usize,
    /// Fraction of GPU memory for cache.
    pub gpu_memory_utilization: f32,
    /// Tensor parallelism degree.
    pub tensor_parallel_size: usize,
    /// Pipeline parallelism degree.
    pub pipeline_parallel_size: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            num_gpus: 1,
            model_name: String::new(),
            block_size: 16,
            gpu_memory_utilization: 0.9,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
        }
    }
}
