//! Configuration trait interfaces.
//!
//! These define the contract each subsystem's config must satisfy.
//! Concrete implementations live in downstream crates.

/// Model architecture and weight configuration.
pub trait ModelConfig: Send + Sync {
    /// Human-readable model name.
    fn model_name(&self) -> &str;
    /// Hidden dimension size.
    fn hidden_size(&self) -> usize;
    /// Number of transformer layers.
    fn num_layers(&self) -> usize;
    /// Number of attention heads.
    fn num_attention_heads(&self) -> usize;
    /// Number of key-value heads (for GQA / MQA).
    fn num_kv_heads(&self) -> usize;
    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
    /// Maximum sequence length the model supports.
    fn max_model_len(&self) -> usize;
}

/// KV-cache and block allocation configuration.
pub trait CacheConfig: Send + Sync {
    /// Number of tokens per cache block.
    fn block_size(&self) -> usize;
    /// Fraction of GPU memory reserved for KV cache.
    fn gpu_memory_utilization(&self) -> f32;
    /// Swap space in bytes on CPU.
    fn swap_space_bytes(&self) -> usize;
}

/// Tensor and pipeline parallelism configuration.
pub trait ParallelConfig: Send + Sync {
    /// Tensor parallelism degree.
    fn tensor_parallel_size(&self) -> usize;
    /// Pipeline parallelism degree.
    fn pipeline_parallel_size(&self) -> usize;
}

/// Scheduler policy configuration.
pub trait SchedulerConfig: Send + Sync {
    /// Maximum number of sequences that can be batched together.
    fn max_num_seqs(&self) -> usize;
    /// Maximum number of tokens per iteration.
    fn max_num_batched_tokens(&self) -> usize;
    /// Maximum padding percentage allowed in a batch.
    fn max_paddings(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyModel;
    impl ModelConfig for DummyModel {
        fn model_name(&self) -> &str { "test" }
        fn hidden_size(&self) -> usize { 4096 }
        fn num_layers(&self) -> usize { 32 }
        fn num_attention_heads(&self) -> usize { 32 }
        fn num_kv_heads(&self) -> usize { 8 }
        fn vocab_size(&self) -> usize { 32000 }
        fn max_model_len(&self) -> usize { 4096 }
    }

    struct DummyCache;
    impl CacheConfig for DummyCache {
        fn block_size(&self) -> usize { 16 }
        fn gpu_memory_utilization(&self) -> f32 { 0.9 }
        fn swap_space_bytes(&self) -> usize { 4 * 1024 * 1024 * 1024 }
    }

    struct DummyParallel;
    impl ParallelConfig for DummyParallel {
        fn tensor_parallel_size(&self) -> usize { 1 }
        fn pipeline_parallel_size(&self) -> usize { 1 }
    }

    struct DummySched;
    impl SchedulerConfig for DummySched {
        fn max_num_seqs(&self) -> usize { 256 }
        fn max_num_batched_tokens(&self) -> usize { 8192 }
        fn max_paddings(&self) -> usize { 256 }
    }

    #[test]
    fn model_config_trait() {
        let m = DummyModel;
        assert_eq!(m.model_name(), "test");
        assert_eq!(m.num_kv_heads(), 8);
    }

    #[test]
    fn cache_config_trait() {
        let c = DummyCache;
        assert_eq!(c.block_size(), 16);
        assert!((c.gpu_memory_utilization() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn parallel_config_trait() {
        let p = DummyParallel;
        assert_eq!(p.tensor_parallel_size(), 1);
    }

    #[test]
    fn scheduler_config_trait() {
        let s = DummySched;
        assert_eq!(s.max_num_seqs(), 256);
    }

    #[test]
    fn traits_are_object_safe() {
        // Ensure we can make trait objects.
        fn _model(_: &dyn ModelConfig) {}
        fn _cache(_: &dyn CacheConfig) {}
        fn _par(_: &dyn ParallelConfig) {}
        fn _sched(_: &dyn SchedulerConfig) {}
    }
}
