//! CLI argument definitions via `clap`.
//!
//! Environment variable overrides are handled in [`crate::load_config`] via
//! the standard `VLLM_*` prefix convention, not through clap's `env` feature.

use clap::Parser;
use rvllm_core::types::Dtype;

use crate::DEFAULT_MAX_MODEL_LEN;

/// Command-line arguments for the vLLM engine.
#[derive(Debug, Clone, Parser)]
#[command(name = "vllm", about = "vLLM inference engine")]
pub struct CliArgs {
    // -- Model --
    /// Path to model weights (HuggingFace repo id or local path).
    #[arg(long)]
    pub model: String,

    /// Path to tokenizer (defaults to model path).
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// Data type for model weights: auto, float32, float16, bfloat16.
    /// "auto" selects float16 on SM >= 7.0 GPUs, float32 otherwise.
    #[arg(long, default_value = "auto")]
    pub dtype: Dtype,

    /// Maximum model context length.
    #[arg(long, default_value_t = DEFAULT_MAX_MODEL_LEN)]
    pub max_model_len: usize,

    /// Trust remote code when loading model.
    #[arg(long, default_value_t = false)]
    pub trust_remote_code: bool,

    // -- Cache --
    /// Tokens per KV-cache block.
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Fraction of GPU memory for KV cache.
    #[arg(long, default_value_t = 0.90)]
    pub gpu_memory_utilization: f32,

    /// VRAM in GiB to leave free for graph/cublas/scratch allocations.
    #[arg(long, default_value_t = 0.0)]
    pub gpu_memory_reserve_gb: f32,

    /// CPU swap space in GiB.
    #[arg(long, default_value_t = 4.0)]
    pub swap_space_gb: f32,

    /// Fixed number of GPU blocks (overrides auto-compute).
    #[arg(long)]
    pub num_gpu_blocks: Option<usize>,

    /// Fixed number of CPU blocks (overrides auto-compute).
    #[arg(long)]
    pub num_cpu_blocks: Option<usize>,

    /// Enable prefix caching to reuse KV blocks for shared prompt prefixes.
    #[arg(long, default_value_t = false)]
    pub enable_prefix_caching: bool,

    /// KV cache data type: "auto" (fp16), "fp8", "fp8_e4m3".
    #[arg(long, default_value = "auto")]
    pub kv_cache_dtype: String,

    // -- Scheduler --
    /// Max concurrent sequences.
    #[arg(long, default_value_t = 256)]
    pub max_num_seqs: usize,

    /// Max tokens per batch.
    #[arg(long, default_value_t = DEFAULT_MAX_MODEL_LEN)]
    pub max_num_batched_tokens: usize,

    /// Max prompt tokens processed per prefill step. 0 disables chunking.
    #[arg(long, default_value_t = 128)]
    pub max_prefill_chunk: usize,

    /// Max padding tokens in a batch.
    #[arg(long, default_value_t = 256)]
    pub max_paddings: usize,

    /// Preemption mode: "swap" or "recompute".
    #[arg(long, default_value = "recompute")]
    pub preemption_mode: String,

    // -- Parallel --
    /// Tensor parallel size.
    #[arg(long, default_value_t = 1)]
    pub tensor_parallel_size: usize,

    /// Pipeline parallel size.
    #[arg(long, default_value_t = 1)]
    pub pipeline_parallel_size: usize,

    // -- Device --
    /// Target device: "cuda", "cpu", "metal".
    #[arg(long, default_value = "cuda")]
    pub device: String,

    // -- Telemetry --
    /// Disable telemetry.
    #[arg(long, default_value_t = false)]
    pub disable_telemetry: bool,

    /// Prometheus metrics port.
    #[arg(long)]
    pub prometheus_port: Option<u16>,

    /// OTLP gRPC endpoint.
    #[arg(long)]
    pub otlp_endpoint: Option<String>,

    /// Log level.
    #[arg(long, default_value = "info")]
    pub log_level: String,

    // -- Config file --
    /// Path to a TOML config file (values override defaults, CLI overrides file).
    #[arg(long)]
    pub config_file: Option<String>,
}
