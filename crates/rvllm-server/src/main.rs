//! rvllm: High-performance LLM inference server in Rust
//!
//! Usage: rvllm serve --model <model_path> [options]
//!
//! Compatible with OpenAI API at http://localhost:8000/v1/

use clap::{Parser, Subcommand};
use tracing::info;

#[derive(Parser)]
#[command(name = "rvllm", about = "High-performance LLM inference server")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
    Serve {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long, default_value = "auto")]
        dtype: String,
        #[arg(long, default_value_t = 2048)]
        max_model_len: usize,
        #[arg(long, default_value_t = 0.90)]
        gpu_memory_utilization: f32,
        #[arg(long, default_value_t = 1)]
        tensor_parallel_size: usize,
        #[arg(long, default_value_t = 256)]
        max_num_seqs: usize,
        #[arg(long)]
        tokenizer: Option<String>,
        #[arg(long, default_value = "info")]
        log_level: String,
        #[arg(long)]
        disable_telemetry: bool,
    },
    /// Show system info (GPU, memory, etc.)
    Info,
    /// Run benchmarks
    Benchmark {
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 100)]
        num_prompts: usize,
        #[arg(long, default_value_t = 128)]
        input_len: usize,
        #[arg(long, default_value_t = 128)]
        output_len: usize,
    },
}

fn init_tracing(log_level: &str) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(log_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();
}

fn detect_gpu_and_log() {
    let devices = rvllm_gpu::prelude::list_devices();
    if devices.is_empty() {
        info!("no GPU devices detected, using mock backend");
    } else {
        for dev in &devices {
            info!(
                id = dev.id,
                name = %dev.name,
                compute = %format!("{}.{}", dev.compute_capability.0, dev.compute_capability.1),
                memory_gb = dev.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                "detected GPU device"
            );
        }
    }
}

/// Detect the appropriate device string based on compiled features and hardware.
fn detect_device_string() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        let devices = rvllm_gpu::prelude::list_devices();
        if !devices.is_empty() {
            return "cuda";
        }
        info!("cuda feature enabled but no CUDA devices found, falling back to cpu");
        "cpu"
    }
    #[cfg(not(feature = "cuda"))]
    {
        "cpu"
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Serve {
            model,
            host,
            port,
            dtype,
            max_model_len,
            gpu_memory_utilization,
            tensor_parallel_size,
            max_num_seqs,
            tokenizer,
            log_level,
            disable_telemetry,
        } => {
            init_tracing(&log_level);
            info!("rvllm v0.1.0");

            detect_gpu_and_log();

            // Build EngineConfig from CLI args
            let config = {
                use rvllm_config::*;
                EngineConfig::builder()
                    .model({
                        let mut m = ModelConfigImpl::builder()
                            .model_path(&model)
                            .dtype(&dtype)
                            .max_model_len(max_model_len);
                        if let Some(ref tok) = tokenizer {
                            m = m.tokenizer_path(tok);
                        }
                        m.build()
                    })
                    .cache(
                        CacheConfigImpl::builder()
                            .gpu_memory_utilization(gpu_memory_utilization)
                            .build(),
                    )
                    .scheduler(
                        SchedulerConfigImpl::builder()
                            .max_num_seqs(max_num_seqs)
                            .build(),
                    )
                    .parallel(
                        ParallelConfigImpl::builder()
                            .tensor_parallel_size(tensor_parallel_size)
                            .build(),
                    )
                    .device(
                        DeviceConfig::builder()
                            .device(detect_device_string())
                            .build(),
                    )
                    .telemetry(
                        TelemetryConfig::builder()
                            .enabled(!disable_telemetry)
                            .log_level(&log_level)
                            .build(),
                    )
                    .build()
            };

            info!(
                model = %model,
                host = %host,
                port = port,
                dtype = %dtype,
                max_model_len = max_model_len,
                gpu_memory_utilization = gpu_memory_utilization,
                tp_size = tensor_parallel_size,
                "starting server"
            );

            // Pass host/port to the server via env vars so rvllm_api::serve
            // can pick them up without changing its public signature.
            std::env::set_var("VLLM_HOST", &host);
            std::env::set_var("VLLM_PORT", port.to_string());

            rvllm_api::serve(config).await?;
        }
        Commands::Info => {
            init_tracing("info");
            info!("rvllm system info");

            detect_gpu_and_log();

            info!(platform = %std::env::consts::OS, arch = %std::env::consts::ARCH, "system");
        }
        Commands::Benchmark {
            model,
            num_prompts,
            input_len,
            output_len,
        } => {
            init_tracing("info");
            info!(
                model = %model,
                num_prompts = num_prompts,
                input_len = input_len,
                output_len = output_len,
                "running benchmark"
            );
        }
    }
    Ok(())
}
