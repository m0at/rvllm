//! rvllm: High-performance LLM inference server in Rust
//!
//! Usage: rvllm serve --model <model_path> [options]
//!
//! Compatible with OpenAI API at http://localhost:8000/v1/

use clap::{Parser, Subcommand};
use rvllm_core::types::Dtype;
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
        dtype: Dtype,
        #[arg(long)]
        max_model_len: Option<usize>,
        #[arg(long, default_value_t = 0.90)]
        gpu_memory_utilization: f32,
        #[arg(long, default_value_t = 0.0)]
        gpu_memory_reserve_gb: f32,
        #[arg(long)]
        num_gpu_blocks: Option<usize>,
        #[arg(long)]
        num_cpu_blocks: Option<usize>,
        #[arg(long, default_value_t = 1)]
        tensor_parallel_size: usize,
        #[arg(long, default_value_t = 256)]
        max_num_seqs: usize,
        #[arg(long, default_value_t = 8192)]
        max_num_batched_tokens: usize,
        #[arg(long, default_value_t = 128)]
        max_prefill_chunk: usize,
        #[arg(long)]
        tokenizer: Option<String>,
        #[arg(long, default_value = "info")]
        log_level: String,
        #[arg(long)]
        disable_telemetry: bool,
    },
    /// Show system info (GPU, memory, etc.)
    Info,
    /// Run throughput benchmarks (direct engine, no HTTP)
    Benchmark {
        #[arg(long)]
        model: String,
        /// Comma-separated batch sizes to sweep, e.g. "1,2,4,8,16,32,64,128"
        #[arg(long, default_value = "1,2,4,8,16,32,64,128,256")]
        n: String,
        /// Tokens to generate per request
        #[arg(long, default_value_t = 512)]
        output_len: usize,
        #[arg(long, default_value = "auto")]
        dtype: Dtype,
        #[arg(long)]
        max_model_len: Option<usize>,
        #[arg(long, default_value_t = 0.90)]
        gpu_memory_utilization: f32,
        #[arg(long, default_value_t = 0.0)]
        gpu_memory_reserve_gb: f32,
        #[arg(long)]
        num_gpu_blocks: Option<usize>,
        #[arg(long)]
        num_cpu_blocks: Option<usize>,
        /// Print results as JSON (for scripted comparison)
        #[arg(long)]
        json: bool,
    },
}

fn init_tracing(log_level: &str) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();
}

fn detect_gpu_and_log() {
    let devices = rvllm_gpu::prelude::list_devices();
    if devices.is_empty() {
        info!("no GPU devices detected");
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
fn detect_device_string() -> anyhow::Result<&'static str> {
    #[cfg(feature = "cuda")]
    {
        let devices = rvllm_gpu::prelude::list_devices();
        if !devices.is_empty() {
            return Ok("cuda");
        }
        anyhow::bail!(
            "cuda feature enabled but no CUDA devices found; refusing to start non-GPU backend"
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok("cpu")
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
            gpu_memory_reserve_gb,
            num_gpu_blocks,
            num_cpu_blocks,
            tensor_parallel_size,
            max_num_seqs,
            max_num_batched_tokens,
            max_prefill_chunk,
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
                let mut cache = CacheConfigImpl::builder()
                    .gpu_memory_utilization(gpu_memory_utilization)
                    .gpu_memory_reserve_gb(gpu_memory_reserve_gb);
                if let Some(v) = num_gpu_blocks {
                    cache = cache.num_gpu_blocks(v);
                }
                if let Some(v) = num_cpu_blocks {
                    cache = cache.num_cpu_blocks(v);
                }
                EngineConfig::builder()
                    .model({
                        let mut m = ModelConfigImpl::builder()
                            .model_path(&model)
                            .dtype(dtype);
                        if let Some(max_model_len) = max_model_len {
                            m = m.max_model_len(max_model_len);
                        }
                        if let Some(ref tok) = tokenizer {
                            m = m.tokenizer_path(tok);
                        }
                        m.build()
                    })
                    .cache(cache.build())
                    .scheduler(
                        SchedulerConfigImpl::builder()
                            .max_num_seqs(max_num_seqs)
                            .max_num_batched_tokens(max_num_batched_tokens)
                            .max_prefill_chunk(max_prefill_chunk)
                            .build(),
                    )
                    .parallel(
                        ParallelConfigImpl::builder()
                            .tensor_parallel_size(tensor_parallel_size)
                            .build(),
                    )
                    .device(
                        DeviceConfig::builder()
                            .device(detect_device_string()?)
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
                max_model_len = config.model.max_model_len,
                gpu_memory_utilization = gpu_memory_utilization,
                gpu_memory_reserve_gb = gpu_memory_reserve_gb,
                num_gpu_blocks = num_gpu_blocks,
                num_cpu_blocks = num_cpu_blocks,
                tp_size = tensor_parallel_size,
                max_num_batched_tokens = max_num_batched_tokens,
                max_prefill_chunk = max_prefill_chunk,
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
            n,
            output_len,
            dtype,
            max_model_len,
            gpu_memory_utilization,
            gpu_memory_reserve_gb,
            num_gpu_blocks,
            num_cpu_blocks,
            json,
        } => {
            init_tracing("warn");
            let multi = n.contains(',');
            if multi {
                eprintln!("rvLLM benchmark -- direct engine, no HTTP");
                eprintln!("model: {model}  output_len: {output_len}  dtype: {dtype}");
            }

            let batch_sizes: Vec<usize> =
                n.split(',').filter_map(|s| s.trim().parse().ok()).collect();

            // Build engine config
            let config = {
                use rvllm_config::*;
                let mut cache = CacheConfigImpl::builder()
                    .gpu_memory_utilization(gpu_memory_utilization)
                    .gpu_memory_reserve_gb(gpu_memory_reserve_gb);
                if let Some(v) = num_gpu_blocks {
                    cache = cache.num_gpu_blocks(v);
                }
                if let Some(v) = num_cpu_blocks {
                    cache = cache.num_cpu_blocks(v);
                }
                EngineConfig::builder()
                    .model({
                        let mut m = ModelConfigImpl::builder()
                            .model_path(&model)
                            .dtype(dtype);
                        if let Some(max_model_len) = max_model_len {
                            m = m.max_model_len(max_model_len);
                        }
                        m.build()
                    })
                    .cache(cache.build())
                    .scheduler(
                        SchedulerConfigImpl::builder()
                            .max_num_seqs(batch_sizes.iter().copied().max().unwrap_or(256).max(256))
                            .build(),
                    )
                    .parallel(ParallelConfigImpl::builder().build())
                    .device(
                        DeviceConfig::builder()
                            .device(detect_device_string()?)
                            .build(),
                    )
                    .telemetry(TelemetryConfig::builder().enabled(false).build())
                    .build()
            };

            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("benchmark requires --features cuda");
                std::process::exit(1);
            }

            #[cfg(feature = "cuda")]
            {
                const BENCH_PROMPTS: &[&str] = &[
                    "Explain the theory of relativity in simple terms.",
                    "Write a Python function to sort a list of integers.",
                    "What are the main differences between TCP and UDP?",
                    "Describe the process of photosynthesis step by step.",
                    "Write a short story about a robot learning to paint.",
                    "Explain how a transformer neural network works.",
                    "What are the advantages of Rust over C++?",
                    "Describe the water cycle in detail.",
                    "Write a haiku about machine learning.",
                    "Explain the concept of recursion with an example.",
                    "What is the difference between a stack and a queue?",
                    "Describe how HTTPS encryption works.",
                    "Write a SQL query to find duplicate records in a table.",
                    "Explain the CAP theorem in distributed systems.",
                    "What are the main principles of object-oriented programming?",
                    "Describe the architecture of a modern CPU.",
                    "Write a regular expression to validate email addresses.",
                    "Explain how garbage collection works in Java.",
                    "What is the difference between concurrency and parallelism?",
                    "Describe the MapReduce programming model.",
                    "Explain how a B-tree index works in databases.",
                    "What are the trade-offs between microservices and monoliths?",
                    "Describe the process of DNS resolution.",
                    "Write pseudocode for the A* pathfinding algorithm.",
                ];

                // Each batch size runs as a separate process to avoid CUDA context
                // poisoning between different graph captures.
                if batch_sizes.len() > 1 {
                    let exe = std::env::current_exe()?;
                    for &batch in &batch_sizes {
                        let mut command = std::process::Command::new(&exe);
                        command
                            .arg("benchmark")
                            .arg("--model").arg(&model)
                            .arg("--dtype").arg(format!("{dtype}"))
                            .arg("--output-len").arg(format!("{output_len}"))
                            .arg("--gpu-memory-utilization").arg(format!("{gpu_memory_utilization}"))
                            .arg("--n").arg(format!("{batch}"))
                            .args(if json { vec!["--json"] } else { vec![] })
                            .env("RVLLM_PTX_DIR", std::env::var("RVLLM_PTX_DIR").unwrap_or_default());
                        if let Some(max_model_len) = max_model_len {
                            command.arg("--max-model-len").arg(format!("{max_model_len}"));
                        }
                        let output = command.output()?;
                        // Print child's stderr (bench results) and stdout (json)
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        for line in stderr.lines() {
                            if !line.is_empty() {
                                eprintln!("{line}");
                            }
                        }
                        if !stdout.is_empty() {
                            println!("{stdout}");
                        }
                    }
                    return Ok(()); // We handled everything via subprocesses
                }

                let mut results = Vec::new();

                for &batch in &batch_sizes {
                    // Single batch size -- run directly in this process on the sync GPU engine.
                    let mut engine = rvllm_engine::GpuLLMEngine::new(config.clone())?;

                    // Warmup: run `batch` short requests to trigger graph capture at this N.
                    {
                        for i in 0..batch {
                            let p = rvllm_core::types::SamplingParams {
                                temperature: 0.0,
                                max_tokens: 5,
                                ignore_eos: true,
                                ..Default::default()
                            };
                            let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()].to_string();
                            engine.add_request_auto_id_with_emit_intermediate(prompt, p, false)?;
                        }
                        let _ = engine.run_count_tokens_only()?;
                    }

                    let params = rvllm_core::types::SamplingParams {
                        temperature: 0.0,
                        max_tokens: output_len,
                        ignore_eos: true,
                        ..Default::default()
                    };

                    let t0 = std::time::Instant::now();
                    for i in 0..batch {
                        let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()].to_string();
                        engine.add_request_auto_id_with_emit_intermediate(
                            prompt,
                            params.clone(),
                            false,
                        )?;
                    }

                    let (total_toks, finished) = engine.run_count_tokens_only()?;
                    let elapsed = t0.elapsed();
                    let failed = batch.saturating_sub(finished);
                    let tps = total_toks as f64 / elapsed.as_secs_f64();

                    if json {
                        results.push(serde_json::json!({
                            "n": batch,
                            "total_tokens": total_toks,
                            "elapsed_ms": elapsed.as_millis(),
                            "tok_per_sec": (tps * 10.0).round() / 10.0,
                            "failed": failed,
                        }));
                    } else {
                        eprintln!(
                            "N={:<4} {:>6} tok  {:>6}ms  {:>8.1} tok/s  {}",
                            batch,
                            total_toks,
                            elapsed.as_millis(),
                            tps,
                            if failed > 0 {
                                format!("({failed} failed)")
                            } else {
                                "ok".into()
                            },
                        );
                    }
                    // engine dropped here (single iteration when running as subprocess)
                }

                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "engine": "rvllm",
                            "model": model,
                            "output_len": output_len,
                            "results": results,
                        }))?
                    );
                }
            } // #[cfg(feature = "cuda")]
        }
    }
    Ok(())
}
