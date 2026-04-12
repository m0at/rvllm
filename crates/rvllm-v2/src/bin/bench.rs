//! Standalone v2 benchmark binary.
//!
//! Usage:
//!   cargo run -p rvllm-v2 --features cuda --bin bench -- \
//!     --model meta-llama/Llama-3-8B \
//!     --n 1,32,64,96,128 \
//!     --output-len 512

use std::time::Instant;

use clap::Parser;
use rvllm_core::prelude::SamplingParams;

use rvllm_v2::integration::{init_engine, V2EngineConfig};

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

#[derive(Parser)]
#[command(name = "rvllm-v2-bench", about = "v2 engine throughput benchmark")]
struct Cli {
    #[arg(long)]
    model: String,

    #[arg(long, default_value = "1,32,64,96,128")]
    n: String,

    #[arg(long, default_value_t = 512)]
    output_len: usize,

    #[arg(long)]
    num_gpu_blocks: Option<usize>,

    #[arg(long)]
    num_cpu_blocks: Option<usize>,

    #[arg(long, default_value_t = 0.90)]
    gpu_memory_utilization: f32,

    #[arg(long, default_value_t = 0.0)]
    gpu_memory_reserve_gb: f32,

    #[arg(long)]
    max_model_len: Option<usize>,

    #[arg(long)]
    json: bool,
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();

    let batch_sizes: Vec<usize> = cli
        .n
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if batch_sizes.len() > 1 {
        eprintln!("rvLLM v2 benchmark -- direct engine, no HTTP");
        eprintln!("model: {}  output_len: {}", cli.model, cli.output_len);
    }

    let mut json_results = Vec::new();

    for &batch in &batch_sizes {
        let config = V2EngineConfig {
            num_gpu_blocks_override: cli.num_gpu_blocks,
            num_cpu_blocks_override: cli.num_cpu_blocks,
            gpu_memory_utilization: cli.gpu_memory_utilization,
            gpu_memory_reserve_gb: cli.gpu_memory_reserve_gb,
            max_model_len: cli.max_model_len.unwrap_or(2048),
            max_num_seqs: batch.max(256),
            ..V2EngineConfig::from_model_path(&cli.model)
        };

        let mut engine = init_engine(&config);

        // Warmup: short requests to trigger graph capture
        let warmup_params = SamplingParams {
            temperature: 0.0,
            max_tokens: 5,
            ignore_eos: true,
            ..Default::default()
        };
        for i in 0..batch {
            let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()].to_string();
            engine.add_request(prompt, warmup_params.clone())?;
        }
        while engine.has_pending_work() {
            engine.step().map_err(|e| anyhow::anyhow!("{e}"))?;
        }

        // Bench run
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: cli.output_len,
            ignore_eos: true,
            ..Default::default()
        };

        for i in 0..batch {
            let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()].to_string();
            engine
                .add_request(prompt, params.clone())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }

        let t0 = Instant::now();
        let mut total_tokens = 0usize;
        let mut finished = 0usize;

        while engine.has_pending_work() {
            let outputs = engine.step().map_err(|e| anyhow::anyhow!("{e}"))?;
            for out in &outputs {
                total_tokens += 1;
                if out.finished {
                    finished += 1;
                }
            }
        }

        let elapsed = t0.elapsed();
        let failed = batch.saturating_sub(finished);
        let tps = total_tokens as f64 / elapsed.as_secs_f64();

        if cli.json {
            json_results.push(serde_json::json!({
                "n": batch,
                "total_tokens": total_tokens,
                "elapsed_ms": elapsed.as_millis(),
                "tok_per_sec": (tps * 10.0).round() / 10.0,
                "failed": failed,
            }));
        } else {
            eprintln!(
                "N={:<4} {:>6} tok  {:>6}ms  {:>8.1} tok/s  {}",
                batch,
                total_tokens,
                elapsed.as_millis(),
                tps,
                if failed > 0 {
                    format!("({failed} failed)")
                } else {
                    "ok".into()
                },
            );
        }
    }

    if cli.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "engine": "rvllm-v2",
                "model": cli.model,
                "output_len": cli.output_len,
                "results": json_results,
            }))?
        );
    }

    Ok(())
}
