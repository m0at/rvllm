#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use rvrlm::{Prompt, RvllmCudaConfig, ServeRequest, ServeService};

#[cfg(feature = "cuda")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

#[cfg(feature = "cuda")]
fn request(prompt: &str) -> ServeRequest {
    ServeRequest {
        prompt: Prompt::from(prompt),
        ..ServeRequest::default()
    }
}

#[cfg(feature = "cuda")]
fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    if let Err(error) = run() {
        eprintln!("rvrlm-bench: {error}");
        std::process::exit(1);
    }
}

#[cfg(feature = "cuda")]
fn run() -> Result<(), String> {
    let prompt = std::env::var("RVRLM_PROMPT")
        .unwrap_or_else(|_| "Write one short sentence about recursive language models.".to_owned());
    let ttft_prompt = std::env::var("RVRLM_TTFT_PROMPT").unwrap_or_else(|_| prompt.clone());
    let ppl_prompt = std::env::var("RVRLM_PPL_PROMPT").ok();
    let max_new_tokens = env_usize("RVRLM_MAX_NEW_TOKENS", 128);
    let iters = env_usize("RVLLM_ITERS", 20);
    let warmup = env_usize("RVLLM_WARMUP", 3);

    let mut ttft_config =
        RvllmCudaConfig::from_env().map_err(|error| format!("config: {error}"))?;
    ttft_config.max_new_tokens = 1;
    let mut ttft_service = ServeService::from_rvllm_cuda(ttft_config)
        .map_err(|error| format!("ttft service init: {error}"))?;

    eprintln!("== rvrlm-bench ==");
    eprintln!("prompt chars = {}", prompt.len());
    eprintln!("iters        = {iters} (warmup {warmup})");
    eprintln!("max_new      = {max_new_tokens}");

    let cold = ttft_service
        .complete(request(&ttft_prompt))
        .map_err(|error| format!("ttft cold: {error}"))?;
    let hot = ttft_service
        .complete(request(&ttft_prompt))
        .map_err(|error| format!("ttft hot: {error}"))?;
    let ttft_cold_ms = cold.completion.execution_time_secs * 1.0e3;
    let ttft_hot_ms = hot.completion.execution_time_secs * 1.0e3;
    drop(ttft_service);

    let ppl_summary = if let Some(ppl_prompt) = ppl_prompt.as_deref() {
        let mut ppl_service = ServeService::from_rvllm_cuda(
            RvllmCudaConfig::from_env().map_err(|error| format!("config: {error}"))?,
        )
        .map_err(|error| format!("ppl service init: {error}"))?;
        Some(
            ppl_service
                .perplexity(request(ppl_prompt))
                .map_err(|error| format!("ppl: {error}"))?,
        )
    } else {
        None
    };

    let mut bench_config =
        RvllmCudaConfig::from_env().map_err(|error| format!("config: {error}"))?;
    bench_config.max_new_tokens = max_new_tokens;
    let mut service = ServeService::from_rvllm_cuda(bench_config)
        .map_err(|error| format!("bench service init: {error}"))?;

    let mut total_input_tokens = 0u64;
    let mut total_output_tokens = 0u64;
    let mut baseline_input_tokens = 0u64;
    let mut baseline_output_tokens = 0u64;

    for _ in 0..warmup {
        let response = service
            .complete(request(&prompt))
            .map_err(|error| format!("warmup completion: {error}"))?;
        baseline_input_tokens = response.completion.usage_summary.total_input_tokens();
        baseline_output_tokens = response.completion.usage_summary.total_output_tokens();
    }

    let started = Instant::now();
    let mut last_input_tokens = baseline_input_tokens;
    let mut last_output_tokens = baseline_output_tokens;
    for _ in 0..iters {
        let response = service
            .complete(request(&prompt))
            .map_err(|error| format!("bench completion: {error}"))?;
        let input_tokens = response.completion.usage_summary.total_input_tokens();
        let output_tokens = response.completion.usage_summary.total_output_tokens();
        total_input_tokens += input_tokens.saturating_sub(last_input_tokens);
        total_output_tokens += output_tokens.saturating_sub(last_output_tokens);
        last_input_tokens = input_tokens;
        last_output_tokens = output_tokens;
    }
    let elapsed_s = started.elapsed().as_secs_f64();
    let tok_per_sec = if elapsed_s > 0.0 {
        total_output_tokens as f64 / elapsed_s
    } else {
        0.0
    };
    let ms_per_request = if iters > 0 {
        elapsed_s * 1.0e3 / iters as f64
    } else {
        0.0
    };
    let avg_prompt_tokens = if iters > 0 {
        total_input_tokens as f64 / iters as f64
    } else {
        0.0
    };
    let avg_output_tokens = if iters > 0 {
        total_output_tokens as f64 / iters as f64
    } else {
        0.0
    };

    match &ppl_summary {
        Some(summary) => eprintln!(
            "bench: batch=1 iters={} -> {:.1} tok/s ({:.3} ms/request) ttft_cold={:.2}ms ttft_hot={:.2}ms ppl={:.4} tokens={}",
            iters,
            tok_per_sec,
            ms_per_request,
            ttft_cold_ms,
            ttft_hot_ms,
            summary.perplexity,
            summary.evaluated_tokens
        ),
        None => eprintln!(
            "bench: batch=1 iters={} -> {:.1} tok/s ({:.3} ms/request) ttft_cold={:.2}ms ttft_hot={:.2}ms",
            iters, tok_per_sec, ms_per_request, ttft_cold_ms, ttft_hot_ms
        ),
    }
    match ppl_summary {
        Some(summary) => println!(
            "{{\"mode\":\"rvrlm\",\"batch\":1,\"iters\":{},\"warmup\":{},\"max_new_tokens\":{},\"tok_per_sec\":{:.1},\"ms_per_request\":{:.4},\"ttft_cold_ms\":{:.3},\"ttft_hot_ms\":{:.3},\"prompt_tokens_avg\":{:.2},\"output_tokens_avg\":{:.2},\"perplexity\":{:.4},\"ppl_evaluated_tokens\":{},\"ppl_total_nll\":{:.6}}}",
            iters,
            warmup,
            max_new_tokens,
            tok_per_sec,
            ms_per_request,
            ttft_cold_ms,
            ttft_hot_ms,
            avg_prompt_tokens,
            avg_output_tokens,
            summary.perplexity,
            summary.evaluated_tokens,
            summary.total_nll
        ),
        None => println!(
            "{{\"mode\":\"rvrlm\",\"batch\":1,\"iters\":{},\"warmup\":{},\"max_new_tokens\":{},\"tok_per_sec\":{:.1},\"ms_per_request\":{:.4},\"ttft_cold_ms\":{:.3},\"ttft_hot_ms\":{:.3},\"prompt_tokens_avg\":{:.2},\"output_tokens_avg\":{:.2}}}",
            iters,
            warmup,
            max_new_tokens,
            tok_per_sec,
            ms_per_request,
            ttft_cold_ms,
            ttft_hot_ms,
            avg_prompt_tokens,
            avg_output_tokens
        ),
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("rvrlm-bench requires --features cuda");
    std::process::exit(1);
}
