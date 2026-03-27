//! Metric name constants and description registration.

use metrics::{describe_counter, describe_gauge, describe_histogram, Unit};

// -- Histograms --
pub const REQUEST_LATENCY: &str = "vllm_request_latency_seconds";
pub const TTFT: &str = "vllm_time_to_first_token_seconds";
pub const ITL: &str = "vllm_inter_token_latency_seconds";
pub const FORWARD_TIME: &str = "vllm_forward_time_seconds";
pub const SAMPLE_TIME: &str = "vllm_sample_time_seconds";

// -- Gauges --
pub const TOKENS_PER_SECOND: &str = "vllm_tokens_per_second";
pub const RUNNING_REQUESTS: &str = "vllm_num_running_requests";
pub const WAITING_REQUESTS: &str = "vllm_num_waiting_requests";
pub const GPU_CACHE_USAGE: &str = "rvllm_gpu_cache_usage_percent";
pub const WORKER_TOKENS_PER_SECOND: &str = "rvllm_worker_tokens_per_second";

// -- Counters --
pub const PREEMPTIONS_TOTAL: &str = "vllm_num_preemptions_total";
pub const REQUESTS_TOTAL: &str = "vllm_num_requests_total";
pub const FINISHED_REQUESTS_TOTAL: &str = "vllm_num_finished_requests_total";
pub const PROMPT_TOKENS_TOTAL: &str = "vllm_prompt_tokens_total";
pub const GENERATION_TOKENS_TOTAL: &str = "vllm_generation_tokens_total";
pub const FORWARD_PASSES_TOTAL: &str = "vllm_forward_passes_total";
pub const TOKENS_SAMPLED_TOTAL: &str = "vllm_tokens_sampled_total";
pub const STEPS_TOTAL: &str = "rvllm_engine_steps_total";

/// Register all metric descriptions with the active recorder.
pub fn register_descriptions() {
    describe_histogram!(REQUEST_LATENCY, Unit::Seconds, "End-to-end request latency");
    describe_histogram!(TTFT, Unit::Seconds, "Time to first token");
    describe_histogram!(ITL, Unit::Seconds, "Inter-token latency");
    describe_histogram!(FORWARD_TIME, Unit::Seconds, "GPU forward pass duration");
    describe_histogram!(SAMPLE_TIME, Unit::Seconds, "Token sampling duration");

    describe_gauge!(TOKENS_PER_SECOND, "Tokens generated per second");
    describe_gauge!(RUNNING_REQUESTS, "Number of currently running requests");
    describe_gauge!(WAITING_REQUESTS, "Number of waiting (queued) requests");
    describe_gauge!(GPU_CACHE_USAGE, "GPU KV-cache usage percentage");
    describe_gauge!(WORKER_TOKENS_PER_SECOND, "Worker-level tokens per second throughput");

    describe_counter!(PREEMPTIONS_TOTAL, "Total number of preemptions");
    describe_counter!(REQUESTS_TOTAL, "Total number of requests received");
    describe_counter!(FINISHED_REQUESTS_TOTAL, "Total number of finished requests");
    describe_counter!(PROMPT_TOKENS_TOTAL, "Total prompt tokens processed");
    describe_counter!(GENERATION_TOKENS_TOTAL, "Total generation tokens produced");
    describe_counter!(FORWARD_PASSES_TOTAL, "Total GPU forward passes executed");
    describe_counter!(TOKENS_SAMPLED_TOTAL, "Total tokens sampled across all requests");
    describe_counter!(STEPS_TOTAL, "Total engine step() invocations");
}
