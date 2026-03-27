use std::collections::HashMap;
use std::time::Instant;

use parking_lot::Mutex;

use crate::metrics::*;

/// Tracks per-request timing state.
struct RequestState {
    start: Instant,
    first_token: Option<Instant>,
    last_token: Option<Instant>,
}

/// Convenience struct for recording vLLM metrics through the `metrics` crate.
pub struct MetricsRecorder {
    requests: Mutex<HashMap<String, RequestState>>,
}

impl MetricsRecorder {
    pub fn new() -> Self {
        Self {
            requests: Mutex::new(HashMap::new()),
        }
    }

    pub fn record_request_start(&self, request_id: &str) {
        metrics::counter!(REQUESTS_TOTAL).increment(1);
        self.requests.lock().insert(
            request_id.to_owned(),
            RequestState {
                start: Instant::now(),
                first_token: None,
                last_token: None,
            },
        );
    }

    pub fn record_first_token(&self, request_id: &str) {
        let now = Instant::now();
        let mut map = self.requests.lock();
        if let Some(state) = map.get_mut(request_id) {
            let ttft = now.duration_since(state.start).as_secs_f64();
            metrics::histogram!(TTFT).record(ttft);
            state.first_token = Some(now);
            state.last_token = Some(now);
        }
    }

    pub fn record_token_generated(&self, request_id: &str) {
        let now = Instant::now();
        let mut map = self.requests.lock();
        if let Some(state) = map.get_mut(request_id) {
            if let Some(last) = state.last_token {
                let itl = now.duration_since(last).as_secs_f64();
                metrics::histogram!(ITL).record(itl);
            }
            state.last_token = Some(now);
        }
    }

    pub fn record_request_finished(
        &self,
        request_id: &str,
        prompt_tokens: usize,
        gen_tokens: usize,
    ) {
        metrics::counter!(FINISHED_REQUESTS_TOTAL).increment(1);
        metrics::counter!(PROMPT_TOKENS_TOTAL).increment(prompt_tokens as u64);
        metrics::counter!(GENERATION_TOKENS_TOTAL).increment(gen_tokens as u64);

        let mut map = self.requests.lock();
        if let Some(state) = map.remove(request_id) {
            let latency = state.start.elapsed().as_secs_f64();
            metrics::histogram!(REQUEST_LATENCY).record(latency);

            if latency > 0.0 && gen_tokens > 0 {
                let tps = gen_tokens as f64 / latency;
                metrics::gauge!(TOKENS_PER_SECOND).set(tps);
            }
        }
    }

    pub fn record_preemption(&self) {
        metrics::counter!(PREEMPTIONS_TOTAL).increment(1);
    }

    pub fn update_cache_usage(&self, percent: f32) {
        metrics::gauge!(GPU_CACHE_USAGE).set(f64::from(percent));
    }

    pub fn update_running_requests(&self, count: usize) {
        metrics::gauge!(RUNNING_REQUESTS).set(count as f64);
    }

    /// Update the number of waiting (queued) requests gauge.
    pub fn update_waiting_requests(&self, count: usize) {
        metrics::gauge!(WAITING_REQUESTS).set(count as f64);
    }

    /// Record the duration of a single GPU forward pass.
    pub fn record_forward_time(&self, seconds: f64) {
        metrics::histogram!(FORWARD_TIME).record(seconds);
        metrics::counter!(FORWARD_PASSES_TOTAL).increment(1);
    }

    /// Record the duration of a single token sampling step.
    pub fn record_sample_time(&self, seconds: f64) {
        metrics::histogram!(SAMPLE_TIME).record(seconds);
    }

    /// Record the number of tokens sampled in one step.
    pub fn record_tokens_sampled(&self, count: usize) {
        metrics::counter!(TOKENS_SAMPLED_TOTAL).increment(count as u64);
    }

    /// Update the worker-level tokens-per-second throughput gauge.
    pub fn update_worker_tps(&self, tps: f64) {
        metrics::gauge!(WORKER_TOKENS_PER_SECOND).set(tps);
    }

    /// Increment the engine step counter.
    pub fn record_step(&self) {
        metrics::counter!(STEPS_TOTAL).increment(1);
    }
}

impl Default for MetricsRecorder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Install a throwaway recorder so the metrics macros don't panic.
    fn install_test_recorder() {
        let _ = metrics_exporter_prometheus::PrometheusBuilder::new()
            .install_recorder();
    }

    #[test]
    fn request_lifecycle() {
        install_test_recorder();
        let rec = MetricsRecorder::new();

        rec.record_request_start("r1");
        rec.record_first_token("r1");
        rec.record_token_generated("r1");
        rec.record_token_generated("r1");
        rec.record_request_finished("r1", 10, 20);

        // After finish the request state is cleaned up.
        assert!(!rec.requests.lock().contains_key("r1"));
    }

    #[test]
    fn preemption_and_gauges() {
        install_test_recorder();
        let rec = MetricsRecorder::new();

        rec.record_preemption();
        rec.update_cache_usage(42.5);
        rec.update_running_requests(3);
    }
}
