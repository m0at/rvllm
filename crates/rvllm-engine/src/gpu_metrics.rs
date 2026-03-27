//! GPU engine metrics instrumentation.
//!
//! Wraps `rvllm_telemetry::MetricsRecorder` to instrument the `GpuLLMEngine`
//! step loop: request_latency, time-to-first-token, inter-token latency,
//! num_running, num_waiting, gpu_cache_usage_percent, and preemption counts.

use std::collections::HashMap;

use rvllm_core::prelude::RequestId;
use rvllm_telemetry::MetricsRecorder;

/// Per-request timing state tracked by the engine metrics layer.
#[derive(Debug)]
struct RequestTiming {
    first_token_emitted: bool,
    prompt_tokens: usize,
    gen_tokens: usize,
}

/// Engine-level metrics tracker that instruments the GpuLLMEngine.
///
/// Holds per-request timing state and delegates to `MetricsRecorder`
/// for the actual metric emission.
pub struct GpuEngineMetrics {
    recorder: MetricsRecorder,
    timings: HashMap<RequestId, RequestTiming>,
}

impl GpuEngineMetrics {
    /// Create a new metrics tracker backed by a fresh `MetricsRecorder`.
    pub fn new() -> Self {
        Self {
            recorder: MetricsRecorder::new(),
            timings: HashMap::new(),
        }
    }

    /// Record that a new request has been added to the engine.
    pub fn on_request_added(&mut self, request_id: RequestId, prompt_tokens: usize) {
        let id_str = request_id.0.to_string();
        self.recorder.record_request_start(&id_str);
        self.timings.insert(request_id, RequestTiming {
            first_token_emitted: false,
            prompt_tokens,
            gen_tokens: 0,
        });
    }

    /// Record that a token was generated for the given request.
    pub fn on_token_generated(&mut self, request_id: RequestId) {
        let id_str = request_id.0.to_string();
        if let Some(timing) = self.timings.get_mut(&request_id) {
            if !timing.first_token_emitted {
                self.recorder.record_first_token(&id_str);
                timing.first_token_emitted = true;
            } else {
                self.recorder.record_token_generated(&id_str);
            }
            timing.gen_tokens += 1;
        }
    }

    /// Record that a request has finished (completed or aborted).
    pub fn on_request_finished(&mut self, request_id: RequestId) {
        let id_str = request_id.0.to_string();
        if let Some(timing) = self.timings.remove(&request_id) {
            self.recorder.record_request_finished(
                &id_str,
                timing.prompt_tokens,
                timing.gen_tokens,
            );
        }
    }

    /// Update scheduler queue gauges at the start of each step.
    pub fn update_queue_sizes(&self, num_running: usize, num_waiting: usize) {
        self.recorder.update_running_requests(num_running);
        self.recorder.update_waiting_requests(num_waiting);
    }

    /// Update GPU KV cache usage percentage.
    pub fn update_cache_usage(&self, percent: f32) {
        self.recorder.update_cache_usage(percent);
    }

    /// Record a preemption event.
    pub fn on_preemption(&self) {
        self.recorder.record_preemption();
    }

    /// Record that an engine step was executed.
    pub fn on_step(&self) {
        self.recorder.record_step();
    }

    /// Access the underlying recorder for direct use.
    pub fn recorder(&self) -> &MetricsRecorder {
        &self.recorder
    }
}

impl Default for GpuEngineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install_test_recorder() {
        let _ = metrics_exporter_prometheus::PrometheusBuilder::new()
            .install_recorder();
    }

    #[test]
    fn full_request_lifecycle() {
        install_test_recorder();
        let mut m = GpuEngineMetrics::new();

        let rid = RequestId(42);
        m.on_request_added(rid, 10);
        m.update_queue_sizes(1, 0);

        // First token => records TTFT
        m.on_token_generated(rid);
        // Subsequent tokens => records ITL
        m.on_token_generated(rid);
        m.on_token_generated(rid);

        m.on_step();
        m.on_request_finished(rid);

        // Timing entry should be cleaned up.
        assert!(!m.timings.contains_key(&rid));
    }

    #[test]
    fn queue_and_cache_gauges() {
        install_test_recorder();
        let m = GpuEngineMetrics::new();

        m.update_queue_sizes(5, 3);
        m.update_cache_usage(72.5);
        m.on_preemption();
    }

    #[test]
    fn multiple_requests() {
        install_test_recorder();
        let mut m = GpuEngineMetrics::new();

        let r1 = RequestId(1);
        let r2 = RequestId(2);

        m.on_request_added(r1, 5);
        m.on_request_added(r2, 8);
        m.update_queue_sizes(2, 0);

        m.on_token_generated(r1);
        m.on_token_generated(r2);
        m.on_step();

        m.on_token_generated(r1);
        m.on_request_finished(r1);

        m.on_token_generated(r2);
        m.on_token_generated(r2);
        m.on_request_finished(r2);

        assert!(m.timings.is_empty());
    }
}
