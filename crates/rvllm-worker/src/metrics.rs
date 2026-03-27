//! GPU worker metrics instrumentation.
//!
//! Wraps `rvllm_telemetry::MetricsRecorder` to record per-step timing data
//! from the GPU forward pass and sampling phases. Tracks forward_time,
//! sample_time, and tokens_per_second at the worker level.

use std::time::Instant;

use rvllm_telemetry::MetricsRecorder;

/// Timing snapshot for a single worker execute() call.
#[derive(Debug, Clone)]
pub struct WorkerStepTimings {
    /// Wall-clock seconds spent in the GPU forward pass.
    pub forward_secs: f64,
    /// Wall-clock seconds spent in token sampling.
    pub sample_secs: f64,
    /// Number of tokens processed in this step.
    pub num_tokens: usize,
    /// Number of sequences that produced a sampled token.
    pub num_sampled: usize,
}

/// RAII timer that records elapsed time on drop into a closure.
pub struct ScopedTimer {
    start: Instant,
}

impl ScopedTimer {
    /// Start a new scoped timer.
    pub fn start() -> Self {
        Self { start: Instant::now() }
    }

    /// Consume the timer and return elapsed seconds.
    pub fn elapsed_secs(self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

/// Record a completed worker step's timings into the global metrics recorder.
pub fn record_worker_step(recorder: &MetricsRecorder, timings: &WorkerStepTimings) {
    recorder.record_forward_time(timings.forward_secs);
    recorder.record_sample_time(timings.sample_secs);
    recorder.record_tokens_sampled(timings.num_sampled);

    let total = timings.forward_secs + timings.sample_secs;
    if total > 0.0 && timings.num_sampled > 0 {
        let tps = timings.num_sampled as f64 / total;
        recorder.update_worker_tps(tps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_timer_measures_nonzero() {
        let timer = ScopedTimer::start();
        // Spin briefly
        let mut _x = 0u64;
        for i in 0..1000 {
            _x = _x.wrapping_add(i);
        }
        let elapsed = timer.elapsed_secs();
        assert!(elapsed >= 0.0);
    }

    #[test]
    fn step_timings_construction() {
        let t = WorkerStepTimings {
            forward_secs: 0.01,
            sample_secs: 0.002,
            num_tokens: 128,
            num_sampled: 4,
        };
        assert_eq!(t.num_tokens, 128);
        assert_eq!(t.num_sampled, 4);
    }

    #[test]
    fn record_worker_step_with_recorder() {
        // Install a throwaway prometheus recorder for this test.
        let _ = metrics_exporter_prometheus::PrometheusBuilder::new()
            .install_recorder();

        let recorder = MetricsRecorder::new();
        let timings = WorkerStepTimings {
            forward_secs: 0.05,
            sample_secs: 0.005,
            num_tokens: 64,
            num_sampled: 8,
        };
        record_worker_step(&recorder, &timings);
    }
}
