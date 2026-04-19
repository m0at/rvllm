//! 1 Hz clock + power sampler for GB10 bench runs.
//!
//! The GB10 firmware clock regime (851 MHz sustained → 507 MHz throttled
//! after ~3 s) inverts the usual "fewer instructions = faster" heuristic
//! for `fp8_gemv`. Without a trace of `clocks.sm` + `power.draw` through
//! a bench run, that paradox lives in kernel comments instead of in
//! numbers we can stare at.
//!
//! This sampler shells out to `nvidia-smi` once per second and appends
//! one JSONL record per sample to a user-provided path. No NVML dep,
//! no C bindings, no new crates — `nvidia-smi` is always present on a
//! machine that can run a CUDA bench.
//!
//! Output format (one JSON object per line):
//!   { "t_ms": <ms since start>,
//!     "clocks_sm_mhz": <u32>,
//!     "power_draw_w": <f32> }
//!
//! Use:
//!   let log = ClockLog::start(Path::new("bench_clocks.jsonl"))?;
//!   run_bench(...);
//!   log.stop();  // joins the sampler thread, flushes the file

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Running sampler. Drop or `stop()` to terminate.
pub struct ClockLog {
    stop_flag: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    out_path: PathBuf,
}

impl ClockLog {
    /// Spawn the background sampler. Samples at 1 Hz by default.
    ///
    /// Returns `Err` only if the output file cannot be opened —
    /// `nvidia-smi` failures during sampling are logged to the JSONL
    /// stream as `{"t_ms": N, "error": "..."}` rather than aborting.
    pub fn start(out_path: impl Into<PathBuf>) -> std::io::Result<Self> {
        Self::start_with_interval(out_path, Duration::from_secs(1))
    }

    /// Same as `start` but with an explicit sampling period. Useful for
    /// tight captures around a suspected throttle transition.
    pub fn start_with_interval(
        out_path: impl Into<PathBuf>,
        interval: Duration,
    ) -> std::io::Result<Self> {
        let out_path: PathBuf = out_path.into();
        let file = File::create(&out_path)?;
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_w = Arc::clone(&stop_flag);

        let worker = thread::Builder::new()
            .name("rvllm-bench-clocklog".into())
            .spawn(move || {
                let mut writer = BufWriter::new(file);
                let t0 = Instant::now();
                while !stop_flag_w.load(Ordering::Relaxed) {
                    let t_ms = t0.elapsed().as_millis() as u64;
                    match sample_nvidia_smi() {
                        Ok((clock_mhz, power_w)) => {
                            let _ = writeln!(
                                writer,
                                "{{\"t_ms\":{t_ms},\"clocks_sm_mhz\":{clock_mhz},\"power_draw_w\":{power_w:.2}}}"
                            );
                        }
                        Err(e) => {
                            // Record the failure at its timestamp so the
                            // trace doesn't silently have gaps. Still
                            // keep sampling — transient nvidia-smi lag
                            // is recoverable.
                            let msg = e.replace('"', "'");
                            let _ = writeln!(
                                writer,
                                "{{\"t_ms\":{t_ms},\"error\":\"{msg}\"}}"
                            );
                        }
                    }
                    let _ = writer.flush();
                    thread::sleep(interval);
                }
            })?;

        Ok(Self {
            stop_flag,
            worker: Some(worker),
            out_path,
        })
    }

    /// Path of the JSONL log.
    pub fn path(&self) -> &Path {
        &self.out_path
    }

    /// Signal the worker to stop and join it. Idempotent.
    pub fn stop(mut self) {
        self.stop_inner();
    }

    fn stop_inner(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

impl Drop for ClockLog {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

/// One shot of `nvidia-smi` → `(clocks.sm MHz, power.draw W)`.
///
/// Pulled out for unit-testability of the parser (see tests below).
fn sample_nvidia_smi() -> Result<(u32, f32), String> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .map_err(|e| format!("spawn nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "nvidia-smi exit {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let line = String::from_utf8_lossy(&out.stdout);
    parse_csv_line(line.trim())
}

fn parse_csv_line(line: &str) -> Result<(u32, f32), String> {
    let mut parts = line.split(',').map(str::trim);
    let clock_s = parts.next().ok_or("no clocks.sm column")?;
    let power_s = parts.next().ok_or("no power.draw column")?;
    let clock = clock_s
        .parse::<u32>()
        .map_err(|e| format!("clocks.sm={clock_s:?}: {e}"))?;
    let power = power_s
        .parse::<f32>()
        .map_err(|e| format!("power.draw={power_s:?}: {e}"))?;
    Ok((clock, power))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sustained_sample() {
        // Realistic GB10 sustained line from nvidia-smi.
        let (clock, power) = parse_csv_line("851, 42.31").unwrap();
        assert_eq!(clock, 851);
        assert!((power - 42.31).abs() < 1e-3);
    }

    #[test]
    fn parses_throttled_sample() {
        let (clock, power) = parse_csv_line("507, 18.04").unwrap();
        assert_eq!(clock, 507);
        assert!((power - 18.04).abs() < 1e-3);
    }

    #[test]
    fn rejects_malformed() {
        assert!(parse_csv_line("").is_err());
        assert!(parse_csv_line("851").is_err());
        assert!(parse_csv_line("NaN, 42.0").is_err());
    }

    #[test]
    fn sampler_writes_jsonl_on_fake_nvidia_smi() {
        // Skip if no nvidia-smi — this test is best-effort on GPU boxes.
        if Command::new("nvidia-smi")
            .arg("--help")
            .output()
            .ok()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: nvidia-smi not runnable");
            return;
        }
        let tmp = std::env::temp_dir().join(format!(
            "rvllm-clocklog-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let log = ClockLog::start_with_interval(&tmp, Duration::from_millis(100)).unwrap();
        std::thread::sleep(Duration::from_millis(350));
        log.stop();
        let body = std::fs::read_to_string(&tmp).unwrap();
        // At 100 ms interval over 350 ms we expect 3-4 lines. Be lax.
        assert!(
            body.lines().count() >= 2,
            "expected >= 2 JSONL lines, got: {body:?}",
        );
        // Each line is either a sample or an error record; both start with `{"t_ms":`.
        for line in body.lines() {
            assert!(line.starts_with("{\"t_ms\":"), "bad line: {line:?}");
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
