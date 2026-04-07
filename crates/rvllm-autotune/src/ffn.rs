use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Args)]
pub struct FfnSweepArgs {
    #[arg(long, default_value = "target/release/rvllm")]
    pub rvllm_bin: PathBuf,
    #[arg(long)]
    pub model: String,
    #[arg(long, default_value = "1,32,64,96,128")]
    pub n: String,
    #[arg(long, default_value_t = 128)]
    pub output_len: usize,
    #[arg(long, default_value_t = 3584)]
    pub hidden: usize,
    #[arg(long, default_value_t = 18944)]
    pub intermediate: usize,
    #[arg(long)]
    pub harness_bin: Option<PathBuf>,
    #[arg(long)]
    pub cutlass_so: Option<PathBuf>,
    #[arg(long)]
    pub out_dir: Option<PathBuf>,
    #[arg(long, default_value = "ffn-sweep")]
    pub label: String,
    #[arg(long, default_value_t = false)]
    pub skip_harness: bool,
    #[arg(long, default_value_t = 20)]
    pub harness_timeout_s: u64,
}

#[derive(Debug, Clone, Args)]
pub struct FfnKernelSweepArgs {
    #[arg(long, default_value = "target/release/rvllm")]
    pub rvllm_bin: PathBuf,
    #[arg(long, default_value = "kernels/build_cutlass_so.sh")]
    pub build_script: PathBuf,
    #[arg(long, default_value = "sm_90")]
    pub arch: String,
    #[arg(long, default_value = "/root/cutlass")]
    pub cutlass_dir: PathBuf,
    #[arg(long)]
    pub model: String,
    #[arg(long, default_value = "1,32,64,96,128")]
    pub n: String,
    #[arg(long, default_value_t = 128)]
    pub output_len: usize,
    #[arg(long, default_value_t = 3584)]
    pub hidden: usize,
    #[arg(long, default_value_t = 18944)]
    pub intermediate: usize,
    #[arg(long)]
    pub harness_bin: PathBuf,
    #[arg(long)]
    pub out_dir: Option<PathBuf>,
    #[arg(long, default_value = "ffn-kernel-sweep")]
    pub label: String,
    #[arg(long, default_value_t = 20)]
    pub harness_timeout_s: u64,
    #[arg(long, default_value_t = false)]
    pub skip_baseline: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnCandidate {
    pub name: String,
    pub batched_gemm_strategy: Option<String>,
    pub cutlass_gate_aux: Option<bool>,
    pub requires_cutlass_gate_harness: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateBuildConfig {
    pub name: String,
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub cluster_m: usize,
    pub cluster_n: usize,
    pub cluster_k: usize,
    pub cooperative: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessResult {
    pub m: usize,
    pub ok: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub tok_per_sec: f64,
    pub elapsed_ms: u64,
    pub total_tokens: u64,
    pub failed: u64,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepRecord {
    pub timestamp_s: u64,
    pub commit: Option<String>,
    pub hostname: Option<String>,
    pub gpu_name: Option<String>,
    pub driver_version: Option<String>,
    pub candidate: FfnCandidate,
    pub gate_build: Option<GateBuildConfig>,
    pub n: usize,
    pub output_len: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub harness: Option<HarnessResult>,
    pub benchmark: Option<BenchmarkResult>,
    pub skipped_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinnerSummary {
    pub n: usize,
    pub winner: Option<FfnCandidate>,
    pub gate_build: Option<GateBuildConfig>,
    pub tok_per_sec: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepSummary {
    pub timestamp_s: u64,
    pub label: String,
    pub commit: Option<String>,
    pub hostname: Option<String>,
    pub gpu_name: Option<String>,
    pub driver_version: Option<String>,
    pub rvllm_bin: String,
    pub model: String,
    pub output_len: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub shapes: Vec<usize>,
    pub records_path: String,
    pub winners: Vec<WinnerSummary>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkJson {
    results: Vec<BenchmarkJsonRow>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkJsonRow {
    elapsed_ms: u64,
    failed: u64,
    n: usize,
    tok_per_sec: f64,
    total_tokens: u64,
}

pub fn run_ffn_sweep(args: &FfnSweepArgs) -> Result<(), String> {
    let shapes = parse_csv_usize(&args.n)?;
    if shapes.is_empty() {
        return Err("shape list is empty".into());
    }

    let out_dir = args
        .out_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("results").join("autotune").join(format!("{}-{}", args.label, now_s())));
    fs::create_dir_all(&out_dir).map_err(|e| format!("create {}: {e}", out_dir.display()))?;
    let records_path = out_dir.join("ffn_sweep.jsonl");
    let summary_path = out_dir.join("summary.json");

    let metadata = probe_metadata();
    let candidates = default_candidates();
    let mut harness_by_shape = BTreeMap::new();

    if !args.skip_harness {
        if let (Some(harness_bin), Some(cutlass_so)) = (&args.harness_bin, &args.cutlass_so) {
            for &m in &shapes {
                let harness = run_harness(
                    harness_bin,
                    cutlass_so,
                    m,
                    args.hidden,
                    args.intermediate,
                    args.harness_timeout_s,
                )?;
                harness_by_shape.insert(m, harness);
            }
        }
    }

    let mut winners = Vec::new();
    for &n in &shapes {
        let harness = harness_by_shape.get(&n).cloned();
        let mut best: Option<(FfnCandidate, f64)> = None;

        for candidate in &candidates {
            let skipped_reason = if candidate.requires_cutlass_gate_harness {
                match harness.as_ref() {
                    Some(h) if !h.ok => Some("cutlass_harness_failed".to_string()),
                    None if !args.skip_harness => Some("cutlass_harness_missing".to_string()),
                    _ => None,
                }
            } else {
                None
            };

            let benchmark = if skipped_reason.is_none() {
                Some(run_benchmark(args, candidate, n)?)
            } else {
                None
            };

            if let Some(bench) = benchmark.as_ref() {
                if bench.failed == 0 {
                    match best {
                        Some((_, current)) if current >= bench.tok_per_sec => {}
                        _ => best = Some((candidate.clone(), bench.tok_per_sec)),
                    }
                }
            }

            let record = SweepRecord {
                timestamp_s: now_s(),
                commit: metadata.commit.clone(),
                hostname: metadata.hostname.clone(),
                gpu_name: metadata.gpu_name.clone(),
                driver_version: metadata.driver_version.clone(),
                candidate: candidate.clone(),
                gate_build: None,
                n,
                output_len: args.output_len,
                hidden: args.hidden,
                intermediate: args.intermediate,
                harness: harness.clone(),
                benchmark,
                skipped_reason,
            };
            append_jsonl(&records_path, &record)?;
        }

        winners.push(WinnerSummary {
            n,
            winner: best.as_ref().map(|(candidate, _)| candidate.clone()),
            gate_build: None,
            tok_per_sec: best.as_ref().map(|(_, tok_per_sec)| *tok_per_sec),
        });
    }

    let summary = SweepSummary {
        timestamp_s: now_s(),
        label: args.label.clone(),
        commit: metadata.commit,
        hostname: metadata.hostname,
        gpu_name: metadata.gpu_name,
        driver_version: metadata.driver_version,
        rvllm_bin: args.rvllm_bin.display().to_string(),
        model: args.model.clone(),
        output_len: args.output_len,
        hidden: args.hidden,
        intermediate: args.intermediate,
        shapes,
        records_path: records_path.display().to_string(),
        winners,
    };
    let summary_json =
        serde_json::to_string_pretty(&summary).map_err(|e| format!("summary json: {e}"))?;
    fs::write(&summary_path, summary_json)
        .map_err(|e| format!("write {}: {e}", summary_path.display()))?;

    println!("{}", summary_path.display());
    Ok(())
}

pub fn run_ffn_kernel_sweep(args: &FfnKernelSweepArgs) -> Result<(), String> {
    let shapes = parse_csv_usize(&args.n)?;
    if shapes.is_empty() {
        return Err("shape list is empty".into());
    }

    let out_dir = args
        .out_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("results").join("autotune").join(format!("{}-{}", args.label, now_s())));
    fs::create_dir_all(&out_dir).map_err(|e| format!("create {}: {e}", out_dir.display()))?;
    let records_path = out_dir.join("ffn_sweep.jsonl");
    let summary_path = out_dir.join("summary.json");

    let metadata = probe_metadata();
    let cutlass_so = built_cutlass_so_path(&args.build_script, &args.arch)?;
    let baseline = FfnCandidate {
        name: "default".into(),
        batched_gemm_strategy: None,
        cutlass_gate_aux: None,
        requires_cutlass_gate_harness: false,
    };
    let gate_candidate = FfnCandidate {
        name: "hybrid_gate_aux_on".into(),
        batched_gemm_strategy: Some("hybrid".into()),
        cutlass_gate_aux: Some(true),
        requires_cutlass_gate_harness: true,
    };
    let configs = default_gate_build_configs();
    let mut winners: Vec<Option<(FfnCandidate, Option<GateBuildConfig>, f64)>> =
        vec![None; shapes.len()];

    if !args.skip_baseline {
        for (idx, &n) in shapes.iter().enumerate() {
            let benchmark = run_benchmark_inner(
                &args.rvllm_bin,
                &args.model,
                args.output_len,
                &baseline,
                n,
            )?;
            let tok = benchmark.tok_per_sec;
            winners[idx] = Some((baseline.clone(), None, tok));
            let record = SweepRecord {
                timestamp_s: now_s(),
                commit: metadata.commit.clone(),
                hostname: metadata.hostname.clone(),
                gpu_name: metadata.gpu_name.clone(),
                driver_version: metadata.driver_version.clone(),
                candidate: baseline.clone(),
                gate_build: None,
                n,
                output_len: args.output_len,
                hidden: args.hidden,
                intermediate: args.intermediate,
                harness: None,
                benchmark: Some(benchmark),
                skipped_reason: None,
            };
            append_jsonl(&records_path, &record)?;
        }
    }

    for config in &configs {
        build_gate_variant(&args.build_script, &args.arch, &args.cutlass_dir, config)?;

        for (idx, &n) in shapes.iter().enumerate() {
            let harness = run_harness(
                &args.harness_bin,
                &cutlass_so,
                n,
                args.hidden,
                args.intermediate,
                args.harness_timeout_s,
            )?;
            let skipped_reason = if harness.ok {
                None
            } else {
                Some("cutlass_harness_failed".to_string())
            };
            let benchmark = if skipped_reason.is_none() {
                Some(run_benchmark_inner(
                    &args.rvllm_bin,
                    &args.model,
                    args.output_len,
                    &gate_candidate,
                    n,
                )?)
            } else {
                None
            };

            if let Some(bench) = benchmark.as_ref() {
                if bench.failed == 0 {
                    match winners[idx].as_ref() {
                        Some((_, _, current)) if *current >= bench.tok_per_sec => {}
                        _ => {
                            winners[idx] =
                                Some((gate_candidate.clone(), Some(config.clone()), bench.tok_per_sec))
                        }
                    }
                }
            }

            let record = SweepRecord {
                timestamp_s: now_s(),
                commit: metadata.commit.clone(),
                hostname: metadata.hostname.clone(),
                gpu_name: metadata.gpu_name.clone(),
                driver_version: metadata.driver_version.clone(),
                candidate: gate_candidate.clone(),
                gate_build: Some(config.clone()),
                n,
                output_len: args.output_len,
                hidden: args.hidden,
                intermediate: args.intermediate,
                harness: Some(harness),
                benchmark,
                skipped_reason,
            };
            append_jsonl(&records_path, &record)?;
        }
    }

    let summary = SweepSummary {
        timestamp_s: now_s(),
        label: args.label.clone(),
        commit: metadata.commit,
        hostname: metadata.hostname,
        gpu_name: metadata.gpu_name,
        driver_version: metadata.driver_version,
        rvllm_bin: args.rvllm_bin.display().to_string(),
        model: args.model.clone(),
        output_len: args.output_len,
        hidden: args.hidden,
        intermediate: args.intermediate,
        shapes: shapes.clone(),
        records_path: records_path.display().to_string(),
        winners: shapes
            .into_iter()
            .enumerate()
            .map(|(idx, n)| WinnerSummary {
                n,
                winner: winners[idx].as_ref().map(|(candidate, _, _)| candidate.clone()),
                gate_build: winners[idx].as_ref().and_then(|(_, gate_build, _)| gate_build.clone()),
                tok_per_sec: winners[idx].as_ref().map(|(_, _, tok_per_sec)| *tok_per_sec),
            })
            .collect(),
    };
    let summary_json =
        serde_json::to_string_pretty(&summary).map_err(|e| format!("summary json: {e}"))?;
    fs::write(&summary_path, summary_json)
        .map_err(|e| format!("write {}: {e}", summary_path.display()))?;

    println!("{}", summary_path.display());
    Ok(())
}

fn run_benchmark(
    args: &FfnSweepArgs,
    candidate: &FfnCandidate,
    n: usize,
) -> Result<BenchmarkResult, String> {
    run_benchmark_inner(
        &args.rvllm_bin,
        &args.model,
        args.output_len,
        candidate,
        n,
    )
}

fn run_benchmark_inner(
    rvllm_bin: &Path,
    model: &str,
    output_len: usize,
    candidate: &FfnCandidate,
    n: usize,
) -> Result<BenchmarkResult, String> {
    let mut cmd = Command::new(rvllm_bin);
    cmd.arg("benchmark")
        .arg("--model")
        .arg(model)
        .arg("--n")
        .arg(n.to_string())
        .arg("--output-len")
        .arg(output_len.to_string())
        .arg("--json");
    cmd.env_remove("RVLLM_PHASE_PROFILE_BATCHES");
    cmd.env_remove("RVLLM_BATCHED_GEMM_STRATEGY");
    cmd.env_remove("RVLLM_CUTLASS_GATE_AUX");
    if let Some(strategy) = &candidate.batched_gemm_strategy {
        cmd.env("RVLLM_BATCHED_GEMM_STRATEGY", strategy);
    }
    if let Some(gate_aux) = candidate.cutlass_gate_aux {
        cmd.env("RVLLM_CUTLASS_GATE_AUX", if gate_aux { "1" } else { "0" });
    }

    let output = cmd.output().map_err(|e| {
        format!(
            "spawn benchmark {} for n={n}: {e}",
            rvllm_bin.display()
        )
    })?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() {
        return Err(format!(
            "benchmark failed for candidate={} n={n}: status={:?}\n{}",
            candidate.name,
            output.status.code(),
            stderr
        ));
    }
    let parsed: BenchmarkJson =
        serde_json::from_str(&stdout).map_err(|e| format!("parse benchmark json: {e}\n{stdout}"))?;
    let row = parsed
        .results
        .into_iter()
        .find(|row| row.n == n)
        .ok_or_else(|| format!("missing n={n} result in benchmark json"))?;
    Ok(BenchmarkResult {
        tok_per_sec: row.tok_per_sec,
        elapsed_ms: row.elapsed_ms,
        total_tokens: row.total_tokens,
        failed: row.failed,
        stdout,
        stderr,
    })
}

fn run_harness(
    harness_bin: &Path,
    cutlass_so: &Path,
    m: usize,
    hidden: usize,
    intermediate: usize,
    timeout_s: u64,
) -> Result<HarnessResult, String> {
    let mut child = Command::new(harness_bin)
        .arg(cutlass_so)
        .arg(m.to_string())
        .arg(hidden.to_string())
        .arg(intermediate.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn harness {}: {e}", harness_bin.display()))?;

    let timeout = Duration::from_secs(timeout_s);
    let started = std::time::Instant::now();
    let output = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|e| format!("poll harness {}: {e}", harness_bin.display()))?
        {
            let out = child
                .wait_with_output()
                .map_err(|e| format!("collect harness output: {e}"))?;
            break (status.code(), out.stdout, out.stderr);
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let out = child
                .wait_with_output()
                .map_err(|e| format!("collect timed out harness output: {e}"))?;
            break (None, out.stdout, out.stderr);
        }
        sleep(Duration::from_millis(100));
    };
    let (exit_code, stdout_bytes, stderr_bytes) = output;
    let stdout = String::from_utf8_lossy(&stdout_bytes).into_owned();
    let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();
    Ok(HarnessResult {
        m,
        ok: exit_code == Some(0),
        exit_code,
        stdout,
        stderr,
    })
}

fn append_jsonl(path: &Path, record: &SweepRecord) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let line = serde_json::to_string(record).map_err(|e| format!("record json: {e}"))?;
    file.write_all(line.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .map_err(|e| format!("append {}: {e}", path.display()))
}

fn default_candidates() -> Vec<FfnCandidate> {
    vec![
        FfnCandidate {
            name: "default".into(),
            batched_gemm_strategy: None,
            cutlass_gate_aux: None,
            requires_cutlass_gate_harness: false,
        },
        FfnCandidate {
            name: "cublas".into(),
            batched_gemm_strategy: Some("cublas".into()),
            cutlass_gate_aux: None,
            requires_cutlass_gate_harness: false,
        },
        FfnCandidate {
            name: "hybrid_gate_aux_on".into(),
            batched_gemm_strategy: Some("hybrid".into()),
            cutlass_gate_aux: Some(true),
            requires_cutlass_gate_harness: true,
        },
        FfnCandidate {
            name: "hybrid_gate_aux_off".into(),
            batched_gemm_strategy: Some("hybrid".into()),
            cutlass_gate_aux: Some(false),
            requires_cutlass_gate_harness: false,
        },
        FfnCandidate {
            name: "cutlass_gate_aux_off".into(),
            batched_gemm_strategy: Some("cutlass".into()),
            cutlass_gate_aux: Some(false),
            requires_cutlass_gate_harness: false,
        },
    ]
}

fn default_gate_build_configs() -> Vec<GateBuildConfig> {
    vec![
        GateBuildConfig {
            name: "tile128x256x64_cluster1x2x1_ws".into(),
            tile_m: 128,
            tile_n: 256,
            tile_k: 64,
            cluster_m: 1,
            cluster_n: 2,
            cluster_k: 1,
            cooperative: false,
        },
        GateBuildConfig {
            name: "tile128x256x64_cluster1x2x1_coop".into(),
            tile_m: 128,
            tile_n: 256,
            tile_k: 64,
            cluster_m: 1,
            cluster_n: 2,
            cluster_k: 1,
            cooperative: true,
        },
        GateBuildConfig {
            name: "tile128x128x64_cluster1x2x1_ws".into(),
            tile_m: 128,
            tile_n: 128,
            tile_k: 64,
            cluster_m: 1,
            cluster_n: 2,
            cluster_k: 1,
            cooperative: false,
        },
        GateBuildConfig {
            name: "tile128x128x64_cluster1x1x1_ws".into(),
            tile_m: 128,
            tile_n: 128,
            tile_k: 64,
            cluster_m: 1,
            cluster_n: 1,
            cluster_k: 1,
            cooperative: false,
        },
        GateBuildConfig {
            name: "tile64x256x64_cluster1x2x1_ws".into(),
            tile_m: 64,
            tile_n: 256,
            tile_k: 64,
            cluster_m: 1,
            cluster_n: 2,
            cluster_k: 1,
            cooperative: false,
        },
        GateBuildConfig {
            name: "tile64x128x64_cluster1x1x1_ws".into(),
            tile_m: 64,
            tile_n: 128,
            tile_k: 64,
            cluster_m: 1,
            cluster_n: 1,
            cluster_k: 1,
            cooperative: false,
        },
    ]
}

fn parse_csv_usize(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid integer '{s}': {e}"))
        })
        .collect()
}

#[derive(Default)]
struct RunMetadata {
    commit: Option<String>,
    hostname: Option<String>,
    gpu_name: Option<String>,
    driver_version: Option<String>,
}

fn probe_metadata() -> RunMetadata {
    let commit = probe_command(&["git", "rev-parse", "HEAD"]);
    let hostname = probe_command(&["hostname"]);
    let gpu_csv = probe_command(&[
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]);
    let (gpu_name, driver_version) = match gpu_csv {
        Some(line) => {
            let mut parts = line.split(',').map(|part| part.trim().to_string());
            (parts.next(), parts.next())
        }
        None => (None, None),
    };
    RunMetadata {
        commit,
        hostname,
        gpu_name,
        driver_version,
    }
}

fn probe_command(argv: &[&str]) -> Option<String> {
    let (program, args) = argv.split_first()?;
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn now_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn built_cutlass_so_path(build_script: &Path, arch: &str) -> Result<PathBuf, String> {
    let kernel_dir = build_script
        .parent()
        .ok_or_else(|| format!("build script has no parent: {}", build_script.display()))?;
    Ok(kernel_dir.join(arch).join("libcutlass_kernels.so"))
}

fn build_gate_variant(
    build_script: &Path,
    arch: &str,
    cutlass_dir: &Path,
    config: &GateBuildConfig,
) -> Result<(), String> {
    let output = Command::new(build_script)
        .arg(arch)
        .arg(cutlass_dir)
        .env("RVLLM_CUTLASS_GATE_ONLY", "1")
        .env("RVLLM_CUTLASS_GATE_SOURCE_GEN", "1")
        .env("RVLLM_CUTLASS_GATE_TILE_M", config.tile_m.to_string())
        .env("RVLLM_CUTLASS_GATE_TILE_N", config.tile_n.to_string())
        .env("RVLLM_CUTLASS_GATE_TILE_K", config.tile_k.to_string())
        .env("RVLLM_CUTLASS_GATE_CLUSTER_M", config.cluster_m.to_string())
        .env("RVLLM_CUTLASS_GATE_CLUSTER_N", config.cluster_n.to_string())
        .env("RVLLM_CUTLASS_GATE_CLUSTER_K", config.cluster_k.to_string())
        .env(
            "RVLLM_CUTLASS_GATE_SCHEDULE",
            if config.cooperative { "1" } else { "0" },
        )
        .output()
        .map_err(|e| format!("spawn {} for {}: {e}", build_script.display(), config.name))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "build {} failed: status={:?}\nstdout:\n{}\nstderr:\n{}",
            config.name,
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}
