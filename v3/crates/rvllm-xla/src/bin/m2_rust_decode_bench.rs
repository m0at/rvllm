use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rvllm_xla::m2_decode_bench::{
    check_m2_rust_decode_bench_json, m2_rust_decode_timing, M2RustDecodeExecutedTiming,
    M2RustDecodeTimingSource, M2_DEFAULT_DECODE_BENCH_BATCHES,
};
use rvllm_xla::m2_runtime::{m2_decode_mlir_execution_blocker, M2RuntimeMode};
use rvllm_xla::{
    m2_decode_graph_mlir_with_mosaic_body, m2_decode_smoke_mlir, plan_m2_rust_decode_bench,
    M2GraphAbi, M2GraphShape, M2RustDecodeBenchConfig, M2WeightUploadPlan, PjrtElementType,
    TpuMosaicSerializedBody, XlaArtifact, XlaTensorSpec, M2_VOCAB,
};

#[cfg(feature = "tpu")]
use rvllm_xla::PjrtClientHandle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    if let Some(path) = args.check {
        let bytes = fs::read(&path)?;
        let json: serde_json::Value = serde_json::from_slice(&bytes)?;
        check_m2_rust_decode_bench_json(&json)?;
        eprintln!("checked {}", path.display());
        return Ok(());
    }
    if args.runtime_mode != M2RuntimeMode::PlanningOnly
        && !args.native_smoke
        && args.decode_layer_body.is_none()
    {
        return Err("real Rust M2 compile/execute needs --decode-layer-body FILE containing serde bytecode or lowered MLIR; use --native-smoke for the zero-logits PJRT smoke".into());
    }
    let artifacts =
        if args.emit_decode_artifacts || args.runtime_mode != M2RuntimeMode::PlanningOnly {
            write_decode_artifacts(&args)?
        } else {
            Vec::new()
        };
    let mut compiled = false;
    let timings = match args.runtime_mode {
        M2RuntimeMode::PlanningOnly => Vec::new(),
        M2RuntimeMode::CompileOnly => {
            compile_decode_artifacts(&artifacts)?;
            compiled = true;
            Vec::new()
        }
        M2RuntimeMode::Execute => {
            if let Some(reason) = artifacts
                .iter()
                .find_map(|artifact| m2_decode_mlir_execution_blocker(&artifact.mlir))
            {
                return Err(reason.into());
            }
            execute_decode_artifacts(&artifacts, &args)?
        }
    };

    let mut report = plan_m2_rust_decode_bench(&M2RustDecodeBenchConfig {
        model_dir: args.model_dir.clone(),
        batches: args.batches.clone(),
        ctx: args.ctx,
        iters: args.iters,
        warmup: args.warmup,
        kv_cache: args.kv_cache.clone(),
        moe_impl: args.moe_impl.clone(),
        artifact_dir: args.artifact_dir.clone(),
        report_path: args.out.clone(),
        ppl_text_path: args.ppl_text.clone(),
        prompt: args.prompt.clone(),
        gen_tokens: args.gen_tokens,
    })?;
    if compiled {
        for item in &mut report.sweep {
            item.status = "compiled";
            item.error = None;
        }
    }
    if !timings.is_empty() {
        for (item, timing) in report.sweep.iter_mut().zip(timings) {
            item.status = "executed";
            item.error = None;
            item.timing = Some(timing);
        }
    }

    let json = serde_json::to_vec_pretty(&report)?;
    if let Some(out) = args.out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, &json)?;
        eprintln!("wrote {}", out.display());
    } else {
        println!("{}", String::from_utf8(json)?);
    }
    Ok(())
}

struct DecodeArtifact {
    mlir: String,
    #[cfg_attr(not(feature = "tpu"), allow(dead_code))]
    mlir_path: PathBuf,
    batch: usize,
    shape: M2GraphShape,
    weight_arena_bytes: usize,
}

fn write_decode_artifacts(args: &Args) -> Result<Vec<DecodeArtifact>, Box<dyn std::error::Error>> {
    fs::create_dir_all(&args.artifact_dir)?;
    let decode_layer_body = match &args.decode_layer_body {
        Some(path) => {
            let bytes = fs::read(path)?;
            let body = match args.decode_layer_body_format.as_str() {
                "serde" | "serialized" | "bytecode" => {
                    TpuMosaicSerializedBody::from_serialized_bytecode(&bytes)?
                }
                "lowered" | "raw" | "mlir" => TpuMosaicSerializedBody::from_lowered_mlir(&bytes)?,
                other => {
                    return Err(format!(
                        "--decode-layer-body-format: expected serde|lowered, got {other:?}"
                    )
                    .into());
                }
            };
            eprintln!(
                "linked {} Mosaic decode-layer body {} ({} bytes)",
                args.decode_layer_body_format,
                path.display(),
                body.byte_len()
            );
            Some(body)
        }
        None => None,
    };
    let mut out = Vec::with_capacity(args.batches.len());
    for &batch in &args.batches {
        let shape = M2GraphShape::decode(batch, args.ctx, kv_bytes_per_elem(&args.kv_cache)?);
        let weight_arena_bytes;
        let mlir = if args.native_smoke {
            weight_arena_bytes = 1;
            m2_decode_smoke_mlir("main", &shape)?
        } else {
            let abi = M2GraphAbi::new(shape.clone())?;
            let weights = M2WeightUploadPlan::from_index_dir(&args.model_dir, &abi)?;
            let arena = weights.flat_arena(128)?;
            weight_arena_bytes = arena.total_bytes.div_ceil(8);
            m2_decode_graph_mlir_with_mosaic_body(
                "main",
                &shape,
                &arena,
                decode_layer_body.as_ref(),
            )?
        };
        let mlir_name = format!("m2_decode_b{batch}.mlir");
        let json_name = format!("m2_decode_b{batch}.json");
        let mlir_path = args.artifact_dir.join(&mlir_name);
        fs::write(&mlir_path, mlir.as_bytes())?;
        let artifact = decode_artifact_manifest(&mlir_name, &shape, weight_arena_bytes);
        fs::write(
            args.artifact_dir.join(&json_name),
            serde_json::to_vec_pretty(&artifact)?,
        )?;
        eprintln!(
            "wrote decode artifact {} and {}",
            args.artifact_dir.join(&mlir_name).display(),
            args.artifact_dir.join(&json_name).display()
        );
        out.push(DecodeArtifact {
            mlir,
            mlir_path,
            batch,
            shape,
            weight_arena_bytes,
        });
    }
    Ok(out)
}

fn decode_artifact_manifest(
    mlir_file: &str,
    shape: &M2GraphShape,
    weight_arena_bytes: usize,
) -> XlaArtifact {
    XlaArtifact {
        mlir_file: mlir_file.to_string(),
        inputs: vec![
            tensor("token_ids", &[shape.batch], PjrtElementType::S32),
            tensor("positions", &[shape.batch], PjrtElementType::S32),
            tensor("kv_cache", &[shape.kv_cache_bytes()], PjrtElementType::S8),
            tensor("weight_arena", &[weight_arena_bytes], PjrtElementType::S8),
        ],
        outputs: vec![
            tensor("logits", &[shape.batch, M2_VOCAB], PjrtElementType::BF16),
            tensor("next_token", &[shape.batch], PjrtElementType::S32),
            tensor("kv_cache", &[shape.kv_cache_bytes()], PjrtElementType::S8),
        ],
        donate_indices: vec![2],
        num_partitions: 8,
    }
}

fn tensor(name: &str, shape: &[usize], dtype: PjrtElementType) -> XlaTensorSpec {
    XlaTensorSpec {
        name: name.to_string(),
        shape: shape.iter().map(|&x| x as i64).collect(),
        dtype,
    }
}

#[cfg(feature = "tpu")]
fn compile_decode_artifacts(
    artifacts: &[DecodeArtifact],
) -> Result<(), Box<dyn std::error::Error>> {
    let client = PjrtClientHandle::new()?;
    for artifact in artifacts {
        let mlir = fs::read_to_string(&artifact.mlir_path)?;
        let _exe = client.compile(&mlir)?;
        eprintln!("compiled decode MLIR through PJRT");
    }
    Ok(())
}

#[cfg(feature = "tpu")]
fn execute_decode_artifacts(
    artifacts: &[DecodeArtifact],
    args: &Args,
) -> Result<Vec<M2RustDecodeExecutedTiming>, Box<dyn std::error::Error>> {
    let client = PjrtClientHandle::new()?;
    let mut out = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        let mlir = fs::read_to_string(&artifact.mlir_path)?;
        let exe = client.compile(&mlir)?;
        let token_buf = client.buffer_from_host(
            &vec![0u8; artifact.shape.batch * 4],
            &[artifact.shape.batch as i64],
            PjrtElementType::S32,
            0,
        )?;
        let pos_buf = client.buffer_from_host(
            &vec![0u8; artifact.shape.batch * 4],
            &[artifact.shape.batch as i64],
            PjrtElementType::S32,
            0,
        )?;
        let mut kv_buf = client.buffer_from_host(
            &vec![0u8; artifact.shape.kv_cache_bytes()],
            &[artifact.shape.kv_cache_bytes() as i64],
            PjrtElementType::S8,
            0,
        )?;
        let weight_arena = client.buffer_from_host(
            &vec![0u8; artifact.weight_arena_bytes],
            &[artifact.weight_arena_bytes as i64],
            PjrtElementType::S8,
            0,
        )?;
        let mut samples = Vec::with_capacity(args.iters);
        for step in 0..(args.warmup + args.iters) {
            let start = Instant::now();
            let inputs = [&token_buf, &pos_buf, &kv_buf, &weight_arena];
            let mut outputs = client.execute(&exe, &inputs)?;
            if outputs.len() != 3 {
                return Err(
                    format!("decode returned {} outputs, expected 3", outputs.len()).into(),
                );
            }
            let next_token = outputs.remove(1);
            let new_kv = outputs.remove(1);
            let mut next_bytes = vec![0u8; artifact.shape.batch * 4];
            client.buffer_to_host(&next_token, &mut next_bytes)?;
            if step >= args.warmup {
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            kv_buf = new_kv;
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        let ms_min = *samples.first().ok_or("no timing samples")?;
        let ms_max = *samples.last().ok_or("no timing samples")?;
        let ms_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let ms_p50 = samples[samples.len() / 2];
        let timing = m2_rust_decode_timing(
            M2RustDecodeTimingSource {
                runtime: "rust_xla",
                executed: true,
                device: "tpu:0".to_string(),
                executable: "pjrt_mlir_decode".to_string(),
            },
            artifact.batch,
            ms_min,
            ms_mean,
            ms_max,
            ms_p50,
        )?;
        eprintln!(
            "executed B={} mean={:.3} ms tok/s={:.2}",
            artifact.batch,
            ms_mean,
            1000.0 * artifact.batch as f64 / ms_mean
        );
        out.push(timing);
    }
    Ok(out)
}

#[cfg(not(feature = "tpu"))]
fn execute_decode_artifacts(
    _artifacts: &[DecodeArtifact],
    _args: &Args,
) -> Result<Vec<M2RustDecodeExecutedTiming>, Box<dyn std::error::Error>> {
    Err("m2_rust_decode_bench execute mode requires --features tpu".into())
}

#[cfg(not(feature = "tpu"))]
fn compile_decode_artifacts(
    _artifacts: &[DecodeArtifact],
) -> Result<(), Box<dyn std::error::Error>> {
    Err("m2_rust_decode_bench compile-only requires --features tpu".into())
}

struct Args {
    model_dir: PathBuf,
    artifact_dir: PathBuf,
    out: Option<PathBuf>,
    check: Option<PathBuf>,
    decode_layer_body: Option<PathBuf>,
    decode_layer_body_format: String,
    emit_decode_artifacts: bool,
    native_smoke: bool,
    runtime_mode: M2RuntimeMode,
    batches: Vec<usize>,
    ctx: usize,
    iters: usize,
    warmup: usize,
    kv_cache: String,
    moe_impl: String,
    ppl_text: Option<PathBuf>,
    prompt: String,
    gen_tokens: usize,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            model_dir: PathBuf::from("/dev/shm/m2-nvfp4"),
            artifact_dir: PathBuf::from("tpu/out/m2/rust_xla_plan"),
            out: None,
            check: None,
            decode_layer_body: None,
            decode_layer_body_format: "serde".to_string(),
            emit_decode_artifacts: false,
            native_smoke: false,
            runtime_mode: M2RuntimeMode::PlanningOnly,
            batches: M2_DEFAULT_DECODE_BENCH_BATCHES.to_vec(),
            ctx: 2048,
            iters: 10,
            warmup: 3,
            kv_cache: "int8".to_string(),
            moe_impl: "auto".to_string(),
            ppl_text: None,
            prompt: "Explain angular momentum.".to_string(),
            gen_tokens: 256,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--model-dir" => {
                    i += 1;
                    out.model_dir = PathBuf::from(value(&args, i, "--model-dir")?);
                }
                "--artifact-dir" => {
                    i += 1;
                    out.artifact_dir = PathBuf::from(value(&args, i, "--artifact-dir")?);
                }
                "--out" => {
                    i += 1;
                    out.out = Some(PathBuf::from(value(&args, i, "--out")?));
                }
                "--check" => {
                    i += 1;
                    out.check = Some(PathBuf::from(value(&args, i, "--check")?));
                }
                "--decode-layer-body" => {
                    i += 1;
                    out.decode_layer_body =
                        Some(PathBuf::from(value(&args, i, "--decode-layer-body")?));
                }
                "--decode-layer-body-format" => {
                    i += 1;
                    out.decode_layer_body_format =
                        value(&args, i, "--decode-layer-body-format")?.to_string();
                }
                "--emit-decode-artifacts" => out.emit_decode_artifacts = true,
                "--native-smoke" => {
                    out.native_smoke = true;
                    out.emit_decode_artifacts = true;
                }
                "--runtime-mode" => {
                    i += 1;
                    out.runtime_mode = value(&args, i, "--runtime-mode")?
                        .parse::<M2RuntimeMode>()
                        .map_err(|e| format!("--runtime-mode: {e}"))?;
                }
                "--compile-decode" => out.runtime_mode = M2RuntimeMode::CompileOnly,
                "--execute-decode" => out.runtime_mode = M2RuntimeMode::Execute,
                "--batch" => {
                    i += 1;
                    out.batches = vec![parse(value(&args, i, "--batch")?, "--batch")?];
                }
                "--batches" => {
                    i += 1;
                    out.batches = parse_batches(value(&args, i, "--batches")?)?;
                }
                "--ctx" => {
                    i += 1;
                    out.ctx = parse(value(&args, i, "--ctx")?, "--ctx")?;
                }
                "--iters" => {
                    i += 1;
                    out.iters = parse(value(&args, i, "--iters")?, "--iters")?;
                }
                "--warmup" => {
                    i += 1;
                    out.warmup = parse(value(&args, i, "--warmup")?, "--warmup")?;
                }
                "--kv-cache" => {
                    i += 1;
                    out.kv_cache = value(&args, i, "--kv-cache")?.to_string();
                }
                "--moe-impl" => {
                    i += 1;
                    out.moe_impl = value(&args, i, "--moe-impl")?.to_string();
                }
                "--ppl-text" => {
                    i += 1;
                    out.ppl_text = Some(PathBuf::from(value(&args, i, "--ppl-text")?));
                }
                "--prompt" => {
                    i += 1;
                    out.prompt = value(&args, i, "--prompt")?.to_string();
                }
                "--gen-tokens" => {
                    i += 1;
                    out.gen_tokens = parse(value(&args, i, "--gen-tokens")?, "--gen-tokens")?;
                }
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown arg {other:?}\n{}", usage())),
            }
            i += 1;
        }
        Ok(out)
    }
}

fn value<'a>(args: &'a [String], i: usize, name: &str) -> Result<&'a str, String> {
    args.get(i)
        .map(String::as_str)
        .ok_or_else(|| format!("{name}: missing value"))
}

fn parse<T: std::str::FromStr>(s: &str, name: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    s.parse::<T>()
        .map_err(|e| format!("{name}: expected number, got {s:?}: {e}"))
}

fn parse_batches(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err("--batches: empty batch entry".to_string());
        }
        out.push(parse(part, "--batches")?);
    }
    if out.is_empty() {
        return Err("--batches: must not be empty".to_string());
    }
    Ok(out)
}

fn kv_bytes_per_elem(kv_cache: &str) -> Result<usize, String> {
    match kv_cache {
        "int8" => Ok(1),
        "bf16" => Ok(2),
        _ => Err("--kv-cache: expected int8|bf16".to_string()),
    }
}

fn usage() -> String {
    "usage: m2_rust_decode_bench [--model-dir DIR] [--artifact-dir DIR] [--out JSON] [--check JSON] [--decode-layer-body FILE] [--decode-layer-body-format serde|lowered] [--runtime-mode planning-only|compile-only|execute] [--emit-decode-artifacts] [--compile-decode] [--execute-decode] [--batches 1,8,16,32|--batch N] [--ctx N] [--iters N] [--warmup N] [--kv-cache int8|bf16] [--moe-impl NAME] [--ppl-text FILE] [--prompt TEXT] [--gen-tokens N]".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_artifact_compile_mode() {
        let args = Args::parse(vec![
            "--emit-decode-artifacts".to_string(),
            "--compile-decode".to_string(),
            "--batches".to_string(),
            "1,8".to_string(),
        ])
        .unwrap();
        assert!(args.emit_decode_artifacts);
        assert_eq!(args.runtime_mode, M2RuntimeMode::CompileOnly);
        assert_eq!(args.batches, vec![1, 8]);
    }

    #[test]
    fn parses_decode_layer_body_path() {
        let args = Args::parse(vec![
            "--decode-layer-body".to_string(),
            "/tmp/layer.mlirbc".to_string(),
            "--decode-layer-body-format".to_string(),
            "lowered".to_string(),
        ])
        .unwrap();
        assert_eq!(
            args.decode_layer_body,
            Some(PathBuf::from("/tmp/layer.mlirbc"))
        );
        assert_eq!(args.decode_layer_body_format, "lowered");
    }

    #[test]
    fn execute_decode_alias_selects_execute_mode() {
        let args = Args::parse(vec!["--execute-decode".to_string()]).unwrap();
        assert_eq!(args.runtime_mode, M2RuntimeMode::Execute);
    }
}
