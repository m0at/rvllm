use std::env;
use std::fs;
use std::path::PathBuf;

use rvllm_xla::m2_decode_bench::{
    check_m2_rust_decode_bench_json, M2_DEFAULT_DECODE_BENCH_BATCHES,
};
use rvllm_xla::m2_runtime::{m2_decode_mlir_execution_blocker, M2RuntimeMode};
use rvllm_xla::{
    m2_decode_graph_mlir, plan_m2_rust_decode_bench, M2GraphAbi, M2GraphShape,
    M2RustDecodeBenchConfig, M2WeightUploadPlan, PjrtElementType, XlaArtifact, XlaTensorSpec,
    M2_VOCAB,
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

    let artifacts =
        if args.emit_decode_artifacts || args.runtime_mode != M2RuntimeMode::PlanningOnly {
            write_decode_artifacts(&args)?
        } else {
            Vec::new()
        };
    match args.runtime_mode {
        M2RuntimeMode::PlanningOnly => {}
        M2RuntimeMode::CompileOnly => {
            compile_decode_artifacts(&artifacts)?;
            for item in &mut report.sweep {
                item.status = "compiled";
                item.error = None;
            }
        }
        M2RuntimeMode::Execute => {
            if let Some(reason) = artifacts
                .iter()
                .find_map(|artifact| m2_decode_mlir_execution_blocker(&artifact.mlir))
            {
                return Err(reason.into());
            }
            return Err("m2_rust_decode_bench execute mode is not wired until decode custom-call bodies are linked".into());
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
}

fn write_decode_artifacts(args: &Args) -> Result<Vec<DecodeArtifact>, Box<dyn std::error::Error>> {
    fs::create_dir_all(&args.artifact_dir)?;
    let mut out = Vec::with_capacity(args.batches.len());
    for &batch in &args.batches {
        let shape = M2GraphShape::decode(batch, args.ctx, kv_bytes_per_elem(&args.kv_cache)?);
        let abi = M2GraphAbi::new(shape.clone())?;
        let weights = M2WeightUploadPlan::from_index_dir(&args.model_dir, &abi)?;
        let arena = weights.flat_arena(128)?;
        let mlir = m2_decode_graph_mlir("main", &shape, &arena)?;
        let mlir_name = format!("m2_decode_b{batch}.mlir");
        let json_name = format!("m2_decode_b{batch}.json");
        fs::write(args.artifact_dir.join(&mlir_name), mlir.as_bytes())?;
        let artifact = decode_artifact_manifest(&mlir_name, &shape, arena.total_bytes);
        fs::write(
            args.artifact_dir.join(&json_name),
            serde_json::to_vec_pretty(&artifact)?,
        )?;
        eprintln!(
            "wrote decode artifact {} and {}",
            args.artifact_dir.join(&mlir_name).display(),
            args.artifact_dir.join(&json_name).display()
        );
        out.push(DecodeArtifact { mlir });
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
        let _exe = client.compile(&artifact.mlir)?;
        eprintln!("compiled decode MLIR through PJRT");
    }
    Ok(())
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
    emit_decode_artifacts: bool,
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
            emit_decode_artifacts: false,
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
                "--emit-decode-artifacts" => out.emit_decode_artifacts = true,
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
    "usage: m2_rust_decode_bench [--model-dir DIR] [--artifact-dir DIR] [--out JSON] [--check JSON] [--runtime-mode planning-only|compile-only|execute] [--emit-decode-artifacts] [--compile-decode] [--execute-decode] [--batches 1,8,16,32|--batch N] [--ctx N] [--iters N] [--warmup N] [--kv-cache int8|bf16] [--moe-impl NAME] [--ppl-text FILE] [--prompt TEXT] [--gen-tokens N]".to_string()
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
    fn execute_decode_alias_selects_execute_mode() {
        let args = Args::parse(vec!["--execute-decode".to_string()]).unwrap();
        assert_eq!(args.runtime_mode, M2RuntimeMode::Execute);
    }
}
