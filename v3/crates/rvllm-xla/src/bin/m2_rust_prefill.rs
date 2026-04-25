use std::env;
use std::path::PathBuf;

use rvllm_fused::M2PrefillKvDType;
use rvllm_xla::{plan_m2_rust_prefill, M2RustPrefillConfig};

#[cfg(feature = "tpu")]
use rvllm_xla::{load_artifact, make_m2_prefill_inputs, PjrtClientHandle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    let cfg = M2RustPrefillConfig {
        model_dir: args.model_dir,
        batch: args.batch,
        prompt_len: args.prompt_len,
        ctx: args.ctx,
        block_size: args.block_size,
        kv_dtype: args.kv_dtype,
    };
    let plan = plan_m2_rust_prefill(&cfg)?;
    eprintln!(
        "m2 rust prefill plan: tensors={} nvfp4_groups={} batch={} prompt_len={} ctx={} kv_bytes={} host_input_bytes={}",
        plan.checkpoint.total_tensors,
        plan.checkpoint.nvfp4_groups,
        plan.shape.batch,
        plan.shape.prompt_len,
        plan.shape.ctx,
        plan.shape.kv_cache_bytes(),
        plan.total_host_input_bytes()
    );

    for spec in &plan.input_specs {
        eprintln!(
            "  input {} {:?} {:?} {} bytes",
            spec.name, spec.shape, spec.dtype, spec.nbytes
        );
    }

    if args.execute {
        execute(args.artifact_dir, &plan)?;
    }
    Ok(())
}

#[cfg(feature = "tpu")]
fn execute(
    artifact_dir: Option<PathBuf>,
    plan: &rvllm_xla::M2RustPrefillPlan,
) -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir =
        artifact_dir.unwrap_or_else(|| PathBuf::from("tpu/out/m2/prefill_scan_artifact"));
    let (artifact, mlir, compile_options) = load_artifact(&artifact_dir)?;
    let host_inputs = make_m2_prefill_inputs(&plan.plan, plan.shape)?;
    let mut client = PjrtClientHandle::new()?;
    if let Some(opts) = compile_options {
        client.set_compile_options(opts)?;
    }
    let exe = client.compile_bytes(&mlir)?;
    let buffers = host_inputs
        .iter()
        .map(|input| client.buffer_from_host(&input.bytes, &input.shape, input.dtype, 0))
        .collect::<rvllm_core::Result<Vec<_>>>()?;
    let refs = buffers.iter().collect::<Vec<_>>();
    let outputs = client.execute(&exe, &refs)?;
    eprintln!(
        "m2 rust prefill executed: artifact_inputs={} host_inputs={} outputs={}",
        artifact.inputs.len(),
        host_inputs.len(),
        outputs.len()
    );
    Ok(())
}

#[cfg(not(feature = "tpu"))]
fn execute(
    _artifact_dir: Option<PathBuf>,
    _plan: &rvllm_xla::M2RustPrefillPlan,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("m2_rust_prefill --execute requires --features tpu".into())
}

struct Args {
    model_dir: PathBuf,
    artifact_dir: Option<PathBuf>,
    batch: usize,
    prompt_len: usize,
    ctx: usize,
    block_size: u32,
    kv_dtype: M2PrefillKvDType,
    execute: bool,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            model_dir: PathBuf::from("/dev/shm/m2-nvfp4"),
            artifact_dir: None,
            batch: 8,
            prompt_len: 20,
            ctx: 2048,
            block_size: 32,
            kv_dtype: M2PrefillKvDType::Int8,
            execute: false,
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
                    out.artifact_dir = Some(PathBuf::from(value(&args, i, "--artifact-dir")?));
                }
                "--batch" => {
                    i += 1;
                    out.batch = parse(value(&args, i, "--batch")?, "--batch")?;
                }
                "--prompt-len" => {
                    i += 1;
                    out.prompt_len = parse(value(&args, i, "--prompt-len")?, "--prompt-len")?;
                }
                "--ctx" => {
                    i += 1;
                    out.ctx = parse(value(&args, i, "--ctx")?, "--ctx")?;
                }
                "--block-size" => {
                    i += 1;
                    out.block_size = parse(value(&args, i, "--block-size")?, "--block-size")?;
                }
                "--kv-dtype" => {
                    i += 1;
                    out.kv_dtype = match value(&args, i, "--kv-dtype")? {
                        "int8" => M2PrefillKvDType::Int8,
                        "bf16" => M2PrefillKvDType::Bf16,
                        other => {
                            return Err(format!("--kv-dtype: expected int8|bf16, got {other:?}"));
                        }
                    };
                }
                "--execute" => out.execute = true,
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

fn usage() -> String {
    "usage: m2_rust_prefill [--model-dir DIR] [--batch N] [--prompt-len N] [--ctx N] [--block-size N] [--kv-dtype int8|bf16] [--artifact-dir DIR] [--execute]".to_string()
}
