use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use rvllm_fused::M2PrefillKvDType;
use rvllm_xla::{plan_m2_rust_prefill_decode, M2RustPrefillDecodeConfig, M2RustPrefillDecodePlan};

#[cfg(feature = "tpu")]
use rvllm_loader::M2SafetensorsReader;
#[cfg(feature = "tpu")]
use rvllm_xla::{
    load_artifact, make_m2_prefill_inputs, M2GraphAbi, M2WeightUploadPlan, PjrtClientHandle,
    PjrtElementType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    let model_dir = args.model_dir.clone();
    let plan = plan_m2_rust_prefill_decode(&M2RustPrefillDecodeConfig {
        model_dir: args.model_dir,
        batch: args.batch,
        prompt_len: args.prompt_len,
        decode_steps: args.decode_steps,
        ctx: args.ctx,
        block_size: args.block_size,
        kv_dtype: args.kv_dtype,
    })?;

    eprintln!(
        "m2 rust prefill+decode plan: batch={} prompt_len={} decode_steps={} ctx={} kv_bytes={} weight_arena_bytes={} weight_entries={} decode_mlir_bytes={}",
        plan.prefill.shape.batch,
        plan.prefill.shape.prompt_len,
        plan.decode_steps,
        plan.prefill.shape.ctx,
        plan.decode_shape.kv_cache_bytes(),
        plan.weight_arena_bytes,
        plan.weight_entries,
        plan.decode_mlir.len()
    );
    eprintln!(
        "  seed decode token_ids={:?} positions={:?}",
        plan.seed_decode_token_ids, plan.seed_decode_positions
    );
    for spec in &plan.decode_input_specs {
        eprintln!(
            "  decode input {} {:?} {:?} {} bytes",
            spec.name, spec.shape, spec.dtype, spec.nbytes
        );
    }

    if let Some(path) = args.emit_decode_mlir {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, plan.decode_mlir.as_bytes())?;
        eprintln!("wrote {}", path.display());
    }
    if args.execute_prefill {
        execute_prefill(args.artifact_dir, &plan)?;
    }
    if args.execute_decode {
        execute_decode(&model_dir, &plan, args.max_weight_arena_bytes)?;
    }
    Ok(())
}

#[cfg(feature = "tpu")]
fn execute_prefill(
    artifact_dir: Option<PathBuf>,
    plan: &M2RustPrefillDecodePlan,
) -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir =
        artifact_dir.unwrap_or_else(|| PathBuf::from("tpu/out/m2/prefill_scan_artifact"));
    let (artifact, mlir, compile_options) = load_artifact(&artifact_dir)?;
    let host_inputs = make_m2_prefill_inputs(&plan.prefill.plan, plan.prefill.shape)?;
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
        "m2 rust batched prefill executed: artifact_inputs={} host_inputs={} outputs={}",
        artifact.inputs.len(),
        host_inputs.len(),
        outputs.len()
    );
    Ok(())
}

#[cfg(feature = "tpu")]
fn execute_decode(
    model_dir: &Path,
    plan: &M2RustPrefillDecodePlan,
    max_weight_arena_bytes: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if max_weight_arena_bytes == 0 {
        return Err("--execute-decode requires --max-weight-arena-bytes".into());
    }
    let reader = M2SafetensorsReader::open(model_dir)?;
    let abi = M2GraphAbi::new(plan.decode_shape.clone())?;
    let weights = M2WeightUploadPlan::from_index_dir(model_dir, &abi)?;
    let arena = weights.flat_arena(128)?;
    let client = PjrtClientHandle::new()?;
    let exe = client.compile(&plan.decode_mlir)?;
    let mut token_ids = plan.seed_decode_token_ids.clone();
    let mut positions = plan.seed_decode_positions.clone();
    let mut kv_buf = client.buffer_from_host(
        &vec![0u8; plan.decode_shape.kv_cache_bytes()],
        &[plan.decode_shape.kv_cache_bytes() as i64],
        PjrtElementType::S8,
        0,
    )?;
    let weight_arena =
        arena.upload_flat_arena_to_pjrt(&reader, &client, 0, max_weight_arena_bytes)?;

    for _step in 0..plan.decode_steps {
        let token_buf = client.buffer_from_host(
            &i32_bytes(&token_ids),
            &[plan.decode_shape.batch as i64],
            PjrtElementType::S32,
            0,
        )?;
        let pos_buf = client.buffer_from_host(
            &i32_bytes(&positions),
            &[plan.decode_shape.batch as i64],
            PjrtElementType::S32,
            0,
        )?;
        let inputs = [&token_buf, &pos_buf, &kv_buf, &weight_arena];
        let mut outputs = client.execute(&exe, &inputs)?;
        if outputs.len() != 3 {
            return Err(format!(
                "decode graph returned {} outputs, expected 3",
                outputs.len()
            )
            .into());
        }
        let next_token = outputs.remove(1);
        let new_kv = outputs.remove(1);
        let mut next_bytes = vec![0u8; plan.decode_shape.batch * 4];
        client.buffer_to_host(&next_token, &mut next_bytes)?;
        token_ids = read_i32s(&next_bytes);
        for pos in &mut positions {
            *pos += 1;
        }
        kv_buf = new_kv;
    }
    eprintln!(
        "m2 rust decode loop executed: steps={} final_positions={:?} last_tokens={:?}",
        plan.decode_steps, positions, token_ids
    );
    Ok(())
}

#[cfg(not(feature = "tpu"))]
fn execute_decode(
    _model_dir: &Path,
    _plan: &M2RustPrefillDecodePlan,
    _max_weight_arena_bytes: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("m2_rust_prefill_decode --execute-decode requires --features tpu".into())
}

#[cfg(not(feature = "tpu"))]
fn execute_prefill(
    _artifact_dir: Option<PathBuf>,
    _plan: &M2RustPrefillDecodePlan,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("m2_rust_prefill_decode --execute-prefill requires --features tpu".into())
}

struct Args {
    model_dir: PathBuf,
    artifact_dir: Option<PathBuf>,
    emit_decode_mlir: Option<PathBuf>,
    batch: usize,
    prompt_len: usize,
    decode_steps: usize,
    ctx: usize,
    block_size: u32,
    kv_dtype: M2PrefillKvDType,
    execute_prefill: bool,
    execute_decode: bool,
    max_weight_arena_bytes: usize,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            model_dir: PathBuf::from("/dev/shm/m2-nvfp4"),
            artifact_dir: None,
            emit_decode_mlir: None,
            batch: 8,
            prompt_len: 20,
            decode_steps: 256,
            ctx: 2048,
            block_size: 32,
            kv_dtype: M2PrefillKvDType::Int8,
            execute_prefill: false,
            execute_decode: false,
            max_weight_arena_bytes: 0,
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
                "--emit-decode-mlir" => {
                    i += 1;
                    out.emit_decode_mlir =
                        Some(PathBuf::from(value(&args, i, "--emit-decode-mlir")?));
                }
                "--batch" => {
                    i += 1;
                    out.batch = parse(value(&args, i, "--batch")?, "--batch")?;
                }
                "--prompt-len" => {
                    i += 1;
                    out.prompt_len = parse(value(&args, i, "--prompt-len")?, "--prompt-len")?;
                }
                "--decode-steps" => {
                    i += 1;
                    out.decode_steps = parse(value(&args, i, "--decode-steps")?, "--decode-steps")?;
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
                            return Err(format!("--kv-dtype: expected int8|bf16, got {other:?}"))
                        }
                    };
                }
                "--execute-prefill" => out.execute_prefill = true,
                "--execute-decode" => out.execute_decode = true,
                "--max-weight-arena-bytes" => {
                    i += 1;
                    out.max_weight_arena_bytes = parse(
                        value(&args, i, "--max-weight-arena-bytes")?,
                        "--max-weight-arena-bytes",
                    )?;
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

#[cfg(feature = "tpu")]
fn i32_bytes(vals: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for val in vals {
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

#[cfg(feature = "tpu")]
fn read_i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

fn usage() -> String {
    "usage: m2_rust_prefill_decode [--model-dir DIR] [--batch N] [--prompt-len N] [--decode-steps N] [--ctx N] [--block-size N] [--kv-dtype int8|bf16] [--emit-decode-mlir FILE] [--artifact-dir DIR] [--execute-prefill] [--execute-decode --max-weight-arena-bytes N]".to_string()
}
