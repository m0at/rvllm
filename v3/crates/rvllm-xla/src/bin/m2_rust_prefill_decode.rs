use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use rvllm_fused::M2PrefillKvDType;
use rvllm_xla::m2_runtime::{m2_decode_execution_blocker, M2RuntimeMode};
use rvllm_xla::{plan_m2_rust_prefill_decode, M2RustPrefillDecodeConfig, M2RustPrefillDecodePlan};
use rvllm_xla::{XlaArtifact, XlaTensorSpec};

#[cfg(feature = "tpu")]
use rvllm_loader::M2SafetensorsReader;
#[cfg(feature = "tpu")]
use rvllm_xla::{
    load_artifact, m2_bf16_argmax_tokens, m2_bf16_logits_nll, m2_gather_embed_bf16,
    m2_ppl_from_nll, make_m2_prefill_inputs, M2GraphAbi, M2WeightUploadPlan, PjrtClientHandle,
    PjrtElementType, M2_HIDDEN, M2_NUM_LAYERS, M2_VOCAB,
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
        weight_format: args.weight_format.clone(),
    })?;

    eprintln!(
        "m2 rust prefill+decode plan: batch={} prompt_len={} decode_steps={} ctx={} kv_bytes={} weight_format={} weight_arena_bytes={} weight_entries={} decode_mlir_bytes={}",
        plan.prefill.shape.batch,
        plan.prefill.shape.prompt_len,
        plan.decode_steps,
        plan.prefill.shape.ctx,
        plan.decode_shape.kv_cache_bytes(),
        plan.weight_format,
        plan.weight_arena_bytes,
        plan.weight_entries,
        plan.decode_mlir.len()
    );
    eprintln!("  runtime_mode={}", args.runtime_mode.as_str());
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
    if let Some(path) = args.emit_decode_artifact.as_ref() {
        write_decode_artifact(path, &plan, args.decode_num_partitions)?;
        eprintln!("wrote decode artifact {}", path.display());
    }
    if args.runtime_mode == M2RuntimeMode::CompileOnly {
        compile_decode(&plan)?;
    }
    if args.execute_prefill {
        execute_prefill(args.artifact_dir, &plan)?;
    }
    if args.runtime_mode == M2RuntimeMode::Execute {
        let target_ids = args
            .ppl_target_ids
            .as_ref()
            .map(|path| read_token_ids(path))
            .transpose()?;
        let report = execute_decode(
            &model_dir,
            &plan,
            args.max_weight_arena_bytes,
            target_ids.as_deref(),
        )?;
        if let Some(path) = args.out_token_ids {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, token_lines(&report.generated_token_ids))?;
            eprintln!("wrote {}", path.display());
        }
    }
    Ok(())
}

fn write_decode_artifact(
    dir: &Path,
    plan: &M2RustPrefillDecodePlan,
    num_partitions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if num_partitions == 0 {
        return Err("--decode-num-partitions must be > 0".into());
    }
    fs::create_dir_all(dir)?;
    let mlir_file = "rvllm_m2_decode.mlir";
    fs::write(dir.join(mlir_file), plan.decode_mlir.as_bytes())?;
    let artifact = XlaArtifact {
        mlir_file: mlir_file.to_string(),
        inputs: plan
            .decode_input_specs
            .iter()
            .map(|spec| XlaTensorSpec {
                name: spec.name.to_string(),
                shape: spec.shape.clone(),
                dtype: spec.dtype,
            })
            .collect(),
        outputs: plan
            .decode_output_specs
            .iter()
            .map(|spec| XlaTensorSpec {
                name: spec.name.to_string(),
                shape: spec.shape.clone(),
                dtype: spec.dtype,
            })
            .collect(),
        donate_indices: vec![2],
        num_partitions,
    };
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}

#[cfg(feature = "tpu")]
fn compile_decode(plan: &M2RustPrefillDecodePlan) -> Result<(), Box<dyn std::error::Error>> {
    let client = PjrtClientHandle::new()?;
    let _exe = client.compile(&plan.decode_mlir)?;
    eprintln!(
        "m2 rust decode compiled through PJRT: mlir_bytes={}",
        plan.decode_mlir.len()
    );
    Ok(())
}

#[cfg(not(feature = "tpu"))]
fn compile_decode(_plan: &M2RustPrefillDecodePlan) -> Result<(), Box<dyn std::error::Error>> {
    Err("m2_rust_prefill_decode compile-only requires --features tpu".into())
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
    ppl_target_ids: Option<&[i32]>,
) -> Result<M2DecodeExecutionReport, Box<dyn std::error::Error>> {
    if let Some(reason) = m2_decode_execution_blocker(plan) {
        return Err(reason.into());
    }
    if max_weight_arena_bytes == 0 {
        return Err("--execute-decode requires --max-weight-arena-bytes".into());
    }
    if let Some(targets) = ppl_target_ids {
        let needed = plan.decode_steps * plan.decode_shape.batch;
        if targets.len() < needed {
            return Err(format!("PPL target ids need at least {needed} ids").into());
        }
    }
    let reader = M2SafetensorsReader::open(model_dir)?;
    let abi = M2GraphAbi::new(plan.decode_shape.clone())?;
    let weights = M2WeightUploadPlan::from_index_dir(model_dir, &abi)?;
    let arena = match plan.weight_format.as_str() {
        "nvfp4" => weights.flat_arena(128)?,
        "int8" => weights.int8_flat_arena(128)?,
        _ => return Err("--weight-format: expected nvfp4|int8".into()),
    };
    let client = PjrtClientHandle::new()?;
    let exe = client.compile(&plan.decode_mlir)?;
    let mut token_ids = plan.seed_decode_token_ids.clone();
    let mut positions = plan.seed_decode_positions.clone();
    let mut generated_token_ids = Vec::with_capacity(plan.decode_steps * plan.decode_shape.batch);
    let mut nll = Vec::new();
    let mut kv_buf = client.buffer_from_host(
        &vec![0u8; plan.decode_shape.kv_cache_bytes()],
        &[plan.decode_shape.kv_cache_bytes() as i64],
        PjrtElementType::S8,
        0,
    )?;
    let row_scale_zero = vec![0u8; M2_NUM_LAYERS * 128 * 4];
    let (weight_arena, row_scale_bytes) = match plan.weight_format.as_str() {
        "nvfp4" => {
            let host = arena.materialize_host_buffer(&reader, max_weight_arena_bytes)?;
            let local = local_weight_arena_shard(&host.bytes, plan.weight_arena_bytes, 0);
            (
                client.buffer_from_host(
                    &local,
                    &[plan.weight_arena_bytes as i64],
                    PjrtElementType::S8,
                    0,
                )?,
                row_scale_zero,
            )
        }
        "int8" => {
            let host = arena.materialize_int8_host_buffer(&reader, max_weight_arena_bytes)?;
            let row_scales = arena.materialize_decode_w1_row_scale_probe_bytes(&host)?;
            let local = local_weight_arena_shard(&host.bytes, plan.weight_arena_bytes, 0);
            (
                client.buffer_from_host(
                    &local,
                    &[plan.weight_arena_bytes as i64],
                    PjrtElementType::S8,
                    0,
                )?,
                row_scales,
            )
        }
        _ => return Err("--weight-format: expected nvfp4|int8".into()),
    };
    let int8_row_scales = client.buffer_from_host(
        &row_scale_bytes,
        &[(M2_NUM_LAYERS * 128) as i64],
        PjrtElementType::F32,
        0,
    )?;
    let final_norm = upload_global_tensor(&reader, &client, "model.norm.weight", &[M2_HIDDEN])?;
    let lm_head = upload_global_tensor(&reader, &client, "lm_head.weight", &[M2_VOCAB, M2_HIDDEN])?;

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
        let input_hidden = m2_gather_embed_bf16(&reader, &token_ids)?;
        let input_hidden_buf = client.buffer_from_host(
            &input_hidden,
            &[plan.decode_shape.batch as i64, M2_HIDDEN as i64],
            PjrtElementType::BF16,
            0,
        )?;
        let inputs = [
            &token_buf,
            &pos_buf,
            &kv_buf,
            &weight_arena,
            &int8_row_scales,
            &input_hidden_buf,
            &final_norm,
            &lm_head,
        ];
        let mut outputs = client.execute(&exe, &inputs)?;
        if outputs.len() != 3 {
            return Err(format!(
                "decode graph returned {} outputs, expected 3",
                outputs.len()
            )
            .into());
        }
        let logits_buf = outputs.remove(0);
        let next_token = outputs.remove(0);
        let new_kv = outputs.remove(0);
        let mut logits = vec![0u8; plan.decode_shape.batch * M2_VOCAB * 2];
        client.buffer_to_host(&logits_buf, &mut logits)?;
        if let Some(targets) = ppl_target_ids {
            let target_start = _step * plan.decode_shape.batch;
            let target_end = target_start + plan.decode_shape.batch;
            nll.extend(m2_bf16_logits_nll(
                &logits,
                &targets[target_start..target_end],
                plan.decode_shape.batch,
                M2_VOCAB,
            )?);
        }
        let _ = next_token;
        token_ids = m2_bf16_argmax_tokens(&logits, plan.decode_shape.batch, M2_VOCAB)?;
        generated_token_ids.extend_from_slice(&token_ids);
        for pos in &mut positions {
            *pos += 1;
        }
        kv_buf = new_kv;
    }
    eprintln!(
        "m2 rust decode loop executed: steps={} final_positions={:?} last_tokens={:?}",
        plan.decode_steps, positions, token_ids
    );
    if !nll.is_empty() {
        let ppl = m2_ppl_from_nll(&nll)?;
        eprintln!(
            "m2 rust ppl: n_tokens_scored={} avg_nll={:.6} ppl={:.6}",
            ppl.n_tokens_scored, ppl.avg_nll, ppl.ppl
        );
    }
    Ok(M2DecodeExecutionReport {
        generated_token_ids,
    })
}

#[cfg(feature = "tpu")]
fn local_weight_arena_shard(bytes: &[u8], local_bytes: usize, device: usize) -> Vec<u8> {
    let start = device.saturating_mul(local_bytes);
    let end = bytes.len().min(start.saturating_add(local_bytes));
    let mut out = vec![0u8; local_bytes];
    if start < end {
        out[..end - start].copy_from_slice(&bytes[start..end]);
    }
    out
}

#[cfg(feature = "tpu")]
fn upload_global_tensor(
    reader: &M2SafetensorsReader,
    client: &PjrtClientHandle,
    name: &'static str,
    shape: &[usize],
) -> Result<rvllm_xla::PjrtBufferHandle, Box<dyn std::error::Error>> {
    let view = reader.tensor(name)?;
    let expected = shape.iter().product::<usize>() * 2;
    if view.bytes.len() != expected {
        return Err(format!(
            "{name}: expected {expected} bytes, got {}",
            view.bytes.len()
        )
        .into());
    }
    let shape = shape.iter().map(|&x| x as i64).collect::<Vec<_>>();
    Ok(client.buffer_from_host(view.bytes, &shape, PjrtElementType::BF16, 0)?)
}

#[cfg(not(feature = "tpu"))]
fn execute_decode(
    _model_dir: &Path,
    plan: &M2RustPrefillDecodePlan,
    _max_weight_arena_bytes: usize,
    _ppl_target_ids: Option<&[i32]>,
) -> Result<M2DecodeExecutionReport, Box<dyn std::error::Error>> {
    if let Some(reason) = m2_decode_execution_blocker(plan) {
        return Err(reason.into());
    }
    Err("m2_rust_prefill_decode --execute-decode requires --features tpu".into())
}

struct M2DecodeExecutionReport {
    generated_token_ids: Vec<i32>,
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
    emit_decode_artifact: Option<PathBuf>,
    decode_num_partitions: usize,
    batch: usize,
    prompt_len: usize,
    decode_steps: usize,
    ctx: usize,
    block_size: u32,
    kv_dtype: M2PrefillKvDType,
    weight_format: String,
    runtime_mode: M2RuntimeMode,
    execute_prefill: bool,
    max_weight_arena_bytes: usize,
    out_token_ids: Option<PathBuf>,
    ppl_target_ids: Option<PathBuf>,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            model_dir: PathBuf::from("/dev/shm/m2-nvfp4"),
            artifact_dir: None,
            emit_decode_mlir: None,
            emit_decode_artifact: None,
            decode_num_partitions: 8,
            batch: 8,
            prompt_len: 20,
            decode_steps: 256,
            ctx: 2048,
            block_size: 32,
            kv_dtype: M2PrefillKvDType::Int8,
            weight_format: "int8".to_string(),
            runtime_mode: M2RuntimeMode::PlanningOnly,
            execute_prefill: false,
            max_weight_arena_bytes: 0,
            out_token_ids: None,
            ppl_target_ids: None,
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
                "--emit-decode-artifact" => {
                    i += 1;
                    out.emit_decode_artifact =
                        Some(PathBuf::from(value(&args, i, "--emit-decode-artifact")?));
                }
                "--decode-num-partitions" => {
                    i += 1;
                    out.decode_num_partitions = parse(
                        value(&args, i, "--decode-num-partitions")?,
                        "--decode-num-partitions",
                    )?;
                }
                "--runtime-mode" => {
                    i += 1;
                    out.runtime_mode = value(&args, i, "--runtime-mode")?
                        .parse::<M2RuntimeMode>()
                        .map_err(|e| format!("--runtime-mode: {e}"))?;
                }
                "--compile-decode" => out.runtime_mode = M2RuntimeMode::CompileOnly,
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
                "--weight-format" => {
                    i += 1;
                    out.weight_format = value(&args, i, "--weight-format")?.to_string();
                    if !matches!(out.weight_format.as_str(), "nvfp4" | "int8") {
                        return Err("--weight-format: expected nvfp4|int8".to_string());
                    }
                }
                "--execute-prefill" => out.execute_prefill = true,
                "--execute-decode" => out.runtime_mode = M2RuntimeMode::Execute,
                "--out-token-ids" => {
                    i += 1;
                    out.out_token_ids = Some(PathBuf::from(value(&args, i, "--out-token-ids")?));
                }
                "--ppl-target-ids" => {
                    i += 1;
                    out.ppl_target_ids = Some(PathBuf::from(value(&args, i, "--ppl-target-ids")?));
                }
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

fn read_token_ids(path: &Path) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for word in text.split_whitespace() {
        out.push(word.parse::<i32>()?);
    }
    Ok(out)
}

fn token_lines(tokens: &[i32]) -> String {
    let mut out = String::new();
    for token in tokens {
        out.push_str(&token.to_string());
        out.push('\n');
    }
    out
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
    "usage: m2_rust_prefill_decode [--model-dir DIR] [--batch N] [--prompt-len N] [--decode-steps N] [--ctx N] [--block-size N] [--kv-dtype int8|bf16] [--weight-format int8|nvfp4] [--runtime-mode planning-only|compile-only|execute] [--emit-decode-mlir FILE] [--emit-decode-artifact DIR] [--decode-num-partitions N] [--artifact-dir DIR] [--execute-prefill] [--compile-decode] [--execute-decode --max-weight-arena-bytes N] [--out-token-ids FILE] [--ppl-target-ids FILE]".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_compile_only_decode_artifact_flags() {
        let args = Args::parse(vec![
            "--runtime-mode".to_string(),
            "compile-only".to_string(),
            "--emit-decode-artifact".to_string(),
            "out/decode".to_string(),
            "--decode-num-partitions".to_string(),
            "4".to_string(),
        ])
        .unwrap();
        assert_eq!(args.runtime_mode, M2RuntimeMode::CompileOnly);
        assert_eq!(args.emit_decode_artifact, Some(PathBuf::from("out/decode")));
        assert_eq!(args.decode_num_partitions, 4);
    }

    #[test]
    fn execute_decode_alias_selects_execute_mode() {
        let args = Args::parse(vec!["--execute-decode".to_string()]).unwrap();
        assert_eq!(args.runtime_mode, M2RuntimeMode::Execute);
    }
}
