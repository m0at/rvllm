use std::env;
use std::fs;
use std::path::PathBuf;
#[cfg(feature = "tpu")]
use std::time::Instant;

use rvllm_xla::m2_decode_bench::{
    check_m2_rust_decode_bench_json, M2RustDecodeExecutedTiming, M2_DEFAULT_DECODE_BENCH_BATCHES,
};
#[cfg(feature = "tpu")]
use rvllm_xla::m2_decode_bench::{m2_rust_decode_timing, M2RustDecodeTimingSource};
use rvllm_xla::m2_runtime::{m2_decode_mlir_execution_blocker, M2RuntimeMode};
use rvllm_xla::{
    m2_decode_graph_mlir_with_mosaic_body, m2_decode_smoke_mlir, plan_m2_rust_decode_bench,
    M2GraphAbi, M2GraphShape, M2RustDecodeBenchConfig, M2WeightUploadPlan, PjrtElementType,
    TpuMosaicSerializedBody, XlaArtifact, XlaTensorSpec, M2_HIDDEN, M2_NUM_LAYERS, M2_VOCAB,
};

#[cfg(feature = "tpu")]
use rvllm_loader::M2SafetensorsReader;
#[cfg(feature = "tpu")]
use rvllm_xla::{m2_gather_embed_bf16, PjrtClientHandle};

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
        && !args.use_existing_artifacts
    {
        return Err("real Rust M2 compile/execute needs --decode-layer-body FILE containing serde bytecode or lowered MLIR; use --native-smoke for the zero-logits PJRT smoke".into());
    }
    let artifacts = if args.use_existing_artifacts {
        load_decode_artifacts(&args)?
    } else if args.emit_decode_artifacts || args.runtime_mode != M2RuntimeMode::PlanningOnly {
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
        weight_format: args.weight_format.clone(),
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

#[cfg_attr(not(feature = "tpu"), allow(dead_code))]
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
            let arena = match args.weight_format.as_str() {
                "nvfp4" => weights.flat_arena(128)?,
                "int8" => weights.int8_flat_arena(128)?,
                _ => return Err("--weight-format: expected nvfp4|int8".into()),
            };
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
        let artifact =
            decode_artifact_manifest(&mlir_name, &shape, weight_arena_bytes, !args.native_smoke);
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

fn load_decode_artifacts(args: &Args) -> Result<Vec<DecodeArtifact>, Box<dyn std::error::Error>> {
    let mut out = Vec::with_capacity(args.batches.len());
    for &batch in &args.batches {
        let json_name = format!("m2_decode_b{batch}.json");
        let mlir_name = format!("m2_decode_b{batch}.mlir");
        let json_path = args.artifact_dir.join(&json_name);
        let artifact: XlaArtifact = serde_json::from_slice(&fs::read(&json_path)?)?;
        if artifact.mlir_file != mlir_name {
            return Err(format!(
                "{} points at {}, expected {}",
                json_path.display(),
                artifact.mlir_file,
                mlir_name
            )
            .into());
        }
        let mlir_path = args.artifact_dir.join(&artifact.mlir_file);
        let mlir = fs::read_to_string(&mlir_path)?;
        let shape = M2GraphShape::decode(batch, args.ctx, kv_bytes_per_elem(&args.kv_cache)?);
        let weight_arena_bytes = artifact
            .inputs
            .iter()
            .find(|input| input.name == "weight_arena")
            .and_then(|input| input.shape.first().copied())
            .ok_or_else(|| format!("{} missing weight_arena input", json_path.display()))?
            as usize;
        eprintln!(
            "loaded existing decode artifact {} and {}",
            mlir_path.display(),
            json_path.display()
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
    include_globals: bool,
) -> XlaArtifact {
    let mut inputs = vec![
        tensor("token_ids", &[shape.batch], PjrtElementType::S32),
        tensor("positions", &[shape.batch], PjrtElementType::S32),
        tensor("kv_cache", &[shape.kv_cache_bytes()], PjrtElementType::S8),
        tensor("weight_arena", &[weight_arena_bytes], PjrtElementType::S8),
        tensor(
            "int8_row_scales",
            &[M2_NUM_LAYERS * 128],
            PjrtElementType::F32,
        ),
    ];
    if include_globals {
        inputs.extend([
            tensor(
                "input_hidden",
                &[shape.batch, M2_HIDDEN],
                PjrtElementType::BF16,
            ),
            tensor("final_norm", &[M2_HIDDEN], PjrtElementType::BF16),
            tensor("lm_head", &[M2_VOCAB, M2_HIDDEN], PjrtElementType::BF16),
        ]);
    }
    XlaArtifact {
        mlir_file: mlir_file.to_string(),
        inputs,
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
    let num_devices = client.num_devices();
    if args.max_weight_arena_bytes == 0 && !args.native_smoke {
        return Err(
            "--execute-decode requires --max-weight-arena-bytes for real weight upload".into(),
        );
    }
    let mut out = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        let mlir = fs::read_to_string(&artifact.mlir_path)?;
        let exe = client.compile(&mlir)?;
        let reader = if args.native_smoke {
            None
        } else {
            Some(M2SafetensorsReader::open(&args.model_dir)?)
        };
        let token_zero = vec![0u8; artifact.shape.batch * 4];
        let pos_zero = vec![0u8; artifact.shape.batch * 4];
        let kv_zero = vec![0u8; artifact.shape.kv_cache_bytes()];
        let token_bufs = (0..num_devices)
            .map(|device| {
                client.buffer_from_host(
                    &token_zero,
                    &[artifact.shape.batch as i64],
                    PjrtElementType::S32,
                    device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pos_bufs = (0..num_devices)
            .map(|device| {
                client.buffer_from_host(
                    &pos_zero,
                    &[artifact.shape.batch as i64],
                    PjrtElementType::S32,
                    device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut kv_bufs = (0..num_devices)
            .map(|device| {
                client.buffer_from_host(
                    &kv_zero,
                    &[artifact.shape.kv_cache_bytes() as i64],
                    PjrtElementType::S8,
                    device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let row_scale_zero = vec![0u8; M2_NUM_LAYERS * 128 * 4];
        let (weight_arenas, row_scale_arenas) = if args.native_smoke {
            let weight_zero = vec![0u8; artifact.weight_arena_bytes];
            let weights = (0..num_devices)
                .map(|device| {
                    client.buffer_from_host(
                        &weight_zero,
                        &[artifact.weight_arena_bytes as i64],
                        PjrtElementType::S8,
                        device,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let scales = (0..num_devices)
                .map(|device| {
                    client.buffer_from_host(
                        &row_scale_zero,
                        &[(M2_NUM_LAYERS * 128) as i64],
                        PjrtElementType::F32,
                        device,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            (weights, scales)
        } else {
            let upload_start = Instant::now();
            let reader = reader.as_ref().expect("reader is present for real execute");
            let abi = M2GraphAbi::new(artifact.shape.clone())?;
            let weights = M2WeightUploadPlan::from_index_dir(&args.model_dir, &abi)?;
            let arena = match args.weight_format.as_str() {
                "nvfp4" => weights.flat_arena(128)?,
                "int8" => weights.int8_flat_arena(128)?,
                _ => return Err("--weight-format: expected nvfp4|int8".into()),
            };
            if arena.total_bytes.div_ceil(8) != artifact.weight_arena_bytes {
                return Err(format!(
                    "artifact local weight bytes {} do not match arena shard bytes {}",
                    artifact.weight_arena_bytes,
                    arena.total_bytes.div_ceil(8)
                )
                .into());
            }
            let host = match args.weight_format.as_str() {
                "nvfp4" => arena.materialize_host_buffer(reader, args.max_weight_arena_bytes)?,
                "int8" => {
                    arena.materialize_int8_host_buffer(reader, args.max_weight_arena_bytes)?
                }
                _ => return Err("--weight-format: expected nvfp4|int8".into()),
            };
            let row_scale_bytes = if args.weight_format == "int8" {
                arena.materialize_decode_w1_row_scale_probe_bytes(&host)?
            } else {
                row_scale_zero.clone()
            };
            let uploaded = (0..num_devices)
                .map(|device| {
                    let local =
                        local_weight_arena_shard(&host.bytes, artifact.weight_arena_bytes, device);
                    client.buffer_from_host(
                        &local,
                        &[artifact.weight_arena_bytes as i64],
                        PjrtElementType::S8,
                        device,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let row_scales = (0..num_devices)
                .map(|device| {
                    client.buffer_from_host(
                        &row_scale_bytes,
                        &[(M2_NUM_LAYERS * 128) as i64],
                        PjrtElementType::F32,
                        device,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            eprintln!(
                "uploaded real {} weight arena B={} total={:.3} GB local={:.3} GB devices={} in {:.2}s",
                args.weight_format,
                artifact.batch,
                host.total_bytes as f64 / 1.0e9,
                artifact.weight_arena_bytes as f64 / 1.0e9,
                num_devices,
                upload_start.elapsed().as_secs_f64()
            );
            (uploaded, row_scales)
        };
        let global_tensors = if args.native_smoke {
            None
        } else {
            let reader = reader.as_ref().expect("reader is present for real execute");
            let token_ids = vec![0; artifact.shape.batch];
            Some(upload_global_tensors(
                &client,
                reader,
                num_devices,
                &token_ids,
            )?)
        };
        let mut samples = Vec::with_capacity(args.iters);
        for step in 0..(args.warmup + args.iters) {
            let start = Instant::now();
            let per_device_inputs = (0..num_devices)
                .map(|device| {
                    let mut inputs = vec![
                        &token_bufs[device],
                        &pos_bufs[device],
                        &kv_bufs[device],
                        &weight_arenas[device],
                        &row_scale_arenas[device],
                    ];
                    if let Some((input_hidden, final_norms, lm_heads)) = &global_tensors {
                        inputs.push(&input_hidden[device]);
                        inputs.push(&final_norms[device]);
                        inputs.push(&lm_heads[device]);
                    }
                    inputs
                })
                .collect::<Vec<_>>();
            let outputs_by_device = client.execute_partitioned(&exe, &per_device_inputs)?;
            let mut next_kv_bufs = Vec::with_capacity(num_devices);
            for mut outputs in outputs_by_device {
                if outputs.len() != 3 {
                    return Err(
                        format!("decode returned {} outputs, expected 3", outputs.len()).into(),
                    );
                }
                let next_token = outputs.remove(1);
                let new_kv = outputs.remove(1);
                let mut next_bytes = vec![0u8; artifact.shape.batch * 4];
                client.buffer_to_host(&next_token, &mut next_bytes)?;
                next_kv_bufs.push(new_kv);
            }
            if step >= args.warmup {
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            kv_bufs = next_kv_bufs;
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
                device: format!("tpu:0-{}", num_devices.saturating_sub(1)),
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

#[cfg(any(feature = "tpu", test))]
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
type GlobalTensorBuffers = (
    Vec<rvllm_xla::PjrtBufferHandle>,
    Vec<rvllm_xla::PjrtBufferHandle>,
    Vec<rvllm_xla::PjrtBufferHandle>,
);

#[cfg(feature = "tpu")]
fn upload_global_tensors(
    client: &PjrtClientHandle,
    reader: &M2SafetensorsReader,
    num_devices: usize,
    token_ids: &[i32],
) -> Result<GlobalTensorBuffers, Box<dyn std::error::Error>> {
    let input_hidden = m2_gather_embed_bf16(reader, token_ids)?;
    let final_norm = reader.tensor("model.norm.weight")?;
    let lm_head = reader.tensor("lm_head.weight")?;
    Ok((
        upload_replicated(
            client,
            num_devices,
            &input_hidden,
            &[token_ids.len() as i64, M2_HIDDEN as i64],
        )?,
        upload_replicated(client, num_devices, final_norm.bytes, &[M2_HIDDEN as i64])?,
        upload_replicated(
            client,
            num_devices,
            lm_head.bytes,
            &[M2_VOCAB as i64, M2_HIDDEN as i64],
        )?,
    ))
}

#[cfg(feature = "tpu")]
fn upload_replicated(
    client: &PjrtClientHandle,
    num_devices: usize,
    bytes: &[u8],
    shape: &[i64],
) -> Result<Vec<rvllm_xla::PjrtBufferHandle>, Box<dyn std::error::Error>> {
    let out = (0..num_devices)
        .map(|device| client.buffer_from_host(bytes, shape, PjrtElementType::BF16, device))
        .collect::<Result<Vec<_>, _>>()?;
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
    weight_format: String,
    moe_impl: String,
    ppl_text: Option<PathBuf>,
    prompt: String,
    gen_tokens: usize,
    max_weight_arena_bytes: usize,
    use_existing_artifacts: bool,
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
            weight_format: "int8".to_string(),
            moe_impl: "auto".to_string(),
            ppl_text: None,
            prompt: "Explain angular momentum.".to_string(),
            gen_tokens: 256,
            max_weight_arena_bytes: 0,
            use_existing_artifacts: false,
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
                "--use-existing-artifacts" => out.use_existing_artifacts = true,
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
                "--weight-format" => {
                    i += 1;
                    out.weight_format = value(&args, i, "--weight-format")?.to_string();
                    if !matches!(out.weight_format.as_str(), "nvfp4" | "int8") {
                        return Err("--weight-format: expected nvfp4|int8".to_string());
                    }
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
    "usage: m2_rust_decode_bench [--model-dir DIR] [--artifact-dir DIR] [--out JSON] [--check JSON] [--decode-layer-body FILE] [--decode-layer-body-format serde|lowered] [--runtime-mode planning-only|compile-only|execute] [--emit-decode-artifacts] [--use-existing-artifacts] [--compile-decode] [--execute-decode] [--batches 1,8,16,32|--batch N] [--ctx N] [--iters N] [--warmup N] [--kv-cache int8|bf16] [--weight-format int8|nvfp4] [--moe-impl NAME] [--ppl-text FILE] [--prompt TEXT] [--gen-tokens N] [--max-weight-arena-bytes N]".to_string()
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

    #[test]
    fn parses_max_weight_arena_bytes() {
        let args = Args::parse(vec![
            "--max-weight-arena-bytes".to_string(),
            "1234".to_string(),
        ])
        .unwrap();
        assert_eq!(args.max_weight_arena_bytes, 1234);
    }

    #[test]
    fn pads_local_weight_arena_shards() {
        assert_eq!(local_weight_arena_shard(&[1, 2, 3, 4, 5], 2, 0), vec![1, 2]);
        assert_eq!(local_weight_arena_shard(&[1, 2, 3, 4, 5], 2, 2), vec![5, 0]);
    }
}
