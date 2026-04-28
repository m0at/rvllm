use std::env;
use std::fs;
#[cfg(feature = "tpu")]
use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "tpu")]
use std::time::Instant;

use rvllm_xla::m2_decode_bench::{
    check_m2_rust_decode_bench_json, M2RustDecodeExecutedTiming, M2RustDecodePplResult,
    M2_DEFAULT_DECODE_BENCH_BATCHES,
};
#[cfg(feature = "tpu")]
use rvllm_xla::m2_decode_bench::{m2_rust_decode_timing_with_metrics, M2RustDecodeTimingSource};
use rvllm_xla::m2_runtime::{m2_decode_mlir_execution_blocker, M2RuntimeMode};
use rvllm_xla::{
    m2_decode_graph_mlir_with_mosaic_body, m2_decode_smoke_mlir, plan_m2_rust_decode_bench,
    M2GraphAbi, M2GraphShape, M2RustDecodeBenchConfig, M2WeightUploadPlan, PjrtElementType,
    TpuMosaicSerializedBody, XlaArtifact, XlaTensorSpec, M2_HIDDEN, M2_NUM_LAYERS, M2_VOCAB,
};

#[cfg(feature = "tpu")]
use rvllm_loader::M2SafetensorsReader;
#[cfg(feature = "tpu")]
use rvllm_xla::{
    analyze_logits_observability, enforce_body_probe_if_enabled, m2_bf16_argmax_tokens,
    m2_bf16_logits_nll, m2_gather_embed_bf16, m2_ppl_from_nll, M2LogitsObservabilityReport,
    PjrtClientHandle,
};

#[cfg(feature = "tpu")]
const M2_EOS_TOKEN_ID: i32 = 200020;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    if let Some(path) = args.check {
        let bytes = fs::read(&path)?;
        let json: serde_json::Value = serde_json::from_slice(&bytes)?;
        check_m2_rust_decode_bench_json(&json)?;
        eprintln!("checked {}", path.display());
        return Ok(());
    }
    let stablehlo_layer =
        std::env::var("RVLLM_M2_LAYER_MODE").unwrap_or_default() != "mosaic"
            && std::env::var("RVLLM_M2_LAYER_MODE").is_ok();
    if args.runtime_mode != M2RuntimeMode::PlanningOnly
        && !args.native_smoke
        && args.decode_layer_body.is_none()
        && !args.use_existing_artifacts
        && !stablehlo_layer
    {
        return Err("real Rust M2 compile/execute needs --decode-layer-body FILE containing serde bytecode or lowered MLIR; use --native-smoke for the zero-logits PJRT smoke, or RVLLM_M2_LAYER_MODE=stablehlo for inline layer ops".into());
    }
    let artifacts = if args.use_existing_artifacts {
        load_decode_artifacts(&args)?
    } else if args.emit_decode_artifacts || args.runtime_mode != M2RuntimeMode::PlanningOnly {
        write_decode_artifacts(&args)?
    } else {
        Vec::new()
    };
    let mut compiled = false;
    let executed = match args.runtime_mode {
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
    if !executed.is_empty() {
        let mut nll = Vec::new();
        let mut top1_hits = 0usize;
        let mut top1_total = 0usize;
        let mut generated_token_ids = Vec::new();
        let mut generated_text = None;
        let mut stopped_eos = false;
        #[cfg(feature = "tpu")]
        let mut last_observability: Option<M2LogitsObservabilityReport> = None;
        for (item, executed) in report.sweep.iter_mut().zip(executed) {
            item.status = "executed";
            item.error = None;
            if let Some(ppl) = executed.ppl.clone() {
                nll.push(ppl);
            }
            top1_hits += executed.top1_hits;
            top1_total += executed.top1_total;
            if generated_token_ids.is_empty() && !executed.generated_token_ids.is_empty() {
                generated_token_ids = executed.generated_token_ids.clone();
                generated_text = executed.generated_text.clone();
                stopped_eos = executed.stopped_eos;
            }
            item.timing = Some(executed.timing);
            #[cfg(feature = "tpu")]
            {
                if let Some(report) = executed.observability.clone() {
                    last_observability = Some(report);
                }
            }
        }
        if let Some(ppl) = nll.last() {
            report.ppl.status = "executed";
            report.ppl.result = Some(serde_json::json!({
                "n_tokens_scored": ppl.n_tokens_scored,
                "avg_nll": ppl.avg_nll,
                "ppl": ppl.ppl,
            }));
        }
        if !generated_token_ids.is_empty() {
            report.generation.status = "executed";
            report.generation.result = Some(serde_json::json!({
                "generated_tokens": generated_token_ids.len(),
                "generated_token_ids": generated_token_ids,
                "stopped_eos": stopped_eos,
                "text": generated_text,
                "top1_match_rate": if top1_total > 0 { Some(top1_hits as f64 / top1_total as f64) } else { None },
                "matched": top1_hits,
                "total": top1_total,
                "note": "free-run argmax token generation; top1 fields are present only when teacher targets are provided",
            }));
        }
        #[cfg(feature = "tpu")]
        if let Some(obs) = last_observability {
            report.body_probe = Some(serde_json::to_value(&obs)?);
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

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "tpu"), allow(dead_code))]
struct ExecutedDecodeArtifact {
    timing: M2RustDecodeExecutedTiming,
    ppl: Option<M2RustDecodePplResult>,
    top1_hits: usize,
    top1_total: usize,
    generated_token_ids: Vec<i32>,
    generated_text: Option<String>,
    stopped_eos: bool,
    #[cfg(feature = "tpu")]
    observability: Option<M2LogitsObservabilityReport>,
}

fn write_decode_artifacts(args: &Args) -> Result<Vec<DecodeArtifact>, Box<dyn std::error::Error>> {
    let stablehlo_layer =
        std::env::var("RVLLM_M2_LAYER_MODE").unwrap_or_default() != "mosaic"
            && std::env::var("RVLLM_M2_LAYER_MODE").is_ok();
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
            weight_arena_bytes = if stablehlo_layer {
                1
            } else {
                arena.total_bytes.div_ceil(8)
            };
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
) -> Result<Vec<ExecutedDecodeArtifact>, Box<dyn std::error::Error>> {
    let stablehlo_layer =
        std::env::var("RVLLM_M2_LAYER_MODE").unwrap_or_default() != "mosaic"
            && std::env::var("RVLLM_M2_LAYER_MODE").is_ok();
    let client = PjrtClientHandle::new()?;
    let num_devices = client.num_devices();
    if args.max_weight_arena_bytes == 0 && !args.native_smoke && !stablehlo_layer {
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
        let kv_zero = vec![0u8; artifact.shape.kv_cache_bytes()];
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
        let (weight_arenas, row_scale_arenas) = if args.native_smoke || stablehlo_layer {
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
            let w1_probe_only = args.weight_format == "int8"
                && env::var("RVLLM_M2_MOE_INT8").as_deref() == Ok("w1");
            let row_scale_bytes = if w1_probe_only {
                arena.materialize_decode_w1_probe_row_scales(reader)?
            } else if args.weight_format == "int8" {
                let host =
                    arena.materialize_int8_host_buffer(reader, args.max_weight_arena_bytes)?;
                arena.materialize_decode_w1_row_scale_probe_bytes(&host)?
            } else {
                row_scale_zero.clone()
            };
            let uploaded = (0..num_devices)
                .map(|device| {
                    let local = if w1_probe_only {
                        arena.materialize_decode_w1_probe_shard(
                            reader,
                            artifact.weight_arena_bytes,
                            device,
                        )?
                    } else {
                        let host = match args.weight_format.as_str() {
                            "nvfp4" => arena
                                .materialize_host_buffer(reader, args.max_weight_arena_bytes)?,
                            "int8" => arena.materialize_int8_host_buffer(
                                reader,
                                args.max_weight_arena_bytes,
                            )?,
                            _ => unreachable!("weight_format is validated during arg parsing"),
                        };
                        local_weight_arena_shard(&host.bytes, artifact.weight_arena_bytes, device)
                    };
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
                arena.total_bytes as f64 / 1.0e9,
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
            Some(upload_static_global_tensors(&client, reader, num_devices)?)
        };
        let dense_weight_bufs: Option<Vec<_>> = if stablehlo_layer && !args.native_smoke {
            let reader = reader.as_ref().expect("reader is present for real execute");
            Some(upload_dense_weight_arena(&client, reader, num_devices)?)
        } else {
            None
        };
        let teacher = if args.native_smoke {
            None
        } else {
            let reader = reader.as_ref().expect("reader is present for real execute");
            Some(prepare_decode_tokens(
                args,
                reader,
                artifact.shape.batch,
                args.warmup + args.iters,
            )?)
        };
        let prompt_prefill_steps = if args.ppl_text.is_none() && args.ppl_token_ids.is_none() {
            teacher
                .as_ref()
                .map(|tokens| tokens.lanes.first().map_or(0, |lane| lane.len().saturating_sub(1)))
                .unwrap_or(0)
        } else {
            0
        };
        let total_steps = args.warmup + args.iters + prompt_prefill_steps;
        let score_teacher = args.ppl_text.is_some() || args.ppl_token_ids.is_some();
        let mut token_ids = teacher
            .as_ref()
            .map(|tokens| tokens.input_at(0))
            .unwrap_or_else(|| vec![0; artifact.shape.batch]);
        let mut positions = vec![0i32; artifact.shape.batch];
        let mut samples = Vec::with_capacity(args.iters);
        let mut ttft_ms = None;
        let mut nll = Vec::new();
        let mut top1_hits = 0usize;
        let mut top1_total = 0usize;
        let mut observability: Option<M2LogitsObservabilityReport> = None;
        let mut generated_token_ids = Vec::new();
        let mut stopped_eos = false;
        for step in 0..total_steps {
            if let Some(tokens) = &teacher {
                if tokens.has_input(step) {
                    token_ids = tokens.input_at(step);
                }
            }
            let token_bytes = i32_bytes(&token_ids);
            let pos_bytes = i32_bytes(&positions);
            let token_bufs = (0..num_devices)
                .map(|device| {
                    client.buffer_from_host(
                        &token_bytes,
                        &[artifact.shape.batch as i64],
                        PjrtElementType::S32,
                        device,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let pos_bufs = (0..num_devices)
                .map(|device| {
                    client.buffer_from_host(
                        &pos_bytes,
                        &[artifact.shape.batch as i64],
                        PjrtElementType::S32,
                        device,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let input_hidden = if args.native_smoke {
                None
            } else {
                let reader = reader.as_ref().expect("reader is present for real execute");
                Some(upload_input_hidden(
                    &client,
                    reader,
                    num_devices,
                    &token_ids,
                    artifact.shape.batch,
                )?)
            };
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
                    if let (Some(input_hidden), Some((final_norms, lm_heads))) =
                        (&input_hidden, &global_tensors)
                    {
                        inputs.push(&input_hidden[device]);
                        inputs.push(&final_norms[device]);
                        inputs.push(&lm_heads[device]);
                    }
                    if let Some(dense) = &dense_weight_bufs {
                        inputs.push(&dense[device]);
                    }
                    inputs
                })
                .collect::<Vec<_>>();
            let outputs_by_device = client.execute_partitioned(&exe, &per_device_inputs)?;
            let mut next_kv_bufs = Vec::with_capacity(num_devices);
            let mut logits = None;
            for (device, mut outputs) in outputs_by_device.into_iter().enumerate() {
                if outputs.len() != 3 {
                    return Err(
                        format!("decode returned {} outputs, expected 3", outputs.len()).into(),
                    );
                }
                let logits_buf = outputs.remove(0);
                let _next_token = outputs.remove(0);
                let new_kv = outputs.remove(0);
                if device == 0 {
                    let mut logits_bytes = vec![0u8; artifact.shape.batch * M2_VOCAB * 2];
                    client.buffer_to_host(&logits_buf, &mut logits_bytes)?;
                    logits = Some(logits_bytes);
                }
                next_kv_bufs.push(new_kv);
            }
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            if step == 0 {
                ttft_ms = Some(elapsed_ms);
            }
            if step >= args.warmup + prompt_prefill_steps {
                samples.push(elapsed_ms);
                if observability.is_none() {
                    if let Some(logits) = logits.as_ref() {
                        observability = Some(analyze_logits_observability(
                            logits,
                            artifact.shape.batch,
                            M2_VOCAB,
                        )?);
                    }
                }
                if score_teacher {
                    if let (Some(tokens), Some(logits)) = (&teacher, logits.as_ref()) {
                        if let Ok(targets) = tokens.target_at(step) {
                            let argmax =
                                m2_bf16_argmax_tokens(logits, artifact.shape.batch, M2_VOCAB)?;
                            top1_hits += argmax
                                .iter()
                                .zip(targets.iter())
                                .filter(|(got, want)| got == want)
                                .count();
                            top1_total += targets.len();
                            nll.extend(m2_bf16_logits_nll(
                                logits,
                                &targets,
                                artifact.shape.batch,
                                M2_VOCAB,
                            )?);
                        }
                    }
                }
            }
            if let Some(logits) = logits.as_ref() {
                let next_tokens = m2_bf16_argmax_tokens(logits, artifact.shape.batch, M2_VOCAB)?;
                if step >= args.warmup + prompt_prefill_steps {
                    generated_token_ids.extend_from_slice(&next_tokens);
                    if artifact.shape.batch == 1
                        && next_tokens.first().copied() == Some(M2_EOS_TOKEN_ID)
                    {
                        stopped_eos = true;
                        break;
                    }
                }
                token_ids = next_tokens;
            }
            for pos in &mut positions {
                *pos += 1;
            }
            kv_bufs = next_kv_bufs;
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        let ms_min = *samples.first().ok_or("no timing samples")?;
        let ms_max = *samples.last().ok_or("no timing samples")?;
        let ms_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let ms_p50 = samples[samples.len() / 2];
        let ppl = if nll.is_empty() {
            None
        } else {
            let ppl = m2_ppl_from_nll(&nll)?;
            Some(M2RustDecodePplResult {
                n_tokens_scored: ppl.n_tokens_scored,
                avg_nll: ppl.avg_nll,
                ppl: ppl.ppl,
            })
        };
        let top1_match_rate = if top1_total == 0 {
            None
        } else {
            Some(top1_hits as f64 / top1_total as f64)
        };
        let timing = m2_rust_decode_timing_with_metrics(
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
            ttft_ms,
            ppl.clone(),
            top1_match_rate,
        )?;
        eprintln!(
            "executed B={} ttft={:.3} ms mean={:.3} ms tok/s={:.2} ppl={} top1={}",
            artifact.batch,
            ttft_ms.unwrap_or(0.0),
            ms_mean,
            1000.0 * artifact.batch as f64 / ms_mean,
            ppl.as_ref()
                .map(|p| format!("{:.4}", p.ppl))
                .unwrap_or_else(|| "-".to_string()),
            top1_match_rate
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "-".to_string())
        );
        if let Some(report) = observability.as_ref() {
            eprintln!(
                "logits_observability B={} var={:.3e} range=[{:.3e},{:.3e}] nonzero_frac={:.3} distinct_top1={} passed={}",
                artifact.batch,
                report.var,
                report.min,
                report.max,
                report.nonzero_frac,
                report.distinct_top1,
                report.passed,
            );
            enforce_body_probe_if_enabled(report)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        }
        let generated_text = if artifact.shape.batch == 1 && !generated_token_ids.is_empty() {
            Some(decode_token_ids(&args.model_dir, &generated_token_ids)?)
        } else {
            None
        };
        out.push(ExecutedDecodeArtifact {
            timing,
            ppl,
            top1_hits,
            top1_total,
            generated_token_ids,
            generated_text,
            stopped_eos,
            observability,
        });
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
);

#[cfg(feature = "tpu")]
fn upload_static_global_tensors(
    client: &PjrtClientHandle,
    reader: &M2SafetensorsReader,
    num_devices: usize,
) -> Result<GlobalTensorBuffers, Box<dyn std::error::Error>> {
    let final_norm = reader.tensor("model.norm.weight")?;
    let lm_head = reader.tensor("lm_head.weight")?;
    Ok((
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
fn upload_dense_weight_arena(
    client: &PjrtClientHandle,
    reader: &M2SafetensorsReader,
    num_devices: usize,
) -> Result<Vec<rvllm_xla::PjrtBufferHandle>, Box<dyn std::error::Error>> {
    let dense_fields: &[&str] = &[
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "block_sparse_moe.gate.weight",
        "block_sparse_moe.e_score_correction_bias",
    ];
    let mut buf = Vec::new();
    for layer_idx in 0..M2_NUM_LAYERS {
        for field in dense_fields {
            let name = format!("model.layers.{layer_idx}.{field}");
            let tensor = reader.tensor(&name)?;
            buf.extend_from_slice(tensor.bytes);
        }
    }
    eprintln!(
        "dense_weight_arena: {} bytes ({:.2} MB) for {} layers",
        buf.len(),
        buf.len() as f64 / 1e6,
        M2_NUM_LAYERS,
    );
    let n_bf16 = buf.len() / 2;
    upload_replicated(client, num_devices, &buf, &[n_bf16 as i64])
}

#[cfg(feature = "tpu")]
fn upload_input_hidden(
    client: &PjrtClientHandle,
    reader: &M2SafetensorsReader,
    num_devices: usize,
    token_ids: &[i32],
    batch: usize,
) -> Result<Vec<rvllm_xla::PjrtBufferHandle>, Box<dyn std::error::Error>> {
    if token_ids.len() != batch {
        return Err(format!("expected {batch} token ids, got {}", token_ids.len()).into());
    }
    let input_hidden = m2_gather_embed_bf16(reader, token_ids)?;
    upload_replicated(
        client,
        num_devices,
        &input_hidden,
        &[token_ids.len() as i64, M2_HIDDEN as i64],
    )
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

#[cfg(feature = "tpu")]
#[derive(Clone, Debug)]
struct DecodeTokenPlan {
    lanes: Vec<Vec<i32>>,
}

#[cfg(feature = "tpu")]
impl DecodeTokenPlan {
    fn has_input(&self, step: usize) -> bool {
        self.lanes.iter().all(|lane| step < lane.len())
    }

    fn input_at(&self, step: usize) -> Vec<i32> {
        self.lanes.iter().map(|lane| lane[step]).collect()
    }

    fn target_at(&self, step: usize) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        if self.lanes.iter().any(|lane| step + 1 >= lane.len()) {
            return Err(format!("not enough target token ids for step {step}").into());
        }
        Ok(self.lanes.iter().map(|lane| lane[step + 1]).collect())
    }
}

#[cfg(feature = "tpu")]
fn prepare_decode_tokens(
    args: &Args,
    _reader: &M2SafetensorsReader,
    batch: usize,
    total_steps: usize,
) -> Result<DecodeTokenPlan, Box<dyn std::error::Error>> {
    let ids = if let Some(path) = &args.ppl_token_ids {
        read_token_ids(path)?
    } else if let Some(path) = &args.ppl_text {
        tokenize_text_file(&args.model_dir, path)?
    } else {
        tokenize_text(&args.model_dir, &args.prompt)?
    };
    if ids.is_empty() {
        return Err("tokenizer produced zero token ids".into());
    }
    if args.ppl_text.is_none() && args.ppl_token_ids.is_none() {
        return Ok(DecodeTokenPlan {
            lanes: vec![ids; batch],
        });
    }
    let stride = if args.ppl_text.is_some() || args.ppl_token_ids.is_some() {
        total_steps + 1
    } else {
        1
    };
    let need = batch * stride;
    let mut lanes = Vec::with_capacity(batch);
    if ids.len() >= need {
        for row in 0..batch {
            let start = row * stride;
            lanes.push(ids[start..start + stride].to_vec());
        }
    } else if stride == 1 {
        for row in 0..batch {
            lanes.push(vec![ids[row.min(ids.len() - 1)]]);
        }
    } else {
        return Err(format!(
            "need at least {need} token ids for B={batch}, steps={total_steps}; got {}",
            ids.len()
        )
        .into());
    }
    Ok(DecodeTokenPlan { lanes })
}

#[cfg(feature = "tpu")]
fn tokenize_text_file(
    model_dir: &Path,
    path: &Path,
) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    tokenize_text(model_dir, &text)
}

#[cfg(feature = "tpu")]
fn tokenize_text(model_dir: &Path, text: &str) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("load tokenizer {}: {e}", tokenizer_path.display()))?;
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| format!("tokenize text: {e}"))?;
    Ok(encoding.get_ids().iter().map(|id| *id as i32).collect())
}

#[cfg(feature = "tpu")]
fn decode_token_ids(
    model_dir: &Path,
    ids: &[i32],
) -> Result<String, Box<dyn std::error::Error>> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("load tokenizer {}: {e}", tokenizer_path.display()))?;
    let ids = ids
        .iter()
        .map(|id| {
            u32::try_from(*id).map_err(|_| format!("generated token id out of range: {id}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    tokenizer
        .decode(&ids, true)
        .map_err(|e| format!("decode token ids: {e}").into())
}

#[cfg(feature = "tpu")]
fn read_token_ids(path: &Path) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for word in text.split_whitespace() {
        out.push(word.parse::<i32>()?);
    }
    Ok(out)
}

#[cfg(feature = "tpu")]
fn i32_bytes(vals: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for val in vals {
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

#[cfg(not(feature = "tpu"))]
fn execute_decode_artifacts(
    _artifacts: &[DecodeArtifact],
    _args: &Args,
) -> Result<Vec<ExecutedDecodeArtifact>, Box<dyn std::error::Error>> {
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
    ppl_token_ids: Option<PathBuf>,
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
            ppl_token_ids: None,
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
                "--ppl-token-ids" => {
                    i += 1;
                    out.ppl_token_ids = Some(PathBuf::from(value(&args, i, "--ppl-token-ids")?));
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
    "usage: m2_rust_decode_bench [--model-dir DIR] [--artifact-dir DIR] [--out JSON] [--check JSON] [--decode-layer-body FILE] [--decode-layer-body-format serde|lowered] [--runtime-mode planning-only|compile-only|execute] [--emit-decode-artifacts] [--use-existing-artifacts] [--compile-decode] [--execute-decode] [--batches 1,8,16,32|--batch N] [--ctx N] [--iters N] [--warmup N] [--kv-cache int8|bf16] [--weight-format int8|nvfp4] [--moe-impl NAME] [--ppl-text FILE|--ppl-token-ids FILE] [--prompt TEXT] [--gen-tokens N] [--max-weight-arena-bytes N]".to_string()
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
