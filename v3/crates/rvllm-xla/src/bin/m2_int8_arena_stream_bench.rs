use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rvllm_loader::M2SafetensorsReader;
use rvllm_xla::{M2GraphAbi, M2GraphShape, M2WeightUploadPlan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    let total_start = Instant::now();

    let reader_start = Instant::now();
    let reader = M2SafetensorsReader::open(&args.model_dir)?;
    let reader_seconds = reader_start.elapsed().as_secs_f64();

    let plan_start = Instant::now();
    let abi = M2GraphAbi::new(M2GraphShape::decode(args.batch, args.ctx, 1))?;
    let weights = M2WeightUploadPlan::from_index_dir(&args.model_dir, &abi)?;
    let arena = weights.int8_flat_arena(args.alignment)?;
    let plan_seconds = plan_start.elapsed().as_secs_f64();

    let stream_start = Instant::now();
    let mut shard_idx = 0usize;
    let mut shard_pos = 0usize;
    let mut shard_count = 0usize;
    let mut streamed_bytes = 0usize;
    let mut chunk_count = 0usize;
    let mut checksum = 0u64;
    let stats = arena.stream_int8_host_buffer(
        &reader,
        args.max_bytes,
        args.copy_dense,
        |_, mut bytes| {
            chunk_count += 1;
            streamed_bytes += bytes.len();
            for &b in bytes.iter().step_by(4096) {
                checksum = checksum.wrapping_mul(131).wrapping_add(b as u64);
            }
            while !bytes.is_empty() {
                if shard_pos == 0 {
                    shard_count += 1;
                    shard_idx += 1;
                }
                let take = bytes.len().min(args.shard_bytes - shard_pos);
                shard_pos += take;
                bytes = &bytes[take..];
                if shard_pos == args.shard_bytes {
                    shard_pos = 0;
                }
            }
            Ok(())
        },
    )?;
    let stream_seconds = stream_start.elapsed().as_secs_f64();

    let report = serde_json::json!({
        "schema": "rvllm.m2.int8_arena_stream_bench.v1",
        "model_dir": args.model_dir.display().to_string(),
        "batch": args.batch,
        "ctx": args.ctx,
        "alignment": args.alignment,
        "copy_dense_tensors": args.copy_dense,
        "dense_slots": if args.copy_dense { "copied_from_checkpoint" } else { "streamed_zero_filled" },
        "weight_format": "int8",
        "source_format": "nvfp4_modelopt",
        "max_bytes": args.max_bytes,
        "shard_bytes": args.shard_bytes,
        "logical_shards": shard_count,
        "last_shard_bytes": if shard_pos == 0 { args.shard_bytes.min(stats.total_bytes) } else { shard_pos },
        "checksum_sample": checksum,
        "streamed_bytes": streamed_bytes,
        "stream_chunks": chunk_count,
        "stats": {
            "total_bytes": stats.total_bytes,
            "streamed_bytes": stats.streamed_bytes,
            "chunks": stats.chunks,
            "entries": stats.entries,
            "converted_weights": stats.converted_weights,
            "zero_bytes": stats.zero_bytes,
        },
        "reader_seconds": reader_seconds,
        "plan_seconds": plan_seconds,
        "stream_seconds": stream_seconds,
        "total_seconds": total_start.elapsed().as_secs_f64(),
    });

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

struct Args {
    model_dir: PathBuf,
    batch: usize,
    ctx: usize,
    alignment: usize,
    max_bytes: usize,
    shard_bytes: usize,
    copy_dense: bool,
    out: Option<PathBuf>,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            model_dir: PathBuf::from("/dev/shm/m2-nvfp4"),
            batch: 8,
            ctx: 2048,
            alignment: 128,
            max_bytes: 300_000_000_000,
            shard_bytes: 8 * 1024 * 1024 * 1024,
            copy_dense: false,
            out: None,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--model-dir" => {
                    i += 1;
                    out.model_dir = PathBuf::from(value(&args, i, "--model-dir")?);
                }
                "--batch" => {
                    i += 1;
                    out.batch = parse(value(&args, i, "--batch")?, "--batch")?;
                }
                "--ctx" => {
                    i += 1;
                    out.ctx = parse(value(&args, i, "--ctx")?, "--ctx")?;
                }
                "--alignment" => {
                    i += 1;
                    out.alignment = parse(value(&args, i, "--alignment")?, "--alignment")?;
                }
                "--max-bytes" => {
                    i += 1;
                    out.max_bytes = parse(value(&args, i, "--max-bytes")?, "--max-bytes")?;
                }
                "--shard-bytes" => {
                    i += 1;
                    out.shard_bytes = parse(value(&args, i, "--shard-bytes")?, "--shard-bytes")?;
                }
                "--copy-dense" => out.copy_dense = true,
                "--out" => {
                    i += 1;
                    out.out = Some(PathBuf::from(value(&args, i, "--out")?));
                }
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown arg {other:?}\n{}", usage())),
            }
            i += 1;
        }
        if out.batch == 0 || out.ctx == 0 || out.shard_bytes == 0 {
            return Err("--batch, --ctx, and --shard-bytes must be > 0".to_string());
        }
        Ok(out)
    }
}

fn value<'a>(args: &'a [String], i: usize, flag: &str) -> Result<&'a str, String> {
    args.get(i)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag}: missing value"))
}

fn parse<T>(s: &str, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    s.parse::<T>().map_err(|e| format!("{flag}: {e}"))
}

fn usage() -> String {
    "usage: m2_int8_arena_stream_bench [--model-dir DIR] [--batch N] [--ctx N] [--alignment N] [--max-bytes N] [--shard-bytes N] [--copy-dense] [--out JSON]".to_string()
}
