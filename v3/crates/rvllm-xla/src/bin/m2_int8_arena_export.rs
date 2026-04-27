use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rvllm_loader::M2SafetensorsReader;
use rvllm_xla::{M2GraphAbi, M2GraphShape, M2WeightUploadPlan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    fs::create_dir_all(&args.out_dir)?;

    let total_start = Instant::now();
    let reader_start = Instant::now();
    let reader = M2SafetensorsReader::open(&args.model_dir)?;
    let reader_seconds = reader_start.elapsed().as_secs_f64();

    let plan_start = Instant::now();
    let abi = M2GraphAbi::new(M2GraphShape::decode(args.batch, args.ctx, 1))?;
    let weights = M2WeightUploadPlan::from_index_dir(&args.model_dir, &abi)?;
    let arena = weights.int8_flat_arena(args.alignment)?;
    let plan_seconds = plan_start.elapsed().as_secs_f64();

    let materialize_start = Instant::now();
    let host = if args.copy_dense {
        arena.materialize_int8_host_buffer(&reader, args.max_bytes)?
    } else {
        arena.materialize_int8_expert_host_buffer(&reader, args.max_bytes)?
    };
    let materialize_seconds = materialize_start.elapsed().as_secs_f64();

    let write_start = Instant::now();
    let mut shards = Vec::new();
    let shard_count = host.total_bytes.div_ceil(args.shard_bytes);
    for idx in 0..shard_count {
        let offset = idx * args.shard_bytes;
        let end = (offset + args.shard_bytes).min(host.total_bytes);
        let filename = format!("arena-{:05}-of-{:05}.bin", idx + 1, shard_count);
        let path = args.out_dir.join(&filename);
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&host.bytes[offset..end])?;
        file.flush()?;
        shards.push(serde_json::json!({
            "file": filename,
            "offset": offset,
            "nbytes": end - offset,
        }));
    }
    let write_seconds = write_start.elapsed().as_secs_f64();

    let manifest = serde_json::json!({
        "schema": "rvllm.m2.int8_arena_export.v1",
        "model": "lukealonso/MiniMax-M2.7-NVFP4",
        "format": "rvllm_flat_int8_arena",
        "source_format": "nvfp4_modelopt",
        "weight_format": "int8",
        "batch": args.batch,
        "ctx": args.ctx,
        "alignment": args.alignment,
        "copy_dense_tensors": args.copy_dense,
        "dense_slots": if args.copy_dense { "copied_from_checkpoint" } else { "reserved_zero_filled" },
        "total_bytes": host.total_bytes,
        "weight_entries": host.entries,
        "shard_bytes": args.shard_bytes,
        "shards": shards,
        "timing": {
            "reader_seconds": reader_seconds,
            "plan_seconds": plan_seconds,
            "materialize_seconds": materialize_seconds,
            "write_seconds": write_seconds,
            "total_seconds": total_start.elapsed().as_secs_f64(),
        },
    });

    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    fs::write(args.out_dir.join("manifest.json"), &manifest_bytes)?;
    println!("{}", String::from_utf8(manifest_bytes)?);
    Ok(())
}

struct Args {
    model_dir: PathBuf,
    out_dir: PathBuf,
    batch: usize,
    ctx: usize,
    alignment: usize,
    max_bytes: usize,
    shard_bytes: usize,
    copy_dense: bool,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            model_dir: PathBuf::from("/dev/shm/m2-nvfp4"),
            out_dir: PathBuf::from("/dev/shm/m2-int8-arena"),
            batch: 8,
            ctx: 2048,
            alignment: 128,
            max_bytes: 300_000_000_000,
            shard_bytes: 8 * 1024 * 1024 * 1024,
            copy_dense: false,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--model-dir" => {
                    i += 1;
                    out.model_dir = PathBuf::from(value(&args, i, "--model-dir")?);
                }
                "--out-dir" => {
                    i += 1;
                    out.out_dir = PathBuf::from(value(&args, i, "--out-dir")?);
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
    "usage: m2_int8_arena_export [--model-dir DIR] [--out-dir DIR] [--batch N] [--ctx N] [--alignment N] [--max-bytes N] [--shard-bytes N] [--copy-dense]".to_string()
}
