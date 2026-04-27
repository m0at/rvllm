use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rvllm_loader::M2SafetensorsReader;
use rvllm_xla::{M2GraphAbi, M2GraphShape, M2WeightUploadPlan};

#[cfg(feature = "tpu")]
use rvllm_xla::{PjrtClientHandle, PjrtElementType};

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

    let materialize_start = Instant::now();
    let host = arena.materialize_int8_host_buffer(&reader, args.max_bytes)?;
    let materialize_seconds = materialize_start.elapsed().as_secs_f64();

    let mut upload_seconds = None;
    if args.upload {
        upload_seconds = Some(upload_int8_arena(&host.bytes, host.total_bytes)?);
    }

    let report = serde_json::json!({
        "schema": "rvllm.m2.int8_arena_bench.v1",
        "model_dir": args.model_dir.display().to_string(),
        "batch": args.batch,
        "ctx": args.ctx,
        "alignment": args.alignment,
        "weight_format": "int8",
        "source_format": "nvfp4_modelopt",
        "weight_entries": host.entries,
        "total_bytes": host.total_bytes,
        "max_bytes": args.max_bytes,
        "reader_seconds": reader_seconds,
        "plan_seconds": plan_seconds,
        "materialize_seconds": materialize_seconds,
        "upload_seconds": upload_seconds,
        "total_seconds": total_start.elapsed().as_secs_f64(),
        "uploaded_to_pjrt": args.upload,
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

#[cfg(feature = "tpu")]
fn upload_int8_arena(bytes: &[u8], total_bytes: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let client = PjrtClientHandle::new()?;
    let start = Instant::now();
    let _buf = client.buffer_from_host(bytes, &[total_bytes as i64], PjrtElementType::S8, 0)?;
    Ok(start.elapsed().as_secs_f64())
}

#[cfg(not(feature = "tpu"))]
fn upload_int8_arena(
    _bytes: &[u8],
    _total_bytes: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    Err("--upload requires rvllm-xla --features tpu".into())
}

struct Args {
    model_dir: PathBuf,
    batch: usize,
    ctx: usize,
    alignment: usize,
    max_bytes: usize,
    upload: bool,
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
            upload: false,
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
                "--upload" => out.upload = true,
                "--out" => {
                    i += 1;
                    out.out = Some(PathBuf::from(value(&args, i, "--out")?));
                }
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown arg {other:?}\n{}", usage())),
            }
            i += 1;
        }
        if out.batch == 0 || out.ctx == 0 {
            return Err("--batch and --ctx must be > 0".to_string());
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
    "usage: m2_int8_arena_bench [--model-dir DIR] [--batch N] [--ctx N] [--alignment N] [--max-bytes N] [--upload] [--out JSON]".to_string()
}
