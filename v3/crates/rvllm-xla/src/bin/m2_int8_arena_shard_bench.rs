use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde::Deserialize;

#[cfg(feature = "tpu")]
use rvllm_xla::{PjrtClientHandle, PjrtElementType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;

    let total_start = Instant::now();
    let manifest_start = Instant::now();
    let manifest: Manifest =
        serde_json::from_slice(&fs::read(args.shard_dir.join("manifest.json"))?)?;
    let manifest_seconds = manifest_start.elapsed().as_secs_f64();

    if manifest.total_bytes > args.max_bytes {
        return Err(format!(
            "arena is {} bytes, above configured max {}",
            manifest.total_bytes, args.max_bytes
        )
        .into());
    }

    let read_start = Instant::now();
    let mut bytes = vec![0u8; manifest.total_bytes];
    for shard in &manifest.shards {
        let start = shard.offset;
        let end = start + shard.nbytes;
        if end > bytes.len() {
            return Err(format!("{} extends past arena", shard.file).into());
        }
        let shard_bytes = fs::read(args.shard_dir.join(&shard.file))?;
        if shard_bytes.len() != shard.nbytes {
            return Err(format!(
                "{} expected {} bytes, got {}",
                shard.file,
                shard.nbytes,
                shard_bytes.len()
            )
            .into());
        }
        bytes[start..end].copy_from_slice(&shard_bytes);
    }
    let read_seconds = read_start.elapsed().as_secs_f64();

    let mut upload_seconds = None;
    if args.upload {
        upload_seconds = Some(upload_int8_arena(&bytes)?);
    }

    let report = serde_json::json!({
        "schema": "rvllm.m2.int8_arena_shard_bench.v1",
        "shard_dir": args.shard_dir.display().to_string(),
        "source_manifest_schema": manifest.schema,
        "format": manifest.format,
        "batch": manifest.batch,
        "ctx": manifest.ctx,
        "copy_dense_tensors": manifest.copy_dense_tensors,
        "dense_slots": manifest.dense_slots,
        "total_bytes": manifest.total_bytes,
        "shards": manifest.shards.len(),
        "manifest_seconds": manifest_seconds,
        "read_seconds": read_seconds,
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
fn upload_int8_arena(bytes: &[u8]) -> Result<f64, Box<dyn std::error::Error>> {
    let client = PjrtClientHandle::new()?;
    let start = Instant::now();
    let _buf = client.buffer_from_host(bytes, &[bytes.len() as i64], PjrtElementType::S8, 0)?;
    Ok(start.elapsed().as_secs_f64())
}

#[cfg(not(feature = "tpu"))]
fn upload_int8_arena(_bytes: &[u8]) -> Result<f64, Box<dyn std::error::Error>> {
    Err("--upload requires rvllm-xla --features tpu".into())
}

#[derive(Deserialize)]
struct Manifest {
    schema: String,
    format: String,
    batch: usize,
    ctx: usize,
    copy_dense_tensors: bool,
    dense_slots: String,
    total_bytes: usize,
    shards: Vec<Shard>,
}

#[derive(Deserialize)]
struct Shard {
    file: String,
    offset: usize,
    nbytes: usize,
}

struct Args {
    shard_dir: PathBuf,
    max_bytes: usize,
    upload: bool,
    out: Option<PathBuf>,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            shard_dir: PathBuf::from("/dev/shm/m2-int8-arena"),
            max_bytes: 300_000_000_000,
            upload: false,
            out: None,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--shard-dir" => {
                    i += 1;
                    out.shard_dir = PathBuf::from(value(&args, i, "--shard-dir")?);
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
    "usage: m2_int8_arena_shard_bench [--shard-dir DIR] [--max-bytes N] [--upload] [--out JSON]"
        .to_string()
}
