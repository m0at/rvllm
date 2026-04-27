use std::env;
use std::fs;
use std::path::PathBuf;

use rvllm_xla::{
    m2_decode_layer_int8_lowered_body_mlir, m2_decode_layer_lowered_body_mlir, M2GraphShape,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    let shape = M2GraphShape::decode(args.batch, args.ctx, args.kv_bytes_per_elem);
    let mlir = match args.format {
        BodyFormat::Base => {
            m2_decode_layer_lowered_body_mlir(&shape, args.weight_arena_local_bytes)?
        }
        BodyFormat::Int8 => {
            m2_decode_layer_int8_lowered_body_mlir(&shape, args.weight_arena_local_bytes)?
        }
    };
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, mlir)?;
    eprintln!("wrote {}", args.out.display());
    Ok(())
}

struct Args {
    out: PathBuf,
    batch: usize,
    ctx: usize,
    kv_bytes_per_elem: usize,
    weight_arena_local_bytes: usize,
    format: BodyFormat,
}

enum BodyFormat {
    Base,
    Int8,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            out: PathBuf::from("tpu/out/m2/rust_xla/m2_decode_layer_body_b8.mlir"),
            batch: 8,
            ctx: 2048,
            kv_bytes_per_elem: 1,
            weight_arena_local_bytes: 1,
            format: BodyFormat::Base,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--out" => {
                    i += 1;
                    out.out = PathBuf::from(value(&args, i, "--out")?);
                }
                "--batch" => {
                    i += 1;
                    out.batch = parse(value(&args, i, "--batch")?, "--batch")?;
                }
                "--ctx" => {
                    i += 1;
                    out.ctx = parse(value(&args, i, "--ctx")?, "--ctx")?;
                }
                "--kv-bytes-per-elem" => {
                    i += 1;
                    out.kv_bytes_per_elem = parse(
                        value(&args, i, "--kv-bytes-per-elem")?,
                        "--kv-bytes-per-elem",
                    )?;
                }
                "--weight-arena-local-bytes" => {
                    i += 1;
                    out.weight_arena_local_bytes = parse(
                        value(&args, i, "--weight-arena-local-bytes")?,
                        "--weight-arena-local-bytes",
                    )?;
                }
                "--format" => {
                    i += 1;
                    out.format = match value(&args, i, "--format")? {
                        "base" | "lowered" => BodyFormat::Base,
                        "int8" => BodyFormat::Int8,
                        other => {
                            return Err(format!("--format: expected base|int8, got {other:?}"))
                        }
                    };
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

fn usage() -> String {
    "usage: m2_emit_decode_layer_body [--out FILE] [--format base|int8] [--batch N] [--ctx N] [--kv-bytes-per-elem 1|2] [--weight-arena-local-bytes N]".to_string()
}
