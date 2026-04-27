use std::env;
use std::fs;
use std::path::PathBuf;

use rvllm_xla::{m2_decode_graph_mlir, M2GraphAbi, M2GraphShape, M2WeightUploadPlan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    let shape = M2GraphShape::decode(args.batch, args.ctx, args.kv_bytes_per_elem);
    let abi = M2GraphAbi::new(shape.clone())?;
    let weights = M2WeightUploadPlan::from_index_dir(&args.model_dir, &abi)?;
    let arena = weights.flat_arena(args.alignment)?;
    let mlir = m2_decode_graph_mlir(&args.kernel_name, &shape, &arena)?;
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, mlir)?;
    eprintln!(
        "wrote {} (batch={} ctx={} kv_bytes={} weight_arena_bytes={} entries={})",
        args.out.display(),
        args.batch,
        args.ctx,
        shape.kv_cache_bytes(),
        arena.total_bytes,
        arena.entries.len()
    );
    Ok(())
}

struct Args {
    model_dir: PathBuf,
    out: PathBuf,
    batch: usize,
    ctx: usize,
    kv_bytes_per_elem: usize,
    alignment: usize,
    kernel_name: String,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            model_dir: PathBuf::from("tpu/harness/m2_checkpoint_schema"),
            out: PathBuf::from("tpu/out/m2/rvllm_m2_decode_graph.mlir"),
            batch: 8,
            ctx: 2048,
            kv_bytes_per_elem: 1,
            alignment: 128,
            kernel_name: "main".to_string(),
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--model-dir" => {
                    i += 1;
                    out.model_dir = PathBuf::from(value(&args, i, "--model-dir")?);
                }
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
                "--alignment" => {
                    i += 1;
                    out.alignment = parse(value(&args, i, "--alignment")?, "--alignment")?;
                }
                "--kernel" => {
                    i += 1;
                    out.kernel_name = value(&args, i, "--kernel")?.to_string();
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
    "usage: m2_emit_decode_graph [--model-dir DIR] [--out FILE] [--batch N] [--ctx N] [--kv-bytes-per-elem 1|2] [--alignment N] [--kernel NAME]".to_string()
}
