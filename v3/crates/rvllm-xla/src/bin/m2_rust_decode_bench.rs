use std::env;
use std::fs;
use std::path::PathBuf;

use rvllm_xla::{plan_m2_rust_decode_bench, M2RustDecodeBenchConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    let report = plan_m2_rust_decode_bench(&M2RustDecodeBenchConfig {
        model_dir: args.model_dir,
        batch: args.batch,
        ctx: args.ctx,
        iters: args.iters,
        warmup: args.warmup,
        kv_cache: args.kv_cache,
        moe_impl: args.moe_impl,
    })?;
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
    out: Option<PathBuf>,
    batch: usize,
    ctx: usize,
    iters: usize,
    warmup: usize,
    kv_cache: String,
    moe_impl: String,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut out = Self {
            model_dir: PathBuf::from("/dev/shm/m2-nvfp4"),
            out: None,
            batch: 8,
            ctx: 2048,
            iters: 10,
            warmup: 3,
            kv_cache: "int8".to_string(),
            moe_impl: "auto".to_string(),
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
                    out.out = Some(PathBuf::from(value(&args, i, "--out")?));
                }
                "--batch" => {
                    i += 1;
                    out.batch = parse(value(&args, i, "--batch")?, "--batch")?;
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
                "--moe-impl" => {
                    i += 1;
                    out.moe_impl = value(&args, i, "--moe-impl")?.to_string();
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
    "usage: m2_rust_decode_bench [--model-dir DIR] [--out JSON] [--batch N] [--ctx N] [--iters N] [--warmup N] [--kv-cache int8|bf16] [--moe-impl NAME]".to_string()
}
