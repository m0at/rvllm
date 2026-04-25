use std::env;
use std::fs;
use std::path::PathBuf;

use rvllm_fused::{M2PrefillKvDType, M2PrefillScanShape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let out = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "tpu/out/m2/rvllm_m2_prefill_scan.mlir".to_string()),
    );
    let batch = parse_arg(args.next(), 1, "batch")?;
    let prompt_len = parse_arg(args.next(), 20, "prompt_len")?;
    let ctx = parse_arg(args.next(), 2048, "ctx")?;
    let kernel_name = args
        .next()
        .unwrap_or_else(|| "rvllm_m2_prefill_scan".to_string());
    let kv_dtype = match args.next().as_deref() {
        Some("bf16") => M2PrefillKvDType::Bf16,
        Some("int8") | None => M2PrefillKvDType::Int8,
        Some(other) => return Err(format!("kv_dtype: expected int8|bf16, got {other:?}").into()),
    };

    let shape = M2PrefillScanShape {
        batch,
        prompt_len,
        hidden: 3072,
        ctx,
        num_layers: 62,
        num_kv_heads: 8,
        head_dim: 128,
        kv_dtype,
    };
    let mlir = shape.mlir(&kernel_name)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, mlir)?;
    eprintln!(
        "wrote {} (batch={batch}, prompt_len={prompt_len}, ctx={ctx}, kernel={kernel_name}, kv={kv_dtype:?})",
        out.display()
    );
    Ok(())
}

fn parse_arg(arg: Option<String>, default: usize, name: &'static str) -> Result<usize, String> {
    match arg {
        Some(s) => s
            .parse::<usize>()
            .map_err(|e| format!("{name}: expected usize, got {s:?}: {e}")),
        None => Ok(default),
    }
}
