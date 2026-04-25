use std::env;
use std::path::PathBuf;

use rvllm_fused::{M2PrefillKvDType, M2PrefillScanShape};
use rvllm_xla::write_m2_prefill_artifact;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "tpu/out/m2/prefill_scan_artifact".to_string()),
    );
    let batch = parse_arg(args.next(), 1, "batch")?;
    let prompt_len = parse_arg(args.next(), 20, "prompt_len")?;
    let ctx = parse_arg(args.next(), 2048, "ctx")?;
    let num_partitions = parse_arg(args.next(), 8, "num_partitions")?;
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
    let mlir = shape.mlir("main")?;
    write_m2_prefill_artifact(&dir, "model.mlir", &mlir, shape, num_partitions)?;
    eprintln!(
        "wrote {} (batch={batch}, prompt_len={prompt_len}, ctx={ctx}, partitions={num_partitions}, kv={kv_dtype:?})",
        dir.display()
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
