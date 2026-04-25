use std::env;
use std::fs;
use std::path::PathBuf;

use rvllm_fused::{M2Nvfp4MatmulShape, M2Nvfp4MosaicTilePlan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let out = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "tpu/out/m2/rvllm_m2_nvfp4_matmul.mlir".to_string()),
    );
    let m = parse_arg(args.next(), 8, "m")?;
    let n = parse_arg(args.next(), 1536, "n")?;
    let k = parse_arg(args.next(), 3072, "k")?;
    let kernel_name = args
        .next()
        .unwrap_or_else(|| "rvllm_m2_nvfp4_matmul".to_string());

    let shape = M2Nvfp4MatmulShape { m, n, k };
    let default_plan = shape.default_mosaic_plan();
    let bm = parse_arg(args.next(), default_plan.bm, "bm")?;
    let bn = parse_arg(args.next(), default_plan.bn, "bn")?;
    let bk = parse_arg(args.next(), default_plan.bk, "bk")?;
    let plan = M2Nvfp4MosaicTilePlan::new(bm, bn, bk);
    let mlir = shape.mosaic_mlir_with_plan(&kernel_name, plan)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, mlir)?;
    eprintln!(
        "wrote {} (m={m}, n={n}, k={k}, kernel={kernel_name}, bm={bm}, bn={bn}, bk={bk})",
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
