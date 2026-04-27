use std::env;
use std::fs;
use std::path::PathBuf;

#[cfg(feature = "tpu")]
use rvllm_xla::PjrtClientHandle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: xla_compile_mlir FILE")?;
    let mlir = fs::read_to_string(&path)?;
    compile(&mlir)?;
    eprintln!("compiled {}", path.display());
    Ok(())
}

#[cfg(feature = "tpu")]
fn compile(mlir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = PjrtClientHandle::new()?;
    let _exe = client.compile(mlir)?;
    Ok(())
}

#[cfg(not(feature = "tpu"))]
fn compile(_mlir: &str) -> Result<(), Box<dyn std::error::Error>> {
    Err("xla_compile_mlir requires --features tpu".into())
}
