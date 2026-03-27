// build.rs for rvllm-gpu
//
// When the `cuda` feature is active AND nvcc is found on PATH,
// compile every .cu file in the workspace kernels/ directory to .ptx
// and place outputs in OUT_DIR/ptx/. Downstream code can embed or
// locate these PTX files via the RVLLM_PTX_DIR env var.
//
// On Mac / CI without CUDA toolkit this is a silent no-op.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(p) = env::var("NVCC") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let output = Command::new("which").arg("nvcc").output().ok()?;
    if output.status.success() {
        let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(PathBuf::from(s));
        }
    }
    None
}

fn compile_kernels(nvcc: &Path, kernel_dir: &Path, out_dir: &Path) {
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());

    let ptx_dir = out_dir.join("ptx");
    fs::create_dir_all(&ptx_dir).expect("failed to create ptx output dir");

    let entries = match fs::read_dir(kernel_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }

        let stem = path.file_stem().unwrap().to_str().unwrap();
        let ptx_path = ptx_dir.join(format!("{}.ptx", stem));

        println!("cargo:rerun-if-changed={}", path.display());

        let status = Command::new(nvcc)
            .args(["-ptx", &format!("-arch={}", arch), "-O3", "-o"])
            .arg(&ptx_path)
            .arg(&path)
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Compiled kernel: {}.cu -> {}.ptx", stem, stem);
            }
            Ok(s) => {
                println!(
                    "cargo:warning=nvcc failed for {}.cu (exit {}), skipping",
                    stem,
                    s.code().unwrap_or(-1)
                );
            }
            Err(e) => {
                println!(
                    "cargo:warning=Failed to run nvcc for {}.cu: {}, skipping",
                    stem, e
                );
            }
        }
    }

    println!("cargo:rustc-env=RVLLM_PTX_DIR={}", ptx_dir.display());
}

fn main() {
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    // kernels/ is two levels up from crates/rvllm-gpu/
    let kernel_dir = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("kernels"))
        .unwrap_or_else(|| manifest_dir.join("kernels"));

    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    match find_nvcc() {
        Some(nvcc) => {
            let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
            println!(
                "cargo:warning=Found nvcc at {}, compiling kernels from {}",
                nvcc.display(),
                kernel_dir.display()
            );
            compile_kernels(&nvcc, &kernel_dir, &out_dir);
        }
        None => {
            println!("cargo:warning=nvcc not found -- CUDA kernels will not be compiled");
            println!("cargo:warning=Install CUDA toolkit or set NVCC env var to enable kernel compilation");
        }
    }
}
