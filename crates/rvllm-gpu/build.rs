// build.rs for rvllm-gpu
//
// When the `cuda` feature is active AND nvcc is found on PATH,
// compile every .cu file in the workspace kernels/ directory to .ptx
// and place outputs in OUT_DIR/ptx/. Downstream code can embed or
// locate these PTX files via the RVLLM_PTX_DIR env var.
//
// Architecture selection:
//   1. CUDA_ARCH env var -- single ("sm_90") or comma-separated ("sm_80,sm_90,sm_120")
//   2. Auto-detect via `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
//   3. Fallback to sm_80
//
// Supported architectures: sm_70 (Volta), sm_75 (Turing), sm_80/sm_86/sm_89 (Ampere),
// sm_90/sm_90a (Hopper), sm_100/sm_100a (Blackwell data-center),
// sm_120/sm_122 (Blackwell consumer).
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

/// Map a compute capability string like "12.0" to an sm_ arch string.
fn cc_to_sm(cc: &str) -> Option<&'static str> {
    match cc.trim() {
        "7.0" => Some("sm_70"),
        "7.5" => Some("sm_75"),
        "8.0" => Some("sm_80"),
        "8.6" => Some("sm_86"),
        "8.9" => Some("sm_89"),
        "9.0" => Some("sm_90"),
        "10.0" => Some("sm_100"),
        "12.0" => Some("sm_120"),
        "12.2" => Some("sm_122"),
        _ => None,
    }
}

/// Auto-detect GPU architectures via nvidia-smi.
/// Returns a deduplicated list of sm_ strings for all installed GPUs.
fn detect_gpu_archs() -> Vec<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut archs: Vec<String> = stdout
        .lines()
        .filter_map(|line| cc_to_sm(line).map(String::from))
        .collect();
    archs.sort();
    archs.dedup();
    archs
}

/// Resolve the list of target architectures to compile for.
fn resolve_archs() -> Vec<String> {
    // Explicit env var takes priority (comma-separated)
    if let Ok(val) = env::var("CUDA_ARCH") {
        let archs: Vec<String> = val
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if !archs.is_empty() {
            return archs;
        }
    }

    // Auto-detect from installed GPUs
    let detected = detect_gpu_archs();
    if !detected.is_empty() {
        return detected;
    }

    // Fallback
    vec!["sm_80".to_string()]
}

fn compile_kernels(nvcc: &Path, kernel_dir: &Path, out_dir: &Path) {
    let archs = resolve_archs();
    println!(
        "cargo:warning=Target CUDA architectures: {}",
        archs.join(", ")
    );

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
        println!("cargo:rerun-if-changed={}", path.display());

        for arch in &archs {
            // Multi-arch: suffix PTX with arch. Single-arch: keep original name.
            let ptx_name = if archs.len() == 1 {
                format!("{}.ptx", stem)
            } else {
                format!("{}_{}.ptx", stem, arch)
            };
            let ptx_path = ptx_dir.join(&ptx_name);

            // Cooperative-groups kernels need cubin (not PTX) to preserve
            // grid-level sync. PTX compilation downgrades grid.sync() to
            // block-level bar.sync which silently breaks the kernel.
            let needs_cubin = stem == "persistent_layer_decode" || stem == "megakernel_decode";
            let arch_flag = format!("-arch={}", arch);

            if needs_cubin {
                // Compile to cubin for cooperative launch
                let cubin_name = if archs.len() == 1 {
                    format!("{}.cubin", stem)
                } else {
                    format!("{}_{}.cubin", stem, arch)
                };
                let cubin_path = ptx_dir.join(&cubin_name);
                let mut nvcc_cmd = Command::new(nvcc);
                nvcc_cmd.args(["-cubin", &arch_flag, "-O3", "--use_fast_math"]);
                nvcc_cmd.arg("-Xptxas");
                nvcc_cmd.arg("-v");
                nvcc_cmd.arg("-o").arg(&cubin_path).arg(&path);
                let output = nvcc_cmd.output();
                match output {
                    Ok(o) if o.status.success() => {
                        println!(
                            "cargo:warning=Compiled kernel: {}.cu -> {} (cubin, {})",
                            stem, cubin_name, arch
                        );
                        // Surface register usage / spill stats from nvcc -Xptxas -v
                        let stderr = String::from_utf8_lossy(&o.stderr);
                        for line in stderr.lines() {
                            println!("cargo:warning=[ptxas] {}", line);
                        }
                    }
                    Ok(o) => {
                        println!(
                            "cargo:warning=nvcc cubin failed for {}.cu [{}] (exit {}), skipping",
                            stem, arch, o.status.code().unwrap_or(-1)
                        );
                        let stderr = String::from_utf8_lossy(&o.stderr);
                        for line in stderr.lines() {
                            println!("cargo:warning=[nvcc stderr] {}", line);
                        }
                    }
                    Err(e) => {
                        println!(
                            "cargo:warning=Failed to run nvcc for {}.cu [{}]: {}, skipping",
                            stem, arch, e
                        );
                    }
                }
            } else {
                let mut nvcc_cmd = Command::new(nvcc);
                nvcc_cmd.args(["-ptx", &arch_flag, "-O3"]);
                nvcc_cmd.arg("-o").arg(&ptx_path).arg(&path);
                let status = nvcc_cmd.status();
                match status {
                    Ok(s) if s.success() => {
                        println!(
                            "cargo:warning=Compiled kernel: {}.cu -> {} ({})",
                            stem, ptx_name, arch
                        );
                    }
                    Ok(s) => {
                        println!(
                            "cargo:warning=nvcc failed for {}.cu [{}] (exit {}), skipping",
                            stem, arch, s.code().unwrap_or(-1)
                        );
                    }
                    Err(e) => {
                        println!(
                            "cargo:warning=Failed to run nvcc for {}.cu [{}]: {}, skipping",
                            stem, arch, e
                        );
                    }
                }
            }
        }
    }

    println!("cargo:rustc-env=RVLLM_PTX_DIR={}", ptx_dir.display());

    // Also copy PTX files to workspace target/ptx/ where the runtime loader
    // checks first (avoids hash-dependent build output paths).
    let workspace_ptx = out_dir
        .ancestors()
        .find(|p| p.ends_with("target") || p.join("release").exists() || p.join("debug").exists())
        .and_then(|p| {
            // Walk up to target/
            let mut cur = p;
            while !cur.ends_with("target") {
                cur = cur.parent()?;
            }
            Some(cur.to_path_buf())
        })
        .unwrap_or_else(|| out_dir.join("../../.."))
        .join("ptx");

    if let Err(e) = fs::create_dir_all(&workspace_ptx) {
        println!("cargo:warning=Could not create {}: {e}", workspace_ptx.display());
    } else {
        if let Ok(entries) = fs::read_dir(&ptx_dir) {
            for entry in entries.flatten() {
                let src = entry.path();
                if matches!(src.extension().and_then(|e| e.to_str()), Some("ptx") | Some("cubin")) {
                    let dst = workspace_ptx.join(src.file_name().unwrap());
                    if let Err(e) = fs::copy(&src, &dst) {
                        println!("cargo:warning=Failed to copy {} -> {}: {e}", src.display(), dst.display());
                    }
                }
            }
        }
    }
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
