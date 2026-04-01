use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let zig_dir = PathBuf::from(&manifest_dir).join("../../zig");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let zig_target = match (target_arch.as_str(), target_os.as_str()) {
        ("aarch64", "macos") => "aarch64-macos",
        ("x86_64", "linux") => "x86_64-linux-gnu",
        ("aarch64", "linux") => "aarch64-linux-gnu",
        ("x86_64", "macos") => "x86_64-macos",
        _ => panic!(
            "rvllm-zig: unsupported target {}-{}",
            target_arch, target_os
        ),
    };

    let obj_path = out_dir.join("rvllm_zig.o");
    let lib_path = out_dir.join("librvllm_zig.a");

    // Build object file
    let status = Command::new("zig")
        .arg("build-obj")
        .arg("src/root.zig")
        .arg("-OReleaseFast")
        .arg("-target")
        .arg(zig_target)
        .arg(format!("-femit-bin={}", obj_path.display()))
        .current_dir(&zig_dir)
        .status()
        .expect("rvllm-zig: zig not found on PATH");

    assert!(status.success(), "rvllm-zig: zig build-obj failed");

    // Create archive with proper alignment
    let _ = std::fs::remove_file(&lib_path);
    let status = Command::new("ar")
        .arg("rcs")
        .arg(&lib_path)
        .arg(&obj_path)
        .status()
        .expect("ar not found");

    assert!(status.success(), "rvllm-zig: ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=rvllm_zig");

    println!("cargo:rerun-if-changed=../../zig/src/simd_math.zig");
    println!("cargo:rerun-if-changed=../../zig/src/weight_convert.zig");
    println!("cargo:rerun-if-changed=../../zig/src/root.zig");
}
