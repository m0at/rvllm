// Codex23-2: emit a compile-time `RVLLM_BUILD_REVISION` env var
// from `git rev-parse --short HEAD`. The PTX manifest carries the
// same string under `revision`; engine init compares and warns on
// drift so a self-consistent stale manifest doesn't silently pair
// with a binary that has a newer launch ABI.
//
// Falls back to "dev" when git is absent (CI tarballs, source-only
// rebuilds). Re-runs whenever HEAD moves so the baked-in value
// tracks the working tree.

use std::process::Command;

fn main() {
    let rev = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "dev".to_string());
    println!("cargo:rustc-env=RVLLM_BUILD_REVISION={rev}");
    // Codex25-3: crate sits in `v3/crates/rvllm-kernels/`, so the
    // repo-root `.git/` is FOUR levels up, not three. The previous
    // 3-level path resolved to `v3/.git/HEAD` (non-existent) so the
    // build.rs never re-ran on commits and the baked revision drifted.
    println!("cargo:rerun-if-changed=../../../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../../../.git/refs/heads");
}
