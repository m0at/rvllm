//! Integration test that locks the Mistral 3.5 layer reference's
//! byte-for-byte determinism into CI.
//!
//! Two paths run side-by-side:
//!
//! 1. Direct: call `mistral_layer_step` twice with identical
//!    inputs + weights and compare the f32 outputs byte-for-byte
//!    (mantissa drift = bug).
//! 2. Probe binary: build & exec
//!    `probe-mistral35-layer-ref` against two distinct
//!    `--out-dir` paths with the same seed and `diff -r` the
//!    resulting f32 dumps.
//!
//! Path 1 catches refactor drift inside the layer-ref module; path
//! 2 catches drift in the probe binary's argument parsing,
//! checksum, and file-write code. Together they pin the entire
//! "deterministic CPU baseline" surface so a future cosine
//! validation harness has a stable reference.
//!
//! No CUDA / no model-dir needed. Path 2 is `#[cfg_attr]`-skipped
//! when `cargo` cannot reach the probe binary (e.g. a workspace
//! reduce that omits the bin entry); path 1 always runs.

use std::path::PathBuf;
use std::process::Command;

use rvllm_loader::mistral35_arch::YarnRopeConfig;
use rvllm_runtime::mistral35_layer_ref::{
    mistral_layer_step, KvCacheF32, LayerDimsF32, LayerWeightsF32,
};
use rvllm_runtime::mistral35_yarn::build_yarn_rope_tables;

fn fixture_dims() -> LayerDimsF32 {
    LayerDimsF32 {
        hidden_size: 8,
        intermediate_size: 16,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        rms_norm_eps: 1e-5,
    }
}

fn fixture_yarn() -> YarnRopeConfig {
    YarnRopeConfig {
        rope_theta: 10_000.0,
        original_max_position_embeddings: 16,
        factor: 1.0,
        beta_fast: 4.0,
        beta_slow: 1.0,
        mscale: 1.0,
        mscale_all_dim: 0.0,
    }
}

/// Same SplitMix32 PRNG the probe binary uses, repeated here so
/// the test stays self-contained (the probe binary's helper is
/// in a `[[bin]]`-private module).
fn next_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_add(0x9E3779B9);
    let mut z = *state;
    z = (z ^ (z >> 16)).wrapping_mul(0x85EBCA6B);
    z = (z ^ (z >> 13)).wrapping_mul(0xC2B2AE35);
    z ^= z >> 16;
    (z >> 8) as f32 / 16_777_216.0
}

fn weights_for_seed(seed: u32, dims: &LayerDimsF32) -> LayerWeightsF32 {
    let mut state = seed;
    let h = dims.hidden_size;
    let q_dim = dims.q_dim();
    let kv_dim = dims.kv_dim();
    let i = dims.intermediate_size;
    let rand = |state: &mut u32, n: usize| -> Vec<f32> {
        (0..n).map(|_| 0.2 * next_f32(state) - 0.1).collect()
    };
    LayerWeightsF32 {
        w_in_norm:   vec![1.0; h],
        w_q:         rand(&mut state, h * q_dim),
        w_k:         rand(&mut state, h * kv_dim),
        w_v:         rand(&mut state, h * kv_dim),
        w_o:         rand(&mut state, q_dim * h),
        w_post_norm: vec![1.0; h],
        w_gate:      rand(&mut state, h * i),
        w_up:        rand(&mut state, h * i),
        w_down:      rand(&mut state, i * h),
    }
}

fn run_layer_chain(seed: u32, prompt_len: usize) -> Vec<Vec<f32>> {
    let dims = fixture_dims();
    let weights = weights_for_seed(seed, &dims);
    let rope = build_yarn_rope_tables(&fixture_yarn(), dims.head_dim, 16);
    let mut kv = KvCacheF32::default();
    let mut outs = Vec::with_capacity(prompt_len);
    for pos in 0..prompt_len {
        let x: Vec<f32> = (0..dims.hidden_size)
            .map(|d| ((pos as f32 + 1.0) * (d as f32 + 1.0)) / 100.0)
            .collect();
        outs.push(mistral_layer_step(&x, pos, &dims, &weights, &mut kv, &rope));
    }
    outs
}

#[test]
fn layer_ref_is_byte_deterministic_across_runs() {
    // Identical seed → identical outputs, bit-for-bit.
    let a = run_layer_chain(42, 4);
    let b = run_layer_chain(42, 4);
    assert_eq!(a.len(), b.len());
    for (i, (av, bv)) in a.iter().zip(&b).enumerate() {
        assert_eq!(av.len(), bv.len(), "pos {i} length mismatch");
        for (j, (x, y)) in av.iter().zip(bv).enumerate() {
            assert_eq!(
                x.to_le_bytes(), y.to_le_bytes(),
                "pos {i} elem {j}: {} != {} (bit-level drift)", x, y
            );
        }
    }
}

#[test]
fn layer_ref_distinguishes_seeds() {
    // Different seeds → different outputs (sanity: prevents a
    // "byte-identical because the function is constant" bug).
    let a = run_layer_chain(42, 4);
    let b = run_layer_chain(43, 4);
    let mut any_diff = false;
    for (av, bv) in a.iter().zip(&b) {
        for (x, y) in av.iter().zip(bv) {
            if (x - y).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        if any_diff { break; }
    }
    assert!(any_diff, "seed 42 and 43 produced identical layer outputs");
}

/// Path 2: build + exec the probe binary; verify byte-identical
/// dumps across two runs with the same seed. Skipped when the
/// CARGO env var isn't set (the integration runner sets it
/// automatically; manual `cargo test --no-run` invocations might
/// not).
#[test]
fn probe_binary_dumps_are_reproducible() {
    let cargo = match std::env::var_os("CARGO") {
        Some(c) => c,
        None => {
            eprintln!("skip: CARGO env not set");
            return;
        }
    };

    let tmp_root: PathBuf = std::env::temp_dir().join(format!(
        "rvllm-mistral35-it-probe-{}", std::process::id()
    ));
    let dir_a = tmp_root.join("a");
    let dir_b = tmp_root.join("b");
    let _ = std::fs::remove_dir_all(&tmp_root);

    let run_once = |out_dir: &PathBuf| {
        let status = Command::new(&cargo)
            .args([
                "run", "--release", "--quiet",
                "-p", "rvllm-runtime",
                "--bin", "probe-mistral35-layer-ref",
                "--",
                "--out-dir", out_dir.to_str().unwrap(),
                "--prompt-len", "4",
                "--seed", "42",
            ])
            .status()
            .expect("spawn cargo run probe");
        assert!(status.success(), "probe binary exited non-zero");
    };
    run_once(&dir_a);
    run_once(&dir_b);

    // Compare every pair of files byte-for-byte.
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir_a)
        .expect("read dir_a")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "probe produced no files");

    for pa in &paths {
        let name = pa.file_name().unwrap();
        let pb = dir_b.join(name);
        let bytes_a = std::fs::read(pa).expect("read a");
        let bytes_b = std::fs::read(&pb).expect("read b");
        assert_eq!(
            bytes_a, bytes_b,
            "probe dump {} differs across runs (probe-binary determinism broken)",
            name.to_string_lossy()
        );
    }

    // Cleanup — failures leave the dir for inspection.
    let _ = std::fs::remove_dir_all(&tmp_root);
}
