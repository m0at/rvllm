//! probe-mistral35-layer-ref — deterministic CPU baseline dump.
//!
//! Runs the pure-Rust Mistral 3.5 layer reference
//! (`mistral35_layer_ref::mistral_layer_step`) on a deterministic
//! synthetic input across a small prompt window and writes per-step
//! intermediate tensors to disk as f32 little-endian raw files.
//! The future cosine-validation harness diffs the CUDA forward
//! against these files to bisect kernel-level regressions.
//!
//! No CUDA / no model directory required — everything runs on the
//! CPU. The fixture uses tiny shapes (hidden=8, intermediate=16,
//! 4 heads / 2 KV heads / head_dim=4) so the dumps stay small
//! (~200 B per step) and the harness can re-run in milliseconds.
//!
//! Usage:
//!
//!     cargo run --bin probe-mistral35-layer-ref -- \
//!         --out-dir /tmp/rvllm-mistral35-ref \
//!         [--prompt-len 4] [--seed 42]
//!
//! Output files (under `out_dir`, all f32 LE):
//!   layer0_hidden_in_pos{P}.bin    [hidden_size]
//!   layer0_hidden_out_pos{P}.bin   [hidden_size]
//!   layer0_kv_k.bin                [seq_len, kv_dim]   (final state)
//!   layer0_kv_v.bin                [seq_len, kv_dim]
//!   layer0_summary.txt             newline-delimited shape + checksums
//!
//! Determinism:
//!   - Weights are seeded via SplitMix32 from the `--seed` arg so a
//!     bit-identical re-run produces the same output bytes.
//!   - Activations: token i → x[d] = ((i + 1) * (d + 1)) / 100.0,
//!     a stable per-(token, channel) ramp.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use rvllm_loader::mistral35_arch::YarnRopeConfig;
use rvllm_runtime::mistral35_layer_ref::{
    mistral_layer_step, KvCacheF32, LayerDimsF32, LayerWeightsF32,
};
use rvllm_runtime::mistral35_yarn::build_yarn_rope_tables;

/// SplitMix32 step: deterministic per-call f32 in [0, 1).
fn next_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_add(0x9E3779B9);
    let mut z = *state;
    z = (z ^ (z >> 16)).wrapping_mul(0x85EBCA6B);
    z = (z ^ (z >> 13)).wrapping_mul(0xC2B2AE35);
    z ^= z >> 16;
    // Use top 24 bits → uniform in [0, 1).
    (z >> 8) as f32 / 16_777_216.0
}

/// SplitMix-driven small-magnitude weight buffer in [-0.1, 0.1).
fn random_weights(state: &mut u32, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // Map [0,1) → [-0.1, 0.1).
        out.push(0.2 * next_f32(state) - 0.1);
    }
    out
}

fn parse_arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let prefix = format!("--{name}=");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if let Some(v) = a.strip_prefix(&prefix) {
            return v.parse().unwrap_or(default);
        }
        if a == format!("--{name}") {
            if let Some(v) = args.next() {
                return v.parse().unwrap_or(default);
            }
        }
    }
    default
}

fn parse_path_arg(name: &str, default: &str) -> PathBuf {
    let prefix = format!("--{name}=");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if let Some(v) = a.strip_prefix(&prefix) {
            return PathBuf::from(v);
        }
        if a == format!("--{name}") {
            if let Some(v) = args.next() {
                return PathBuf::from(v);
            }
        }
    }
    PathBuf::from(default)
}

fn write_f32_le(path: &std::path::Path, data: &[f32]) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    let mut buf = Vec::with_capacity(data.len() * 4);
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
}

fn checksum(data: &[f32]) -> f64 {
    // Stable order-independent-ish checksum (sum of magnitudes +
    // sum of values; together these catch sign-bit drifts that
    // a plain sum would miss when individual entries cancel).
    let s: f64 = data.iter().map(|v| *v as f64).sum();
    let a: f64 = data.iter().map(|v| (*v as f64).abs()).sum();
    s + 1e-3 * a
}

fn main() {
    let prompt_len: usize = parse_arg("prompt-len", 4usize);
    let seed: u32 = parse_arg("seed", 42u32);
    let out_dir = parse_path_arg("out-dir", "/tmp/rvllm-mistral35-ref");
    fs::create_dir_all(&out_dir).expect("create out-dir");

    println!(
        "probe-mistral35-layer-ref: prompt_len={prompt_len} seed={seed} \
         out_dir={}", out_dir.display()
    );

    let dims = LayerDimsF32 {
        hidden_size: 8,
        intermediate_size: 16,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        rms_norm_eps: 1e-5,
    };

    let mut state = seed;
    let h = dims.hidden_size;
    let q_dim = dims.q_dim();
    let kv_dim = dims.kv_dim();
    let i = dims.intermediate_size;
    let weights = LayerWeightsF32 {
        w_in_norm:   vec![1.0; h],
        w_q:         random_weights(&mut state, h * q_dim),
        w_k:         random_weights(&mut state, h * kv_dim),
        w_v:         random_weights(&mut state, h * kv_dim),
        w_o:         random_weights(&mut state, q_dim * h),
        w_post_norm: vec![1.0; h],
        w_gate:      random_weights(&mut state, h * i),
        w_up:        random_weights(&mut state, h * i),
        w_down:      random_weights(&mut state, i * h),
    };

    let yarn_cfg = YarnRopeConfig {
        rope_theta: 10_000.0,
        original_max_position_embeddings: 16,
        factor: 1.0,
        beta_fast: 4.0,
        beta_slow: 1.0,
        mscale: 1.0,
        mscale_all_dim: 0.0,
    };
    let rope = build_yarn_rope_tables(&yarn_cfg, dims.head_dim, 16);

    let mut kv = KvCacheF32::default();
    let mut summary = String::new();
    summary.push_str(&format!(
        "dims: hidden={} intermediate={} num_heads={} num_kv_heads={} \
         head_dim={} gqa_ratio={} rms_eps={}\n",
        dims.hidden_size, dims.intermediate_size,
        dims.num_attention_heads, dims.num_key_value_heads,
        dims.head_dim, dims.gqa_ratio(), dims.rms_norm_eps,
    ));
    summary.push_str(&format!("yarn: mscale={} factor={}\n", rope.mscale, yarn_cfg.factor));

    for pos in 0..prompt_len {
        let x: Vec<f32> = (0..h)
            .map(|d| ((pos as f32 + 1.0) * (d as f32 + 1.0)) / 100.0)
            .collect();
        let in_path = out_dir.join(format!("layer0_hidden_in_pos{pos}.bin"));
        write_f32_le(&in_path, &x).expect("write hidden_in");

        let y = mistral_layer_step(&x, pos, &dims, &weights, &mut kv, &rope);
        let out_path = out_dir.join(format!("layer0_hidden_out_pos{pos}.bin"));
        write_f32_le(&out_path, &y).expect("write hidden_out");

        let cs_in = checksum(&x);
        let cs_out = checksum(&y);
        summary.push_str(&format!(
            "pos={pos:>2}  cs_in={cs_in:+.6}  cs_out={cs_out:+.6}  kv_seq={}\n",
            kv.seq_len(kv_dim),
        ));
    }

    let k_path = out_dir.join("layer0_kv_k.bin");
    write_f32_le(&k_path, &kv.k).expect("write kv_k");
    let v_path = out_dir.join("layer0_kv_v.bin");
    write_f32_le(&v_path, &kv.v).expect("write kv_v");
    summary.push_str(&format!(
        "final kv_k.len={} kv_v.len={} cs_k={:+.6} cs_v={:+.6}\n",
        kv.k.len(), kv.v.len(),
        checksum(&kv.k), checksum(&kv.v),
    ));

    let summary_path = out_dir.join("layer0_summary.txt");
    fs::write(&summary_path, &summary).expect("write summary");

    print!("{summary}");
    println!("dumps written to: {}", out_dir.display());
}
