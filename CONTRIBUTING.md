# Contributing to rvLLM

rvLLM is a Rust + CUDA inference engine on GPU and a pure-JAX inference engine on TPU. Both live in this repo. Contributions should land against the **v3 workspace** (`v3/crates/*`) on the GPU side and `tpu/harness/` on the TPU side.

This doc tells you where things are, how to build and test them, and what's open to work on right now.

## Repo layout

```
rvllm/
  v3/                      GPU runtime (Rust + CUDA). All new GPU work lands here.
    Cargo.toml             workspace root -- use --manifest-path v3/Cargo.toml
    crates/
      rvllm-core           typed errors, IDs, dtype, shape, config, env
      rvllm-mem            HbmArena, Region, Stream, Event, PinnedBuf, CudaContextHandle
      rvllm-kernels        SHA-pinned kernel manifest, PTX loader, kernel catalog
      rvllm-fused          fused-kernel launchers + pure-Rust f32 references
      rvllm-attention      FA3 SM90 paged decode/prefill (dlopen)
      rvllm-cutlass        FP8 variant catalog + schedule/epilogue pairing + cuBLASLt wrapper
      rvllm-metadata       frozen-layout metadata per bucket (single upload path)
      rvllm-loader         safetensors mmap -> HBM + CPU-path FP8 quant + clamp gate
      rvllm-sampling       argmax tail, pinned DtoH
      rvllm-graph          captured-graph pool keyed on MetaLayoutHash
      rvllm-runtime        Engine, scheduler, layer_exec, bring_up
      rvllm-serve          OpenAI-compatible HTTP serve loop (Phase D scaffold)
      rvllm-bench          RVLLM_* env-driven bench binary
      rvllm-deploy         SHA-pinned tarball + manifest.json verification (Phase D scaffold)
      rvllm-invariants     DAG-dep test, no-megakernel gate
    specs/                 numbered specs referenced by each crate's Cargo.toml
    GEMMA4_SPEC.md         31B Gemma 4 architecture and weight shapes
    SPEC.md, IMPL_PLAN.md  v3 rewrite plan and agent specs

  kernels/                 .cu sources + build scripts
    build.sh               compiles .cu -> .ptx into kernels/sm_90/
    build_cutlass_so.sh    builds libcutlass_kernels.so
    build_fa3.sh           builds libfa3_kernels.so
    sm_90/                 build output (populated by the three scripts above)

  tpu/harness/             pure-JAX TPU path -- single-file model + server
    gemma4_tpu_infer.py    model + forward pass (E4B / 26B-A4B / 31B auto-detect)
    api_server.py          OpenAI-compatible HTTP server (imports from above)
    eagle3_infer.py        EAGLE-3 speculative-decoding inference
    eagle3_train.py        EAGLE-3 draft head training

  docs/                    architecture, benchmarks, model-support notes
```

The top-level `crates/`, `Makefile`, `bench_harness.py`, and other root-level files predate v3 and are kept only for backward compatibility. **Do not extend them -- extend `v3/crates/*` instead.**

## Setup

```bash
git clone https://github.com/m0at/rvllm.git
cd rvllm

# Workspace compile check (no GPU required, works on macOS)
cargo check --manifest-path v3/Cargo.toml --workspace

# Run the CPU-safe tests (no CUDA)
cargo test  --manifest-path v3/Cargo.toml --workspace
```

`install.sh` installs Rust + Python + tooling if you want a one-shot bootstrap. Read it before running it.

## Building on a GPU box

```bash
# One-time: build kernel artifacts into kernels/sm_90/
bash kernels/build.sh                # fused PTX
bash kernels/build_cutlass_so.sh     # libcutlass_kernels.so
bash kernels/build_fa3.sh            # libfa3_kernels.so

# Build the bench binary
cargo build --release --features cuda --manifest-path v3/Cargo.toml -p rvllm-bench

# Build everything CUDA-enabled
cargo build --release --features cuda --manifest-path v3/Cargo.toml --workspace
```

All rvLLM GPU binaries share the same required env set: `RVLLM_MODEL_DIR`, `RVLLM_KERNELS_DIR`, `RVLLM_CUTLASS_SO`, `RVLLM_FA3_SO`, `RVLLM_POLICY`. See `README.md` for the full run recipe.

## Building on a TPU box

```bash
pip3 install 'jax[tpu]' huggingface_hub tokenizers \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

That's it. No Rust toolchain, no custom kernels. Run the harness directly:

```bash
python3 tpu/harness/gemma4_tpu_infer.py --model-dir ~/models/gemma-4-E4B-it --max-tokens 200
```

## Open contribution tracks

Each of these is a well-scoped, meaningful landing. Pick one, open a PR.

### 1. Additional CUTLASS channelscale tile shapes (GPU)

**Where:** `kernels/cutlass_fp8_gemm_channelscale.cu`, `v3/crates/rvllm-cutlass/`
**Difficulty:** Medium
**Impact:** High -- directly moves B=1..8 throughput

The current CUTLASS channelscale kernel uses a single 128x128x128 tile. For low-batch decode (M <= 16) this leaves perf on the table. Add a 64x64x128 tile variant (same EVT epilogue: per-token ColBroadcast for activation scale, per-channel RowBroadcast for weight scale, F32 accumulate, F16 output), wire it into the FP8 variant catalog in `rvllm-cutlass`, and extend the autotune entry in `policy.json` so small M hits the new tile.

Acceptance: measurable improvement at B=1, B=4, B=8 without regressing B=128+; invariants crate passes.

### 2. GPU OpenAI-compatible HTTP server (`rvllm-serve`)

**Where:** `v3/crates/rvllm-serve/`
**Difficulty:** Medium-Hard
**Impact:** Huge -- removes the only remaining Python in the GPU serving path

`rvllm-serve` is a Phase-D scaffold today (`main.rs` exits with "not yet implemented"). The spec is `v3/specs/05-concurrency.md`. The serve loop should reuse `step_launch` / `step_collect` from `rvllm-runtime`, run a single-worker tokio task against a batch queue, and expose `/v1/completions` and `/v1/chat/completions` with streaming.

Reference shape: `tpu/harness/api_server.py` has the request/response contract already working on the TPU side -- port the surface, not the implementation.

### 3. `rvllm-deploy` tarball + manifest tool

**Where:** `v3/crates/rvllm-deploy/`
**Difficulty:** Medium
**Impact:** Medium -- needed before the GPU stack can be shipped as a pinned artifact

Also a Phase-D scaffold. The spec is `v3/specs/16-deploy.md`. Build: produce a SHA-pinned tarball containing the `rvllm-server` binary, the PTX blobs, `libcutlass_kernels.so`, `libfa3_kernels.so`, `policy.json`, and a `manifest.json` that records the SHA of every artifact plus the git SHA of the build. Verify: load the tarball, recompute SHAs, refuse to start on any mismatch.

No runtime dependency on `rvllm-runtime` -- this crate is build/ship tooling only.

### 4. FA3 prefill path (GPU)

**Where:** `v3/crates/rvllm-attention/`, `kernels/flash_attention_3_prefill.cu`
**Difficulty:** Hard
**Impact:** High -- drops TTFT on long prompts

The current attention path is decode-focused. A first-class prefill variant (chunked over sequence, same SM90 wrapper style) would complete the path and let the scheduler batch prefill and decode cleanly.

### 5. Additional fused-kernel variants

**Where:** `v3/crates/rvllm-fused/`, `v3/crates/rvllm-kernels/` (manifest), `kernels/*.cu`
**Difficulty:** Medium
**Impact:** Medium -- each fusion removes graph nodes and one kernel launch per layer

Candidates visible in `kernels/` today include persistent-kernel variants (`persistent_gemm.cu`, `persistent_layer_*.cu`) and `megakernel_decode.cu`. The discipline: every kernel gets a SHA-pinned entry in the manifest, a launcher in `rvllm-fused`, and a pure-Rust f32 reference test. No dispatch fallback chains.

### 6. EAGLE-3 training data expansion (TPU)

**Where:** `tpu/harness/eagle3_train.py`, `tpu/harness/EAGLE3_SPEC.md`
**Difficulty:** Medium
**Impact:** High -- moves the speculative path from "pipeline validated" to "production tau"

README's measured numbers use a 2K-example draft head (loss 7.1). Production tau needs 50K+ examples. The pipeline is validated end-to-end; the work is data curation + a longer training run + checkpoint delivery.

### 7. New TPU model coverage

**Where:** `tpu/harness/gemma4_tpu_infer.py`
**Difficulty:** Medium
**Impact:** Medium -- `~500 lines of JAX` currently covers three Gemma 4 variants; adding another family (Qwen, Llama, Mistral) exercises the dual-path architecture

Keep it pure JAX. No custom kernels, no torch, no vLLM.

## How to add a CUDA kernel

1. **Write the .cu** under `kernels/`. Keep it self-contained (no headers outside the kernels dir). Use `extern "C"` for any symbol the Rust side dlopens.

2. **Wire it into a build script.** Either extend `kernels/build.sh` (for PTX) or one of the `.so` build scripts (for SM90 CUTLASS / FA3 wrappers). Never bake per-arch flags into the kernel -- the build script decides `-arch=sm_90`.

3. **Pin it in the manifest.** Add a SHA-pinned entry in `v3/crates/rvllm-kernels` so the runtime refuses to load a drifted artifact.

4. **Write a launcher in `rvllm-fused` (or the relevant crate).** The launcher owns the workspace contract, the bucket/layout, and the stream. No `unwrap()` in library code -- return `Result<T, RvllmError>`.

5. **Write a pure-Rust f32 reference test.** Every fused launcher in this repo has one. It must match the CUDA output within tolerance on a pinned input. This is non-negotiable: correctness is gated on the reference path.

6. **Update `policy.json`** if the kernel has tile variants that need autotune selection.

No fallback chains. Missing kernel = engine panic at bring-up, not a runtime surprise.

## How to add a fused kernel launcher in pure Rust

Pattern lives in `v3/crates/rvllm-fused/`. Every launcher:

- Takes a `CudaContextHandle` and a `Stream`.
- Owns a workspace `Region` from the `HbmArena`.
- Reads frozen metadata from `rvllm-metadata` (no per-step allocation).
- Returns `Result<(), RvllmError>`.
- Has a `tests/` entry with a pure-Rust f32 reference against pinned input.

Read two existing launchers before writing a third.

## Testing

```bash
# CPU-safe workspace tests (macOS, CI)
cargo test --manifest-path v3/Cargo.toml --workspace

# CUDA feature tests (H100 box)
cargo test --manifest-path v3/Cargo.toml --workspace --features cuda

# Invariants gate (DAG dependency, no-megakernel)
cargo test --manifest-path v3/Cargo.toml -p rvllm-invariants
```

TPU-side: the harness is a single-file model; run it against a small `--max-tokens` and diff against the HuggingFace BF16 reference (PPL is the canonical signal -- README lists the reference numbers).

## Code style

- `cargo fmt` before committing.
- `cargo clippy --manifest-path v3/Cargo.toml --workspace` must be warning-free.
- All public items need `///` doc comments.
- Use `tracing::{info, debug, warn, error}` for logging, never `println!`.
- Error handling: `Result<T, RvllmError>` end-to-end with structured context. **No `unwrap()` in libraries.** Binaries may `unwrap` at `main` boundaries only.
- All CUDA code behind `#[cfg(feature = "cuda")]`.
- No dispatch fallback chains. Missing variant = panic at bring-up, not silent degradation.

## Project principles

These are enforced, not aspirational:

1. **No fallbacks.** Missing autotune entry = panic. Missing `.so` = refuse start.
2. **Graph-capture invariant.** Metadata buffer layout is frozen per `(bucket, max_blocks_per_seq)`. Captured graphs bind exact offsets.
3. **CUTLASS schedule/epilogue pairing.** Mainloop and epilogue schedules must match -- enforced via `static_assert`.
4. **No `unwrap()` in libraries.** `Result<T, RvllmError>` with structured context, end-to-end.
5. **Real block-change detection.** The scheduler emits block-table updates; missing signals = stale KV reads caught at the type level.

## License

Contributions are accepted under Apache-2.0, matching the repo.
