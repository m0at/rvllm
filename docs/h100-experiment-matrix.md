# H100 Experiment Matrix

`scripts/h100_experiment_matrix.sh` is a bounded local runner for an already-idle H100 box, including Lambda-style instances. It does not provision machines, call Vast, SSH elsewhere, or run an open-ended loop.

## Build Once

Build kernels and binaries before the matrix run:

```bash
bash kernels/build.sh sm_90
bash kernels/build_cutlass_so.sh sm_90
bash kernels/build_fa3.sh
CUTLASS_DIR=/workspace/cutlass bash kernels/build_w4a8.sh
cargo build --release --features cuda --manifest-path v3/Cargo.toml -p rvllm-bench
cargo build --release --features cuda,cublaslt --manifest-path v3/Cargo.toml -p rvllm-serve
```

If the Rust binaries are not built yet, the runner can do only that part:

```bash
H100_MATRIX_BUILD=1 scripts/h100_experiment_matrix.sh
```

## Required Paths

Set these paths in the shell that runs the matrix:

```bash
export CUDA_ARCH=sm_90
export RVLLM_MODEL_DIR=/workspace/models/gemma-4-31B-it-FP8-Dynamic
export RVLLM_KERNELS_DIR=/workspace/rvllm/kernels
export RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so
export RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so
export RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json
export RVLLM_W4A8_SO=/workspace/rvllm/kernels/sm_90/libw4a8_gemm.so
```

`RVLLM_W4A8_SO` is only required for the `w4a8` experiment preset. The default arena is `74` GB through `H100_MATRIX_ARENA_GB`; lower it if the instance has less free VRAM.

The runner hashes `${RVLLM_KERNELS_DIR}/manifest.json` into every summary row. If the manifest lives somewhere else, set:

```bash
export H100_MATRIX_KERNEL_MANIFEST=/workspace/rvllm/kernels/sm_90/manifest.json
```

## Run

Default run:

```bash
scripts/h100_experiment_matrix.sh
```

This writes logs and summaries under:

```text
logs/h100-experiment-matrix/<utc-run-id>/
```

The main outputs are `summary.jsonl` and `summary.tsv`. Each row includes exact git branch/SHA, kernel manifest path and SHA-256, lane env label, status, elapsed time, and GPU memory used before/after the command.

Default tests:

```text
w4a8_smoke,server,bench,ppl
```

They cover:

- optional `kernels/w4a8_smoke`
- `rvllm-server` readiness, `/health`, `/v1/models`, and an angular-momentum `/v1/chat/completions` prompt
- `rvllm-bench` at `RVLLM_BATCH=1` and `RVLLM_BATCH=128`
- `rvllm-ppl` with one small chunk

Optional bounded memcheck:

```bash
H100_MATRIX_TESTS=w4a8_smoke,w4a8_memcheck scripts/h100_experiment_matrix.sh
```

`w4a8_memcheck` runs `compute-sanitizer --tool memcheck --error-exitcode 99 kernels/w4a8_smoke` with `H100_MATRIX_W4A8_MEMCHECK_TIMEOUT`, default `300` seconds. It is skipped if `compute-sanitizer` or `kernels/w4a8_smoke` is missing.

## Experiment Presets

Run one or more comma-separated variants:

```bash
H100_MATRIX_EXPERIMENTS=baseline,w4a8,rotor_cl3,planar2,iso4 scripts/h100_experiment_matrix.sh
```

Presets:

- `baseline`: unsets `RVLLM_EXPERIMENT_*`, `RVLLM_W4A8`, and `RVLLM_ROTORQUANT`
- `current`: inherits the caller's optional experiment env
- `w4a8`: sets `RVLLM_EXPERIMENT_WEIGHT=w4a8-awq` and `RVLLM_W4A8=1`
- `rotor_cl3`, `planar2`, `iso4`: set `RVLLM_EXPERIMENT_KV=rotorquant` and the corresponding `RVLLM_ROTORQUANT` mode
- `exp:<value>`: sets `RVLLM_EXPERIMENT=<value>`

If any controller or legacy experiment env is already exported and `H100_MATRIX_EXPERIMENTS` is unset, the runner uses `current`. Otherwise it uses `baseline`.

Controller env:

- `RVLLM_EXPERIMENT_WEIGHT`: `fp8-default`, `w4a8-awq`, `awq-metadata-only`
- `RVLLM_EXPERIMENT_KV`: `f16`, `fp8`, `rotorquant`
- `RVLLM_EXPERIMENT_ATTENTION`: `auto`, `fa3`, `fa2-fallback`
- `RVLLM_EXPERIMENT_ARCH`: `auto`, `force-sm75-compat`, `force-hopper`
- `RVLLM_EXPERIMENT_VALIDATION`: `smoke`, `ppl`, `throughput`, `chat`

The runner sets `RVLLM_EXPERIMENT_VALIDATION` per lane so server headers and logs show whether a process was launched for chat, throughput, or PPL validation.
Those lane labels are also written to `summary.jsonl`, `summary.tsv`, and each command log.

Useful bounds:

```bash
H100_MATRIX_B1_ITERS=4 \
H100_MATRIX_B128_ITERS=3 \
H100_MATRIX_PPL_CHUNK=32 \
H100_MATRIX_PPL_CHUNKS=1 \
H100_MATRIX_CHAT_MAX_TOKENS=32 \
scripts/h100_experiment_matrix.sh
```

To run only part of the matrix:

```bash
H100_MATRIX_TESTS=server,bench scripts/h100_experiment_matrix.sh
```

## Avoid Active Loops

Do not keep an SSH session spinning with `while true`, repeated `nvidia-smi`, or tight readiness polling. This runner uses:

- `timeout --kill-after=15s` around bounded commands
- a cleanup trap that stops `rvllm-server` on exit, interrupt, or failure
- readiness polling at one-second cadence with a hard deadline
- finite bench and PPL sizes
- per-command logs instead of live terminal watch loops
- before/after `nvidia-smi` memory snapshots per command, without a watcher process

For long remote work, start the matrix once under `tmux`, `screen`, or `nohup`, then disconnect:

```bash
nohup scripts/h100_experiment_matrix.sh > h100-matrix.out 2>&1 &
```

Check `summary.tsv` or individual logs after it exits rather than keeping an active client loop open.
