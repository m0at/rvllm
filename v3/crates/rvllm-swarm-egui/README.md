# rvLLM Swarm Commander

Native `egui` control surface for a 30-agent rvLLM swarm running on a single
H100. One **master 4B controller** decomposes goals and dispatches work;
**30 4B worker agents** execute with real filesystem, git, and shell tools
inside isolated git worktrees. The GUI is tuned for a 45" monitor (6x5 agent
tile grid + master strip + dispatch/log side panel).

This crate is intentionally **standalone** inside the rvllm workspace: the
build does not require editing any files outside this directory. The real
backend is the native rvllm runtime behind `--features cuda`.

## Topology

```
            +-------------------------------------------------------+
            |  MASTER 4B AGENT  (controller + decomposer)           |
            |    - owns task queue + tasks.jsonl journal            |
            |    - owns git main branch, authorizes merges          |
            |    - picks workers by load, persona, worktree locality|
            +----+--------------+--------------+--------------+-----+
                 |              |              |              |
          +------v---+   +------v---+   +------v---+   +------v---+
          | WORKER 1 |   | WORKER 2 |   |   ...    |   | WORKER 30|
          | persona  |   | persona  |   |          |   | persona  |
          | worktree |   | worktree |   |          |   | worktree |
          | rhai env |   | rhai env |   |          |   | rhai env |
          +----------+   +----------+   +----------+   +----------+
```

All 31 logical agents are 4B models (Gemma 4 E4B). Because one H100 can only
hold a small number of KV caches at once, the scheduler keeps `N_LIVE` agents
resident (default 4) and swaps others on an LRU/priority basis. This is
multiplexed concurrency, not parallel execution.

For operator fanout tests, the master strip also supports a **single-submit
broadcast** path: one input goes in once, one shared response is generated,
and that response is stamped across all 30 agent histories/tasks without
issuing 30 separate submissions.

## Run (mock backend, any machine)

```bash
cd v3/crates/rvllm-swarm-egui
cargo run --release
```

Launches with 30 mock 4B workers. Useful for iterating on UI and scheduler.

## Run (real backend, H100)

Known-good E4B FP8 mode uses the FP8 checkpoint with the first 9 PLI layers
dequantized to F16. Full all-layer FP8 is a known-bad diagnostic mode; it
produces junk unless `RVLLM_UNSAFE_FULL_FP8=1` is set deliberately.

```bash
cd v3/crates/rvllm-swarm-egui
RVLLM_MODEL_DIR=/workspace/models/gemma-4-E4B-it-FP8 \
RVLLM_KERNELS_DIR=/workspace/rvllm/kernels/sm_90 \
RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so \
RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so \
RVLLM_FA_FALLBACK_SO=/workspace/rvllm/kernels/sm_90/libfa_sm89_kernels.so \
RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json \
RVLLM_F16_LAYERS=9 \
RVLLM_F16_KV=1 \
RVLLM_DECODE_BATCH_TARGET=30 \
cargo run --release --features cuda
```

Before trusting the path after kernel/runtime edits, run the H100 gate:

```bash
RVLLM_ROOT=/workspace/rvllm ./deploy/run_4b_fp8_gate.sh
```

## Headless

```bash
cargo run --release --bin swarm-cli -- --goal "wire worker pool"
cargo run --release --bin swarm-cli -- --broadcast "answer this once for all 30 agents"
```

## Opt-in to the workspace (optional)

If you want `cargo build -p rvllm-swarm-egui` to work from the repo root, add
`"crates/rvllm-swarm-egui"` to `v3/Cargo.toml`'s `[workspace] members`. This
crate is written to work either way.

## Design docs

See [`docs/00_OVERVIEW.md`](docs/00_OVERVIEW.md) and onward. Each doc is a
self-contained handoff for one sub-swarm implementation agent.

## License

Apache-2.0, same as the rest of rvllm.
