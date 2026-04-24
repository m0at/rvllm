# 13 — Benchmarks

## What we measure

Three different scales:

1. **Microbench** — how fast does a single agent run a single task?
2. **Swarm macrobench** — how fast does the whole 30-agent setup finish a
   representative multi-step goal?
3. **Saturator bench** — how fast can the same backend run when the operator
   view stays at 30 agents but the on-card decode target is much larger?

Microbench is mostly a proxy for the native rvllm backend; we include it so a regression in
the backend is obvious from the swarm harness too.

## Microbench

Command:

```bash
cargo run --release --bin swarm-cli -- \
    --agents 1 --bench micro \
    --prompt "Write one short sentence about worktrees." \
    --iterations 30 --warmup 5
```

Emits JSON on stdout:

```json
{
  "kind": "micro",
  "agent": "agent-01",
  "persona": "misc",
  "warmup": 5,
  "iterations": 30,
  "mean_tok_s": 77.8,
  "p50_tok_s": 78.2,
  "p99_tok_s": 62.1,
  "mean_ttft_ms": 31.5,
  "mean_elapsed_s": 2.44
}
```

Target numbers on H100 (Gemma 4 E4B BF16 text checkpoint): `mean_tok_s`
should be in the same neighborhood as the rvLLM bring-up benches. Recent
single-study checks measured about 135 tok/s at B=1 and about 3.9k tok/s at
B=30.

## Swarm macrobench

Command:

```bash
cargo run --release --bin swarm-cli -- \
    --agents 30 --bench swarm \
    --mode operator-30 \
    --goal-file bench/goals/port_tile.yaml
```

Where `port_tile.yaml` is a prebuilt goal with a known subtask
decomposition (to keep the master's cost out of the measurement).

We measure, per goal:

| Metric                     | Target (v1)         | Notes                               |
|----------------------------|---------------------|-------------------------------------|
| end-to-end wall time       | ≤ 8 × single-agent  | 30 visible agents                   |
| tokens per second (swarm)  | ≥ 3k aggregate      | B=30 operator path + overhead       |
| tasks merged / minute      | ≥ 2                 | assuming 20-subtask goals           |
| HBM peak                   | ≤ 45 GB             | 4 live + master, ctx=4k             |
| scheduler starvation       | 0 denies            | no `Deny(AllSlotsBusy)` in journal  |

The `swarm-cli` binary emits a single JSON line per completed goal plus
a final summary line. Pipe it to `jq` or plot with whatever script.

## Saturator bench

Command:

```bash
cargo run --release --bin swarm-cli -- \
    --agents 30 --bench swarm \
    --mode saturator \
    --decode-batch 512 \
    --goal-file bench/goals/port_tile.yaml
```

Recent H100 single-study decode sweeps for the BF16 text 4B path:

| Batch | Decode tok/s |
|-------|--------------|
| 1     | 135          |
| 30    | 3,952        |
| 64    | 7,857        |
| 128   | 14,016       |
| 256   | 22,300       |
| 512   | 29,406       |

Use `operator-30` when correctness and the visible 30-bot workflow matter.
Use `saturator` when the question is whether the H100 is being fed hard
enough.

## Soak test

Command:

```bash
cargo run --release --bin swarm-cli -- \
    --agents 30 --bench soak \
    --goal-file bench/goals/mixed_20x.yaml \
    --duration 1h
```

Runs the same goal 20 times back to back. Pass criteria:

- no worker enters `Failed`,
- HBM does not grow monotonically (no leaks),
- PPL spot-check variance < 1 % across the hour,
- journal replay after the hour reconstructs identical final state.

This is the regression harness we run before merging changes to the
scheduler or the tool sandbox.

## UI bench

We also eyeball the UI's frame budget with the debug overlay (`F9`)
during a swarm macrobench, and the acceptance criterion is **≥ 60 FPS**
steady state with all 30 tiles repainting on every event.

## Where benches live

`bench/` inside the crate:

```
bench/
├── goals/
│   ├── port_tile.yaml
│   └── mixed_20x.yaml
└── micro_prompts.txt
```

These are plain data files, not part of the default build.
