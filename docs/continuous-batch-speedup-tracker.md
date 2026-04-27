# Continuous Batch Speedup Tracker

Goal: make rvLLM serve Gemma 4B/31B at the fastest honest `C=1`, `C=8`, and `C=32` continuous concurrency on one H100, while preserving the fixed-batch speed path as a separate benchmark lane.

Linked context: [current forward model](current-forward-model.md), [profiling guide](profiling.md), [vLLM comparison spec](vllm_comparison_spec.md), [benchmark history](benchmark-history.md), [architecture](arch.md), [autokernel fit](autokernel-fit.md).

Artifact root: `results/continuous_batch_speedup/YYYYMMDD_HHMMSS/`

## Status Key

| Status | Meaning |
|---|---|
| `todo` | not started |
| `active` | in progress |
| `blocked` | waiting on dependency, repro, or hardware |
| `verify` | implementation exists, benchmark/correctness pending |
| `done` | accepted with artifacts |
| `drop` | intentionally abandoned |

## Benchmark Contract

All speed claims must pass this contract before updating docs or bench tables.

| Field | Required value |
|---|---|
| GPU | same H100, clean GPU before each engine |
| Model | same checkpoint path, same quantization, same tokenizer |
| Prompt set | shared prompt file; same prompts for rvLLM and vLLM |
| Decode | greedy, `temperature=0`, `ignore_eos=true`, same `max_new` |
| Targets | `C=1`, `C=8`, `C=32` |
| Metrics | row tok/s, decode-only row tok/s, total generate row tok/s, TTFT, p50/p99 step ms, GPU memory, SM active, memcpy counts |
| Fairness rule | never compare rvLLM same-prompt broadcast against vLLM varied prompts unless both rows are labeled that way |

Known latest H100 reference from the Gemma 4B bring-up: fixed/broadcast `B=30` rvLLM reached about `1563 row tok/s` after graph + device metadata, while vLLM offline `B=30` was about `4400 row tok/s`. Treat this as a blocker signal, not a `C=32` serving result.

## Scoreboard

Fill this only from the benchmark matrix below.

| Target | rvLLM direct row tok/s | rvLLM HTTP row tok/s | vLLM row tok/s | rvLLM/vLLM | p50 step ms | p99 step ms | SM active | Top blocker | Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `C=1` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | benchmark contract first | TBD |
| `C=8` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | benchmark contract first | TBD |
| `C=32` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | benchmark contract first | TBD |

## Benchmark Matrix

| Case | Target | Engine | Path | Command / Harness | Required artifact |
|---|---|---|---|---|---|
| `BM-C1-RV-DIRECT` | `C=1` | rvLLM | direct worker | `RVLLM_BATCH=1 RVLLM_MAX_TOKENS=128 target/release/rvllm-eval` | `direct/rvllm_c1.json` |
| `BM-C8-RV-DIRECT` | `C=8` | rvLLM | direct worker | `RVLLM_BATCH=8 RVLLM_MAX_TOKENS=128 target/release/rvllm-eval` | `direct/rvllm_c8.json` |
| `BM-C32-RV-DIRECT` | `C=32` | rvLLM | direct worker | `RVLLM_BATCH=32 RVLLM_MAX_TOKENS=128 target/release/rvllm-eval` | `direct/rvllm_c32.json` |
| `BM-C1-RV-HTTP` | `C=1` | rvLLM | continuous HTTP | `deploy/benchmark_client.py --concurrent 1 --max-tokens 128` | `http/rvllm_c1.json` |
| `BM-C8-RV-HTTP` | `C=8` | rvLLM | continuous HTTP | `deploy/benchmark_client.py --concurrent 8 --max-tokens 128` | `http/rvllm_c8.json` |
| `BM-C32-RV-HTTP` | `C=32` | rvLLM | continuous HTTP | `deploy/benchmark_client.py --concurrent 32 --max-tokens 128` | `http/rvllm_c32.json` |
| `BM-C1-VLLM` | `C=1` | vLLM | direct/offline | `deploy/vllm_direct_bench.py --concurrency 1 --ignore-eos` | `vllm/vllm_c1.json` |
| `BM-C8-VLLM` | `C=8` | vLLM | direct/offline | `deploy/vllm_direct_bench.py --concurrency 8 --ignore-eos` | `vllm/vllm_c8.json` |
| `BM-C32-VLLM` | `C=32` | vLLM | direct/offline | `deploy/vllm_direct_bench.py --concurrency 32 --ignore-eos` | `vllm/vllm_c32.json` |

## Task Graph

| ID | Status | Target | Depends | Gate |
|---|---|---|---|---|
| [CB-000](#cb-000-freeze-11-benchmark-contract) | `todo` | all | none | same prompts, same tokens, same output length |
| [CB-010](#cb-010-run-baseline-matrix) | `todo` | all | `CB-000` | scoreboard has rvLLM and vLLM rows |
| [CB-020](#cb-020-profile-current-hotspots) | `todo` | all | `CB-010` | nsys kernel/memcpy reports for `C=1,8,32` |
| [CB-030](#cb-030-add-real-context-decode-bench) | `todo` | all | `CB-000` | decode bench supports real context length, not `context_lens=1` only |
| [CB-100](#cb-100-extract-gemma4-worker) | `todo` | all | `CB-020` | `Engine` uses the real Gemma4 path, not a bench-only path |
| [CB-110](#cb-110-continuous-scheduler-and-kv-blocks) | `todo` | all | `CB-100` | request join/finish/cancel releases KV correctly |
| [CB-120](#cb-120-ragged-active-slot-packing) | `todo` | `C=8,C=32` | `CB-110` | live rows compact into graph buckets without KV ownership bugs |
| [CB-130](#cb-130-graph-pool-for-c1-c8-c32) | `todo` | all | `CB-120` | no steady-state graph capture |
| [CB-200](#cb-200-kill-split-global-decode) | `todo` | `C=8,C=32` | `CB-020` | global attention is batched, not per-row launches |
| [CB-210](#cb-210-h100-global-gqa-kernel) | `todo` | all | `CB-200` | specialized `head_dim=512` GQA path beats fallback |
| [CB-220](#cb-220-fa3-small-c-metadata-path) | `todo` | `C=1,C=8,C=32` | `CB-020` | FA3 per-launch metadata no longer top 5 overhead |
| [CB-300](#cb-300-c1-fast-path) | `todo` | `C=1` | `CB-020` | graph + GPU greedy active at `C=1` |
| [CB-310](#cb-310-skip-softcap-for-greedy) | `todo` | all | `CB-300` | identical greedy token IDs, fewer tail kernels |
| [CB-320](#cb-320-generation-lm-head-fp8-path) | `todo` | all | `CB-310` | PPL/token parity passes, row tok/s improves |
| [CB-330](#cb-330-argmax-underfill-fix) | `todo` | `C=1,C=8` | `CB-310` | argmax no longer one CTA per row |
| [CB-400](#cb-400-http-continuous-admission) | `todo` | all | `CB-110` | HTTP `C=1,8,32` within 95% of direct worker |
| [CB-900](#cb-900-correctness-gate) | `todo` | all | active speed item | PPL or token spotcheck passes |
| [CB-910](#cb-910-regression-sweep) | `todo` | all | `CB-900` | full matrix rerun and archived |
| [CB-920](#cb-920-docs-and-bench-update) | `todo` | all | `CB-910` | public docs updated only from artifacts |

## Task Details

### CB-000 Freeze 1:1 Benchmark Contract

Owner: benchmark lane.

Code anchors: [rvllm_eval.rs](../v3/crates/rvllm-bench/src/bin/rvllm_eval.rs), [vllm_direct_bench.py](../deploy/vllm_direct_bench.py), [vLLM comparison spec](vllm_comparison_spec.md).

Work:

- Add shared prompt-file mode to both rvLLM and vLLM harnesses.
- Add same-prompt broadcast mode to vLLM only for labeled broadcast tests.
- Emit one JSON schema for both engines: `engine`, `model`, `concurrency`, `prompt_tokens`, `output_tokens`, `row_tok_per_sec`, `decode_row_tok_per_sec`, `generate_s`, `ttft_ms`, `p50_step_ms`, `p99_step_ms`.
- Store exact env, git SHA, GPU name, driver, CUDA version, and checkpoint path in every artifact.

Verify:

```bash
cd /root/rvllm-single-v3
for C in 1 8 32; do
  RVLLM_BATCH=$C RVLLM_MAX_TOKENS=128 RVLLM_IGNORE_EOS=1 \
  RVLLM_PROMPT_FILE=/root/prompts/gemma4_speed_prompts.txt \
  ./v3/target/release/rvllm-eval
done
```

### CB-010 Run Baseline Matrix

Owner: benchmark lane.

Work:

- Run all nine benchmark matrix rows on the same H100.
- Keep fixed-batch `B=30` as a reference row only.
- Kill stale Python/vLLM/rvLLM processes before each engine.

Gate: `Scoreboard` filled with artifact links and no unlabeled apples-to-oranges rows.

### CB-020 Profile Current Hotspots

Owner: profiling lane.

Code anchors: [profiling guide](profiling.md), [Gemma4 bring-up](../v3/crates/rvllm-runtime/src/gemma4_bring_up.rs).

Work:

- Run `nsys` for `C=1`, `C=8`, and `C=32`.
- Export `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum`, and CUDA API summaries.
- Track `cuStreamSynchronize`, HtoD metadata, DtoH token readback, FA3 time, fallback attention time, LM head time, softcap time, and argmax time.

Verify:

```bash
nsys profile --stats=true --force-overwrite=true -o /tmp/rvllm_c32 \
  env RVLLM_BATCH=32 RVLLM_MAX_TOKENS=128 RVLLM_IGNORE_EOS=1 \
  ./v3/target/release/rvllm-eval
nsys stats --force-export=true -r cuda_gpu_kern_sum /tmp/rvllm_c32.nsys-rep
nsys stats --force-export=true -r cuda_gpu_mem_time_sum /tmp/rvllm_c32.nsys-rep
```

### CB-030 Add Real Context Decode Bench

Owner: benchmark lane.

Code anchors: [run_bench](../v3/crates/rvllm-runtime/src/gemma4_bring_up.rs), [rvllm-bench main](../v3/crates/rvllm-bench/src/main.rs).

Problem: the current raw decode-step bench can use `context_lens=1`, which overstates attention performance versus real generation.

Work:

- Add `RVLLM_BENCH_CONTEXT_LEN`.
- Seed KV/block tables for context lengths `1`, `16`, `128`, `512`, and `2048`.
- Report attention-only and full-layer rows separately.

Gate: raw bench no longer gets quoted without context length.

### CB-100 Extract Gemma4 Worker

Owner: runtime lane.

Code anchors: [Gemma4 bring-up](../v3/crates/rvllm-runtime/src/gemma4_bring_up.rs), [Engine](../v3/crates/rvllm-runtime/src/engine.rs), [Scheduler](../v3/crates/rvllm-runtime/src/scheduler.rs).

Work:

- Extract fixed-batch Gemma4 generation into a resident `Gemma4DecodeWorker`.
- Expose `step_launch(plan)` and `step_collect()` so scheduling can overlap with GPU work.
- Keep fixed-batch `rvllm-eval` on the same worker so direct and serving paths share kernels.

Gate: direct worker reaches at least 95% of the current fixed-batch path at `C=1,8,32`.

### CB-110 Continuous Scheduler And KV Blocks

Owner: scheduler lane.

Code anchors: [scheduler.rs](../v3/crates/rvllm-runtime/src/scheduler.rs), [sched_state.rs](../v3/crates/rvllm-runtime/src/sched_state.rs), [kv_layout.rs](../v3/crates/rvllm-mem/src/kv_layout.rs), [metadata plan](../v3/crates/rvllm-metadata/src/plan.rs).

Work:

- Replace skeletal `BatchPlan` with real prefill/decode plans.
- Add KV block allocate/free per request.
- Finish, EOS, cancel, and disconnect must remove the request from the next plan.
- Add randomized scheduler tests for 10k join/finish/cancel cycles.

Gate: zero KV leaks and no finished request appears in the next plan.

### CB-120 Ragged Active Slot Packing

Owner: scheduler + metadata lane.

Code anchors: [metadata pack](../v3/crates/rvllm-metadata/src/pack.rs), [decode metadata kernel](../v3/kernels/decode_metadata.cu).

Work:

- Compact live requests into graph buckets `1,2,4,8,16,32`.
- Preserve request-to-KV block ownership while physical rows move.
- Pad inactive graph rows without emitting outputs for them.

Gate: churn workload reaches at least 90% of fixed full-bucket throughput.

### CB-130 Graph Pool For C1 C8 C32

Owner: graph lane.

Code anchors: [graph pool](../v3/crates/rvllm-graph/src/pool.rs), [Gemma4 graph gate](../v3/crates/rvllm-runtime/src/gemma4_bring_up.rs).

Work:

- Pre-capture decode graph buckets at startup for `C=1`, `C=8`, and `C=32`.
- Remove request-time graph capture.
- Patch only graph-safe input buffers.

Gate: replay overhead under 50 us/step and no steady-state capture in `nsys`.

### CB-200 Kill Split Global Decode

Owner: attention lane.

Code anchors: [Gemma4 layer exec](../v3/crates/rvllm-runtime/src/gemma4_layer_exec.rs), [paged attention fallback](../v3/kernels/paged_attention_sm89.cu).

Problem: global `head_dim=512` decode currently falls into `split_global_decode`, creating per-row attention launches. At `B=30` and 10 global layers, this can become 300 tiny launches per token instead of 10 batched launches.

Work:

- Add a batched global decode path for Gemma4 global layers.
- Keep the current fallback only behind an explicit env flag.
- Add `nsys` gate for fallback launch count.

Gate: global attention launch count scales by layers, not by `layers * batch`.

### CB-210 H100 Global GQA Kernel

Owner: CUDA attention lane.

Code anchors: [paged_attention_sm89.cu](../v3/kernels/paged_attention_sm89.cu), [FA3 wrapper](../v3/kernels/fa3_sm90_wrapper.cu).

Work:

- Specialize a H100 kernel for Gemma4 global attention, `head_dim=512`, GQA packed by `(batch, kv_head)`.
- Reuse K/V across the 8 query heads that share a KV head.
- Support F16 KV first, then FP8 KV after correctness passes.

Gate: beats the current fallback at `C=1`, `C=8`, and `C=32` with identical token outputs under greedy.

### CB-220 FA3 Small-C Metadata Path

Owner: CUDA attention lane.

Code anchors: [FA3 wrapper](../v3/kernels/fa3_sm90_wrapper.cu), [decode metadata kernel](../v3/kernels/decode_metadata.cu).

Work:

- Avoid per-launch metadata zero/prepare work for homogeneous decode.
- Reuse static scheduler metadata when bucket shape and context pattern allow it.
- Keep a safe dynamic fallback for ragged/churn cases.

Gate: FA3 metadata kernels are not a top 5 cost at `C=1,8,32`.

### CB-300 C1 Fast Path

Owner: latency lane.

Code anchors: [device greedy gate](../v3/crates/rvllm-runtime/src/gemma4_bring_up.rs), [argmax launcher](../v3/crates/rvllm-fused/src/launcher.rs).

Work:

- Enable graph replay and GPU greedy at `C=1`.
- Remove host logits readback from default `C=1`.
- Keep a debug env for host logits readback only.

Gate: `C=1` logs `decode graph enabled` and has no steady-state full-logits DtoH.

### CB-310 Skip Softcap For Greedy

Owner: LM-head lane.

Code anchors: [logit softcap kernel](../v3/kernels/logit_softcap.cu), [Gemma4 generation LM head](../v3/crates/rvllm-runtime/src/gemma4_bring_up.rs).

Reason: `cap * tanh(x / cap)` is monotonic, so greedy argmax does not need the softcap transform.

Work:

- Add greedy-only `RVLLM_NO_SOFTCAP=1` path.
- Compare token IDs with and without softcap.
- Never use this path for sampling until sampling correctness is separately proven.

Gate: identical greedy token IDs and lower LM-head tail time.

### CB-320 Generation LM Head FP8 Path

Owner: LM-head lane.

Code anchors: [Gemma4 generation LM head](../v3/crates/rvllm-runtime/src/gemma4_bring_up.rs), [Gemma4 loader](../v3/crates/rvllm-loader/src/gemma4_load.rs).

Work:

- Reuse the benchmark FP8 channel-scale LM head in generation.
- Remove F16 LM-head weight bandwidth and F32 logits where correctness allows.
- Gate with PPL and token spotchecks because prior FP8 LM-head work regressed.

Gate: PPL/token parity passes and row tok/s improves at `C=8` and `C=32`.

### CB-330 Argmax Underfill Fix

Owner: CUDA tail lane.

Code anchors: [argmax.cu](../v3/kernels/argmax.cu), [fused launcher](../v3/crates/rvllm-fused/src/launcher.rs).

Problem: one CTA per row underfills the H100 badly at `C=1` and `C=8`.

Work:

- Add chunked partial argmax by `(row, vocab_chunk)` plus reduce, or fuse greedy selection into the LM-head tail.
- Keep existing simple argmax as fallback.

Gate: argmax is no longer a visible bottleneck in `C=1`/`C=8` traces.

### CB-400 HTTP Continuous Admission

Owner: serving lane.

Code anchors: [serve main](../v3/crates/rvllm-serve/src/main.rs), [serve lib](../v3/crates/rvllm-serve/src/lib.rs).

Work:

- Wire HTTP admission into the real scheduler/worker.
- Request disconnect maps to cancellation.
- Stream outputs without blocking the decode loop.

Gate: HTTP `C=1,8,32` reaches at least 95% of direct worker throughput.

### CB-900 Correctness Gate

Owner: quality lane.

Work:

- Run PPL smoke on fixed prompt corpus.
- Run deterministic greedy token spotchecks for `C=1`, `C=8`, `C=32`.
- Run compute-sanitizer on a short matrix.

Gate: no speedup is accepted without correctness artifacts.

### CB-910 Regression Sweep

Owner: benchmark lane.

Work:

- Re-run the full benchmark matrix.
- Capture `nvidia-smi`, `nsys`, stdout JSON, stderr logs, env, git SHA.
- Compare against previous artifact root.

Gate: scoreboard updated from artifact JSON only.

### CB-920 Docs And Bench Update

Owner: docs lane.

Work:

- Update `bench.html`, `benchmark-history.md`, and any public docs only from `CB-910`.
- Label fixed-batch, direct continuous, and HTTP continuous separately.
- Keep failed or invalid runs in the artifact notes, not in headline tables.

Gate: docs clearly separate `B` fixed batch from `C` continuous concurrency.

## Sprint Order

1. Do `CB-000`, `CB-010`, and `CB-020` first.
2. If profiles show launch explosion, do `CB-200` then `CB-210`.
3. If profiles show scheduler/metadata gaps, do `CB-100` through `CB-130`.
4. If `C=1` remains bad, do `CB-300`, `CB-310`, and `CB-330`.
5. If tail kernels dominate, do `CB-320`.
6. After any speed item, run `CB-900`, then `CB-910`, then `CB-920`.

## H100 Utilization Target

For `C=32`, the target is real on-card utilization: high SM active, low launch gaps, no per-row global attention, no blocking host sync, and no avoidable HtoD/DtoH in the steady decode loop.

For `C=1`, do not fake a 98% utilization claim with wasted work. The target is minimum latency per generated token. Only use extra on-card parallelism if it buys latency or throughput for a real workload, such as speculative decode, parallel heads/chunked vocab work, or multiple independent requests.
