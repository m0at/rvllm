# Experiment Go/No-Go Summary

Branch: `codex/awq-rotorquant`

This branch is a control-plane and metadata groundwork branch. It should not be
read as production benchmark evidence for SM75, AWQ/W4A8 serving, or
RotorQuant serving. The H100 runs below verify the baseline 31B serving path,
experiment gates, W4A8 kernel smoke, and bounded regression lanes.

## Top Line

| Lane | Decision | Why |
|---|---|---|
| SM75 / T4 | No-go for serving | `sm_75` is modeled and audited, but there is no SM75 PTX manifest, validated attention backend, or non-Hopper W4A8/FP8 kernel route. |
| AWQ / W4A8 | No-go for production serving | The direction is good, and W4A8 kernel/binding work plus a research dispatch harness exist. Real calibrated AWQ tensor ingest, packing, and production dispatch do not. |
| RotorQuant | No-go for serving | Metadata, KV layout sizing, fallback policy, and CPU reference helpers exist; attention kernels and runtime KV integration do not. |
| Serving validation | Go for baseline and instrumentation | H100 baseline/current/W4A8-gated/RotorQuant-gated lanes pass health, models, chat, B=1/B=128 throughput, and PPL smoke. |

## SM75 / T4

Implemented:

- Compute capability 7.5 maps to `sm_75`.
- The capability matrix says SM75 has no native FP8 tensor cores, W4A8 CUTLASS,
  FA3, FA2 PTX, or RotorQuant KV support yet.
- The experiment controller can label a forced SM75 compatibility lane.
- The static kernel audit and FA2 fallback plan are documented in
  `docs/sm75-support.md`.

No-go blockers:

- Build and ship an `sm_75` PTX manifest.
- Convert the static audit into an SM75 compile matrix.
- Define and test a T4-compatible attention fallback.

## AWQ / W4A8

The suggestion to add AWQ/W4A8 is good: it targets the right memory-bandwidth
pressure point for H100. The current branch should still be read as groundwork
and research isolation, not incorporated performance work.

Implemented:

- AWQ metadata detection covers `qweight/qzeros/scales/g_idx`, packed
  `weight/scale/zero_point` names, and common quantization config fields.
- Metadata validation tracks bits, group size, zero-point presence, ready layer
  count, config-only checkpoints, and mixed formats.
- Loader-side W4A8 candidate classification distinguishes ready, metadata-only,
  missing, mixed-layout, and invalid-config tensor sets.
- Tiny CPU reference helpers unpack/dequantize 4-bit AWQ `qweight`, `qzeros`,
  and `scales` tensors for parity fixtures.
- W4A8 shared-object loading is gated behind `RVLLM_EXPERIMENT_WEIGHT=w4a8-awq`
  or legacy `RVLLM_W4A8=1`.
- Flag-only H100 `w4a8` lanes verify experiment labels and fallback behavior.
  They were not active W4A8 layer dispatch.
- A real-dispatch research harness can encode F16 source weights with
  `RVLLM_W4A8_ENCODE_LAYERS` and route selected modules via
  `RVLLM_W4A8_MODULES`.
- The experiment controller rejects W4A8 under forced SM75 compatibility.
- The H100 runner has a W4A8 preset and optional W4A8 smoke executable slot.

No-go blockers:

- Load real AWQ tensors into the typed W4A8 weight slots.
- Add runtime candidate checks for supported shapes, groups, and tensor coverage.
- Replace the current symmetric F16-source encoder with calibrated AWQ packing.
- Fix or reject the symmetric real-dispatch path until PPL smoke is sane; H100
  isolation found all tested modules explode PPL versus baseline.
- Add true W4A8-vs-baseline serving benchmarks after production AWQ packing
  exists.

Metadata-only verdict:

- `qweight/qzeros/scales/g_idx` and packed `weight + scale + optional zero_point`
  formats are detected.
- Config-only AWQ is detected but not serving-ready.
- Mixed or partial tensor sets are inspection results, not accepted serving
  routes.
- Symmetric INT4 from F16 source weights is an isolation tool, not AWQ.

Real-dispatch research smoke on H100, May 17 2026:

| Modules | PPL |
| --- | ---: |
| FP8 baseline | 49.3464 |
| qkv, 1 W4A8 layer | 39842466772750.7891 |
| o, 1 W4A8 layer | 4765193586.1566 |
| gate_up, 1 W4A8 layer | 15948788191217.1602 |
| down, 1 W4A8 layer | 11196819598943.0723 |
| down, 8 W4A8 layers | 2520490153081667.0000 |

Down-proj-only synthetic throughput was fast at 8 layers (`254.9 tok/s` B=1,
`25468.8 tok/s` B=128), but quality is unusable, so this remains a research
lane only.
Calibrated AWQ packing is required before presenting this as incorporated
W4A8/AWQ performance work.

## RotorQuant

Implemented:

- RotorQuant metadata validates mode, bits, chunk size, residual bits, and
  codebook sizing.
- RotorQuant KV layout sizing summarizes values, residuals, rotation params,
  and codebook bytes.
- V1 fallback policy keeps prefill on FP8/F16 and decode on FP8/F16 until
  kernels exist.
- Tiny CPU reference helpers cover packing/unpacking, codebook dequantization,
  Planar2 rotation, and Iso4 rotation round trips.
- Controller gates exist for `rotor_cl3`, `planar2`, and `iso4`.

No-go blockers:

- Add decode-only attention integration behind an off-by-default gate.
- Wire the layout into runtime KV allocation and attention.
- Add H100 decode smoke once kernels exist.
- Measure encode/decode overhead.

Residual bits decision:

- Residual bits stay metadata-only for this branch. They are deferred from the
  first serving integration.

## Serving Validation

Implemented:

- Server responses can stamp experiment headers from env.
- The aggregate `x-rvllm-experiment` metadata value is available for logs and
  response inspection.
- The matrix runner can exercise health, models, chat, B=1 throughput,
  B=128 throughput, and a small PPL lane.
- The runner records per-test memory before/after, branch SHA, and kernel
  manifest SHA when run with the updated script.

## H100 Matrix Results

Primary run:

- Run ID: `codex-20260517T133727Z`
- H100 scratch path: `/workspace/runs/awq-rotorquant/rvllm-branch`
- Summary path: `logs/h100-experiment-matrix/codex-20260517T133727Z/summary.jsonl`
- Variants: `baseline`, `w4a8`, `rotor_cl3`
- Tests: `w4a8_smoke`, `server`, `bench`, `ppl`
- Result: 22/22 summary rows passed.

Measured smoke metrics:

| Variant | B=1 tok/s | B=1 ms/step | B=128 tok/s | B=128 ms/step | PPL |
|---|---:|---:|---:|---:|---:|
| baseline | 48.4 | 20.6549 | 5230.3 | 24.4730 | 37.4067 |
| w4a8 gate | 48.7 | 20.5463 | 5204.9 | 24.5921 | 37.4067 |
| rotor_cl3 gate | 48.4 | 20.6522 | 5213.3 | 24.5524 | 37.4067 |

Angular momentum chat response, all variants:

> When a figure skater pulls in their arms, they decrease their moment of inertia, which causes their rotational speed to increase to keep their total angular momentum constant.

Metadata and sanitizer run:

- Run ID: `codex-finalmeta-20260517T134744Z`
- Tests: `w4a8_smoke`, `w4a8_memcheck`, `server`
- Result: all summary rows passed.
- W4A8 smoke: `max_abs=0.000000`, `workspace=0`.
- Compute sanitizer: `ERROR SUMMARY: 0 errors`.
- Kernel manifest: `/workspace/runs/awq-rotorquant/rvllm-branch/kernels/sm_90/manifest.json`.
- Kernel manifest SHA-256: `e6c305ed4f6f7d4583e578609fe78d660619c212d4814bbf1aad12a00b9258df`.
- GPU memory before/after values are populated for each summary row.

Interpretation:

- 31B serving is healthy on H100 for the bounded regression lanes.
- The W4A8 and RotorQuant numbers are not quantized production serving numbers;
  the flag-only W4A8 lanes verify experiment plumbing and fallback behavior.
- The separate real-dispatch W4A8 smoke is a research harness using symmetric
  INT4, not AWQ.
- SM75, real AWQ dispatch, and RotorQuant attention remain no-go until the
  blockers above are implemented and separately benchmarked.

Expected source of truth after a run:

- `logs/h100-experiment-matrix/<run-id>/summary.jsonl`
- `logs/h100-experiment-matrix/<run-id>/summary.tsv`
- Per-test logs under the same run directory
