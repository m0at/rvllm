# MiniMax-M2.7-NVFP4 v6e-8 Full Equivalent Bench

Run commit: `bb800cc216ca5e0caf895a78e6ec97a23a60ede4`

Remote output directory: `/tmp/m2_full_equiv_bb800cc21`

Runtime:

- TPU: v6e-8, `europe-west4-a`
- Context: 2048
- KV cache: int8
- MoE: `M2_MOE=shardmap`, `RVLLM_M2_MOE_IMPL=auto`
- Prompt: `Explain angular momentum.`
- Gate generation: 256 tokens

Batch sweep:

| Batch | ms/step | tok/s | Status |
|---:|---:|---:|---|
| 1 | 726.75 | 1.38 | pass |
| 8 | 145.21 | 55.09 | pass |
| 16 | 154.63 | 103.47 | pass |
| 32 | 186.87 | 171.25 | pass |
| 64 | - | - | OOM: 992 MB allocation with 850 MB free |

Correctness gate:

- PPL: 6.73205 on 2047 scored tokens
- Candidate PPL delta vs baseline: 0.0
- Generation common prefix: 736 chars
- Gate result: pass

Generation note:

The 256-token sample starts coherently, then falls into repeated math text
around `\vec{p}`. Treat this as: model numerics are working and the gate passes,
but long-form coherence still needs decode/sampling/spec-decode follow-up.
