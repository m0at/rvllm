# Mistral 3.5 NVFP4 — investigation log (linear, single source of truth)

**Last updated:** 2026-05-10
**Branch:** `rusty_sm121_vision`
**Service:** `rvllm-serve.service`, port 8010

The pre-2026-05-10 stream-of-consciousness log (with contradictory
"verified correct / reverses the verdict / resolved" cycles) is
archived at `.archive/MISTRAL35_BUG_HUNT_pre-2026-05-10.md`. Do
**not** read it as a current diagnosis — pull facts forward into
this file with date + commit + env + prompt + token-IDs + oracle
before treating them as load-bearing.

---

## Current state

The forward path is byte-faithful at every layer/op verified so far
**versus a numpy oracle that dequantizes the W4A16 weights to F32 and
runs the same arithmetic in F32**. End-to-end the model still
produces a universal token attractor (`\n\n\n` or `'aimerais` for
every prompt at greedy temperature=0).

Two interpretations remain on the table; we have NOT distinguished
between them yet:

1. The W4A16-quantized Mistral 3.5 checkpoint genuinely behaves this
   way under greedy decoding. (Plausible — heavy quantization can
   collapse generation to mode-locked attractors.)
2. There is a small systematic numerical bias in our forward that
   accumulates across 88 layers into a wrong-mode attractor.

The next concrete step to settle this is a **W4A16-equivalent
reference run** — vllm or HF transformers loading the same NVFP4
checkpoint, decoding the same prompt with the same sampler. We do
not yet have that reference up and running on this machine.

## Verified oracles (2026-05-10)

Each row: stage / verification recipe / result. Every "verified"
claim must continue to follow this format.

| Stage | Recipe | Result |
|---|---|---|
| Embed gather | rvllm `post_embed` vs HF `embed_tokens.weight[token]` | byte-identical |
| RMSNorm input | rvllm `post_rmsnorm` (layer 0) vs numpy `rsqrt(mean(x^2)+eps)*x*g` | cos = 1.000000 |
| W4A16 GEMV (fused) | rvllm `q_out` (layer 0) vs numpy `e2m1 * e4m3 * (1/gs) / FP4_MAX @ x` (F32 accumulate, BF16 output round) | cos = 0.999996, ratio = 1.000 |
| Multi-key attention | rvllm `attn_out` (layer 0, past_len 1..4) vs numpy multi-key softmax attention | cos = 0.999996+ |
| Final RMSNorm | rvllm `h_after_final_norm` vs numpy `rsqrt(...)*x*g` | cos = 1.0 |
| LM head argmax | rvllm `predicted_token` vs numpy `argmax(h @ lm_head.T)` | identical token ID |

**Caveat (#1 — open):** the numpy oracle uses the *same* numerical
contract as the fused GEMV path (F32 dequant + F32 accumulate +
single BF16 round at the output). This is necessarily self-
consistent. It does **not** prove the contract matches what
vllm / HF produce on the same checkpoint — that requires a real
external W4A16 reference run.

The legacy W4A16 path (`RVLLM_W4A16_GEMV=0`, dequant→BF16→bf16_gemm)
introduces a second BF16 rounding step. The parity probe at
`v3/tools/mistral35_w4a16_gemv_check.py` measures fused-vs-legacy
within `cos ≥ 0.9999`, `rms_diff/rms < 5e-3` — both paths are
internally consistent but diverge from each other at BF16 ULP-level
in the expected direction (legacy is the lossier of the two).

## Open risks

- **Oracle contract not pinned** (#1). The fused F32-accumulate
  contract may diverge from vllm/HF at the systematic-bias level.
  Resolution requires running the same NVFP4 checkpoint through
  vllm v1 (currently broken on this host) or HF transformers, then
  diffing layer 0 / 40 / 80 / 87 q/k/v outputs.
- **Universal token attractor in greedy decode**. Until an external
  W4A16 reference confirms this is the model's real behaviour, we
  cannot distinguish quant-induced from rvllm-induced.
- **Pixtral splice not wired** — vision-bearing requests are
  rejected at admission. Tracked separately in
  `MISTRAL35_PIXTRAL_VISION_PLAN.md`.
- **Greedy-only sampler**. Non-greedy requests are rejected. Top-k /
  top-p / temperature need wiring before any sampling-driven
  diversity test.

## What NOT to claim without proof

- "rvllm output matches vllm" — we have no working vllm reference
  on this host as of 2026-05-10.
- "the model output is correct" — we have no end-to-end string-
  level reference; only per-op numerical agreement against our own
  numpy oracle, which uses the same contract.
- "the bug is fixed" — until #1 is resolved we cannot say there is
  or isn't a bug to fix.

## Format for new entries

When adding a verification, record exactly:

- Date (ISO)
- Commit SHA (short)
- Env vars in effect (full list of `RVLLM_*` set at run time)
- Prompt (verbatim, including chat template if any)
- Tokenized prompt IDs (for reproducibility)
- Compared-against oracle (numpy script path / vllm version /
  HF transformers version / etc)
- Numerical result (cos / rms ratio / max abs diff)

If any of these is missing, the entry is a hypothesis, not a
verification. File hypotheses under "Open risks" or in a separate
working note, not under "Verified oracles".
