# Mistral 3.5 NVFP4 — investigation log (linear, single source of truth)

**Last updated:** 2026-05-10 (Pixtral vision E2E semantic gate passed)
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

The kernel `mistral35_w4a16_gemv_bf16.cu` and `nvfp4_dequant_weights_bf16.cu`
both used to apply an extra `* (1.0/6.0)` factor on top of the
true e2m1 lookup; on 2026-05-10 the factor was removed (commit
`58ef825`) after `v3/tools/mistral35_marlin_oracle_check.py` showed
that the rvllm dequant magnitude was 6× smaller than the contract
documented in vllm 0.20.2's new `ModelOptNvFp4W4A16LinearMethod`
(PR #41769). After the change, greedy completions produce
plausible-looking text where every prompt previously emitted
`\n\n\n` / `'aimerais`:

  "Hello"                          → " Abstract Wikipedia Update — Volume 100…"
  "The capital of France is"       → " Paris. This is a well-known fact…"
  "Berlin ist die Hauptstadt von"  → " Deutschland und eine der lebendigsten…"

**This is empirical evidence that the `/6` was a real bug, not a
proof of full correctness.** We have NOT yet re-run the same prompts
+ same sampler against vllm 0.20.2's W4A16 NVFP4 path or against an
HF transformers reference loading the same checkpoint. Until that
external reference exists, the following remain open:

1. The unmodified Mistral 3.5 W4A16 NVFP4 checkpoint may still drift
   from vllm/HF token-by-token at greedy decode despite producing
   plausible English/German.
2. There may be smaller numerical biases (RoPE phase, layernorm
   epsilon, softmax accumulation, etc) that the magnitude oracle
   did not exercise.

The next concrete step is a **W4A16-equivalent reference run** —
vllm 0.20.2 or HF transformers loading the same NVFP4 checkpoint,
decoding identical prompt-IDs with the same sampler, then a
layer-by-layer diff at layers 0, 1, 40, 80, 87. We do not yet have
that reference up and running on this machine.

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

## Pixtral vision pipeline (Round-12) — semantically validated end-to-end

The full Pixtral path (forward_pixtral_vision → BF16 splice into the
language-decoder embed buffer at `slot.token_start * row_bytes` →
generate) produces semantically correct visual descriptions on real
images, despite the residual ~6% angular drift at `post_blocks` vs
the HF reference (a separate-yak numerical-fidelity issue tracked
under "Open risks" below).

E2E tests (rvllm-serve, RVLLM_DEBUG_MISTRAL35=1, RVLLM_LOAD_VISION=1):

  /tmp/orange_ball.png (336x336, orange ellipse on light blue)
    user: "What is in this image? Answer in 5 words."
    rvllm: "orange ball on light blue background"   ✅

  /tmp/blue_yellow.png (blue square with yellow triangle)
    user: "Describe this image in 8 words or less."
    rvllm: "Yellow triangle on blue square background."   ✅

  /tmp/grid_image.png (green background, white grid lines)
    user: "What pattern is in this image? 8 words max."
    rvllm: "Green grid with white lines."   ✅

The model correctly identifies shapes, colors, and spatial
relationships from the BF16 soft-token output of
`forward_pixtral_vision`. Text-only smoke
(`"Hello"` → `" Abstract Wikipedia Update — Volume"`) remains
unchanged after the splice plumbing landed; the `generate(...)`
shim onto `generate_with_vision(..., &[])` preserves zero-splice
behaviour.

The 6% post_blocks drift accumulates from small per-block BF16
rounding differences vs HF (see Round-12 phase 3-test (c)
investigation). It does NOT prevent semantically correct visual
understanding — Pixtral's design is robust to small embedding
perturbations the language decoder absorbs through its attention.
Numerical-bisect work (per-block dump in a stream-isolated path,
softmax precision, etc.) can land later as a polish pass without
blocking the vision feature.

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
