# M2 Native Handoff

Date: 2026-04-28

## Current truth

We have a working full-model MiniMax-M2.7 path on TPU through the JAX reference harness. It loads the real `MiniMax-M2.7-NVFP4` checkpoint, runs all 62 layers, uses the real NVFP4 MoE, and can produce coherent text under a 1k generation cap.

We do not yet have a working native Rust/StableHLO M2 language model. The native StableHLO graph executes attention across all 62 layers, updates KV, and produces live logits, but the MoE block is still bypassed. Because M2 is an MoE model, that makes native StableHLO generation and PPL invalid as model-quality evidence.

## Verified artifacts

- Full JAX PPL reference: `tpu/out/m2/m2_jax_ppl_corpus.json`, PPL `4.922`.
- Full JAX 1k-cap coherent sample: `tpu/out/m2/full_model_samples/full_model_1k_answerprefix2_20260428_125426.json`.
- Full JAX raw 1k sample: `tpu/out/m2/full_model_samples/full_model_1k_thinking_20260428_120854.json`, shows the model can get stuck in a reasoning loop if sampled naively.
- Full JAX no-prefix 1k-cap attempt: `tpu/out/m2/full_model_samples/full_model_1k_clean_20260428_173736.json`, stopped at 91 tokens but is not accepted because the answer contains a physics phrasing error.
- Native StableHLO attention execution is documented on `docs/roger/index.html`, but PPL/top-1 are bad because MoE is bypassed.

## Most recent coherent sample

Prompt:

```text
Explain angular momentum clearly and correctly in one coherent answer. Keep it under 180 words and end after the answer.
```

Visible answer:

```text
Angular momentum is a vector quantity that measures rotational motion about a chosen point or axis. For a particle, the angular momentum about a point O is defined as L = r x p, where r is the particle's position vector from O, and p = mv is its linear momentum. The magnitude |L| = |r| |p| sin(theta), where theta is the angle between r and v.
```

Metrics:

- Runtime: full JAX M2.7, 62 layers, real NVFP4 MoE.
- Generated tokens: `64`.
- Mean decode latency: `529.05 ms/token`.
- Throughput: `1.89 tok/s`.
- Stop condition: complete sentence under a 1k cap.

## Native StableHLO state

Working:

- StableHLO syntax/parsing issues fixed.
- Dense bf16 weight arena split out of the giant flat weight arena.
- Placeholder native weight arena avoids HBM overflow in StableHLO mode.
- RMSNorm, QKV attention, RoPE, causal mask, KV update, O projection, final norm, and lm_head execute.
- Rust bench now does prompt prefill before free-running generation.
- EOS stop is recorded in the bench JSON.

Not working:

- MoE is bypassed at `v3/crates/rvllm-xla/src/m2_decode_graph.rs`.
- Native StableHLO PPL is therefore bad by construction, not a metric bug.
- Native StableHLO free-run text is garbled.
- B=32 native StableHLO is blocked by XLA shape/indexing limits on the monolithic KV tensor.

## Fix list

1. Wire MoE into the native path.

   Shortest correct route: consume a JAX-emitted full StableHLO decode graph, or import the JAX MoE lowering into the Rust-emitted StableHLO graph. Do not treat attention-only output as a working model.

2. Add a hard correctness gate before publishing native numbers.

   Required pass condition: native run must produce sane 1k-cap text and PPL near the JAX reference on `m2_ppl_corpus.txt`. Performance numbers should stay marked invalid until this passes.

3. Decide the MoE weight strategy.

   Full W1+W2+W3 int8 expert arena is too tight on v6e-8 HBM. Options are expert paging, projection-sized operands, JAX-emitted StableHLO constants, or a split custom-call design that keeps each operand under TPU plugin span limits.

4. Split KV for B=32.

   Current B=32 shape hits XLA indexing limits with the monolithic KV tensor. Split per layer or per K/V stream before claiming B=32 native execution.

5. Remove sampling crutches from the final demo.

   The reference path works, but clean product demo should not need an answer prefix. The latest no-prefix run stopped cleanly at 91 tokens but had a physics phrasing error, so it is not the accepted sample. Use direct-answer chat formatting, suppress reasoning tags if needed, and stop on EOS or a complete sentence under max tokens.

## Commands

Run full-model 1k-cap reference sample on the TPU:

```bash
gcloud compute tpus tpu-vm ssh rvllm-m2 \
  --project finance-484520 \
  --zone europe-west4-a \
  --command 'cd /workspace/runs/m2bench && python3 tpu/harness/m2_full_sample_1k.py \
    --model-dir /workspace/models/m2/MiniMax-M2.7-NVFP4 \
    --prompt "Explain angular momentum clearly and correctly in one coherent answer. Keep it under 180 words and end after the answer." \
    --max-new-tokens 1000 \
    --direct-answer \
    --empty-think \
    --suppress-reasoning \
    --temperature 0.25 \
    --top-p 0.9 \
    --top-k 40 \
    --repetition-penalty 1.08 \
    --stop-after-sentence-min-tokens 45 \
    --out /workspace/models/runs/m2_full_1k_sample.json'
```

Run native StableHLO bench:

```bash
RVLLM_M2_LAYER_MODE=stablehlo \
RVLLM_M2_BODY_PROBE=1 \
MAX_WEIGHT_ARENA_BYTES=170000000000 \
bash tpu/harness/run_m2_rust_xla.sh
```

Do not mark the native path working until the MoE bypass is removed and the PPL/text gate passes.
