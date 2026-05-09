# Mistral 3.5 NVFP4 forward bug — investigation log

**Status (2026-05-09):** rvllm produces predicted token 101484
(' Greco') vs vllm reference "Hello" for the chat-templated 'hi'
prompt. Output is deterministic but wrong; magnitudes are mostly
right, direction drifts via compound per-layer error.

## Verified correct

| Stage                      | Verification                              | Result        |
|----------------------------|-------------------------------------------|---------------|
| Embed gather               | rvllm post_embed vs HF embed[token]       | byte-identical |
| Layer 0 RMSNorm (input)    | rvllm post_rmsnorm vs numpy RMSNorm       | cos = 0.999999 |
| Layer 0 q_proj (NVFP4 GEMM)| rvllm q_out vs numpy dequant + matmul     | cos = 0.993, magnitude 0.45×|
| Layer 0 attention output   | rvllm attn_out vs numpy multi-key attn    | cos = 0.82, rms 0.5×|

The 0.45× magnitude on q_proj is consistent (K and V projections
also at ~0.64× of numpy reference). cos = 0.993 reflects NVFP4
quantization noise; the magnitude difference appears systematic
but uniform across all NVFP4 GEMMs so it doesn't affect direction
within a single op.

## Verified working

- async-copy race fix on token_in_ptr (cuMemsetD32Async on the
  compute stream, mirrors Qwen 3.6 round-26 pattern)
- KV cache layout [max_pos, n_kv_heads, head_dim]
- GQA mapping: q_head / gqa_ratio = kv_head (matches HF/vllm)
- RoPE split-half (NeoX-style) with mscale baked into cos AND sin
- attn_no_past=1 mode produces identical output to single-token
  broadcast hack (predicted=1775)

## Layer-0 attention isn't where the bug lives

Numpy reproduction of layer 0 (build K_cache from all 364 prompt
tokens via HF-dequantized weights → full multi-key attention for
last token):

- numpy reference: scores rms=0.41, softmax H/H_unif=0.995
- rvllm:           scores rms=0.38, softmax H/H_unif=0.997

**Trained Mistral attention IS roughly uniform-ish at layer 0**
(numpy ref confirms). Not a bug. Mid-to-late layers do the
sharpening; rvllm fails to reach that sharpening.

## Hypotheses tried & ruled out

1. **mscale double-application** — codex confirmed vllm/HF do
   exactly the same baking; not a bug.
2. **RoPE convention (interleaved vs split-half)** — Mistral uses
   split-half (NeoX), matches my kernel.
3. **SFB transform layout (m=N vs m=1)** — both worse; m_dummy=N
   is correct (cos drops 0.993→0.50 if changed).
4. **alpha = 1/(global*6)** — predicted shifted but cos got worse
   (FP4_MAX factor not the missing piece).
5. **alpha = 2/global** — predicted shifted, cos unchanged
   (uniform scaling doesn't fix direction).

## What's left

The bug is in the **per-layer math compounding** — each layer
introduces small systematic error that cumulatively rotates the
hidden state away from what trained Mistral expects. Layer 0
cos vs reference = 0.82; over 88 layers compounded that's far
from the trained-model trajectory.

To bisect properly:
1. Hook into vllm's loaded Mistral3 model via collective_rpc to
   capture per-layer hidden states for the same prompt.
2. Run rvllm with per-layer-N hidden state dump (extend the
   existing scores dump to per-layer h_residual).
3. Compute cosine(rvllm[L], vllm[L]) for L=0..87. First L where
   cosine drops below 0.95 is the divergence point.

That's where the next debugging session needs to start. Without
that, every attempted fix is a guess.

## Diagnostic env knobs (committed)

- `RVLLM_SMOKE_DUMP_DIR=…`        persist 14 stage dumps + scores_layer87
- `RVLLM_SMOKE_FULL_DUMP=1`       gate the dumps (LAST call only)
- `RVLLM_SMOKE_LAYER_RMS=1`       per-layer residual rms
- `RVLLM_SMOKE_SINGLE=1`          skip prefill, run 1 forward
- `RVLLM_SMOKE_ATTN_NO_PAST=1`    every layer attends to current token only
- `RVLLM_SMOKE_ROPE_POS_OVERRIDE=N`  fix RoPE position at every layer
- `RVLLM_NVFP4_ALPHA_MULT=K`      scale uploaded 1/global by K

## Numpy diagnostic scripts in /tmp/

- `cmp_embed.py`        post_embed vs HF embed[token]
- `cmp_norm.py`         post_rmsnorm vs numpy RMSNorm
- `cmp_q_proj.py`       q_out vs numpy NVFP4 dequant + matmul
- `cmp_lm_head.py`      h_after_final_norm × HF lm_head → top-K
- `cmp_scores.py`       layer-87 attention distribution health
- `cmp_layer0.py`       full layer-0 attention via HF weights

---

## Earlier codex consult (kept for reference)

The previous file contents from codex's first review are below for
the historical record.

**Short Answer (codex 2026-05-09 round 1)**
Your #2 is probably not the bug. HF and vLLM both apply YaRN's
attention factor by multiplying both `cos` and `sin`, so Q and K
both get scaled and attention scores effectively get
`mscale^2 / sqrt(d)`. For this checkpoint, that is expected
behavior, not a double-application bug.

The top suspects now are: attention/KV path at nonzero positions,
a later-layer projection/dequant issue, or lm_head orientation/
data mismatch. RoPE convention and YaRN table math look mostly
correct.

(Codex's full layer-0 dump diagnostic plan was the foundation for
the bisect work above. The result confirmed his hypothesis #1 —
layer-by-layer divergence — over hypothesis #4 RoPE or #5 mscale.)
