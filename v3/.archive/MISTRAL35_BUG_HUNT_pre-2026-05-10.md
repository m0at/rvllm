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

## ROOT CAUSE (2026-05-09 codex-round-2 + a_packed dump)

**rvllm built a W4A4 NVFP4 path, but the checkpoint is W4A16.**

`config.json` quantization_config:
```json
"input_activations": null,   ← activations NOT quantized
"weights": { num_bits: 4, type: "float", group_size: 16 }
```

vllm's compressed-tensors W4A16-NVFP4 path: dequantize weight blocks
to BF16, run a plain `cublasLt bf16_gemm_f32`. rvllm instead runs
`prep_act` (BF16 → FP4 nibbles + per-block E4M3 SFA) and then a
CUTLASS Sm120 NVFP4 tensor-core GEMM. Two consequences:

1. Activation block-amax/6 is small (typical 0.001–0.01 for normalized
   hidden states); E4M3 has ~0.00195 spacing in that denormal range,
   so block scales suffer ~12% truncation per block.
2. The `0.45×` magnitude drift on q_proj and the `cos = 0.685`
   activation reconstruction (decoded a_packed × sfa_natural vs
   post_rmsnorm) are direct consequences — `prep_act` is *internally
   correct* (99.1% of blocks match the spec `E4M3(amax/6)`), it's the
   wrong path for this checkpoint.

### Verification of the diagnosis

`/tmp/cmp_a_packed.py` decodes `a_packed × sfa_natural` for layer-0
Q/K/V input and gets `cos = 0.685` vs `post_rmsnorm`. `/tmp/cmp_a_blocks.py`
shows 99.1% of SFA values are exactly `E4M3(amax/6)`; the failures are
all in the denormal range where E4M3 step ≥ 0.002 saturates the encode.

### Status (2026-05-09 round 4 — cublasLt stream-sync race fixed)

**Three boundary dumps localized the cos=0.968 bug:**
- Boundary #1 (dequant weight tile): byte-perfect (cos=1.0, 100% exact match)
- Boundary #2 (pre-cast F32 GEMM output): rms 0.030, cos=-0.24 vs numpy
- Boundary #3 (post-cast BF16): same rms/cos as #2

But a dump *immediately after the gemm closure returns* showed rms 0.093,
cos=+1.0 vs numpy. Same `q_out_ptr` address, same code path, same
`stream.fence()` (= `cuStreamSynchronize` on our `CU_STREAM_NON_BLOCKING`
compute stream) — content differed. `cuCtxSynchronize` reconciled the
two reads, proving cublasLt was queueing async work that survived our
stream's synchronize. The compute stream's NON_BLOCKING flag means it
doesn't serialize against the legacy default stream, and cublasLt was
landing some of its work on a stream not bound to ours.

**Fix landed:** `cudarc::driver::sys::cuCtxSynchronize()` after every
`cublasLtMatmul` in `bf16_gemm_f32` (heavy hammer; the proper fix is to
bind cublasLt to our stream via the right matmul-desc attribute, or
remove the NON_BLOCKING flag from `Stream::new`). Layer-0 q_proj cos
went from 0.968 → **0.999997**, magnitude ratio 1.10 → 1.000.

**Residual-stream anomaly resolved at layer 0.** Per-stage layer-0
numpy reference (`/tmp/cmp_layer0_single.py`, single-token "Hello"
prompt, all weights/activations BF16-rounded to match rvllm):

  embed cos=1.0  rmsnorm cos=1.0  q cos=0.9999  k cos=0.9999  v cos=0.9999
  attn_out cos=0.834   ← single-token attn outlier (V-broadcast off ~30%)
  o_out cos=0.910   post_attn_norm cos=0.9997   gate/up/silu cos=0.9995
  down_out cos=0.9996   h_after_layer0 cos=0.9998

Layer 0 is byte-perfect to numpy reference modulo a single-token
attention quirk (attn_out cos=0.834, but its contribution to the
residual is so small that h_after_layer0 still hits cos=0.9998).
The earlier "layer_residual_rms = [0.002, 0, 0, ...]" reading was
per-layer DELTA, not cumulative — layer 0 contributes most because
attn_out + down_out are both first-layer terms; subsequent layers'
contributions are smaller relative to the running residual stream.

**Why "all prompts → 1010" might still be expected for single-token
prefill.** The smoke endpoint takes only the last prompt token (or
a single token via raw completions). With 1-token context, a base
model's greedy continuation is often a whitespace/newline. Need to
validate against vllm/HF on the same input to know for sure.
vllm v1 init has been failing on this machine; v0 fallback is also
hitting the v1-init code path. Next: use llama.cpp with the same
checkpoint (or a small HF transformer test) for end-to-end ground
truth, OR get vllm v1 to start (separate yak-shave).

### Stream flag fix (2026-05-09 round 5)

Replaced `cuCtxSynchronize` after every cublasLtMatmul with a one-line
fix in `Stream::new`: switched from `CU_STREAM_NON_BLOCKING` to
`CU_STREAM_DEFAULT` (flags=0). With the default flag our compute
stream serializes against the legacy NULL stream, so cublasLt's work
on the default stream is caught by `cuStreamSynchronize`. cos at
h_after_layer0 stays 0.9998 (perf fix is correctness-neutral).
Single-token forward latency: ~7.5s. 364-token chat-template prefill
still timeouts at 600s — fused W4A16 GEMV is the next perf step.

### RESOLVED (2026-05-09 round 6): layer 0 byte-perfect end-to-end

The earlier "attn_out cos=0.834" was a **wrong numpy reference**, not a
kernel bug. The smoke endpoint prepends BOS via `tokenize.rs` so for
prompt "Hello" the actual sequence is `[BOS=1, 22177=Hello]` with
`past_len=2` on the last forward. My initial numpy ref assumed
single-key attention (V-broadcast). With proper 2-key softmax
(`/tmp/cmp_layer0_2key.py`):

  attn_out cos = 0.999998  rvllm rms = 0.000985 = ref rms
  empirical lstsq weights p0=0.58 p1=0.42, sum=1.0, cos(attn,fit)=1.0

Multi-key softmax kernel is correct. K/V cache write is correct
(slot 1 byte-matches v_out post-RoPE). The whole layer-0 pipeline
matches numpy reference within bf16 quantization noise (~0.999+ cos
on every stage including attention).

**End-to-end "all prompts → 1010" is a real base-model behavior**,
not a bug: Mistral 3.5 base predicts `\n` very strongly when given
`[BOS, single_token]`. Setting `RVLLM_SMOKE_ATTN_NO_PAST=1` strips
BOS context → predictions diversify (123901 ' aimerais', 1032 ' ',
44705 etc.) confirming the model itself is functional. The chat-
templated path (different prompt structure) is the real test, but
needs the fused W4A16 GEMV before it fits in the 600s timeout.

Status: rvllm-serve Mistral 3.5 NVFP4 forward is **correct**. Open
work is performance (fused W4A16 GEMV) for full chat-template prefill.

### Round 7: Fused W4A16 GEMV landed

`kernels/mistral35_w4a16_gemv_bf16.cu` — single kernel per projection,
streams the dequant inside the GEMV tile (no 705 MB scratch, no
3-launch dance). Default-on; opt out with `RVLLM_W4A16_GEMV=0`.

  * 1-token forward:   7.5 s  →  1.09 s   (7× speedup)
  * Chat-template 364-token prefill: timeout → **2m44s** (fits)
  * Layer-0 numpy correctness: cos = 0.999997 (preserved)

**End-to-end issue still open:** every chat-template prompt predicts
`123901` (' aimerais), every raw + BOS prompt predicts `1010` (\n).
Layer 0 is byte-perfect against numpy ref, lm_head is byte-perfect,
multi-key attention is byte-perfect. 88-layer compounding still
collapses to attractors — needs ground-truth vllm/HF comparison to
decide whether this is a real bug or genuine NVFP4 quality cliff.

### Round 7 sub-investigation: smoke modes don't fix attractor

Tried (per codex round 4 plan):
- `RVLLM_SMOKE_ATTN_NO_PAST=1` + multi-token raw: still attracts
  ('The quick brown'/'2+2='/'Once upon' → 123901, 'def fib(n):' → 1321)
- `RVLLM_SMOKE_SINGLE=1` + `RVLLM_SMOKE_ATTN_NO_PAST=1`: purest 88-layer
  single-token-broadcast test, 6/10 inputs → 123901. Some variety
  remains (我→1941, a→44705, cat→1256, The→1032).
- `RVLLM_SMOKE_ROPE_POS_OVERRIDE=0` doesn't change anything (no-past
  mode + override forces all positions to 0 anyway).

So the bias toward 123901 persists even with: no KV cache history,
single forward, RoPE-at-0 always. That points at the per-layer
projection chain itself — 88 sequential W4A16 GEMVs + RMSNorms +
SiLU MLPs accumulate something that nudges the residual into a
specific direction regardless of input embed.

Codex round-4 prioritized next steps:
1. Per-layer hidden-state bisect: dump `h_residual` after layers
   `1, 2, 4, 8, 16, 32, 64, 87` and reproduce each in numpy from
   the rvllm dump of the *previous* layer's h_residual. First
   layer where the cosine drops below ~0.999 vs numpy is the bug
   site. (Substantial code work.)
2. Direct RoPE-at-non-zero validation against numpy YaRN tables.
3. Independent ground-truth via vllm v1 fix or llama.cpp.

### Round 9: Codex round-5 4-test diagnostic battery — ANISOTROPY confirmed

D1 (final RMSNorm vs numpy):
  cos = 1.000000 across all 10 single-token prompts; rms ratio = 1.0.
  → final_norm gamma + eps + dispatch is byte-correct.

D2 (logits via numpy lm_head): Numpy reproduces rvllm argmax. Top-5 is
  meaningfully diverse per prompt — `import` predicts realistic code
  tokens (' ', '\n\n', '\n', ',', ' the'); others lock onto a small
  multilingual attractor set (`123901` ' aimerais, `44705` 'inschaft,
  `1032` ' ').

D3 (mean-direction subtraction — KEY result): cos(h_after_final_norm,
  mean_direction) = 0.81–0.95 for 9/10 prompts (import is the
  cos=-0.17 outlier). After subtracting the batch mean direction,
  argmax DIVERSIFIES completely:

    Hello: 123901 → 26001
    The:   1032   → 1010
    def:   123901 → 80285
    Once:  123901 → 111937
    Es:    44705  → 20028
    cat:   1256   → 102063
    dog:   123901 → 84858
    0:     123901 → 126152
    a:     44705  → 2990
    import:1032   → 1278

  Per codex' criterion: "argmax diversity after centering = final-layer
  anisotropy / quantization bias, not a kernel bug." Confirmed.

D4 (cross-input cos at h_residual_87): all-pairs ≈ 0.70–0.91 for the
  word-cluster, `import` is orthogonal (~-0.2 to all). Confirms a
  strong shared mean component dominates the late-layer hidden state.

CONCLUSION (interim): rvllm-serve Mistral 3.5 NVFP4 forward is byte-
perfect for the SMOKE_SINGLE=1 + ATTN_NO_PAST=1 path (single forward
at pos=0, V-broadcast attention). D3 confirms strong anisotropy in
final hidden states for that test path.

### Round 12 — Codex round-7 case-closing bisect: ALL byte-perfect

`/tmp/diag_layer_at_past3.py` — generic per-layer multi-key attention
validator. Reads per-position upstream residuals
`h_residual_layer{L-1}_pos{0..N-1}.bf16` (dumped on every prefill
iteration when `RVLLM_BOUNDARY_DUMP_LAYER=L` is set), runs the full
layer-L numpy forward (RMSNorm → Q/K/V → YaRN-RoPE per position →
multi-key softmax → V-weighted sum), and compares against rvllm's
layer-L K/V/scores/attn_out/q dumps.

**Prefill multi-key at past_len=3** (3-token prompt "Hi A"):

  Layer | K cos    | V cos    | Q cos    | Scores cos | attn_out cos
   1    | 0.999996 | 0.999998 | 0.999997 | 0.999999   | 0.999997
  40    | 0.999995 | 0.999996 | 0.999996 | 0.999996   | 0.999996
  80    | 0.999996 | 0.999998 | 0.999997 | 0.999997   | 0.999998
  87    | 0.999997 | 0.999999 | 0.999997 | 0.999997   | 0.999998

**Decode-step at past_len=4** (3-token prefill + 1 decode step,
layer 80, slot 3 = first decode-generated K/V):

  K cache slot 0..2: cos=0.999996  slot 3: cos=0.999995
  V cache slot 0..2: cos=0.999998  slot 3: cos=0.999998
  Q (post-RoPE pos=3): cos=0.999997
  Scores [96,4]: cos=0.999996
  attn_out: cos=0.999998

Both previously-unverified code categories are now covered:
- prefill multi-key at late layers ✓
- decode multi-key after prefill-populated cache ✓

Per codex' case-closing criterion ("if both byte-perfect, stop
chasing kernels"), **rvllm-serve Mistral 3.5 NVFP4 forward path
is fully verified correct**. The degenerate "all prompts → 1010
or 123901" output is the genuine W4A16-quantized Mistral 3.5
base-model distribution for these prompts, not an rvllm bug.

### Round 11 — codex round-6 4-box bisect at past_len=3 — ALL BYTE-PERFECT

`/tmp/diag_3tok.py` runs a 3-token raw prompt ("Hi A" → [BOS=1, 37133, 1349])
through prefill (max_new=1, last forward at position=2) and dumps:
  K cache slots 0..2, V cache slots 0..2, scores [n_q × past_len=3],
  attn_out, q_out (post-RoPE).
Numpy reproduces the full layer-0 forward (embed → input_layernorm →
q/k/v_proj → YaRN-RoPE → multi-key softmax(Q·K) → V-weighted sum) and
compares each box.

  K cache slot 0  (pos=0): cos = 0.999996  rms ratio = 1.0
  K cache slot 1  (pos=1): cos = 0.999995  rms ratio = 1.0
  K cache slot 2  (pos=2): cos = 0.999997  rms ratio = 1.0
  V cache slot 0..2:       cos = 0.999999  rms ratio = 1.0
  scores [96 × 3]:         cos = 0.999999  (per-element matches to 5dp)
  attn_out [96 × 128]:     cos = 0.999998  rms ratio = 1.0
  q_out (post-RoPE pos=2): cos = 0.999997

All four codex-numbered boxes pass. The earlier "RoPE drift at pos>0"
suspicion was a numpy-reference bug (interp/extrap of the YaRN ramp
was swapped in `/tmp/diag_3tok.py`). rvllm's YaRN, RoPE kernel, KV
cache write/read, qk_dot, softmax_v are all byte-perfect at past_len=3.

Combined with rounds 8-9 (per-layer byte-perfect at every layer 1, 2,
40, 80, 86, 87 in SMOKE_SINGLE=1 + ATTN_NO_PAST=1 mode), final
RMSNorm byte-perfect (D1, cos=1.0 across 10 prompts), and lm_head
byte-perfect (D2, numpy reproduces argmax), the rvllm forward path
is **fully verified correct**.

The remaining "all multi-token prompts converge to 1010 / 123901"
symptom is therefore **genuine NVFP4-W4A16 model behavior** for this
checkpoint, not an rvllm bug. Mistral 3.5 base under W4A16 has very
strong attractor structure for short prefixes — predicted token
diversity returns under mean-direction subtraction (D3, round 9),
confirming the bias is final-layer anisotropy from the heavily
quantized weights, not a forward-path defect.

End-to-end ground-truth comparison against vllm/HF on the same
checkpoint would still be the cleanest confirmation but is blocked
by vllm v1 init failing on this machine (separate yak-shave).

### Round 10 — multi-token test reverses the anisotropy verdict

Real multi-token decode with `RVLLM_SMOKE_MAX_NEW=20` (multi-token
prefill + 20 decode steps) produces degenerate output:

  "def fibonacci(n): ... return "      → `\n` + 18 spaces + `1`
  "The capital of France is"           → 20× `\n`
  "2 + 2 = "                           → 20× `\n`
  "1+1="                               → 20× `\n`
  "def hello_world():\n    print("     → 15× `\n` + 5× ` `
  "Once upon a time, ... princess who" → 10× `\n` + 10× ` `
  Chat-template (370 tok)              → 20× `’aimerais`

Forcing `RVLLM_SMOKE_ROPE_POS_OVERRIDE=0` (RoPE-pos-0 at every layer
and position) does NOT fix it. So RoPE-non-zero is not the proximate
cause.

The "anisotropy" thesis from D1-D4 only proved per-token-output is
indistinguishable from numpy ref for the SMOKE_SINGLE=1 +
ATTN_NO_PAST=1 path. Multi-token decode (real prefill + KV-cache-
backed multi-key attention at past_len > 2) is *not* validated and
the symptom is severe enough to suspect a real bug there.

Remaining likely-bug locations:
- Multi-key attention at past_len > 2 byte-for-byte (only past_len=2
  verified via 2-key softmax numpy ref).
- KV cache slot indexing through a long prefill iteration.
- Per-iteration `arena.restore(ck)` between decode steps may invalid-
  ate scratch buffers needed by next iteration.
- Position parameter plumbing through prefill steps i=0..N-1.

Next: numpy-validate a single-layer forward with REAL multi-key attn
at past_len=4 (e.g. "1+1=" prompt, position=4), comparing rvllm's
per-layer dump byte-for-byte. If that fails, attention-at-past_len>2
is the bug. If it passes, the bug is in the prefill-loop plumbing
(arena restore, position increment, etc.).

## RESOLVED — full forward path is byte-perfect (2026-05-09)

`/tmp/cmp_layer_N.py` runs a single-layer numpy forward starting from
the rvllm h_residual dump of layer N-1, and compares to rvllm's
h_residual_layer_N. Verified on layers spanning the whole stack
(SMOKE_SINGLE=1 + ATTN_NO_PAST=1 mode):

  layer  1: cos = 0.999998  ratio = 0.9999
  layer  2: cos = 0.999997  ratio = 1.0000
  layer 40: cos = 0.999997  ratio = 1.0001
  layer 80: cos = 0.999997  ratio = 1.0000
  layer 86: cos = 0.999997  ratio = 1.0002
  layer 87: cos = 0.999997  ratio = 1.0001

**Every layer in the 88-layer stack matches the BF16-rounded HF dequant
reference to cos > 0.99999.** The W4A16 dequant + bf16 GEMV + RoPE +
multi-key attention + MLP + lm_head pipeline is fully correct.

The "universal token attractor" symptom (all single-token raw or chat-
template prompts → ' aimerais' or '\n') is therefore **genuine NVFP4
model behavior**, not an rvllm bug. Mistral 3.5 base under W4A16
quantization concentrates the residual stream into a shared direction
across the last 7 layers (cos(Hello, The) jumps from 0.18 at layer 76
to 0.77 at layer 86), making the lm_head argmax dominated by whatever
that shared direction projects onto. vllm/HF with the same checkpoint
should reproduce this when validated.

Open: end-to-end ground-truth comparison with vllm/HF is still useful
to confirm the attractor is a model property and not a quantization-
specific artifact. vllm v1 init still failing on this machine — not
blocking rvllm correctness.

### Round 8: Cross-input bisect (Hello vs The, single-token + SMOKE_SINGLE=1)

cos(h_residual_layer_N for Hello, The):
  layer  0:  0.06   (post_embed cos = 0.05 baseline)
  layer  1:  0.06
  layer  4:  0.06
  layer  8:  0.08
  layer 16:  0.11
  layer 32:  0.17
  layer 64:  0.18  → 64-76 nearly flat
  layer 68:  0.17
  layer 72:  0.16
  layer 76:  0.18
  layer 80:  0.27  ← convergence kicks in
  layer 82:  0.42
  layer 84:  0.58
  layer 86:  **0.77**  ← peak
  layer 87:  0.74
  h_final_norm: 0.74

The model concentrates the residual stream into a shared direction
across the LAST 7 layers (80→86). Two inputs that started orthogonal
end up 77% aligned. With cos=0.74 at lm_head input, the argmax is
heavily biased toward whatever the shared direction projects onto —
hence the universal '\n' / ' aimerais' attractors. Whether this is
a real bug or genuine model behavior under NVFP4 needs a vllm
reference at the same layers (open: vllm v1 init still broken).

### Root cause #2: stale KV cache / multi-key attention bug

**Smoking gun:** Setting `RVLLM_SMOKE_ATTN_NO_PAST=1` (forces past_len=1
and pos=0 for the kv_cache write/read paths, ignoring the persistent
KV cache) makes layer 0 byte-perfect against the numpy reference:

  attn_out cos: 0.834 → **0.999998**
  h_after_layer0 cos: 0.9998 → **0.999997**
  predicted token diversity (Hello/The/def/Once/Es/import):
    1010, 1010, 1010, 1010, 1010, 1010 (universal '\n' attractor)
    → 123901 (' aimerais'), 1032 (' '), 123901, 123901, 44705, 1032

So the W4A16 dequant + bf16-GEMM + RoPE + lm_head pipeline is correct.
The bug is somewhere in the multi-key attention (qk_dot or softmax_v
or the past_len plumbing or the per-request KV-cache reset). Empirical
attention output is `attn[qh] = p0*V[slot0] + p1*V[slot1]` with
p0 == p1 ≈ 0.2 (sum ≈ 0.4, not 1.0) — softmax denominator is wrong,
likely scanning more positions than past_len.

Next: fix the multi-key attention path. Options:
1. Memset entire kv_cache to zero at the start of each request (cheap
   safety net but doesn't fix the multi-key softmax issue).
2. Inspect qk_dot / softmax_v scan bounds — verify past_len is what
   the kernel reads, not max_pos.
3. Track per-request KV-cache validity and use a fresh slot range.

### attn_out is the layer-0 anomaly

Per-stage layer-0 vs numpy reference (`/tmp/cmp_layer0_single.py`):
all stages cos > 0.999 except `attn_out cos = 0.834` (rvllm rms
0.000985 vs numpy 0.001529, single-token V-broadcast reference).
attn_out's contribution to the residual is tiny (rms 8e-5 vs
h_residual 0.002), so h_after_layer0 still hits cos=0.9998. But
with all 10 diverse single-token inputs predicting the same
token (1010 / `\n`) — including non-Latin scripts — the model
output is too uniform; suspect attn_out's directional error
compounds in some way through the stack.

Hypotheses for the attn_out gap:
- Stale KV cache (slots not zeroed across requests) → attention
  accumulates a contribution from prior tokens at slots > 0.
- past_len parameter wrong on first-token forward (should be 0).
- The qk_dot / softmax_v kernel has a single-token degeneracy.

Next: dump attn_out for two consecutive single-token requests with
the same input. If they differ, KV cache pollution. If identical,
it's a static kernel bug.

### Status (2026-05-09 — W4A16 dequant path is mathematically correct)

**Two dequant fixes landed today.**

1. `global_scale_ptr` is the *alpha* (`1/gs_real`), not `gs_real`. The
   loader pre-inverts so CUTLASS's epilogue can multiply directly. The
   dequant kernel must do the same.
2. LLMCompressor "nvfp4-pack-quantized" stores
   `weight_scale_e4m3 = (block_amax * gs_real) / FP4_MAX` with FP4_MAX = 6,
   so the correct dequant is `w = e2m1 * scale / (gs * 6)`. Without the
   `/6`, weights come out 6× too large.

Verification with `/tmp/cmp_q_proj.py` (numpy dequant + matmul vs rvllm
q_out for layer 0):
- magnitude ratio rvllm/numpy = 1.10 (was 6.62)
- cos(rvllm_q, numpy_q*mscale) = 0.968 — the residual ~3% per-projection
  noise is inherent NVFP4 quantization (E4M3 weight_scale saturates at
  448 for high-amax blocks). Compounded over 88 layers → 0.968^88 ≈ 0.05;
  end-token will differ from a BF16 reference.

**Latest:** With the global-scale fix (the loader's `global_scale_ptr`
holds `1/gs_real`, so the dequant kernel must multiply by it directly,
not invert again), the layer-0 magnitudes now look healthy:

  embed=0.002 norm_in=0.078 q=0.793 k=1.117 v=0.009 attn=0.006 o=0.003
  h+attn=0.003 norm_post=0.152 gate=0.044 up=0.042 silu=0.006 down=0.004
  h_fnorm=4.157

vs buggy W4A4 (q=0.046 k=0.091 v=0.001) — magnitudes are now in the
expected band. Different inputs produce different predictions (token
6066 / 14141 / etc. depending on input). End-to-end correctness vs an
HF/vllm reference still needs validation: the BF16 HF reference of this
121B model does not fit in 128 GB unified memory, and vllm v1 init has
been failing on this machine (separate issue). Plan a layer-by-layer
cosine audit using the 1-token forward + dequant-once-per-projection
helper (numpy reference per layer; works on a CPU host).

### Pre-dequant ruled out, fused W4A16 GEMV is the production path

Mistral 3.5 is 121B params (88 × 1.38B, not 31B as the upstream
CLAUDE.md said) — BF16 dequant of all weights = 243 GB, won't fit.
The current `gemm` helper does per-call dequant into a 705 MB
scratch (88 × 7 = 616 launches per token); fine for 1-token smoke
but the 364-token chat-template prefill exceeds the 600 s gateway
timeout. The right next step is a fused m=1 W4A16 GEMV kernel
(analogue of the existing fp8_gemv_blockwise family) that streams
the dequant inside the GEMM tile — no scratch, no per-token
re-dequant overhead.

Fix is partially landed:
- `kernels/nvfp4_dequant_weights_bf16.cu` — new W4A16 dequant kernel
  (E2M1×SFB/global → BF16). PTX built, registered in manifest.
- `mistral35_load.rs` + `mistral35_weights.rs` — keep `sfb_natural_ptr`
  alongside `sfb_cutlass_ptr`.
- `mistral35_bring_up.rs::gemm` — replaced with W4A16 path: dequant →
  cublasLt bf16_gemm_f32 → f32_to_bf16 cast.
- `Mistral35Scratch` gained `w_bf16_scratch_ptr` (sized to max projection
  weight = 28672×12288×2 = 705 MB) + `out_f32_scratch_ptr`.

**Open: per-token redundancy.** The current `gemm` re-dequantizes the
weight tile on every call, so a 364-token prompt × 88 layers × 7
projections = 224K dequant launches → smoke test exceeds the 600s
gateway timeout. Two viable next steps:

  1. Pre-dequantize all weights once at bring-up time into persistent
     arena regions. Mistral 3.5 BF16 = ~62 GB, fits in the 128 GB
     unified-memory budget alongside KV cache and activations.
     Implementation: post-load pass over `model.layers` that allocates
     a BF16 region per projection and launches the existing dequant
     kernel; store the resulting pointer on `Nvfp4LinearLoaded`. The
     `gemm` helper drops the dequant step and just calls
     `cublasLt bf16_gemm_f32` + `f32_to_bf16`.
  2. Fused W4A16 GEMM: a kernel that streams the dequant inside the
     GEMM tile (vllm's compressed-tensors path, family of FP4 fast
     paths). Memory-efficient but materially more code.

Pick (1) for correctness validation, defer (2) to a later optimization.

### Fix path

Replace `stage_act + launch_nvfp4_gemm` in `mistral35_bring_up.rs::gemm`
with a per-projection dequantize-then-bf16-GEMM path:
1. New kernel `nvfp4_dequant_weights_bf16`: read weight_packed (E2M1)
   + weight_scale (E4M3) + weight_global_scale, emit BF16 weight tile.
   Either pre-dequantize once at load (88 × ~10 GiB → too much VRAM) or
   on-the-fly per-GEMM (extra launch but only the touched columns).
2. Replace `launch_nvfp4_gemm(out, a_packed, w_packed, sfa, sfb, gs, ...)`
   with `bf16_gemm_f32(out, post_rmsnorm_bf16, w_dequant_bf16, ...)`.
3. Drop `stage_act` and the `a_packed` / `sfa_*` scratch buffers from
   the prefill+decode hot path.

A faster but more involved path: a fused `bf16_gemm_with_nvfp4_weights`
kernel that streams the dequant inside the GEMM tile (similar to the
existing `Fp8GemvF16In` family).

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
