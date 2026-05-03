# NVFP4 on sm_121 in rvllm — Architecture & Algorithms

A reference for the next LLM to come at this codebase. Skip the
process notes (read CLAUDE.md for that). This is the algorithmic
picture you need so you don't re-derive it.

---

## 1. The macro pipeline

For one Gemma 4 layer, decoding step:

```
residual ──► QKV-rmsnorm ──► QKV-proj (FP8 GEMM)
                                  │
                                  ▼  (interleaved [num_tokens, q_dim+2·kv_dim])
                                Q | K | V (f16)
                                  │
                              fused_qkv_rmsnorm
                              (per-head Q-norm + K-norm gammas)
                                  │
                              ┌───┴────────────┐
                              ▼                ▼
                       rope_partial_nvfp4kv kernel
                       ├ RoPE(Q, K)              (positions tbl)
                       ├ Hadamard R(Q, K, ?V)    (per-layer signs)
                       ├ amax→scale per (token,head) for Q
                       ├ amax→scale per 16-elem block for K, V
                       ├ MSE-search-pick scale (V) / amax6 (K)
                       └ pack:
                          Q_fp8[token,head,d]      (FP8 E4M3, per-token-head scale)
                          K_packed[slot,kvh,d/2]   (NVFP4 nibbles)
                          V_packed[slot,kvh,d/2]   (NVFP4 nibbles)
                          K_scale[slot,kvh,d/16]   (E4M3 per-16)
                          V_scale[slot,kvh,d/16]   (E4M3 per-16)
                          Q_scale[token,head]      (f32)
                                  │
                                  ▼
                       FA2-decode-NVFP4 / split-decode kernel
                       ├ load Q row, dequant FP8→f32, fold scale
                       ├ tile loop:
                       │   ├ dequant K tile NVFP4→f16 smem
                       │   ├ Q · K^T MMA (m16n8k16 f16)
                       │   ├ online softmax (f32)
                       │   ├ dequant V tile NVFP4→f16 smem
                       │   └ P · V MMA, f32 accumulator
                       └ epilogue: acc / row_sum → f16 attn_out
                                  │
                                  ▼ (if HADAMARD_V=1)
                       hadamard_unrotate_f16 (apply R^T to attn_out)
                                  │
                                  ▼
                       O-proj (FP8 GEMM) → residual add
```

Math invariants:

- `R = H · diag(D)` where `H` is normalized Walsh-Hadamard (so
  `H · H^T = I`) and `D` is a fixed per-layer ±1 sign vector. Therefore
  `R · R^T = I`, i.e. orthogonal.
- `Q_rot · K_rot^T = (Q·R)·(K·R)^T = Q·R·R^T·K^T = Q·K^T` —
  rotation-invariant. Hence Q+K can be rotated together with no
  unrotate step.
- `P · V_rot = P·V·R`. Recovering `P·V` requires multiplying by `R^T`
  on the right. That's what the unrotate kernel does, AFTER attention,
  BEFORE O-proj. Skipping it = O-proj sees rotated input = garbage out
  with no obvious diagnostic signal.

---

## 2. NVFP4 packed-block format

Every K and V row of `head_dim` elements is stored as:

- `head_dim / 2` **packed bytes** (two FP4 nibbles each, lo nibble in
  bits [0..3], hi nibble in bits [4..7]).
- `head_dim / 16` **E4M3 micro-scales**, one per 16-element block.

FP4 magnitude levels: `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`, plus
sign bit. Zero-equality for sign means there is **no negative zero** in
the encoding (saves one of 16 codes).

Effective bits per element ≈ 4 + (8/16) = 4.5 bits. Memory ratio vs FP8:
~56 %; vs F16: ~28 %.

Block scale derivation:

- `amax6` (range-preserving, OCP baseline): `scale = peak / 6.0`,
  where `peak = max |v_i|`. Maps the max element to FP4 = ±6.
- `mse` (6-candidate): try `{peak/6, peak/4, peak/5, second/6,
  second/4, second/5}`, E4M3-round each, pick whichever yields the
  smallest sum-of-squared-error over the 16-element block when
  applied through the full `quantize → dequantize` round-trip.
  `second` = second-largest magnitude. Picking `second/4` clips the
  block's peak (treats it as outlier) and uses more granularity for
  the rest. The E4M3 rounding is INSIDE the SSE comparison so the
  search picks the actual storable scale, not the abstract one.

The 6-candidate set was empirically the sweet spot. 8 candidates with
`/3` over-clips typical (non-outlier-heavy) blocks; `/4.5` interpolation
ties to over-clipping under E4M3 rounding. Per-block MSE is an
APPROXIMATE proxy for downstream model quality — minimizing the proxy
more aggressively can hurt because cross-block correlations the
per-block metric can't see.

Asymmetric K/V policy is deliberate:

- **K → range-preserving (amax6)**. K participates in `Q·K^T → softmax`,
  so a single mis-quantized K can mis-route attention. Clipping K to
  bring the second-largest into range hurts the softmax distribution.
- **V → outlier-aware MSE**. V is averaged against softmax weights.
  Quantization noise on individual V channels averages out under the
  weighted sum. The MSE-best scale per block minimizes reconstruction
  error and that's what matters.

---

## 3. Hadamard rotation algorithm

The rotation flattens outliers so the per-block FP4 quantization has
less work to do.

Per layer `l`, channel `c`, sign:

```
  seed = SplitMix32(l, c)              # NOT FNV-1a — see §10
  D[l,c] = (seed & 1) ? -1 : +1
```

The signs table is uploaded once at bring-up to `nvfp4_hadamard_signs`
arena region (length `num_layers × head_dim` bytes).

Within the rope kernel, per (token, head):

```
  v[tid] = post-RoPE value at thread `tid`
  s_buf[tid] = v
  __syncthreads()
  apply_signs(s_buf, signs, head_dim)   # element-wise × D
  fwht_inplace(s_buf, head_dim)         # Walsh-Hadamard, lower-bit-cleared
                                        # adds, upper-bit-set subtracts;
                                        # apply 1/sqrt(D) at the end
  v = s_buf[tid]                        # = R · v
```

Treating `apply_signs` as left-mult by `diag(D)` and `fwht` as left-mult
by `H`, the composition gives `(H · diag(D)) · v = R · v`. For row
vectors that's `v · R`.

Unrotate (`hadamard_unrotate_f16.cu`) reverses the order — FWHT first,
then signs — to apply `R^T = diag(D) · H` (since `H` is symmetric and
self-inverse modulo the 1/√D normalization).

The K and V sides reuse the **same** `D` per layer (one sign vector
shared between K and V). Q has its own sign vector but the **same R**
(see §4 — the signs vectors hold the per-layer D, and Q's vector is
derived from the same SplitMix32 seed family).

V rotation is opt-in (`RVLLM_NVFP4_HADAMARD_V`). Without it, V is
quantized in its native distribution; attention output `P·V` is
already correct, no unrotate. With it, V quantization sees a flatter
distribution (Hadamard mixing tends to gaussianize), the 6-candidate
MSE search lands tighter scales, and the unrotate kernel recovers
`P·V` from `P·V·R`.

---

## 4. Per-(token, head) Q dynamic scale

Q stays FP8 per-tensor per-head; K/V go NVFP4 per-block. The Q scale is
computed at quantize time:

```
  for each (token, head):
    a = block_reduce(amax(rope-and-rotated Q[token,head,:]))
    scale = max(a / 448.0f, 1e-12f)        # E4M3 max = 448
    q_scale_cache[token, head] = scale     # f32, written to global
    quantize Q[token,head,d] = e4m3(v / scale)   # for each d
```

FA decode reads `q_scale_cache[token, head]` and multiplies the
dequantized Q by `q_scale * attn_scale` at register-load time. Required
when Hadamard is on, because rotated Q can grow up to ~√D per channel
and saturates the static `RVLLM_Q_SCALE` (typically 0.1).

The cache buffer is sized `[max_tokens, num_heads]`. The same buffer is
reused across prefill chunks (each chunk writes [0..chunk_q) and the
attention call reads the same range — no chunk-relative offset bug).

---

## 5. Attention kernel layout

### 5.1 Decode (M=1) — `flash_attention_2_decode_nvfp4kv_kernel`

Grid `(num_seqs, num_heads)`, block `FA2_THREADS=128`, head_dim ≤ 256
(BC=32) or head_dim=512 (BC=16, separate `_bc16` symbol).

Smem layout (f32 K and V — full precision):

```
  s_key   : [BC × head_dim] f32
  s_val   : [BC × head_dim] f32
  s_score : [BC]             f32
  s_reduce: [FA2_THREADS/32] f32
```

One CTA per (seq, head). Tile loop iterates KV-history in BC-sized
tiles. Per tile: dequant K → f32 smem; thread-per-dim Q·K^T dot;
online-softmax max+sum update; dequant V → f32 smem; thread-per-dim
P·V accumulate. f32 accumulator throughout.

### 5.2 Split decode (long context) — `flash_attention_2_decode_nvfp4kv_split_kernel`

Grid `(num_seqs, num_heads, max_num_partitions)`. Each CTA processes a
KV-history slice of `partition_size` (default 1024) tokens. Per
partition: same online-softmax math but local — output is **per-partition
attention output divided by per-partition row_sum**, plus per-partition
`max_logits` and `exp_sums` saved for the reduce phase.

Phase-2 reduce (`paged_attention_reduce_f16_kernel`): combines all
partitions:

```
  global_max = max(per-partition max_logits)
  for each partition p:
    rescaled_sum[p] = exp_sum[p] · exp(max_logits[p] - global_max)
  global_exp_sum = Σ rescaled_sum
  for each output dim d:
    out[d] = (Σ_p tmp_out[p, d] · rescaled_sum[p]) / global_exp_sum
```

Algebra (`tmp_out[p,d] = numerator_p[d] / exp_sum_p`,
`numerator_p = Σ_t∈p exp(s_t - max_p) · V_t`):

```
  out[d] = Σ_p (numerator_p / exp_sum_p) · (exp_sum_p · exp(max_p - global_max)) / global_exp_sum
         = exp(-global_max) · Σ_p Σ_t∈p exp(s_t) · V_t / Σ_p exp_sum_p · exp(max_p - global_max)
         = (Σ_all_t exp(s_t) · V_t) / (Σ_all_t exp(s_t))
```

I.e. fully equivalent to monolithic decode. The split exists for
parallelism (long contexts otherwise have one CTA doing 14k-token
sequential tile loops).

The split kernel uses **f16 K+V smem** (not f32) to halve smem
footprint and fit head_dim=512 under the Blackwell consumer 99 KB
limit. This costs one f16→f32 round-trip per element on read but is
the only configuration that fits — `cuFuncSetAttribute` raises the
dynamic smem cap to fit `2·BC·D·2 + BC·4 + reduce_scratch`.

`tmp_out` is **f32** (cycle 21 codex fix). Was f16 originally; the
f16 round-trip on per-partition partial outputs accumulated visible
error. f32 partials are essential for split-decode quality.

### 5.3 Unified prefill — `flash_attention_2_prefill_nvfp4kv_unified_kernel`

Grid `(total_num_q_blocks, num_kv_heads)`. Each CTA handles `BLOCK_M=16`
query rows against the full KV history (causal masked). Uses
`m16n8k16` f16 MMA tensor cores.

Per tile (size `tile_size`, default 32 for sliding / 16 for global):

```
  load K tile (NVFP4 → f16 smem, dequant via cvt.rn.f16x2.e2m1x2)
  Q · K^T MMA (4 warps × 1 n-tile of 8 wide each)
  online softmax row-update (f32 m, l, alpha; f16 P fragment)
  load V tile (NVFP4 → f16 smem)
  P · V MMA, accumulating into s_acc (f32, BLOCK_M × head_dim)
  epilogue: out[m, d] = s_acc[m, d] / s_l[m]
```

Causal mask (in `mask_pack` lambda):

```
  query_abs = prefix_len + q_pos_in_seq
  kv_pos = tile_base + t
  valid = (q_pos_in_seq < query_len) ∧ (q_head < num_heads)
        ∧ (kv_pos ≤ query_abs)                       # causal
        ∧ (window_size_left ≤ 0 ∨ query_abs - kv_pos < window_size_left)  # SWA
        ∧ (kv_pos < max_seq_prefix_len)
  return valid ? dot : -FLT_MAX
```

The online softmax has masked-tile guards: `(s > -FLT_MAX + 1.0f) ? expf(s - new_M) : 0.0f`
and `(prev_M > -FLT_MAX ∧ new_M > -FLT_MAX) ? expf(prev_M - new_M) : 0.0f`.
This is necessary because `expf(-FLT_MAX - -FLT_MAX) = expf(0) = 1`
would otherwise pollute row sums on all-masked tiles.

---

## 6. The chunk-size cliff (UNRESOLVED)

`RVLLM_PREFILL_CHUNK_SIZE` (number of new query tokens per prefill
kernel call) has a hard quality cliff at ~288–320. Chunk=256 produces
clean output across all tested prompts; chunk=384 produces garbage
cycles even with all quality fixes (V-Hadamard + 6-candidate MSE +
hybrid configs).

The kernel is **mathematically chunk-invariant**:

- KV cache state at slot `p` is identical regardless of chunking.
  K/V quantization is per-token and deterministic.
- Each query CTA processes `BLOCK_M=16` rows independently. No
  cross-CTA state.
- Causal mask is per-(q_abs, k_pos), independent of chunk_q.
- Each query reads K from `[0, query_abs]` (causally), the same range
  whether the chunk is 256 or 2048 wide.

So the cliff is empirical, not algebraic. Suspected mechanism:
**FP-accumulation-order dependence in MMA reductions that compounds
nonlinearly with tile_count**. Larger chunks mean each q-block iterates
more masked tiles even if those tiles contribute zero to the row sum
(load is still issued, MMA still runs, the 0-weighted P·V add is
still emitted). Across hundreds of tile-iterations, f32-accumulator
rounding could drift in chunk-dependent ways.

Has not been pinned down — would require kernel-side instrumentation
(per-tile intermediate-state dumps for query positions both well below
chunk_q and at the last position of the chunk, comparing chunk=256 to
chunk=2048 traces).

If solved, prefill speedup is large: chunk=2048 finishes a 14.7 k-token
prompt in ~30 s vs chunk=256's ~95 s, with no quality cost.

---

## 7. Gemma 4 specifics that drive choices

### 7.1 60 layers split: 50 sliding + 10 global

| | Sliding | Global |
|---|---|---|
| head_dim | 256 | 512 |
| num_kv_heads | 16 | 4 |
| num_q_per_kv | 2 | 8 |
| RoPE θ | 10 000 | 1 000 000 |
| rotary_dim | 256 (full) | 128 (partial 0.25) |
| sliding_window | 1024 | none |
| `attention_k_eq_v` | false | **true** (K weight serves both K and V; no v_proj) |

The hybrid env knobs (`RVLLM_NVFP4_HYBRID_GLOBAL_FP8`,
`RVLLM_NVFP4_HYBRID_SLIDING_FP8`) let you pin individual subsets of
layers to FP8 KV instead of NVFP4. They exist because earlier debugging
suspected the global layers were the cliff (hypothesis H1: outlier
pressure on the 10 globals dominates). Empirically that hypothesis
turned out to be wrong on Gemma 4 — the cliff was V quantization, not
global-layer-specific.

### 7.2 The K-norm / Q-norm gamma quirk

Gemma 4 has per-head Q-norm and K-norm rmsnorms. The gamma weights for
K-norm in layer 0 are a **near-scalar ~0.1221 ≈ 1/√67** across all 256
channels. After RMSnorm normalization, multiplying by 0.1221 gives Q
and K with element std ≈ 0.1221. Q·K^T over head_dim=256 then has
expected std ≈ √256 · (0.1221)² ≈ 0.238 — already softmax-friendly.

Consequence: **the kernel call passes `attn_scale = 1.0`** instead of
`1/√head_dim`. Gemma 4 has absorbed the attention scaling into the
QK-norm gammas at training time. Adding `1/√d` here would over-scale
and destroy the softmax distribution.

If you ever port to a model that does NOT have this absorption (e.g.
Llama, Qwen), you must pass the proper `1/√head_dim` scale.

### 7.3 Logit softcap

Gemma applies `logit ← cap · tanh(logit / cap)` with `cap = 30` before
argmax. For `|logit| ≪ 30` this is near-linear; for huge logits it
saturates at ±30. The kernel `logit_softcap_kernel` is run on the f16
logit tensor in-place, after the LM-head GEMM, before any sampling /
repetition-penalty logic.

---

## 8. The FP8 GEMM dispatch landscape

Three GEMM paths fire under different (M, weight-shape) regimes for
the QKV / O / gate_up / down projections:

| M | weight has blockscale | path | precision | speed |
|---|---|---|---|---|
| 1 | yes | `Fp8GemvF16In` (per-row GEMV, full 2-D blockscale) | full | best for M=1 |
| 1 | no (fused QKV) | `fp8_gemm_channelscale_or_fallback` | per-row blockscale → 1-D approximation | OK |
| 2..127 | yes | `Fp8GemvF16In` (per-row GEMV) | full 2-D blockscale | covers M<128 |
| 2..127 | no | cuBLASLt-scalar + `scale_cols_f32` | **lossy** (1-D blockscale approximation) | acceptable |
| ≥ 128 | yes | CUTLASS SM120 blockwise FP8 GEMM (`cutlass_fp8_gemm_blockscale_sm120`) | full 2-D blockscale | best for M≥128 |
| ≥ 128 | no | cuBLASLt-scalar + `scale_cols_f32` | lossy | acceptable |

`FAST_PATH_M_MAX = 127` is the hard cutoff between the GEMV fast path
and the CUTLASS / cuBLASLt fallback. The fused QKV weight has
**no 2-D blockscale** (only a 1-D channel scale); the same is true for
gate_up. So those weights take the fallback at any M ≥ 128. O-proj
and down_proj do have blockscales and route through CUTLASS at M ≥ 128.

The CUTLASS SM120 blockwise FP8 kernel needs scale-factor preparation
that's specific to the kernel's MN-major expected layout: SFA is
per-token amax-reduced into `[ceil(M/128), K/128]` f32 CUTLASS MN-major;
SFB transposes Gemma 4's row-major `b_chscale` (N-tile outer, K-block
inner) into MN-major (N-tile inner, K-block outer). Both stage into
`scratch_f32`. M=128 ≈ 102 TFLOPS, M=256 ≈ 136 TFLOPS at the QKV shape
(N=4608, K=5376).

---

## 9. The bf16 residual chain (cycle 53–55)

Residual stream is held in **bf16** (8-bit exponent, 7-bit mantissa)
between layers; narrowing to **f16** (5-bit exponent, 10-bit mantissa)
happens at projection input via `bf16_to_f16_sat` (saturating cast,
clamps to ±65504). Inside layers all math stays f32. The bf16 storage
recovers ~3 mantissa bits of dynamic range vs the pre-cycle-54 f16
chain (the absolute residual magnitude grows across 60 layers; f16's
5-bit exponent saturates earlier than bf16's 8-bit exponent).

Embedding gather writes f16; an in-place `f16_to_bf16` widening runs
once per chunk at chunk entry to bring the residual into bf16 land.

The mantissa narrows back to 10 bits at every QKV/MLP entry — the
stage-2.1 narrowing — because the FP8 GEMV fast paths consume f16
input. Wholesale bf16-input kernels (`Fp8GemvBf16In`, the
"full-chain" gate) regress quality even on short context, so they're
left wired but disabled.

---

## 10. Numerical / kernel pitfalls (so you don't re-find these)

### 10.1 Hadamard signs MUST use a SplitMix32 finalizer

FNV-1a's `(h & 1)` collapses to a stride-2 pattern `[+1,-1,+1,-1,...]`.
That's not random — it's a permutation that DEFEATS Hadamard rotation
(the FWHT factors out the alternating sign trivially, leaving the
distribution un-flattened).

Always use SplitMix32 (or any avalanche hash with good low-bit
randomness). See `sign_byte_for()` in `fused_rope_partial_nvfp4kv.cu`.

### 10.2 V-rotation requires unrotation BEFORE O-proj

If V is rotated and the unrotate kernel is missing/skipped, the
O-projection sees a rotated input. The model output is wrong with NO
diagnostic signal — no NaN, no obvious garbage, just semantically off.
The unrotate runs after `decode.launch{,_split}` writes `attn_out`
and before the O-projection reads it. Both decode and prefill paths
need it.

### 10.3 `attn_scale = 1.0` for Gemma 4 (see §7.2)

Don't "fix" this by adding `1/√head_dim`. The QK-norm gammas already
absorb it.

### 10.4 Synchronization on logit DtoH/HtoD

The repetition-penalty path does `cuMemcpyDtoH_v2(logits) → host
modify → cuMemcpyHtoD_v2(logits)` and then the next stream op
(argmax). The DtoH is synchronous WRT the host but only implicitly
syncs the **NULL stream** with named streams in legacy mode. Always
fence the named stream before the DtoH. The HtoD followed by argmax
in the same named stream IS stream-serialized so doesn't need an
explicit fence.

### 10.5 Online softmax masked-tile guards

Both `expf(s - new_M)` for individual scores and
`expf(prev_M - new_M)` for the alpha rescale need explicit guards
against `-FLT_MAX`. `expf(-FLT_MAX - -FLT_MAX) = expf(0) = 1`; without
the guards every all-masked tile contributes `tile_size · 1` to the
row sum.

The guards are in place in the unified prefill, split decode, and the
non-split decode. If you write a new attention kernel, port them.

### 10.6 `Region` is not `Drop`

In `rvllm_mem`, `Region` is a borrow-handle over the arena's bump
pointer. Bytes stay reserved when the bump pointer is advanced, NOT
when the wrapper is held. So `std::mem::forget(region)` is a no-op
masquerading as "leak this region intentionally" — the bytes are
already kept by virtue of the arena's monotonic bump state. Use
explicit comments to describe the bump-pointer + checkpoint
ordering, not `mem::forget`.

### 10.7 Token IDs to memorize

- `48` = `<|tool_call>` (open). Long-context tool-eligible prompts
  often need a +4.0 logit bias here to overcome margin compression.
- `49` = `<tool_call|>` (close).
- `759` = `' la'` — the historical "la la la" cycle target on
  margin-collapsed long contexts. Seeing `759` dominate >40 % of a
  32-token decode window is the canonical signature of NVFP4
  V-quantization noise hitting the cliff.

---

## 11. Provenance / prefix-cache invalidation

`PrefixProvenance` (in `gemma4_bring_up.rs`) tracks every env knob
that changes the BYTE STATE of the K/V cache:

```
chunk_size, kv_dtype, hybrid_global_fp8, hybrid_sliding_fp8,
fp8_kv_layers, scale_policy, k_scale_policy, v_scale_policy,
hadamard, hadamard_v, per_token_q_scale, batch_prefill,
unified_prefill, split_kv, stoch_round_v
```

On request entry, `PrefixProvenance::from_env()` is compared against
the cached `PrefixCacheState.provenance`. Mismatch → invalidate
(treat as cache-miss). This is the only line of defense against
"oh yeah I flipped Hadamard mid-session and the cache reused old K
bytes with new dequant" silent corruption.

Same-request chunk-shape capping: `committed_prefix_len` is set to
`(prompt_len / chunk_size) * chunk_size` — i.e. the last
fully-written chunk boundary. Slots after that boundary were written
by a SHORT trailing chunk and are unsafe to reuse on the next
request, because the next request might use a different chunk_size
that disagrees on the trailing slot.

If you add a new knob that affects K/V byte content, **add it to
`PrefixProvenance`**. Otherwise it'll silently corrupt across requests.

---

## 12. Per-arch PTX manifest

`kernels/sm_<NN>/manifest.json` is the SHA-pinned catalog of every
loaded artifact. The Rust runtime reads the manifest at start-up,
recomputes sha256 of every listed file, and refuses to load if any
hash mismatches. This catches stale PTX after a source edit that
wasn't re-built.

Manifest entries have only `{path, sha256, bytes}`. The version
emitted by the Python `make_manifest.py` builder also includes
`source_sha256` (so re-runs skip rebuilds), but the shipped manifest
format is the simpler `gen_manifest.sh` output. They're compatible
because the loader only verifies the artifact hash.

CUTLASS arch suffix: most kernels build with `-arch=sm_121`. The
CUTLASS sm_120 blockwise FP8 GEMM specifically needs `sm_121a` on
DGX Spark (not `sm_120a` — the family-conditional macro is false on
sm_121 for `120a`).

---

## 13. The 14 800-token-long elephant

The user's primary inference workload is zeroclaw-style prompts
(persona + history + tool descriptions) of 14 000–17 000 tokens with
a one-line user question at the end. This is at the upper end of
what Gemma 4 can handle coherently, and quantization noise compounds
across 60 layers in ways that compress logit margins:

- WHO@~16k clean prompt: top-1 margin ~22.9 (decisive, model is
  confident).
- WEATHER@~14.7k tool-eligible: top-1 margin can drop below 1.0 —
  multiple competing tokens (open-tag, content tokens, garbage
  tokens) within a few logits of each other. Greedy lands wherever
  quantization noise pushed it.

The cycle-53 logit-bias-on-token-48 fix exists exactly for this case:
boosting the tool-call open-tag by +4.0 lifts it above the noise
floor and restores the canonical training-shape continuation.

Quantization improvements (Hadamard + 6-cand MSE + per-token Q
scale) lift the noise floor across the board, but the bias is the
explicit fix for tool-eligible prompts where long-context margins
naturally collapse.

---

## 14. Where each algorithm lives in source

| Concern | File |
|---|---|
| Per-block FP4 quantize + scale search (NVFP4) | `v3/kernels/fused_rope_partial_nvfp4kv.cu` |
| bf16-input variant (when `RVLLM_RESIDUAL_BF16=1` is also enabling Fp8GemvBf16In) | `v3/kernels/fused_rope_partial_nvfp4kv_bf16in.cu` |
| Per-token FP8 quantize + per-(token,head) Q scale | `v3/kernels/fused_rope_partial_fp8kv*.cu` |
| Decode (NVFP4 KV, M=1, BC=32 / BC=16) | `kernels/flash_attention_nvfp4kv.cu` |
| Split decode (long ctx, NVFP4 KV) | `kernels/flash_attention_split_decode_nvfp4kv.cu` |
| Unified prefill (NVFP4 KV, MMA) | `kernels/flash_attention_unified_prefill_nvfp4kv.cu` |
| Hadamard rotation primitives (FWHT + signs) | `kernels/hadamard.cuh` |
| Hadamard unrotate (apply R^T to attn_out) | `kernels/hadamard_unrotate_f16.cu` |
| FP4 / NVFP4 unpack helpers (asm-using `cvt.rn.f16x2.e2m1x2`) | `kernels/nvfp4_utils.cuh` |
| Per-layer dispatch (KV dtype, kernel fn pick, layer-type routing) | `v3/crates/rvllm-runtime/src/gemma4_layer_exec.rs` |
| Run-loop (chunked prefill + per-token decode + sampling + repetition guards) | `v3/crates/rvllm-runtime/src/gemma4_bring_up.rs` |
| Attention-launcher Rust glue (validates pointers, picks kernel symbol, sets dynamic smem) | `v3/crates/rvllm-attention/src/{decode,prefill}.rs` |
| Prefix-cache provenance + chunk-shape cap | `gemma4_bring_up.rs` (`PrefixProvenance`, `init_prefix_cache`, end of `run_generate`) |
| Logit softcap | `v3/kernels/logit_softcap.cu` |
| Bf16↔f16 conversions | `v3/kernels/{bf16_to_f16_sat,f16_to_bf16}.cu` |
| Manifest-verified PTX loader | `v3/crates/rvllm-kernels/src/{lib,manifest,loader}.rs` |
| Bump-arena (HBM-or-host-stub) | `v3/crates/rvllm-mem/src/{hbm,unified}.rs` |

---

## 15. Decision rules for new work

- **New quantization knob** → add to `PrefixProvenance` if it affects
  K/V byte content.
- **New decode kernel** → port the masked-tile softmax guards
  (§10.5).
- **New scale-search variant** → keep MSE-min monotone (only ADD
  candidates, don't remove). Test against the WHO/WEATHER/HA smoke;
  more candidates can hurt because per-block MSE is an APPROXIMATE
  proxy.
- **Change attn_scale** → re-derive against the QK-norm gamma scale;
  Gemma 4's gammas already absorb 1/√d (§7.2).
- **Edit a kernel** → re-run `nvcc -ptx -arch=sm_121 -O3
  --use_fast_math -o kernels/sm_121/<name>.ptx <src>` AND
  `bash kernels/gen_manifest.sh kernels/sm_121 <revision>` to refresh
  the SHA. Otherwise the manifest verifier rejects the new PTX at load.
- **Touching CUTLASS** → arch suffix is `sm_121a` on this hardware.
- **Removing dead code** → confirm with a workspace-wide grep for the
  symbol, including bench / probe binaries. The runtime-loaded
  `compute_qkv_scales_kernel` was a real dead-code finding (loaded but
  never launched); cleanup commit `8a68433`.
- **Tweaking the residual chain** → it's bf16 between layers, f16 at
  projection inputs, f32 inside math kernels. Don't widen f16 to bf16
  inside the QKV/MLP GEMV path — wholesale bf16 chain regresses even
  at short context.

---

## 16. Tactical knobs by goal

**Lowest memory, pure NVFP4, validated quality**:
Hadamard Q+K+V on, K=amax6, V=mse (6-candidate), per-token-Q-scale,
chunk=256, NVFP4 KV across all 60 layers.

**Highest FP8 quality, more memory**:
NVFP4_KV=0 + FP8_KV=1, per-token-Q-scale, chunk=2048 (FP8 has no
chunk cliff), no Hadamard needed (FP8 has more dynamic range).

**Bench / regression baseline**:
Strip every quality knob (Hadamard off, V=amax6, no per-token Q,
no repetition penalty, chunk=0). Produces correct short-ctx output;
DON'T ship this — long-ctx hits the documented margin-compression
cliff.

See `best_configs_sm121.md` for the full top-5-each tables.

---

## 17. Known unsolved problems

1. **Chunk-size cliff at ~288**. Pinning down requires per-tile
   intermediate-state dumps. Closing this is the single largest
   prefill speedup available.
2. **HA tool-result-followup occasionally goes off-topic** at the
   second decode call (post tool execution). Variable, not garbage —
   suspected model-coherence at this prompt length, not quantization.
3. **TFLOPS headroom on prefill** is real. Decode is bandwidth-bound
   on weights, but prefill GEMMs aren't fully utilizing the SM120
   tensor cores at our shapes. Worth a CUTLASS schedule audit.
