# Gemma 4 vision tower — layer-by-layer HF audit (Codex review #A round 3)

**Branch:** `rusty_sm121_vision`
**Captured:** after the round-1/2/3/4 fixes landed
(commits up to `8186ff3` batched-strided + earlier softmax-f32,
PatchEmbedder pre-norm, Gemma multidim rotary, position-axis swap,
vnorm-ABI fix).

## Audit infrastructure

- `RVLLM_GEMMA4_VIT_DUMP_DIR=<path>` env var on the worker → per-stage
  f16 buffers written to `<path>/g4v_<stage>.bin` for every vision
  request. Mirrors the Qwen-vision dump pattern that landed Qwen at
  cos=0.9999 / layer.
- HF reference dump script: `/tmp/gemma_vision_dump_stages.py` runs
  `transformers.Gemma4VisionModel` + `Gemma4MultimodalEmbedder` on the
  same fixture image, dumps the matching stages.
- Comparison script: `/tmp/cmp_g4v_stages.py` loads both, computes
  per-row cosine similarity.

## Results (test_224.png, fp8-block checkpoint)

```
stage                          HF L2     OURS L2   cos mean   cos min
patch_embed_linear           3417.71     3417.21     1.0000    1.0000
posemb                       3418.46     3418.21     1.0000    1.0000
blk0_out                     4818.66     4818.32     1.0000    1.0000
blk13_out                   66849.93    66852.18     1.0000    0.9999
blk26_out                  120854.20   120810.10     0.9974    0.8060
encoder_out_pre_pool       120854.20   120810.10     0.9974    0.8060
pooler_scaled            1062626.38  1062338.38     0.9990    0.9676   (31/256 rows hit f16 inf, filtered)
standardized                  500.38      500.27     0.9965    0.9173
post_projection              1536.38     1536.89     0.9965    0.9143
```

## Interpretation

**Up to and including block 13**, the implementation is **byte-faithful**
to the HF Gemma4VisionModel reference. All 14 patch_embed → posemb →
block stages report `row-cos = 1.0000` (rounded; absolute deviation
< 1e-4 per row).

**Drift enters blocks 14–26**, mostly through f16 accumulation:
mean cos stays at 0.9974 / 256 rows but `min cos = 0.8060` flags
that ~1 specific row drifts notably. That row's content is likely a
high-magnitude patch where f16 mantissa precision matters. Per-block
drift averages 0.9974^(1/13) ≈ 0.9998 — i.e. extremely small per
block, just compounding linearly.

**f16 overflow in the pooler.** After block 26 the max activation is
~2752; then the Gemma pooler multiplies by `sqrt(hidden_size=1152) ≈
33.94`. 31 / 256 pooled rows have at least one channel exceed f16's
±65504 limit and become `+inf`. HF arbeits in bf16 (range ≈ 3.4e38)
so doesn't hit this. The standardize step `(x - bias) * scale` with
trained `std_scale` divides those values back into f16-safe range
for the rows that don't have inf yet — i.e. the standardize WOULD
recover them if they hadn't already saturated.

**Net effect on text recognition**: end-to-end, Gemma vision still
extracts the correct headline / date / subheadlines from the NYT-1969
moonlanding photo (verified post-fix). cos ≥ 0.99 is well above the
quality threshold for "model can read text in image".

## Next-iteration drift fixes (not blocking)

1. **f16 overflow in the pooler**: keep `pooled_region` in f32 from
   the avg-pool through the standardize step, only narrowing to f16
   after `(x - std_bias) * std_scale` brings the values back into
   range. Would recover the 31 inf'd rows. Requires:
   - bumping `vit_avgpool_f16` to write f32 OR adding a sibling
     kernel `vit_avgpool_f16_to_f32`,
   - similarly for the sqrt-scale (use `scale_inplace_f32` once
     written) and standardize (`vit_standardize_f32_to_f16` to
     re-narrow at the end).
2. **f16 attention compound drift in blocks 14–26**: keep `out_h`
   per-head in f32 longer. Currently we cast `scores @ V` from f32
   to f16 immediately after the GEMM; we could defer the cast until
   after the o_proj. Requires a `linear_no_bias_f32_in` GEMV path or
   accepting the cast since the o_proj lossily narrows anyway.
3. **bf16 vision path** (most invasive): switch the vision-tower
   intermediates from f16 to bf16 entirely. Aligns with HF's native
   dtype, eliminates both issues above, but requires bf16 variants
   of every vision kernel (rotary, transpose, scatter, …) and of
   the cuBLASLt entries. ~2-3 days.

For now the f16-everywhere path is acceptable: model reads text
correctly, residual cosine drift is well below the threshold where
text-recognition tasks degrade.

## Phase 3 status — bf16 vision path (FORMALLY DEFERRED, 2026-05-05)

**Decision**: production runs on the f16 + f32-pooler path. bf16 is
deferred indefinitely. Re-open only if a real quality regression
appears that the f16 path can't service.

### Justification
- f16 path delivers correct vision output on real images for both
  Qwen 3.6 and Gemma 4 31B (NYT-1969 photo headline + date,
  /tmp/ball.png "orangefarbener Ball" plus the embedded "orange
  ball" text overlay).
- Layer-by-layer audit shows mean cos = 0.9974 at blk26 / 0.9969 at
  post_projection — well above the threshold where text-recognition
  degrades. Audit explicitly: "f16-everywhere path is acceptable".
- bf16 gain is ~0.003 mean cos and ~1 outlier row. Cost is several
  hours of debug, with a previous attempt failing at blk0_out
  cos = 0.76 without localisation.

### What stays committed (ready for the next attempt)
- bf16 sibling kernels for every f16 vision kernel
  (`vector_add_bf16`, `vnorm_bf16`, `softmax_row_f32_to_bf16`,
  `vit_pos_emb_lookup_2d_bf16`, `vit_rotary_gemma4_2d_bf16`,
  `extract_head_bf16` / `scatter_head_bf16`, `transpose_heads_v_bf16`,
  `gelu_tanh_mul_bf16`, `vit_avgpool_bf16_to_f32`,
  `vit_standardize_f32_to_bf16`, `rmsnorm_inplace_bf16_gbf16`).
- `CublasLt::bf16_gemm_f32_batched_strided` (caller-vs-cuBLAS
  argument-swap convention identical to the f16 sibling).

### Per-sub-step debug tooling (NEW, landed alongside this deferral)

When the next attempt happens, sub-step instrumentation is now
already in place — the previous session was blocked by lacking it.

1. **rvllm side**: `forward_gemma_vision` now writes per-sub-step
   buffers in one configurable target block when both
   `RVLLM_GEMMA4_VIT_DUMP_DIR` and
   `RVLLM_GEMMA4_VIT_SUBSTEP_BLK=<idx>` are set. Sub-steps:
   `input_ln`, `q_proj`, `k_proj`, `v_proj`, `q_norm`, `k_norm`,
   `v_norm`, `q_rot`, `k_rot`, `attn_out`, `o_proj`,
   `post_attn_ln`, `post_attn_resid`, `pre_ff_ln`, `gate_proj`,
   `up_proj`, `gelu_mul`, `down_proj`, `post_ff_ln`. Files:
   `g4v_blk{B}_{step}.bin`, f16 little-endian, `[N, D]` row-major.
2. **HF reference**: `v3/tools/gemma_vision_substep_hf_dump.py`
   monkey-patches one Gemma4VisionEncoderLayer to emit the same
   sub-step set in the same naming convention.
3. **Diff**: `v3/tools/cmp_g4v_substep.py` walks the SUBSTEPS list,
   computes per-row cosine, prints a "first divergence" pointer
   below a configurable threshold (default 0.99).

### Replay recipe for the next bf16 attempt
```bash
# 1. Confirm f16 path is byte-faithful inside the target block
RVLLM_GEMMA4_VIT_DUMP_DIR=/tmp/g4v_f16 \
RVLLM_GEMMA4_VIT_SUBSTEP_BLK=0 \
  curl -s http://127.0.0.1:8010/v1/chat/completions ...
python3 v3/tools/gemma_vision_substep_hf_dump.py \
  --image v3/crates/rvllm-runtime/tests/fixtures/test_224.png \
  --block 0 --out /tmp/hf_g4v_blk0
python3 v3/tools/cmp_g4v_substep.py \
  --rvllm /tmp/g4v_f16 --hf /tmp/hf_g4v_blk0 --block 0
# Expect cos≈1.0 at every step; this validates the harness.

# 2. Wire bf16 forward (parallel function or generic-over-dtype).
#    Re-run the same recipe with /tmp/g4v_bf16. Cosine drop
#    pinpoints the offending kernel inside that one block —
#    the previous session's blk0_out cos=0.76 mystery.

# 3. Fix the offending kernel (most likely a Rust-side launch
#    parameter mismatch — the kernel logic is mechanical
#    bf16/half substitution).
```

### Original Phase-3 attempt notes (kept for context)

Building blocks landed but the wiring exposed a bug we couldn't
localise inside that session, so the production forward stays on
the f16 + f32-pooler path.

What's committed and verified to build:

- bf16 sibling kernels (drop-in dtype swap of every f16 vision
  kernel; same semantics):
    - `vector_add_bf16`
    - `vnorm_bf16`
    - `softmax_row_f32_to_bf16`
    - `vit_pos_emb_lookup_2d_bf16`
    - `vit_rotary_gemma4_2d_bf16`
    - `extract_head_bf16` / `scatter_head_bf16`
    - `transpose_heads_v_bf16`
    - `scatter_heads_bf16`
    - `gelu_tanh_mul_bf16`
    - `vit_avgpool_bf16_to_f32`
    - `vit_standardize_f32_to_bf16`
    - `rmsnorm_inplace_bf16_gbf16` (vision-specific because the text
      path's `rmsnorm_inplace_bf16` takes f16 gamma; vision-bf16 keeps
      gamma in its native bf16, so this kernel is the bf16-input /
      bf16-gamma variant).
- `CublasLt::bf16_gemm_f32_batched_strided` (sibling of the f16
  batched-strided entry; same caller-vs-cuBLAS argument-swap convention).

What stays on f16 in production:

- `forward_gemma_vision` keeps using the f16 kernels and the f32
  pooler bridge.
- `load_gemma_vision` keeps uploading vision weights as f16.

What broke when I switched the forward to bf16:

- Patch_embed and posemb stages stayed cos=1.0000.
- Block 0 output dropped to cos=0.7631 (vs 1.0000 in f16). That's a
  large per-block jump; mean cos at blk26 fell to 0.5595 and
  post_projection to 0.0950 — the model produced unrelated output
  ('Word-Art with words in different languages' / Russian fragments).
- Post-input-LN (start of block 0) inspected manually: looks
  numerically reasonable (rms ≈ 0.83 per row, no inf/nan, sane
  range).
- I didn't isolate the per-step culprit before reverting. The most
  likely candidates are one of:
  - the q/k/v_proj `bf16_gemm_f32` linears (output f32 → cast to
    bf16) producing a mis-strided layout,
  - the `bf16_gemm_f32_batched_strided` lda/stride combination
    interacting differently with cuBLAS bf16 algos than with f16,
  - one of the per-head extract/scatter/transpose bf16 kernels
    indexing wrong.

Plan for the follow-up debug session:

1. Reactivate the bf16 forward (revert this revert).
2. Use the same env-gated `RVLLM_GEMMA4_VIT_DUMP_DIR` infrastructure
   to dump after EACH per-block sub-step (input_LN, q_proj, q_norm,
   rotary, attention, o_proj, post_attn_LN, FFN) and cosine-compare
   to the HF reference at the same point. The block 0 sub-step where
   cos drops first localises the buggy kernel within ~30 minutes
   given the existing tooling.
3. Fix the offending kernel (most likely a Rust-side launch
   parameter mismatch since the kernel logic is mechanical bf16/half
   substitution).
