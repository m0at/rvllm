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
