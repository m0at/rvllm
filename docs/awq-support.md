# AWQ support

The AWQ/W4A8 suggestion is good: it targets weight bandwidth directly on H100.
This branch adds AWQ metadata inspection, W4A8 candidate classification, W4A8
kernel bindings, CPU reference helpers, a W4A8-specific calibrated symmetric
scale planner, and a calibrated-scale encoder ABI. It also has a research-only
dispatch harness that encodes selected F16 source weights as symmetric INT4.

That harness is not AWQ. Symmetric INT4 lacks AWQ's calibrated per-group packing
and activation-aware scale choices, so it should not be messaged as incorporated
performance work. The C ABI can now accept positive calibrated per-group f32
scales before packing them for the CUTLASS LUT, but those scales still need to
come from a real AWQ search/checkpoint and pass H100 PPL/throughput gates before
serving or benchmark claims.

## Supported metadata detection

- Safetensors tensor-name inspection without reading tensor payloads.
- AutoAWQ-style names:
  - `<base>.qweight`
  - `<base>.qzeros`
  - `<base>.scales`
  - `<base>.g_idx`
- Packed weight/scale names:
  - `<base>.weight` plus `<base>.weight_scale`
  - `<base>.weight` plus `<base>.weight_scales`
  - `<base>.weight` plus `<base>.weight_zero_point`
  - `<base>.weight` plus `<base>.weight_zeros`
- JSON-ish config metadata for:
  - `quant_method` / `quantization_method`
  - `bits`, `w_bit`, `weight_bits`, `num_bits`
  - `group_size`, `q_group_size`
  - `zero_point`, `zp`
  - `symmetric` as the inverse of `zero_point`

`AwqTensorSet::is_ready()` requires complete tensor names plus valid AWQ semantics. Missing `bits` or `group_size` is rejected. Missing `zero_point` is inferred from zero-point tensors: qzeros present means asymmetric zero-point AWQ; no zero-point tensor means symmetric AWQ.

`AwqTensorSet::w4a8_candidate()` is the loader-side gate for the next phase. It
returns:

- `Ready` only for a complete, single-layout, 4-bit AWQ tensor set.
- `MetadataOnly` for config-only checkpoints.
- `MissingTensors` for partial tensor sets.
- `UnsupportedFormat` for mixed AWQ naming/layout conventions.
- `InvalidConfig` for unsupported bits, missing group size, or zero-point
  contradictions.

## Supported CPU reference helpers

The CPU helpers in `rvllm_loader::awq` are for tests and format sanity checks only.

- 4-bit AWQ only.
- `qweight` is interpreted as `u32` words packing 8 low-to-high nibbles along the logical row/input dimension.
- `qzeros` is interpreted as `u32` words packing 8 low-to-high nibbles along the output-column dimension.
- AWQ zero-points use the usual stored-minus-one convention: unpacked `qzeros` values are incremented by 1 before dequantization.
- Scales are group-major with shape `[ceil(rows / group_size), cols]`.
- Dequantization computes `(q - zero) * scale` for asymmetric AWQ and `q * scale` for symmetric AWQ.
- `quantize_awq_groups_ref()` can run an activation-weighted clipping search and return packed reference qweight/qzeros/scales plus protected-lane metadata for fixtures.
- `AwqActivationStatsRef` accumulates mean absolute activation importance as
  `[K]` from calibration batches for reference/offline AWQ experiments.
- `calibrate_w4a8_symmetric_scales_ref()` is the W4A8-specific helper for the
  current CUTLASS encoder ABI. It consumes Gemma weights as row-major `[N, K]`,
  activation importance as `[K]`, searches clip ratios per `(row, K-group)`,
  and emits positive f32 scales as `[N, K / group_size]` for
  `rvllm_w4a8_encode_weight_fp16_with_scales`.

## Current dispatch status

- Flag-only `w4a8` lanes verify experiment labels and fallback behavior. They
  are not active W4A8 layer dispatch.
- Real W4A8 dispatch exists only as an off-by-default research harness using
  `RVLLM_W4A8_ENCODE_LAYERS` and `RVLLM_W4A8_MODULES`.
- The W4A8 encoder has an optional calibrated-scale symbol
  `rvllm_w4a8_encode_weight_fp16_with_scales`, but runtime Gemma loading still
  does not ingest complete AWQ qweight/qzeros/scales tensors.
- H100 full-model isolation on May 17 2026 used a branch-local SM90 kernel
  manifest and matched FP8 baseline/flag-only controls. With baseline PPL
  `2433.6324`, one real W4A8 layer produced PPL `15761.0562` for `qkv`,
  nonfinite logits for `o`, PPL `14910.9735` for `gate_up`, and PPL
  `2379.3528` for `down`.
- Down-only 8-layer full-model PPL passed the short quality gate
  (`2475.9716` versus `2433.6324`, ratio `1.0174`), but it lost throughput:
  B=1 was `46.7 tok/s` versus `48.6`, and B=128 was `4958.3 tok/s` versus
  `5222.6`. The truncated debug lanes remain non-reportable.

## Rejected formats

- 2-bit, 3-bit, 8-bit, or mixed-bit AWQ.
- Missing `bits` or `group_size`.
- `zero_point=true` without qzeros.
- `zero_point=false` with qzeros.
- Shape/length mismatches for qweight, qzeros, or scales.
- Production serving-time AWQ selection.
- Runtime weight loading from complete AWQ tensors.
- Asymmetric qzeros dispatch in the current W4A8 CUTLASS kernel.
- Symmetric F16-source INT4 as a substitute for AWQ.
- `g_idx` act-order dequant behavior in the CPU helper. The metadata parser records `g_idx` presence, but the reference dequant path uses contiguous `row / group_size` grouping.

## References consulted

- Google Gemma 4 docs: 31B is a dense model, available at default precision or
  lower precision, and Google lists approximate 31B inference memory as
  58.3 GB BF16, 30.4 GB SFP8, and 17.4 GB Q4_0.
- Hugging Face Transformers AWQ docs: AWQ checkpoints are identified by
  `quant_method: awq`, commonly with `bits: 4`, `group_size: 128`, and
  `zero_point` metadata.
- AWQ paper, arXiv:2306.00978: AWQ is activation-aware, uses offline
  activation statistics to protect salient channels, and is not equivalent to
  uncalibrated symmetric int4 packing.
