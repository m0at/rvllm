# AWQ support

The AWQ/W4A8 suggestion is good: it targets weight bandwidth directly on H100.
This branch adds AWQ metadata inspection, W4A8 candidate classification, W4A8
kernel bindings, tiny CPU reference helpers, and a calibrated-scale encoder ABI.
It also has a research-only dispatch harness that encodes selected F16 source
weights as symmetric INT4.

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

## Current dispatch status

- Flag-only `w4a8` lanes verify experiment labels and fallback behavior. They
  are not active W4A8 layer dispatch.
- Real W4A8 dispatch exists only as an off-by-default research harness using
  `RVLLM_W4A8_ENCODE_LAYERS` and `RVLLM_W4A8_MODULES`.
- The W4A8 encoder has an optional calibrated-scale symbol
  `rvllm_w4a8_encode_weight_fp16_with_scales`, but runtime Gemma loading still
  does not ingest complete AWQ qweight/qzeros/scales tensors.
- H100 smoke on May 17 2026 used a branch-local SM90 kernel manifest and found
  the symmetric real-dispatch path numerically unusable: baseline PPL was
  `49.3464`, while one-layer W4A8 isolation produced PPL `3.984e13` for `qkv`,
  `4.765e9` for `o`, `1.595e13` for `gate_up`, and `1.120e13` for `down`.
- Down-only 8-layer throughput was fast in the synthetic bench (`254.9 tok/s`
  at B=1 and `25468.8 tok/s` at B=128), but PPL was `2.520e15`, so it is not
  valid performance work.

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
