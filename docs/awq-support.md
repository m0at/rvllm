# AWQ support

This branch only adds AWQ metadata inspection and tiny CPU reference helpers. It does not add serving dispatch, GPU kernels, or weight loading.

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

## Supported CPU reference helpers

The CPU helpers in `rvllm_loader::awq` are for tests and format sanity checks only.

- 4-bit AWQ only.
- `qweight` is interpreted as `u32` words packing 8 low-to-high nibbles along the logical row/input dimension.
- `qzeros` is interpreted as `u32` words packing 8 low-to-high nibbles along the output-column dimension.
- AWQ zero-points use the usual stored-minus-one convention: unpacked `qzeros` values are incremented by 1 before dequantization.
- Scales are group-major with shape `[ceil(rows / group_size), cols]`.
- Dequantization computes `(q - zero) * scale` for asymmetric AWQ and `q * scale` for symmetric AWQ.

## Rejected formats

- 2-bit, 3-bit, 8-bit, or mixed-bit AWQ.
- Missing `bits` or `group_size`.
- `zero_point=true` without qzeros.
- `zero_point=false` with qzeros.
- Shape/length mismatches for qweight, qzeros, or scales.
- GPU dispatch or serving-time AWQ selection.
- Runtime weight loading from AWQ tensors.
- `g_idx` act-order dequant behavior in the CPU helper. The metadata parser records `g_idx` presence, but the reference dequant path uses contiguous `row / group_size` grouping.
