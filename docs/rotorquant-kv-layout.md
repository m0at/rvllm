# RotorQuant KV Cache Layout

Status: metadata and CPU reference only. No RotorQuant KV kernel path is enabled in v1.

## Modes

Supported metadata modes:

- `rotor_cl3`
- `planar2`
- `iso4`

Supported quantized coordinate widths are 2, 3, and 4 bits. The v1 KV chunk is 128 dimensions, so Gemma KV heads split into independent 128-d chunks.

## Cache Shape

The packed coordinate payload is summarized as:

```text
[layers, kind(K/V), blocks, block_size, kv_heads, chunks_per_head, packed_chunk_bytes]
```

Where:

- `kind(K/V)` is fixed at 2.
- `chunks_per_head = head_dim / chunk_dim`.
- `packed_chunk_bytes = ceil(chunk_dim * bits / 8)`.
- `chunk_dim` is 128 for v1 helpers.

For a 256-d KV head at 4 bits, each head has 2 chunks and each chunk uses 64 bytes.

Rotation parameters are metadata keyed per `(layer, kind, head, chunk)`. The loader helper counts compact f32 parameters as:

- `rotor_cl3`: 4 f32 values per chunk.
- `planar2`: 2 f32 values per chunk.
- `iso4`: 8 f32 values per chunk.

This is a sizing convention for metadata/reference checks. Runtime kernels may choose a different packed representation later, but must keep the same logical layout.

## Residual Bits

V1 decision: residual bits are disabled by default.

The metadata type can describe `residual_bits = Some(bits)` for experiments, and the sizing helper reports the extra residual stream as:

```text
[layers, kind(K/V), blocks, block_size, kv_heads, chunks_per_head, residual_chunk_bytes]
```

The production v1 helper uses `residual_bits = None`. QJL-style residual sign bits are deferred until there is a decode kernel and an attention-logit correction path.

## Prefill Fallback Rule

Prefill always uses the existing FP8/F16 KV path in v1.

Decode may use RotorQuant only when RotorQuant KV kernels are present and explicitly selected. If kernels are absent, unsupported for the target, or not wired into dispatch, the path must fall back to FP8/F16. The current v1 constant marks RotorQuant KV kernels unavailable, so both prefill and decode fall back.

## Kernel Boundary

The loader module owns only:

- metadata validation,
- packed byte sizing,
- codebook/dequant reference helpers,
- tiny 2D/4D CPU rotation sanity checks.

CUDA, CUTLASS, Metal, or runtime dispatch kernels are future work and should be treated as no-go until an implementation exists and has parity tests against these helpers.
