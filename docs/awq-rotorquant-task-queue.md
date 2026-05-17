# AWQ and RotorQuant Task Queue

Target branch: `codex/awq-rotorquant`
Idle verification GPU: Lambda `h100-grab-2` (`ubuntu@192.222.54.34`)

## AWQ / W4A8

- [x] 01. Confirm existing W4A8 CUDA wrapper ABI against CUTLASS 4 example 55.
- [x] 02. Build `libw4a8_gemm.so` on H100 with current vendored/remote CUTLASS.
- [ ] 03. Add W4A8 build to deploy scripts without affecting FP8 deploys.
- [x] 04. Add runtime env/config gate for W4A8.
- [x] 05. Add typed weight representation for W4A8 int4 weights and packed scales.
- [ ] 06. Add loader detection for AWQ-style model metadata.
- [ ] 07. Add loader path for pre-packed AWQ tensors.
- [ ] 08. Add loader path for on-GPU f16 to W4A8 encoding.
- [ ] 09. Add group-size validation and shape diagnostics.
- [x] 10. Wire W4A8 library loading into bring-up.
- [ ] 11. Thread W4A8 handles into layer execution.
- [ ] 12. Add QKV W4A8 dispatch path.
- [ ] 13. Add O-proj W4A8 dispatch path with residual behavior.
- [ ] 14. Add gate/up W4A8 dispatch path.
- [ ] 15. Add down-proj W4A8 dispatch path with residual behavior.
- [ ] 16. Add workspace sizing for all W4A8 callsites.
- [ ] 17. Add CPU reference dequant/matmul parity harness.
- [x] 18. Add CUDA W4A8 single-GEMM smoke harness.
- [ ] 19. Add layer-level cosine checks.
- [ ] 20. Add full-model PPL check.
- [ ] 21. Add decode throughput check at B=1,8,32,128.
- [ ] 22. Add graph capture/replay check.
- [x] 23. Add failure messages for missing W4A8 .so or symbols.
- [ ] 24. Document supported AWQ formats and current limits.
- [ ] 25. Decide whether real AWQ activation protection lands in v1 or follows.
- [ ] 26. If v1, add calibration-scale tensor format.
- [ ] 27. If v1, apply activation protection in runtime inputs.
- [ ] 28. Validate quality delta vs FP8 baseline.
- [ ] 29. Package artifact paths for deploy.
- [ ] 30. Push AWQ-ready branch after H100 verification.

## RotorQuant

- [x] 31. Pin upstream RotorQuant paper/repo references.
- [x] 32. Extract exact KV quantization math and metadata layout.
- [x] 33. Define rvLLM RotorQuant KV cache format.
- [x] 34. Add config/env gate for RotorQuant KV.
- [ ] 35. Add typed scale/rotation metadata structures.
- [ ] 36. Add offline/loader path for rotation parameters.
- [ ] 37. Add CUDA reference encode kernel for K/V blocks.
- [ ] 38. Add CUDA reference decode path inside attention.
- [ ] 39. Add CPU reference encode/decode parity harness.
- [ ] 40. Add H100 microbench for encode/decode overhead.
- [ ] 41. Add paged decode attention integration.
- [ ] 42. Add paged prefill attention integration.
- [ ] 43. Add fallback to existing FP8/F16 KV when disabled.
- [ ] 44. Add metadata sizing to HBM arena planning.
- [ ] 45. Add graph capture/replay checks.
- [ ] 46. Add numerical parity checks on synthetic KV.
- [ ] 47. Add layer-level attention cosine checks.
- [ ] 48. Add full-model PPL check.
- [ ] 49. Add long-context throughput check.
- [ ] 50. Add 128K context memory check.
- [ ] 51. Tune block size / group size on H100.
- [ ] 52. Tune dequant placement in attention hot loop.
- [ ] 53. Add guardrails for unsupported head dims.
- [ ] 54. Add guardrails for unsupported GPU arch.
- [ ] 55. Add deploy artifact manifest entries.
- [ ] 56. Add docs for enabling RotorQuant.
- [ ] 57. Add regression tests for disabled path.
- [ ] 58. Compare against FP8 KV baseline.
- [ ] 59. Prepare user-facing status/notification.
- [ ] 60. Push RotorQuant-ready branch after H100 verification.

## Pinned References

- RotorQuant paper: https://www.scrya.com/rotorquant.pdf
- RotorQuant repository: https://github.com/scrya-com/rotorquant

## RotorQuant Integration Notes

Upstream RotorQuant replaces TurboQuant's dense `d x d` random rotation with block-diagonal Clifford rotors. The paper's concrete path chunks vectors into 3D groups, applies a sparse rotor sandwich product, quantizes rotated coordinates to 2-4 bits, and optionally carries QJL residual bits for unbiased inner-product estimation. The public repo README also documents the later production direction: PlanarQuant/IsoQuant use 2D/4D blocks for lower FMA count, with llama.cpp integration as the practical reference.

rvLLM v1 format should be chunked at 128 dimensions because the paper validates `d=128` and Gemma4 KV heads are 256 for sliding layers and 512 for global layers. Each KV head is stored as multiple independent 128-d chunks.

Proposed cache layout:

- `mode`: off, rotor_cl3, planar2, iso4.
- `bits`: 2, 3, or 4.
- `chunk_dim`: 128.
- `values`: packed quantized coordinates, `[layers, kind(K/V), blocks, block_size, kv_heads, chunks, packed_dim]`.
- `rot_params`: per `(layer, kind, head, chunk)` rotation parameters; Cl(3,0) uses normalized rotor components, planar2/iso4 use compact 2D/4D parameters.
- `codebook`: per mode/bits Lloyd-Max codebook or scalar quantizer table.
- `residual_bits`: optional QJL sign bits for attention-logit correction.
- `fallback`: existing FP8/F16 KV cache remains the disabled path and the correctness oracle.

Execution plan:

- Prefill: keep the current FP8/F16 attention path first, then add deferred compression of completed KV pages.
- Decode: add dequant + inverse rotation in the paged attention load path, then fuse it once parity is stable.
- Verification: synthetic encode/decode parity, attention cosine by layer, full PPL, long-context memory, and H100 decode throughput against the FP8 KV baseline.
