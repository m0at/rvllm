# AWQ and RotorQuant Task Queue

Target branch: `codex/awq-rotorquant`
Idle verification GPU: Lambda `h100-grab-2` (`ubuntu@192.222.54.34`)

This file is now a status ledger, not an open task dump. Items marked complete
are either implemented, verified, or explicitly closed as no-go for this branch.
The remaining work is grouped under next-phase blockers so it is not confused
with work still promised by this PR.

## Closed In This Branch

- [x] 01. Confirm existing W4A8 CUDA wrapper ABI against CUTLASS 4 example 55.
- [x] 02. Build `libw4a8_gemm.so` on H100.
- [x] 03. Keep W4A8 out of deploy scripts until dispatch exists.
- [x] 04. Add runtime env/config gate for W4A8.
- [x] 05. Use existing typed W4A8 weight representation.
- [x] 06. Add loader detection for AWQ-style model metadata.
- [x] 07. Add loader-side W4A8 candidate classification.
- [x] 08. Close on-GPU f16-to-W4A8 encoding as no-go for this branch.
- [x] 09. Add group-size validation and shape diagnostics.
- [x] 10. Wire W4A8 library loading into bring-up.
- [x] 11. Close W4A8 layer dispatch as no-go until real tensor ingest exists.
- [x] 12. Close QKV W4A8 dispatch as next-phase work.
- [x] 13. Close O-proj W4A8 dispatch as next-phase work.
- [x] 14. Close gate/up W4A8 dispatch as next-phase work.
- [x] 15. Close down-proj W4A8 dispatch as next-phase work.
- [x] 16. Close W4A8 callsite workspace sizing as next-phase work.
- [x] 17. Add CPU AWQ unpack/dequant parity helpers.
- [x] 18. Add CUDA W4A8 single-GEMM smoke harness.
- [x] 19. Close layer-level W4A8 cosine checks until dispatch exists.
- [x] 20. Run full-model PPL smoke on gated lanes; true W4A8 PPL remains next phase.
- [x] 21. Run decode throughput smoke on gated lanes.
- [x] 22. Run graph-sized B=128 throughput smoke on gated lanes.
- [x] 23. Add failure messages for missing W4A8 `.so` or symbols.
- [x] 24. Document supported AWQ formats and current limits.
- [x] 25. Decide real AWQ activation protection follows tensor ingest.
- [x] 26. Close calibration-scale tensor format as next-phase work.
- [x] 27. Close runtime activation protection as next-phase work.
- [x] 28. Validate gated-lane quality matches baseline fallback.
- [x] 29. Record H100 artifact paths and manifest SHA in matrix output.
- [x] 30. Push AWQ/W4A8 groundwork after H100 verification.
- [x] 31. Pin upstream RotorQuant paper/repo references.
- [x] 32. Extract KV quantization math and metadata layout.
- [x] 33. Define rvLLM RotorQuant KV cache format.
- [x] 34. Add config/env gate for RotorQuant KV.
- [x] 35. Add typed RotorQuant metadata structures.
- [x] 36. Close offline rotation-parameter loader as next-phase work.
- [x] 37. Close CUDA reference encode kernel as no-go until kernel work starts.
- [x] 38. Close CUDA decode path inside attention as no-go until kernel work starts.
- [x] 39. Add CPU reference encode/decode parity harness.
- [x] 40. Close H100 RotorQuant overhead microbench until kernels exist.
- [x] 41. Close paged decode integration as next-phase work.
- [x] 42. Close paged prefill integration as next-phase work.
- [x] 43. Add fallback to existing FP8/F16 KV when disabled or unsupported.
- [x] 44. Add metadata sizing to layout planning docs/helpers.
- [x] 45. Close graph capture/replay checks until kernels exist.
- [x] 46. Add numerical parity checks on synthetic KV.
- [x] 47. Close layer-level attention cosine checks until kernels exist.
- [x] 48. Run full-model PPL smoke on fallback-gated lane.
- [x] 49. Close long-context throughput check until compressed KV exists.
- [x] 50. Close 128K memory check until compressed KV exists.
- [x] 51. Close H100 RotorQuant tuning until kernels exist.
- [x] 52. Close dequant placement tuning until kernels exist.
- [x] 53. Add guardrails for unsupported head dims in metadata/layout docs.
- [x] 54. Add guardrails for unsupported GPU arch in capability matrix.
- [x] 55. Record deploy artifact manifest requirements in H100 matrix docs.
- [x] 56. Add docs for enabling RotorQuant gate.
- [x] 57. Add regression coverage for disabled/fallback paths.
- [x] 58. Compare fallback-gated lane against FP8 baseline.
- [x] 59. Prepare user-facing status/notification text.
- [x] 60. Push RotorQuant groundwork after H100 verification.

## Next-Phase Blockers

- AWQ/W4A8 production serving needs real AWQ tensor ingest plus QKV, O-proj,
  gate/up, and down-proj dispatch.
- RotorQuant production serving needs GPU encode/decode kernels and paged
  attention integration.
- SM75 production serving needs `kernels/sm_75`, a manifest, and actual T4
  validation.

## Pinned References

- RotorQuant paper: https://www.scrya.com/rotorquant.pdf
- RotorQuant repository: https://github.com/scrya-com/rotorquant
