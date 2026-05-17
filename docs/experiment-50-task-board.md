# rvLLM Quantization and Architecture Experiment Board

Branch: `codex/awq-rotorquant`

Goal: turn the community request into a controlled experiment lane for SM75,
AWQ/W4A8, RotorQuant KV, and serving validation. Each item should either land
behind an explicit controller gate or leave a clear no-go result.

Status note: `[x]` means implemented, documented, or verified on this branch.
`[ ]` means pending, blocked, or no-go for this branch. H100 results live in
`docs/experiment-go-nogo.md`.

## Control Plane

- [x] 01. Add core GPU capability matrix for SM75, SM80, SM89, SM90, and SM121.
- [x] 02. Add runtime experiment controller with quant, KV, attention, arch, and validation axes.
- [x] 03. Add server experiment labels for logs/responses.
- [x] 04. Wire controller summary into Gemma 4 bring-up logs.
- [x] 05. Add controller docs for every `RVLLM_EXPERIMENT_*` env.
- [x] 06. Add a single H100 matrix runner that records every trial to JSONL.
- [x] 07. Keep default production path identical when no env gate is set.
- [x] 08. Add failure-mode notes for missing kernels, missing .so files, and unsupported arch.

## SM75 / Wider Architecture Support

- [x] 09. Map compute capability 7.5 to `sm_75`.
- [x] 10. Define SM75 support as compatibility-only unless kernels prove out.
- [ ] 11. Build PTX manifest for SM75 fused kernels. No-go until kernels exist.
- [ ] 12. Identify kernels using FP8 tensor cores or Hopper-only instructions. Pending detailed audit.
- [x] 13. Add clear unsupported errors for FP8/W4A8 routes on SM75.
- [ ] 14. Add FA2 fallback plan for SM75 attention. No-go for this branch.
- [ ] 15. Add a small local manifest validation test for `sm_75`. Pending manifest.
- [x] 16. Document expected SM75 limits and what would be needed for T4.

## AWQ / W4A8

- [x] 17. Add AWQ metadata detection for common tensor naming conventions.
- [x] 18. Add AWQ group-size/bits/zero-point validation.
- [x] 19. Add AWQ tensor-set summary for Gemma 4 layers.
- [x] 20. Add CPU AWQ unpack/dequant reference for tiny matrices.
- [ ] 21. Add loader-side W4A8 candidate selection behind metadata-only mode. Pending real tensor ingest.
- [x] 22. Add W4A8 dispatch candidate checks to the controller.
- [x] 23. Add H100 W4A8 smoke to the matrix runner.
- [x] 24. Add H100 W4A8 compute-sanitizer lane.
- [ ] 25. Add a real AWQ checkpoint detection fixture when available. Pending fixture.
- [ ] 26. Benchmark W4A8 small GEMM vs FP8 on H100. Reserved; no result claimed.
- [ ] 27. Benchmark W4A8 serving only after layer dispatch is wired. No-go until dispatch exists.
- [x] 28. Document AWQ formats supported and rejected.

## RotorQuant

- [ ] 29. Move runtime RotorQuant config into a reusable module. Pending; metadata helpers exist.
- [x] 30. Add RotorQuant metadata validation for mode, bits, chunk, residuals, codebook.
- [x] 31. Add tiny CPU reference rotation/dequant helpers.
- [x] 32. Add packed-byte sizing helpers.
- [x] 33. Add RotorQuant KV cache layout summary.
- [x] 34. Add controller gating for `rotor_cl3`, `planar2`, and `iso4`.
- [ ] 35. Add decode-only attention integration point behind an off-by-default gate. No-go for this branch.
- [x] 36. Add prefill fallback rule when RotorQuant is enabled.
- [x] 37. Add parity harness for tiny KV blocks.
- [ ] 38. Add H100 decode smoke once kernels exist. Reserved; no result claimed.
- [ ] 39. Measure RotorQuant overhead before fusion. Reserved; no result claimed.
- [x] 40. Decide whether residual bits land in v1 or a later pass.

## Serving Quality and Benchmarks

- [x] 41. Keep Gemma 4 chat template aligned with the model template.
- [x] 42. Keep stop-token handling sourced from `generation_config.json`.
- [x] 43. Add angular momentum chat smoke to H100 matrix.
- [x] 44. Add PPL smoke with fixed prompt and chunk size.
- [x] 45. Add B=1 decode tok/s lane.
- [x] 46. Add B=128 graph tok/s lane.
- [x] 47. Record GPU memory before and after every H100 run.
- [x] 48. Record exact branch SHA and kernel manifest SHA.
- [x] 49. Produce a go/no-go summary for each experiment lane.
- [x] 50. Push only lanes that pass local checks and bounded H100 verification.

## Worker Ownership

- Worker 1: core architecture capability surface.
- Worker 2: runtime experiment controller.
- Worker 3: AWQ metadata detection.
- Worker 4: RotorQuant metadata/reference helpers.
- Worker 5: H100 experiment matrix runner and docs.
- Worker 6: server experiment labels.

## Literature Guardrails

- AWQ is weight-only PTQ in its common serving form: W4A16/W4A8 experiments
  should preserve its activation-aware scale/search semantics instead of
  treating all int4 checkpoints as equivalent.
- Hugging Face AWQ support expects explicit quantization metadata; rvLLM should
  reject ambiguous int4 tensor sets rather than guessing silently.
- RotorQuant/PlanarQuant/IsoQuant are KV-cache compression routes; keep them
  separate from weight quantization and test attention fidelity before any
  throughput claims.
- SM75 support is a compatibility lane, not a Hopper parity promise: FP8 tensor
  core and W4A8 CUTLASS paths should stay gated off until equivalent kernels
  exist.
