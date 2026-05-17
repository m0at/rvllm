# rvllm v3 — implementation plan

Consolidates `v3/specs/*.md` into a build order, resolves conflicts, sets the first 3 milestones.

## 1. Locked decisions

### 1.1 Crate layout (16 crates)

Adopting agent 01's DAG with two simplifications:
- `rvllm-config` collapses into `rvllm-core` (one fewer crate, config has no deps anyway).
- `rvllm-stream` collapses into `rvllm-mem` (Stream/Event/CaptureScope must coordinate with Region/Tensor lifetimes; splitting them creates a circular borrow problem).

Final list (target: each crate ≤800 LoC, each file ≤500 LoC):

```
rvllm-core      — error.rs ids.rs dtype.rs shape.rs config/{model,runtime,builder,validate}.rs env.rs
rvllm-mem       — hbm.rs region.rs tensor.rs pinned.rs kv_layout.rs stream.rs event.rs capture.rs graph_safe.rs
rvllm-kernels   — loader.rs sigs.rs artifacts.rs (PTX/CUBIN/.so wrappers)
rvllm-cutlass   — variants.rs plan.rs autotune_policy.rs workspace.rs ffi.rs build.rs
rvllm-attention — fa3_decode.rs fa3_prefill.rs gqa.rs ffi.rs
rvllm-fused     — add_norm_quant.rs rope_kv.rs silu_mul.rs argmax.rs residual.rs ffi.rs
rvllm-metadata  — layout.rs pack.rs handle.rs
rvllm-graph     — capture.rs replay.rs pool.rs validate.rs fingerprint.rs
rvllm-loader    — safetensors.rs fp8_quant.rs placement.rs
rvllm-sampling  — greedy.rs topk_topp.rs dtoh_pinned.rs rng.rs
rvllm-runtime   — layer_exec.rs scheduler.rs lifecycle.rs sched_state.rs engine.rs
rvllm-serve     — http.rs openai.rs
rvllm-bench     — harness.rs gates.rs profile.rs
rvllm-deploy    — tarball.rs spawn.rs deploy_and_bench.rs
rvllm-zig       — bpe.zig topk.zig metapack.zig (FFI shim in rvllm-core)
tools/          — parity, perplexity, bench-gate, autotune (binaries, not crates)
```

DAG enforced by `cargo deny` + `tests/dag.rs`. Cycles forbidden. Runtime stays
the main orchestrator; `rvllm-serve` may forward CUDA feature dependencies, and
`rvllm-bench` may depend on low-level crates used by its direct GPU harnesses.

### 1.2 Conflict resolutions

| Conflict | Specs | Resolution |
|---|---|---|
| `CaptureScope` location | 04, 05, 14 | Lives in `rvllm-mem::capture` (next to `Stream`/`Event`/`Tensor`/`GraphSafe`). `rvllm-graph` USES it; doesn't redefine. |
| Layer forward signature | 01 (`execute_layer`), 09 (`forward`) | Use agent 09's: `fn forward(input, weights, kv, scratch, meta, out, scope) -> Result<()>`. The `scope` borrow forces capture-region awareness. |
| FP8 GEMM API shape | 01 (`plan_fp8_gemm`/`launch`), 11 (`Cutlass::run(VariantId, ...)`) | Both. `plan_fp8_gemm(shape, variant) -> Plan` + `workspace_bytes(&Plan) -> usize` for engine-init workspace sizing. `Cutlass::run(plan, args, ws, scope)` for hot-path launch. Plan caches the variant fn pointer. |
| `Default` impls | 01 (forbid), 02 (LogLevel only) | Forbid everywhere except `LogLevel`. Lint via `clippy.toml`. |
| `LayerScratch` ownership | 04 (Region), 09 (`&mut LayerScratch`) | Allocated from `HbmArena` once; runtime borrows `&mut` per step. Stays inside `CaptureScope` because borrow doesn't outlive the closure. |
| Bucket list | 02, 07, 14 | One source of truth: `RuntimeConfig::graph_capture: GraphMode::Buckets(Vec<u32>)`. Default candidate (NOT a Default impl): `[1,2,4,8,16,32,48,64,96,128,160,192,256]`. Caller must provide explicitly. |
| Workspace coverage | 04, 11 | `cutlass-variants` table is iterated × bucket sizes × every shape (qkv, gate_up, o_proj, down_proj, lm_head if FP8). NO fallback "default" kernel — every callable variant is in the policy.json, and policy.json drives what `max_workspace_size` queries. Eliminates today's class of bug structurally. |

### 1.3 Forbidden in v3 (carried-forward bug class avoidance)

- No `Default` impl on any config or runtime struct (except `LogLevel`).
- No second metadata upload path. ONE `MetaPack::upload(layout, batch, scope)`. No "patch" alternative.
- No `&mut HbmArena` in scope inside a `CaptureScope`. Compile error.
- No `unwrap`/`expect` outside tests. No `Box<dyn Error>`. No `anyhow` in libs.
- No silent fallback chain in dispatch. Missing autotune entry → engine refuses to start.
- No PTX fallback for FA3. `libfa3_kernels.so` REQUIRED, verified at engine init.
- No fused FP8 GEMM with non-cooperative mainloop + cooperative epilogue. CUTLASS variant table rejects mismatched pairs at compile time via `static_assert`.
- No `String` errors crossing crate boundaries. All `Result<T, RvllmError>`.
- No mid-execution alloc/realloc. All buffers sized at engine init from `RuntimeConfig`.
- No "lazy graph capture during warmup". Pre-capture all buckets in `Engine::init`.
- No two execution paths (sync vs pipelined). One `step_launch`/`step_collect` pair.
- No `i32` for token IDs. `TokenId(u32)` newtype.

## 2. Build order (bottom-up)

Each step ships with tests. Next step blocked until previous is green under `compute-sanitizer memcheck` and `clippy -D warnings`.

### Phase A — foundations (~1 week)
1. `rvllm-core`: error enum, IDs, DType, Shape, ModelConfig, RuntimeConfig, RuntimeConfigBuilder, env whitelist.
2. `rvllm-mem`: HbmArena, Region, Tensor, Stream, Event, CaptureScope, GraphSafe trait, PinnedPool, KvLayout.
3. Tests: arena bump + Drop frees; Tensor borrow lifetimes; capture-scope rejects `&mut HbmArena` (`trybuild`); pinned alloc.
4. `rvllm-kernels`: kernel loader (PTX + .so), signature wrappers, artifact manifest.
5. CI gate: `cargo deny check`, `tests/dag.rs`, `tests/loc_budget.rs`.

### Phase B — kernel layer (~1 week)
6. `rvllm-fused`: 8 kernels (embedding_gather, fused_add_rmsnorm_fp8_quant, fused_rmsnorm_fp8_quant, quantize_fp8_per_token, fused_rope_kv_write, fused_silu_mul_fp8_quant, argmax, residual_add). Each kernel ships:
   - .cu source
   - Rust binding with shape/alignment validation
   - Pure-Rust f32 reference
   - Cosine ≥0.999 unit test
7. `rvllm-attention`: `paged_decode` + `paged_prefill` from FA3 SM90 source. .so build script. head_dim=128 hard gate. compute-sanitizer green at every bucket.
8. `rvllm-cutlass`: variant catalog generated by `build.rs` from one source (Rust + CUDA tables sync). `static_assert` on schedule pairs in CUDA. Workspace contract. autotune binary (`tools/autotune`) emits `policy.json`.

### Phase C — orchestration (~1 week)
9. `rvllm-metadata`: frozen layout per `(bucket, max_blocks)`. ONE upload path.
10. `rvllm-loader`: HF safetensors → GPU. GPU-side FP8 quant. Per-tensor scale. Hard-fail on >0.001% clamp.
11. `rvllm-sampling`: greedy argmax + double-buffered DtoH with type-state `DtoHTicket`. Optional top-k/p kernel.
12. `rvllm-graph`: `CaptureScope::record` closure API. `GraphPool::capture_all` at engine init. Fingerprint walk. NO lazy capture.
13. `rvllm-runtime`: `Engine::init` (load → arena alloc → pre-flight → autotune policy load → graph capture). `Engine::step_launch`/`step_collect`. Scheduler emits `BatchPlan::{Prefill, Decode, Idle}`.

### Phase D — validation + ship (~1 week)
14. `tools/parity`: HF reference per-layer cosine compare for Qwen2.5-7B.
15. `tools/perplexity`: WikiText-2 10K tokens, ±0.5% gate vs HF.
16. `rvllm-bench`: harness + JSON output for `tools/bench-gate`.
17. `rvllm-serve`: HTTP loop reusing `step_launch`/`step_collect`.
18. `rvllm-deploy`: SHA-pinned tarball with manifest.json. CI builds artifacts on H100 runner. Refuse deploy if SHA drift.

## 3. Milestones

- **M1 (end of Phase A)**: `RuntimeConfig` builder validates Qwen2.5-7B; HbmArena allocates KV cache and scratch; nothing else.
- **M2 (end of Phase B)**: `rvllm-fused` cosine tests all green; `rvllm-attention` paged_decode passes compute-sanitizer at all buckets on H100; CUTLASS catalog compiles with all schedule-pair `static_asserts`.
- **M3 (end of Phase C)**: Qwen2.5-7B FP8 N=1 greedy run produces identical tokens to HF for 32 prompts × 64 tokens. NO graphs yet.
- **M4 (end of Phase D)**: Graph capture green at all buckets. Bench at N=128 ≥ April 15 baseline (19,259 tok/s). Compute-sanitizer clean. Deploy via tarball+manifest verification.

## 4. Carried-forward facts (from today's debugging)

These go into v3 as test fixtures or compile-time constraints:
- CUTLASS variants with mismatched WS↔Coop schedules cause ILLEGAL_ADDRESS only inside graph replay. Agent 11 enumerated v2's offenders: `cutlass_fp8_gemm_v{0,2,3,4,6,8,11,12,13,14}` and `cutlass_fp8_gemm_residual_v{0,2,4,5,6}`. v3's CUDA `static_assert` rejects any new mismatched variant at compile time.
- `fa3_v3_decode_gqa_kernel` reads `block_tables[seq * max_blocks + page_idx]` for `page_idx` derived from `context_lens[seq]`. Must predicate on `context_lens[i]==0` and never exceed `ceil(context_lens/block_size)`. Test fixture: long-context replay with one zero context_len padded slot.
- Captured graph + non-padded metadata upload = silent OOB. v3 has no non-padded path.
- vectorized `quantize_fp8_per_token` (uint4 loads) needs `dim % 8 == 0` AND every kernel must ship a cosine-vs-scalar reference test.
- Stale `~/.cache/rvllm/cutlass_autotune.json` from previous deploys causes wrong-variant dispatch. v3 ships policy.json IN the tarball, never reads `~/.cache`.

## 5. Open questions (to resolve before Phase A)

1. **Tokenizer**: keep `tokenizers` crate (Python-pretrained BPE from HF), or implement BPE in Zig per `rvllm-zig`? Decision: keep `tokenizers` for now; revisit if it's the bench bottleneck.
2. **Multiple GPUs**: scope to single-GPU only for v3.0. Multi-GPU is v3.1.
3. **Speculative decode**: out of scope for v3.0. Spec for it lives in `v3/specs/future/`.
4. **gRPC/OpenAI compat**: HTTP only for v3.0. gRPC is v3.1.
5. **CUDA version pinning**: lock to CUDA 12.4 (matches the H100 runner). 12.5+ untested.

## 6. Next action

1. Stop work on v2 (the engine works at 9.5K tok/s; that's the v2 ceiling).
2. Begin Phase A: scaffold `rvllm-core` and `rvllm-mem` Cargo.toml + lib.rs stubs.
3. First PR: just the workspace Cargo.toml + 16-crate skeleton + `cargo deny` config + `tests/dag.rs` + `tests/loc_budget.rs`. No business logic.
