// =============================================================
//  cutlass_nvfp4_gemm_sm120.cu — Mistral 3.5 NVFP4 GEMM (sm_121a)
// =============================================================
//
// Mistral Medium 3.5 ships with all decoder linears in the
// `nvfp4-pack-quantized` format:
//
//   weight_packed        U8       [N, K/2]    (two E2M1 / byte, low first)
//   weight_scale         E4M3     [N, K/16]   (one scale per 16-K block)
//   weight_global_scale  F32      [1]
//
// Activations enter the GEMM dynamically quantized to FP8 E4M3 with
// per-token scales; this is the standard Blackwell FP8 × NVFP4
// tensor-core path. The compiled output is BF16/F16, matching the
// surrounding runtime convention.
//
// THIS FILE IS A SKELETON. The Rust-side ABI scaffolding in
// `rvllm-cutlass::lib_so::CutlassSm120Lib` is fully wired (Step 4
// Rust scaffolding); the actual CUTLASS kernel implementation is
// the next milestone. Until it lands, the source is intentionally
// NOT included in `kernels/build_cutlass_sm120_so.sh::SOURCES` so
// that the resulting `libcutlass_sm120.so` is missing every NVFP4
// symbol — `CutlassBackend::require_nvfp4()` then refuses Mistral
// 3.5 startup with a clear error rather than silently routing
// through a stub.
//
// To bring this in:
//
//   1. Implement the four entry points below using
//      `cutlass::CollectiveBuilder<Sm120, ...>` with an FP8 ×
//      NVFP4 tile shape — reference Blackwell FP4 examples in
//      cutlass/4.4.2 (see examples/82_blackwell_gemm_fp4 and
//      57_hopper_grouped_gemm_fp8).
//   2. Append the filename to `SOURCES=` in
//      `kernels/build_cutlass_sm120_so.sh`.
//   3. Rebuild via `bash kernels/build_cutlass_sm120_so.sh sm_121a`.
//   4. Verify `CutlassBackend::nvfp4_active()` returns true at
//      Mistral startup; the `--model-family mistral35` path then
//      proceeds into Mistral35Bringup::load.
//
// ABI declared here for documentation / future implementation. The
// Rust types live in `v3/crates/rvllm-cutlass/src/lib_so.rs` —
// keep both sides in lock-step when changing.
//
//   extern "C" int cutlass_nvfp4_gemm_sm120(
//       void*       output,                  // BF16 or F16, [M, N]
//       const void* a_fp8,                   // FP8 E4M3,    [M, K]
//       const void* b_packed,                // U8,          [N, K/2]
//       const void* sfa,                     // staged per-token scales
//       const void* b_scale_e4m3,            // E4M3,        [N, K/16]
//       const void* b_global_scale_f32,      // F32,         [1]
//       int m, int n, int k,
//       void* workspace, size_t workspace_size,
//       cudaStream_t stream);
//
//   extern "C" int cutlass_nvfp4_gemm_sm120_decode_m1(
//       /* identical signature; specialised for m == 1 */ );
//
//   extern "C" size_t cutlass_nvfp4_gemm_sm120_workspace(int m, int n, int k);
//
//   extern "C" size_t cutlass_nvfp4_gemm_sm120_sfa_bytes(int m, int k);
//
//   extern "C" int cutlass_nvfp4_gemm_sm120_prep_sfa(
//       const void* a_input_fp16_or_bf16,    // f16 (dtype=0) or bf16 (=1)
//       void*       a_fp8_out,
//       void*       sfa_out,
//       int m, int k,
//       int a_input_dtype,                   // 0=F16, 1=BF16
//       cudaStream_t stream);
//
// All entry points return 0 on success, non-zero on failure. The
// Rust wrappers in `CutlassSm120Lib::launch_nvfp4_gemm` /
// `launch_nvfp4_prep_sfa` map non-zero rc to
// `CutlassError::KernelLaunchFailed` so callers see a clean error
// rather than a silent NaN propagation through the decoder.
