// rvllm-bench — scaffold only.
//   pub mod harness;  // deterministic bench loop
//   pub mod gates;    // regression gate output (JSON)
//   pub mod profile;  // nsys/ncu hooks
//
// GB10 clock/power sampling lives in v3/tools/fp8_gemv_bench.py —
// the actual bench workflow is Python (cuda-python) because the
// Rust runtime's launch path isn't wired up for fp8_gemv yet.
