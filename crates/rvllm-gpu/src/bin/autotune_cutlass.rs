//! Standalone CUTLASS autotune benchmark: HGEMM, oproj+residual, gateup+SiLU.
//!
//! For each (M, N, K) shape used by Qwen2.5-7B decode, benchmarks all compiled
//! CUTLASS WGMMA tile/cluster/schedule variants plus cuBLAS baseline.
//! Picks the fastest variant per shape and writes the results to JSON cache.
//!
//! Usage:
//!   cargo run --release --features cuda --bin autotune-cutlass [-- --so <path> --out <path>]

use cudarc::cublas::sys as cublas_sys;
use cudarc::driver::sys as cu_sys;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::f16;
use rvllm_gpu::cutlass_autotune::CutlassAutotuneCache;
use rvllm_gpu::cutlass_ffi::{CutlassKernels, HGEMM_VARIANTS, OPROJ_RESIDUAL_VARIANTS, GATEUP_SILU_VARIANTS, FP8_GEMM_VARIANTS, FP8_GEMM_RESIDUAL_VARIANTS};
use std::ffi::c_void;
use std::path::PathBuf;

// -- Qwen2.5-7B decode shapes: (N, K, label) --
const SHAPES: &[(usize, usize, &str)] = &[
    (4608, 3584, "QKV"),
    (3584, 3584, "O-proj"),
    (37888, 3584, "gate_up"),
    (3584, 18944, "down"),
    (152064, 3584, "LM head"),
];

// Fused family shapes: oproj+residual and gateup+silu
// oproj: C = A @ W + residual, shape (M, hidden, q_dim) = (M, 3584, 3584)
// down_proj+residual reuses same kernel: (M, hidden, intermediate) = (M, 3584, 18944)
const OPROJ_SHAPES: &[(usize, usize, &str)] = &[
    (3584, 3584, "O-proj+residual"),
    (3584, 18944, "down_proj+residual"),
];

// gateup+silu: C = SiLU(A @ W), shape (M, gate_up_dim, hidden) = (M, 37888, 3584)
const GATEUP_SHAPES: &[(usize, usize, &str)] = &[
    (37888, 3584, "gate_up+SiLU"),
];

// FP8 GEMM shapes: same weight matrices, FP8 precision
// Skip LM head (done in F16)
const FP8_SHAPES: &[(usize, usize, &str)] = &[
    (4608, 3584, "QKV"),
    (3584, 3584, "O-proj"),
    (37888, 3584, "gate_up"),
    (3584, 18944, "down"),
    (152064, 3584, "lm_head"),
];

// FP8 GEMM + residual shapes: down-proj fused with residual add
// C = A @ W + residual, shape (M, hidden, intermediate) = (M, 3584, 18944)
// FP8 GEMM + residual shapes: O-proj and down-proj fused with residual add
const FP8_RESIDUAL_SHAPES: &[(usize, usize, &str)] = &[
    (3584, 3584, "O-proj+residual"),
    (3584, 18944, "down+residual"),
];

const M_VALUES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256];

const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;
const WORKSPACE_BYTES: usize = 16 * 1024 * 1024;

const VARIANT_NAMES: [&str; HGEMM_VARIANTS] = [
    // v0-v19: base WS/Coop/PP variants
    " 64x 64x 64 1x1x1 WS",
    " 64x128x 64 1x1x1 WS",
    "128x128x128 1x1x1 Coop",
    " 64x256x 64 1x1x1 WS",
    "256x256x 64 1x1x1 Coop",
    " 64x256x 64 1x2x1 WS",
    "128x128x 64 1x1x1 WS",
    "128x128x 64 1x1x1 Coop",
    "128x128x 64 2x1x1 WS",
    "128x128x128 1x1x1 WS",
    "128x256x 64 1x1x1 WS",
    "128x256x 64 1x1x1 Coop",
    "128x256x 64 1x2x1 WS",
    "128x256x 64 1x2x1 Coop",
    "128x256x 64 1x2x1 PP",
    "128x256x128 1x2x1 WS",
    "256x128x 64 1x1x1 WS",
    "256x128x 64 2x1x1 WS",
    "256x128x 64 2x1x1 Coop",
    "256x256x 64 1x1x1 WS",
    // v20-v23: 4-SM cluster variants
    "128x256x 64 2x2x1 WS",
    "128x256x 64 2x2x1 Coop",
    " 64x256x 64 1x4x1 WS",
    "128x256x 64 1x4x1 WS",
    // v24-v29: PingPong mirrors
    " 64x128x 64 1x1x1 PP",
    " 64x256x 64 1x1x1 PP",
    "128x128x 64 1x1x1 PP",
    "128x256x 64 1x1x1 PP",
    "256x128x 64 2x1x1 PP",
    "128x256x 64 2x2x1 PP",
    // v30-v35: explicit pipeline stages
    "128x256x 64 1x2x1 WS s2",
    "128x256x 64 1x2x1 WS s4",
    "128x128x 64 1x1x1 WS s2",
    "128x128x 64 1x1x1 WS s4",
    " 64x256x 64 1x2x1 WS s2",
    " 64x256x 64 1x2x1 WS s4",
    // v36-v41: swizzle rasterization
    "128x256x 64 1x2x1 WS z2",
    "128x256x 64 1x2x1 WS z4",
    "128x256x 64 1x1x1 WS z2",
    "128x256x 64 1x1x1 WS z4",
    " 64x256x 64 1x2x1 WS z2",
    " 64x256x 64 1x2x1 WS z4",
    // v42-v50: split-K M>=128 (Coop + StreamKScheduler)
    "128x256x 64 1x1x1 SK2",
    "128x256x 64 1x1x1 SK4",
    "128x256x 64 1x1x1 SK8",
    "256x128x 64 1x1x1 SK2",
    "256x128x 64 1x1x1 SK4",
    "256x128x 64 1x1x1 SK8",
    "128x128x 64 1x1x1 SK2",
    "128x128x 64 1x1x1 SK4",
    "128x128x 64 1x1x1 SK8",
    // v51-v54: stream-K M>=128
    "256x128x 64 1x1x1 StK",
    "128x128x 64 1x1x1 StK",
    "128x256x 64 1x1x1 StK",
    "256x256x 64 1x1x1 StK",
    // v55-v62: M=64 WS/PP additional (Coop/split-K need M>=128)
    " 64x256x 64 1x1x1 WS s2",
    " 64x256x 64 1x1x1 WS s4",
    " 64x128x 64 1x1x1 WS s2",
    " 64x128x 64 1x1x1 WS s4",
    " 64x256x 64 1x2x1 PP",
    " 64x128x 64 1x2x1 PP",
    " 64x256x 64 1x1x1 WS z2",
    " 64x256x 64 1x1x1 WS z4",
    // v63-v66: 128xN split-K/stream-K for M=32 (M-waste but fills SMs)
    "128x128x 64 1x1x1 SK16",
    "128x256x 64 1x1x1 SK16",
    "128x256x 64 1x2x1 StK",
    "128x128x 64 1x2x1 StK",
];

// -- CUDA event timing (RAII) --

struct CuEvent(cu_sys::CUevent);

impl Drop for CuEvent {
    fn drop(&mut self) {
        unsafe { cu_sys::cuEventDestroy_v2(self.0); }
    }
}

impl CuEvent {
    fn new() -> Self {
        let mut ev: cu_sys::CUevent = std::ptr::null_mut();
        let r = unsafe { cu_sys::cuEventCreate(&mut ev, 0) };
        assert_eq!(r, cu_sys::CUresult::CUDA_SUCCESS, "cuEventCreate: {r:?}");
        Self(ev)
    }

    fn record(&self, stream: cu_sys::CUstream) {
        let r = unsafe { cu_sys::cuEventRecord(self.0, stream) };
        assert_eq!(r, cu_sys::CUresult::CUDA_SUCCESS, "cuEventRecord: {r:?}");
    }

    fn sync(&self) {
        let r = unsafe { cu_sys::cuEventSynchronize(self.0) };
        assert_eq!(r, cu_sys::CUresult::CUDA_SUCCESS, "cuEventSynchronize: {r:?}");
    }

    fn elapsed_ms(&self, start: &CuEvent) -> f32 {
        let mut ms: f32 = 0.0;
        let r = unsafe { cu_sys::cuEventElapsedTime(&mut ms, start.0, self.0) };
        assert_eq!(r, cu_sys::CUresult::CUDA_SUCCESS, "cuEventElapsedTime: {r:?}");
        ms
    }
}

// -- cuBLAS baseline handle --

struct CublasBaseline {
    handle: cublas_sys::cublasHandle_t,
}

impl Drop for CublasBaseline {
    fn drop(&mut self) {
        unsafe { cublas_sys::cublasDestroy_v2(self.handle); }
    }
}

impl CublasBaseline {
    fn new(stream: cu_sys::CUstream) -> Self {
        let mut handle: cublas_sys::cublasHandle_t = std::ptr::null_mut();
        let r = unsafe { cublas_sys::cublasCreate_v2(&mut handle) };
        assert_eq!(r, cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cublasCreate: {r:?}");
        let r = unsafe { cublas_sys::cublasSetStream_v2(handle, stream as _) };
        assert_eq!(r, cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cublasSetStream: {r:?}");
        Self { handle }
    }

    /// C[m,n] = A[m,k] @ B[n,k]^T  (row-major all)
    fn hgemm(&self, m: usize, n: usize, k: usize, a: u64, b: u64, c: u64) {
        use cublas_sys::{
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
            cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            cudaDataType_t::CUDA_R_16F,
        };
        static ALPHA: f32 = 1.0;
        static BETA: f32 = 0.0;
        let status = unsafe {
            cublas_sys::cublasGemmEx(
                self.handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n as i32, m as i32, k as i32,
                &ALPHA as *const f32 as *const c_void,
                b as *const c_void, CUDA_R_16F, k as i32,
                a as *const c_void, CUDA_R_16F, k as i32,
                &BETA as *const f32 as *const c_void,
                c as *mut c_void, CUDA_R_16F, n as i32,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        };
        assert_eq!(status, CUBLAS_STATUS_SUCCESS, "cublasGemmEx: {status:?}");
    }
}

// -- Timing helpers --

fn median(times: &mut Vec<f32>) -> f32 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times.len();
    if n == 0 { return 0.0; }
    if n % 2 == 0 {
        (times[n / 2 - 1] + times[n / 2]) / 2.0
    } else {
        times[n / 2]
    }
}

/// Time a single CUTLASS HGEMM variant. Returns median microseconds or None if variant unavailable.
fn bench_cutlass_variant(
    cutlass: &CutlassKernels,
    variant: usize,
    cu_stream: cu_sys::CUstream,
    m: usize, n: usize, k: usize,
    a_ptr: u64, b_ptr: u64, c_ptr: u64,
    ws_ptr: u64, ws_size: usize, stream_ptr: u64,
) -> Option<f32> {
    // Check variant is loaded
    let ws_needed = cutlass.hgemm_variant_workspace_size(variant, m as i32, n as i32, k as i32)?;
    if ws_needed > ws_size {
        eprintln!("  v{variant} needs {} bytes workspace, have {} -- skip", ws_needed, ws_size);
        return None;
    }

    // Probe: does the variant succeed on this shape?
    if cutlass.hgemm_variant(variant, c_ptr, a_ptr, b_ptr, m as i32, n as i32, k as i32,
                              ws_ptr, ws_size, stream_ptr).is_err() {
        return None;
    }
    let r = unsafe { cu_sys::cuStreamSynchronize(cu_stream) };
    if r != cu_sys::CUresult::CUDA_SUCCESS { return None; }

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = cutlass.hgemm_variant(variant, c_ptr, a_ptr, b_ptr, m as i32, n as i32, k as i32,
                                       ws_ptr, ws_size, stream_ptr);
    }
    let r = unsafe { cu_sys::cuStreamSynchronize(cu_stream) };
    if r != cu_sys::CUresult::CUDA_SUCCESS { return None; }

    // Timed iterations
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = CuEvent::new();
        let stop = CuEvent::new();
        start.record(cu_stream);
        let _ = cutlass.hgemm_variant(variant, c_ptr, a_ptr, b_ptr, m as i32, n as i32, k as i32,
                                       ws_ptr, ws_size, stream_ptr);
        stop.record(cu_stream);
        stop.sync();
        times.push(stop.elapsed_ms(&start) * 1000.0); // ms -> us
    }

    Some(median(&mut times))
}

/// Time cuBLAS HGEMM. Returns median microseconds.
fn bench_cublas(
    cublas: &CublasBaseline,
    cu_stream: cu_sys::CUstream,
    m: usize, n: usize, k: usize,
    a_ptr: u64, b_ptr: u64, c_ptr: u64,
) -> f32 {
    // Warmup
    for _ in 0..WARMUP_ITERS {
        cublas.hgemm(m, n, k, a_ptr, b_ptr, c_ptr);
    }
    unsafe { cu_sys::cuStreamSynchronize(cu_stream); }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = CuEvent::new();
        let stop = CuEvent::new();
        start.record(cu_stream);
        cublas.hgemm(m, n, k, a_ptr, b_ptr, c_ptr);
        stop.record(cu_stream);
        stop.sync();
        times.push(stop.elapsed_ms(&start) * 1000.0);
    }

    median(&mut times)
}

/// Time a single oproj+residual variant. Returns median microseconds or None.
fn bench_oproj_residual_variant(
    cutlass: &CutlassKernels,
    variant: usize,
    cu_stream: cu_sys::CUstream,
    m: usize, n: usize, k: usize,
    a_ptr: u64, b_ptr: u64, c_ptr: u64, r_ptr: u64,
    ws_ptr: u64, ws_size: usize, stream_ptr: u64,
) -> Option<f32> {
    let ws_needed = cutlass.oproj_residual_variant_workspace_size(variant, m as i32, n as i32, k as i32)?;
    if ws_needed > ws_size { return None; }

    if cutlass.oproj_residual_variant(variant, c_ptr, a_ptr, b_ptr, r_ptr,
        m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr).is_err() {
        return None;
    }
    let r = unsafe { cu_sys::cuStreamSynchronize(cu_stream) };
    if r != cu_sys::CUresult::CUDA_SUCCESS { return None; }

    for _ in 0..WARMUP_ITERS {
        let _ = cutlass.oproj_residual_variant(variant, c_ptr, a_ptr, b_ptr, r_ptr,
            m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr);
    }
    unsafe { cu_sys::cuStreamSynchronize(cu_stream); }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = CuEvent::new();
        let stop = CuEvent::new();
        start.record(cu_stream);
        let _ = cutlass.oproj_residual_variant(variant, c_ptr, a_ptr, b_ptr, r_ptr,
            m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr);
        stop.record(cu_stream);
        stop.sync();
        times.push(stop.elapsed_ms(&start) * 1000.0);
    }
    Some(median(&mut times))
}

/// Time a single gateup+silu variant. Returns median microseconds or None.
fn bench_gateup_silu_variant(
    cutlass: &CutlassKernels,
    variant: usize,
    cu_stream: cu_sys::CUstream,
    m: usize, n: usize, k: usize,
    a_ptr: u64, b_ptr: u64, c_ptr: u64,
    ws_ptr: u64, ws_size: usize, stream_ptr: u64,
) -> Option<f32> {
    let ws_needed = cutlass.gateup_silu_variant_workspace_size(variant, m as i32, n as i32, k as i32)?;
    if ws_needed > ws_size { return None; }

    if cutlass.gateup_silu_variant(variant, c_ptr, a_ptr, b_ptr,
        m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr).is_err() {
        return None;
    }
    let r = unsafe { cu_sys::cuStreamSynchronize(cu_stream) };
    if r != cu_sys::CUresult::CUDA_SUCCESS { return None; }

    for _ in 0..WARMUP_ITERS {
        let _ = cutlass.gateup_silu_variant(variant, c_ptr, a_ptr, b_ptr,
            m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr);
    }
    unsafe { cu_sys::cuStreamSynchronize(cu_stream); }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = CuEvent::new();
        let stop = CuEvent::new();
        start.record(cu_stream);
        let _ = cutlass.gateup_silu_variant(variant, c_ptr, a_ptr, b_ptr,
            m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr);
        stop.record(cu_stream);
        stop.sync();
        times.push(stop.elapsed_ms(&start) * 1000.0);
    }
    Some(median(&mut times))
}

/// Time a single FP8 GEMM variant. Returns median microseconds or None.
fn bench_fp8_gemm_variant(
    cutlass: &CutlassKernels,
    variant: usize,
    cu_stream: cu_sys::CUstream,
    m: usize, n: usize, k: usize,
    a_ptr: u64, b_ptr: u64, c_ptr: u64,
    as_ptr: u64, bs_ptr: u64,
    ws_ptr: u64, ws_size: usize, stream_ptr: u64,
) -> Option<f32> {
    let ws_needed = cutlass.fp8_gemm_variant_workspace_size(variant, m as i32, n as i32, k as i32)?;
    if ws_needed > ws_size {
        eprintln!("  fp8 v{variant} needs {} bytes workspace, have {} -- skip", ws_needed, ws_size);
        return None;
    }

    if cutlass.fp8_gemm_variant(variant, c_ptr, a_ptr, b_ptr, as_ptr, bs_ptr,
        m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr).is_err() {
        return None;
    }
    let r = unsafe { cu_sys::cuStreamSynchronize(cu_stream) };
    if r != cu_sys::CUresult::CUDA_SUCCESS { return None; }

    for _ in 0..WARMUP_ITERS {
        let _ = cutlass.fp8_gemm_variant(variant, c_ptr, a_ptr, b_ptr, as_ptr, bs_ptr,
            m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr);
    }
    let r = unsafe { cu_sys::cuStreamSynchronize(cu_stream) };
    if r != cu_sys::CUresult::CUDA_SUCCESS { return None; }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = CuEvent::new();
        let stop = CuEvent::new();
        start.record(cu_stream);
        let _ = cutlass.fp8_gemm_variant(variant, c_ptr, a_ptr, b_ptr, as_ptr, bs_ptr,
            m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr);
        stop.record(cu_stream);
        stop.sync();
        times.push(stop.elapsed_ms(&start) * 1000.0);
    }
    Some(median(&mut times))
}

/// Time a single FP8 GEMM + residual variant. Returns median microseconds or None.
fn bench_fp8_gemm_residual_variant(
    cutlass: &CutlassKernels,
    variant: usize,
    cu_stream: cu_sys::CUstream,
    m: usize, n: usize, k: usize,
    a_ptr: u64, b_ptr: u64, c_ptr: u64,
    as_ptr: u64, bs_ptr: u64, r_ptr: u64,
    ws_ptr: u64, ws_size: usize, stream_ptr: u64,
) -> Option<f32> {
    let ws_needed = cutlass.fp8_gemm_residual_workspace_size(variant, m as i32, n as i32, k as i32)?;
    if ws_needed > ws_size { return None; }

    if cutlass.fp8_gemm_residual(variant, c_ptr, a_ptr, b_ptr, as_ptr, bs_ptr, r_ptr,
        m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr).is_err() {
        return None;
    }
    let r = unsafe { cu_sys::cuStreamSynchronize(cu_stream) };
    if r != cu_sys::CUresult::CUDA_SUCCESS { return None; }

    for _ in 0..WARMUP_ITERS {
        let _ = cutlass.fp8_gemm_residual(variant, c_ptr, a_ptr, b_ptr, as_ptr, bs_ptr, r_ptr,
            m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr);
    }
    let r = unsafe { cu_sys::cuStreamSynchronize(cu_stream) };
    if r != cu_sys::CUresult::CUDA_SUCCESS { return None; }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = CuEvent::new();
        let stop = CuEvent::new();
        start.record(cu_stream);
        let _ = cutlass.fp8_gemm_residual(variant, c_ptr, a_ptr, b_ptr, as_ptr, bs_ptr, r_ptr,
            m as i32, n as i32, k as i32, ws_ptr, ws_size, stream_ptr);
        stop.record(cu_stream);
        stop.sync();
        times.push(stop.elapsed_ms(&start) * 1000.0);
    }
    Some(median(&mut times))
}

// -- Detailed results for the report JSON --

#[derive(serde::Serialize)]
struct DetailedReport {
    gpu: String,
    shapes: Vec<ShapeReport>,
}

#[derive(serde::Serialize)]
struct ShapeReport {
    m: usize,
    n: usize,
    k: usize,
    proj: String,
    winner: String,
    winner_us: f32,
    cublas_us: f32,
    speedup: f32,
    variants: Vec<VariantResult>,
}

#[derive(serde::Serialize)]
struct VariantResult {
    id: usize,
    name: String,
    time_us: Option<f32>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut so_path = PathBuf::from("kernels/sm_90/libcutlass_kernels.so");
    let mut out_path = CutlassAutotuneCache::cache_path();
    let mut report_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--so" if i + 1 < args.len() => { so_path = PathBuf::from(&args[i + 1]); i += 2; }
            "--out" if i + 1 < args.len() => { out_path = PathBuf::from(&args[i + 1]); i += 2; }
            "--report" if i + 1 < args.len() => { report_path = Some(PathBuf::from(&args[i + 1])); i += 2; }
            _ => { i += 1; }
        }
    }

    println!("CUTLASS Full Autotune Benchmark");
    println!("================================");
    println!("Library: {}", so_path.display());
    println!("Output:  {}", out_path.display());
    println!();

    // Init CUDA
    let ctx = CudaContext::new(0).expect("CUDA context init");
    let stream = ctx.new_stream().expect("CUDA stream");
    let cu_stream = stream.cu_stream();
    let stream_ptr = cu_stream as u64;

    // GPU name
    let gpu_name = unsafe {
        let mut name = [0u8; 256];
        let r = cu_sys::cuDeviceGetName(name.as_mut_ptr() as *mut i8, 256, 0);
        if r == cu_sys::CUresult::CUDA_SUCCESS {
            let s = std::ffi::CStr::from_ptr(name.as_ptr() as *const i8);
            s.to_string_lossy().to_string()
        } else {
            "unknown".to_string()
        }
    };
    println!("GPU: {gpu_name}");

    // Load CUTLASS
    let cutlass = CutlassKernels::load(&so_path).expect("load CUTLASS .so");
    let hgemm_loaded = cutlass.hgemm_variant_count();
    let oproj_loaded = cutlass.oproj_residual_variant_count();
    let gateup_loaded = cutlass.gateup_silu_variant_count();
    let fp8_loaded = cutlass.fp8_gemm_variant_count();
    let fp8_res_loaded = cutlass.fp8_gemm_residual_variant_count();
    println!("Loaded {hgemm_loaded}/{HGEMM_VARIANTS} HGEMM variants");
    println!("Loaded {oproj_loaded}/{OPROJ_RESIDUAL_VARIANTS} oproj+residual variants");
    println!("Loaded {gateup_loaded}/{GATEUP_SILU_VARIANTS} gateup+SiLU variants");
    println!("Loaded {fp8_loaded}/{FP8_GEMM_VARIANTS} FP8 GEMM variants");
    println!("Loaded {fp8_res_loaded}/{FP8_GEMM_RESIDUAL_VARIANTS} FP8 GEMM+residual variants");
    if hgemm_loaded == 0 {
        eprintln!("ERROR: No HGEMM autotune variants in .so -- build cutlass_hgemm_autotune.cu first");
        std::process::exit(1);
    }

    // cuBLAS baseline
    let cublas = CublasBaseline::new(cu_stream);

    // Pre-allocate GPU buffers for largest shapes (including fused family shapes)
    let max_m = *M_VALUES.iter().max().unwrap();
    let all_shapes: Vec<(usize, usize)> = SHAPES.iter().map(|&(n, k, _)| (n, k))
        .chain(OPROJ_SHAPES.iter().map(|&(n, k, _)| (n, k)))
        .chain(GATEUP_SHAPES.iter().map(|&(n, k, _)| (n, k)))
        .chain(FP8_SHAPES.iter().map(|&(n, k, _)| (n, k)))
        .chain(FP8_RESIDUAL_SHAPES.iter().map(|&(n, k, _)| (n, k)))
        .collect();
    let max_a = all_shapes.iter().map(|&(_, k)| max_m * k).max().unwrap();
    let max_b = all_shapes.iter().map(|&(n, k)| n * k).max().unwrap();
    let max_c = all_shapes.iter().map(|&(n, _)| max_m * n).max().unwrap();

    // Compute max workspace needed (stream-K/split-K variants need more)
    let mut dyn_ws_bytes = WORKSPACE_BYTES;
    for &(n, k, _) in FP8_SHAPES {
        for &m in M_VALUES {
            for v in 0..FP8_GEMM_VARIANTS {
                if let Some(ws) = cutlass.fp8_gemm_variant_workspace_size(v, m as i32, n as i32, k as i32) {
                    dyn_ws_bytes = dyn_ws_bytes.max(ws);
                }
            }
        }
    }
    for &(n, k, _) in FP8_RESIDUAL_SHAPES {
        for &m in M_VALUES {
            for v in 0..FP8_GEMM_RESIDUAL_VARIANTS {
                if let Some(ws) = cutlass.fp8_gemm_residual_workspace_size(v, m as i32, n as i32, k as i32) {
                    dyn_ws_bytes = dyn_ws_bytes.max(ws);
                }
            }
        }
    }
    let ws_bytes = dyn_ws_bytes;

    let a_buf = stream.alloc_zeros::<f16>(max_a).expect("alloc A");
    let b_buf = stream.alloc_zeros::<f16>(max_b).expect("alloc B");
    let mut c_buf = stream.alloc_zeros::<f16>(max_c).expect("alloc C");
    let mut ws_buf = stream.alloc_zeros::<u8>(ws_bytes).expect("alloc workspace");
    // Extra residual buffer for oproj+residual (same size as output)
    let r_buf = stream.alloc_zeros::<f16>(max_c).expect("alloc residual");

    // FP8 buffers: u8 for FP8 data, f32 for scales
    let fp8_a_buf = stream.alloc_zeros::<u8>(max_a).expect("alloc FP8 A");
    let fp8_b_buf = stream.alloc_zeros::<u8>(max_b).expect("alloc FP8 B");
    let fp8_as_buf = stream.alloc_zeros::<f32>(max_m).expect("alloc FP8 A scales");
    let fp8_bs_buf = stream.alloc_zeros::<f32>(1).expect("alloc FP8 B scale");

    let (a_ptr, _ag) = DevicePtr::device_ptr(&a_buf, &stream);
    let (b_ptr, _bg) = DevicePtr::device_ptr(&b_buf, &stream);
    let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(&mut c_buf, &stream);
    let (ws_ptr, _wg) = DevicePtrMut::device_ptr_mut(&mut ws_buf, &stream);
    let (r_ptr, _rg) = DevicePtr::device_ptr(&r_buf, &stream);
    let (fp8_a_ptr, _fp8ag) = DevicePtr::device_ptr(&fp8_a_buf, &stream);
    let (fp8_b_ptr, _fp8bg) = DevicePtr::device_ptr(&fp8_b_buf, &stream);
    let (fp8_as_ptr, _fp8asg) = DevicePtr::device_ptr(&fp8_as_buf, &stream);
    let (fp8_bs_ptr, _fp8bsg) = DevicePtr::device_ptr(&fp8_bs_buf, &stream);

    let a_ptr = a_ptr as u64;
    let b_ptr = b_ptr as u64;
    let c_ptr = c_ptr as u64;
    let ws_ptr = ws_ptr as u64;
    let r_ptr = r_ptr as u64;
    let fp8_a_ptr = fp8_a_ptr as u64;
    let fp8_b_ptr = fp8_b_ptr as u64;
    let fp8_as_ptr = fp8_as_ptr as u64;
    let fp8_bs_ptr = fp8_bs_ptr as u64;

    // Benchmark all shapes
    let mut cache = CutlassAutotuneCache::default();
    let mut shape_reports = Vec::new();
    let mut total = 0usize;
    let mut cutlass_wins = 0usize;
    let mut cublas_wins = 0usize;

    println!();
    println!("Benchmarking ({WARMUP_ITERS} warmup, {BENCH_ITERS} timed, median)");
    println!("{}", "=".repeat(72));

    for &(n, k, proj) in SHAPES {
        println!();
        println!("--- {proj} (N={n}, K={k}) ---");

        for &m in M_VALUES {
            println!();
            println!("Shape M={m} N={n} K={k}:");

            let mut variant_results = Vec::new();
            let mut best_cutlass: Option<(usize, f32)> = None;

            for v in 0..HGEMM_VARIANTS {
                match bench_cutlass_variant(
                    &cutlass, v, cu_stream, m, n, k,
                    a_ptr, b_ptr, c_ptr, ws_ptr, ws_bytes, stream_ptr,
                ) {
                    Some(us) => {
                        println!("  v{v} ({name}): {us:8.1} us", name = VARIANT_NAMES[v]);
                        variant_results.push(VariantResult { id: v, name: VARIANT_NAMES[v].to_string(), time_us: Some(us) });
                        match best_cutlass {
                            None => best_cutlass = Some((v, us)),
                            Some((_, bt)) if us < bt => best_cutlass = Some((v, us)),
                            _ => {}
                        }
                    }
                    None => {
                        println!("  v{v} ({name}):   --skip--", name = VARIANT_NAMES[v]);
                        variant_results.push(VariantResult { id: v, name: VARIANT_NAMES[v].to_string(), time_us: None });
                    }
                }
            }

            let cublas_us = bench_cublas(&cublas, cu_stream, m, n, k, a_ptr, b_ptr, c_ptr);
            println!("  cuBLAS:                      {cublas_us:8.1} us");

            total += 1;

            let (winner_str, winner_us, speedup) = match best_cutlass {
                Some((v, us)) if us < cublas_us => {
                    let spd = cublas_us / us;
                    cache.insert_hgemm(m, n, k, v);
                    cutlass_wins += 1;
                    let pct = (1.0 - us / cublas_us) * 100.0;
                    println!("  BEST: CUTLASS v{v} ({us:.1} us, {pct:.1}% faster than cuBLAS)");
                    (format!("v{v}"), us, spd)
                }
                Some((_, us)) => {
                    cublas_wins += 1;
                    println!("  BEST: cuBLAS wins ({cublas_us:.1} us vs CUTLASS {us:.1} us) -- not stored");
                    ("cublas".to_string(), cublas_us, 1.0)
                }
                None => {
                    cublas_wins += 1;
                    println!("  NO CUTLASS VARIANT WORKED for M={m} N={n} K={k}");
                    ("none".to_string(), cublas_us, 1.0)
                }
            };

            shape_reports.push(ShapeReport {
                m, n, k,
                proj: proj.to_string(),
                winner: winner_str,
                winner_us,
                cublas_us,
                speedup,
                variants: variant_results,
            });
        }
    }

    // ===================================================================
    // Fused oproj+residual variants (output = input @ weight + residual)
    // ===================================================================
    if oproj_loaded > 0 {
        println!();
        println!("Benchmarking oproj+residual ({oproj_loaded} variants)");
        println!("{}", "=".repeat(72));

        for &(n, k, proj) in OPROJ_SHAPES {
            println!();
            println!("--- {proj} (N={n}, K={k}) ---");

            for &m in M_VALUES {
                println!();
                println!("Shape M={m} N={n} K={k}:");

                let mut best: Option<(usize, f32)> = None;
                for v in 0..OPROJ_RESIDUAL_VARIANTS {
                    match bench_oproj_residual_variant(
                        &cutlass, v, cu_stream, m, n, k,
                        a_ptr, b_ptr, c_ptr, r_ptr, ws_ptr, ws_bytes, stream_ptr,
                    ) {
                        Some(us) => {
                            println!("  v{v}: {us:8.1} us");
                            match best {
                                None => best = Some((v, us)),
                                Some((_, bt)) if us < bt => best = Some((v, us)),
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }

                // cuBLAS baseline (plain GEMM, no residual add)
                let cublas_us = bench_cublas(&cublas, cu_stream, m, n, k, a_ptr, b_ptr, c_ptr);
                println!("  cuBLAS (GEMM only, no residual): {cublas_us:8.1} us");

                total += 1;
                match best {
                    Some((v, us)) if us < cublas_us => {
                        cache.insert_oproj_residual(m, n, k, v);
                        cutlass_wins += 1;
                        let speedup = (1.0 - us / cublas_us) * 100.0;
                        println!("  BEST: CUTLASS v{v} ({us:.1} us, {speedup:.1}% faster than cuBLAS)");
                    }
                    Some((_, us)) => {
                        cublas_wins += 1;
                        println!("  BEST: cuBLAS wins ({cublas_us:.1} us vs CUTLASS {us:.1} us) -- not stored");
                    }
                    None => {
                        cublas_wins += 1;
                        println!("  NO OPROJ VARIANT WORKED for M={m} N={n} K={k}");
                    }
                }
            }
        }
    }

    // ===================================================================
    // Fused gateup+silu variants (output = SiLU(input @ weight))
    // ===================================================================
    if gateup_loaded > 0 {
        println!();
        println!("Benchmarking gateup+SiLU ({gateup_loaded} variants)");
        println!("{}", "=".repeat(72));

        for &(n, k, proj) in GATEUP_SHAPES {
            println!();
            println!("--- {proj} (N={n}, K={k}) ---");

            for &m in M_VALUES {
                println!();
                println!("Shape M={m} N={n} K={k}:");

                let mut best: Option<(usize, f32)> = None;
                for v in 0..GATEUP_SILU_VARIANTS {
                    match bench_gateup_silu_variant(
                        &cutlass, v, cu_stream, m, n, k,
                        a_ptr, b_ptr, c_ptr, ws_ptr, ws_bytes, stream_ptr,
                    ) {
                        Some(us) => {
                            println!("  v{v}: {us:8.1} us");
                            match best {
                                None => best = Some((v, us)),
                                Some((_, bt)) if us < bt => best = Some((v, us)),
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }

                // cuBLAS baseline (GEMM only, no SiLU -- unfair comparison but logged for reference)
                let cublas_us = bench_cublas(&cublas, cu_stream, m, n, k, a_ptr, b_ptr, c_ptr);
                println!("  cuBLAS (GEMM only, no SiLU): {cublas_us:8.1} us");

                total += 1;
                match best {
                    Some((v, us)) => {
                        cache.insert_gateup_silu(m, n, k, v);
                        cutlass_wins += 1;
                        println!("  BEST: v{v} ({us:.1} us) -- fused GEMM+SiLU, always stored");
                    }
                    None => {
                        cublas_wins += 1;
                        println!("  NO GATEUP VARIANT WORKED for M={m} N={n} K={k}");
                    }
                }
            }
        }
    }

    // ===================================================================
    // FP8 GEMM variants (FP8 E4M3 with fused per-row/per-tensor scaling)
    // ===================================================================
    if fp8_loaded > 0 {
        println!();
        println!("Benchmarking FP8 GEMM ({fp8_loaded} variants)");
        println!("{}", "=".repeat(72));

        for &(n, k, proj) in FP8_SHAPES {
            println!();
            println!("--- FP8 {proj} (N={n}, K={k}) ---");

            for &m in M_VALUES {
                println!();
                println!("FP8 Shape M={m} N={n} K={k}:");

                let mut best: Option<(usize, f32)> = None;
                for v in 0..FP8_GEMM_VARIANTS {
                    match bench_fp8_gemm_variant(
                        &cutlass, v, cu_stream, m, n, k,
                        fp8_a_ptr, fp8_b_ptr, c_ptr,
                        fp8_as_ptr, fp8_bs_ptr,
                        ws_ptr, ws_bytes, stream_ptr,
                    ) {
                        Some(us) => {
                            println!("  fp8_v{v}: {us:8.1} us");
                            match best {
                                None => best = Some((v, us)),
                                Some((_, bt)) if us < bt => best = Some((v, us)),
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }

                // cuBLAS F16 baseline for comparison
                let cublas_us = Some(bench_cublas(&cublas, cu_stream, m, n, k, a_ptr, b_ptr, c_ptr));

                total += 1;
                match best {
                    Some((v, us)) => {
                        let speedup = cublas_us.map(|c| c / us).unwrap_or(0.0);
                        cache.insert_fp8_gemm(m, n, k, v);
                        cutlass_wins += 1;
                        println!("  BEST: fp8_v{v} ({us:.1} us) vs cuBLAS F16 ({:.1} us) = {speedup:.2}x",
                                 cublas_us.unwrap_or(0.0));
                    }
                    None => {
                        cublas_wins += 1;
                        println!("  NO FP8 VARIANT WORKED for M={m} N={n} K={k} (cuBLAS F16: {:.1} us)",
                                 cublas_us.unwrap_or(0.0));
                    }
                }
            }
        }
    }

    // ===================================================================
    // FP8 GEMM + residual variants (output = FP8 GEMM + residual add)
    // ===================================================================
    if fp8_res_loaded > 0 {
        println!();
        println!("Benchmarking FP8 GEMM + residual ({fp8_res_loaded} variants)");
        println!("{}", "=".repeat(72));

        for &(n, k, proj) in FP8_RESIDUAL_SHAPES {
            println!();
            println!("--- FP8+residual {proj} (N={n}, K={k}) ---");

            for &m in M_VALUES {
                println!();
                println!("FP8+residual Shape M={m} N={n} K={k}:");

                let mut best: Option<(usize, f32)> = None;
                for v in 0..FP8_GEMM_RESIDUAL_VARIANTS {
                    match bench_fp8_gemm_residual_variant(
                        &cutlass, v, cu_stream, m, n, k,
                        fp8_a_ptr, fp8_b_ptr, c_ptr,
                        fp8_as_ptr, fp8_bs_ptr, r_ptr,
                        ws_ptr, ws_bytes, stream_ptr,
                    ) {
                        Some(us) => {
                            println!("  fp8res_v{v}: {us:8.1} us");
                            match best {
                                None => best = Some((v, us)),
                                Some((_, bt)) if us < bt => best = Some((v, us)),
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }

                // Compare against best non-residual FP8 GEMM (the alternative path)
                let fp8_baseline = {
                    let mut best_fp8: Option<f32> = None;
                    for v in 0..FP8_GEMM_VARIANTS {
                        if let Some(us) = bench_fp8_gemm_variant(
                            &cutlass, v, cu_stream, m, n, k,
                            fp8_a_ptr, fp8_b_ptr, c_ptr,
                            fp8_as_ptr, fp8_bs_ptr,
                            ws_ptr, ws_bytes, stream_ptr,
                        ) {
                            match best_fp8 {
                                None => best_fp8 = Some(us),
                                Some(bt) if us < bt => best_fp8 = Some(us),
                                _ => {}
                            }
                        }
                    }
                    best_fp8
                };
                if let Some(fp8_us) = fp8_baseline {
                    println!("  FP8 GEMM only (no residual): {fp8_us:8.1} us");
                }

                total += 1;
                match best {
                    Some((v, us)) => {
                        cache.insert_fp8_gemm_residual(m, n, k, v);
                        cutlass_wins += 1;
                        let note = if let Some(fp8_us) = fp8_baseline {
                            format!(" (vs FP8-only {fp8_us:.1} us = {:.1}% overhead for free residual)", (us / fp8_us - 1.0) * 100.0)
                        } else { String::new() };
                        println!("  BEST: fp8res_v{v} ({us:.1} us){note}");
                    }
                    None => {
                        cublas_wins += 1;
                        println!("  NO FP8+RESIDUAL VARIANT WORKED for M={m} N={n} K={k}");
                    }
                }
            }
        }
    }

    // Summary
    println!();
    println!("{}", "=".repeat(72));
    println!("Summary: {total} shapes benchmarked");
    println!("  CUTLASS wins: {cutlass_wins}");
    println!("  cuBLAS  wins: {cublas_wins}");

    // Save runtime cache (CutlassAutotuneCache format, readable by layer.rs)
    cache.save(&out_path).expect("save autotune cache");
    println!();
    println!("Runtime cache saved to: {}", out_path.display());
    println!("  hgemm entries: {}", cache.hgemm.len());
    println!("  oproj_residual entries: {}", cache.oproj_residual.len());
    println!("  gateup_silu entries: {}", cache.gateup_silu.len());
    println!("  fp8_gemm entries: {}", cache.fp8_gemm.len());
    println!("  fp8_gemm_residual entries: {}", cache.fp8_gemm_residual.len());

    // Compact table
    println!();
    println!("{:<6} {:<8} {:<8} {:<10} {:<10} {:<10} {:<8}",
             "M", "N", "K", "Winner", "Time(us)", "cuBLAS", "Speedup");
    println!("{}", "-".repeat(68));

    for sr in &shape_reports {
        println!("{:<6} {:<8} {:<8} {:<10} {:<10.1} {:<10.1} {:<.2}x",
                 sr.m, sr.n, sr.k, sr.winner, sr.winner_us, sr.cublas_us, sr.speedup);
    }

    // Save detailed report if requested
    if let Some(rp) = report_path {
        let report = DetailedReport { gpu: gpu_name, shapes: shape_reports };
        let json = serde_json::to_string_pretty(&report).expect("serialize report");
        if let Some(parent) = rp.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&rp, json).expect("write report");
        println!();
        println!("Detailed report saved to: {}", rp.display());
    }
}
