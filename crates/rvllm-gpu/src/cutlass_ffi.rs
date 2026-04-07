//! FFI bindings to CUTLASS shared library (libcutlass_kernels.so).
//!
//! CUTLASS kernels are self-launching -- they compute their own grid dimensions
//! internally via the CUTLASS device adapter. You cannot launch them as raw PTX
//! via cuLaunchKernel. Instead, compile to .so and call the C wrapper functions
//! from host code (same pattern as vLLM).

use std::ffi::c_void;
use std::path::Path;

/// Handle to the loaded CUTLASS shared library.
/// Holds function pointers resolved at load time for zero-cost dispatch.
pub struct CutlassKernels {
    _lib: libloading::Library,
    // Resolved function pointers (cached at load time, not per-call dlsym)
    fn_qkv_bias: QkvBiasFn,
    fn_qkv_bias_ws: WorkspaceSizeFn,
    fn_oproj_residual: OprojResidualFn,
    fn_oproj_residual_ws: WorkspaceSizeFn,
    fn_gateup_silu: GateUpSiluFn,
    fn_gateup_silu_ws: WorkspaceSizeFn,
    fn_gate_silu_mul: GateSiluMulFn,
    fn_gate_silu_mul_ws: WorkspaceSizeFn,
    fn_hgemm: HgemmFn,
    fn_hgemm_ws: WorkspaceSizeFn,
}

// Function pointer types matching the extern "C" signatures in the .cu files.
type QkvBiasFn = unsafe extern "C" fn(
    output: *mut c_void,
    input: *const c_void,
    weight: *const c_void,
    bias: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void, // cudaStream_t
) -> i32;

type OprojResidualFn = unsafe extern "C" fn(
    output: *mut c_void,
    input: *const c_void,
    weight: *const c_void,
    residual: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

type GateUpSiluFn = unsafe extern "C" fn(
    output: *mut c_void,
    input: *const c_void,
    weight: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

type GateSiluMulFn = unsafe extern "C" fn(
    output: *mut c_void,
    input: *const c_void,
    gate_weight: *const c_void,
    aux_up: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

type HgemmFn = unsafe extern "C" fn(
    output: *mut c_void,
    input: *const c_void,
    weight: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

type WorkspaceSizeFn = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

unsafe impl Send for CutlassKernels {}
unsafe impl Sync for CutlassKernels {}

impl CutlassKernels {
    /// Load the CUTLASS shared library and resolve all function pointers.
    pub fn load(lib_path: &Path) -> Result<Self, String> {
        let lib = unsafe { libloading::Library::new(lib_path) }
            .map_err(|e| format!("dlopen {}: {e}", lib_path.display()))?;

        unsafe {
            let fn_qkv_bias: QkvBiasFn = *lib
                .get(b"cutlass_qkv_bias_gemm\0")
                .map_err(|e| format!("cutlass_qkv_bias_gemm: {e}"))?;
            let fn_qkv_bias_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_qkv_bias_workspace_size\0")
                .map_err(|e| format!("cutlass_qkv_bias_workspace_size: {e}"))?;
            let fn_oproj_residual: OprojResidualFn = *lib
                .get(b"cutlass_oproj_residual_gemm\0")
                .map_err(|e| format!("cutlass_oproj_residual_gemm: {e}"))?;
            let fn_oproj_residual_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_oproj_residual_workspace_size\0")
                .map_err(|e| format!("cutlass_oproj_residual_workspace_size: {e}"))?;
            let fn_gateup_silu: GateUpSiluFn = *lib
                .get(b"cutlass_gateup_silu\0")
                .map_err(|e| format!("cutlass_gateup_silu: {e}"))?;
            let fn_gateup_silu_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_gateup_silu_workspace_size\0")
                .map_err(|e| format!("cutlass_gateup_silu_workspace_size: {e}"))?;
            let fn_gate_silu_mul: GateSiluMulFn = *lib
                .get(b"cutlass_gate_silu_mul\0")
                .map_err(|e| format!("cutlass_gate_silu_mul: {e}"))?;
            let fn_gate_silu_mul_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_gate_silu_mul_workspace_size\0")
                .map_err(|e| format!("cutlass_gate_silu_mul_workspace_size: {e}"))?;
            let fn_hgemm: HgemmFn = *lib
                .get(b"cutlass_hgemm\0")
                .map_err(|e| format!("cutlass_hgemm: {e}"))?;
            let fn_hgemm_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_hgemm_workspace_size\0")
                .map_err(|e| format!("cutlass_hgemm_workspace_size: {e}"))?;

            Ok(Self {
                _lib: lib,
                fn_qkv_bias,
                fn_qkv_bias_ws,
                fn_oproj_residual,
                fn_oproj_residual_ws,
                fn_gateup_silu,
                fn_gateup_silu_ws,
                fn_gate_silu_mul,
                fn_gate_silu_mul_ws,
                fn_hgemm,
                fn_hgemm_ws,
            })
        }
    }

    /// QKV projection with fused bias add.
    /// All pointers are raw device pointers (u64 from cudarc DevicePtr).
    pub fn qkv_bias_gemm(
        &self,
        output: u64,
        input: u64,
        weight: u64,
        bias: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_qkv_bias)(
                output as *mut c_void,
                input as *const c_void,
                weight as *const c_void,
                bias as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_qkv_bias_gemm failed: {status}"));
        }
        Ok(())
    }

    /// Query workspace size for QKV+bias GEMM.
    pub fn qkv_bias_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_qkv_bias_ws)(m, n, k) }
    }

    /// O-projection GEMM with fused residual add.
    pub fn oproj_residual_gemm(
        &self,
        output: u64,
        input: u64,
        weight: u64,
        residual: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_oproj_residual)(
                output as *mut c_void,
                input as *const c_void,
                weight as *const c_void,
                residual as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_oproj_residual_gemm failed: {status}"));
        }
        Ok(())
    }

    /// Query workspace size for O-proj+residual GEMM.
    pub fn oproj_residual_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_oproj_residual_ws)(m, n, k) }
    }

    /// Gate+Up projection GEMM with fused SiLU*Mul activation.
    /// N is the full gate+up width (2 * intermediate_size).
    /// Output is [M, N/2] after SiLU activation.
    pub fn gateup_silu(
        &self,
        output: u64,
        input: u64,
        weight: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_gateup_silu)(
                output as *mut c_void,
                input as *const c_void,
                weight as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_gateup_silu failed: {status}"));
        }
        Ok(())
    }

    /// Query workspace size for GateUp+SiLU.
    pub fn gateup_silu_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_gateup_silu_ws)(m, n, k) }
    }

    /// Gate-only GEMM with Sm90 epilogue: SiLU(gate) * aux_up.
    pub fn gate_silu_mul(
        &self,
        output: u64,
        input: u64,
        gate_weight: u64,
        aux_up: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_gate_silu_mul)(
                output as *mut c_void,
                input as *const c_void,
                gate_weight as *const c_void,
                aux_up as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_gate_silu_mul failed: {status}"));
        }
        Ok(())
    }

    pub fn gate_silu_mul_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_gate_silu_mul_ws)(m, n, k) }
    }

    /// Plain half-precision GEMM (no epilogue fusion).
    pub fn hgemm(
        &self,
        output: u64,
        input: u64,
        weight: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_hgemm)(
                output as *mut c_void,
                input as *const c_void,
                weight as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_hgemm failed: {status}"));
        }
        Ok(())
    }

    /// Query workspace size for plain HGEMM.
    pub fn hgemm_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_hgemm_ws)(m, n, k) }
    }
}
