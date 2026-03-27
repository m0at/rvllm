//! CUDA dispatch for activation kernels: SiLU, GELU, fused SiLU*mul.
//!
//! Launches the element-wise kernels defined in `kernels/activation.cu` via cudarc.
//! All three kernels use the same launch config:
//!   Grid:  (ceil(n / 256), 1, 1)
//!   Block: (256, 1, 1)
//!   Shared memory: 0
//!
//! Gated behind `#[cfg(feature = "cuda")]` -- mock-gpu builds never see the inner types.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaDevice, CudaFunction, CudaSlice, CudaStream, LaunchAsync, LaunchConfig,
    };
    use tracing::trace;

    use rvllm_core::prelude::{LLMError, Result};

    const BLOCK_SIZE: u32 = 256;
    const MODULE_NAME: &str = "activation";

    fn launch_cfg(n: u32) -> LaunchConfig {
        let grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// CUDA SiLU (Swish) activation: x / (1 + exp(-x)).
    ///
    /// Holds a preloaded kernel handle to avoid repeated PTX lookups.
    pub struct CudaSiLU {
        device: Arc<CudaDevice>,
        func: CudaFunction,
    }

    impl CudaSiLU {
        /// Load the activation PTX and extract `silu_kernel`.
        ///
        /// `ptx_bytes` should be the compiled PTX of `kernels/activation.cu`.
        pub fn new(device: Arc<CudaDevice>, ptx_bytes: &[u8]) -> Result<Self> {
            load_module_if_needed(&device, ptx_bytes)?;
            let func = device
                .get_func(MODULE_NAME, "silu_kernel")
                .ok_or_else(|| {
                    LLMError::GpuError("silu_kernel not found in activation module".into())
                })?;
            trace!("CudaSiLU: loaded silu_kernel");
            Ok(Self { device, func })
        }

        /// Apply SiLU element-wise, returning a new device buffer.
        pub fn forward(
            &self,
            input: &CudaSlice<f32>,
            stream: &CudaStream,
        ) -> Result<CudaSlice<f32>> {
            let n = input.len();
            let output = self
                .device
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("SiLU alloc failed: {e}")))?;
            let cfg = launch_cfg(n as u32);
            // SAFETY: kernel reads `n` f32 from `input`, writes `n` f32 to `output`.
            // Both device slices have length >= n. The i32 `n` matches the kernel
            // signature `int n`.
            unsafe {
                self.func
                    .clone()
                    .launch_on_stream(stream, cfg, (&output, input, n as i32))
                    .map_err(|e| LLMError::GpuError(format!("silu_kernel launch failed: {e}")))?;
            }
            trace!(n, "silu_kernel launched");
            Ok(output)
        }

        /// Apply SiLU in-place (output overwrites input).
        pub fn forward_inplace(
            &self,
            data: &mut CudaSlice<f32>,
            stream: &CudaStream,
        ) -> Result<()> {
            let n = data.len();
            let cfg = launch_cfg(n as u32);
            // SAFETY: The silu_kernel reads input[idx] then writes output[idx] per thread.
            // Each thread handles exactly one independent element, so aliasing output==input
            // is safe -- the read completes before the write within a single thread.
            unsafe {
                self.func
                    .clone()
                    .launch_on_stream(stream, cfg, (data, &*data, n as i32))
                    .map_err(|e| {
                        LLMError::GpuError(format!("silu_kernel inplace launch failed: {e}"))
                    })?;
            }
            trace!(n, "silu_kernel launched (inplace)");
            Ok(())
        }

        pub fn device(&self) -> &Arc<CudaDevice> {
            &self.device
        }
    }

    /// CUDA GELU activation (tanh approximation).
    pub struct CudaGELU {
        device: Arc<CudaDevice>,
        func: CudaFunction,
    }

    impl CudaGELU {
        /// Load the activation PTX and extract `gelu_kernel`.
        pub fn new(device: Arc<CudaDevice>, ptx_bytes: &[u8]) -> Result<Self> {
            load_module_if_needed(&device, ptx_bytes)?;
            let func = device
                .get_func(MODULE_NAME, "gelu_kernel")
                .ok_or_else(|| {
                    LLMError::GpuError("gelu_kernel not found in activation module".into())
                })?;
            trace!("CudaGELU: loaded gelu_kernel");
            Ok(Self { device, func })
        }

        /// Apply GELU element-wise, returning a new device buffer.
        pub fn forward(
            &self,
            input: &CudaSlice<f32>,
            stream: &CudaStream,
        ) -> Result<CudaSlice<f32>> {
            let n = input.len();
            let output = self
                .device
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("GELU alloc failed: {e}")))?;
            let cfg = launch_cfg(n as u32);
            // SAFETY: kernel reads `n` f32 from `input`, writes `n` f32 to `output`.
            // Both slices are device-allocated with length >= n.
            unsafe {
                self.func
                    .clone()
                    .launch_on_stream(stream, cfg, (&output, input, n as i32))
                    .map_err(|e| LLMError::GpuError(format!("gelu_kernel launch failed: {e}")))?;
            }
            trace!(n, "gelu_kernel launched");
            Ok(output)
        }

        /// Apply GELU in-place.
        pub fn forward_inplace(
            &self,
            data: &mut CudaSlice<f32>,
            stream: &CudaStream,
        ) -> Result<()> {
            let n = data.len();
            let cfg = launch_cfg(n as u32);
            // SAFETY: pure element-wise -- same aliasing rationale as CudaSiLU::forward_inplace.
            unsafe {
                self.func
                    .clone()
                    .launch_on_stream(stream, cfg, (data, &*data, n as i32))
                    .map_err(|e| {
                        LLMError::GpuError(format!("gelu_kernel inplace launch failed: {e}"))
                    })?;
            }
            trace!(n, "gelu_kernel launched (inplace)");
            Ok(())
        }

        pub fn device(&self) -> &Arc<CudaDevice> {
            &self.device
        }
    }

    /// Fused SiLU(gate) * up on GPU -- single kernel, saves a full memory traversal
    /// and a temporary buffer compared to separate SiLU + element-wise multiply.
    pub struct CudaFusedSiLUMul {
        device: Arc<CudaDevice>,
        func: CudaFunction,
    }

    impl CudaFusedSiLUMul {
        /// Load the activation PTX and extract `fused_silu_mul_kernel`.
        pub fn new(device: Arc<CudaDevice>, ptx_bytes: &[u8]) -> Result<Self> {
            load_module_if_needed(&device, ptx_bytes)?;
            let func = device
                .get_func(MODULE_NAME, "fused_silu_mul_kernel")
                .ok_or_else(|| {
                    LLMError::GpuError(
                        "fused_silu_mul_kernel not found in activation module".into(),
                    )
                })?;
            trace!("CudaFusedSiLUMul: loaded fused_silu_mul_kernel");
            Ok(Self { device, func })
        }

        /// Compute silu(gate) * up element-wise, returning a new device buffer.
        ///
        /// `gate` and `up` must have the same length.
        pub fn forward(
            &self,
            gate: &CudaSlice<f32>,
            up: &CudaSlice<f32>,
            stream: &CudaStream,
        ) -> Result<CudaSlice<f32>> {
            let n = gate.len();
            if up.len() != n {
                return Err(LLMError::GpuError(format!(
                    "fused_silu_mul: gate len {} != up len {}",
                    n,
                    up.len()
                )));
            }
            let output = self
                .device
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul alloc failed: {e}")))?;
            let cfg = launch_cfg(n as u32);
            // SAFETY: kernel reads `n` elements each from `gate` and `up`, writes `n`
            // elements to `output`. All three slices are device-allocated with length >= n.
            unsafe {
                self.func
                    .clone()
                    .launch_on_stream(stream, cfg, (&output, gate, up, n as i32))
                    .map_err(|e| {
                        LLMError::GpuError(format!("fused_silu_mul_kernel launch failed: {e}"))
                    })?;
            }
            trace!(n, "fused_silu_mul_kernel launched");
            Ok(output)
        }

        /// Compute silu(gate) * up, writing the result into `gate` in-place.
        ///
        /// `gate` and `up` must have the same length.
        pub fn forward_inplace(
            &self,
            gate: &mut CudaSlice<f32>,
            up: &CudaSlice<f32>,
            stream: &CudaStream,
        ) -> Result<()> {
            let n = gate.len();
            if up.len() != n {
                return Err(LLMError::GpuError(format!(
                    "fused_silu_mul inplace: gate len {} != up len {}",
                    n,
                    up.len()
                )));
            }
            let cfg = launch_cfg(n as u32);
            // SAFETY: output ptr aliases gate. The kernel reads gate[idx] then writes
            // output[idx] for the same idx in a single thread -- no cross-element deps,
            // so in-place aliasing is safe for this element-wise operation.
            unsafe {
                self.func
                    .clone()
                    .launch_on_stream(stream, cfg, (gate, &*gate, up, n as i32))
                    .map_err(|e| {
                        LLMError::GpuError(format!(
                            "fused_silu_mul_kernel inplace launch failed: {e}"
                        ))
                    })?;
            }
            trace!(n, "fused_silu_mul_kernel launched (inplace)");
            Ok(())
        }

        pub fn device(&self) -> &Arc<CudaDevice> {
            &self.device
        }
    }

    /// Load the activation PTX module into the device if not already present.
    ///
    /// All three activation structs share the same module, so only the first
    /// call actually loads; subsequent calls are no-ops.
    fn load_module_if_needed(device: &Arc<CudaDevice>, ptx_bytes: &[u8]) -> Result<()> {
        if device.has_func(MODULE_NAME, "silu_kernel") {
            return Ok(());
        }
        let ptx_str = std::str::from_utf8(ptx_bytes).map_err(|e| {
            LLMError::GpuError(format!("activation PTX is not valid UTF-8: {e}"))
        })?;
        device
            .load_ptx(
                cudarc::nvrtc::Ptx::from_src(ptx_str),
                MODULE_NAME,
                &["silu_kernel", "gelu_kernel", "fused_silu_mul_kernel"],
            )
            .map_err(|e| LLMError::GpuError(format!("failed to load activation PTX: {e}")))?;
        trace!("activation module loaded with 3 kernels");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use inner::{CudaFusedSiLUMul, CudaGELU, CudaSiLU};

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        // Compile-only sanity check under default (mock-gpu) features.
    }
}
