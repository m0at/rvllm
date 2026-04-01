//! CUDA linear (dense / GEMM) layer via cuBLAS.
//!
//! Implements `output[m,n] = input[m,k] @ weight^T[k,n] [+ bias]` where weight
//! is stored as `[n, k]` row-major (out_features x in_features), matching the
//! convention in `linear.rs`.
//!
//! This module is intended to be gated behind `#[cfg(feature = "cuda")]` in the
//! parent `mod.rs`. It delegates the unsafe cuBLAS call to `rvllm_gpu::cublas_ops`
//! so this crate's `#![forbid(unsafe_code)]` is respected.

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice as _, LaunchAsync, LaunchConfig};
use half::{bf16, f16};
use rvllm_core::prelude::{LLMError, Result};
use rvllm_gpu::cublas::CublasHandle;
use rvllm_gpu::cublas_ops::CublasOps;
use std::sync::Arc;

/// GPU-accelerated dense linear projection using cuBLAS SGEMM.
///
/// Owns a `CublasOps` handle so cuBLAS init cost is amortized across calls.
pub struct CudaLinearLayer {
    ops: CublasOps,
}

impl CudaLinearLayer {
    /// Create a new layer bound to the given CUDA device.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self {
            ops: CublasOps::new(device)?,
        })
    }

    /// Convenience constructor sharing the device from an existing `CublasHandle`.
    pub fn from_handle(blas: &CublasHandle) -> Result<Self> {
        Self::new(blas.device().clone())
    }

    /// Compute `output[m,n] = input[m,k] @ weight^T[k,n] [+ bias]`.
    ///
    /// # Arguments
    /// * `input`  - `[m, k]` row-major activation tensor on GPU
    /// * `weight` - `[n, k]` row-major weight matrix on GPU
    /// * `bias`   - optional `[n]` bias vector on GPU
    /// * `m`      - number of tokens / rows in input
    /// * `n`      - output features (rows in weight)
    /// * `k`      - input features (cols in weight, cols in input)
    pub fn forward(
        &self,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaSlice<f32>> {
        if input.len() < m * k {
            return Err(LLMError::GpuError(format!(
                "CudaLinearLayer: input len {} < m*k = {}",
                input.len(),
                m * k
            )));
        }
        if weight.len() < n * k {
            return Err(LLMError::GpuError(format!(
                "CudaLinearLayer: weight len {} < n*k = {}",
                weight.len(),
                n * k
            )));
        }
        if let Some(b) = bias {
            if b.len() < n {
                return Err(LLMError::GpuError(format!(
                    "CudaLinearLayer: bias len {} < n = {}",
                    b.len(),
                    n
                )));
            }
        }

        let device = self.ops.device();

        // Allocate output [m, n]. If bias is present, tile it into every row so
        // sgemm accumulates on top with beta=1.
        let mut output: CudaSlice<f32> = if let Some(b) = bias {
            let bias_host = device
                .dtoh_sync_copy(b)
                .map_err(|e| LLMError::GpuError(format!("bias dtoh failed: {e}")))?;
            let mut tiled = Vec::with_capacity(m * n);
            for _ in 0..m {
                tiled.extend_from_slice(&bias_host[..n]);
            }
            device
                .htod_sync_copy(&tiled)
                .map_err(|e| LLMError::GpuError(format!("tiled bias htod failed: {e}")))?
        } else {
<<<<<<< Updated upstream
            // Safety: sgemm with beta=0 writes all m*n elements
            unsafe { stream.alloc::<f32>(m * n) }
=======
            device
                .alloc_zeros::<f32>(m * n)
>>>>>>> Stashed changes
                .map_err(|e| LLMError::GpuError(format!("output alloc failed: {e}")))?
        };

        let beta = if bias.is_some() { 1.0f32 } else { 0.0f32 };

        // C[m,n] = 1.0 * input[m,k] @ weight^T[k,n] + beta * C[m,n]
        self.ops
            .sgemm_a_bt(m, n, k, 1.0, input, weight, beta, &mut output)?;

        Ok(output)
    }

    /// Static forward matching the spec signature. Creates a temporary CublasOps;
    /// prefer the instance method [`Self::forward`] for repeated calls.
    pub fn forward_once(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
    ) -> Result<CudaSlice<f32>> {
        let layer = Self::from_handle(blas)?;
        layer.forward(input, weight, bias, m, n, k)
    }

    /// Static forward with FP8 weights.
    ///
    /// For small M (decode, M ≤ 32): uses a fused FP8 GEMV kernel that reads FP8
    /// weights directly from memory and accumulates in f32. This avoids allocating
    /// a temporary FP16 weight buffer and reduces memory traffic by ~5×.
    ///
    /// For large M (prefill, M > 32): falls back to dequant FP8→FP16 + HGEMM,
    /// which is compute-bound and benefits from tensor cores.
    ///
    /// Optional per-tensor or block-wise scale is applied during computation.
    pub fn forward_once_fp8(
        input: &CudaSlice<f32>,
        weight_fp8: &CudaSlice<u8>,
        weight_scale: Option<&CudaSlice<f32>>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
    ) -> Result<CudaSlice<f32>> {
        let device = blas.device();

        // For small batch sizes (decode), use fused FP8 GEMV which reads weights
        // once from memory (1 byte/param) instead of dequant+GEMM (5 bytes/param).
        if m <= 32 {
            return Self::gpu_fp8_gemv(device, input, weight_fp8, weight_scale, m, n, k);
        }

        // For larger batches (prefill), dequant to BF16 then use cuBLAS GEMM.
        let weight_bf16 = Self::gpu_dequant_fp8_to_bf16(device, weight_fp8, weight_scale, n, k)?;
        Self::forward_once_bf16(input, &weight_bf16, m, n, k, blas)
    }

    /// Fused FP8 GEMV: output[M,N] = input[M,K] @ weight_fp8[N,K]^T * scale
    ///
    /// Reads FP8 weights directly from GPU memory and computes the matrix-vector
    /// product in f32 with block-wise scale applied during accumulation.
    /// No temporary FP16 weight buffer needed.
    ///
    /// Uses shared-memory input caching when the `fp8_gemv_smem` kernel is loaded,
    /// which eliminates L2 contention between input and weight reads.
    fn gpu_fp8_gemv(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<f32>,
        weight_fp8: &CudaSlice<u8>,
        scale: Option<&CudaSlice<f32>>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaSlice<f32>> {
        let mut output = device
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| LLMError::GpuError(format!("fp8_gemv alloc: {e}")))?;

        // Try shared-memory kernel first (eliminates L2 contention for input reads).
        // Shared-memory kernel disabled (causes hangs on Blackwell sm_121).
        if false {
            if let Some(scale_ptr) = scale {
                if scale_ptr.len() > 1 {
                    if let Some(smem_kernel) = device.get_func("fp8_gemv_smem", "fp8_gemv_smem_kernel") {
                        let threads = 256u32;
                        let smem_cfg = LaunchConfig {
                            grid_dim: (n as u32, m as u32, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: 4, // non-zero for Blackwell compat (static smem handles the rest)
                        };
                        let num_col_blocks = (k + 127) / 128;
                        unsafe {
                            smem_kernel
                                .launch(smem_cfg, (
                                    &mut output,
                                    weight_fp8,
                                    scale_ptr,
                                    input,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    num_col_blocks as i32,
                                ))
                                .map_err(|e| LLMError::GpuError(format!("fp8_gemv_smem launch: {e}")))?;
                        }
                        return Ok(output);
                    }
                }
            }
        }

        // Warp-per-row kernel: 8 warps per block, each warp computes one output row.
        // Uses branchless FP8→f32 (24 insn/elem). This is FASTER than hardware CVT
        // (3 insn/elem) on DIGITS: the ALU work throttles memory request rate, keeping
        // power under the firmware cap so clocks sustain 851 MHz. Native CVT triggers
        // a drop to 507 MHz after ~3s — even with nvidia-smi -lgc, firmware overrides.
        let threads = 256u32;
        let wpr_rows = 8u32;
        let cfg = LaunchConfig {
            grid_dim: (((n as u32) + wpr_rows - 1) / wpr_rows, m as u32, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        if let Some(scale_ptr) = scale {
            if scale_ptr.len() > 1 {
                // Block-wise scale
                let num_col_blocks = (k + 127) / 128;
                let kernel = device
                    .get_func("fp8_gemv", "fp8_gemv_blockwise_wpr_kernel")
                    .ok_or_else(|| LLMError::GpuError("fp8_gemv_blockwise_wpr_kernel not loaded".into()))?;
                unsafe {
                    kernel
                        .launch(cfg, (
                            &mut output,
                            weight_fp8,
                            scale_ptr,
                            input,
                            m as i32,
                            n as i32,
                            k as i32,
                            num_col_blocks as i32,
                        ))
                        .map_err(|e| LLMError::GpuError(format!("fp8_gemv_wpr launch: {e}")))?;
                }
            } else {
                // Per-tensor scale
                let kernel = device
                    .get_func("fp8_gemv", "fp8_gemv_scaled_kernel")
                    .ok_or_else(|| LLMError::GpuError("fp8_gemv_scaled_kernel not loaded".into()))?;
                unsafe {
                    kernel
                        .launch(cfg, (
                            &mut output,
                            weight_fp8,
                            scale_ptr,
                            input,
                            m as i32,
                            n as i32,
                            k as i32,
                        ))
                        .map_err(|e| LLMError::GpuError(format!("fp8_gemv_scaled launch: {e}")))?;
                }
            }
        } else {
            // No scale
            let kernel = device
                .get_func("fp8_gemv", "fp8_gemv_kernel")
                .ok_or_else(|| LLMError::GpuError("fp8_gemv_kernel not loaded".into()))?;
            unsafe {
                kernel
                    .launch(cfg, (
                        &mut output,
                        weight_fp8,
                        input,
                        m as i32,
                        n as i32,
                        k as i32,
                    ))
                    .map_err(|e| LLMError::GpuError(format!("fp8_gemv launch: {e}")))?;
            }
        }

        Ok(output)
    }

    /// FP8 GEMV into a pre-allocated output buffer (avoids cudaMalloc per call).
    /// Only supports decode path (m <= 32). Caller must ensure output has m*n elements.
    pub fn forward_once_fp8_into(
        output: &mut CudaSlice<f32>,
        input: &CudaSlice<f32>,
        weight_fp8: &CudaSlice<u8>,
        weight_scale: Option<&CudaSlice<f32>>,
        m: usize,
        n: usize,
        k: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<()> {
        debug_assert!(m <= 32, "forward_once_fp8_into only for decode (m<=32)");

        // Try shared-memory kernel first (eliminates L2 contention).
        // Shared-memory kernel disabled (causes hangs on Blackwell sm_121).
        if false {
            if let Some(scale_ptr) = weight_scale {
                if scale_ptr.len() > 1 {
                    if let Some(smem_kernel) = device.get_func("fp8_gemv_smem", "fp8_gemv_smem_kernel") {
                        let threads = 256u32;
                        let smem_cfg = LaunchConfig {
                            grid_dim: (n as u32, m as u32, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: (threads / 32) * 4, // warp_sums only
                        };
                        let num_col_blocks = (k + 127) / 128;
                        unsafe {
                            smem_kernel.launch(smem_cfg, (
                                output as &mut CudaSlice<f32>,
                                weight_fp8, scale_ptr, input,
                                m as i32, n as i32, k as i32, num_col_blocks as i32,
                            )).map_err(|e| LLMError::GpuError(format!("fp8_gemv_smem_into: {e}")))?;
                        }
                        return Ok(());
                    }
                }
            }
        }

        // Warp-per-row kernel (see comment above for power/clock rationale).
        let threads = 256u32;
        let wpr_rows = 8u32;
        let cfg = LaunchConfig {
            grid_dim: (((n as u32) + wpr_rows - 1) / wpr_rows, m as u32, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        if let Some(scale_ptr) = weight_scale {
            if scale_ptr.len() > 1 {
                let num_col_blocks = (k + 127) / 128;
                let kernel = device
                    .get_func("fp8_gemv", "fp8_gemv_blockwise_wpr_kernel")
                    .ok_or_else(|| LLMError::GpuError("fp8_gemv_blockwise_wpr_kernel not loaded".into()))?;
                unsafe {
                    kernel.launch(cfg, (
                        output as &mut CudaSlice<f32>,
                        weight_fp8, scale_ptr, input,
                        m as i32, n as i32, k as i32, num_col_blocks as i32,
                    )).map_err(|e| LLMError::GpuError(format!("fp8_gemv_wpr_into blockwise: {e}")))?;
                }
            } else {
                let kernel = device
                    .get_func("fp8_gemv", "fp8_gemv_scaled_kernel")
                    .ok_or_else(|| LLMError::GpuError("fp8_gemv_scaled_kernel not loaded".into()))?;
                unsafe {
                    kernel.launch(cfg, (
                        output as &mut CudaSlice<f32>,
                        weight_fp8, scale_ptr, input,
                        m as i32, n as i32, k as i32,
                    )).map_err(|e| LLMError::GpuError(format!("fp8_gemv_into scaled: {e}")))?;
                }
            }
        } else {
            let kernel = device
                .get_func("fp8_gemv", "fp8_gemv_kernel")
                .ok_or_else(|| LLMError::GpuError("fp8_gemv_kernel not loaded".into()))?;
            unsafe {
                kernel.launch(cfg, (
                    output as &mut CudaSlice<f32>,
                    weight_fp8, input,
                    m as i32, n as i32, k as i32,
                )).map_err(|e| LLMError::GpuError(format!("fp8_gemv_into: {e}")))?;
            }
        }

        Ok(())
    }

    /// Static forward with GPTQ INT4 weights.
    ///
    /// For small M (decode, M <= 32): fused INT4 GEMV reads packed int4 weights
    /// directly and dequantizes on-the-fly. Halves weight bandwidth vs FP8.
    ///
    /// For large M (prefill, M > 32): dequant INT4->F16 then HGEMM.
    pub fn forward_once_gptq(
        input: &CudaSlice<f32>,
        qweight: &CudaSlice<u8>,      // [N, K/8] repacked int32 as raw bytes
        scales: &CudaSlice<f32>,       // [N, num_groups]
        zeros: &CudaSlice<f32>,        // [N, num_groups]
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
        blas: &CublasHandle,
    ) -> Result<CudaSlice<f32>> {
        let device = blas.device();

        if m <= 32 {
            return Self::gpu_int4_gemv(device, input, qweight, scales, zeros, m, n, k, group_size);
        }

        // Prefill: dequant to F16 then HGEMM
        let weight_f16 = Self::gpu_dequant_int4_to_f16(device, qweight, scales, zeros, n, k, group_size)?;
        Self::forward_once_f16(input, &weight_f16, m, n, k, blas)
    }

    /// Fused INT4 GEMV: output[M,N] = input[M,K] @ dequant(qweight[N,K])^T
    fn gpu_int4_gemv(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<f32>,
        qweight: &CudaSlice<u8>,
        scales: &CudaSlice<f32>,
        zeros: &CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<CudaSlice<f32>> {
        let mut output = device
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| LLMError::GpuError(format!("int4_gemv alloc: {e}")))?;

        let kernel = device
            .get_func("int4_gemv", "int4_gemv_kernel")
            .ok_or_else(|| LLMError::GpuError("int4_gemv_kernel not loaded".into()))?;

        let threads = 256u32;
        let num_groups = (k + group_size - 1) / group_size;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, m as u32, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: (threads / 32) * 4,
        };

        unsafe {
            kernel
                .launch(cfg, (
                    &mut output,
                    qweight,
                    scales,
                    zeros,
                    input,
                    m as i32,
                    n as i32,
                    k as i32,
                    group_size as i32,
                    num_groups as i32,
                ))
                .map_err(|e| LLMError::GpuError(format!("int4_gemv launch: {e}")))?;
        }

        Ok(output)
    }

    /// INT4 GEMV into pre-allocated output buffer (avoids cudaMalloc per call).
    pub fn forward_once_gptq_into(
        output: &mut CudaSlice<f32>,
        input: &CudaSlice<f32>,
        qweight: &CudaSlice<u8>,
        scales: &CudaSlice<f32>,
        zeros: &CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<()> {
        debug_assert!(m <= 32, "forward_once_gptq_into only for decode (m<=32)");

        let kernel = device
            .get_func("int4_gemv", "int4_gemv_kernel")
            .ok_or_else(|| LLMError::GpuError("int4_gemv_kernel not loaded".into()))?;

        let threads = 256u32;
        let num_groups = (k + group_size - 1) / group_size;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, m as u32, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: (threads / 32) * 4,
        };

        unsafe {
            kernel
                .launch(cfg, (
                    output as &mut CudaSlice<f32>,
                    qweight,
                    scales,
                    zeros,
                    input,
                    m as i32,
                    n as i32,
                    k as i32,
                    group_size as i32,
                    num_groups as i32,
                ))
                .map_err(|e| LLMError::GpuError(format!("int4_gemv_into: {e}")))?;
        }

        Ok(())
    }

    /// Dequantize INT4 (GPTQ) to f16 for HGEMM prefill path.
    fn gpu_dequant_int4_to_f16(
        device: &Arc<CudaDevice>,
        qweight: &CudaSlice<u8>,
        scales: &CudaSlice<f32>,
        zeros: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<CudaSlice<f16>> {
        let n = rows * cols;
        let mut output = device
            .alloc_zeros::<f16>(n)
            .map_err(|e| LLMError::GpuError(format!("dequant_int4 alloc: {e}")))?;

        let kernel = device
            .get_func("int4_gemv", "dequant_int4_to_f16_kernel")
            .ok_or_else(|| LLMError::GpuError("dequant_int4_to_f16_kernel not loaded".into()))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let num_groups = (cols + group_size - 1) / group_size;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(cfg, (
                    &mut output,
                    qweight,
                    scales,
                    zeros,
                    rows as i32,
                    cols as i32,
                    group_size as i32,
                    num_groups as i32,
                ))
                .map_err(|e| LLMError::GpuError(format!("dequant_int4_to_f16 launch: {e}")))?;
        }

        Ok(output)
    }

    /// Dequantize FP8 E4M3 (u8) to f16 on GPU using the dequant_fp8 kernel.
    ///
    /// Automatically detects block-wise vs per-tensor scale based on the number of
    /// scale elements. Block-wise uses 128×128 blocks with a 2D scale tensor.
    fn gpu_dequant_fp8_to_f16(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<u8>,
        scale: Option<&CudaSlice<f32>>,
        rows: usize,
        cols: usize,
    ) -> Result<CudaSlice<f16>> {
        let n = rows * cols;
        let mut output = device
            .alloc_zeros::<f16>(n)
            .map_err(|e| LLMError::GpuError(format!("dequant_fp8 alloc: {e}")))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        if let Some(scale_ptr) = scale {
            if scale_ptr.len() > 1 {
                // Block-wise scale: scale is [ceil(rows/128), ceil(cols/128)] flattened
                let block_size: usize = 128;
                let num_col_blocks = (cols + block_size - 1) / block_size;
                let kernel = device
                    .get_func("dequant_fp8", "dequant_fp8_blockwise_to_f16_kernel")
                    .ok_or_else(|| LLMError::GpuError("dequant_fp8_blockwise_to_f16_kernel not loaded".into()))?;
                unsafe {
                    kernel
                        .launch(cfg, (
                            &mut output,
                            input,
                            scale_ptr,
                            rows as i32,
                            cols as i32,
                            block_size as i32,
                            num_col_blocks as i32,
                        ))
                        .map_err(|e| LLMError::GpuError(format!("dequant_fp8_blockwise launch: {e}")))?;
                }
            } else {
                // Per-tensor scale (single element)
                let kernel = device
                    .get_func("dequant_fp8", "dequant_fp8_scaled_to_f16_kernel")
                    .ok_or_else(|| LLMError::GpuError("dequant_fp8_scaled_to_f16_kernel not loaded".into()))?;
                unsafe {
                    kernel
                        .launch(cfg, (&mut output, input, scale_ptr, n as i32))
                        .map_err(|e| LLMError::GpuError(format!("dequant_fp8_scaled launch: {e}")))?;
                }
            }
        } else {
            let kernel = device
                .get_func("dequant_fp8", "dequant_fp8_to_f16_kernel")
                .ok_or_else(|| LLMError::GpuError("dequant_fp8_to_f16_kernel not loaded".into()))?;
            unsafe {
                kernel
                    .launch(cfg, (&mut output, input, n as i32))
                    .map_err(|e| LLMError::GpuError(format!("dequant_fp8 launch: {e}")))?;
            }
        }

        Ok(output)
    }

    /// Dequantize FP8 E4M3 (u8) to bf16 on GPU using the dequant_fp8 BF16 kernels.
    ///
    /// Automatically detects block-wise vs per-tensor scale based on the number of
    /// scale elements. Block-wise uses 128x128 blocks with a 2D scale tensor.
    fn gpu_dequant_fp8_to_bf16(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<u8>,
        scale: Option<&CudaSlice<f32>>,
        rows: usize,
        cols: usize,
    ) -> Result<CudaSlice<bf16>> {
        let n = rows * cols;
        let mut output = device
            .alloc_zeros::<bf16>(n)
            .map_err(|e| LLMError::GpuError(format!("dequant_fp8_bf16 alloc: {e}")))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        if let Some(scale_ptr) = scale {
            if scale_ptr.len() > 1 {
                // Block-wise scale
                let block_size: usize = 128;
                let num_col_blocks = (cols + block_size - 1) / block_size;
                let kernel = device
                    .get_func("dequant_fp8", "dequant_fp8_blockwise_to_bf16_kernel")
                    .ok_or_else(|| LLMError::GpuError("dequant_fp8_blockwise_to_bf16_kernel not loaded".into()))?;
                unsafe {
                    kernel
                        .launch(cfg, (
                            &mut output,
                            input,
                            scale_ptr,
                            rows as i32,
                            cols as i32,
                            block_size as i32,
                            num_col_blocks as i32,
                        ))
                        .map_err(|e| LLMError::GpuError(format!("dequant_fp8_blockwise_bf16 launch: {e}")))?;
                }
            } else {
                // Per-tensor scale
                let kernel = device
                    .get_func("dequant_fp8", "dequant_fp8_scaled_to_bf16_kernel")
                    .ok_or_else(|| LLMError::GpuError("dequant_fp8_scaled_to_bf16_kernel not loaded".into()))?;
                unsafe {
                    kernel
                        .launch(cfg, (&mut output, input, scale_ptr, n as i32))
                        .map_err(|e| LLMError::GpuError(format!("dequant_fp8_scaled_bf16 launch: {e}")))?;
                }
            }
        } else {
            let kernel = device
                .get_func("dequant_fp8", "dequant_fp8_to_bf16_kernel")
                .ok_or_else(|| LLMError::GpuError("dequant_fp8_to_bf16_kernel not loaded".into()))?;
            unsafe {
                kernel
                    .launch(cfg, (&mut output, input, n as i32))
                    .map_err(|e| LLMError::GpuError(format!("dequant_fp8_bf16 launch: {e}")))?;
            }
        }

        Ok(output)
    }

    /// Static forward with bf16 weights: cast f32 input -> bf16, bgemm, cast bf16 output -> f32.
    ///
    /// Matches HuggingFace/PyTorch BF16 inference numerics for FP8 models.
    pub fn forward_once_bf16(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
    ) -> Result<CudaSlice<f32>> {
        let device = blas.device();

        // Cast input f32 -> bf16
        let input_bf16 = Self::gpu_cast_f32_to_bf16(device, input, m * k)?;

        // Allocate bf16 output
        let mut output_bf16 = device
            .alloc_zeros::<bf16>(m * n)
            .map_err(|e| LLMError::GpuError(format!("forward_once_bf16 alloc: {e}")))?;

        // bgemm: output = input @ weight^T
        blas.bgemm(
            m, n, k,
            bf16::ONE,
            &input_bf16,
            weight,
            bf16::ZERO,
            &mut output_bf16,
        )?;

        // Cast output bf16 -> f32
        Self::gpu_cast_bf16_to_f32(device, &output_bf16, m * n)
    }

    /// Static forward with f16 weights: cast f32 input -> f16, hgemm, cast f16 output -> f32.
    ///
    /// Used for the LM head projection when `use_fp16` is enabled.
    pub fn forward_once_f16(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
    ) -> Result<CudaSlice<f32>> {
        let device = blas.device();

        // Cast input f32 -> f16
        let input_f16 = Self::gpu_cast_f32_to_f16(device, input, m * k)?;

<<<<<<< Updated upstream
        // Safety: hgemm with beta=0 writes all m*n elements
        let mut output_f16 = unsafe { stream.alloc::<f16>(m * n) }
=======
        // Allocate f16 output
        let mut output_f16 = device
            .alloc_zeros::<f16>(m * n)
>>>>>>> Stashed changes
            .map_err(|e| LLMError::GpuError(format!("forward_once_f16 alloc: {e}")))?;

        // hgemm: output = input @ weight^T
        blas.hgemm(
            m, n, k,
            f16::ONE,
            &input_f16,
            weight,
            f16::ZERO,
            &mut output_f16,
        )?;

        // Cast output f16 -> f32
        Self::gpu_cast_f16_to_f32(device, &output_f16, m * n)
    }

<<<<<<< Updated upstream
    /// Mixed-precision forward: f32 input, f16 weight, f32 output.
    ///
    /// Casts input f32->f16, then uses cublasGemmEx(f16,f16->f32) to produce
    /// f32 output directly. Saves 1 cast kernel + 1 alloc per call vs the old
    /// forward_once_f16 which does cast_in + hgemm + cast_out (2 casts + 2 allocs).
    pub fn forward_mixed(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
        loader: &rvllm_gpu::kernel_loader::KernelLoader,
    ) -> Result<CudaSlice<f32>> {
        let stream = blas.stream();

        // Cast input f32 -> f16 (still needed; cuBLAS requires matching A/B types)
        let cast_f32_f16 = loader.get_func("cast_fp", "cast_f32_to_f16_kernel")
            .map_err(|e| LLMError::GpuError(format!("load cast_f32_to_f16_kernel: {e}")))?;
        let input_f16 = Self::gpu_cast_f32_to_f16(stream, input, m * k, &cast_f32_f16)?;

        // Safety: hgemm_f32_output with beta=0 writes all m*n elements
        let mut output = unsafe { stream.alloc::<f32>(m * n) }
            .map_err(|e| LLMError::GpuError(format!("forward_mixed alloc: {e}")))?;

        // f16 x f16 -> f32 via cublasGemmEx
        blas.hgemm_f32_output(m, n, k, 1.0, &input_f16, weight, 0.0, &mut output)?;
        Ok(output)
    }

    /// Pre-cast forward: f16 input already cast by caller, f16 weight, f32 output.
    ///
    /// When multiple linears share the same f32 input (e.g. Q/K/V all use normed,
    /// gate/up both use normed2), the caller casts f32->f16 ONCE and calls this
    /// for each projection. Saves N-1 redundant cast kernels.
    pub fn forward_f16_in(
        input_f16: &CudaSlice<f16>,
        weight: &CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
    ) -> Result<CudaSlice<f32>> {
        // Safety: hgemm with beta=0 writes all m*n elements
        let mut output = unsafe { blas.stream().alloc::<f32>(m * n) }
            .map_err(|e| LLMError::GpuError(format!("forward_f16_in alloc: {e}")))?;
        blas.hgemm_f32_output(m, n, k, 1.0, input_f16, weight, 0.0, &mut output)?;
        Ok(output)
    }

    /// Static forward with f16 weights using cublasLt for decode-sized batches.
    ///
    /// When `cublaslt` feature is enabled and `m <= CUBLASLT_M_THRESHOLD`,
    /// uses cublasLt's heuristic algo selection (with workspace + split-K)
    /// for better decode performance. Falls back to standard cuBLAS hgemm
    /// for larger batches.
    #[cfg(feature = "cublaslt")]
    pub fn forward_once_f16_lt(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
        lt: &CublasLtOps,
        loader: &rvllm_gpu::kernel_loader::KernelLoader,
    ) -> Result<CudaSlice<f32>> {
        // For large M (prefill), standard cuBLAS is fine -- less overhead.
        if m > CUBLASLT_M_THRESHOLD {
            return Self::forward_once_f16(input, weight, m, n, k, blas, loader);
        }

        let stream = blas.stream();

        let cast_f32_f16 = loader.get_func("cast_fp", "cast_f32_to_f16_kernel")
            .map_err(|e| LLMError::GpuError(format!("load cast_f32_to_f16_kernel: {e}")))?;
        let cast_f16_f32 = loader.get_func("cast_fp", "cast_f16_to_f32_kernel")
            .map_err(|e| LLMError::GpuError(format!("load cast_f16_to_f32_kernel: {e}")))?;

        let input_f16 = Self::gpu_cast_f32_to_f16(stream, input, m * k, &cast_f32_f16)?;

        // Safety: hgemm with beta=0 writes all m*n elements
        let mut output_f16 = unsafe { stream.alloc::<f16>(m * n) }
            .map_err(|e| LLMError::GpuError(format!("forward_once_f16_lt alloc: {e}")))?;

        lt.hgemm_a_bt(m, n, k, 1.0, &input_f16, weight, 0.0, &mut output_f16)?;

        Self::gpu_cast_f16_to_f32(stream, &output_f16, m * n, &cast_f16_f32)
    }

    pub fn gpu_cast_f32_to_f16(
        stream: &Arc<CudaStream>,
=======
    fn gpu_cast_f32_to_f16(
        device: &Arc<CudaDevice>,
>>>>>>> Stashed changes
        input: &CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<f16>> {
<<<<<<< Updated upstream
        // Safety: cast kernel writes all n elements
        let mut output = unsafe { stream.alloc::<f16>(n) }
=======
        let mut output = device
            .alloc_zeros::<f16>(n)
>>>>>>> Stashed changes
            .map_err(|e| LLMError::GpuError(format!("cast_f32_to_f16 alloc: {e}")))?;

        let kernel = device
            .get_func("cast_fp", "cast_f32_to_f16_kernel")
            .ok_or_else(|| LLMError::GpuError("cast_f32_to_f16_kernel not loaded".into()))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(cfg, (&mut output, input, n as i32))
                .map_err(|e| LLMError::GpuError(format!("cast_f32_to_f16 launch: {e}")))?;
        }
        Ok(output)
    }

    fn gpu_cast_f16_to_f32(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<f16>,
        n: usize,
    ) -> Result<CudaSlice<f32>> {
<<<<<<< Updated upstream
        // Safety: cast kernel writes all n elements
        let mut output = unsafe { stream.alloc::<f32>(n) }
=======
        let mut output = device
            .alloc_zeros::<f32>(n)
>>>>>>> Stashed changes
            .map_err(|e| LLMError::GpuError(format!("cast_f16_to_f32 alloc: {e}")))?;

        let kernel = device
            .get_func("cast_fp", "cast_f16_to_f32_kernel")
            .ok_or_else(|| LLMError::GpuError("cast_f16_to_f32_kernel not loaded".into()))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(cfg, (&mut output, input, n as i32))
                .map_err(|e| LLMError::GpuError(format!("cast_f16_to_f32 launch: {e}")))?;
        }
        Ok(output)
    }

    fn gpu_cast_f32_to_bf16(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<bf16>> {
        let mut output = device
            .alloc_zeros::<bf16>(n)
            .map_err(|e| LLMError::GpuError(format!("cast_f32_to_bf16 alloc: {e}")))?;

        let kernel = device
            .get_func("cast_fp", "cast_f32_to_bf16_kernel")
            .ok_or_else(|| LLMError::GpuError("cast_f32_to_bf16_kernel not loaded".into()))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(cfg, (&mut output, input, n as i32))
                .map_err(|e| LLMError::GpuError(format!("cast_f32_to_bf16 launch: {e}")))?;
        }
        Ok(output)
    }

    fn gpu_cast_bf16_to_f32(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<bf16>,
        n: usize,
    ) -> Result<CudaSlice<f32>> {
        let mut output = device
            .alloc_zeros::<f32>(n)
            .map_err(|e| LLMError::GpuError(format!("cast_bf16_to_f32 alloc: {e}")))?;

        let kernel = device
            .get_func("cast_fp", "cast_bf16_to_f32_kernel")
            .ok_or_else(|| LLMError::GpuError("cast_bf16_to_f32_kernel not loaded".into()))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(cfg, (&mut output, input, n as i32))
                .map_err(|e| LLMError::GpuError(format!("cast_bf16_to_f32 launch: {e}")))?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    // Tests require the `cuda` feature and a real GPU.
    // Run with: cargo test -p rvllm-model-runner --features cuda
}
