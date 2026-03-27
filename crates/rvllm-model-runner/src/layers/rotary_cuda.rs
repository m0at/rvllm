//! CUDA-accelerated Rotary Positional Embedding (RoPE).
//!
//! Dispatches to the `rotary_embedding_kernel` in `rotary_embedding.cu`.
//! Precomputes cos/sin cache on GPU at construction time, then applies
//! rotary embedding to query + key tensors in-place on each forward call.
//!
//! Gated behind `#[cfg(feature = "cuda")]` -- mock-gpu builds never see this.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaDevice, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchAsync, LaunchConfig,
    };
    use tracing::debug;

    use rvllm_core::error::{LLMError, Result};

    /// CUDA-backed rotary positional embedding with precomputed cos/sin caches
    /// resident on GPU memory.
    pub struct CudaRotaryEmbedding {
        cos_cache: CudaSlice<f32>, // [max_position, half_dim]
        sin_cache: CudaSlice<f32>, // [max_position, half_dim]
        device: Arc<CudaDevice>,
        head_dim: usize,
        half_dim: usize,
        max_position: usize,
        kernel_fn: CudaFunction,
    }

    impl CudaRotaryEmbedding {
        /// Build a new RoPE cache and upload it to GPU.
        ///
        /// * `head_dim`     -- dimension per attention head (must be even)
        /// * `max_position` -- maximum sequence position to precompute
        /// * `base`         -- RoPE frequency base (typically 10000.0)
        /// * `device`       -- CUDA device for allocation
        /// * `ptx`          -- compiled PTX bytes for `rotary_embedding.cu`
        pub fn new(
            head_dim: usize,
            max_position: usize,
            base: f32,
            device: Arc<CudaDevice>,
            ptx: &[u8],
        ) -> Result<Self> {
            assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

            let half_dim = head_dim / 2;

            // --- precompute cos/sin tables on CPU ---
            let inv_freq: Vec<f32> = (0..half_dim)
                .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
                .collect();

            let table_len = max_position * half_dim;
            let mut cos_host = Vec::with_capacity(table_len);
            let mut sin_host = Vec::with_capacity(table_len);

            for pos in 0..max_position {
                for &freq in &inv_freq {
                    let theta = pos as f32 * freq;
                    cos_host.push(theta.cos());
                    sin_host.push(theta.sin());
                }
            }

            // --- upload to GPU ---
            let cos_cache = device
                .htod_sync_copy(&cos_host)
                .map_err(|e| LLMError::GpuError(format!("RoPE cos cache upload: {e}")))?;
            let sin_cache = device
                .htod_sync_copy(&sin_host)
                .map_err(|e| LLMError::GpuError(format!("RoPE sin cache upload: {e}")))?;

            debug!(
                head_dim,
                max_position,
                table_bytes = table_len * 4 * 2,
                "CudaRotaryEmbedding: cos/sin caches uploaded to GPU"
            );

            // --- load PTX module ---
            let module_name = "rotary_embedding";
            device
                .load_ptx(
                    cudarc::nvrtc::Ptx::from_src(
                        std::str::from_utf8(ptx)
                            .map_err(|e| LLMError::GpuError(format!("invalid PTX UTF-8: {e}")))?,
                    ),
                    module_name,
                    &["rotary_embedding_kernel"],
                )
                .map_err(|e| LLMError::GpuError(format!("RoPE PTX load: {e}")))?;

            let kernel_fn = device
                .get_func(module_name, "rotary_embedding_kernel")
                .ok_or_else(|| {
                    LLMError::GpuError(
                        "rotary_embedding_kernel not found after PTX load".into(),
                    )
                })?;

            Ok(Self {
                cos_cache,
                sin_cache,
                device,
                head_dim,
                half_dim,
                max_position,
                kernel_fn,
            })
        }

        /// Apply RoPE to query and key tensors **in-place** on GPU.
        ///
        /// # Layout
        /// * `query`      -- `[num_tokens, num_heads * head_dim]` flattened f32 on device
        /// * `key`        -- `[num_tokens, num_kv_heads * head_dim]` flattened f32 on device
        /// * `positions`  -- `[num_tokens]` i32 on device (position index per token)
        /// * `num_tokens` -- number of tokens in the batch
        /// * `num_heads`  -- number of query attention heads
        /// * `num_kv_heads` -- number of key/value attention heads (for GQA)
        /// * `stream`     -- CUDA stream for async execution
        ///
        /// The kernel grid is `(num_tokens, max(num_heads, num_kv_heads), 1)` with
        /// block size `(head_dim / 2, 1, 1)`. The kernel internally skips key
        /// updates for head indices >= num_kv_heads (GQA support).
        pub fn forward(
            &self,
            query: &mut CudaSlice<f32>,
            key: &mut CudaSlice<f32>,
            positions: &CudaSlice<i32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            stream: &CudaStream,
        ) -> Result<()> {
            if num_tokens == 0 {
                return Ok(());
            }

            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (self.half_dim as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            // SAFETY: kernel arguments match the rotary_embedding_kernel signature
            // exactly. All CudaSlice pointers are valid device memory owned by
            // the caller. The launch config ensures thread counts stay within
            // the allocated buffer bounds.
            unsafe {
                self.kernel_fn
                    .clone()
                    .launch_on_stream(
                        stream,
                        cfg,
                        (
                            query,                         // float* query
                            key,                           // float* key
                            &self.cos_cache,               // const float* cos_cache
                            &self.sin_cache,               // const float* sin_cache
                            positions,                     // const int* positions
                            num_tokens as i32,             // int num_tokens
                            num_heads as i32,              // int num_heads
                            num_kv_heads as i32,           // int num_kv_heads
                            self.head_dim as i32,          // int head_dim
                        ),
                    )
                    .map_err(|e| LLMError::GpuError(format!("RoPE kernel launch: {e}")))?;
            }

            Ok(())
        }

        /// Convenience: upload host positions to GPU, apply RoPE, synchronize.
        ///
        /// Useful when positions are only available on CPU (e.g. freshly computed
        /// by the scheduler). For the hot path prefer the fully-on-GPU `forward`.
        pub fn forward_with_host_positions(
            &self,
            query: &mut CudaSlice<f32>,
            key: &mut CudaSlice<f32>,
            positions_host: &[i32],
            num_heads: usize,
            num_kv_heads: usize,
            stream: &CudaStream,
        ) -> Result<()> {
            let positions_dev = self
                .device
                .htod_sync_copy(positions_host)
                .map_err(|e| LLMError::GpuError(format!("RoPE positions upload: {e}")))?;

            self.forward(
                query,
                key,
                &positions_dev,
                positions_host.len(),
                num_heads,
                num_kv_heads,
                stream,
            )
        }

        pub fn head_dim(&self) -> usize {
            self.head_dim
        }

        pub fn max_position(&self) -> usize {
            self.max_position
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::CudaRotaryEmbedding;
