//! GPU forward pass orchestrator (Agent 13).
//!
//! `GpuModelRunner` drives the full Llama-family forward pass on CUDA:
//! token embedding lookup -> N transformer layers -> final RMSNorm -> LM head -> logits.
//!
//! All CUDA code is gated behind `#[cfg(feature = "cuda")]`. Under `mock-gpu`
//! (the default), this module provides a compile-compatible stub that returns
//! an error at runtime so existing Mac-side tests keep working.

// =========================================================================
//  CUDA implementation
// =========================================================================
/// Output of a forward pass -- either full logits or just argmax token IDs.
#[derive(Debug, Clone)]
pub enum ForwardOutput {
    /// Full logits buffer: [num_tokens * vocab_size] f32.
    Logits(Vec<f32>),
    /// GPU-side argmax token IDs: [num_tokens] i32 (greedy fast path).
    TokenIds(Vec<i32>),
    /// Async DtoH in progress: token IDs are in a pinned host buffer but
    /// the stream has not been synchronized yet. The caller must call
    /// `sync_stream()` + read from the pinned buffer before using the data.
    /// Fields: (actual_batch_size,) -- the pinned buffer lives on the worker.
    TokenIdsPending { actual_batch: usize },
}

/// Extract the raw CUfunction handle from a CudaFunction for raw launch APIs.
#[cfg(feature = "cuda")]
unsafe fn extract_cu_function_from_loader(
    func: &cudarc::driver::CudaFunction,
) -> cudarc::driver::sys::CUfunction {
    std::ptr::read(
        (func as *const cudarc::driver::CudaFunction)
            .cast::<cudarc::driver::sys::CUfunction>(),
    )
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::cell::RefCell;
    use std::sync::Arc;

    use std::cell::Cell;

    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
    use half::f16;
    use tracing::{debug, info, trace};

    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

    use crate::gpu_layer::{
        GpuLayerConfig, GpuLayerInput, GpuLayerWeights, GpuTransformerLayer,
    };
    use crate::layers::linear_cuda::CudaLinearLayer;

    use rvllm_gpu::kernel_loader::KernelLoader;
    use rvllm_gpu::prelude::CublasHandle;
    use rvllm_kv_cache::engine_cuda::CudaCacheEngine;
    use rvllm_model_loader::gpu_weights::GpuModelWeights;

    use super::ForwardOutput;

    /// Reusable GPU buffer that grows as needed, eliminating per-step CUDA
    /// allocations on the hot decode path.
    struct ReusableGpuBuf {
        buf: Option<CudaSlice<i32>>,
    }

    impl ReusableGpuBuf {
        fn new() -> Self {
            Self { buf: None }
        }

        /// Upload `data` into the reusable buffer. If the existing GPU allocation
        /// is large enough, copies via `memcpy_htod` (zero alloc). Otherwise
        /// allocates a new buffer with 2x headroom and copies into that.
        fn upload(
            &mut self,
            data: &[i32],
            stream: &Arc<CudaStream>,
        ) -> std::result::Result<(), cudarc::driver::result::DriverError> {
            let need = data.len();
            if need == 0 {
                // Ensure we have at least a 1-element buffer so references are valid.
                if self.buf.is_none() {
                    self.buf = Some(stream.alloc_zeros::<i32>(1)?);
                }
                return Ok(());
            }
            let have = self.buf.as_ref().map_or(0, |b| b.len());
            if have < need {
                // Grow with 2x headroom to amortize future resizes.
                let cap = need.max(have * 2).max(64);
                self.buf = Some(stream.alloc_zeros::<i32>(cap)?);
            }
            stream.memcpy_htod(data, self.buf.as_mut().unwrap())?;
            Ok(())
        }

        fn slice(&self) -> &CudaSlice<i32> {
            self.buf.as_ref().expect("upload() must be called first")
        }
    }

    /// Pre-allocated f16 scratch buffers for the forward pass.
    /// Sized for max_batch_tokens. Reused across all layers (sequential execution).
    pub struct F16LayerScratch {
        pub qkv: CudaSlice<f16>,      // [max_tokens * qkv_dim]
        pub attn_out: CudaSlice<f16>, // [max_tokens * q_dim]
        pub o_proj: CudaSlice<f16>,   // [max_tokens * hidden]
        pub normed: CudaSlice<f16>,   // [max_tokens * hidden]
        pub residual: CudaSlice<f16>, // [max_tokens * hidden]
        pub gate_up: CudaSlice<f16>,  // [max_tokens * intermediate * 2]
        pub silu_out: CudaSlice<f16>, // [max_tokens * intermediate]
        pub down: CudaSlice<f16>,     // [max_tokens * hidden]
    }

    /// Scratch buffers for the persistent cooperative-groups layer kernel.
    /// Tiny (< 100KB total), allocated once, reused across all layers.
    pub struct PersistentLayerScratch {
        pub qkv: CudaSlice<f16>,       // [qkv_dim]
        pub attn: CudaSlice<f16>,      // [q_dim]
        pub oproj: CudaSlice<f16>,     // [hidden_size]
        pub gateup: CudaSlice<f16>,    // [gate_up_dim]
    }

    /// Element offsets into the packed metadata GPU buffer.
    #[derive(Clone, Copy, Default)]
    struct PackedMetaOffsets {
        token_ids: usize,
        positions: usize,
        context_lens: usize,
        block_tables: usize,
        slot_mapping: usize,
        seq_start_pos: usize,
        num_token_ids: usize,
        num_positions: usize,
        num_context_lens: usize,
        num_block_tables: usize,
        num_slot_mapping: usize,
        num_seq_start_pos: usize,
    }

    pub struct GpuModelRunner {
        weights: GpuModelWeights,
        cache: CudaCacheEngine,
        blas: CublasHandle,
        #[cfg(feature = "cublaslt")]
        blas_lt: Option<rvllm_gpu::cublaslt_ops::CublasLtOps>,
        loader: Arc<KernelLoader>,
        config: ModelRunnerConfig,
        device: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        layers: Vec<GpuTransformerLayer>,
        embed_tokens: CudaSlice<f16>,
        final_norm_weight: CudaSlice<f16>,
        lm_head_weight: CudaSlice<f16>,
        rms_norm_eps: f32,
        /// Precomputed RoPE cos table on GPU: [max_position, head_dim/2]
        rope_cos: CudaSlice<f32>,
        /// Precomputed RoPE sin table on GPU: [max_position, head_dim/2]
        rope_sin: CudaSlice<f32>,
        /// Single packed GPU buffer for all per-step metadata (token_ids,
        /// positions, context_lens, block_tables, slot_mapping, seq_start_pos).
        /// One memcpy_htod per decode step instead of 6 separate transfers.
        meta_packed: RefCell<ReusableGpuBuf>,
        /// Element offsets into meta_packed for each metadata field.
        meta_packed_offsets: Cell<PackedMetaOffsets>,
        /// Persistent GPU output buffer for CUDA graph capture/replay.
        graph_output: RefCell<Option<CudaSlice<i32>>>,
        /// Reusable CPU scratch buffer for packing metadata.
        cpu_scratch: RefCell<Vec<i32>>,
        /// Fixed max blocks per sequence for CUDA graph capture/replay.
        graph_max_blocks: usize,
        /// Fused QKV weights per layer: [q_dim + kv_dim + kv_dim, hidden] f16.
        /// One GEMM instead of 3 per layer. Populated by fuse_weights().
        fused_qkv_weights: Vec<CudaSlice<half::f16>>,
        /// Fused gate+up weights per layer: [intermediate*2, hidden] f16.
        /// One GEMM instead of 2 per layer. Populated by fuse_weights().
        fused_gate_up_weights: Vec<CudaSlice<half::f16>>,
        /// Fused QKV bias per layer (None if model has no QKV bias).
        fused_qkv_bias: Vec<Option<CudaSlice<half::f16>>>,
        /// Pre-allocated scratch buffers for the forward pass.
        f16_scratch: Option<F16LayerScratch>,
        /// Scratch buffers for persistent cooperative layer kernel (M=1 decode).
        persistent_scratch: Option<PersistentLayerScratch>,
        /// Cached max cooperative grid size for the persistent layer kernel.
        persistent_max_grid: u32,
    }

    impl GpuModelRunner {
        pub fn new(
            weights: GpuModelWeights,
            cache: CudaCacheEngine,
            blas: CublasHandle,
            loader: KernelLoader,
            config: ModelRunnerConfig,
            device: Arc<CudaContext>,
            stream: Arc<CudaStream>,
        ) -> Result<Self> {
            let loader = Arc::new(loader);
            debug!(
                num_layers = config.num_layers,
                hidden = config.hidden_size,
                vocab = config.vocab_size,
                "GpuModelRunner::new"
            );

            let embed_tokens = weights
                .get("model.embed_tokens.weight")
                .ok_or_else(|| LLMError::GpuError("missing model.embed_tokens.weight".into()))?
                .clone();

            let final_norm_weight = weights
                .get("model.norm.weight")
                .ok_or_else(|| LLMError::GpuError("missing model.norm.weight".into()))?
                .clone();

            let lm_head_weight = weights
                .get("lm_head.weight")
                .or_else(|| weights.get("model.embed_tokens.weight"))
                .ok_or_else(|| {
                    LLMError::GpuError(
                        "missing lm_head.weight and model.embed_tokens.weight".into(),
                    )
                })?
                .clone();

            let mut layers = Vec::with_capacity(config.num_layers);
            for i in 0..config.num_layers {
                let layer_cfg = GpuLayerConfig {
                    hidden_size: config.hidden_size,
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.head_dim,
                    intermediate_size: config.intermediate_size,
                    rms_norm_eps: config.rms_norm_eps,
                    layer_idx: i,
                };
                layers.push(GpuTransformerLayer::new(layer_cfg, Arc::clone(&stream), Arc::clone(&loader)));
            }
            info!(
                rms_norm_eps = config.rms_norm_eps,
                rope_theta = config.rope_theta,
                num_heads = config.num_heads,
                num_kv_heads = config.num_kv_heads,
                head_dim = config.head_dim,
                hidden = config.hidden_size,
                vocab = config.vocab_size,
                "model config verified"
            );

            // Precompute RoPE cos/sin tables
            let head_dim = config.head_dim;
            let max_pos = config.max_position.min(8192);
            let half_dim = head_dim / 2;
            let rope_theta = config.rope_theta;
            let mut cos_table = vec![0.0f32; max_pos * half_dim];
            let mut sin_table = vec![0.0f32; max_pos * half_dim];
            for pos in 0..max_pos {
                for i in 0..half_dim {
                    let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                    let theta = pos as f32 * freq;
                    cos_table[pos * half_dim + i] = theta.cos();
                    sin_table[pos * half_dim + i] = theta.sin();
                }
            }
            let rope_cos = stream
                .clone_htod(&cos_table)
                .map_err(|e| LLMError::GpuError(format!("rope cos HtoD: {e}")))?;
            let rope_sin = stream
                .clone_htod(&sin_table)
                .map_err(|e| LLMError::GpuError(format!("rope sin HtoD: {e}")))?;
            info!(max_pos, half_dim, "RoPE tables uploaded to GPU");

            let block_size = cache.block_size();
            let graph_max_blocks = (config.max_position + block_size - 1) / block_size;
            let rms_norm_eps = config.rms_norm_eps;
            info!(graph_max_blocks, block_size, max_position = config.max_position,
                  "fixed block_tables stride for CUDA graph stability");

            #[cfg(feature = "cublaslt")]
            let blas_lt = rvllm_gpu::cublaslt_ops::CublasLtOps::new(stream.clone()).ok();

            Ok(Self {
                weights,
                cache,
                blas,
                #[cfg(feature = "cublaslt")]
                blas_lt,
                loader,
                config,
                device,
                stream,
                layers,
                embed_tokens,
                final_norm_weight,
                lm_head_weight,
                rms_norm_eps,
                rope_cos,
                rope_sin,
                meta_packed: RefCell::new(ReusableGpuBuf::new()),
                meta_packed_offsets: Cell::new(PackedMetaOffsets::default()),
                graph_output: RefCell::new(None),
                cpu_scratch: RefCell::new(Vec::with_capacity(4096)),
                graph_max_blocks,
                fused_qkv_weights: Vec::new(),
                fused_gate_up_weights: Vec::new(),
                fused_qkv_bias: Vec::new(),
                f16_scratch: None,
                persistent_scratch: None,
                persistent_max_grid: 0,
            })
        }

        /// Fuse QKV and gate+up weights for each layer.
        /// Concatenates on GPU: q_proj || k_proj || v_proj -> fused_qkv
        /// and gate_proj || up_proj -> fused_gate_up.
        /// Reduces 5 GEMMs to 2 per layer (3 QKV->1, 2 gate+up->1).
        pub fn fuse_weights(&mut self) -> Result<()> {
            let num_layers = self.layers.len();
            let hidden = self.config.hidden_size;
            let q_dim = self.config.num_heads * self.config.head_dim;
            let kv_dim = self.config.num_kv_heads * self.config.head_dim;
            let qkv_dim = q_dim + kv_dim + kv_dim;
            let intermediate = self.config.intermediate_size;
            let gate_up_dim = intermediate * 2;

            for i in 0..num_layers {
                // Fuse QKV: concat [q_dim, hidden] + [kv_dim, hidden] + [kv_dim, hidden]
                let q_w = self.weights.get(
                    &format!("model.layers.{i}.self_attn.q_proj.weight")
                ).ok_or_else(|| LLMError::GpuError(format!("missing q_proj layer {i}")))?;
                let k_w = self.weights.get(
                    &format!("model.layers.{i}.self_attn.k_proj.weight")
                ).ok_or_else(|| LLMError::GpuError(format!("missing k_proj layer {i}")))?;
                let v_w = self.weights.get(
                    &format!("model.layers.{i}.self_attn.v_proj.weight")
                ).ok_or_else(|| LLMError::GpuError(format!("missing v_proj layer {i}")))?;

                let mut fused_qkv = self.stream.alloc_zeros::<half::f16>(qkv_dim * hidden)
                    .map_err(|e| LLMError::GpuError(format!("fused_qkv alloc: {e}")))?;
                self.stream.memcpy_dtod(q_w, &mut fused_qkv.slice_mut(..q_dim * hidden))
                    .map_err(|e| LLMError::GpuError(format!("fused_qkv q copy: {e}")))?;
                self.stream.memcpy_dtod(k_w, &mut fused_qkv.slice_mut(q_dim * hidden..(q_dim + kv_dim) * hidden))
                    .map_err(|e| LLMError::GpuError(format!("fused_qkv k copy: {e}")))?;
                self.stream.memcpy_dtod(v_w, &mut fused_qkv.slice_mut((q_dim + kv_dim) * hidden..qkv_dim * hidden))
                    .map_err(|e| LLMError::GpuError(format!("fused_qkv v copy: {e}")))?;
                self.fused_qkv_weights.push(fused_qkv);

                // Fuse gate+up: concat [intermediate, hidden] + [intermediate, hidden]
                let gate_w = self.weights.get(
                    &format!("model.layers.{i}.mlp.gate_proj.weight")
                ).ok_or_else(|| LLMError::GpuError(format!("missing gate_proj layer {i}")))?;
                let up_w = self.weights.get(
                    &format!("model.layers.{i}.mlp.up_proj.weight")
                ).ok_or_else(|| LLMError::GpuError(format!("missing up_proj layer {i}")))?;

                let mut fused_gu = self.stream.alloc_zeros::<half::f16>(gate_up_dim * hidden)
                    .map_err(|e| LLMError::GpuError(format!("fused_gate_up alloc: {e}")))?;
                self.stream.memcpy_dtod(gate_w, &mut fused_gu.slice_mut(..intermediate * hidden))
                    .map_err(|e| LLMError::GpuError(format!("fused_gate_up gate copy: {e}")))?;
                self.stream.memcpy_dtod(up_w, &mut fused_gu.slice_mut(intermediate * hidden..gate_up_dim * hidden))
                    .map_err(|e| LLMError::GpuError(format!("fused_gate_up up copy: {e}")))?;
                self.fused_gate_up_weights.push(fused_gu);

                // Fuse QKV biases: concat q_bias || k_bias || v_bias (already f16)
                let q_bias = self.weights.get(&format!("model.layers.{i}.self_attn.q_proj.bias"));
                let k_bias = self.weights.get(&format!("model.layers.{i}.self_attn.k_proj.bias"));
                let v_bias = self.weights.get(&format!("model.layers.{i}.self_attn.v_proj.bias"));
                if let (Some(qb), Some(kb), Some(vb)) = (q_bias, k_bias, v_bias) {
                    let mut fused_bias = self.stream.alloc_zeros::<half::f16>(qkv_dim)
                        .map_err(|e| LLMError::GpuError(format!("fused_bias alloc: {e}")))?;
                    self.stream.memcpy_dtod(qb, &mut fused_bias.slice_mut(..q_dim))
                        .map_err(|e| LLMError::GpuError(format!("fused_bias q copy: {e}")))?;
                    self.stream.memcpy_dtod(kb, &mut fused_bias.slice_mut(q_dim..q_dim + kv_dim))
                        .map_err(|e| LLMError::GpuError(format!("fused_bias k copy: {e}")))?;
                    self.stream.memcpy_dtod(vb, &mut fused_bias.slice_mut(q_dim + kv_dim..qkv_dim))
                        .map_err(|e| LLMError::GpuError(format!("fused_bias v copy: {e}")))?;
                    self.fused_qkv_bias.push(Some(fused_bias));
                } else {
                    self.fused_qkv_bias.push(None);
                }
            }

            info!(num_layers, qkv_dim, gate_up_dim, "fused QKV and gate+up weights (f16)");

            self.alloc_scratch()?;
            Ok(())
        }

        /// Pre-allocate a reusable set of f16 scratch buffers for the forward pass.
        /// Sized for max padded batch (256). Since all layers are processed
        /// sequentially, one set of buffers covers every layer.
        fn alloc_scratch(&mut self) -> Result<()> {
            let max_tokens: usize = 512; // support up to N=512 batch decode
            let hidden = self.config.hidden_size;
            let q_dim = self.config.num_heads * self.config.head_dim;
            let kv_dim = self.config.num_kv_heads * self.config.head_dim;
            let qkv_dim = q_dim + kv_dim + kv_dim;
            let intermediate = self.config.intermediate_size;

            let alloc = |n: usize| -> Result<CudaSlice<f16>> {
                self.stream.alloc_zeros::<f16>(n)
                    .map_err(|e| LLMError::GpuError(format!("f16 scratch alloc ({n} elems): {e}")))
            };

            let scratch = F16LayerScratch {
                qkv: alloc(max_tokens * qkv_dim)?,
                attn_out: alloc(max_tokens * q_dim)?,
                o_proj: alloc(max_tokens * hidden)?,
                normed: alloc(max_tokens * hidden)?,
                residual: alloc(max_tokens * hidden)?,
                gate_up: alloc(max_tokens * intermediate * 2)?,
                silu_out: alloc(max_tokens * intermediate)?,
                down: alloc(max_tokens * hidden)?,
            };

            let total_bytes = (max_tokens * (qkv_dim + q_dim + hidden * 3 + intermediate * 3)) * 2;
            info!(max_tokens, total_bytes, "f16 layer scratch allocated");
            self.f16_scratch = Some(scratch);

            // Allocate persistent cooperative layer scratch (M=1 only, tiny).
            let gate_up_dim = intermediate * 2;
            let p_scratch = PersistentLayerScratch {
                qkv: alloc(qkv_dim)?,
                attn: alloc(q_dim)?,
                oproj: alloc(hidden)?,
                gateup: alloc(gate_up_dim)?,
            };
            let p_bytes = (qkv_dim + q_dim + hidden + gate_up_dim) * 2;
            info!(p_bytes, "persistent layer scratch allocated");
            self.persistent_scratch = Some(p_scratch);

            // Query cooperative grid capacity if the kernel is loaded.
            if self.loader.has_module("persistent_layer_decode") {
                self.init_persistent_grid(hidden)?;
            }

            Ok(())
        }

        /// Query and cache the max cooperative grid size for the persistent layer kernel.
        fn init_persistent_grid(&mut self, hidden_size: usize) -> Result<()> {
            use rvllm_gpu::cooperative;

            let func = self.loader.get_func(
                "persistent_layer_decode",
                "persistent_layer_decode_f16",
            )?;
            let cu_func = unsafe {
                super::extract_cu_function_from_loader(&func)
            };
            let shared_bytes = (hidden_size * 4 + 32) as u32;
            match unsafe {
                cooperative::max_cooperative_grid(cu_func, 256, shared_bytes, &self.device)
            } {
                Ok(max_grid) => {
                    self.persistent_max_grid = max_grid.min(256);
                    info!(
                        max_grid,
                        capped = self.persistent_max_grid,
                        "persistent layer cooperative grid"
                    );
                }
                Err(e) => {
                    info!("cooperative grid query failed ({e:?}), persistent path disabled");
                    self.persistent_max_grid = 0;
                }
            }
            Ok(())
        }

        /// Whether the persistent cooperative layer kernel is available.
        pub fn has_persistent_layer(&self) -> bool {
            self.persistent_max_grid > 0 && self.persistent_scratch.is_some()
        }

        /// Access persistent layer scratch buffers.
        pub fn persistent_scratch(&self) -> &PersistentLayerScratch {
            self.persistent_scratch.as_ref().expect("persistent_scratch not allocated")
        }

        /// Max cooperative grid size for the persistent layer kernel.
        pub fn persistent_max_grid(&self) -> u32 {
            self.persistent_max_grid
        }

        /// Access the pre-allocated f16 scratch buffers.
        /// Panics if called before fuse_weights().
        pub fn f16_scratch(&self) -> &F16LayerScratch {
            self.f16_scratch.as_ref().expect("f16_scratch not allocated; call fuse_weights() first")
        }

        /// Run a single transformer layer using the persistent cooperative-groups
        /// kernel. Only valid for M=1 decode (single token).
        ///
        /// Returns (residual, mlp_out) just like the multi-kernel path.
        ///
        /// # Safety
        /// Caller must ensure metadata is uploaded and all weight/cache pointers are valid.
        pub unsafe fn forward_persistent_layer(
            &self,
            layer_idx: usize,
            prev_residual: &CudaSlice<f16>,
            prev_mlp: Option<&CudaSlice<f16>>,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<(CudaSlice<f16>, CudaSlice<f16>)> {
            use rvllm_gpu::cooperative;

            let hidden = self.config.hidden_size;
            let q_dim = self.config.num_heads * self.config.head_dim;
            let kv_dim = self.config.num_kv_heads * self.config.head_dim;
            let qkv_dim = q_dim + kv_dim + kv_dim;
            let intermediate = self.config.intermediate_size;
            let gate_up_dim = intermediate * 2;

            let ps = self.persistent_scratch.as_ref()
                .ok_or_else(|| LLMError::GpuError("persistent scratch not allocated".into()))?;

            // Outputs: residual_out and mlp_out (newly allocated, kernel writes all elements)
            let residual_out = self.stream.alloc::<f16>(hidden)
                .map_err(|e| LLMError::GpuError(format!("persistent residual alloc: {e}")))?;
            let mlp_out = self.stream.alloc::<f16>(hidden)
                .map_err(|e| LLMError::GpuError(format!("persistent mlp alloc: {e}")))?;

            // Cache pointers (kernel reads + writes KV cache)
            let gpu_cache = self.cache.gpu_cache();
            let (key_cache, value_cache) = &gpu_cache[layer_idx];

            // Metadata from packed buffer
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();

            // Weight pointers
            let weights = self.layer_weights(layer_idx)?;
            let fused_qkv = weights.fused_qkv
                .ok_or_else(|| LLMError::GpuError("persistent layer needs fused QKV".into()))?;
            let fused_gate_up = weights.fused_gate_up
                .ok_or_else(|| LLMError::GpuError("persistent layer needs fused gate_up".into()))?;

            let func = self.loader.get_func(
                "persistent_layer_decode",
                "persistent_layer_decode_f16",
            )?;
            let cu_func = super::extract_cu_function_from_loader(&func);

            // Shared memory: max(hidden_size * 4 + 32, FA_BC * head_dim * 4 + MAX_HPG * FA_BC * 4 + 32)
            let fa_bc: usize = 64;
            let fa_max_hpg: usize = 8;
            let head_dim = self.config.head_dim;
            let smem_norm = hidden * 4 + 32;
            let smem_attn = fa_bc * head_dim * 4 + fa_max_hpg * fa_bc * 4 + 32;
            let shared_mem = smem_norm.max(smem_attn) as u32;

            let grid_x = self.persistent_max_grid;

            // Extract raw device pointers. DevicePtr::device_ptr works for all
            // (read-only trait, kernel handles mutability). All on same stream so
            // ordering is implicit.
            let (mlp_out_ptr, _g0) = DevicePtr::device_ptr(&mlp_out, &self.stream);
            let (res_out_ptr, _g1) = DevicePtr::device_ptr(&residual_out, &self.stream);
            let (prev_res_ptr, _g2) = DevicePtr::device_ptr(prev_residual, &self.stream);

            // prev_mlp: NULL (0u64) for layer 0
            let _g3_opt;
            let prev_mlp_dev: u64;
            if let Some(pm) = prev_mlp {
                let (p, g) = DevicePtr::device_ptr(pm, &self.stream);
                prev_mlp_dev = p;
                _g3_opt = Some(g);
            } else {
                prev_mlp_dev = 0;
                _g3_opt = None;
            }

            let (kc_ptr, _g4) = DevicePtr::device_ptr(key_cache, &self.stream);
            let (vc_ptr, _g5) = DevicePtr::device_ptr(value_cache, &self.stream);

            // Metadata views
            let bt_view = packed_buf.slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables);
            let cl_view = packed_buf.slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens);
            let pos_view = packed_buf.slice(offsets.positions..offsets.positions + offsets.num_positions);
            let sm_view = packed_buf.slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping);

            let (bt_ptr, _g6) = DevicePtr::device_ptr(&bt_view, &self.stream);
            let (cl_ptr, _g7) = DevicePtr::device_ptr(&cl_view, &self.stream);
            let (pos_ptr, _g8) = DevicePtr::device_ptr(&pos_view, &self.stream);
            let (sm_ptr, _g9) = DevicePtr::device_ptr(&sm_view, &self.stream);

            let (rcos_ptr, _g10) = DevicePtr::device_ptr(&self.rope_cos, &self.stream);
            let (rsin_ptr, _g11) = DevicePtr::device_ptr(&self.rope_sin, &self.stream);

            // Weights
            let (norm_w_ptr, _g12) = DevicePtr::device_ptr(weights.input_layernorm, &self.stream);
            let (qkv_w_ptr, _g13) = DevicePtr::device_ptr(fused_qkv, &self.stream);
            let (_g14_opt, qkv_bias_dev);
            if let Some(b) = weights.qkv_bias {
                let (p, g) = DevicePtr::device_ptr(b, &self.stream);
                qkv_bias_dev = p;
                _g14_opt = Some(g);
            } else {
                qkv_bias_dev = 0;
                _g14_opt = None;
            }
            let (o_w_ptr, _g15) = DevicePtr::device_ptr(weights.o_proj, &self.stream);
            let (post_norm_ptr, _g16) = DevicePtr::device_ptr(weights.post_attention_layernorm, &self.stream);
            let (gu_w_ptr, _g17) = DevicePtr::device_ptr(fused_gate_up, &self.stream);
            let (down_w_ptr, _g18) = DevicePtr::device_ptr(weights.down_proj, &self.stream);

            // Scratch buffers
            let (qkv_s_ptr, _g19) = DevicePtr::device_ptr(&ps.qkv, &self.stream);
            let (attn_s_ptr, _g20) = DevicePtr::device_ptr(&ps.attn, &self.stream);
            let (oproj_s_ptr, _g21) = DevicePtr::device_ptr(&ps.oproj, &self.stream);
            let (gu_s_ptr, _g22) = DevicePtr::device_ptr(&ps.gateup, &self.stream);

            // Scalar config
            let eps = self.rms_norm_eps;
            let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
            let hidden_i = hidden as i32;
            let q_dim_i = q_dim as i32;
            let kv_dim_i = kv_dim as i32;
            let qkv_dim_i = qkv_dim as i32;
            let num_heads_i = self.config.num_heads as i32;
            let num_kv_heads_i = self.config.num_kv_heads as i32;
            let head_dim_i = head_dim as i32;
            let intermediate_i = intermediate as i32;
            let gate_up_dim_i = gate_up_dim as i32;
            let block_size_i = block_size as i32;
            let max_context_i = max_context_len as i32;
            let max_blocks_per_seq_i = (max_context_len as usize / block_size + 1) as i32;

            // Build args array -- order must match kernel signature exactly.
            // Each element is a pointer to the local variable holding the CUdeviceptr or scalar.
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                // Outputs
                &mlp_out_ptr as *const u64 as *mut std::ffi::c_void,
                &res_out_ptr as *const u64 as *mut std::ffi::c_void,
                // Inputs
                &prev_res_ptr as *const u64 as *mut std::ffi::c_void,
                &prev_mlp_dev as *const u64 as *mut std::ffi::c_void,
                // Attention I/O
                &kc_ptr as *const u64 as *mut std::ffi::c_void,
                &vc_ptr as *const u64 as *mut std::ffi::c_void,
                &bt_ptr as *const u64 as *mut std::ffi::c_void,
                &cl_ptr as *const u64 as *mut std::ffi::c_void,
                &pos_ptr as *const u64 as *mut std::ffi::c_void,
                &sm_ptr as *const u64 as *mut std::ffi::c_void,
                &rcos_ptr as *const u64 as *mut std::ffi::c_void,
                &rsin_ptr as *const u64 as *mut std::ffi::c_void,
                // Weights
                &norm_w_ptr as *const u64 as *mut std::ffi::c_void,
                &qkv_w_ptr as *const u64 as *mut std::ffi::c_void,
                &qkv_bias_dev as *const u64 as *mut std::ffi::c_void,
                &o_w_ptr as *const u64 as *mut std::ffi::c_void,
                &post_norm_ptr as *const u64 as *mut std::ffi::c_void,
                &gu_w_ptr as *const u64 as *mut std::ffi::c_void,
                &down_w_ptr as *const u64 as *mut std::ffi::c_void,
                // Scratch
                &qkv_s_ptr as *const u64 as *mut std::ffi::c_void,
                &attn_s_ptr as *const u64 as *mut std::ffi::c_void,
                &oproj_s_ptr as *const u64 as *mut std::ffi::c_void,
                &gu_s_ptr as *const u64 as *mut std::ffi::c_void,
                // Config scalars
                &eps as *const f32 as *mut std::ffi::c_void,
                &attn_scale as *const f32 as *mut std::ffi::c_void,
                &hidden_i as *const i32 as *mut std::ffi::c_void,
                &q_dim_i as *const i32 as *mut std::ffi::c_void,
                &kv_dim_i as *const i32 as *mut std::ffi::c_void,
                &qkv_dim_i as *const i32 as *mut std::ffi::c_void,
                &num_heads_i as *const i32 as *mut std::ffi::c_void,
                &num_kv_heads_i as *const i32 as *mut std::ffi::c_void,
                &head_dim_i as *const i32 as *mut std::ffi::c_void,
                &intermediate_i as *const i32 as *mut std::ffi::c_void,
                &gate_up_dim_i as *const i32 as *mut std::ffi::c_void,
                &block_size_i as *const i32 as *mut std::ffi::c_void,
                &max_context_i as *const i32 as *mut std::ffi::c_void,
                &max_blocks_per_seq_i as *const i32 as *mut std::ffi::c_void,
            ];

            self.device.bind_to_thread()
                .map_err(|e| LLMError::GpuError(format!("CUDA bind: {e}")))?;

            cooperative::launch_cooperative(
                cu_func,
                (grid_x, 1, 1),
                (256, 1, 1),
                shared_mem,
                self.stream.cu_stream(),
                &mut args,
            ).map_err(|e| LLMError::GpuError(format!(
                "persistent_layer_decode cooperative launch (layer {layer_idx}): {e}"
            )))?;

            Ok((residual_out, mlp_out))
        }

        pub fn forward(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            is_prefill: bool,
        ) -> Result<Vec<f32>> {
            match self.forward_ex(token_ids, positions, attn_meta, is_prefill, false)? {
                ForwardOutput::Logits(logits) => Ok(logits),
                ForwardOutput::TokenIds(_) | ForwardOutput::TokenIdsPending { .. } => {
                    unreachable!("greedy_only=false must return Logits")
                }
            }
        }

        /// Extended forward: when `greedy_only` is true, runs argmax on GPU and
        /// returns only token IDs (num_tokens * 4 bytes DtoH instead of
        /// num_tokens * vocab_size * 4 bytes).
        pub fn forward_ex(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            is_prefill: bool,
            greedy_only: bool,
        ) -> Result<ForwardOutput> {
            let num_tokens = token_ids.len();
            let num_seqs = attn_meta.context_lens.len();
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            debug!(num_tokens, num_seqs, is_prefill, greedy_only, "GpuModelRunner::forward_ex");

            let max_context_len = attn_meta.max_context_len;

            // Single packed upload: all 6 metadata fields in one memcpy_htod.
            self.upload_metadata(token_ids, positions, attn_meta)?;

            // Step 1: token embedding lookup from packed buffer
            info!("gpu_runner: embedding lookup");

            // === f16 forward path ===
            let debug_fwd = std::env::var("RVLLM_DEBUG").is_ok();
            let mut hidden_f16 = self.embedding_lookup_from_meta(num_tokens)?;

            if debug_fwd {
                let vals: Vec<f16> = self.stream.clone_dtoh(&hidden_f16)
                    .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                let first10: Vec<f32> = vals.iter().take(10).map(|v| v.to_f32()).collect();
                let has_nan = vals.iter().any(|v| v.to_f32().is_nan());
                let has_inf = vals.iter().any(|v| v.to_f32().is_infinite());
                let max_abs = vals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                info!("DEBUG embed: first10={first10:?} nan={has_nan} inf={has_inf} max={max_abs}");
            }

            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            let use_persistent = num_tokens == 1 && !is_prefill && self.has_persistent_layer();
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;

            if use_persistent {
                // Persistent cooperative-groups path: 1 kernel launch per layer
                // instead of 5-6. forward_persistent_layer borrows meta_packed internally.
                for layer_idx in 0..num_layers {
                    let (residual, mlp_out) = unsafe {
                        self.forward_persistent_layer(
                            layer_idx,
                            &hidden_f16,
                            prev_mlp_out.as_ref(),
                            max_context_len,
                            block_size,
                        )?
                    };
                    hidden_f16 = residual;
                    prev_mlp_out = Some(mlp_out);
                }
            } else {
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let input = GpuLayerInput {
                    hidden_states: &hidden_f16,
                    positions: packed_buf.slice(offsets.positions..offsets.positions + offsets.num_positions),
                    key_cache,
                    value_cache,
                    block_tables: packed_buf.slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                    context_lens: packed_buf.slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                    slot_mapping: packed_buf.slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: packed_buf.slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                };
                let weights = self.layer_weights(layer_idx)?;
                let (residual, mlp_out) = layer.forward(&input, &weights, &self.blas, prev_mlp_out.as_ref(), self.cublaslt_ref())?;
                hidden_f16 = residual;
                prev_mlp_out = Some(mlp_out);

                if debug_fwd && (layer_idx < 3 || layer_idx == num_layers - 1) {
                    let vals: Vec<f16> = self.stream.clone_dtoh(&hidden_f16)
                        .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                    let first5: Vec<f32> = vals.iter().take(5).map(|v| v.to_f32()).collect();
                    let has_nan = vals.iter().any(|v| v.to_f32().is_nan());
                    let has_inf = vals.iter().any(|v| v.to_f32().is_infinite());
                    let max_abs = vals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                    info!("DEBUG layer {layer_idx} residual: first5={first5:?} nan={has_nan} inf={has_inf} max={max_abs}");

                    let mvals: Vec<f16> = self.stream.clone_dtoh(prev_mlp_out.as_ref().unwrap())
                        .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                    let mfirst5: Vec<f32> = mvals.iter().take(5).map(|v| v.to_f32()).collect();
                    let mnan = mvals.iter().any(|v| v.to_f32().is_nan());
                    let mmax = mvals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                    info!("DEBUG layer {layer_idx} mlp_out: first5={mfirst5:?} nan={mnan} max={mmax}");
                }
            }
            } // end else (fallback multi-kernel path)

            // Final: fuse last layer's residual add with final RMSNorm
            let normed_f16 = if let Some(ref last_mlp) = prev_mlp_out {
                let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                    &self.stream, &self.loader,
                    &hidden_f16, last_mlp, &self.final_norm_weight,
                    self.rms_norm_eps, num_tokens, hidden_size,
                )?;
                n
            } else {
                self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
            };

            if debug_fwd {
                // Check normed hidden and top logits
                let nvals: Vec<f16> = self.stream.clone_dtoh(&normed_f16)
                    .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                let nfirst5: Vec<f32> = nvals.iter().take(5).map(|v| v.to_f32()).collect();
                let nnan = nvals.iter().any(|v| v.to_f32().is_nan());
                let nmax = nvals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                info!("DEBUG normed: first5={nfirst5:?} nan={nnan} max={nmax}");

                // Compute full logits and find top-5
                let logits_dbg = CudaLinearLayer::forward_f16_in(
                    &normed_f16, &self.lm_head_weight, num_tokens, vocab_size, hidden_size,
                    &self.blas,
                )?;
                let logits_cpu: Vec<f32> = self.stream.clone_dtoh(&logits_dbg)
                    .map_err(|e| LLMError::GpuError(format!("debug logits dtoh: {e}")))?;
                // Find top-5 for last token
                let last_start = (num_tokens - 1) * vocab_size;
                let last_logits = &logits_cpu[last_start..last_start + vocab_size];
                let mut indexed: Vec<(usize, f32)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top5: Vec<(usize, f32)> = indexed[..5.min(indexed.len())].to_vec();
                info!("DEBUG top5_logits: {:?}", top5);
                info!("DEBUG logits range: min={:.2} max={:.2} mean={:.4}",
                    last_logits.iter().cloned().fold(f32::INFINITY, f32::min),
                    last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    last_logits.iter().sum::<f32>() / last_logits.len() as f32,
                );
            }

            // LM head + argmax: f16 hidden -> fused argmax
            if num_tokens == 1 && greedy_only {
                let token_ids_gpu = self.gpu_fused_lm_head_argmax_f16_hidden(
                    &normed_f16, &self.lm_head_weight, vocab_size, hidden_size)?;
                let token_ids_cpu = self.stream.clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head token DtoH: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Full logits path: hgemm f16 hidden x f16 lm_head -> f32 logits
            let logits_gpu = CudaLinearLayer::forward_f16_in(
                &normed_f16, &self.lm_head_weight, num_tokens, vocab_size, hidden_size,
                &self.blas,
            )?;

            if greedy_only {
                let token_ids_gpu = self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?;
                let token_ids_cpu = self.stream.clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("argmax DtoH: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            let logits_cpu = self.stream.clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;
            Ok(ForwardOutput::Logits(logits_cpu))
        }

        /// Upload all per-step metadata into persistent GPU buffers.
        ///
        /// This MUST be called before `forward_graph_body` (or before replaying
        /// a captured CUDA graph). The memcpy_htod calls update the data at
        /// stable GPU pointers that the graph's kernels will read.
        pub fn upload_metadata(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
        ) -> Result<()> {
            let num_tokens = token_ids.len();
            let num_seqs = attn_meta.context_lens.len();
            let max_blocks = self.graph_max_blocks;

            let mut scratch = self.cpu_scratch.borrow_mut();
            scratch.clear();

            // Pack all 6 metadata fields contiguously, recording offsets.
            let token_ids_off = scratch.len();
            scratch.extend(token_ids.iter().map(|&t| t as i32));
            let num_token_ids = scratch.len() - token_ids_off;

            let positions_off = scratch.len();
            scratch.extend(positions.iter().map(|&p| p as i32));
            let num_positions = scratch.len() - positions_off;

            let context_lens_off = scratch.len();
            scratch.extend(attn_meta.context_lens.iter().map(|&c| c as i32));
            let num_context_lens = scratch.len() - context_lens_off;

            // block_tables: [num_seqs, graph_max_blocks], zero-padded.
            let block_tables_off = scratch.len();
            let bt_len = num_seqs * max_blocks;
            let new_len = scratch.len() + bt_len;
            scratch.resize(new_len, 0i32);
            for (s, row) in attn_meta.block_tables.iter().enumerate() {
                for (b, &blk) in row.iter().enumerate() {
                    scratch[block_tables_off + s * max_blocks + b] = blk as i32;
                }
            }

            let slot_mapping_off = scratch.len();
            scratch.extend(attn_meta.slot_mapping.iter().map(|&s| s as i32));
            let num_slot_mapping = scratch.len() - slot_mapping_off;

            let seq_start_pos_off = scratch.len();
            let mut pos = 0i32;
            for &ql in &attn_meta.query_lens {
                scratch.push(pos);
                pos += ql as i32;
            }
            scratch.push(num_tokens as i32);
            let num_seq_start_pos = scratch.len() - seq_start_pos_off;

            // Single packed upload (1 memcpy_htod instead of 6).
            self.meta_packed.borrow_mut().upload(&scratch, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("packed metadata HtoD: {e}")))?;

            self.meta_packed_offsets.set(PackedMetaOffsets {
                token_ids: token_ids_off,
                positions: positions_off,
                context_lens: context_lens_off,
                block_tables: block_tables_off,
                slot_mapping: slot_mapping_off,
                seq_start_pos: seq_start_pos_off,
                num_token_ids,
                num_positions,
                num_context_lens,
                num_block_tables: bt_len,
                num_slot_mapping,
                num_seq_start_pos,
            });

            Ok(())
        }

        /// Upload metadata padded to `padded_batch` tokens.
        /// Extra slots are filled with dummy data (duplicating seq 0's metadata).
        pub fn upload_metadata_padded(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            padded_batch: usize,
        ) -> Result<()> {
            let actual = token_ids.len();
            if actual >= padded_batch {
                return self.upload_metadata(token_ids, positions, attn_meta);
            }
            let pad_count = padded_batch - actual;

            // Pad token_ids with 0 (padding token)
            let mut padded_tokens: Vec<u32> = token_ids.to_vec();
            padded_tokens.resize(padded_batch, 0);

            // Pad positions with 0
            let mut padded_positions: Vec<u32> = positions.to_vec();
            padded_positions.resize(padded_batch, 0);

            // Pad attention metadata
            let mut padded_meta = attn_meta.clone();
            let dummy_ctx = if attn_meta.context_lens.is_empty() { 1 } else { attn_meta.context_lens[0] };
            padded_meta.context_lens.resize(padded_batch, dummy_ctx);
            padded_meta.query_lens.resize(padded_batch, 1);
            let dummy_bt = if attn_meta.block_tables.is_empty() { vec![] } else { attn_meta.block_tables[0].clone() };
            for _ in 0..pad_count {
                padded_meta.block_tables.push(dummy_bt.clone());
            }
            padded_meta.slot_mapping.resize(padded_batch, 0);

            self.upload_metadata(&padded_tokens, &padded_positions, &padded_meta)
        }

        /// Run the forward pass using already-uploaded metadata buffers.
        ///
        /// Call `upload_metadata()` first. This method does NOT upload metadata --
        /// it reads from the persistent GPU buffers populated by upload_metadata.
        /// This separation is required for CUDA graph capture: the uploads happen
        /// outside the graph, the forward pass is captured into the graph.
        pub fn forward_graph_body(
            &self,
            num_tokens: usize,
            num_seqs: usize,
            max_context_len: u32,
            is_prefill: bool,
            greedy_only: bool,
        ) -> Result<ForwardOutput> {
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            let mut hidden_f16 = self.embedding_lookup_from_meta(num_tokens)?;

            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            let use_persistent = num_tokens == 1 && !is_prefill && self.has_persistent_layer();
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;

            if use_persistent {
                for layer_idx in 0..num_layers {
                    let (residual, mlp_out) = unsafe {
                        self.forward_persistent_layer(
                            layer_idx,
                            &hidden_f16,
                            prev_mlp_out.as_ref(),
                            max_context_len,
                            block_size,
                        )?
                    };
                    hidden_f16 = residual;
                    prev_mlp_out = Some(mlp_out);
                }
            } else {
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let input = GpuLayerInput {
                    hidden_states: &hidden_f16,
                    positions: packed_buf.slice(offsets.positions..offsets.positions + offsets.num_positions),
                    key_cache,
                    value_cache,
                    block_tables: packed_buf.slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                    context_lens: packed_buf.slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                    slot_mapping: packed_buf.slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: packed_buf.slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                };
                let weights = self.layer_weights(layer_idx)?;
                let (residual, mlp_out) = layer.forward(&input, &weights, &self.blas, prev_mlp_out.as_ref(), self.cublaslt_ref())?;
                hidden_f16 = residual;
                prev_mlp_out = Some(mlp_out);
            }
            } // end else (fallback multi-kernel path)

            // Final: fuse last layer's residual add with final RMSNorm
            let normed_f16 = if let Some(ref last_mlp) = prev_mlp_out {
                let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                    &self.stream, &self.loader,
                    &hidden_f16, last_mlp, &self.final_norm_weight,
                    self.rms_norm_eps, num_tokens, hidden_size,
                )?;
                n
            } else {
                self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
            };

            // LM head + argmax: f16 hidden -> fused argmax
            if num_tokens == 1 && greedy_only {
                let token_ids_gpu = self.gpu_fused_lm_head_argmax_f16_hidden(
                    &normed_f16, &self.lm_head_weight, vocab_size, hidden_size)?;
                let token_ids_cpu = self.stream.clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head token DtoH: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Full logits path: hgemm f16 hidden x f16 lm_head -> f32 logits
            let logits_gpu = CudaLinearLayer::forward_f16_in(
                &normed_f16, &self.lm_head_weight, num_tokens, vocab_size, hidden_size,
                &self.blas,
            )?;

            if greedy_only {
                let token_ids_gpu = self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?;
                let token_ids_cpu = self.stream.clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("argmax DtoH: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            let logits_cpu = self.stream.clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;
            Ok(ForwardOutput::Logits(logits_cpu))
        }

        /// Get a reference to the CUDA stream used by this runner.
        pub fn cuda_stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }

        /// Get cublasLt reference (None when feature is off).
        fn cublaslt_ref(&self) -> Option<&crate::CublasLtRef> {
            #[cfg(feature = "cublaslt")]
            { self.blas_lt.as_ref() }
            #[cfg(not(feature = "cublaslt"))]
            { None }
        }


        /// Prepare the runner for CUDA graph capture.
        ///
        /// Pre-allocates the cuBLAS workspace (required by cuBLAS for graph
        /// capture) and ensures the stream has async alloc support.
        pub fn prepare_for_graph_capture(&mut self) -> rvllm_core::error::Result<()> {
            self.blas.prepare_for_graph_capture()?;
            if !self.stream.context().has_async_alloc() {
                tracing::warn!(
                    "GPU does not support async memory allocation (cuMemAllocAsync). \
                     CUDA graph capture may fail. Consider upgrading the CUDA driver."
                );
            }
            // Pre-allocate the packed metadata buffer large enough for max batch size
            // so it never reallocates (which would invalidate graph-captured pointers).
            // Layout per step: token_ids(N) + positions(N) + context_lens(N) +
            //   block_tables(N * graph_max_blocks) + slot_mapping(N) + seq_start_pos(N+1)
            let max_seqs = 256usize; // must match max_num_seqs
            let max_meta = max_seqs * (1 + 1 + 1 + self.graph_max_blocks + 1 + 1) + 1;
            let mut meta = self.meta_packed.borrow_mut();
            let dummy = vec![0i32; max_meta];
            meta.upload(&dummy, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("pre-alloc meta: {e}")))?;
            info!(max_meta, "pre-allocated packed metadata buffer for graph stability");
            // Pre-allocate graph output buffer too (same reason -- stable pointer).
            {
                let mut out = self.graph_output.borrow_mut();
                *out = Some(self.stream.alloc_zeros::<i32>(max_seqs)
                    .map_err(|e| LLMError::GpuError(format!("pre-alloc graph_output: {e}")))?);
            }
            Ok(())
        }

        /// GPU-only forward pass for CUDA graph capture.
        ///
        /// Runs the full forward pass (embedding -> layers -> norm -> argmax)
        /// writing the argmax result into the persistent `graph_output` buffer.
        /// Does NOT do any DtoH copy, so this is safe to capture in a CUDA graph.
        ///
        /// Call `upload_metadata()` first. After this, call `read_graph_output()`
        /// to get the host-side token IDs.
        pub fn forward_gpu_only(
            &self,
            num_tokens: usize,
            num_seqs: usize,
            max_context_len: u32,
            is_prefill: bool,
        ) -> Result<()> {
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            let mut hidden_f16 = self.embedding_lookup_from_meta(num_tokens)?;

            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            // NOTE: No profiling in forward_gpu_only -- this is captured into CUDA graphs.
            // stream.synchronize() during capture would invalidate the graph.
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let input = GpuLayerInput {
                    hidden_states: &hidden_f16,
                    positions: packed_buf.slice(offsets.positions..offsets.positions + offsets.num_positions),
                    key_cache,
                    value_cache,
                    block_tables: packed_buf.slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                    context_lens: packed_buf.slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                    slot_mapping: packed_buf.slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: packed_buf.slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                };
                let weights = self.layer_weights(layer_idx)?;
                let (residual, mlp_out) = layer.forward(&input, &weights, &self.blas, prev_mlp_out.as_ref(), self.cublaslt_ref())?;
                hidden_f16 = residual;
                prev_mlp_out = Some(mlp_out);
            }

            // Final: fuse last layer's residual add with final RMSNorm
            let normed_f16 = if let Some(ref last_mlp) = prev_mlp_out {
                let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                    &self.stream, &self.loader,
                    &hidden_f16, last_mlp, &self.final_norm_weight,
                    self.rms_norm_eps, num_tokens, hidden_size,
                )?;
                n
            } else {
                self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
            };

            let token_ids_gpu = if num_tokens == 1 {
                self.gpu_fused_lm_head_argmax_f16_hidden(
                    &normed_f16, &self.lm_head_weight, vocab_size, hidden_size)?
            } else {
                // Multi-token: hgemm f16 hidden x f16 lm_head -> f32 logits -> argmax
                let logits_gpu = CudaLinearLayer::forward_f16_in(
                    &normed_f16, &self.lm_head_weight, num_tokens, vocab_size, hidden_size,
                    &self.blas,
                )?;
                self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?
            };

            // Copy argmax result into the persistent output buffer.
            // On first call this allocates; on subsequent calls with the same
            // num_tokens it reuses the same GPU pointer (crucial for graph replay).
            let mut out = self.graph_output.borrow_mut();
            let need = num_tokens;
            let have = out.as_ref().map_or(0, |b| b.len());
            if have < need {
                *out = Some(self.stream.alloc_zeros::<i32>(need)
                    .map_err(|e| LLMError::GpuError(format!("graph_output alloc: {e}")))?);
            }
            let dst = out.as_mut().unwrap();
            self.stream.memcpy_dtod(&token_ids_gpu, dst)
                .map_err(|e| LLMError::GpuError(format!("graph_output dtod: {e}")))?;

            Ok(())
        }

        /// Read the argmax token IDs from the persistent graph output buffer.
        ///
        /// Call after `forward_gpu_only()` or after replaying a CUDA graph.
        /// This performs a DtoH copy (outside the graph).
        pub fn read_graph_output(&self, num_tokens: usize) -> Result<Vec<i32>> {
            let out = self.graph_output.borrow();
            let buf = out.as_ref().ok_or_else(|| {
                LLMError::GpuError("graph_output not populated -- call forward_gpu_only first".into())
            })?;
            // Copy only the needed elements
            let full = self.stream.clone_dtoh(buf)
                .map_err(|e| LLMError::GpuError(format!("graph_output DtoH: {e}")))?;
            Ok(full[..num_tokens].to_vec())
        }

        /// Enqueue an async DtoH copy of graph output into a pinned host buffer.
        ///
        /// Unlike `read_graph_output`, this does NOT synchronize the stream.
        /// The caller must call `sync_stream()` before reading from `dst`.
        /// `dst` MUST be pinned host memory for truly async behavior; with
        /// pageable memory cuMemcpyDtoHAsync degrades to synchronous.
        pub fn read_graph_output_async(
            &self,
            num_tokens: usize,
            dst: &mut [i32],
        ) -> Result<()> {
            let out = self.graph_output.borrow();
            let buf = out.as_ref().ok_or_else(|| {
                LLMError::GpuError("graph_output not populated -- call forward_gpu_only first".into())
            })?;
            // Only copy num_tokens elements (not the full padded buffer)
            let src_view = buf.slice(..num_tokens);
            self.stream.memcpy_dtoh(&src_view, &mut dst[..num_tokens])
                .map_err(|e| LLMError::GpuError(format!("graph_output async DtoH: {e}")))?;
            Ok(())
        }

        /// Synchronize the runner's CUDA stream, blocking until all enqueued
        /// work (graph replay + async DtoH) completes.
        pub fn sync_stream(&self) -> Result<()> {
            self.stream.synchronize()
                .map_err(|e| LLMError::GpuError(format!("stream sync: {e}")))?;
            Ok(())
        }

        /// Launch argmax kernel on GPU, returning [num_tokens] i32 token IDs.
        fn gpu_argmax(
            &self,
            logits_gpu: &CudaSlice<f32>,
            num_tokens: usize,
            vocab_size: usize,
        ) -> Result<CudaSlice<i32>> {
            let kernel = self
                .loader
                .get_func("argmax", "argmax_kernel")?;

            let output: CudaSlice<i32> = self
                .stream
                .alloc_zeros::<i32>(num_tokens)
                .map_err(|e| LLMError::GpuError(format!("argmax alloc: {e}")))?;

            let block_dim = vocab_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.stream
                    .launch_builder(&kernel)
                    .arg(logits_gpu)
                    .arg(&output)
                    .arg(&(vocab_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("argmax_kernel launch: {e}")))?;
            }

            Ok(output)
        }

        /// Fused LM-head matvec + argmax for single-token greedy decode (f32 weights).
        /// Skips materializing the full [vocab_size] logits tensor entirely.
        fn gpu_fused_lm_head_argmax(
            &self,
            hidden_state: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            vocab_size: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<i32>> {
            let pass1 = self
                .loader
                .get_func("fused_lm_head_argmax", "fused_lm_head_argmax_kernel")?;
            let pass2 = self
                .loader
                .get_func("fused_lm_head_argmax", "fused_lm_head_argmax_reduce_kernel")?;

            let num_blocks = (vocab_size + 255) / 256;

            let partial_val: CudaSlice<f32> = self
                .stream
                .alloc_zeros::<f32>(num_blocks)
                .map_err(|e| LLMError::GpuError(format!("fused_lm_head partial_val alloc: {e}")))?;
            let partial_idx: CudaSlice<i32> = self
                .stream
                .alloc_zeros::<i32>(num_blocks)
                .map_err(|e| LLMError::GpuError(format!("fused_lm_head partial_idx alloc: {e}")))?;
            let output: CudaSlice<i32> = self
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| LLMError::GpuError(format!("fused_lm_head output alloc: {e}")))?;

            // Pass 1: per-block dot + local argmax
            let cfg1 = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (hidden_size * std::mem::size_of::<f32>()) as u32,
            };
            unsafe {
                self.stream
                    .launch_builder(&pass1)
                    .arg(weight)
                    .arg(hidden_state)
                    .arg(&partial_val)
                    .arg(&partial_idx)
                    .arg(&(vocab_size as i32))
                    .arg(&(hidden_size as i32))
                    .launch(cfg1)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head_argmax_kernel launch: {e}")))?;
            }

            // Pass 2: reduce partials to single token ID
            let reduce_threads = num_blocks.min(1024) as u32;
            let cfg2 = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (reduce_threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&pass2)
                    .arg(&partial_val)
                    .arg(&partial_idx)
                    .arg(&output)
                    .arg(&(num_blocks as i32))
                    .launch(cfg2)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head_argmax_reduce_kernel launch: {e}")))?;
            }

            Ok(output)
        }

        /// Fused LM-head matvec + argmax for single-token greedy decode (f16 weights).
        fn gpu_fused_lm_head_argmax_f16(
            &self,
            hidden_state: &CudaSlice<f32>,
            weight: &CudaSlice<f16>,
            vocab_size: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<i32>> {
            let pass1 = self
                .loader
                .get_func("fused_lm_head_argmax_f16", "fused_lm_head_argmax_f16_kernel")?;
            let pass2 = self
                .loader
                .get_func("fused_lm_head_argmax", "fused_lm_head_argmax_reduce_kernel")?;

            let num_blocks = (vocab_size + 255) / 256;

            let partial_val: CudaSlice<f32> = self
                .stream
                .alloc_zeros::<f32>(num_blocks)
                .map_err(|e| LLMError::GpuError(format!("fused_lm_head_f16 partial_val alloc: {e}")))?;
            let partial_idx: CudaSlice<i32> = self
                .stream
                .alloc_zeros::<i32>(num_blocks)
                .map_err(|e| LLMError::GpuError(format!("fused_lm_head_f16 partial_idx alloc: {e}")))?;
            let output: CudaSlice<i32> = self
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| LLMError::GpuError(format!("fused_lm_head_f16 output alloc: {e}")))?;

            // Pass 1: per-block dot + local argmax (f16 weight, f32 hidden)
            let cfg1 = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (hidden_size * std::mem::size_of::<f32>()) as u32,
            };
            unsafe {
                self.stream
                    .launch_builder(&pass1)
                    .arg(weight)
                    .arg(hidden_state)
                    .arg(&partial_val)
                    .arg(&partial_idx)
                    .arg(&(vocab_size as i32))
                    .arg(&(hidden_size as i32))
                    .launch(cfg1)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head_argmax_f16_kernel launch: {e}")))?;
            }

            // Pass 2: reduce partials to single token ID
            let reduce_threads = num_blocks.min(1024) as u32;
            let cfg2 = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (reduce_threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&pass2)
                    .arg(&partial_val)
                    .arg(&partial_idx)
                    .arg(&output)
                    .arg(&(num_blocks as i32))
                    .launch(cfg2)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head_argmax_f16_reduce launch: {e}")))?;
            }

            Ok(output)
        }

        /// Per-layer weight references into the GPU weight map.
        fn layer_weights(&self, i: usize) -> Result<GpuLayerWeights<'_>> {
            let g = |name: &str| -> Result<&CudaSlice<f16>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            Ok(GpuLayerWeights {
                input_layernorm: g(&format!("model.layers.{i}.input_layernorm.weight"))?,
                q_proj: g(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj: g(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj: g(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj: g(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                post_attention_layernorm: g(&format!(
                    "model.layers.{i}.post_attention_layernorm.weight"
                ))?,
                gate_proj: g(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj: g(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj: g(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
                fused_qkv: self.fused_qkv_weights.get(i),
                fused_gate_up: self.fused_gate_up_weights.get(i),
                qkv_bias: self.fused_qkv_bias.get(i).and_then(|o| o.as_ref()),
            })
        }

        /// Embedding lookup using the pre-uploaded token IDs in the packed
        /// metadata buffer. Call upload_metadata() first to populate.
        fn embedding_lookup_from_meta(&self, num_tokens: usize) -> Result<CudaSlice<f16>> {
            let hidden_size = self.config.hidden_size;

            let kernel = self.loader
                .get_func("embedding_gather_f16", "embedding_gather_f16_kernel")?;

            // Safety: embedding gather kernel writes all num_tokens * hidden_size elements
            let output = unsafe { self.stream.alloc::<f16>(num_tokens * hidden_size) }
                .map_err(|e| LLMError::GpuError(format!("embed alloc: {e}")))?;

            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            let token_ids_view = packed_buf.slice(
                offsets.token_ids..offsets.token_ids + offsets.num_token_ids,
            );

            let block_dim = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.stream
                    .launch_builder(&kernel)
                    .arg(&output)
                    .arg(&self.embed_tokens)
                    .arg(&token_ids_view)
                    .arg(&(hidden_size as i32))
                    .arg(&(self.config.vocab_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("embedding_gather launch: {e}")))?;
            }

            Ok(output)
        }

        /// RMSNorm f16: f16 input, f16 weight, f16 output.
        fn rms_norm_f16_runner(
            &self,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            hidden_size: usize,
        ) -> Result<CudaSlice<f16>> {
            let num_tokens = input.len() / hidden_size;
            // Safety: rms_norm kernel writes all elements
            let mut output = unsafe { self.stream.alloc::<f16>(input.len()) }
                .map_err(|e| LLMError::GpuError(format!("rms_norm_f16 alloc: {e}")))?;
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = self.loader.get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                self.stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(weight)
                    .arg(&self.rms_norm_eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_f16 launch: {e}")))?;
            }
            Ok(output)
        }

        /// In-place RMSNorm f16: normalizes `input` directly, no output allocation.
        fn rms_norm_f16_inplace_runner(
            &self,
            input: &mut CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            hidden_size: usize,
        ) -> Result<()> {
            let num_tokens = input.len() / hidden_size;
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = self.loader.get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                let (raw_ptr, _guard) = DevicePtrMut::device_ptr_mut(input, &self.stream);
                self.stream.launch_builder(&kernel)
                    .arg(&raw_ptr)
                    .arg(&raw_ptr)
                    .arg(weight)
                    .arg(&self.rms_norm_eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_f16_inplace launch: {e}")))?;
            }
            Ok(())
        }

        /// Fused LM-head matvec + argmax for single-token greedy decode (f16 weights, f16 hidden).
        /// Casts f16 hidden -> f32 internally since the kernel expects f32 hidden.
        fn gpu_fused_lm_head_argmax_f16_hidden(
            &self,
            hidden_state_f16: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            vocab_size: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<i32>> {
            // Cast f16 hidden -> f32 for the LM head kernel
            let cast_kernel = self.loader.get_func("cast_fp", "cast_f16_to_f32_kernel")?;
            // Safety: cast kernel writes all hidden_size elements
            let mut hidden_f32 = unsafe { self.stream.alloc::<f32>(hidden_size) }
                .map_err(|e| LLMError::GpuError(format!("lm_head f16->f32 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((hidden_size as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&cast_kernel)
                    .arg(&mut hidden_f32)
                    .arg(hidden_state_f16)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("lm_head f16->f32 launch: {e}")))?;
            }
            self.gpu_fused_lm_head_argmax_f16(&hidden_f32, weight, vocab_size, hidden_size)
        }

        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }

        pub fn cache(&self) -> &CudaCacheEngine {
            &self.cache
        }

        pub fn cache_mut(&mut self) -> &mut CudaCacheEngine {
            &mut self.cache
        }

    }
}

// Re-export under cuda feature gate.
#[cfg(feature = "cuda")]
pub use cuda_impl::GpuModelRunner;

// =========================================================================
//  Mock-GPU stub (default feature)
// =========================================================================
#[cfg(not(feature = "cuda"))]
mod mock_impl {
    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;
    use super::ForwardOutput;

    /// Stub GpuModelRunner for non-CUDA builds.
    ///
    /// Allows downstream code to reference the type without conditional
    /// compilation everywhere. All methods return an error at runtime.
    pub struct GpuModelRunner {
        config: ModelRunnerConfig,
    }

    impl GpuModelRunner {
        /// Returns an error -- real CUDA is required.
        pub fn forward(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _attn_meta: &crate::bridge::AttentionMetadata,
            _is_prefill: bool,
        ) -> Result<Vec<f32>> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn forward_ex(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _attn_meta: &crate::bridge::AttentionMetadata,
            _is_prefill: bool,
            _greedy_only: bool,
        ) -> Result<ForwardOutput> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }

        pub fn upload_metadata(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _attn_meta: &crate::bridge::AttentionMetadata,
        ) -> Result<()> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn forward_graph_body(
            &self,
            _num_tokens: usize,
            _num_seqs: usize,
            _max_context_len: u32,
            _is_prefill: bool,
            _greedy_only: bool,
        ) -> Result<ForwardOutput> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn forward_gpu_only(
            &self,
            _num_tokens: usize,
            _num_seqs: usize,
            _max_context_len: u32,
            _is_prefill: bool,
        ) -> Result<()> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn read_graph_output(&self, _num_tokens: usize) -> Result<Vec<i32>> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn read_graph_output_async(&self, _dst: &mut [i32]) -> Result<()> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn sync_stream(&self) -> Result<()> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

    }
}

#[cfg(not(feature = "cuda"))]
pub use mock_impl::GpuModelRunner;

// =========================================================================
//  Tests (run under mock-gpu / default features)
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_runner_returns_error() {
        #[cfg(not(feature = "cuda"))]
        {
            let config = ModelRunnerConfig {
                num_layers: 2,
                hidden_size: 64,
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 16,
                intermediate_size: 128,
                vocab_size: 100,
                max_position: 512,
                rms_norm_eps: 1e-5, rope_theta: 10000.0,
                dtype: "float32".to_string(),
                architecture: "LlamaForCausalLM".to_string(),
            };
            let runner = GpuModelRunner { config };
            let result = runner.forward(&[1, 2, 3], &[0, 1, 2], &[], &[]);
            assert!(result.is_err());
            let err_msg = format!("{}", result.unwrap_err());
            assert!(err_msg.contains("cuda"));
        }
    }

    #[test]
    fn config_accessible() {
        #[cfg(not(feature = "cuda"))]
        {
            let config = ModelRunnerConfig {
                num_layers: 4,
                hidden_size: 256,
                num_heads: 8,
                num_kv_heads: 8,
                head_dim: 32,
                intermediate_size: 512,
                vocab_size: 32000,
                max_position: 2048,
                rms_norm_eps: 1e-5, rope_theta: 10000.0,
                dtype: "float16".to_string(),
                architecture: "LlamaForCausalLM".to_string(),
            };
            let runner = GpuModelRunner { config };
            assert_eq!(runner.config().num_layers, 4);
            assert_eq!(runner.config().vocab_size, 32000);
        }
    }
}
