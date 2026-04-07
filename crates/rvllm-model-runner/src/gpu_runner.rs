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

#[derive(Debug, Clone, Copy)]
pub struct DecodeExecutionPlan {
    pub actual_tokens: usize,
    pub graph_tokens: usize,
    pub use_graphed_decode: bool,
    pub use_batched_v2: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeGraphRuntimeState {
    pub graphs_enabled: bool,
    pub exact_graph_available: bool,
    pub warmup_complete: bool,
    pub capture_attempted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeGraphAction {
    Raw,
    Replay,
    Capture,
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeGraphDispatch {
    pub execution: DecodeExecutionPlan,
    pub action: DecodeGraphAction,
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::cell::RefCell;
    use std::sync::Arc;

    use std::cell::Cell;
    use std::time::Instant;

    use cudarc::driver::{
        CudaContext, CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut, LaunchConfig,
        PushKernelArg,
    };
    use half::f16;
    use tracing::{debug, info, trace};

    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

    use crate::gpu_layer::{
        BatchedLayerPhaseTimings, BatchedV2PolicyConfig, ForwardPath, GemmStrategy,
        GpuLayerConfig, GpuLayerInput, GpuLayerWeights, GpuTransformerLayer, LayerScratchRef,
    };
    use crate::layers::linear_cuda::CudaLinearLayer;

    use rvllm_gpu::kernel_loader::KernelLoader;
    use rvllm_gpu::prelude::CublasHandle;
    use rvllm_kv_cache::engine_cuda::CudaCacheEngine;
    use rvllm_model_loader::gpu_weights::GpuModelWeights;

    use super::{
        DecodeExecutionPlan, DecodeGraphAction, DecodeGraphDispatch, DecodeGraphRuntimeState,
        ForwardOutput,
    };

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

        fn ensure_len(
            &mut self,
            len: usize,
            stream: &Arc<CudaStream>,
        ) -> std::result::Result<(), cudarc::driver::result::DriverError> {
            let need = len.max(1);
            let have = self.buf.as_ref().map_or(0, |b| b.len());
            if have < need {
                let cap = need.max(have * 2).max(64);
                self.buf = Some(stream.alloc_zeros::<i32>(cap)?);
            }
            Ok(())
        }

        fn upload_range(
            &mut self,
            offset: usize,
            data: &[i32],
            stream: &Arc<CudaStream>,
        ) -> std::result::Result<(), cudarc::driver::result::DriverError> {
            let end = offset + data.len();
            self.ensure_len(end, stream)?;
            let mut dst = self
                .buf
                .as_mut()
                .expect("ensure_len() must populate buffer")
                .slice_mut(offset..end);
            stream.memcpy_htod(data, &mut dst)?;
            Ok(())
        }
    }

    #[derive(Default)]
    struct DecodeMetadataV2State {
        padded_batch: usize,
        token_ids: Vec<i32>,
        positions: Vec<i32>,
        context_lens: Vec<i32>,
        slot_mapping: Vec<i32>,
        seq_start_pos: Vec<i32>,
        block_tables: Vec<i32>,
    }

    impl DecodeMetadataV2State {
        fn resize(&mut self, padded_batch: usize, max_blocks: usize) {
            self.padded_batch = padded_batch;
            self.token_ids.resize(padded_batch, 0);
            self.positions.resize(padded_batch, 0);
            self.context_lens.resize(padded_batch, 0);
            self.slot_mapping.resize(padded_batch, 0);
            self.seq_start_pos.resize(padded_batch + 1, 0);
            self.block_tables.resize(padded_batch * max_blocks, 0);
        }
    }

    struct PersistentDecodeMetadataBuffers {
        token_ids: RefCell<ReusableGpuBuf>,
        positions: RefCell<ReusableGpuBuf>,
        context_lens: RefCell<ReusableGpuBuf>,
        block_tables: RefCell<ReusableGpuBuf>,
        slot_mapping: RefCell<ReusableGpuBuf>,
        seq_start_pos: RefCell<ReusableGpuBuf>,
    }

    impl PersistentDecodeMetadataBuffers {
        fn new() -> Self {
            Self {
                token_ids: RefCell::new(ReusableGpuBuf::new()),
                positions: RefCell::new(ReusableGpuBuf::new()),
                context_lens: RefCell::new(ReusableGpuBuf::new()),
                block_tables: RefCell::new(ReusableGpuBuf::new()),
                slot_mapping: RefCell::new(ReusableGpuBuf::new()),
                seq_start_pos: RefCell::new(ReusableGpuBuf::new()),
            }
        }

        fn ensure_len(
            &self,
            padded: usize,
            max_blocks: usize,
            stream: &Arc<CudaStream>,
        ) -> std::result::Result<(), cudarc::driver::result::DriverError> {
            self.token_ids.borrow_mut().ensure_len(padded, stream)?;
            self.positions.borrow_mut().ensure_len(padded, stream)?;
            self.context_lens.borrow_mut().ensure_len(padded, stream)?;
            self.block_tables
                .borrow_mut()
                .ensure_len(padded * max_blocks, stream)?;
            self.slot_mapping.borrow_mut().ensure_len(padded, stream)?;
            self.seq_start_pos.borrow_mut().ensure_len(padded + 1, stream)?;
            Ok(())
        }
    }

    /// Pre-allocated f16 scratch buffers for the forward pass.
    /// Sized for max_batch_tokens. Reused across all layers (sequential execution).
    pub struct F16LayerScratch {
        pub max_tokens: usize,
        pub qkv: CudaSlice<f16>,      // [max_tokens * qkv_dim]
        pub attn_out: CudaSlice<f16>, // [max_tokens * q_dim]
        pub o_proj: CudaSlice<f16>,   // [max_tokens * hidden]
        pub normed: CudaSlice<f16>,   // [max_tokens * hidden]
        pub gate_up: CudaSlice<f16>,  // [max_tokens * intermediate * 2]
        pub gateup_ws: CudaSlice<u8>, // CUTLASS gateup workspace
        pub silu_out: CudaSlice<f16>, // [max_tokens * intermediate]
        // Double-buffered: layer N writes to pair A, reads from pair B.
        // Layer N+1 writes to pair B, reads from pair A. Zero alloc, zero copy.
        pub residual_a: CudaSlice<f16>, // [max_tokens * hidden]
        pub down_a: CudaSlice<f16>,     // [max_tokens * hidden]
        pub residual_b: CudaSlice<f16>, // [max_tokens * hidden]
        pub down_b: CudaSlice<f16>,     // [max_tokens * hidden]
    }

    struct PersistentV3Scratch {
        max_splits: usize,
        qkv: CudaSlice<f16>,
        attn: CudaSlice<f16>,
        oproj: CudaSlice<f16>,
        gateup: CudaSlice<f16>,
        split_max: CudaSlice<u8>,
        split_sum: CudaSlice<u8>,
        split_acc: CudaSlice<u8>,
        sync_flags: CudaSlice<i32>,
        residual_a: CudaSlice<f16>,
        residual_b: CudaSlice<f16>,
        mlp_a: CudaSlice<f16>,
        mlp_b: CudaSlice<f16>,
    }

    const DEFAULT_F16_SCRATCH_TOKENS: usize = 512;

    struct DecodePhaseProfileSummary {
        target_bucket: usize,
        pre_attn_norm_ns: Vec<u64>,
        qkv_ns: Vec<u64>,
        rope_cache_ns: Vec<u64>,
        attn_ns: Vec<u64>,
        oproj_norm_ns: Vec<u64>,
        gateup_silu_ns: Vec<u64>,
        down_ns: Vec<u64>,
        final_norm_ns: u64,
        lm_head_ns: u64,
    }

    impl DecodePhaseProfileSummary {
        fn new(target_bucket: usize) -> Self {
            Self {
                target_bucket,
                pre_attn_norm_ns: Vec::new(),
                qkv_ns: Vec::new(),
                rope_cache_ns: Vec::new(),
                attn_ns: Vec::new(),
                oproj_norm_ns: Vec::new(),
                gateup_silu_ns: Vec::new(),
                down_ns: Vec::new(),
                final_norm_ns: 0,
                lm_head_ns: 0,
            }
        }

        fn observe_layer(&mut self, t: &BatchedLayerPhaseTimings) {
            self.pre_attn_norm_ns.push(t.pre_attn_norm.as_nanos() as u64);
            self.qkv_ns.push(t.qkv.as_nanos() as u64);
            self.rope_cache_ns.push(t.rope_cache.as_nanos() as u64);
            self.attn_ns.push(t.attn.as_nanos() as u64);
            self.oproj_norm_ns.push(t.oproj_norm.as_nanos() as u64);
            self.gateup_silu_ns.push(t.gateup_silu.as_nanos() as u64);
            self.down_ns.push(t.down.as_nanos() as u64);
        }

        fn median_us(values: &[u64]) -> f64 {
            if values.is_empty() {
                return 0.0;
            }
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            let mid = sorted.len() / 2;
            let ns = if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2
            } else {
                sorted[mid]
            };
            ns as f64 / 1_000.0
        }

        fn log(&self, batch: usize, num_layers: usize) {
            info!(
                target_bucket = self.target_bucket,
                batch,
                num_layers,
                pre_attn_norm_us = Self::median_us(&self.pre_attn_norm_ns),
                qkv_us = Self::median_us(&self.qkv_ns),
                rope_cache_us = Self::median_us(&self.rope_cache_ns),
                attn_us = Self::median_us(&self.attn_ns),
                oproj_norm_us = Self::median_us(&self.oproj_norm_ns),
                gateup_silu_us = Self::median_us(&self.gateup_silu_ns),
                down_us = Self::median_us(&self.down_ns),
                final_norm_us = self.final_norm_ns as f64 / 1_000.0,
                lm_head_us = self.lm_head_ns as f64 / 1_000.0,
                "decode phase profile"
            );
        }
    }

    fn forward_path_graph_capture_supported(path: ForwardPath) -> bool {
        matches!(
            path,
            ForwardPath::Fp8Decode
                | ForwardPath::FusedDecode
                | ForwardPath::PersistentDecode
                | ForwardPath::PersistentV3Decode
                | ForwardPath::CublasGemvDecode
                | ForwardPath::Batched
                | ForwardPath::BatchedV2
        )
    }

    #[derive(Debug, Clone, Copy)]
    struct ForwardExecutionPlan {
        path: ForwardPath,
        use_scratch: bool,
        graph_capture_supported: bool,
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
        cutlass: Option<Arc<rvllm_gpu::cutlass_ffi::CutlassKernels>>,
        persistent_v2: Option<Arc<rvllm_gpu::persistent_v2_ffi::PersistentV2Kernels>>,
        gemm_strategy: GemmStrategy,
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
        /// Host-side mirror for persistent V2 decode metadata updates.
        decode_meta_v2_state: RefCell<DecodeMetadataV2State>,
        /// Persistent device-side decode metadata state for BatchedV2.
        persistent_decode_meta: PersistentDecodeMetadataBuffers,
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
        /// FP8 quantized weights + per-row scales (when RVLLM_FP8_WEIGHTS=1).
        fp8_fused_qkv: Vec<CudaSlice<u8>>,
        fp8_fused_qkv_scale: Vec<CudaSlice<half::f16>>,
        fp8_o_proj: Vec<CudaSlice<u8>>,
        fp8_o_proj_scale: Vec<CudaSlice<half::f16>>,
        fp8_fused_gate_up: Vec<CudaSlice<u8>>,
        fp8_fused_gate_up_scale: Vec<CudaSlice<half::f16>>,
        fp8_down_proj: Vec<CudaSlice<u8>>,
        fp8_down_proj_scale: Vec<CudaSlice<half::f16>>,
        /// FP8 input scratch buffer for cublasLt FP8 GEMM (reused across layers).
        fp8_input_scratch: Option<CudaSlice<u8>>,
        /// Pre-allocated scratch buffers for the forward pass (RefCell for interior mutability).
        f16_scratch: RefCell<Option<F16LayerScratch>>,
        /// Pre-allocated batch-1 scratch for persistent v3 decode.
        persistent_v3_scratch: RefCell<Option<PersistentV3Scratch>>,
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

            let get_weight = |name: &str| {
                weights.get(name).or_else(|| {
                    name.strip_prefix("model.")
                        .and_then(|rest| weights.get(&format!("model.language_model.{rest}")))
                })
            };

            let embed_tokens = get_weight("model.embed_tokens.weight")
                .ok_or_else(|| LLMError::GpuError("missing model.embed_tokens.weight".into()))?
                .clone();

            let final_norm_weight = get_weight("model.norm.weight")
                .ok_or_else(|| LLMError::GpuError("missing model.norm.weight".into()))?
                .clone();

            let lm_head_weight = weights
                .get("lm_head.weight")
                .or_else(|| get_weight("model.embed_tokens.weight"))
                .ok_or_else(|| {
                    LLMError::GpuError(
                        "missing lm_head.weight and model.embed_tokens.weight".into(),
                    )
                })?
                .clone();

            let mut layers = Vec::with_capacity(config.num_layers);
            let batched_v2_policy = BatchedV2PolicyConfig {
                use_cutlass_qkv: std::env::var("RVLLM_V2_CUTLASS_QKV")
                    .map_or(false, |v| v == "1"),
                use_cutlass_oproj: std::env::var("RVLLM_V2_CUTLASS_OPROJ")
                    .map_or(false, |v| v == "1"),
                use_cutlass_gateup: std::env::var("RVLLM_V2_CUTLASS_GATEUP")
                    .map_or(false, |v| v == "1"),
                use_cutlass_gate_aux: std::env::var("RVLLM_V2_CUTLASS_GATE_AUX")
                    .map_or(false, |v| v == "1"),
            };
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
                layers.push(GpuTransformerLayer::new(
                    layer_cfg,
                    Arc::clone(&stream),
                    Arc::clone(&loader),
                    batched_v2_policy,
                ));
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
            let max_pos = config.max_position.min(32768);
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
            info!(
                graph_max_blocks,
                block_size,
                max_position = config.max_position,
                "fixed block_tables stride for CUDA graph stability"
            );

            #[cfg(feature = "cublaslt")]
            let blas_lt = rvllm_gpu::cublaslt_ops::CublasLtOps::new(stream.clone()).ok();

            // Load CUTLASS shared library (compiled by build_cutlass_so.sh).
            // Try arch-specific path first, then fallback paths.
            let cutlass = {
                use rvllm_gpu::cutlass_ffi::CutlassKernels;
                let candidates = [
                    "kernels/sm_90/libcutlass_kernels.so",
                    "kernels/sm_80/libcutlass_kernels.so",
                    "/usr/local/lib/libcutlass_kernels.so",
                ];
                let mut loaded = None;
                for path in &candidates {
                    let p = std::path::Path::new(path);
                    if p.exists() {
                        match CutlassKernels::load(p) {
                            Ok(k) => {
                                info!(?p, "CUTLASS shared library loaded");
                                loaded = Some(Arc::new(k));
                                break;
                            }
                            Err(e) => {
                                debug!(?p, error = %e, "CUTLASS .so found but failed to load");
                            }
                        }
                    }
                }
                if loaded.is_none() {
                    info!("CUTLASS .so not found");
                } else {
                    info!("CUTLASS kernels loaded");
                }
                loaded
            };

            let gemm_strategy = match std::env::var("RVLLM_BATCHED_GEMM_STRATEGY").ok().as_deref() {
                Some("cublas") => GemmStrategy::Cublas,
                Some("cutlass") if cutlass.is_some() => GemmStrategy::Cutlass,
                Some("cutlass") => {
                    info!(
                        "RVLLM_BATCHED_GEMM_STRATEGY=cutlass requested but CUTLASS is unavailable"
                    );
                    GemmStrategy::Cublas
                }
                Some("hybrid") if cutlass.is_some() => GemmStrategy::Hybrid,
                Some("hybrid") => {
                    info!(
                        "RVLLM_BATCHED_GEMM_STRATEGY=hybrid requested but CUTLASS is unavailable"
                    );
                    GemmStrategy::Cublas
                }
                None => GemmStrategy::Cublas,
                Some(other) => {
                    info!(
                        value = other,
                        "unknown RVLLM_BATCHED_GEMM_STRATEGY, defaulting"
                    );
                    GemmStrategy::Cublas
                }
            };

            match gemm_strategy {
                GemmStrategy::Hybrid => info!("Using batched GEMM policy: cuBLAS/cublasLt for QKV/O-proj/down, CUTLASS for gateup+silu"),
                GemmStrategy::Cutlass => info!("Using batched GEMM policy: CUTLASS for QKV/O-proj/gateup, cuBLAS/cublasLt for down"),
                GemmStrategy::Cublas => info!("Using batched GEMM policy: cuBLAS/cublasLt for all batched GEMMs"),
            }

            // Load persistent_v2 cubins (cooperative TC GEMV + split-KV kernels).
            let persistent_v2 = {
                use rvllm_gpu::persistent_v2_ffi::PersistentV2Kernels;
                match PersistentV2Kernels::load(device.clone()) {
                    Ok(k) => {
                        info!(
                            layer = k.has_layer_kernel(),
                            mega = k.has_mega_kernel(),
                            "persistent_v2 cubins loaded"
                        );
                        Some(Arc::new(k))
                    }
                    Err(e) => {
                        debug!("persistent_v2 cubins not found: {e}");
                        None
                    }
                }
            };

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
                gemm_strategy,
                cutlass,
                persistent_v2,
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
                decode_meta_v2_state: RefCell::new(DecodeMetadataV2State::default()),
                persistent_decode_meta: PersistentDecodeMetadataBuffers::new(),
                graph_max_blocks,
                fused_qkv_weights: Vec::new(),
                fused_gate_up_weights: Vec::new(),
                fused_qkv_bias: Vec::new(),
                fp8_fused_qkv: Vec::new(),
                fp8_fused_qkv_scale: Vec::new(),
                fp8_o_proj: Vec::new(),
                fp8_o_proj_scale: Vec::new(),
                fp8_fused_gate_up: Vec::new(),
                fp8_fused_gate_up_scale: Vec::new(),
                fp8_down_proj: Vec::new(),
                fp8_down_proj_scale: Vec::new(),
                fp8_input_scratch: None,
                f16_scratch: RefCell::new(None),
                persistent_v3_scratch: RefCell::new(None),
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
                let q_w = self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.q_proj.weight"))
                    .ok_or_else(|| LLMError::GpuError(format!("missing q_proj layer {i}")))?;
                let k_w = self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.k_proj.weight"))
                    .ok_or_else(|| LLMError::GpuError(format!("missing k_proj layer {i}")))?;
                let v_w = self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.v_proj.weight"))
                    .ok_or_else(|| LLMError::GpuError(format!("missing v_proj layer {i}")))?;

                let mut fused_qkv = self
                    .stream
                    .alloc_zeros::<half::f16>(qkv_dim * hidden)
                    .map_err(|e| LLMError::GpuError(format!("fused_qkv alloc: {e}")))?;
                self.stream
                    .memcpy_dtod(q_w, &mut fused_qkv.slice_mut(..q_dim * hidden))
                    .map_err(|e| LLMError::GpuError(format!("fused_qkv q copy: {e}")))?;
                self.stream
                    .memcpy_dtod(
                        k_w,
                        &mut fused_qkv.slice_mut(q_dim * hidden..(q_dim + kv_dim) * hidden),
                    )
                    .map_err(|e| LLMError::GpuError(format!("fused_qkv k copy: {e}")))?;
                self.stream
                    .memcpy_dtod(
                        v_w,
                        &mut fused_qkv.slice_mut((q_dim + kv_dim) * hidden..qkv_dim * hidden),
                    )
                    .map_err(|e| LLMError::GpuError(format!("fused_qkv v copy: {e}")))?;
                self.fused_qkv_weights.push(fused_qkv);

                // Fuse gate+up: concat [intermediate, hidden] + [intermediate, hidden]
                let gate_w = self
                    .weights
                    .get(&format!("model.layers.{i}.mlp.gate_proj.weight"))
                    .ok_or_else(|| LLMError::GpuError(format!("missing gate_proj layer {i}")))?;
                let up_w = self
                    .weights
                    .get(&format!("model.layers.{i}.mlp.up_proj.weight"))
                    .ok_or_else(|| LLMError::GpuError(format!("missing up_proj layer {i}")))?;

                let mut fused_gu = self
                    .stream
                    .alloc_zeros::<half::f16>(gate_up_dim * hidden)
                    .map_err(|e| LLMError::GpuError(format!("fused_gate_up alloc: {e}")))?;
                self.stream
                    .memcpy_dtod(gate_w, &mut fused_gu.slice_mut(..intermediate * hidden))
                    .map_err(|e| LLMError::GpuError(format!("fused_gate_up gate copy: {e}")))?;
                self.stream
                    .memcpy_dtod(
                        up_w,
                        &mut fused_gu.slice_mut(intermediate * hidden..gate_up_dim * hidden),
                    )
                    .map_err(|e| LLMError::GpuError(format!("fused_gate_up up copy: {e}")))?;
                self.fused_gate_up_weights.push(fused_gu);

                // Fuse QKV biases: concat q_bias || k_bias || v_bias (already f16)
                let q_bias = self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.q_proj.bias"));
                let k_bias = self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.k_proj.bias"));
                let v_bias = self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.v_proj.bias"));
                if let (Some(qb), Some(kb), Some(vb)) = (q_bias, k_bias, v_bias) {
                    let mut fused_bias = self
                        .stream
                        .alloc_zeros::<half::f16>(qkv_dim)
                        .map_err(|e| LLMError::GpuError(format!("fused_bias alloc: {e}")))?;
                    self.stream
                        .memcpy_dtod(qb, &mut fused_bias.slice_mut(..q_dim))
                        .map_err(|e| LLMError::GpuError(format!("fused_bias q copy: {e}")))?;
                    self.stream
                        .memcpy_dtod(kb, &mut fused_bias.slice_mut(q_dim..q_dim + kv_dim))
                        .map_err(|e| LLMError::GpuError(format!("fused_bias k copy: {e}")))?;
                    self.stream
                        .memcpy_dtod(vb, &mut fused_bias.slice_mut(q_dim + kv_dim..qkv_dim))
                        .map_err(|e| LLMError::GpuError(format!("fused_bias v copy: {e}")))?;
                    self.fused_qkv_bias.push(Some(fused_bias));
                } else {
                    self.fused_qkv_bias.push(None);
                }
            }

            info!(
                num_layers,
                qkv_dim, gate_up_dim, "fused QKV and gate+up weights (f16)"
            );

            // FP8 weight quantization for ALL projection weights (when RVLLM_FP8_WEIGHTS=1)
            // Note: FP8 only helps for M=1 decode (bandwidth-bound GEMV). For batched
            // decode (M>=8), f16 tensor cores already saturate compute and FP8 adds
            // cast overhead with no throughput gain. Use for latency-sensitive single-
            // stream workloads, not high-throughput batch serving.
            if std::env::var("RVLLM_FP8_WEIGHTS").map_or(false, |v| v == "1") {
                use rvllm_gpu::fp8_quantize::quantize_weight_fp8;
                info!("quantizing ALL weights to FP8 E4M3 (per-row scales)...");
                tracing::warn!("FP8 weights: improves single-stream decode latency but does NOT improve batched throughput. For high-concurrency serving, f16 is equivalent or faster.");

                let q_dim = self.config.num_heads * self.config.head_dim;
                let intermediate = self.config.intermediate_size;
                let gate_up_dim = intermediate * 2;

                for i in 0..num_layers {
                    // QKV: [qkv_dim, hidden]
                    let mut host = vec![half::f16::ZERO; self.fused_qkv_weights[i].len()];
                    self.stream
                        .memcpy_dtoh(&self.fused_qkv_weights[i], &mut host)
                        .map_err(|e| LLMError::GpuError(format!("fp8 dtoh qkv: {e}")))?;
                    let q = quantize_weight_fp8(&host, qkv_dim, hidden);
                    let mut fp8 = unsafe { self.stream.alloc::<u8>(q.data.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 alloc qkv: {e}")))?;
                    self.stream
                        .memcpy_htod(&q.data, &mut fp8)
                        .map_err(|e| LLMError::GpuError(format!("fp8 htod qkv: {e}")))?;
                    let mut sc = unsafe { self.stream.alloc::<half::f16>(q.scales.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale qkv: {e}")))?;
                    self.stream
                        .memcpy_htod(&q.scales, &mut sc)
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale htod qkv: {e}")))?;
                    self.fp8_fused_qkv.push(fp8);
                    self.fp8_fused_qkv_scale.push(sc);

                    // O-proj: [hidden, q_dim]
                    let o_name = format!("model.layers.{i}.self_attn.o_proj.weight");
                    let o_w = self
                        .weights
                        .get(&o_name)
                        .ok_or_else(|| LLMError::GpuError(format!("missing {o_name}")))?;
                    let mut host = vec![half::f16::ZERO; o_w.len()];
                    self.stream
                        .memcpy_dtoh(o_w, &mut host)
                        .map_err(|e| LLMError::GpuError(format!("fp8 dtoh o: {e}")))?;
                    let q = quantize_weight_fp8(&host, hidden, q_dim);
                    let mut fp8 = unsafe { self.stream.alloc::<u8>(q.data.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 alloc o: {e}")))?;
                    self.stream
                        .memcpy_htod(&q.data, &mut fp8)
                        .map_err(|e| LLMError::GpuError(format!("fp8 htod o: {e}")))?;
                    let mut sc = unsafe { self.stream.alloc::<half::f16>(q.scales.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale o: {e}")))?;
                    self.stream
                        .memcpy_htod(&q.scales, &mut sc)
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale htod o: {e}")))?;
                    self.fp8_o_proj.push(fp8);
                    self.fp8_o_proj_scale.push(sc);

                    // Gate+up: [gate_up_dim, hidden]
                    let mut host = vec![half::f16::ZERO; self.fused_gate_up_weights[i].len()];
                    self.stream
                        .memcpy_dtoh(&self.fused_gate_up_weights[i], &mut host)
                        .map_err(|e| LLMError::GpuError(format!("fp8 dtoh gu: {e}")))?;
                    let q = quantize_weight_fp8(&host, gate_up_dim, hidden);
                    let mut fp8 = unsafe { self.stream.alloc::<u8>(q.data.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 alloc gu: {e}")))?;
                    self.stream
                        .memcpy_htod(&q.data, &mut fp8)
                        .map_err(|e| LLMError::GpuError(format!("fp8 htod gu: {e}")))?;
                    let mut sc = unsafe { self.stream.alloc::<half::f16>(q.scales.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale gu: {e}")))?;
                    self.stream
                        .memcpy_htod(&q.scales, &mut sc)
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale htod gu: {e}")))?;
                    self.fp8_fused_gate_up.push(fp8);
                    self.fp8_fused_gate_up_scale.push(sc);

                    // Down: [hidden, intermediate]
                    let down_name = format!("model.layers.{i}.mlp.down_proj.weight");
                    let down_w = self
                        .weights
                        .get(&down_name)
                        .ok_or_else(|| LLMError::GpuError(format!("missing {down_name}")))?;
                    let mut host = vec![half::f16::ZERO; down_w.len()];
                    self.stream
                        .memcpy_dtoh(down_w, &mut host)
                        .map_err(|e| LLMError::GpuError(format!("fp8 dtoh down: {e}")))?;
                    let q = quantize_weight_fp8(&host, hidden, intermediate);
                    let mut fp8 = unsafe { self.stream.alloc::<u8>(q.data.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 alloc down: {e}")))?;
                    self.stream
                        .memcpy_htod(&q.data, &mut fp8)
                        .map_err(|e| LLMError::GpuError(format!("fp8 htod down: {e}")))?;
                    let mut sc = unsafe { self.stream.alloc::<half::f16>(q.scales.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale down: {e}")))?;
                    self.stream
                        .memcpy_htod(&q.scales, &mut sc)
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale htod down: {e}")))?;
                    self.fp8_down_proj.push(fp8);
                    self.fp8_down_proj_scale.push(sc);
                }
                // Allocate FP8 input scratch (max of all input dimensions)
                let max_k = *[hidden, q_dim, intermediate].iter().max().unwrap();
                self.fp8_input_scratch = Some(
                    unsafe { self.stream.alloc::<u8>(max_k) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 input scratch: {e}")))?,
                );
                info!(
                    num_layers,
                    "FP8 weight quantization complete (all projections)"
                );
            }

            // Autotune cublasLt algorithms now that weights are loaded and GPU
            // memory is settled. Scratch buffers (~10MB each) are dropped after.
            #[cfg(feature = "cublaslt")]
            if let Some(ref lt) = self.blas_lt {
                let gpu_name = rvllm_gpu::prelude::list_devices()
                    .first()
                    .map(|d| d.name.clone())
                    .unwrap_or_else(|| "unknown".into());
                match rvllm_gpu::CublasAutotuner::autotune_model(
                    lt,
                    rvllm_gpu::GemmDtype::F16,
                    hidden,
                    q_dim,
                    qkv_dim,
                    intermediate,
                    gate_up_dim,
                    &gpu_name,
                ) {
                    Ok(tuner) => {
                        info!(
                            shapes = tuner.len(),
                            max_ws = tuner.max_workspace_size(),
                            "cublasLt autotuning complete, installing into hot path"
                        );
                        lt.install_autotuned(&tuner);
                    }
                    Err(e) => {
                        tracing::warn!(%e, "cublasLt autotuning failed, using heuristics");
                    }
                }
            }

            self.alloc_scratch(DEFAULT_F16_SCRATCH_TOKENS)?;
            Ok(())
        }

        /// Pre-allocate a reusable set of f16 scratch buffers for the forward pass.
        /// Sized for max padded batch tokens. Since all layers are processed
        /// sequentially, one set of buffers covers every layer.
        fn alloc_scratch(&self, max_tokens: usize) -> Result<()> {
            let hidden = self.config.hidden_size;
            let q_dim = self.config.num_heads * self.config.head_dim;
            let kv_dim = self.config.num_kv_heads * self.config.head_dim;
            let qkv_dim = q_dim + kv_dim + kv_dim;
            let intermediate = self.config.intermediate_size;

            let alloc = |n: usize| -> Result<CudaSlice<f16>> {
                // Safety: scratch buffers are immediately overwritten by kernels each layer
                unsafe { self.stream.alloc::<f16>(n) }
                    .map_err(|e| LLMError::GpuError(format!("f16 scratch alloc ({n} elems): {e}")))
            };
            let alloc_u8 = |n: usize| -> Result<CudaSlice<u8>> {
                unsafe { self.stream.alloc::<u8>(n.max(1)) }
                    .map_err(|e| LLMError::GpuError(format!("u8 scratch alloc ({n} elems): {e}")))
            };
            let gateup_ws_bytes = self.cutlass.as_ref().map_or(1usize, |ck| {
                let gateup_ws = ck.gateup_silu_workspace_size(
                    max_tokens as i32,
                    (intermediate * 2) as i32,
                    hidden as i32,
                );
                let gate_aux_ws =
                    ck.gate_silu_mul_workspace_size(max_tokens as i32, intermediate as i32, hidden as i32);
                gateup_ws.max(gate_aux_ws).max(1)
            });

            let scratch = F16LayerScratch {
                max_tokens,
                qkv: alloc(max_tokens * qkv_dim)?,
                attn_out: alloc(max_tokens * q_dim)?,
                o_proj: alloc(max_tokens * hidden)?,
                normed: alloc(max_tokens * hidden)?,
                gate_up: alloc(max_tokens * intermediate * 2)?,
                gateup_ws: alloc_u8(gateup_ws_bytes)?,
                silu_out: alloc(max_tokens * intermediate)?,
                residual_a: alloc(max_tokens * hidden)?,
                down_a: alloc(max_tokens * hidden)?,
                residual_b: alloc(max_tokens * hidden)?,
                down_b: alloc(max_tokens * hidden)?,
            };

            let total_bytes = (max_tokens * (qkv_dim + q_dim + hidden * 3 + intermediate * 3)) * 2;
            info!(max_tokens, total_bytes, "f16 layer scratch allocated");
            *self.f16_scratch.borrow_mut() = Some(scratch);
            Ok(())
        }

        fn ensure_scratch_capacity(&self, num_tokens: usize) -> Result<()> {
            let required_tokens = num_tokens.max(DEFAULT_F16_SCRATCH_TOKENS);
            let current_tokens = self
                .f16_scratch
                .borrow()
                .as_ref()
                .map_or(0, |scratch| scratch.max_tokens);
            if current_tokens >= required_tokens {
                return Ok(());
            }

            info!(
                current_tokens,
                required_tokens, "growing f16 layer scratch for batched forward"
            );
            self.alloc_scratch(required_tokens)
        }

        fn alloc_persistent_v3_scratch(&self, max_splits: usize) -> Result<()> {
            use rvllm_gpu::persistent_v2_ffi::PersistentV2Kernels;

            let hidden = self.config.hidden_size;
            let q_dim = self.config.num_heads * self.config.head_dim;
            let kv_dim = self.config.num_kv_heads * self.config.head_dim;
            let qkv_dim = q_dim + kv_dim + kv_dim;
            let gate_up_dim = self.config.intermediate_size * 2;
            let heads_per_group = self.config.num_heads / self.config.num_kv_heads;
            let head_dim = self.config.head_dim;
            let num_kv_heads = self.config.num_kv_heads;

            let alloc_f16 = |n: usize| -> Result<CudaSlice<f16>> {
                unsafe { self.stream.alloc::<f16>(n) }
                    .map_err(|e| LLMError::GpuError(format!("pv3 scratch alloc ({n} elems): {e}")))
            };
            let alloc_u8 = |n: usize| -> Result<CudaSlice<u8>> {
                unsafe { self.stream.alloc::<u8>(n) }
                    .map_err(|e| LLMError::GpuError(format!("pv3 scratch alloc ({n} bytes): {e}")))
            };

            let (max_sz, sum_sz, acc_sz) = PersistentV2Kernels::split_kv_scratch_size(
                num_kv_heads,
                max_splits,
                heads_per_group,
                head_dim,
            );

            let scratch = PersistentV3Scratch {
                max_splits,
                qkv: alloc_f16(qkv_dim)?,
                attn: alloc_f16(q_dim)?,
                oproj: alloc_f16(hidden)?,
                gateup: alloc_f16(gate_up_dim)?,
                split_max: alloc_u8(max_sz)?,
                split_sum: alloc_u8(sum_sz)?,
                split_acc: alloc_u8(acc_sz)?,
                sync_flags: self
                    .stream
                    .alloc_zeros::<i32>(6)
                    .map_err(|e| LLMError::GpuError(format!("pv3 sync alloc: {e}")))?,
                residual_a: alloc_f16(hidden)?,
                residual_b: alloc_f16(hidden)?,
                mlp_a: alloc_f16(hidden)?,
                mlp_b: alloc_f16(hidden)?,
            };

            info!(max_splits, "persistent v3 scratch allocated");
            *self.persistent_v3_scratch.borrow_mut() = Some(scratch);
            Ok(())
        }

        fn ensure_persistent_v3_scratch(&self, max_splits: usize) -> Result<()> {
            let current = self
                .persistent_v3_scratch
                .borrow()
                .as_ref()
                .map_or(0, |scratch| scratch.max_splits);
            if current >= max_splits {
                return Ok(());
            }
            self.alloc_persistent_v3_scratch(max_splits)
        }

        /// Prepare the minimal unfused fp16 runtime state.
        pub fn prepare_fp16_runtime(&mut self) -> Result<()> {
            if self.f16_scratch.borrow().is_none() {
                info!(
                    num_layers = self.layers.len(),
                    "prepared zero-copy unfused fp16 runtime state"
                );
                self.alloc_scratch(256)?;
            }
            Ok(())
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

        fn forward_inner(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            is_prefill: bool,
            greedy_only: bool,
            mut phase_profile: Option<&mut DecodePhaseProfileSummary>,
        ) -> Result<ForwardOutput> {
            let num_tokens = token_ids.len();
            let num_seqs = attn_meta.context_lens.len();
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            debug!(
                num_tokens,
                num_seqs, is_prefill, greedy_only, "GpuModelRunner::forward_ex"
            );

            let max_context_len = attn_meta.max_context_len;

            // Single packed upload: all 6 metadata fields in one memcpy_htod.
            self.upload_metadata(token_ids, positions, attn_meta)?;

            // Step 1: token embedding lookup from packed buffer
            trace!("gpu_runner: embedding lookup");

            // === f16 forward path ===
            let debug_fwd = std::env::var("RVLLM_DEBUG").is_ok();
            let mut hidden_f16 = self.embedding_lookup_from_meta(num_tokens)?;

            if debug_fwd {
                let vals: Vec<f16> = self
                    .stream
                    .clone_dtoh(&hidden_f16)
                    .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                let first10: Vec<f32> = vals.iter().take(10).map(|v| v.to_f32()).collect();
                let has_nan = vals.iter().any(|v| v.to_f32().is_nan());
                let has_inf = vals.iter().any(|v| v.to_f32().is_infinite());
                let max_abs = vals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                info!("DEBUG embed: first10={first10:?} nan={has_nan} inf={has_inf} max={max_abs}");
            }

            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            let plan = self.forward_execution_plan(num_tokens, is_prefill);
            let path = plan.path;
            if is_prefill || !matches!(path, ForwardPath::Batched | ForwardPath::BatchedV2) {
                phase_profile = None;
            }
            if plan.use_scratch {
                self.ensure_scratch_capacity(num_tokens)?;
            }
            let mut scratch_borrow = self.f16_scratch.borrow_mut();
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;

            // For scratch double-buffering: copy embedding into residual_a so the
            // first layer can read from it while writing to residual_b.
            if plan.use_scratch {
                if let Some(ref mut s) = *scratch_borrow {
                    self.stream
                        .memcpy_dtod(
                            &hidden_f16,
                            &mut s.residual_a.slice_mut(..num_tokens * hidden_size),
                        )
                        .map_err(|e| LLMError::GpuError(format!("embed->scratch: {e}")))?;
                }
            }

            // === MEGAKERNEL V2: all layers in one cooperative launch (TC GEMV + split-KV) ===
            if path == ForwardPath::MegakernelV2Decode {
                return self.forward_megakernel_v2(
                    &hidden_f16,
                    gpu_cache,
                    packed_buf,
                    offsets,
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                );
            }
            // === PERSISTENT V3: non-cooperative, 1024 blocks, vectorized GEMV ===
            if path == ForwardPath::PersistentV3Decode {
                return self.forward_persistent_v3(
                    &hidden_f16,
                    gpu_cache,
                    packed_buf,
                    offsets,
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    greedy_only,
                );
            }
            // === PERSISTENT V2: per-layer cooperative TC GEMV + split-KV ===
            if path == ForwardPath::PersistentV2Decode {
                return self.forward_persistent_v2(
                    &hidden_f16,
                    gpu_cache,
                    packed_buf,
                    offsets,
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    greedy_only,
                );
            }
            // === MEGAKERNEL V1: all layers + LM head in one kernel launch ===
            if path == ForwardPath::MegakernelDecode {
                return self.forward_megakernel(
                    &hidden_f16,
                    gpu_cache,
                    packed_buf,
                    offsets,
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    greedy_only,
                );
            }

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];

                // Double-buffer: even layers write A read B, odd write B read A.
                let use_double_buf = plan.use_scratch && scratch_borrow.is_some();
                let (scratch_ref_opt, hs_ref, pmo_ref): (
                    Option<LayerScratchRef<'_>>,
                    &CudaSlice<f16>,
                    Option<&CudaSlice<f16>>,
                ) = if use_double_buf {
                    let s = scratch_borrow.as_mut().unwrap();
                    let even = layer_idx % 2 == 0;
                    let (write_res, write_down, read_res, read_down) = if even {
                        (&mut s.residual_b, &mut s.down_b, &s.residual_a, &s.down_a)
                    } else {
                        (&mut s.residual_a, &mut s.down_a, &s.residual_b, &s.down_b)
                    };
                    let pmo = if layer_idx > 0 {
                        Some(read_down as &CudaSlice<f16>)
                    } else {
                        None
                    };
                    (
                        Some(LayerScratchRef {
                            normed: &mut s.normed,
                            residual: write_res,
                            qkv: &mut s.qkv,
                            attn_out: &mut s.attn_out,
                            o_proj: &mut s.o_proj,
                            gate_up: &mut s.gate_up,
                            gateup_ws: &mut s.gateup_ws,
                            silu_out: &mut s.silu_out,
                            down: write_down,
                        }),
                        read_res as &CudaSlice<f16>,
                        pmo,
                    )
                } else {
                    (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                };

                let input = GpuLayerInput {
                    hidden_states: hs_ref,
                    positions: packed_buf
                        .slice(offsets.positions..offsets.positions + offsets.num_positions),
                    key_cache,
                    value_cache,
                    block_tables: packed_buf.slice(
                        offsets.block_tables..offsets.block_tables + offsets.num_block_tables,
                    ),
                    context_lens: packed_buf.slice(
                        offsets.context_lens..offsets.context_lens + offsets.num_context_lens,
                    ),
                    slot_mapping: packed_buf.slice(
                        offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping,
                    ),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: packed_buf.slice(
                        offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos,
                    ),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                    fp8_input_scratch_ptr: self.fp8_input_scratch.as_ref().map_or(0u64, |s| {
                        let (p, _) = DevicePtr::device_ptr(s, &self.stream);
                        p
                    }),
                    fp8_input_scratch_len: self.fp8_input_scratch.as_ref().map_or(0, |s| s.len()),
                };
                let weights = self.layer_weights(layer_idx)?;
                let mut scratch_ref = scratch_ref_opt;
                let result = if let Some(summary) = phase_profile.as_deref_mut() {
                    let mut layer_phase = BatchedLayerPhaseTimings::default();
                    let result = layer.forward_profiled(
                        path,
                        &input,
                        &weights,
                        &self.blas,
                        if use_double_buf {
                            pmo_ref
                        } else {
                            prev_mlp_out.as_ref()
                        },
                        self.cublaslt_ref(),
                        scratch_ref.as_mut(),
                        &mut layer_phase,
                        self.gemm_strategy,
                        self.cutlass.as_deref(),
                    )?;
                    summary.observe_layer(&layer_phase);
                    result
                } else {
                    layer.forward(
                        path,
                        &input,
                        &weights,
                        &self.blas,
                        if use_double_buf {
                            pmo_ref
                        } else {
                            prev_mlp_out.as_ref()
                        },
                        self.cublaslt_ref(),
                        scratch_ref.as_mut(),
                        self.gemm_strategy,
                        self.cutlass.as_deref(),
                    )?
                };
                if let Some((residual, mlp_out)) = result {
                    // Non-scratch path: take ownership
                    hidden_f16 = residual;
                    prev_mlp_out = Some(mlp_out);
                }
                // Scratch path (None): results are in s.residual/s.down,
                // next iteration reads them via the double-buffer swap.

                if debug_fwd && (layer_idx < 3 || layer_idx == num_layers - 1) {
                    let vals: Vec<f16> = self
                        .stream
                        .clone_dtoh(&hidden_f16)
                        .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                    let first5: Vec<f32> = vals.iter().take(5).map(|v| v.to_f32()).collect();
                    let has_nan = vals.iter().any(|v| v.to_f32().is_nan());
                    let has_inf = vals.iter().any(|v| v.to_f32().is_infinite());
                    let max_abs = vals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                    info!("DEBUG layer {layer_idx} residual: first5={first5:?} nan={has_nan} inf={has_inf} max={max_abs}");

                    let mvals: Vec<f16> = self
                        .stream
                        .clone_dtoh(prev_mlp_out.as_ref().unwrap())
                        .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                    let mfirst5: Vec<f32> = mvals.iter().take(5).map(|v| v.to_f32()).collect();
                    let mnan = mvals.iter().any(|v| v.to_f32().is_nan());
                    let mmax = mvals
                        .iter()
                        .map(|v| v.to_f32().abs())
                        .fold(0.0f32, f32::max);
                    info!(
                        "DEBUG layer {layer_idx} mlp_out: first5={mfirst5:?} nan={mnan} max={mmax}"
                    );
                }
            }

            // Double-buffer: extract final layer results from scratch (1 copy at end, not 28)
            if plan.use_scratch {
                if let Some(ref s) = *scratch_borrow {
                    let last_even = (num_layers - 1) % 2 == 0;
                    let (res_src, down_src) = if last_even {
                        (&s.residual_b, &s.down_b)
                    } else {
                        (&s.residual_a, &s.down_a)
                    };
                    let n = num_tokens * hidden_size;
                    let mut res_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("final res: {e}")))?;
                    self.stream
                        .memcpy_dtod(&res_src.slice(..n), &mut res_out)
                        .map_err(|e| LLMError::GpuError(format!("final res dtod: {e}")))?;
                    hidden_f16 = res_out;
                    let mut down_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("final mlp: {e}")))?;
                    self.stream
                        .memcpy_dtod(&down_src.slice(..n), &mut down_out)
                        .map_err(|e| LLMError::GpuError(format!("final mlp dtod: {e}")))?;
                    prev_mlp_out = Some(down_out);
                }
            }
            drop(scratch_borrow);

            // Final: fuse last layer's residual add with final RMSNorm
            let final_norm_start = if phase_profile.is_some() {
                self.stream
                    .synchronize()
                    .map_err(|e| LLMError::GpuError(format!("phase profile sync: {e}")))?;
                Some(Instant::now())
            } else {
                None
            };
            let normed_f16 = if let Some(ref last_mlp) = prev_mlp_out {
                let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                    &self.stream,
                    &self.loader,
                    &hidden_f16,
                    last_mlp,
                    &self.final_norm_weight,
                    self.rms_norm_eps,
                    num_tokens,
                    hidden_size,
                )?;
                n
            } else {
                self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
            };
            if let Some(summary) = phase_profile.as_mut() {
                self.stream
                    .synchronize()
                    .map_err(|e| LLMError::GpuError(format!("phase profile sync: {e}")))?;
                summary.final_norm_ns = final_norm_start.unwrap().elapsed().as_nanos() as u64;
            }

            if debug_fwd {
                // Check normed hidden and top logits
                let nvals: Vec<f16> = self
                    .stream
                    .clone_dtoh(&normed_f16)
                    .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                let nfirst5: Vec<f32> = nvals.iter().take(5).map(|v| v.to_f32()).collect();
                let nnan = nvals.iter().any(|v| v.to_f32().is_nan());
                let nmax = nvals
                    .iter()
                    .map(|v| v.to_f32().abs())
                    .fold(0.0f32, f32::max);
                info!("DEBUG normed: first5={nfirst5:?} nan={nnan} max={nmax}");

                // Compute full logits and find top-5
                let logits_dbg = CudaLinearLayer::forward_f16_in(
                    &normed_f16,
                    &self.lm_head_weight,
                    num_tokens,
                    vocab_size,
                    hidden_size,
                    &self.blas,
                )?;
                let logits_cpu: Vec<f32> = self
                    .stream
                    .clone_dtoh(&logits_dbg)
                    .map_err(|e| LLMError::GpuError(format!("debug logits dtoh: {e}")))?;
                // Find top-5 for last token
                let last_start = (num_tokens - 1) * vocab_size;
                let last_logits = &logits_cpu[last_start..last_start + vocab_size];
                let mut indexed: Vec<(usize, f32)> = last_logits
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i, v))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top5: Vec<(usize, f32)> = indexed[..5.min(indexed.len())].to_vec();
                info!("DEBUG top5_logits: {:?}", top5);
                info!(
                    "DEBUG logits range: min={:.2} max={:.2} mean={:.4}",
                    last_logits.iter().cloned().fold(f32::INFINITY, f32::min),
                    last_logits
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max),
                    last_logits.iter().sum::<f32>() / last_logits.len() as f32,
                );
            }

            // LM head + argmax: f16 hidden -> fused argmax
            let lm_head_start = if phase_profile.is_some() {
                self.stream
                    .synchronize()
                    .map_err(|e| LLMError::GpuError(format!("phase profile sync: {e}")))?;
                Some(Instant::now())
            } else {
                None
            };
            if num_tokens == 1 && greedy_only {
                let token_ids_gpu = self.gpu_fused_lm_head_argmax_f16_hidden(
                    &normed_f16,
                    &self.lm_head_weight,
                    vocab_size,
                    hidden_size,
                )?;
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head token DtoH: {e}")))?;
                if let Some(summary) = phase_profile.as_mut() {
                    self.stream
                        .synchronize()
                        .map_err(|e| LLMError::GpuError(format!("phase profile sync: {e}")))?;
                    summary.lm_head_ns = lm_head_start.unwrap().elapsed().as_nanos() as u64;
                    summary.log(num_tokens, num_layers);
                }
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Full logits path: hgemm f16 hidden x f16 lm_head -> f32 logits
            let logits_gpu = CudaLinearLayer::forward_f16_in(
                &normed_f16,
                &self.lm_head_weight,
                num_tokens,
                vocab_size,
                hidden_size,
                &self.blas,
            )?;

            if greedy_only {
                let token_ids_gpu = self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?;
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("argmax DtoH: {e}")))?;
                if let Some(summary) = phase_profile.as_mut() {
                    self.stream
                        .synchronize()
                        .map_err(|e| LLMError::GpuError(format!("phase profile sync: {e}")))?;
                    summary.lm_head_ns = lm_head_start.unwrap().elapsed().as_nanos() as u64;
                    summary.log(num_tokens, num_layers);
                }
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            let logits_cpu = self
                .stream
                .clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;
            if let Some(summary) = phase_profile.as_mut() {
                self.stream
                    .synchronize()
                    .map_err(|e| LLMError::GpuError(format!("phase profile sync: {e}")))?;
                summary.lm_head_ns = lm_head_start.unwrap().elapsed().as_nanos() as u64;
                summary.log(num_tokens, num_layers);
            }
            Ok(ForwardOutput::Logits(logits_cpu))
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
            self.forward_inner(token_ids, positions, attn_meta, is_prefill, greedy_only, None)
        }

        pub fn profile_decode_bucket(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            greedy_only: bool,
            target_bucket: usize,
        ) -> Result<ForwardOutput> {
            let mut phase_profile = DecodePhaseProfileSummary::new(target_bucket);
            self.forward_inner(
                token_ids,
                positions,
                attn_meta,
                false,
                greedy_only,
                Some(&mut phase_profile),
            )
        }

        /// Partial forward: run only the first `max_layers` transformer layers,
        /// then apply final RMSNorm + LM head to produce logits.
        /// Used by self-draft speculative decoding to get approximate predictions
        /// from a fraction of the model's depth.
        pub fn forward_partial(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            is_prefill: bool,
            max_layers: usize,
        ) -> Result<Vec<f32>> {
            let num_tokens = token_ids.len();
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            self.upload_metadata(token_ids, positions, attn_meta)?;

            let mut hidden_f16 = self.embedding_lookup_from_meta(num_tokens)?;

            let gpu_cache = self.cache.gpu_cache();
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            let num_seqs = attn_meta.context_lens.len();
            let max_context_len = attn_meta.max_context_len;
            let layers_to_run = max_layers.min(self.layers.len());
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;
            let plan = self.forward_execution_plan(num_tokens, is_prefill);
            let path = plan.path;
            if plan.use_scratch {
                self.ensure_scratch_capacity(num_tokens)?;
            }
            let mut scratch_borrow = self.f16_scratch.borrow_mut();

            // For Batched path: copy embedding into residual_a for double-buffer read
            if plan.use_scratch {
                if let Some(ref mut s) = *scratch_borrow {
                    self.stream
                        .memcpy_dtod(
                            &hidden_f16,
                            &mut s.residual_a.slice_mut(..num_tokens * hidden_size),
                        )
                        .map_err(|e| LLMError::GpuError(format!("partial embed->scratch: {e}")))?;
                }
            }

            for (layer_idx, layer) in self.layers.iter().take(layers_to_run).enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];

                let (mut scratch_ref_opt, hs_ref, pmo_ref): (
                    Option<LayerScratchRef<'_>>,
                    &CudaSlice<f16>,
                    Option<&CudaSlice<f16>>,
                ) = if plan.use_scratch {
                    if let Some(ref mut s) = *scratch_borrow {
                        let even = layer_idx % 2 == 0;
                        let (write_res, write_down, read_res, read_down) = if even {
                            (&mut s.residual_b, &mut s.down_b, &s.residual_a, &s.down_a)
                        } else {
                            (&mut s.residual_a, &mut s.down_a, &s.residual_b, &s.down_b)
                        };
                        let pmo = if layer_idx > 0 {
                            Some(read_down as &CudaSlice<f16>)
                        } else {
                            None
                        };
                        let hs: &CudaSlice<f16> =
                            if layer_idx > 0 { read_res } else { &hidden_f16 };
                        (
                            Some(LayerScratchRef {
                                normed: &mut s.normed,
                                residual: write_res,
                                qkv: &mut s.qkv,
                                attn_out: &mut s.attn_out,
                                o_proj: &mut s.o_proj,
                                gate_up: &mut s.gate_up,
                                gateup_ws: &mut s.gateup_ws,
                                silu_out: &mut s.silu_out,
                                down: write_down,
                            }),
                            hs,
                            pmo,
                        )
                    } else {
                        return Err(LLMError::GpuError(
                            "Batched path requires scratch buffers".into(),
                        ));
                    }
                } else {
                    (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                };

                let input = GpuLayerInput {
                    hidden_states: hs_ref,
                    positions: packed_buf
                        .slice(offsets.positions..offsets.positions + offsets.num_positions),
                    key_cache,
                    value_cache,
                    block_tables: packed_buf.slice(
                        offsets.block_tables..offsets.block_tables + offsets.num_block_tables,
                    ),
                    context_lens: packed_buf.slice(
                        offsets.context_lens..offsets.context_lens + offsets.num_context_lens,
                    ),
                    slot_mapping: packed_buf.slice(
                        offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping,
                    ),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: packed_buf.slice(
                        offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos,
                    ),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                    fp8_input_scratch_ptr: self.fp8_input_scratch.as_ref().map_or(0u64, |s| {
                        let (p, _) = DevicePtr::device_ptr(s, &self.stream);
                        p
                    }),
                    fp8_input_scratch_len: self.fp8_input_scratch.as_ref().map_or(0, |s| s.len()),
                };
                let weights = self.layer_weights(layer_idx)?;
                let result = layer.forward(
                    path,
                    &input,
                    &weights,
                    &self.blas,
                    if plan.use_scratch {
                        pmo_ref
                    } else {
                        prev_mlp_out.as_ref()
                    },
                    self.cublaslt_ref(),
                    scratch_ref_opt.as_mut(),
                    self.gemm_strategy,
                    self.cutlass.as_deref(),
                )?;
                if let Some((residual, mlp_out)) = result {
                    hidden_f16 = residual;
                    prev_mlp_out = Some(mlp_out);
                }
            }

            // Extract final results from scratch double-buffer
            if plan.use_scratch {
                if let Some(ref s) = *scratch_borrow {
                    let last_even = (layers_to_run - 1) % 2 == 0;
                    let (res_src, down_src) = if last_even {
                        (&s.residual_b, &s.down_b)
                    } else {
                        (&s.residual_a, &s.down_a)
                    };
                    let n = num_tokens * hidden_size;
                    let mut res_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("partial final res: {e}")))?;
                    self.stream
                        .memcpy_dtod(&res_src.slice(..n), &mut res_out)
                        .map_err(|e| LLMError::GpuError(format!("partial final res dtod: {e}")))?;
                    hidden_f16 = res_out;
                    let mut down_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("partial final mlp: {e}")))?;
                    self.stream
                        .memcpy_dtod(&down_src.slice(..n), &mut down_out)
                        .map_err(|e| LLMError::GpuError(format!("partial final mlp dtod: {e}")))?;
                    prev_mlp_out = Some(down_out);
                }
            }

            // Final norm + LM head (same as full forward)
            let normed_f16 = if let Some(ref last_mlp) = prev_mlp_out {
                let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                    &self.stream,
                    &self.loader,
                    &hidden_f16,
                    last_mlp,
                    &self.final_norm_weight,
                    self.rms_norm_eps,
                    num_tokens,
                    hidden_size,
                )?;
                n
            } else {
                self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
            };

            let logits_gpu = CudaLinearLayer::forward_f16_in(
                &normed_f16,
                &self.lm_head_weight,
                num_tokens,
                vocab_size,
                hidden_size,
                &self.blas,
            )?;

            let logits_cpu = self
                .stream
                .clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("forward_partial logits DtoH: {e}")))?;
            Ok(logits_cpu)
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
            self.meta_packed
                .borrow_mut()
                .upload(&scratch, &self.stream)
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
            let dummy_ctx = if attn_meta.context_lens.is_empty() {
                1
            } else {
                attn_meta.context_lens[0]
            };
            padded_meta.context_lens.resize(padded_batch, dummy_ctx);
            padded_meta.query_lens.resize(padded_batch, 1);
            let dummy_bt = if attn_meta.block_tables.is_empty() {
                vec![]
            } else {
                attn_meta.block_tables[0].clone()
            };
            for _ in 0..pad_count {
                padded_meta.block_tables.push(dummy_bt.clone());
            }
            padded_meta.slot_mapping.resize(padded_batch, 0);

            self.upload_metadata(&padded_tokens, &padded_positions, &padded_meta)
        }

        /// Upload decode metadata for the V2 graphed path using a stable padded
        /// packed layout without cloning a temporary metadata object.
        pub fn upload_decode_metadata_v2(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            padded_batch: usize,
        ) -> Result<()> {
            let actual = token_ids.len();
            let padded = padded_batch.max(actual);
            let max_blocks = self.graph_max_blocks;
            let bt_len = padded * max_blocks;
            let token_ids_off = 0usize;
            let positions_off = token_ids_off + padded;
            let context_lens_off = positions_off + padded;
            let block_tables_off = context_lens_off + padded;
            let slot_mapping_off = block_tables_off + bt_len;
            let seq_start_pos_off = slot_mapping_off + padded;
            let total_len = seq_start_pos_off + padded + 1;

            let dummy_ctx = attn_meta.context_lens.first().copied().unwrap_or(1) as i32;
            let dummy_bt = attn_meta.block_tables.first();
            let mut host = self.decode_meta_v2_state.borrow_mut();
            host.resize(padded, max_blocks);
            for (dst, &src) in host.token_ids.iter_mut().zip(token_ids.iter()) {
                *dst = src as i32;
            }
            for (dst, &src) in host.positions.iter_mut().zip(positions.iter()) {
                *dst = src as i32;
            }
            for (dst, &src) in host
                .context_lens
                .iter_mut()
                .zip(attn_meta.context_lens.iter())
            {
                *dst = src as i32;
            }
            for seq_idx in 0..padded {
                let row = attn_meta
                    .block_tables
                    .get(seq_idx)
                    .or(dummy_bt)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let row_base = seq_idx * max_blocks;
                for (blk_idx, &blk) in row.iter().take(max_blocks).enumerate() {
                    host.block_tables[row_base + blk_idx] = blk as i32;
                }
            }
            for (dst, &src) in host.slot_mapping.iter_mut().zip(attn_meta.slot_mapping.iter()) {
                *dst = src as i32;
            }
            let mut seq_pos = 0i32;
            for idx in 0..padded {
                host.seq_start_pos[idx] = seq_pos;
                seq_pos += attn_meta.query_lens.get(idx).copied().unwrap_or(1) as i32;
            }
            host.seq_start_pos[padded] = seq_pos;

            self.persistent_decode_meta
                .ensure_len(padded, max_blocks, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 persistent meta alloc: {e}")))?;

            self.persistent_decode_meta
                .token_ids
                .borrow_mut()
                .upload(&host.token_ids, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 token_ids HtoD: {e}")))?;
            self.persistent_decode_meta
                .positions
                .borrow_mut()
                .upload(&host.positions, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 positions HtoD: {e}")))?;
            self.persistent_decode_meta
                .context_lens
                .borrow_mut()
                .upload(&host.context_lens, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 context_lens HtoD: {e}")))?;
            self.persistent_decode_meta
                .block_tables
                .borrow_mut()
                .upload(&host.block_tables, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 block_tables HtoD: {e}")))?;
            self.persistent_decode_meta
                .slot_mapping
                .borrow_mut()
                .upload(&host.slot_mapping, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 slot_mapping HtoD: {e}")))?;
            self.persistent_decode_meta
                .seq_start_pos
                .borrow_mut()
                .upload(&host.seq_start_pos, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 seq_start_pos HtoD: {e}")))?;

            let mut meta = self.meta_packed.borrow_mut();
            meta.ensure_len(total_len, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 meta alloc: {e}")))?;
            let packed = meta.buf.as_mut().expect("meta_packed allocated");

            {
                let src = self.persistent_decode_meta.token_ids.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(token_ids_off..token_ids_off + padded),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 token_ids DtoD: {e}")))?;
            }
            {
                let src = self.persistent_decode_meta.positions.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(positions_off..positions_off + padded),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 positions DtoD: {e}")))?;
            }
            {
                let src = self.persistent_decode_meta.context_lens.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(context_lens_off..context_lens_off + padded),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 context_lens DtoD: {e}")))?;
            }
            {
                let src = self.persistent_decode_meta.block_tables.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(block_tables_off..block_tables_off + bt_len),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 block_tables DtoD: {e}")))?;
            }
            {
                let src = self.persistent_decode_meta.slot_mapping.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(slot_mapping_off..slot_mapping_off + padded),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 slot_mapping DtoD: {e}")))?;
            }
            {
                let src = self.persistent_decode_meta.seq_start_pos.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(seq_start_pos_off..seq_start_pos_off + padded + 1),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 seq_start_pos DtoD: {e}")))?;
            }

            self.meta_packed_offsets.set(PackedMetaOffsets {
                token_ids: token_ids_off,
                positions: positions_off,
                context_lens: context_lens_off,
                block_tables: block_tables_off,
                slot_mapping: slot_mapping_off,
                seq_start_pos: seq_start_pos_off,
                num_token_ids: padded,
                num_positions: padded,
                num_context_lens: padded,
                num_block_tables: bt_len,
                num_slot_mapping: padded,
                num_seq_start_pos: padded + 1,
            });

            Ok(())
        }

        pub fn upload_decode_metadata_v2_flat(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            query_lens: &[u32],
            context_lens: &[u32],
            slot_mapping: &[u32],
            block_tables_flat: &[u32],
            padded_batch: usize,
        ) -> Result<()> {
            let actual = token_ids.len();
            let padded = padded_batch.max(actual);
            let max_blocks = self.graph_max_blocks;
            let expected_bt_len = actual * max_blocks;
            if block_tables_flat.len() < expected_bt_len {
                return Err(LLMError::GpuError(format!(
                    "decode v2 flat block tables too short: have {}, need {}",
                    block_tables_flat.len(),
                    expected_bt_len
                )));
            }

            let bt_len = padded * max_blocks;
            let token_ids_off = 0usize;
            let positions_off = token_ids_off + padded;
            let context_lens_off = positions_off + padded;
            let block_tables_off = context_lens_off + padded;
            let slot_mapping_off = block_tables_off + bt_len;
            let seq_start_pos_off = slot_mapping_off + padded;
            let total_len = seq_start_pos_off + padded + 1;

            let dummy_ctx = context_lens.first().copied().unwrap_or(1) as i32;
            let mut host = self.decode_meta_v2_state.borrow_mut();
            host.resize(padded, max_blocks);

            host.token_ids.fill(0);
            for (dst, &src) in host.token_ids.iter_mut().zip(token_ids.iter()) {
                *dst = src as i32;
            }

            host.positions.fill(0);
            for (dst, &src) in host.positions.iter_mut().zip(positions.iter()) {
                *dst = src as i32;
            }

            host.context_lens.fill(dummy_ctx);
            for (dst, &src) in host.context_lens.iter_mut().zip(context_lens.iter()) {
                *dst = src as i32;
            }

            host.block_tables.fill(0);
            for seq_idx in 0..actual {
                let src_off = seq_idx * max_blocks;
                let dst_off = src_off;
                for (dst, &src) in host.block_tables[dst_off..dst_off + max_blocks]
                    .iter_mut()
                    .zip(block_tables_flat[src_off..src_off + max_blocks].iter())
                {
                    *dst = src as i32;
                }
            }
            if actual > 0 {
                let dummy_row = host.block_tables[..max_blocks].to_vec();
                for seq_idx in actual..padded {
                    let dst_off = seq_idx * max_blocks;
                    host.block_tables[dst_off..dst_off + max_blocks].copy_from_slice(&dummy_row);
                }
            }

            host.slot_mapping.fill(0);
            for (dst, &src) in host.slot_mapping.iter_mut().zip(slot_mapping.iter()) {
                *dst = src as i32;
            }

            let mut seq_pos = 0i32;
            for idx in 0..padded {
                host.seq_start_pos[idx] = seq_pos;
                seq_pos += query_lens.get(idx).copied().unwrap_or(1) as i32;
            }
            host.seq_start_pos[padded] = seq_pos;

            self.persistent_decode_meta
                .ensure_len(padded, max_blocks, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 persistent meta alloc: {e}")))?;

            self.persistent_decode_meta
                .token_ids
                .borrow_mut()
                .upload(&host.token_ids, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 token_ids HtoD: {e}")))?;
            self.persistent_decode_meta
                .positions
                .borrow_mut()
                .upload(&host.positions, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 positions HtoD: {e}")))?;
            self.persistent_decode_meta
                .context_lens
                .borrow_mut()
                .upload(&host.context_lens, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 context_lens HtoD: {e}")))?;
            self.persistent_decode_meta
                .block_tables
                .borrow_mut()
                .upload(&host.block_tables, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 block_tables HtoD: {e}")))?;
            self.persistent_decode_meta
                .slot_mapping
                .borrow_mut()
                .upload(&host.slot_mapping, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 slot_mapping HtoD: {e}")))?;
            self.persistent_decode_meta
                .seq_start_pos
                .borrow_mut()
                .upload(&host.seq_start_pos, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 seq_start_pos HtoD: {e}")))?;

            let mut meta = self.meta_packed.borrow_mut();
            meta.ensure_len(total_len, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("decode v2 meta alloc: {e}")))?;
            let packed = meta.buf.as_mut().expect("meta_packed allocated");

            {
                let src = self.persistent_decode_meta.token_ids.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(token_ids_off..token_ids_off + padded),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 token_ids DtoD: {e}")))?;
            }
            {
                let src = self.persistent_decode_meta.positions.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(positions_off..positions_off + padded),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 positions DtoD: {e}")))?;
            }
            {
                let src = self.persistent_decode_meta.context_lens.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(context_lens_off..context_lens_off + padded),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 context_lens DtoD: {e}")))?;
            }
            {
                let src = self.persistent_decode_meta.block_tables.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(block_tables_off..block_tables_off + bt_len),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 block_tables DtoD: {e}")))?;
            }
            {
                let src = self.persistent_decode_meta.slot_mapping.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(slot_mapping_off..slot_mapping_off + padded),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 slot_mapping DtoD: {e}")))?;
            }
            {
                let src = self.persistent_decode_meta.seq_start_pos.borrow();
                self.stream
                    .memcpy_dtod(
                        src.slice(),
                        &mut packed.slice_mut(seq_start_pos_off..seq_start_pos_off + padded + 1),
                    )
                    .map_err(|e| LLMError::GpuError(format!("decode v2 seq_start_pos DtoD: {e}")))?;
            }

            self.meta_packed_offsets.set(PackedMetaOffsets {
                token_ids: token_ids_off,
                positions: positions_off,
                context_lens: context_lens_off,
                block_tables: block_tables_off,
                slot_mapping: slot_mapping_off,
                seq_start_pos: seq_start_pos_off,
                num_token_ids: padded,
                num_positions: padded,
                num_context_lens: padded,
                num_block_tables: bt_len,
                num_slot_mapping: padded,
                num_seq_start_pos: padded + 1,
            });

            Ok(())
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
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            let plan = self.forward_execution_plan(num_tokens, is_prefill);
            let path = plan.path;
            if path == ForwardPath::PersistentV3Decode {
                return self.forward_persistent_v3(
                    &hidden_f16,
                    gpu_cache,
                    packed_buf,
                    offsets,
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    greedy_only,
                );
            }
            // Double-buffered scratch for batched decode/prefill and optional
            // batch-1 decode experiments: zero per-layer allocations.
            if plan.use_scratch {
                self.ensure_scratch_capacity(num_tokens)?;
            }
            let mut scratch_borrow = self.f16_scratch.borrow_mut();
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let even = layer_idx % 2 == 0;

                let (mut scratch_ref_opt, hs_ref, pmo_ref): (
                    Option<LayerScratchRef<'_>>,
                    &CudaSlice<f16>,
                    Option<&CudaSlice<f16>>,
                ) = if plan.use_scratch {
                    if let Some(ref mut s) = *scratch_borrow {
                        let (write_res, write_down, read_res, read_down) = if even {
                            (&mut s.residual_b, &mut s.down_b, &s.residual_a, &s.down_a)
                        } else {
                            (&mut s.residual_a, &mut s.down_a, &s.residual_b, &s.down_b)
                        };
                        let pmo = if layer_idx > 0 {
                            Some(read_down as &CudaSlice<f16>)
                        } else {
                            None
                        };
                        let hs: &CudaSlice<f16> =
                            if layer_idx > 0 { read_res } else { &hidden_f16 };
                        (
                            Some(LayerScratchRef {
                                normed: &mut s.normed,
                                residual: write_res,
                                qkv: &mut s.qkv,
                                attn_out: &mut s.attn_out,
                                o_proj: &mut s.o_proj,
                                gate_up: &mut s.gate_up,
                                gateup_ws: &mut s.gateup_ws,
                                silu_out: &mut s.silu_out,
                                down: write_down,
                            }),
                            hs,
                            pmo,
                        )
                    } else {
                        (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                    }
                } else {
                    (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                };

                let input = GpuLayerInput {
                    hidden_states: hs_ref,
                    positions: packed_buf
                        .slice(offsets.positions..offsets.positions + offsets.num_positions),
                    key_cache,
                    value_cache,
                    block_tables: packed_buf.slice(
                        offsets.block_tables..offsets.block_tables + offsets.num_block_tables,
                    ),
                    context_lens: packed_buf.slice(
                        offsets.context_lens..offsets.context_lens + offsets.num_context_lens,
                    ),
                    slot_mapping: packed_buf.slice(
                        offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping,
                    ),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: packed_buf.slice(
                        offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos,
                    ),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                    fp8_input_scratch_ptr: self.fp8_input_scratch.as_ref().map_or(0u64, |s| {
                        let (p, _) = DevicePtr::device_ptr(s, &self.stream);
                        p
                    }),
                    fp8_input_scratch_len: self.fp8_input_scratch.as_ref().map_or(0, |s| s.len()),
                };
                let weights = self.layer_weights(layer_idx)?;
                let result = layer.forward(
                    path,
                    &input,
                    &weights,
                    &self.blas,
                    if plan.use_scratch {
                        pmo_ref
                    } else {
                        prev_mlp_out.as_ref()
                    },
                    self.cublaslt_ref(),
                    scratch_ref_opt.as_mut(),
                    self.gemm_strategy,
                    self.cutlass.as_deref(),
                )?;
                if let Some((residual, mlp_out)) = result {
                    hidden_f16 = residual;
                    prev_mlp_out = Some(mlp_out);
                }
            }

            // Final: fuse last layer's residual add with final RMSNorm.
            // For the graphed batched path, read the final residual/down buffers
            // directly from scratch instead of copying them out first.
            let normed_f16 = if plan.use_scratch {
                if let Some(ref s) = *scratch_borrow {
                    let last_even = (num_layers - 1) % 2 == 0;
                    let (res_src, down_src) = if last_even {
                        (&s.residual_b, &s.down_b)
                    } else {
                        (&s.residual_a, &s.down_a)
                    };
                    let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                        &self.stream,
                        &self.loader,
                        res_src,
                        down_src,
                        &self.final_norm_weight,
                        self.rms_norm_eps,
                        num_tokens,
                        hidden_size,
                    )?;
                    n
                } else if let Some(ref last_mlp) = prev_mlp_out {
                    let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                        &self.stream,
                        &self.loader,
                        &hidden_f16,
                        last_mlp,
                        &self.final_norm_weight,
                        self.rms_norm_eps,
                        num_tokens,
                        hidden_size,
                    )?;
                    n
                } else {
                    self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
                }
            } else if let Some(ref last_mlp) = prev_mlp_out {
                let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                    &self.stream,
                    &self.loader,
                    &hidden_f16,
                    last_mlp,
                    &self.final_norm_weight,
                    self.rms_norm_eps,
                    num_tokens,
                    hidden_size,
                )?;
                n
            } else {
                self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
            };
            drop(scratch_borrow);

            // LM head + argmax: f16 hidden -> fused argmax
            if num_tokens == 1 && greedy_only {
                let token_ids_gpu = self.gpu_fused_lm_head_argmax_f16_hidden(
                    &normed_f16,
                    &self.lm_head_weight,
                    vocab_size,
                    hidden_size,
                )?;
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head token DtoH: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Full logits path: hgemm f16 hidden x f16 lm_head -> f32 logits
            let logits_gpu = CudaLinearLayer::forward_f16_in(
                &normed_f16,
                &self.lm_head_weight,
                num_tokens,
                vocab_size,
                hidden_size,
                &self.blas,
            )?;

            if greedy_only {
                let token_ids_gpu = self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?;
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("argmax DtoH: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            let logits_cpu = self
                .stream
                .clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;
            Ok(ForwardOutput::Logits(logits_cpu))
        }

        /// Get a reference to the CUDA stream used by this runner.
        pub fn cuda_stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }

        fn padded_decode_batch_size(batch: usize) -> usize {
            if batch <= 1 {
                return 1;
            }
            if batch <= 2 {
                return 2;
            }
            if batch <= 4 {
                return 4;
            }
            if batch <= 8 {
                return 8;
            }
            if batch <= 256 {
                return (batch + 7) & !7;
            }
            (batch + 31) & !31
        }

        pub fn decode_execution_plan(
            &self,
            num_tokens: usize,
            is_prefill: bool,
            is_pure_decode: bool,
        ) -> DecodeExecutionPlan {
            let graph_tokens = if is_pure_decode {
                Self::padded_decode_batch_size(num_tokens)
            } else {
                num_tokens
            };
            let graph_plan = self.forward_execution_plan(graph_tokens, is_prefill);
            DecodeExecutionPlan {
                actual_tokens: num_tokens,
                graph_tokens,
                use_graphed_decode: is_pure_decode && graph_plan.graph_capture_supported,
                use_batched_v2: matches!(graph_plan.path, ForwardPath::BatchedV2),
            }
        }

        pub fn decode_graph_dispatch(
            &self,
            num_tokens: usize,
            is_prefill: bool,
            is_pure_decode: bool,
            state: DecodeGraphRuntimeState,
        ) -> DecodeGraphDispatch {
            let execution = self.decode_execution_plan(num_tokens, is_prefill, is_pure_decode);
            let action = if !execution.use_graphed_decode || !state.graphs_enabled {
                DecodeGraphAction::Raw
            } else if state.exact_graph_available {
                DecodeGraphAction::Replay
            } else if state.warmup_complete && !state.capture_attempted {
                DecodeGraphAction::Capture
            } else {
                DecodeGraphAction::Raw
            };
            DecodeGraphDispatch { execution, action }
        }

        /// Whether the selected forward path can be executed by
        /// `forward_gpu_only()` and therefore participate in CUDA graph capture.
        pub fn graph_capture_supported(&self, num_tokens: usize, is_prefill: bool) -> bool {
            self.forward_execution_plan(num_tokens, is_prefill)
                .graph_capture_supported
        }

        pub fn graph_max_blocks(&self) -> usize {
            self.graph_max_blocks
        }

        /// Get cublasLt reference (None when feature is off).
        fn cublaslt_ref(&self) -> Option<&crate::CublasLtRef> {
            #[cfg(feature = "cublaslt")]
            {
                self.blas_lt.as_ref()
            }
            #[cfg(not(feature = "cublaslt"))]
            {
                None
            }
        }

        fn resolve_forward_path(&self, num_tokens: usize, is_prefill: bool) -> ForwardPath {
            if num_tokens != 1 || is_prefill {
                if std::env::var("RVLLM_BATCHED_PIPELINE_V2").map_or(false, |v| v == "0") {
                    ForwardPath::Batched
                } else {
                    ForwardPath::BatchedV2
                }
            } else if let Some(path) = self.experimental_decode_path_override() {
                path
            } else if std::env::var("RVLLM_BATCHED_DECODE_1").map_or(true, |v| v != "0") {
                ForwardPath::Batched
            } else if std::env::var("RVLLM_CUBLAS_DECODE").map_or(true, |v| v != "0") {
                ForwardPath::CublasGemvDecode
            } else {
                #[cfg(feature = "cublaslt")]
                if self.blas_lt.is_some()
                    && !self.fp8_fused_qkv.is_empty()
                    && self.fp8_input_scratch.is_some()
                {
                    return ForwardPath::Fp8Decode;
                }
                ForwardPath::FusedDecode
            }
        }

        fn forward_execution_plan(
            &self,
            num_tokens: usize,
            is_prefill: bool,
        ) -> ForwardExecutionPlan {
            let path = self.resolve_forward_path(num_tokens, is_prefill);
            ForwardExecutionPlan {
                path,
                use_scratch: num_tokens > 1
                    || is_prefill
                    || matches!(path, ForwardPath::Batched | ForwardPath::BatchedV2),
                graph_capture_supported: forward_path_graph_capture_supported(path),
            }
        }

        fn experimental_decode_path_override(&self) -> Option<ForwardPath> {
            if std::env::var("RVLLM_PERSISTENT_V3").map_or(false, |v| v == "1")
                && self
                    .persistent_v2
                    .as_ref()
                    .map_or(false, |k| k.has_v3_kernel())
            {
                return Some(ForwardPath::PersistentV3Decode);
            }
            if std::env::var("RVLLM_MEGAKERNEL_V2").map_or(false, |v| v == "1")
                && self
                    .persistent_v2
                    .as_ref()
                    .map_or(false, |k| k.has_mega_kernel())
            {
                return Some(ForwardPath::MegakernelV2Decode);
            }
            if std::env::var("RVLLM_PERSISTENT_V2").map_or(false, |v| v == "1")
                && self
                    .persistent_v2
                    .as_ref()
                    .map_or(false, |k| k.has_layer_kernel())
            {
                return Some(ForwardPath::PersistentV2Decode);
            }
            if std::env::var("RVLLM_MEGAKERNEL").map_or(false, |v| v == "1") {
                return Some(ForwardPath::MegakernelDecode);
            }
            if std::env::var("RVLLM_PERSISTENT").map_or(false, |v| v == "1") {
                return Some(ForwardPath::PersistentDecode);
            }
            None
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
            self.persistent_decode_meta
                .ensure_len(max_seqs, self.graph_max_blocks, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("pre-alloc persistent v2 meta: {e}")))?;
            let mut meta = self.meta_packed.borrow_mut();
            meta.ensure_len(max_meta, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("pre-alloc meta: {e}")))?;
            info!(
                max_meta,
                "pre-allocated packed metadata buffer for graph stability"
            );
            // Pre-allocate graph output buffer too (same reason -- stable pointer).
            {
                let mut out = self.graph_output.borrow_mut();
                *out = Some(
                    self.stream
                        .alloc_zeros::<i32>(max_seqs)
                        .map_err(|e| LLMError::GpuError(format!("pre-alloc graph_output: {e}")))?,
                );
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
            let plan = self.forward_execution_plan(num_tokens, is_prefill);
            let path = plan.path;
            if path == ForwardPath::PersistentV3Decode {
                return self.forward_persistent_v3_graph(
                    &hidden_f16,
                    gpu_cache,
                    packed_buf,
                    offsets,
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                );
            }
            // Double-buffered scratch for batched decode/prefill and optional
            // batch-1 decode experiments: zero per-layer allocations.
            if plan.use_scratch {
                self.ensure_scratch_capacity(num_tokens)?;
            }
            let mut scratch_borrow = self.f16_scratch.borrow_mut();
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let even = layer_idx % 2 == 0;

                let (mut scratch_ref_opt, hs_ref, pmo_ref): (
                    Option<LayerScratchRef<'_>>,
                    &CudaSlice<f16>,
                    Option<&CudaSlice<f16>>,
                ) = if plan.use_scratch {
                    if let Some(ref mut s) = *scratch_borrow {
                        let (write_res, write_down, read_res, read_down) = if even {
                            (&mut s.residual_b, &mut s.down_b, &s.residual_a, &s.down_a)
                        } else {
                            (&mut s.residual_a, &mut s.down_a, &s.residual_b, &s.down_b)
                        };
                        let pmo = if layer_idx > 0 {
                            Some(read_down as &CudaSlice<f16>)
                        } else {
                            None
                        };
                        let hs: &CudaSlice<f16> =
                            if layer_idx > 0 { read_res } else { &hidden_f16 };
                        (
                            Some(LayerScratchRef {
                                normed: &mut s.normed,
                                residual: write_res,
                                qkv: &mut s.qkv,
                                attn_out: &mut s.attn_out,
                                o_proj: &mut s.o_proj,
                                gate_up: &mut s.gate_up,
                                gateup_ws: &mut s.gateup_ws,
                                silu_out: &mut s.silu_out,
                                down: write_down,
                            }),
                            hs,
                            pmo,
                        )
                    } else {
                        (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                    }
                } else {
                    (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                };

                let input = GpuLayerInput {
                    hidden_states: hs_ref,
                    positions: packed_buf
                        .slice(offsets.positions..offsets.positions + offsets.num_positions),
                    key_cache,
                    value_cache,
                    block_tables: packed_buf.slice(
                        offsets.block_tables..offsets.block_tables + offsets.num_block_tables,
                    ),
                    context_lens: packed_buf.slice(
                        offsets.context_lens..offsets.context_lens + offsets.num_context_lens,
                    ),
                    slot_mapping: packed_buf.slice(
                        offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping,
                    ),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: packed_buf.slice(
                        offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos,
                    ),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                    fp8_input_scratch_ptr: self.fp8_input_scratch.as_ref().map_or(0u64, |s| {
                        let (p, _) = DevicePtr::device_ptr(s, &self.stream);
                        p
                    }),
                    fp8_input_scratch_len: self.fp8_input_scratch.as_ref().map_or(0, |s| s.len()),
                };
                let weights = self.layer_weights(layer_idx)?;
                let result = layer.forward(
                    path,
                    &input,
                    &weights,
                    &self.blas,
                    if plan.use_scratch {
                        pmo_ref
                    } else {
                        prev_mlp_out.as_ref()
                    },
                    self.cublaslt_ref(),
                    scratch_ref_opt.as_mut(),
                    self.gemm_strategy,
                    self.cutlass.as_deref(),
                )?;
                if let Some((residual, mlp_out)) = result {
                    hidden_f16 = residual;
                    prev_mlp_out = Some(mlp_out);
                }
            }

            // Final: fuse last layer's residual add with final RMSNorm.
            // For the graphed batched path, read the final residual/down buffers
            // directly from scratch instead of copying them out first.
            let normed_f16 = if plan.use_scratch {
                if let Some(ref s) = *scratch_borrow {
                    let last_even = (num_layers - 1) % 2 == 0;
                    let (res_src, down_src) = if last_even {
                        (&s.residual_b, &s.down_b)
                    } else {
                        (&s.residual_a, &s.down_a)
                    };
                    let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                        &self.stream,
                        &self.loader,
                        res_src,
                        down_src,
                        &self.final_norm_weight,
                        self.rms_norm_eps,
                        num_tokens,
                        hidden_size,
                    )?;
                    n
                } else if let Some(ref last_mlp) = prev_mlp_out {
                    let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                        &self.stream,
                        &self.loader,
                        &hidden_f16,
                        last_mlp,
                        &self.final_norm_weight,
                        self.rms_norm_eps,
                        num_tokens,
                        hidden_size,
                    )?;
                    n
                } else {
                    self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
                }
            } else if let Some(ref last_mlp) = prev_mlp_out {
                let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                    &self.stream,
                    &self.loader,
                    &hidden_f16,
                    last_mlp,
                    &self.final_norm_weight,
                    self.rms_norm_eps,
                    num_tokens,
                    hidden_size,
                )?;
                n
            } else {
                self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
            };
            drop(scratch_borrow);

            if num_tokens == 1 {
                let token_ids_gpu = self.gpu_fused_lm_head_argmax_f16_hidden(
                    &normed_f16,
                    &self.lm_head_weight,
                    vocab_size,
                    hidden_size,
                )?;
                self.write_graph_output(&token_ids_gpu, num_tokens)?;
            } else {
                let logits_gpu = CudaLinearLayer::forward_f16_in(
                    &normed_f16,
                    &self.lm_head_weight,
                    num_tokens,
                    vocab_size,
                    hidden_size,
                    &self.blas,
                )?;
                let mut out = self.graph_output.borrow_mut();
                let dst = out.as_mut().ok_or_else(|| {
                    LLMError::GpuError(
                        "graph_output not preallocated -- call prepare_for_graph_capture first"
                            .into(),
                    )
                })?;
                if dst.len() < num_tokens {
                    return Err(LLMError::GpuError(format!(
                        "graph_output too small for {num_tokens} tokens (have {})",
                        dst.len()
                    )));
                }
                self.gpu_argmax_into(&logits_gpu, dst, num_tokens, vocab_size)?;
            }

            Ok(())
        }

        fn write_graph_output(
            &self,
            token_ids_gpu: &CudaSlice<i32>,
            num_tokens: usize,
        ) -> Result<()> {
            let mut out = self.graph_output.borrow_mut();
            let need = num_tokens;
            let have = out.as_ref().map_or(0, |b| b.len());
            if have < need {
                *out = Some(
                    self.stream
                        .alloc_zeros::<i32>(need)
                        .map_err(|e| LLMError::GpuError(format!("graph_output alloc: {e}")))?,
                );
            }
            let dst = out.as_mut().unwrap();
            self.stream
                .memcpy_dtod(token_ids_gpu, dst)
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
                LLMError::GpuError(
                    "graph_output not populated -- call forward_gpu_only first".into(),
                )
            })?;
            // Copy only the needed elements
            let full = self
                .stream
                .clone_dtoh(buf)
                .map_err(|e| LLMError::GpuError(format!("graph_output DtoH: {e}")))?;
            Ok(full[..num_tokens].to_vec())
        }

        /// Enqueue an async DtoH copy of graph output into a pinned host buffer.
        ///
        /// Unlike `read_graph_output`, this does NOT synchronize the stream.
        /// The caller must call `sync_stream()` before reading from `dst`.
        /// `dst` MUST be pinned host memory for truly async behavior; with
        /// pageable memory cuMemcpyDtoHAsync degrades to synchronous.
        pub fn read_graph_output_async(&self, num_tokens: usize, dst: &mut [i32]) -> Result<()> {
            let out = self.graph_output.borrow();
            let buf = out.as_ref().ok_or_else(|| {
                LLMError::GpuError(
                    "graph_output not populated -- call forward_gpu_only first".into(),
                )
            })?;
            // Only copy num_tokens elements (not the full padded buffer)
            let src_view = buf.slice(..num_tokens);
            self.stream
                .memcpy_dtoh(&src_view, &mut dst[..num_tokens])
                .map_err(|e| LLMError::GpuError(format!("graph_output async DtoH: {e}")))?;
            Ok(())
        }

        /// Synchronize the runner's CUDA stream, blocking until all enqueued
        /// work (graph replay + async DtoH) completes.
        pub fn sync_stream(&self) -> Result<()> {
            self.stream
                .synchronize()
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
            let output: CudaSlice<i32> = self
                .stream
                .alloc_zeros::<i32>(num_tokens)
                .map_err(|e| LLMError::GpuError(format!("argmax alloc: {e}")))?;

            self.gpu_argmax_into(logits_gpu, &output, num_tokens, vocab_size)?;

            Ok(output)
        }

        fn gpu_argmax_into(
            &self,
            logits_gpu: &CudaSlice<f32>,
            output: &CudaSlice<i32>,
            num_tokens: usize,
            vocab_size: usize,
        ) -> Result<()> {
            let kernel = self.loader.get_func("argmax", "argmax_kernel")?;

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
                    .arg(output)
                    .arg(&(vocab_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("argmax_kernel launch: {e}")))?;
            }

            Ok(())
        }

        /// Fused LM-head matvec + argmax for single-token greedy decode (f16 weights).
        fn gpu_fused_lm_head_argmax_f16(
            &self,
            hidden_state: &CudaSlice<f32>,
            weight: &CudaSlice<f16>,
            vocab_size: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<i32>> {
            let pass1 = self.loader.get_func(
                "fused_lm_head_argmax_f16",
                "fused_lm_head_argmax_f16_kernel",
            )?;
            let pass2 = self
                .loader
                .get_func("fused_lm_head_argmax", "fused_lm_head_argmax_reduce_kernel")?;

            let num_blocks = (vocab_size + 255) / 256;

            let partial_val: CudaSlice<f32> =
                self.stream.alloc_zeros::<f32>(num_blocks).map_err(|e| {
                    LLMError::GpuError(format!("fused_lm_head_f16 partial_val alloc: {e}"))
                })?;
            let partial_idx: CudaSlice<i32> =
                self.stream.alloc_zeros::<i32>(num_blocks).map_err(|e| {
                    LLMError::GpuError(format!("fused_lm_head_f16 partial_idx alloc: {e}"))
                })?;
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
                    .map_err(|e| {
                        LLMError::GpuError(format!("fused_lm_head_argmax_f16_kernel launch: {e}"))
                    })?;
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
                    .map_err(|e| {
                        LLMError::GpuError(format!("fused_lm_head_argmax_f16_reduce launch: {e}"))
                    })?;
            }

            Ok(output)
        }

        /// Megakernel decode: all layers + LM head in one kernel launch.
        fn forward_megakernel(
            &self,
            embedding: &CudaSlice<f16>,
            gpu_cache: &[(CudaSlice<f16>, CudaSlice<f16>)],
            packed_buf: &CudaSlice<i32>,
            offsets: PackedMetaOffsets,
            num_tokens: usize,
            _num_seqs: usize,
            max_context_len: u32,
            block_size: usize,
            _greedy_only: bool,
        ) -> Result<ForwardOutput> {
            use crate::megakernel;
            use std::ffi::c_void;

            let cfg = &self.config;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;
            let vocab = cfg.vocab_size;
            let num_layers = self.layers.len();

            // Build instruction tape
            let has_bias =
                !self.fused_qkv_bias.is_empty() && self.fused_qkv_bias.iter().any(|b| b.is_some());
            let tape = megakernel::build_instruction_tape(
                num_layers,
                hidden,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate,
                vocab,
                self.rms_norm_eps,
                has_bias,
            );
            let num_instructions = tape.len() as i32;

            // Upload instruction tape
            let tape_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tape.as_ptr() as *const u8, tape.len() * 64) };
            let tape_gpu = self
                .stream
                .clone_htod(tape_bytes)
                .map_err(|e| LLMError::GpuError(format!("mk tape upload: {e}")))?;

            // Collect weight pointers (7 per layer + 2 global)
            let mut wptrs: Vec<u64> = Vec::with_capacity(megakernel::weight_ptr_count(num_layers));
            for i in 0..num_layers {
                let w = self.layer_weights(i)?;
                // input_layernorm
                wptrs.push(DevicePtr::device_ptr(w.input_layernorm, &self.stream).0);
                // fused_qkv (required for megakernel)
                let qkv_w = w
                    .fused_qkv
                    .ok_or_else(|| LLMError::GpuError("megakernel requires fused_qkv".into()))?;
                wptrs.push(DevicePtr::device_ptr(qkv_w, &self.stream).0);
                // qkv_bias (0 if none)
                wptrs.push(
                    w.qkv_bias
                        .map_or(0u64, |b| DevicePtr::device_ptr(b, &self.stream).0),
                );
                // o_proj
                wptrs.push(DevicePtr::device_ptr(w.o_proj, &self.stream).0);
                // post_attention_layernorm
                wptrs.push(DevicePtr::device_ptr(w.post_attention_layernorm, &self.stream).0);
                // fused_gate_up (required)
                let gu_w = w.fused_gate_up.ok_or_else(|| {
                    LLMError::GpuError("megakernel requires fused_gate_up".into())
                })?;
                wptrs.push(DevicePtr::device_ptr(gu_w, &self.stream).0);
                // down_proj
                wptrs.push(DevicePtr::device_ptr(w.down_proj, &self.stream).0);
            }
            // Global: final_norm, lm_head
            wptrs.push(DevicePtr::device_ptr(&self.final_norm_weight, &self.stream).0);
            wptrs.push(DevicePtr::device_ptr(&self.lm_head_weight, &self.stream).0);

            let wptrs_gpu = self
                .stream
                .clone_htod(&wptrs)
                .map_err(|e| LLMError::GpuError(format!("mk wptrs upload: {e}")))?;

            // Allocate scratch buffer
            let scratch_bytes = megakernel::scratch_size_bytes(
                hidden,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate,
                vocab,
            );
            let mut scratch_gpu = unsafe { self.stream.alloc::<u8>(scratch_bytes) }
                .map_err(|e| LLMError::GpuError(format!("mk scratch: {e}")))?;

            // Copy embedding into residual_a slot in scratch via raw memcpy
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_dim = q_dim + 2 * kv_dim;
            let gate_up_dim = intermediate * 2;
            let res_a_byte_offset = (qkv_dim + q_dim + hidden + gate_up_dim + hidden) * 2;
            {
                let embed_bytes = num_tokens * hidden * 2;
                let src_ptr = DevicePtr::device_ptr(embedding, &self.stream).0;
                let dst_ptr =
                    DevicePtr::device_ptr(&scratch_gpu, &self.stream).0 + res_a_byte_offset as u64;
                unsafe {
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr,
                        src_ptr,
                        embed_bytes as usize,
                        self.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| LLMError::GpuError(format!("mk embed copy: {e}")))?;
                }
            }

            // Sync counters (zeroed)
            let num_counters = megakernel::sync_counter_count(num_layers);
            let sync_gpu = self
                .stream
                .clone_htod(&vec![0i32; num_counters])
                .map_err(|e| LLMError::GpuError(format!("mk sync: {e}")))?;

            // Collect per-layer KV cache pointers
            let mut key_ptrs: Vec<u64> = Vec::with_capacity(num_layers);
            let mut val_ptrs: Vec<u64> = Vec::with_capacity(num_layers);
            for (kc, vc) in gpu_cache.iter().take(num_layers) {
                key_ptrs.push(DevicePtr::device_ptr(kc, &self.stream).0);
                val_ptrs.push(DevicePtr::device_ptr(vc, &self.stream).0);
            }
            let key_ptrs_gpu = self
                .stream
                .clone_htod(&key_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk key_ptrs: {e}")))?;
            let val_ptrs_gpu = self
                .stream
                .clone_htod(&val_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk val_ptrs: {e}")))?;

            // Output token buffer
            let output_gpu = self
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| LLMError::GpuError(format!("mk output: {e}")))?;

            // Get device pointers for kernel args
            let p_tape = DevicePtr::device_ptr(&tape_gpu, &self.stream).0;
            let p_wptrs = DevicePtr::device_ptr(&wptrs_gpu, &self.stream).0;
            let p_scratch = DevicePtr::device_ptr(&scratch_gpu, &self.stream).0;
            let p_sync = DevicePtr::device_ptr(&sync_gpu, &self.stream).0;
            let p_keys = DevicePtr::device_ptr(&key_ptrs_gpu, &self.stream).0;
            let p_vals = DevicePtr::device_ptr(&val_ptrs_gpu, &self.stream).0;
            // Compute metadata pointers via base + byte offset (avoids slice bounds issues)
            let p_packed_base = DevicePtr::device_ptr(packed_buf, &self.stream).0;
            let p_block_tables = p_packed_base + (offsets.block_tables * 4) as u64;
            let p_context_lens = p_packed_base + (offsets.context_lens * 4) as u64;
            let p_positions = p_packed_base + (offsets.positions * 4) as u64;
            let p_slot_mapping = p_packed_base + (offsets.slot_mapping * 4) as u64;
            let p_rope_cos = DevicePtr::device_ptr(&self.rope_cos, &self.stream).0;
            let p_rope_sin = DevicePtr::device_ptr(&self.rope_sin, &self.stream).0;
            let p_output = DevicePtr::device_ptr(&output_gpu, &self.stream).0;
            let i_block_size = block_size as i32;
            let i_max_ctx = max_context_len as i32;
            let i_hidden = hidden as i32;

            // smem: hidden_size * 4 + scratch (match attention needs)
            let smem = std::cmp::max(
                (hidden * 4 + 32) as u32,
                (64 * head_dim * 4 + 8 * 64 * 4 + 32) as u32,
            );

            #[allow(clippy::cast_ptr_alignment)]
            let mut args: [*mut c_void; 17] = [
                &p_tape as *const u64 as *mut c_void,
                &num_instructions as *const i32 as *mut c_void,
                &p_wptrs as *const u64 as *mut c_void,
                &p_scratch as *const u64 as *mut c_void,
                &p_sync as *const u64 as *mut c_void,
                &p_keys as *const u64 as *mut c_void,
                &p_vals as *const u64 as *mut c_void,
                &p_block_tables as *const u64 as *mut c_void,
                &p_context_lens as *const u64 as *mut c_void,
                &p_positions as *const u64 as *mut c_void,
                &p_slot_mapping as *const u64 as *mut c_void,
                &p_rope_cos as *const u64 as *mut c_void,
                &p_rope_sin as *const u64 as *mut c_void,
                &p_output as *const u64 as *mut c_void,
                &i_block_size as *const i32 as *mut c_void,
                &i_max_ctx as *const i32 as *mut c_void,
                &i_hidden as *const i32 as *mut c_void,
            ];

            if smem > 49152 {
                let cu_func = self
                    .loader
                    .get_cubin_func("megakernel_decode", "megakernel_decode_f16")?;
                unsafe {
                    cudarc::driver::sys::cuFuncSetAttribute(
                        cu_func,
                        cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem as i32,
                    ).result().map_err(|e| LLMError::GpuError(format!("mk smem attr: {e}")))?;
                }
            }

            unsafe {
                self.loader.launch_cubin_raw(
                    "megakernel_decode",
                    "megakernel_decode_f16",
                    LaunchConfig {
                        grid_dim: (256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: smem,
                    },
                    &mut args,
                )?;
            }

            // Read output token
            let output_cpu: Vec<i32> = self
                .stream
                .clone_dtoh(&output_gpu)
                .map_err(|e| LLMError::GpuError(format!("mk output dtoh: {e}")))?;

            Ok(ForwardOutput::TokenIds(output_cpu))
        }

        // ===================================================================
        // Persistent V2: per-layer cooperative TC GEMV + split-KV attention
        // ===================================================================

        fn forward_persistent_v2(
            &self,
            embedding: &CudaSlice<f16>,
            gpu_cache: &[(CudaSlice<f16>, CudaSlice<f16>)],
            packed_buf: &CudaSlice<i32>,
            offsets: PackedMetaOffsets,
            num_tokens: usize,
            _num_seqs: usize,
            max_context_len: u32,
            block_size: usize,
            greedy_only: bool,
        ) -> Result<ForwardOutput> {
            use rvllm_gpu::persistent_v2_ffi::PersistentV2Kernels;

            let pv2 = self
                .persistent_v2
                .as_ref()
                .ok_or_else(|| LLMError::GpuError("persistent_v2 not loaded".into()))?;

            let cfg = &self.config;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;
            let num_layers = self.layers.len();
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_dim = q_dim + 2 * kv_dim;
            let gate_up_dim = intermediate * 2;
            let heads_per_group = num_heads / num_kv_heads;
            let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

            let smem = PersistentV2Kernels::layer_shared_mem(hidden, head_dim, heads_per_group);

            // Query max cooperative grid size
            let max_grid = pv2
                .layer_max_grid(smem)
                .map_err(|e| LLMError::GpuError(e))?;
            let grid_blocks = max_grid.min(256);

            // Split-KV config: max 8 splits (256 blocks / 32 kv_heads)
            let max_splits = (grid_blocks as usize / num_kv_heads).min(8).max(1) as i32;

            // Allocate per-layer scratch (reused across layers)
            let mut qkv_scratch = unsafe { self.stream.alloc::<f16>(qkv_dim) }
                .map_err(|e| LLMError::GpuError(format!("pv2 alloc: {e}")))?;
            let mut attn_scratch = unsafe { self.stream.alloc::<f16>(q_dim) }
                .map_err(|e| LLMError::GpuError(format!("pv2 alloc: {e}")))?;
            let mut oproj_scratch = unsafe { self.stream.alloc::<f16>(hidden) }
                .map_err(|e| LLMError::GpuError(format!("pv2 alloc: {e}")))?;
            let mut gateup_scratch = unsafe { self.stream.alloc::<f16>(gate_up_dim) }
                .map_err(|e| LLMError::GpuError(format!("pv2 alloc: {e}")))?;

            // Split-KV scratch
            let (max_sz, sum_sz, acc_sz) = PersistentV2Kernels::split_kv_scratch_size(
                num_kv_heads,
                max_splits as usize,
                heads_per_group,
                head_dim,
            );
            let split_max_buf = unsafe { self.stream.alloc::<u8>(max_sz) }
                .map_err(|e| LLMError::GpuError(format!("pv2 split_max: {e}")))?;
            let split_sum_buf = unsafe { self.stream.alloc::<u8>(sum_sz) }
                .map_err(|e| LLMError::GpuError(format!("pv2 split_sum: {e}")))?;
            let split_acc_buf = unsafe { self.stream.alloc::<u8>(acc_sz) }
                .map_err(|e| LLMError::GpuError(format!("pv2 split_acc: {e}")))?;

            // Sync flags (zeroed before each layer launch)
            let sync_flags = self
                .stream
                .clone_htod(&[0i32; 6])
                .map_err(|e| LLMError::GpuError(format!("pv2 sync: {e}")))?;

            // Double-buffer residuals
            let mut residual_a = unsafe { self.stream.alloc::<f16>(hidden) }
                .map_err(|e| LLMError::GpuError(format!("pv2 alloc: {e}")))?;
            let mut residual_b = unsafe { self.stream.alloc::<f16>(hidden) }
                .map_err(|e| LLMError::GpuError(format!("pv2 alloc: {e}")))?;
            let mut mlp_a = unsafe { self.stream.alloc::<f16>(hidden) }
                .map_err(|e| LLMError::GpuError(format!("pv2 alloc: {e}")))?;
            let mut mlp_b = unsafe { self.stream.alloc::<f16>(hidden) }
                .map_err(|e| LLMError::GpuError(format!("pv2 alloc: {e}")))?;

            // Copy embedding into residual_a
            self.stream
                .memcpy_dtod(embedding, &mut residual_a.slice_mut(..hidden))
                .map_err(|e| LLMError::GpuError(format!("pv2 embed copy: {e}")))?;

            let max_blocks = (self.graph_max_blocks) as i32;
            let cu_stream = self.stream.cu_stream();

            // Metadata pointers
            let p_packed_base = DevicePtr::device_ptr(packed_buf, &self.stream).0;
            let p_block_tables = p_packed_base + (offsets.block_tables * 4) as u64;
            let p_context_lens = p_packed_base + (offsets.context_lens * 4) as u64;
            let p_positions = p_packed_base + (offsets.positions * 4) as u64;
            let p_slot_mapping = p_packed_base + (offsets.slot_mapping * 4) as u64;
            let p_rope_cos = DevicePtr::device_ptr(&self.rope_cos, &self.stream).0;
            let p_rope_sin = DevicePtr::device_ptr(&self.rope_sin, &self.stream).0;

            let p_qkv_scratch = DevicePtr::device_ptr(&qkv_scratch, &self.stream).0;
            let p_attn_scratch = DevicePtr::device_ptr(&attn_scratch, &self.stream).0;
            let p_oproj_scratch = DevicePtr::device_ptr(&oproj_scratch, &self.stream).0;
            let p_gateup_scratch = DevicePtr::device_ptr(&gateup_scratch, &self.stream).0;
            let p_split_max = DevicePtr::device_ptr(&split_max_buf, &self.stream).0;
            let p_split_sum = DevicePtr::device_ptr(&split_sum_buf, &self.stream).0;
            let p_split_acc = DevicePtr::device_ptr(&split_acc_buf, &self.stream).0;
            let p_sync = DevicePtr::device_ptr(&sync_flags, &self.stream).0;

            for layer_idx in 0..num_layers {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let w = self.layer_weights(layer_idx)?;
                let fused_qkv = w
                    .fused_qkv
                    .ok_or_else(|| LLMError::GpuError("PersistentV2 requires fused_qkv".into()))?;
                let fused_gate_up = w.fused_gate_up.ok_or_else(|| {
                    LLMError::GpuError("PersistentV2 requires fused_gate_up".into())
                })?;

                let even = layer_idx % 2 == 0;
                let (read_res, write_res) = if even {
                    (&residual_a, &residual_b)
                } else {
                    (&residual_b, &residual_a)
                };
                let (read_mlp, write_mlp) = if even {
                    (&mlp_a, &mlp_b)
                } else {
                    (&mlp_b, &mlp_a)
                };

                let p_prev_residual = DevicePtr::device_ptr(read_res, &self.stream).0;
                let p_residual_out = DevicePtr::device_ptr(write_res, &self.stream).0;
                let p_prev_mlp = if layer_idx > 0 {
                    DevicePtr::device_ptr(read_mlp, &self.stream).0
                } else {
                    0u64
                };
                let p_mlp_out = DevicePtr::device_ptr(write_mlp, &self.stream).0;

                // Zero sync flags before each layer
                unsafe {
                    cudarc::driver::sys::cuMemsetD32Async(p_sync, 0, 6, cu_stream)
                        .result()
                        .map_err(|e| LLMError::GpuError(format!("pv2 zero sync: {e}")))?;
                }

                pv2.launch_layer(
                    cu_stream,
                    grid_blocks,
                    smem,
                    p_mlp_out,
                    p_residual_out,
                    p_prev_residual,
                    p_prev_mlp,
                    DevicePtr::device_ptr(key_cache, &self.stream).0,
                    DevicePtr::device_ptr(value_cache, &self.stream).0,
                    p_block_tables,
                    p_context_lens,
                    p_positions,
                    p_slot_mapping,
                    p_rope_cos,
                    p_rope_sin,
                    DevicePtr::device_ptr(w.input_layernorm, &self.stream).0,
                    DevicePtr::device_ptr(fused_qkv, &self.stream).0,
                    w.qkv_bias
                        .map_or(0u64, |b| DevicePtr::device_ptr(b, &self.stream).0),
                    DevicePtr::device_ptr(w.o_proj, &self.stream).0,
                    DevicePtr::device_ptr(w.post_attention_layernorm, &self.stream).0,
                    DevicePtr::device_ptr(fused_gate_up, &self.stream).0,
                    DevicePtr::device_ptr(w.down_proj, &self.stream).0,
                    p_qkv_scratch,
                    p_attn_scratch,
                    p_oproj_scratch,
                    p_gateup_scratch,
                    p_split_max,
                    p_split_sum,
                    p_split_acc,
                    max_splits,
                    self.rms_norm_eps,
                    attn_scale,
                    hidden as i32,
                    q_dim as i32,
                    kv_dim as i32,
                    qkv_dim as i32,
                    num_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    intermediate as i32,
                    gate_up_dim as i32,
                    block_size as i32,
                    max_context_len as i32,
                    max_blocks,
                    p_sync,
                )
                .map_err(|e| LLMError::GpuError(e))?;
            }

            // After all layers: final residual is in write buffer of last layer
            let even_last = (num_layers - 1) % 2 == 0;
            let final_residual = if even_last { &residual_b } else { &residual_a };
            let final_mlp = if even_last { &mlp_b } else { &mlp_a };

            // Final: fuse last residual add + RMSNorm
            let (normed_f16, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                &self.stream,
                &self.loader,
                final_residual,
                final_mlp,
                &self.final_norm_weight,
                self.rms_norm_eps,
                num_tokens,
                hidden,
            )?;

            // LM head + argmax
            let vocab_size = self.config.vocab_size;
            if greedy_only {
                let token_ids_gpu = self.gpu_fused_lm_head_argmax_f16_hidden(
                    &normed_f16,
                    &self.lm_head_weight,
                    vocab_size,
                    hidden,
                )?;
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("pv2 argmax dtoh: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }
            let logits_gpu = CudaLinearLayer::forward_f16_in(
                &normed_f16,
                &self.lm_head_weight,
                1,
                vocab_size,
                hidden,
                &self.blas,
            )?;
            let logits_cpu = self
                .stream
                .clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("pv2 logits dtoh: {e}")))?;
            Ok(ForwardOutput::Logits(logits_cpu))
        }

        // ===================================================================
        // Persistent V3: non-cooperative, 1024 blocks, vectorized GEMV, BC=32
        // ===================================================================

        fn forward_persistent_v3(
            &self,
            embedding: &CudaSlice<f16>,
            gpu_cache: &[(CudaSlice<f16>, CudaSlice<f16>)],
            packed_buf: &CudaSlice<i32>,
            offsets: PackedMetaOffsets,
            num_tokens: usize,
            _num_seqs: usize,
            max_context_len: u32,
            block_size: usize,
            greedy_only: bool,
        ) -> Result<ForwardOutput> {
            let normed_f16 = self.forward_persistent_v3_hidden(
                embedding,
                gpu_cache,
                packed_buf,
                offsets,
                num_tokens,
                max_context_len,
                block_size,
            )?;
            let vocab_size = self.config.vocab_size;
            let hidden = self.config.hidden_size;
            if greedy_only {
                let token_ids_gpu = self.gpu_fused_lm_head_argmax_f16_hidden(
                    &normed_f16,
                    &self.lm_head_weight,
                    vocab_size,
                    hidden,
                )?;
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("pv3 argmax dtoh: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }
            let logits_gpu = CudaLinearLayer::forward_f16_in(
                &normed_f16,
                &self.lm_head_weight,
                num_tokens,
                vocab_size,
                hidden,
                &self.blas,
            )?;
            let logits_cpu = self
                .stream
                .clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("pv3 logits dtoh: {e}")))?;
            Ok(ForwardOutput::Logits(logits_cpu))
        }

        fn forward_persistent_v3_graph(
            &self,
            embedding: &CudaSlice<f16>,
            gpu_cache: &[(CudaSlice<f16>, CudaSlice<f16>)],
            packed_buf: &CudaSlice<i32>,
            offsets: PackedMetaOffsets,
            num_tokens: usize,
            _num_seqs: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<()> {
            let normed_f16 = self.forward_persistent_v3_hidden(
                embedding,
                gpu_cache,
                packed_buf,
                offsets,
                num_tokens,
                max_context_len,
                block_size,
            )?;
            let token_ids_gpu = self.gpu_fused_lm_head_argmax_f16_hidden(
                &normed_f16,
                &self.lm_head_weight,
                self.config.vocab_size,
                self.config.hidden_size,
            )?;
            self.write_graph_output(&token_ids_gpu, num_tokens)
        }

        fn forward_persistent_v3_hidden(
            &self,
            embedding: &CudaSlice<f16>,
            gpu_cache: &[(CudaSlice<f16>, CudaSlice<f16>)],
            packed_buf: &CudaSlice<i32>,
            offsets: PackedMetaOffsets,
            num_tokens: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f16>> {
            use rvllm_gpu::persistent_v2_ffi::PersistentV2Kernels;

            let pv2 = self
                .persistent_v2
                .as_ref()
                .ok_or_else(|| LLMError::GpuError("persistent_v3 not loaded".into()))?;

            let cfg = &self.config;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;
            let num_layers = self.layers.len();
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_dim = q_dim + 2 * kv_dim;
            let gate_up_dim = intermediate * 2;
            let heads_per_group = num_heads / num_kv_heads;
            let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

            let smem =
                PersistentV2Kernels::v3_shared_mem(hidden, head_dim, heads_per_group, intermediate);

            // V3: non-cooperative, up to 1024 blocks
            let grid_blocks = pv2.v3_max_grid(smem).map_err(|e| LLMError::GpuError(e))?;

            // Split-KV config
            let max_splits = (grid_blocks as usize / num_kv_heads).min(8).max(1) as i32;

            self.ensure_persistent_v3_scratch(max_splits as usize)?;
            let mut scratch_borrow = self.persistent_v3_scratch.borrow_mut();
            let scratch = scratch_borrow
                .as_mut()
                .ok_or_else(|| LLMError::GpuError("persistent_v3 scratch missing".into()))?;

            // Copy embedding into residual_a
            self.stream
                .memcpy_dtod(embedding, &mut scratch.residual_a.slice_mut(..hidden))
                .map_err(|e| LLMError::GpuError(format!("pv3 embed copy: {e}")))?;

            let max_blocks = (self.graph_max_blocks) as i32;
            let cu_stream = self.stream.cu_stream();

            // Metadata pointers
            let p_packed_base = DevicePtr::device_ptr(packed_buf, &self.stream).0;
            let p_block_tables = p_packed_base + (offsets.block_tables * 4) as u64;
            let p_context_lens = p_packed_base + (offsets.context_lens * 4) as u64;
            let p_positions = p_packed_base + (offsets.positions * 4) as u64;
            let p_slot_mapping = p_packed_base + (offsets.slot_mapping * 4) as u64;
            let p_rope_cos = DevicePtr::device_ptr(&self.rope_cos, &self.stream).0;
            let p_rope_sin = DevicePtr::device_ptr(&self.rope_sin, &self.stream).0;

            let p_qkv_scratch = DevicePtr::device_ptr(&scratch.qkv, &self.stream).0;
            let p_attn_scratch = DevicePtr::device_ptr(&scratch.attn, &self.stream).0;
            let p_oproj_scratch = DevicePtr::device_ptr(&scratch.oproj, &self.stream).0;
            let p_gateup_scratch = DevicePtr::device_ptr(&scratch.gateup, &self.stream).0;
            let p_split_max = DevicePtr::device_ptr(&scratch.split_max, &self.stream).0;
            let p_split_sum = DevicePtr::device_ptr(&scratch.split_sum, &self.stream).0;
            let p_split_acc = DevicePtr::device_ptr(&scratch.split_acc, &self.stream).0;
            let p_sync = DevicePtr::device_ptr(&scratch.sync_flags, &self.stream).0;

            debug!(grid_blocks, smem, max_splits, "persistent_v3 launching");

            for layer_idx in 0..num_layers {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let w = self.layer_weights(layer_idx)?;
                let fused_qkv = w
                    .fused_qkv
                    .ok_or_else(|| LLMError::GpuError("PersistentV3 requires fused_qkv".into()))?;
                let fused_gate_up = w.fused_gate_up.ok_or_else(|| {
                    LLMError::GpuError("PersistentV3 requires fused_gate_up".into())
                })?;

                let even = layer_idx % 2 == 0;
                let (read_res, write_res) = if even {
                    (&scratch.residual_a, &scratch.residual_b)
                } else {
                    (&scratch.residual_b, &scratch.residual_a)
                };
                let (read_mlp, write_mlp) = if even {
                    (&scratch.mlp_a, &scratch.mlp_b)
                } else {
                    (&scratch.mlp_b, &scratch.mlp_a)
                };

                let p_prev_residual = DevicePtr::device_ptr(read_res, &self.stream).0;
                let p_residual_out = DevicePtr::device_ptr(write_res, &self.stream).0;
                let p_prev_mlp = if layer_idx > 0 {
                    DevicePtr::device_ptr(read_mlp, &self.stream).0
                } else {
                    0u64
                };
                let p_mlp_out = DevicePtr::device_ptr(write_mlp, &self.stream).0;

                // Zero sync flags before each layer
                unsafe {
                    cudarc::driver::sys::cuMemsetD32Async(p_sync, 0, 6, cu_stream)
                        .result()
                        .map_err(|e| LLMError::GpuError(format!("pv3 zero sync: {e}")))?;
                }

                pv2.launch_v3_layer(
                    cu_stream,
                    grid_blocks,
                    smem,
                    p_mlp_out,
                    p_residual_out,
                    p_prev_residual,
                    p_prev_mlp,
                    DevicePtr::device_ptr(key_cache, &self.stream).0,
                    DevicePtr::device_ptr(value_cache, &self.stream).0,
                    p_block_tables,
                    p_context_lens,
                    p_positions,
                    p_slot_mapping,
                    p_rope_cos,
                    p_rope_sin,
                    DevicePtr::device_ptr(w.input_layernorm, &self.stream).0,
                    DevicePtr::device_ptr(fused_qkv, &self.stream).0,
                    w.qkv_bias
                        .map_or(0u64, |b| DevicePtr::device_ptr(b, &self.stream).0),
                    DevicePtr::device_ptr(w.o_proj, &self.stream).0,
                    DevicePtr::device_ptr(w.post_attention_layernorm, &self.stream).0,
                    DevicePtr::device_ptr(fused_gate_up, &self.stream).0,
                    DevicePtr::device_ptr(w.down_proj, &self.stream).0,
                    p_qkv_scratch,
                    p_attn_scratch,
                    p_oproj_scratch,
                    p_gateup_scratch,
                    p_split_max,
                    p_split_sum,
                    p_split_acc,
                    max_splits,
                    self.rms_norm_eps,
                    attn_scale,
                    hidden as i32,
                    q_dim as i32,
                    kv_dim as i32,
                    qkv_dim as i32,
                    num_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    intermediate as i32,
                    gate_up_dim as i32,
                    block_size as i32,
                    max_context_len as i32,
                    max_blocks,
                    p_sync,
                )
                .map_err(|e| LLMError::GpuError(e))?;
            }

            // After all layers: final residual is in write buffer of last layer
            let even_last = (num_layers - 1) % 2 == 0;
            let final_residual = if even_last {
                &scratch.residual_b
            } else {
                &scratch.residual_a
            };
            let final_mlp = if even_last {
                &scratch.mlp_b
            } else {
                &scratch.mlp_a
            };

            // Final: fuse last residual add + RMSNorm
            let (normed_f16, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                &self.stream,
                &self.loader,
                final_residual,
                final_mlp,
                &self.final_norm_weight,
                self.rms_norm_eps,
                1,
                hidden,
            )?;

            Ok(normed_f16)
        }

        // ===================================================================
        // Megakernel V2: all layers + embed + LM head in one cooperative launch
        // ===================================================================

        fn forward_megakernel_v2(
            &self,
            _embedding: &CudaSlice<f16>,
            gpu_cache: &[(CudaSlice<f16>, CudaSlice<f16>)],
            packed_buf: &CudaSlice<i32>,
            offsets: PackedMetaOffsets,
            num_tokens: usize,
            _num_seqs: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<ForwardOutput> {
            use rvllm_gpu::persistent_v2_ffi::PersistentV2Kernels;

            let pv2 = self
                .persistent_v2
                .as_ref()
                .ok_or_else(|| LLMError::GpuError("persistent_v2 not loaded".into()))?;

            let cfg = &self.config;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;
            let vocab = cfg.vocab_size;
            let num_layers = self.layers.len();
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_dim = q_dim + 2 * kv_dim;
            let gate_up_dim = intermediate * 2;
            let heads_per_group = num_heads / num_kv_heads;
            let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

            let smem = PersistentV2Kernels::layer_shared_mem(hidden, head_dim, heads_per_group);
            let max_grid = pv2.mega_max_grid(smem).map_err(|e| LLMError::GpuError(e))?;
            let grid_blocks = max_grid.min(256);
            let max_splits = (grid_blocks as usize / num_kv_heads).min(8).max(1) as i32;

            // Build 7 per-layer weight pointer arrays
            let mut norm_ptrs = Vec::with_capacity(num_layers);
            let mut qkv_ptrs = Vec::with_capacity(num_layers);
            let mut bias_ptrs = Vec::with_capacity(num_layers);
            let mut o_ptrs = Vec::with_capacity(num_layers);
            let mut post_norm_ptrs = Vec::with_capacity(num_layers);
            let mut gateup_ptrs = Vec::with_capacity(num_layers);
            let mut down_ptrs = Vec::with_capacity(num_layers);

            for i in 0..num_layers {
                let w = self.layer_weights(i)?;
                let fused_qkv = w
                    .fused_qkv
                    .ok_or_else(|| LLMError::GpuError("MegakernelV2 requires fused_qkv".into()))?;
                let fused_gate_up = w.fused_gate_up.ok_or_else(|| {
                    LLMError::GpuError("MegakernelV2 requires fused_gate_up".into())
                })?;

                norm_ptrs.push(DevicePtr::device_ptr(w.input_layernorm, &self.stream).0);
                qkv_ptrs.push(DevicePtr::device_ptr(fused_qkv, &self.stream).0);
                bias_ptrs.push(
                    w.qkv_bias
                        .map_or(0u64, |b| DevicePtr::device_ptr(b, &self.stream).0),
                );
                o_ptrs.push(DevicePtr::device_ptr(w.o_proj, &self.stream).0);
                post_norm_ptrs
                    .push(DevicePtr::device_ptr(w.post_attention_layernorm, &self.stream).0);
                gateup_ptrs.push(DevicePtr::device_ptr(fused_gate_up, &self.stream).0);
                down_ptrs.push(DevicePtr::device_ptr(w.down_proj, &self.stream).0);
            }

            // Upload all pointer arrays to GPU
            let g_norm = self
                .stream
                .clone_htod(&norm_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk2 upload: {e}")))?;
            let g_qkv = self
                .stream
                .clone_htod(&qkv_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk2 upload: {e}")))?;
            let g_bias = self
                .stream
                .clone_htod(&bias_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk2 upload: {e}")))?;
            let g_o = self
                .stream
                .clone_htod(&o_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk2 upload: {e}")))?;
            let g_post = self
                .stream
                .clone_htod(&post_norm_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk2 upload: {e}")))?;
            let g_gateup = self
                .stream
                .clone_htod(&gateup_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk2 upload: {e}")))?;
            let g_down = self
                .stream
                .clone_htod(&down_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk2 upload: {e}")))?;

            // Per-layer KV cache pointer arrays
            let mut key_ptrs = Vec::with_capacity(num_layers);
            let mut val_ptrs = Vec::with_capacity(num_layers);
            for (kc, vc) in gpu_cache.iter().take(num_layers) {
                key_ptrs.push(DevicePtr::device_ptr(kc, &self.stream).0);
                val_ptrs.push(DevicePtr::device_ptr(vc, &self.stream).0);
            }
            let g_keys = self
                .stream
                .clone_htod(&key_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk2 upload: {e}")))?;
            let g_vals = self
                .stream
                .clone_htod(&val_ptrs)
                .map_err(|e| LLMError::GpuError(format!("mk2 upload: {e}")))?;

            // Scratch buffers
            let (h_sz, m_sz, qkv_sz, a_sz, o_sz, gu_sz, l_sz) =
                PersistentV2Kernels::megakernel_scratch_elems(
                    hidden,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    intermediate,
                    vocab,
                );
            let hidden_buf = unsafe { self.stream.alloc::<f16>(h_sz) }
                .map_err(|e| LLMError::GpuError(format!("mk2 alloc: {e}")))?;
            let mlp_buf = unsafe { self.stream.alloc::<f16>(m_sz) }
                .map_err(|e| LLMError::GpuError(format!("mk2 alloc: {e}")))?;
            let qkv_scratch = unsafe { self.stream.alloc::<f16>(qkv_sz) }
                .map_err(|e| LLMError::GpuError(format!("mk2 alloc: {e}")))?;
            let attn_scratch = unsafe { self.stream.alloc::<f16>(a_sz) }
                .map_err(|e| LLMError::GpuError(format!("mk2 alloc: {e}")))?;
            let oproj_scratch = unsafe { self.stream.alloc::<f16>(o_sz) }
                .map_err(|e| LLMError::GpuError(format!("mk2 alloc: {e}")))?;
            let gateup_scratch = unsafe { self.stream.alloc::<f16>(gu_sz) }
                .map_err(|e| LLMError::GpuError(format!("mk2 alloc: {e}")))?;
            let logits_scratch = unsafe { self.stream.alloc::<f16>(l_sz) }
                .map_err(|e| LLMError::GpuError(format!("mk2 alloc: {e}")))?;

            // Split-KV scratch
            let (max_sz, sum_sz, acc_sz) = PersistentV2Kernels::split_kv_scratch_size(
                num_kv_heads,
                max_splits as usize,
                heads_per_group,
                head_dim,
            );
            let split_max_buf = unsafe { self.stream.alloc::<u8>(max_sz) }
                .map_err(|e| LLMError::GpuError(format!("mk2 alloc: {e}")))?;
            let split_sum_buf = unsafe { self.stream.alloc::<u8>(sum_sz) }
                .map_err(|e| LLMError::GpuError(format!("mk2 alloc: {e}")))?;
            let split_acc_buf = unsafe { self.stream.alloc::<u8>(acc_sz) }
                .map_err(|e| LLMError::GpuError(format!("mk2 alloc: {e}")))?;

            // Sync flags (zeroed)
            let sync_count = PersistentV2Kernels::megakernel_sync_flags_size(num_layers) / 4;
            let sync_flags = self
                .stream
                .clone_htod(&vec![0i32; sync_count])
                .map_err(|e| LLMError::GpuError(format!("mk2 sync: {e}")))?;

            // Input token (from packed metadata, positions[0] area has the token_id)
            let p_packed_base = DevicePtr::device_ptr(packed_buf, &self.stream).0;
            let p_input_token = p_packed_base; // token_ids are at offset 0
            let p_positions = p_packed_base + (offsets.positions * 4) as u64;
            let p_slot_mapping = p_packed_base + (offsets.slot_mapping * 4) as u64;
            let p_block_tables = p_packed_base + (offsets.block_tables * 4) as u64;
            let p_context_lens = p_packed_base + (offsets.context_lens * 4) as u64;

            // Output token
            let output_gpu = self
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| LLMError::GpuError(format!("mk2 output: {e}")))?;

            let max_blocks = self.graph_max_blocks as i32;

            pv2.launch_megakernel(
                self.stream.cu_stream(),
                grid_blocks,
                smem,
                // Per-layer weight arrays
                DevicePtr::device_ptr(&g_norm, &self.stream).0,
                DevicePtr::device_ptr(&g_qkv, &self.stream).0,
                DevicePtr::device_ptr(&g_bias, &self.stream).0,
                DevicePtr::device_ptr(&g_o, &self.stream).0,
                DevicePtr::device_ptr(&g_post, &self.stream).0,
                DevicePtr::device_ptr(&g_gateup, &self.stream).0,
                DevicePtr::device_ptr(&g_down, &self.stream).0,
                // KV caches
                DevicePtr::device_ptr(&g_keys, &self.stream).0,
                DevicePtr::device_ptr(&g_vals, &self.stream).0,
                // Global weights
                DevicePtr::device_ptr(&self.embed_tokens, &self.stream).0,
                DevicePtr::device_ptr(&self.final_norm_weight, &self.stream).0,
                DevicePtr::device_ptr(&self.lm_head_weight, &self.stream).0,
                // Input metadata
                p_input_token,
                p_positions,
                p_slot_mapping,
                p_block_tables,
                p_context_lens,
                DevicePtr::device_ptr(&self.rope_cos, &self.stream).0,
                DevicePtr::device_ptr(&self.rope_sin, &self.stream).0,
                // Scratch
                DevicePtr::device_ptr(&hidden_buf, &self.stream).0,
                DevicePtr::device_ptr(&mlp_buf, &self.stream).0,
                DevicePtr::device_ptr(&qkv_scratch, &self.stream).0,
                DevicePtr::device_ptr(&attn_scratch, &self.stream).0,
                DevicePtr::device_ptr(&oproj_scratch, &self.stream).0,
                DevicePtr::device_ptr(&gateup_scratch, &self.stream).0,
                DevicePtr::device_ptr(&logits_scratch, &self.stream).0,
                // Split-KV
                DevicePtr::device_ptr(&split_max_buf, &self.stream).0,
                DevicePtr::device_ptr(&split_sum_buf, &self.stream).0,
                DevicePtr::device_ptr(&split_acc_buf, &self.stream).0,
                max_splits,
                // Sync
                DevicePtr::device_ptr(&sync_flags, &self.stream).0,
                // Config
                num_layers as i32,
                hidden as i32,
                q_dim as i32,
                kv_dim as i32,
                qkv_dim as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                intermediate as i32,
                gate_up_dim as i32,
                block_size as i32,
                max_context_len as i32,
                max_blocks,
                vocab as i32,
                self.rms_norm_eps,
                attn_scale,
                // Output
                DevicePtr::device_ptr(&output_gpu, &self.stream).0,
            )
            .map_err(|e| LLMError::GpuError(e))?;

            let output_cpu: Vec<i32> = self
                .stream
                .clone_dtoh(&output_gpu)
                .map_err(|e| LLMError::GpuError(format!("mk2 output dtoh: {e}")))?;

            Ok(ForwardOutput::TokenIds(output_cpu))
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
                fused_qkv_fp8: self.fp8_fused_qkv.get(i),
                fused_qkv_fp8_scale: self.fp8_fused_qkv_scale.get(i),
                o_proj_fp8: self.fp8_o_proj.get(i),
                o_proj_fp8_scale: self.fp8_o_proj_scale.get(i),
                fused_gate_up_fp8: self.fp8_fused_gate_up.get(i),
                fused_gate_up_fp8_scale: self.fp8_fused_gate_up_scale.get(i),
                down_proj_fp8: self.fp8_down_proj.get(i),
                down_proj_fp8_scale: self.fp8_down_proj_scale.get(i),
            })
        }

        /// Embedding lookup using the pre-uploaded token IDs in the packed
        /// metadata buffer. Call upload_metadata() first to populate.
        fn embedding_lookup_from_meta(&self, num_tokens: usize) -> Result<CudaSlice<f16>> {
            let hidden_size = self.config.hidden_size;

            let kernel = self
                .loader
                .get_func("embedding_gather_f16", "embedding_gather_f16_kernel")?;

            // Safety: embedding gather kernel writes all num_tokens * hidden_size elements
            let output = unsafe { self.stream.alloc::<f16>(num_tokens * hidden_size) }
                .map_err(|e| LLMError::GpuError(format!("embed alloc: {e}")))?;

            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            let token_ids_view =
                packed_buf.slice(offsets.token_ids..offsets.token_ids + offsets.num_token_ids);

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
            let kernel = self
                .loader
                .get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                self.stream
                    .launch_builder(&kernel)
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
                self.stream
                    .launch_builder(&cast_kernel)
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
    use super::{
        DecodeExecutionPlan, DecodeGraphAction, DecodeGraphDispatch, DecodeGraphRuntimeState,
        ForwardOutput,
    };
    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

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

        pub fn profile_decode_bucket(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _attn_meta: &crate::bridge::AttentionMetadata,
            _greedy_only: bool,
            _target_bucket: usize,
        ) -> Result<ForwardOutput> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }

        pub fn decode_execution_plan(
            &self,
            num_tokens: usize,
            _is_prefill: bool,
            _is_pure_decode: bool,
        ) -> DecodeExecutionPlan {
            DecodeExecutionPlan {
                actual_tokens: num_tokens,
                graph_tokens: num_tokens,
                use_graphed_decode: false,
                use_batched_v2: false,
            }
        }

        pub fn decode_graph_dispatch(
            &self,
            num_tokens: usize,
            is_prefill: bool,
            is_pure_decode: bool,
            _state: DecodeGraphRuntimeState,
        ) -> DecodeGraphDispatch {
            DecodeGraphDispatch {
                execution: self.decode_execution_plan(num_tokens, is_prefill, is_pure_decode),
                action: DecodeGraphAction::Raw,
            }
        }

        pub fn upload_decode_metadata_v2(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _attn_meta: &crate::bridge::AttentionMetadata,
            _padded_batch: usize,
        ) -> Result<()> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
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

        pub fn forward_partial(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _attn_meta: &crate::bridge::AttentionMetadata,
            _is_prefill: bool,
            _max_layers: usize,
        ) -> Result<Vec<f32>> {
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
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
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
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                dtype: "float16".to_string(),
                architecture: "LlamaForCausalLM".to_string(),
            };
            let runner = GpuModelRunner { config };
            assert_eq!(runner.config().num_layers, 4);
            assert_eq!(runner.config().vocab_size, 32000);
        }
    }
}
