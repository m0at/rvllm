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
    /// Compatibility variant for worker async graph paths.
    TokenIdsPending { actual_batch: usize },
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::cell::RefCell;
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use half::f16;
    use tracing::{debug, info, trace};

    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

    use crate::gpu_layer::{
        GptOssMoeLayerWeights, GpuLayerConfig, GpuLayerInput, GpuLayerWeights, GpuLayerWeightsF16,
        GpuTransformerLayer,
    };
    use crate::layers::linear_cuda::CudaLinearLayer;
    use crate::layers::norm_cuda::CudaRMSNorm;
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

    pub struct GpuModelRunner {
        weights: GpuModelWeights,
        cache: CudaCacheEngine,
        blas: CublasHandle,
        loader: Arc<KernelLoader>,
        config: ModelRunnerConfig,
        device: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        layers: Vec<GpuTransformerLayer>,
        embed_tokens: CudaSlice<f32>,
        final_norm_weight: CudaSlice<f32>,
        lm_head_weight: CudaSlice<f32>,
        rms_norm_eps: f32,
        /// Precomputed RoPE cos table on GPU: [max_position, head_dim/2]
        rope_cos: CudaSlice<f32>,
        /// Precomputed RoPE sin table on GPU: [max_position, head_dim/2]
        rope_sin: CudaSlice<f32>,
        /// When true, use hgemm with f16 projection weights instead of sgemm.
        use_fp16: bool,
        /// Pre-allocated GPU buffers for per-step metadata, reused across
        /// forward calls to avoid CUDA malloc/free on the decode hot path.
        /// Interior mutability via RefCell since forward_ex takes &self.
        meta_positions: RefCell<ReusableGpuBuf>,
        meta_context_lens: RefCell<ReusableGpuBuf>,
        meta_block_tables: RefCell<ReusableGpuBuf>,
        meta_slot_mapping: RefCell<ReusableGpuBuf>,
        meta_seq_start_pos: RefCell<ReusableGpuBuf>,
        /// Pre-allocated GPU buffer for token IDs (embedding lookup input).
        /// Using a persistent buffer ensures the GPU pointer is stable across
        /// forward calls, which is required for CUDA graph capture/replay.
        meta_token_ids: RefCell<ReusableGpuBuf>,
        /// Persistent GPU output buffer for CUDA graph capture/replay.
        /// Stores the argmax token IDs from the last forward pass. The graph
        /// writes to this buffer's stable pointer; after replay we DtoH copy
        /// from here to read updated results.
        graph_output: RefCell<Option<CudaSlice<i32>>>,
        /// Reusable CPU scratch buffer for packing metadata (avoids per-step
        /// heap allocations for the common small-batch decode case).
        cpu_scratch: RefCell<Vec<i32>>,
        /// Fixed max blocks per sequence for CUDA graph capture/replay.
        /// Computed as ceil(max_position / block_size) so the block_tables
        /// stride is constant across all forward calls, preventing stale
        /// scalar kernel args when a graph is replayed.
        graph_max_blocks: usize,
        /// Optional host-side GPT-OSS routed-expert weights, one slot per layer.
        gpt_oss_moe_layers: Vec<Option<GptOssMoeLayerWeights>>,
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
            let gpt_oss_moe_layers =
                Self::build_gpt_oss_moe_layers(&weights, &config, &stream)?;
            info!(graph_max_blocks, block_size, max_position = config.max_position,
                  "fixed block_tables stride for CUDA graph stability");

            Ok(Self {
                weights,
                cache,
                blas,
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
                use_fp16: false,
                meta_positions: RefCell::new(ReusableGpuBuf::new()),
                meta_context_lens: RefCell::new(ReusableGpuBuf::new()),
                meta_block_tables: RefCell::new(ReusableGpuBuf::new()),
                meta_slot_mapping: RefCell::new(ReusableGpuBuf::new()),
                meta_seq_start_pos: RefCell::new(ReusableGpuBuf::new()),
                meta_token_ids: RefCell::new(ReusableGpuBuf::new()),
                graph_output: RefCell::new(None),
                cpu_scratch: RefCell::new(Vec::with_capacity(4096)),
                graph_max_blocks,
                gpt_oss_moe_layers,
            })
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
                ForwardOutput::TokenIds(_) => unreachable!("greedy_only=false must return Logits"),
                ForwardOutput::TokenIdsPending { .. } => Err(LLMError::GpuError(
                    "greedy_only=false returned pending token IDs".into(),
                )),
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

            // Upload per-step metadata into reusable GPU buffers.
            // On the decode hot path, buffer sizes are stable so this is pure
            // memcpy_htod with zero CUDA malloc/free. The CPU scratch vec is
            // also reused to avoid per-step heap allocations.
            // Use fixed stride for block_tables so the layout matches what
            // CUDA graph kernels expect (p_max_blocks is baked as a scalar).
            let max_blocks = self.graph_max_blocks;
            let max_context_len = attn_meta.max_context_len;

            {
                let mut scratch = self.cpu_scratch.borrow_mut();

                // positions (i32)
                scratch.clear();
                scratch.extend(positions.iter().map(|&p| p as i32));
                self.meta_positions.borrow_mut().upload(&scratch, &self.stream)
                    .map_err(|e| LLMError::GpuError(format!("positions HtoD: {e}")))?;

                // context_lens (i32)
                scratch.clear();
                scratch.extend(attn_meta.context_lens.iter().map(|&c| c as i32));
                self.meta_context_lens.borrow_mut().upload(&scratch, &self.stream)
                    .map_err(|e| LLMError::GpuError(format!("context_lens HtoD: {e}")))?;

                // block_tables flattened [num_seqs, graph_max_blocks] (i32)
                scratch.clear();
                scratch.resize(num_seqs * max_blocks, 0i32);
                for (s, row) in attn_meta.block_tables.iter().enumerate() {
                    for (b, &blk) in row.iter().enumerate() {
                        scratch[s * max_blocks + b] = blk as i32;
                    }
                }
                self.meta_block_tables.borrow_mut().upload(&scratch, &self.stream)
                    .map_err(|e| LLMError::GpuError(format!("block_tables HtoD: {e}")))?;

                // slot_mapping (i32)
                scratch.clear();
                scratch.extend(attn_meta.slot_mapping.iter().map(|&s| s as i32));
                self.meta_slot_mapping.borrow_mut().upload(&scratch, &self.stream)
                    .map_err(|e| LLMError::GpuError(format!("slot_mapping HtoD: {e}")))?;

                // seq_start_pos from query_lens [num_seqs + 1] (i32)
                scratch.clear();
                let mut pos = 0i32;
                for &ql in &attn_meta.query_lens {
                    scratch.push(pos);
                    pos += ql as i32;
                }
                scratch.push(num_tokens as i32);
                self.meta_seq_start_pos.borrow_mut().upload(&scratch, &self.stream)
                    .map_err(|e| LLMError::GpuError(format!("seq_start_pos HtoD: {e}")))?;
            }

            let meta_pos = self.meta_positions.borrow();
            let meta_cl = self.meta_context_lens.borrow();
            let meta_bt = self.meta_block_tables.borrow();
            let meta_sm = self.meta_slot_mapping.borrow();
            let meta_ssp = self.meta_seq_start_pos.borrow();

            // Step 1: token embedding lookup
            info!("gpu_runner: embedding lookup");
            let mut hidden_states = self.embedding_lookup(token_ids)?;

            // Step 2: transformer layers
            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if layer_idx == 0 || layer_idx == num_layers - 1 {
                    info!(layer = layer_idx, use_fp16 = self.use_fp16, "gpu_runner: layer start");
                }
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let input = GpuLayerInput {
                    hidden_states: &hidden_states,
                    positions: meta_pos.slice(),
                    key_cache,
                    value_cache,
                    block_tables: meta_bt.slice(),
                    context_lens: meta_cl.slice(),
                    slot_mapping: meta_sm.slice(),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: meta_ssp.slice(),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                };
                hidden_states = if self.use_fp16 {
                    let weights = self.layer_weights_f16(layer_idx)?;
                    layer.forward_f16(&input, &weights, &self.blas)?
                } else {
                    let weights = self.layer_weights(layer_idx)?;
                    layer.forward(&input, &weights, &self.blas)?
                };
                if layer_idx == 0 || layer_idx == num_layers - 1 {
                    info!(layer = layer_idx, "gpu_runner: layer done");
                }
            }

            // Step 3: final RMSNorm (all on stream 0, no sync needed)
            let normed = CudaRMSNorm::forward(
                &hidden_states,
                &self.final_norm_weight,
                self.rms_norm_eps,
                hidden_size,
                &self.loader,
                &self.stream,
            )?;

            // Step 4+5: fused LM-head + argmax for single-token greedy decode
            if num_tokens == 1 && greedy_only {
                let token_ids_gpu = if self.use_fp16 {
                    let lm_f16 = self
                        .weights
                        .get_f16("lm_head.weight")
                        .or_else(|| self.weights.get_f16("model.embed_tokens.weight"));
                    if let Some(lm_w) = lm_f16 {
                        self.gpu_fused_lm_head_argmax_f16(&normed, lm_w, vocab_size, hidden_size)?
                    } else {
                        self.gpu_fused_lm_head_argmax(&normed, &self.lm_head_weight, vocab_size, hidden_size)?
                    }
                } else {
                    self.gpu_fused_lm_head_argmax(&normed, &self.lm_head_weight, vocab_size, hidden_size)?
                };
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head token DtoH: {e}")))?;
                debug!("forward_ex complete (fused lm_head+argmax, 4 bytes DtoH)");
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Step 4: LM head  normed [num_tokens, hidden] @ lm_head^T [hidden, vocab]
            let logits_gpu = if self.use_fp16 {
                let lm_f16 = self
                    .weights
                    .get_f16("lm_head.weight")
                    .or_else(|| self.weights.get_f16("model.embed_tokens.weight"));
                if let Some(lm_w) = lm_f16 {
                    CudaLinearLayer::forward_once_f16(
                        &normed, lm_w, num_tokens, vocab_size, hidden_size, &self.blas, &self.loader,
                    )?
                } else {
                    CudaLinearLayer::forward_once(
                        &normed, &self.lm_head_weight, None,
                        num_tokens, vocab_size, hidden_size, &self.blas,
                    )?
                }
            } else {
                CudaLinearLayer::forward_once(
                    &normed, &self.lm_head_weight, None,
                    num_tokens, vocab_size, hidden_size, &self.blas,
                )?
            };

            // Step 5: greedy fast path -- argmax on GPU, copy only token IDs
            if greedy_only {
                let token_ids_gpu = self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?;
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("argmax token_ids DtoH: {e}")))?;
                debug!(
                    num_tokens,
                    "forward_ex complete (greedy, {} bytes DtoH)",
                    num_tokens * 4
                );
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Step 5 (fallback): full logits DtoH for temperature>0 sampling
            let logits_cpu = self
                .stream
                .clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;

            debug!(
                logits_len = logits_cpu.len(),
                expected = num_tokens * vocab_size,
                "forward_ex complete (full logits)"
            );
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
            // Use the fixed max_blocks stride so that block_tables.len()/num_seqs
            // is always the same value. This is critical for CUDA graph replay:
            // the attention kernel receives p_max_blocks as a scalar arg baked
            // into the graph, and the data layout must match that stride.
            let max_blocks = self.graph_max_blocks;

            let mut scratch = self.cpu_scratch.borrow_mut();

            // token_ids (i32)
            scratch.clear();
            scratch.extend(token_ids.iter().map(|&t| t as i32));
            self.meta_token_ids.borrow_mut().upload(&scratch, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("token_ids HtoD: {e}")))?;

            // positions (i32)
            scratch.clear();
            scratch.extend(positions.iter().map(|&p| p as i32));
            self.meta_positions.borrow_mut().upload(&scratch, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("positions HtoD: {e}")))?;

            // context_lens (i32)
            scratch.clear();
            scratch.extend(attn_meta.context_lens.iter().map(|&c| c as i32));
            self.meta_context_lens.borrow_mut().upload(&scratch, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("context_lens HtoD: {e}")))?;

            // block_tables flattened [num_seqs, graph_max_blocks] (i32)
            // Always padded to fixed stride for CUDA graph pointer stability.
            scratch.clear();
            scratch.resize(num_seqs * max_blocks, 0i32);
            for (s, row) in attn_meta.block_tables.iter().enumerate() {
                for (b, &blk) in row.iter().enumerate() {
                    scratch[s * max_blocks + b] = blk as i32;
                }
            }
            self.meta_block_tables.borrow_mut().upload(&scratch, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("block_tables HtoD: {e}")))?;

            // slot_mapping (i32)
            scratch.clear();
            scratch.extend(attn_meta.slot_mapping.iter().map(|&s| s as i32));
            self.meta_slot_mapping.borrow_mut().upload(&scratch, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("slot_mapping HtoD: {e}")))?;

            // seq_start_pos from query_lens [num_seqs + 1] (i32)
            scratch.clear();
            let mut pos = 0i32;
            for &ql in &attn_meta.query_lens {
                scratch.push(pos);
                pos += ql as i32;
            }
            scratch.push(num_tokens as i32);
            self.meta_seq_start_pos.borrow_mut().upload(&scratch, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("seq_start_pos HtoD: {e}")))?;

            Ok(())
        }

        pub fn upload_metadata_padded(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            _padded_batch: usize,
        ) -> Result<()> {
            self.upload_metadata(token_ids, positions, attn_meta)
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

            let meta_pos = self.meta_positions.borrow();
            let meta_cl = self.meta_context_lens.borrow();
            let meta_bt = self.meta_block_tables.borrow();
            let meta_sm = self.meta_slot_mapping.borrow();
            let meta_ssp = self.meta_seq_start_pos.borrow();

            // Step 1: token embedding lookup from persistent buffer
            let mut hidden_states = self.embedding_lookup_from_meta(num_tokens)?;

            // Step 2: transformer layers
            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if layer_idx == 0 || layer_idx == num_layers - 1 {
                    trace!(layer = layer_idx, use_fp16 = self.use_fp16, "gpu_runner graph body: layer");
                }
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let input = GpuLayerInput {
                    hidden_states: &hidden_states,
                    positions: meta_pos.slice(),
                    key_cache,
                    value_cache,
                    block_tables: meta_bt.slice(),
                    context_lens: meta_cl.slice(),
                    slot_mapping: meta_sm.slice(),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: meta_ssp.slice(),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                };
                hidden_states = if self.use_fp16 {
                    let weights = self.layer_weights_f16(layer_idx)?;
                    layer.forward_f16(&input, &weights, &self.blas)?
                } else {
                    let weights = self.layer_weights(layer_idx)?;
                    layer.forward(&input, &weights, &self.blas)?
                };
            }

            // Step 3: final RMSNorm
            let normed = CudaRMSNorm::forward(
                &hidden_states,
                &self.final_norm_weight,
                self.rms_norm_eps,
                hidden_size,
                &self.loader,
                &self.stream,
            )?;

            // Step 4+5: fused LM-head + argmax for single-token greedy decode
            if num_tokens == 1 && greedy_only {
                let token_ids_gpu = if self.use_fp16 {
                    let lm_f16 = self
                        .weights
                        .get_f16("lm_head.weight")
                        .or_else(|| self.weights.get_f16("model.embed_tokens.weight"));
                    if let Some(lm_w) = lm_f16 {
                        self.gpu_fused_lm_head_argmax_f16(&normed, lm_w, vocab_size, hidden_size)?
                    } else {
                        self.gpu_fused_lm_head_argmax(&normed, &self.lm_head_weight, vocab_size, hidden_size)?
                    }
                } else {
                    self.gpu_fused_lm_head_argmax(&normed, &self.lm_head_weight, vocab_size, hidden_size)?
                };
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head token DtoH: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Step 4: LM head
            let logits_gpu = if self.use_fp16 {
                let lm_f16 = self
                    .weights
                    .get_f16("lm_head.weight")
                    .or_else(|| self.weights.get_f16("model.embed_tokens.weight"));
                if let Some(lm_w) = lm_f16 {
                    CudaLinearLayer::forward_once_f16(
                        &normed, lm_w, num_tokens, vocab_size, hidden_size, &self.blas, &self.loader,
                    )?
                } else {
                    CudaLinearLayer::forward_once(
                        &normed, &self.lm_head_weight, None,
                        num_tokens, vocab_size, hidden_size, &self.blas,
                    )?
                }
            } else {
                CudaLinearLayer::forward_once(
                    &normed, &self.lm_head_weight, None,
                    num_tokens, vocab_size, hidden_size, &self.blas,
                )?
            };

            // Step 5: greedy fast path
            if greedy_only {
                let token_ids_gpu = self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?;
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("argmax DtoH: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Full logits DtoH
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

            let meta_pos = self.meta_positions.borrow();
            let meta_cl = self.meta_context_lens.borrow();
            let meta_bt = self.meta_block_tables.borrow();
            let meta_sm = self.meta_slot_mapping.borrow();
            let meta_ssp = self.meta_seq_start_pos.borrow();

            // Step 1: token embedding lookup from persistent buffer
            let mut hidden_states = self.embedding_lookup_from_meta(num_tokens)?;

            // Step 2: transformer layers
            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let input = GpuLayerInput {
                    hidden_states: &hidden_states,
                    positions: meta_pos.slice(),
                    key_cache,
                    value_cache,
                    block_tables: meta_bt.slice(),
                    context_lens: meta_cl.slice(),
                    slot_mapping: meta_sm.slice(),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: meta_ssp.slice(),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                };
                hidden_states = if self.use_fp16 {
                    let weights = self.layer_weights_f16(layer_idx)?;
                    layer.forward_f16(&input, &weights, &self.blas)?
                } else {
                    let weights = self.layer_weights(layer_idx)?;
                    layer.forward(&input, &weights, &self.blas)?
                };
            }

            // Step 3: final RMSNorm
            let normed = CudaRMSNorm::forward(
                &hidden_states,
                &self.final_norm_weight,
                self.rms_norm_eps,
                hidden_size,
                &self.loader,
                &self.stream,
            )?;

            // Step 4+5: fused LM-head + argmax into persistent output buffer
            let token_ids_gpu = if num_tokens == 1 {
                if self.use_fp16 {
                    let lm_f16 = self
                        .weights
                        .get_f16("lm_head.weight")
                        .or_else(|| self.weights.get_f16("model.embed_tokens.weight"));
                    if let Some(lm_w) = lm_f16 {
                        self.gpu_fused_lm_head_argmax_f16(&normed, lm_w, vocab_size, hidden_size)?
                    } else {
                        self.gpu_fused_lm_head_argmax(&normed, &self.lm_head_weight, vocab_size, hidden_size)?
                    }
                } else {
                    self.gpu_fused_lm_head_argmax(&normed, &self.lm_head_weight, vocab_size, hidden_size)?
                }
            } else {
                // Multi-token: full LM head then argmax
                let logits_gpu = if self.use_fp16 {
                    let lm_f16 = self
                        .weights
                        .get_f16("lm_head.weight")
                        .or_else(|| self.weights.get_f16("model.embed_tokens.weight"));
                    if let Some(lm_w) = lm_f16 {
                        CudaLinearLayer::forward_once_f16(
                            &normed, lm_w, num_tokens, vocab_size, hidden_size,
                            &self.blas, &self.loader,
                        )?
                    } else {
                        CudaLinearLayer::forward_once(
                            &normed, &self.lm_head_weight, None,
                            num_tokens, vocab_size, hidden_size, &self.blas,
                        )?
                    }
                } else {
                    CudaLinearLayer::forward_once(
                        &normed, &self.lm_head_weight, None,
                        num_tokens, vocab_size, hidden_size, &self.blas,
                    )?
                };
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

        pub fn read_graph_output_async(
            &self,
            num_tokens: usize,
            dst: &mut [i32],
        ) -> Result<()> {
            let ids = self.read_graph_output(num_tokens)?;
            dst[..num_tokens].copy_from_slice(&ids[..num_tokens]);
            Ok(())
        }

        pub fn sync_stream(&self) -> Result<()> {
            self.stream
                .synchronize()
                .map_err(|e| LLMError::GpuError(format!("stream sync: {e}")))?;
            Ok(())
        }

        pub fn fuse_weights(&mut self) -> Result<()> {
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
            let g = |name: &str| -> Result<&CudaSlice<f32>> {
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
                q_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.q_proj.bias")),
                k_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.k_proj.bias")),
                v_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.v_proj.bias")),
                post_attention_layernorm: g(&format!(
                    "model.layers.{i}.post_attention_layernorm.weight"
                ))?,
                gate_proj: self.weights.get(&format!("model.layers.{i}.mlp.gate_proj.weight")),
                up_proj: self.weights.get(&format!("model.layers.{i}.mlp.up_proj.weight")),
                down_proj: self.weights.get(&format!("model.layers.{i}.mlp.down_proj.weight")),
                gpt_oss_moe: self.gpt_oss_moe_layers[i].as_ref(),
            })
        }

        fn embedding_lookup(&self, token_ids: &[u32]) -> Result<CudaSlice<f32>> {
            // Upload token IDs into persistent buffer (stable pointer for graph replay).
            {
                let mut scratch = self.cpu_scratch.borrow_mut();
                scratch.clear();
                scratch.extend(token_ids.iter().map(|&t| t as i32));
                self.meta_token_ids.borrow_mut().upload(&scratch, &self.stream)
                    .map_err(|e| LLMError::GpuError(format!("token_ids HtoD: {e}")))?;
            }
            self.embedding_lookup_from_meta(token_ids.len())
        }

        /// Embedding lookup using the pre-uploaded token IDs in meta_token_ids.
        /// Call upload_metadata() first to populate the buffer.
        fn embedding_lookup_from_meta(&self, num_tokens: usize) -> Result<CudaSlice<f32>> {
            let hidden_size = self.config.hidden_size;

            let kernel = self
                .loader
                .get_func("embedding_gather", "embedding_gather_kernel")?;

            let output = self
                .stream
                .alloc_zeros::<f32>(num_tokens * hidden_size)
                .map_err(|e| LLMError::GpuError(format!("embed alloc: {e}")))?;

            let meta_tid = self.meta_token_ids.borrow();

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
                    .arg(meta_tid.slice())
                    .arg(&(hidden_size as i32))
                    .arg(&(self.config.vocab_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("embedding_gather launch: {e}")))?;
            }

            Ok(output)
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

        /// Enable f16 inference mode.
        ///
        /// After calling this, `forward_ex` will use hgemm with f16 projection
        /// weights for all linear layers. The `GpuModelWeights` must already
        /// have f16 weights populated via `insert_f16` / the f16 loader.
        pub fn enable_fp16(&mut self) {
            self.use_fp16 = true;
            info!(use_fp16 = true, "GpuModelRunner: fp16 mode enabled");
        }

        pub fn use_fp16(&self) -> bool {
            self.use_fp16
        }

        pub fn supports_cuda_graphs(&self) -> bool {
            if self.config.architecture != "GptOssForCausalLM" {
                return true;
            }
            self.use_fp16
                && self
                    .gpt_oss_moe_layers
                    .iter()
                    .flatten()
                    .all(|layer| layer.supports_gpu_decode())
        }

        pub fn prepare_gpt_oss_graph_decode(&mut self) -> Result<()> {
            if self.config.architecture != "GptOssForCausalLM" || !self.use_fp16 {
                return Ok(());
            }

            self.prune_fp32_projection_weights_for_fp16();

            for i in 0..self.gpt_oss_moe_layers.len() {
                let Some(layer) = self.gpt_oss_moe_layers[i].as_mut() else {
                    continue;
                };
                if layer.supports_gpu_decode() {
                    continue;
                }

                let prefix = format!("model.layers.{i}.mlp.experts");
                let gate_up_blocks_name = format!("{prefix}.gate_up_proj_blocks");
                let gate_up_scales_name = format!("{prefix}.gate_up_proj_scales");
                let down_blocks_name = format!("{prefix}.down_proj_blocks");
                let down_scales_name = format!("{prefix}.down_proj_scales");

                if layer.gate_up_blocks_gpu.is_none() {
                    let data = self.weights.get_u8(&gate_up_blocks_name).ok_or_else(|| {
                        LLMError::GpuError(format!("missing GPT-OSS weight: {gate_up_blocks_name}"))
                    })?;
                    layer.gate_up_blocks_gpu = Some(
                        self.stream.clone_htod(data).map_err(|e| {
                            LLMError::GpuError(format!("HtoD {gate_up_blocks_name}: {e}"))
                        })?,
                    );
                }
                if layer.gate_up_scales_gpu.is_none() {
                    let data = self.weights.get_u8(&gate_up_scales_name).ok_or_else(|| {
                        LLMError::GpuError(format!("missing GPT-OSS weight: {gate_up_scales_name}"))
                    })?;
                    layer.gate_up_scales_gpu = Some(
                        self.stream.clone_htod(data).map_err(|e| {
                            LLMError::GpuError(format!("HtoD {gate_up_scales_name}: {e}"))
                        })?,
                    );
                }
                if layer.down_blocks_gpu.is_none() {
                    let data = self.weights.get_u8(&down_blocks_name).ok_or_else(|| {
                        LLMError::GpuError(format!("missing GPT-OSS weight: {down_blocks_name}"))
                    })?;
                    layer.down_blocks_gpu = Some(
                        self.stream.clone_htod(data).map_err(|e| {
                            LLMError::GpuError(format!("HtoD {down_blocks_name}: {e}"))
                        })?,
                    );
                }
                if layer.down_scales_gpu.is_none() {
                    let data = self.weights.get_u8(&down_scales_name).ok_or_else(|| {
                        LLMError::GpuError(format!("missing GPT-OSS weight: {down_scales_name}"))
                    })?;
                    layer.down_scales_gpu = Some(
                        self.stream.clone_htod(data).map_err(|e| {
                            LLMError::GpuError(format!("HtoD {down_scales_name}: {e}"))
                        })?,
                    );
                }
            }

            Ok(())
        }

        fn prune_fp32_projection_weights_for_fp16(&mut self) {
            if self.config.architecture != "GptOssForCausalLM" || !self.use_fp16 {
                return;
            }

            let mut removed = 0usize;
            for i in 0..self.config.num_layers {
                for suffix in [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                    "mlp.router.weight",
                    "mlp.router.bias",
                ] {
                    let name = format!("model.layers.{i}.{suffix}");
                    if self.weights.remove(&name).is_some() {
                        removed += 1;
                    }
                }
                for suffix in [
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ] {
                    let name = format!("model.layers.{i}.{suffix}");
                    if self.weights.remove_f16(&name).is_some() {
                        removed += 1;
                    }
                }
            }

            for shared in ["model.embed_tokens.weight", "lm_head.weight"] {
                if self.weights.remove(shared).is_some() {
                    removed += 1;
                }
                if self.weights.remove_f16(shared).is_some() {
                    removed += 1;
                }
            }

            if removed > 0 {
                info!(removed, "pruned unused GPT-OSS fp32 projection weights after enabling fp16");
            }
        }

        /// Per-layer f16 weight references into the GPU weight map.
        fn layer_weights_f16(&self, i: usize) -> Result<GpuLayerWeightsF16<'_>> {
            let g_f16 = |name: &str| -> Result<&CudaSlice<f16>> {
                self.weights
                    .get_f16(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing f16 weight: {name}")))
            };
            let g_f32 = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            Ok(GpuLayerWeightsF16 {
                input_layernorm: g_f32(&format!("model.layers.{i}.input_layernorm.weight"))?,
                q_proj: g_f16(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj: g_f16(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj: g_f16(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj: g_f16(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                q_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.q_proj.bias")),
                k_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.k_proj.bias")),
                v_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.v_proj.bias")),
                post_attention_layernorm: g_f32(&format!(
                    "model.layers.{i}.post_attention_layernorm.weight"
                ))?,
                gate_proj: self.weights.get_f16(&format!("model.layers.{i}.mlp.gate_proj.weight")),
                up_proj: self.weights.get_f16(&format!("model.layers.{i}.mlp.up_proj.weight")),
                down_proj: self.weights.get_f16(&format!("model.layers.{i}.mlp.down_proj.weight")),
                gpt_oss_moe: self.gpt_oss_moe_layers[i].as_ref(),
            })
        }

        fn build_gpt_oss_moe_layers(
            weights: &GpuModelWeights,
            config: &ModelRunnerConfig,
            stream: &Arc<CudaStream>,
        ) -> Result<Vec<Option<GptOssMoeLayerWeights>>> {
            if config.architecture != "GptOssForCausalLM" {
                return Ok((0..config.num_layers).map(|_| None).collect());
            }

            let mut layers = Vec::with_capacity(config.num_layers);
            for i in 0..config.num_layers {
                let prefix = format!("model.layers.{i}.mlp.experts");
                let gate_up_blocks_name = format!("{prefix}.gate_up_proj_blocks");
                let down_blocks_name = format!("{prefix}.down_proj_blocks");

                let Some(gate_up_blocks) = weights.get_u8(&gate_up_blocks_name) else {
                    layers.push(None);
                    continue;
                };
                let down_blocks = weights.get_u8(&down_blocks_name).ok_or_else(|| {
                    LLMError::GpuError(format!("missing GPT-OSS weight: {down_blocks_name}"))
                })?;
                let gate_up_scales_name = format!("{prefix}.gate_up_proj_scales");
                let down_scales_name = format!("{prefix}.down_proj_scales");
                let gate_up_scales = weights.get_u8(&gate_up_scales_name).ok_or_else(|| {
                    LLMError::GpuError(format!("missing GPT-OSS weight: {gate_up_scales_name}"))
                })?;
                let down_scales = weights.get_u8(&down_scales_name).ok_or_else(|| {
                    LLMError::GpuError(format!("missing GPT-OSS weight: {down_scales_name}"))
                })?;

                let dtoh = |name: &str| -> Result<Vec<f32>> {
                    let weight = weights
                        .get(name)
                        .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))?;
                    stream
                        .clone_dtoh(weight)
                        .map_err(|e| LLMError::GpuError(format!("DtoH {name}: {e}")))
                };
                let router_weight_name = format!("model.layers.{i}.mlp.router.weight");
                let router_bias_name = format!("model.layers.{i}.mlp.router.bias");
                let gate_up_bias_host = dtoh(&format!("{prefix}.gate_up_proj_bias"))?;
                let down_bias_host = dtoh(&format!("{prefix}.down_proj_bias"))?;
                let gate_up_bias_gpu = gate_up_bias_host
                    .chunks(config.intermediate_size * 2)
                    .map(|chunk| {
                        stream
                            .clone_htod(chunk)
                            .map_err(|e| LLMError::GpuError(format!("HtoD gate_up bias layer {i}: {e}")))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let down_bias_gpu = down_bias_host
                    .chunks(config.hidden_size)
                    .map(|chunk| {
                        stream
                            .clone_htod(chunk)
                            .map_err(|e| LLMError::GpuError(format!("HtoD down bias layer {i}: {e}")))
                    })
                    .collect::<Result<Vec<_>>>()?;

                layers.push(Some(GptOssMoeLayerWeights {
                    router_weight: dtoh(&router_weight_name)?,
                    router_bias: dtoh(&router_bias_name)?,
                    gate_up_blocks: gate_up_blocks.to_vec(),
                    gate_up_scales: gate_up_scales.to_vec(),
                    gate_up_bias: gate_up_bias_host,
                    down_blocks: down_blocks.to_vec(),
                    down_scales: down_scales.to_vec(),
                    down_bias: down_bias_host,
                    hidden_size: config.hidden_size,
                    intermediate_size: config.intermediate_size,
                    num_local_experts: config.num_local_experts,
                    num_experts_per_tok: config.num_experts_per_tok,
                    router_weight_gpu: Some(
                        weights
                            .get(&router_weight_name)
                            .ok_or_else(|| {
                                LLMError::GpuError(format!(
                                    "missing GPT-OSS weight: {router_weight_name}"
                                ))
                            })?
                            .clone(),
                    ),
                    router_bias_gpu: Some(
                        weights
                            .get(&router_bias_name)
                            .ok_or_else(|| {
                                LLMError::GpuError(format!(
                                    "missing GPT-OSS weight: {router_bias_name}"
                                ))
                            })?
                            .clone(),
                    ),
                    gate_up_blocks_gpu: None,
                    gate_up_scales_gpu: None,
                    gate_up_bias_gpu: Some(gate_up_bias_gpu),
                    down_blocks_gpu: None,
                    down_scales_gpu: None,
                    down_bias_gpu: Some(down_bias_gpu),
                }));
            }

            Ok(layers)
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

        pub fn enable_fp16(&mut self) {}

        pub fn use_fp16(&self) -> bool {
            false
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
    use rvllm_core::types::Dtype;

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
                partial_rotary_factor: 1.0,
                attn_logit_softcapping: 0.0,
                attention_bias: false,
                sliding_window: None,
                layer_types: Vec::new(),
                num_local_experts: 0,
                num_experts_per_tok: 0,
                dtype: Dtype::Float32,
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
                partial_rotary_factor: 1.0,
                attn_logit_softcapping: 0.0,
                attention_bias: false,
                sliding_window: None,
                layer_types: Vec::new(),
                num_local_experts: 0,
                num_experts_per_tok: 0,
                dtype: Dtype::Float16,
                architecture: "LlamaForCausalLM".to_string(),
            };
            let runner = GpuModelRunner { config };
            assert_eq!(runner.config().num_layers, 4);
            assert_eq!(runner.config().vocab_size, 32000);
        }
    }
}
