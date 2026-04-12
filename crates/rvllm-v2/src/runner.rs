use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};
use half::f16;

use rvllm_core::prelude::{LLMError, Result};
use rvllm_gpu::cublas::CublasHandle;
use rvllm_gpu::cutlass_ffi::CutlassKernels;
use rvllm_gpu::kernel_loader::KernelLoader;

use crate::kv_cache::CudaKVCache;
use crate::layer::{AttentionMeta, F16LayerScratch, GemmStrategy, GpuTransformerLayer, LayerWeights};
use crate::types::GpuBatchInput;

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub block_size: usize,
    pub max_seq_len: usize,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
}

struct ModelWeightsStore {
    #[allow(dead_code)]
    fused_qkv: Vec<CudaSlice<f16>>,
    #[allow(dead_code)]
    fused_gate_up: Vec<CudaSlice<f16>>,
    #[allow(dead_code)]
    o_proj: Vec<CudaSlice<f16>>,
    #[allow(dead_code)]
    down_proj: Vec<CudaSlice<f16>>,
    #[allow(dead_code)]
    input_layernorm: Vec<CudaSlice<f16>>,
    #[allow(dead_code)]
    post_attn_layernorm: Vec<CudaSlice<f16>>,
}

/// Offsets into the packed metadata GPU buffer (i32 elements).
struct PackedMetaOffsets {
    token_ids: usize,
    num_token_ids: usize,
    positions: usize,
    num_positions: usize,
    context_lens: usize,
    num_context_lens: usize,
    block_tables: usize,
    num_block_tables: usize,
    slot_mapping: usize,
    num_slot_mapping: usize,
    seq_start_pos: usize,
    num_seq_start_pos: usize,
}

pub struct GpuModelRunner {
    config: RunnerConfig,
    layers: Vec<GpuTransformerLayer>,
    gemm_strategy: GemmStrategy,
    cutlass: Option<Arc<CutlassKernels>>,
    cublas: CublasHandle,
    stream: Arc<CudaStream>,
    loader: Arc<KernelLoader>,
    weights: ModelWeightsStore,
    embed_tokens: CudaSlice<f16>,
    lm_head_weight: CudaSlice<f16>,
    final_norm_weight: CudaSlice<f16>,
    rope_cos: CudaSlice<f32>,
    rope_sin: CudaSlice<f32>,
    // Reusable CPU scratch for packing metadata
    cpu_scratch: Vec<i32>,
    // Reusable GPU packed metadata buffer
    meta_packed: CudaSlice<i32>,
    // Max blocks per seq for block table padding
    graph_max_blocks: usize,
    // Pre-allocated reusable GPU scratch (sized for max_num_seqs tokens)
    scratch: F16LayerScratch,
    residual_a: CudaSlice<f16>,
    residual_b: CudaSlice<f16>,
    down_a: CudaSlice<f16>,
    down_b: CudaSlice<f16>,
    final_normed: CudaSlice<f16>,
    residual_tmp: CudaSlice<f16>,
    logits_gpu: CudaSlice<f32>,
    embed_output: CudaSlice<f16>,
    argmax_output: CudaSlice<i32>,
}

impl GpuModelRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: RunnerConfig,
        layers: Vec<GpuTransformerLayer>,
        cutlass: Option<Arc<CutlassKernels>>,
        cublas: CublasHandle,
        stream: Arc<CudaStream>,
        loader: Arc<KernelLoader>,
        embed_tokens: CudaSlice<f16>,
        lm_head_weight: CudaSlice<f16>,
        final_norm_weight: CudaSlice<f16>,
        fused_qkv_weights: Vec<CudaSlice<f16>>,
        fused_gate_up_weights: Vec<CudaSlice<f16>>,
        o_proj_weights: Vec<CudaSlice<f16>>,
        down_proj_weights: Vec<CudaSlice<f16>>,
        input_layernorm_weights: Vec<CudaSlice<f16>>,
        post_attn_layernorm_weights: Vec<CudaSlice<f16>>,
        rope_cos: CudaSlice<f32>,
        rope_sin: CudaSlice<f32>,
    ) -> Result<Self> {
        let gemm_strategy = if cutlass.is_some() {
            GemmStrategy::Hybrid
        } else {
            GemmStrategy::Cublas
        };

        let graph_max_blocks = config.max_seq_len / config.block_size + 1;

        // Pre-allocate packed metadata buffer for max capacity
        // Worst case: 256 seqs * (max_seq_len tokens + metadata)
        let max_meta_elems = 256 * (config.max_seq_len + graph_max_blocks + 10);
        let meta_packed = stream
            .alloc_zeros::<i32>(max_meta_elems.max(4096))
            .map_err(|e| LLMError::GpuError(format!("meta_packed alloc: {e}")))?;

        // Pre-allocate reusable GPU scratch for max tokens per step
        let max_t = config.max_num_batched_tokens.max(config.max_num_seqs);
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let layer_config = layers[0].config_ref();
        let scratch = F16LayerScratch::alloc(&stream, layer_config, max_t)?;
        let max_n = max_t * hidden_size;
        let alloc_f16 = |label: &str, count: usize| -> Result<CudaSlice<f16>> {
            stream.alloc_zeros::<f16>(count)
                .map_err(|e| LLMError::GpuError(format!("{label} alloc: {e}")))
        };
        let residual_a = alloc_f16("residual_a", max_n)?;
        let residual_b = alloc_f16("residual_b", max_n)?;
        let down_a = alloc_f16("down_a", max_n)?;
        let down_b = alloc_f16("down_b", max_n)?;
        let final_normed = alloc_f16("final_normed", max_n)?;
        let residual_tmp = alloc_f16("residual_tmp", max_n)?;
        let logits_gpu = stream.alloc_zeros::<f32>(max_t * vocab_size)
            .map_err(|e| LLMError::GpuError(format!("logits alloc: {e}")))?;
        let embed_output = alloc_f16("embed_output", max_n)?;
        let argmax_output = stream.alloc_zeros::<i32>(max_t)
            .map_err(|e| LLMError::GpuError(format!("argmax_output alloc: {e}")))?;

        Ok(Self {
            config,
            layers,
            gemm_strategy,
            cutlass,
            cublas,
            stream,
            loader,
            weights: ModelWeightsStore {
                fused_qkv: fused_qkv_weights,
                fused_gate_up: fused_gate_up_weights,
                o_proj: o_proj_weights,
                down_proj: down_proj_weights,
                input_layernorm: input_layernorm_weights,
                post_attn_layernorm: post_attn_layernorm_weights,
            },
            embed_tokens,
            lm_head_weight,
            final_norm_weight,
            rope_cos,
            rope_sin,
            cpu_scratch: Vec::with_capacity(65536),
            meta_packed,
            graph_max_blocks,
            scratch,
            residual_a,
            residual_b,
            down_a,
            down_b,
            final_normed,
            residual_tmp,
            logits_gpu,
            embed_output,
            argmax_output,
        })
    }

    pub fn gemm_strategy(&self) -> GemmStrategy {
        self.gemm_strategy
    }

    pub fn forward(&mut self, input: &GpuBatchInput, kv_cache: &CudaKVCache) -> Result<Vec<f32>> {
        let num_tokens = total_tokens(input);
        if num_tokens == 0 {
            return Ok(Vec::new());
        }

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let num_seqs = input.num_seqs;
        let num_layers = self.config.num_layers;
        let gemm_strategy = self.gemm_strategy;

        // 1. Pack and upload metadata (single HtoD memcpy)
        let offsets = self.upload_metadata(input)?;

        // 2. Embedding lookup (into pre-allocated buffer)
        self.embedding_lookup(num_tokens, &offsets)?;

        // Destructure self to allow borrowing disjoint fields in the layer loop
        let Self {
            ref layers,
            ref cutlass,
            ref cublas,
            ref stream,
            ref meta_packed,
            ref rope_cos,
            ref rope_sin,
            ref weights,
            ref final_norm_weight,
            ref lm_head_weight,
            ref mut scratch,
            ref mut residual_a,
            ref mut residual_b,
            ref mut down_a,
            ref mut down_b,
            ref mut final_normed,
            ref mut residual_tmp,
            ref mut logits_gpu,
            ref embed_output,
            ..
        } = *self;

        let cutlass_ref: Option<&CutlassKernels> = cutlass.as_deref();

        // 4. Layer loop -- layers write directly to double-buffer targets (zero copies)
        for layer_idx in 0..num_layers {
            let (key_cache, value_cache) = &kv_cache.gpu_cache[layer_idx];

            let attn = AttentionMeta {
                positions: meta_packed
                    .slice(offsets.positions..offsets.positions + offsets.num_positions),
                key_cache,
                value_cache,
                block_tables: meta_packed
                    .slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                context_lens: meta_packed
                    .slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                slot_mapping: meta_packed
                    .slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                seq_start_pos: meta_packed
                    .slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                num_tokens,
                num_seqs,
                max_context_len: input.max_context_len,
                is_prefill: !input.is_all_decode,
                rope_cos,
                rope_sin,
            };

            let layer_weights = LayerWeights {
                qkv_weight: &weights.fused_qkv[layer_idx],
                o_proj_weight: &weights.o_proj[layer_idx],
                gate_up_weight: &weights.fused_gate_up[layer_idx],
                down_proj_weight: &weights.down_proj[layer_idx],
                input_layernorm_weight: &weights.input_layernorm[layer_idx],
                post_attention_layernorm_weight: &weights.post_attn_layernorm[layer_idx],
            };

            if layer_idx == 0 {
                // First layer: read embedding, write to a
                layers[layer_idx].forward_batched_v2(
                    embed_output, &attn, &layer_weights, scratch, None,
                    residual_a, down_a,
                    gemm_strategy, cutlass_ref, cublas,
                )?;
            } else if layer_idx % 2 == 1 {
                // Odd layers: read from a, write to b
                layers[layer_idx].forward_batched_v2(
                    &*residual_a, &attn, &layer_weights, scratch, Some(&*down_a),
                    residual_b, down_b,
                    gemm_strategy, cutlass_ref, cublas,
                )?;
            } else {
                // Even layers > 0: read from b, write to a
                layers[layer_idx].forward_batched_v2(
                    &*residual_b, &attn, &layer_weights, scratch, Some(&*down_b),
                    residual_a, down_a,
                    gemm_strategy, cutlass_ref, cublas,
                )?;
            }
        }

        // Final output: last odd layer wrote to b, last even (>0) layer wrote to a
        let last_wrote_to_b = num_layers > 1 && (num_layers - 1) % 2 == 1;
        let (final_residual, final_down): (&CudaSlice<f16>, &CudaSlice<f16>) =
            if last_wrote_to_b {
                (residual_b, down_b)
            } else {
                (residual_a, down_a)
            };

        // 5. Final RMSNorm
        layers[0].fused_residual_rmsnorm_pub(
            final_residual, final_down, final_norm_weight,
            num_tokens, hidden_size, final_normed, residual_tmp,
        )?;

        // 6. LM head: f16 hidden x f16 lm_head -> f32 logits
        cublas.hgemm_f32_output(
            num_tokens, vocab_size, hidden_size,
            1.0, &*final_normed, lm_head_weight, 0.0, logits_gpu,
        )?;

        // 7. DtoH logits
        let logits_cpu = stream
            .clone_dtoh(&logits_gpu.slice(..num_tokens * vocab_size))
            .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;

        Ok(logits_cpu)
    }

    /// Fast greedy decode path: runs full forward, GPU argmax, returns only token IDs.
    /// Avoids the massive DtoH of full logits (N * vocab_size * 4 bytes).
    pub fn forward_greedy(&mut self, input: &GpuBatchInput, kv_cache: &CudaKVCache) -> Result<Vec<i32>> {
        let num_tokens = total_tokens(input);
        if num_tokens == 0 {
            return Ok(Vec::new());
        }

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let num_seqs = input.num_seqs;
        let num_layers = self.config.num_layers;
        let gemm_strategy = self.gemm_strategy;

        let offsets = self.upload_metadata(input)?;
        self.embedding_lookup(num_tokens, &offsets)?;

        let Self {
            ref layers,
            ref cutlass,
            ref cublas,
            ref stream,
            ref meta_packed,
            ref rope_cos,
            ref rope_sin,
            ref weights,
            ref final_norm_weight,
            ref lm_head_weight,
            ref mut scratch,
            ref mut residual_a,
            ref mut residual_b,
            ref mut down_a,
            ref mut down_b,
            ref mut final_normed,
            ref mut residual_tmp,
            ref mut logits_gpu,
            ref embed_output,
            ref loader,
            ref mut argmax_output,
            ..
        } = *self;

        let cutlass_ref: Option<&CutlassKernels> = cutlass.as_deref();

        for layer_idx in 0..num_layers {
            let (key_cache, value_cache) = &kv_cache.gpu_cache[layer_idx];
            let attn = AttentionMeta {
                positions: meta_packed
                    .slice(offsets.positions..offsets.positions + offsets.num_positions),
                key_cache,
                value_cache,
                block_tables: meta_packed
                    .slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                context_lens: meta_packed
                    .slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                slot_mapping: meta_packed
                    .slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                seq_start_pos: meta_packed
                    .slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                num_tokens,
                num_seqs,
                max_context_len: input.max_context_len,
                is_prefill: !input.is_all_decode,
                rope_cos,
                rope_sin,
            };
            let layer_weights = LayerWeights {
                qkv_weight: &weights.fused_qkv[layer_idx],
                o_proj_weight: &weights.o_proj[layer_idx],
                gate_up_weight: &weights.fused_gate_up[layer_idx],
                down_proj_weight: &weights.down_proj[layer_idx],
                input_layernorm_weight: &weights.input_layernorm[layer_idx],
                post_attention_layernorm_weight: &weights.post_attn_layernorm[layer_idx],
            };
            if layer_idx == 0 {
                layers[layer_idx].forward_batched_v2(
                    embed_output, &attn, &layer_weights, scratch, None,
                    residual_a, down_a,
                    gemm_strategy, cutlass_ref, cublas,
                )?;
            } else if layer_idx % 2 == 1 {
                layers[layer_idx].forward_batched_v2(
                    &*residual_a, &attn, &layer_weights, scratch, Some(&*down_a),
                    residual_b, down_b,
                    gemm_strategy, cutlass_ref, cublas,
                )?;
            } else {
                layers[layer_idx].forward_batched_v2(
                    &*residual_b, &attn, &layer_weights, scratch, Some(&*down_b),
                    residual_a, down_a,
                    gemm_strategy, cutlass_ref, cublas,
                )?;
            }
        }

        let last_wrote_to_b = num_layers > 1 && (num_layers - 1) % 2 == 1;
        let (final_residual, final_down): (&CudaSlice<f16>, &CudaSlice<f16>) =
            if last_wrote_to_b {
                (residual_b, down_b)
            } else {
                (residual_a, down_a)
            };

        layers[0].fused_residual_rmsnorm_pub(
            final_residual, final_down, final_norm_weight,
            num_tokens, hidden_size, final_normed, residual_tmp,
        )?;

        // LM head GEMM
        cublas.hgemm_f32_output(
            num_tokens, vocab_size, hidden_size,
            1.0, &*final_normed, lm_head_weight, 0.0, logits_gpu,
        )?;

        // GPU-side argmax (no full logits DtoH)
        let argmax_kernel = loader.get_func("argmax", "argmax_kernel")?;
        let block_dim = vocab_size.min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&argmax_kernel)
                .arg(&*logits_gpu)
                .arg(&*argmax_output)
                .arg(&(vocab_size as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("argmax launch: {e}")))?;
        }

        // DtoH only the token IDs (num_seqs * 4 bytes instead of num_seqs * vocab * 4)
        let token_ids = stream
            .clone_dtoh(&argmax_output.slice(..num_tokens))
            .map_err(|e| LLMError::GpuError(format!("argmax DtoH: {e}")))?;

        Ok(token_ids)
    }

    /// Pack all metadata fields into one contiguous GPU buffer (1 memcpy).
    fn upload_metadata(&mut self, input: &GpuBatchInput) -> Result<PackedMetaOffsets> {
        let num_tokens = total_tokens(input);
        let num_seqs = input.num_seqs;
        let max_blocks = self.graph_max_blocks;

        self.cpu_scratch.clear();

        // token_ids
        let token_ids_off = self.cpu_scratch.len();
        if input.is_all_decode {
            for req_idx in 0..num_seqs {
                self.cpu_scratch.push(input.token_ids[req_idx] as i32);
            }
        } else {
            for &t in &input.prefill_tokens {
                self.cpu_scratch.push(t as i32);
            }
            for req_idx in input.num_prefill_seqs..num_seqs {
                self.cpu_scratch.push(input.token_ids[req_idx] as i32);
            }
        }
        let num_token_ids = self.cpu_scratch.len() - token_ids_off;

        // positions
        let positions_off = self.cpu_scratch.len();
        if input.is_all_decode {
            for &p in &input.position_ids {
                self.cpu_scratch.push(p as i32);
            }
        } else {
            for &p in &input.prefill_positions {
                self.cpu_scratch.push(p as i32);
            }
            for idx in input.num_prefill_seqs..num_seqs {
                self.cpu_scratch.push(input.position_ids[idx] as i32);
            }
        }
        let num_positions = self.cpu_scratch.len() - positions_off;

        // context_lens
        let context_lens_off = self.cpu_scratch.len();
        for &c in &input.context_lens {
            self.cpu_scratch.push(c as i32);
        }
        let num_context_lens = self.cpu_scratch.len() - context_lens_off;

        // block_tables: [num_seqs, max_blocks], zero-padded
        let block_tables_off = self.cpu_scratch.len();
        let bt_len = num_seqs * max_blocks;
        let old_len = self.cpu_scratch.len();
        self.cpu_scratch.resize(old_len + bt_len, 0i32);
        // Unpack the flat block_tables from GpuBatchInput
        let max_blocks_input = input.max_blocks_per_seq;
        for s in 0..num_seqs {
            let src_start = s * max_blocks_input;
            let copy_len = max_blocks_input.min(max_blocks);
            for b in 0..copy_len {
                if src_start + b < input.block_tables_flat.len() {
                    self.cpu_scratch[block_tables_off + s * max_blocks + b] =
                        input.block_tables_flat[src_start + b] as i32;
                }
            }
        }
        let num_block_tables = bt_len;

        // slot_mapping
        let slot_mapping_off = self.cpu_scratch.len();
        if input.is_all_decode {
            for &s in &input.slot_mapping {
                self.cpu_scratch.push(s as i32);
            }
        } else {
            for &s in &input.prefill_slot_mapping {
                self.cpu_scratch.push(s as i32);
            }
            for idx in input.num_prefill_seqs..num_seqs {
                self.cpu_scratch.push(input.slot_mapping[idx] as i32);
            }
        }
        let num_slot_mapping = self.cpu_scratch.len() - slot_mapping_off;

        // seq_start_pos: prefix sums of query_lens + total
        let seq_start_pos_off = self.cpu_scratch.len();
        let mut pos = 0i32;
        for &ql in &input.query_lens {
            self.cpu_scratch.push(pos);
            pos += ql as i32;
        }
        self.cpu_scratch.push(num_tokens as i32);
        let num_seq_start_pos = self.cpu_scratch.len() - seq_start_pos_off;

        // Single packed upload
        let total_elems = self.cpu_scratch.len();
        if total_elems > self.meta_packed.len() {
            // Reallocate if needed
            self.meta_packed = self.stream
                .alloc_zeros::<i32>(total_elems * 2)
                .map_err(|e| LLMError::GpuError(format!("meta_packed realloc: {e}")))?;
        }

        self.stream
            .memcpy_htod(&self.cpu_scratch, &mut self.meta_packed.slice_mut(..total_elems))
            .map_err(|e| LLMError::GpuError(format!("packed metadata HtoD: {e}")))?;

        Ok(PackedMetaOffsets {
            token_ids: token_ids_off,
            num_token_ids,
            positions: positions_off,
            num_positions,
            context_lens: context_lens_off,
            num_context_lens,
            block_tables: block_tables_off,
            num_block_tables,
            slot_mapping: slot_mapping_off,
            num_slot_mapping,
            seq_start_pos: seq_start_pos_off,
            num_seq_start_pos,
        })
    }

    /// Embedding gather: token_ids -> f16 hidden states via custom kernel.
    fn embedding_lookup(
        &mut self,
        num_tokens: usize,
        offsets: &PackedMetaOffsets,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;

        let kernel = self
            .loader
            .get_func("embedding_gather_f16", "embedding_gather_f16_kernel")?;

        let token_ids_view: CudaView<'_, i32> = self
            .meta_packed
            .slice(offsets.token_ids..offsets.token_ids + offsets.num_token_ids);

        let block_dim = hidden_size.min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&self.embed_output)
                .arg(&self.embed_tokens)
                .arg(&token_ids_view)
                .arg(&(hidden_size as i32))
                .arg(&(self.config.vocab_size as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("embedding_gather launch: {e}")))?;
        }

        Ok(())
    }
}

fn total_tokens(input: &GpuBatchInput) -> usize {
    if input.is_all_decode {
        input.num_seqs
    } else {
        input.prefill_tokens.len() + input.num_decode_seqs
    }
}
