//! rvllm-eval: single-sequence text generation for quality measurement.
//!
//! Loads a HF tokenizer, tokenizes a prompt, runs prefill + decode in
//! eager mode (no graph capture), reads back sampled token IDs via
//! pinned DtoH, detokenizes, and prints the output.
//!
//! Env vars (same as rvllm-bench plus):
//!   RVLLM_MODEL_DIR   = HF snapshot dir (must contain tokenizer.json)
//!   RVLLM_KERNELS_DIR, RVLLM_CUTLASS_SO, RVLLM_FA3_SO, RVLLM_POLICY
//!   RVLLM_MAX_TOKENS  = max output tokens (default 256)
//!   RVLLM_PROMPT      = prompt string (alternative to stdin)

use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use rvllm_core::{DType, ModelArch as HfModelArch, ModelConfig};
use rvllm_runtime::{Bringup, EnginePaths};
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};

fn env_path(k: &str) -> Result<PathBuf, String> {
    std::env::var(k)
        .map_err(|_| format!("missing env var: {k}"))
        .map(PathBuf::from)
}

fn is_gemma4_model_dir(model_dir: &std::path::Path) -> Result<bool, String> {
    Ok(matches!(
        ModelConfig::load_hf(model_dir)
            .map_err(|e| format!("config parse {}: {e}", model_dir.display()))?
            .architecture,
        HfModelArch::Gemma4
    ))
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    if let Err(e) = run() {
        eprintln!("rvllm-eval: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;

    // -- tokenizer --
    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| format!("tokenizer load {}: {e}", tok_path.display()))?;

    // -- prompt --
    let prompt = if let Ok(p) = std::env::var("RVLLM_PROMPT") {
        p
    } else {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("stdin: {e}"))?;
        buf
    };
    if prompt.is_empty() {
        return Err("empty prompt (set RVLLM_PROMPT or pipe to stdin)".into());
    }

    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| format!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_ids.len() as u32;
    eprintln!("prompt: {} tokens", prompt_len);

    let max_new: u32 = std::env::var("RVLLM_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    // -- bringup --
    let paths = EnginePaths {
        model_dir: model_dir.clone(),
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path("RVLLM_CUTLASS_SO")?,
        fa3_so: env_path("RVLLM_FA3_SO")?,
        policy_json: env_path("RVLLM_POLICY")?,
    };
    let arena_bytes: usize = if let Some(gb) = std::env::var("RVLLM_ARENA_GB").ok().and_then(|s| s.parse::<usize>().ok()) {
        gb * 1024 * 1024 * 1024
    } else {
        #[cfg(feature = "cuda")]
        {
            let free = {
                let mut free: usize = 0;
                let mut total: usize = 0;
                // Borrow a context just long enough to probe free memory.
                // `CudaContextHandle::init` uses primary-context retain
                // (works on CUDA 11/12/13 — see rvllm-mem/src/context.rs
                // module docs); the legacy `cuCtxCreate_v2` this code
                // used to call directly is unwrapped by cudarc only for
                // CUDA ≤ 12.09, so on CUDA 13 hosts the old path
                // failed to resolve.
                let _probe_ctx = rvllm_mem::context::CudaContextHandle::init(0)
                    .map_err(|e| format!("probe ctx: {e}"))?;
                unsafe {
                    cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _);
                }
                drop(_probe_ctx);
                free
            };
            let reserve = 512 * 1024 * 1024; // 512MB headroom
            let arena = if free > reserve { free - reserve } else { free };
            eprintln!("[eval] auto-sized arena: {:.1} GB ({:.1} GB free)", arena as f64 / 1e9, free as f64 / 1e9);
            arena
        }
        #[cfg(not(feature = "cuda"))]
        { 32 * 1024 * 1024 * 1024 }
    };
    let t0 = Instant::now();

    if is_gemma4_model_dir(&model_dir)? {
        let g4_paths = Gemma4EnginePaths {
            model_dir: paths.model_dir.clone(),
            kernels_dir: paths.kernels_dir,
            cutlass_so: paths.cutlass_so,
            fa3_so: paths.fa3_so,
            policy_json: paths.policy_json,
        };
        let g4 = Gemma4Bringup::load(g4_paths, arena_bytes)
            .map_err(|e| format!("gemma4 bringup: {e}"))?;
        eprintln!("bringup: {:.2}s", t0.elapsed().as_secs_f64());

        let embed_mod = g4.kernels
            .load_ptx("embedding_gather_f16")
            .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
        let fn_embed = embed_mod
            .get_function("embedding_gather_f16_kernel")
            .map_err(|e| format!("get embedding_gather_f16_kernel: {e}"))?;

        let argmax_mod = g4.kernels
            .load_ptx("argmax")
            .map_err(|e| format!("load argmax: {e}"))?;
        let fn_argmax = argmax_mod
            .get_function("argmax_kernel")
            .map_err(|e| format!("get argmax_kernel: {e}"))?;

        let prompt_with_bos: Vec<u32> = std::iter::once(2u32)
            .chain(prompt_ids.iter().copied())
            .collect();
        let eos_ids: Vec<u32> = vec![1, 2, 107];

        let t_gen = Instant::now();
        let output_ids = unsafe {
            g4.run_generate(fn_embed, fn_argmax, &prompt_with_bos, max_new as usize, &eos_ids)
        }.map_err(|e| format!("gemma4 generate: {e}"))?;

        let elapsed = t_gen.elapsed();
        let n = output_ids.len();
        eprintln!("generated {} tokens in {:.2}s ({:.1} tok/s)", n, elapsed.as_secs_f64(), n as f64 / elapsed.as_secs_f64());

        let text = tokenizer.decode(&output_ids, true)
            .map_err(|e| format!("detokenize: {e}"))?;
        println!("{text}");
        return Ok(());
    }

    let br = Bringup::load(paths, arena_bytes).map_err(|e| format!("bringup: {e}"))?;
    eprintln!("bringup: {:.2}s", t0.elapsed().as_secs_f64());

    // Load embedding_gather kernel (not loaded by default bring_up).
    let embed_mod = br
        .kernels
        .load_ptx("embedding_gather_f16")
        .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
    let fn_embed = embed_mod
        .get_function("embedding_gather_f16_kernel")
        .map_err(|e| format!("get embedding_gather_f16_kernel: {e}"))?;

    // -- run generation (all unsafe: raw device pointers) --
    let output_ids = unsafe { generate(&br, fn_embed, &prompt_ids, max_new) }
        .map_err(|e| format!("generate: {e}"))?;

    // -- detokenize --
    let text = tokenizer
        .decode(&output_ids, true)
        .map_err(|e| format!("detokenize: {e}"))?;
    println!("{text}");
    Ok(())
}

#[cfg(feature = "cuda")]
unsafe fn generate(
    br: &Bringup,
    fn_embed: rvllm_kernels::KernelFn,
    prompt_ids: &[u32],
    max_new: u32,
) -> Result<Vec<u32>, String> {
    use rvllm_cutlass::Fp8GemmPlan;
    use rvllm_fused::require_multiple;
    use rvllm_runtime::layer_exec;

    let arch = &br.arch;
    let hidden = arch.hidden_size as u32;
    let head_dim = arch.head_dim as u32;
    let nh = arch.num_attention_heads as u32;
    let nkvh = arch.num_key_value_heads as u32;
    let inter = arch.intermediate_size as u32;
    let q_dim = nh * head_dim;
    let kv_dim = nkvh * head_dim;
    let qkv_rows = (nh + 2 * nkvh) * head_dim;
    let vocab = arch.vocab_size as u32;
    let prompt_len = prompt_ids.len() as u32;
    require_multiple(hidden as usize, 8, "hidden").map_err(|e| e.to_string())?;

    let num_seqs: u32 = 1;
    let max_tokens = prompt_len.max(num_seqs); // prefill needs prompt_len slots
    let block_size: u32 = 64;
    let num_blocks_total: u32 = 1024;
    let max_blocks_per_seq: u32 = num_blocks_total; // single seq gets all pages
    let kv_per_layer = 2 * num_blocks_total * block_size * nkvh * head_dim;

    let arena = &br.arena;

    // Scratch regions sized for max(prompt_len, 1) tokens.
    let hidden_fp8 = arena.region("hidden_fp8", (max_tokens * hidden) as usize, 16)
        .map_err(|e| e.to_string())?;
    let hidden_scale = arena.region("hidden_scale", (max_tokens * 4) as usize, 16)
        .map_err(|e| e.to_string())?;
    let qkv_out_bytes = (max_tokens * qkv_rows * 2) as usize;
    let qkv_out = arena.region("qkv_out", qkv_out_bytes, 16).map_err(|e| e.to_string())?;
    let q_base = qkv_out.device_ptr();
    // For decode (num_tokens=1) and prefill (num_tokens=prompt_len) the
    // K/V offsets into the packed QKV buffer differ.
    let k_base_decode = q_base + (num_seqs as u64) * (q_dim as u64) * 2;
    let v_base_decode = k_base_decode + (num_seqs as u64) * (kv_dim as u64) * 2;
    let k_base_prefill = q_base + (prompt_len as u64) * (q_dim as u64) * 2;
    let v_base_prefill = k_base_prefill + (prompt_len as u64) * (kv_dim as u64) * 2;
    let attn_out = arena.region("attn_out", (max_tokens * q_dim * 2) as usize, 16)
        .map_err(|e| e.to_string())?;
    let attn_out_fp8 = arena.region("attn_out_fp8", (max_tokens * q_dim) as usize, 16)
        .map_err(|e| e.to_string())?;
    let attn_out_scale = arena.region("attn_out_scale", (max_tokens * 4) as usize, 16)
        .map_err(|e| e.to_string())?;
    let gate_up_out = arena.region("gate_up_out", (max_tokens * 2 * inter * 2) as usize, 16)
        .map_err(|e| e.to_string())?;
    let gate_up_fp8 = arena.region("gate_up_fp8", (max_tokens * 2 * inter) as usize, 16)
        .map_err(|e| e.to_string())?;
    let gate_up_scale = arena.region("gate_up_scale", (max_tokens * 4) as usize, 16)
        .map_err(|e| e.to_string())?;
    let mlp_out_fp8 = arena.region("mlp_out_fp8", (max_tokens * inter) as usize, 16)
        .map_err(|e| e.to_string())?;
    let mlp_out_scale = arena.region("mlp_out_scale", (max_tokens * 4) as usize, 16)
        .map_err(|e| e.to_string())?;

    let kv_cache = arena.region(
        "kv_cache",
        (arch.num_hidden_layers as u64 * kv_per_layer as u64) as usize,
        256,
    ).map_err(|e| e.to_string())?;
    let q_fp8 = arena.region("q_fp8", (max_tokens * q_dim) as usize, 16)
        .map_err(|e| e.to_string())?;
    let q_scale_region = arena.region("q_scale", 4, 4).map_err(|e| e.to_string())?;
    let kv_scale_region = arena.region("kv_scale", 4, 4).map_err(|e| e.to_string())?;
    {
        let seed: f32 = 1.0 / 448.0;
        q_scale_region.copy_from_host(&seed.to_le_bytes()).map_err(|e| e.to_string())?;
        kv_scale_region.copy_from_host(&seed.to_le_bytes()).map_err(|e| e.to_string())?;
    }

    let cutlass_ws = arena.region("cutlass_ws", 16 * 1024 * 1024, 256).map_err(|e| e.to_string())?;
    let fa3_ws = arena.region("fa3_ws", 64 * 1024 * 1024, 256).map_err(|e| e.to_string())?;
    let residual = arena.region("residual", (max_tokens * hidden * 2) as usize, 16)
        .map_err(|e| e.to_string())?;

    // Metadata regions.
    let positions = arena.region("positions", (max_tokens * 4) as usize, 16)
        .map_err(|e| e.to_string())?;
    let slot_mapping = arena.region("slot_mapping", (max_tokens * 4) as usize, 16)
        .map_err(|e| e.to_string())?;
    let context_lens = arena.region("context_lens", (num_seqs * 4) as usize, 16)
        .map_err(|e| e.to_string())?;
    let block_tables = arena.region(
        "block_tables",
        (num_seqs * max_blocks_per_seq * 4) as usize,
        16,
    ).map_err(|e| e.to_string())?;

    // Token ID upload region (for embedding gather input).
    let token_ids_region = arena.region("token_ids", (max_tokens * 4) as usize, 16)
        .map_err(|e| e.to_string())?;
    // Logits + sampled output.
    let logits = arena.region("logits", (num_seqs * vocab * 2) as usize, 16)
        .map_err(|e| e.to_string())?;
    let sampled_tokens = arena.region("sampled_tokens", (num_seqs * 4) as usize, 16)
        .map_err(|e| e.to_string())?;

    // Prefill cu_seqlens_q.
    let cu_seqlens_q = arena.region("cu_seqlens_q", ((num_seqs + 1) * 4) as usize, 16)
        .map_err(|e| e.to_string())?;

    // Block tables: seq 0 owns blocks 0..max_blocks_per_seq.
    {
        let bt: Vec<i32> = (0..max_blocks_per_seq as i32).collect();
        block_tables.copy_from_host(bytemuck_i32(&bt)).map_err(|e| e.to_string())?;
    }

    // GEMM plans for N=1 (decode) -- prefill plans use prompt_len as M.
    let plan_qkv_d = Fp8GemmPlan::from_policy(&br.policy, 1, qkv_rows, hidden, DType::Fp8E4M3)
        .map_err(|e| e.to_string())?;
    let plan_o_d = Fp8GemmPlan::from_policy_residual(&br.policy, 1, hidden, q_dim, DType::Fp8E4M3)
        .map_err(|e| e.to_string())?;
    let plan_gate_up_d = Fp8GemmPlan::from_policy(&br.policy, 1, 2 * inter, hidden, DType::Fp8E4M3)
        .map_err(|e| e.to_string())?;
    let plan_down_d = Fp8GemmPlan::from_policy_residual(&br.policy, 1, hidden, inter, DType::Fp8E4M3)
        .map_err(|e| e.to_string())?;

    // Prefill plans (M = prompt_len).
    let plan_qkv_p = Fp8GemmPlan::from_policy(&br.policy, prompt_len, qkv_rows, hidden, DType::Fp8E4M3)
        .map_err(|e| e.to_string())?;
    let plan_o_p = Fp8GemmPlan::from_policy_residual(&br.policy, prompt_len, hidden, q_dim, DType::Fp8E4M3)
        .map_err(|e| e.to_string())?;
    let plan_gate_up_p = Fp8GemmPlan::from_policy(&br.policy, prompt_len, 2 * inter, hidden, DType::Fp8E4M3)
        .map_err(|e| e.to_string())?;
    let plan_down_p = Fp8GemmPlan::from_policy_residual(&br.policy, prompt_len, hidden, inter, DType::Fp8E4M3)
        .map_err(|e| e.to_string())?;

    let stream = br.stream.raw();

    let dims_base = layer_exec::LayerDims {
        num_tokens: 1,
        hidden,
        num_heads: nh,
        num_kv_heads: nkvh,
        head_dim,
        intermediate: inter,
        block_size,
        max_blocks_per_seq,
        num_blocks_total,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        rms_eps: 1e-6,
    };
    let kernels = layer_exec::LayerKernels {
        fused_rmsnorm: br.fused_modules.fn_rmsnorm,
        fused_add_rmsnorm: br.fused_modules.fn_add_rmsnorm,
        fused_rope_cache_fp8kv: br.fused_modules.fn_rope_cache_fp8kv,
        fused_silu_mul: br.fused_modules.fn_silu_mul,
        fused_gelu_mul: br.fused_modules.fn_gelu_mul,
        mlp_activation: layer_exec::MlpActivation::from_config_str(
            br.arch.hidden_activation.as_deref(),
        ),
        quantize_fp8_per_token: br.fused_modules.fn_quantize,
        add_bias_f16: br.fused_modules.fn_add_bias_f16,
    };
    let residual_ptr = residual.device_ptr();

    // Helper: run all layers + lm_head + argmax for a given phase.
    // Wraps all unsafe kernel/GEMM calls in a single unsafe block.
    let run_forward = |phase: layer_exec::LayerPhase,
                       plans: &layer_exec::LayerGemmPlans,
                       num_tokens: u32,
                       k_base: u64,
                       v_base: u64|
        -> Result<(), String> { unsafe {
        let mut dims = dims_base;
        dims.num_tokens = num_tokens;
        for (layer_idx, layer) in br.model.layers.iter().enumerate() {
            let layer_kv_base =
                kv_cache.device_ptr() + (layer_idx as u64) * (kv_per_layer as u64);
            let w = layer_exec::LayerWeights {
                attn_norm_gamma: layer.input_layernorm.offset_bytes,
                qkv_fp8: layer.qkv.offset_bytes,
                qkv_scale: layer.qkv.scale_ptr,
                qkv_bias: layer.qkv_bias.as_ref().map_or(0, |b| b.offset_bytes),
                o_fp8: layer.o_proj.offset_bytes,
                o_scale: layer.o_proj.scale_ptr,
                mlp_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                gate_up_fp8: layer.gate_up.offset_bytes,
                gate_up_scale: layer.gate_up.scale_ptr,
                down_fp8: layer.down_proj.offset_bytes,
                down_scale: layer.down_proj.scale_ptr,
            };
            let scratch = layer_exec::LayerScratch {
                hidden_fp8: hidden_fp8.device_ptr(),
                hidden_scale: hidden_scale.device_ptr(),
                q_out: q_base,
                k_out: k_base,
                v_out: v_base,
                q_fp8: q_fp8.device_ptr(),
                k_cache: layer_kv_base,
                v_cache: layer_kv_base + (kv_per_layer / 2) as u64,
                q_scale_ptr: q_scale_region.device_ptr(),
                kv_scale_ptr: kv_scale_region.device_ptr(),
                attn_out: attn_out.device_ptr(),
                attn_out_fp8: attn_out_fp8.device_ptr(),
                attn_out_scale: attn_out_scale.device_ptr(),
                gate_up_out: gate_up_out.device_ptr(),
                gate_up_fp8: gate_up_fp8.device_ptr(),
                gate_up_scale: gate_up_scale.device_ptr(),
                mlp_out_fp8: mlp_out_fp8.device_ptr(),
                mlp_out_scale: mlp_out_scale.device_ptr(),
                cutlass_workspace: cutlass_ws.device_ptr(),
                cutlass_workspace_bytes: 16 * 1024 * 1024,
                fa3_workspace: fa3_ws.device_ptr(),
            };
            let meta = layer_exec::MetadataPtrs {
                positions: positions.device_ptr(),
                slot_mapping: slot_mapping.device_ptr(),
                cos: br.model.rope_cos.offset_bytes,
                sin: br.model.rope_sin.offset_bytes,
                block_tables: block_tables.device_ptr(),
                context_lens: context_lens.device_ptr(),
            };
            layer_exec::forward_phase(
                dims, &kernels, &w, &scratch, &meta, plans,
                &br.cutlass, &br.cublaslt, &br.fa3,
                residual_ptr, stream, phase,
                br.arch.layer_types[layer_idx],
            ).map_err(|e| e.to_string())?;
        }
        // Final norm + LM head + argmax (only for decode, prefill skips).
        if matches!(phase, layer_exec::LayerPhase::Decode) {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: 1,
                hidden,
                eps: 1e-6,
            }
            .launch(
                br.fused_modules.fn_rmsnorm,
                hidden_fp8.device_ptr(),
                hidden_scale.device_ptr(),
                residual_ptr,
                br.model.final_norm.offset_bytes,
                stream,
            ).map_err(|e| e.to_string())?;
            br.cublaslt.fp8_gemm(
                hidden_fp8.device_ptr(),
                br.model.lm_head_fp8.offset_bytes,
                logits.device_ptr(),
                1,
                vocab as i32,
                hidden as i32,
                hidden_scale.device_ptr(),
                br.model.lm_head_fp8.scale_ptr,
                stream,
            ).map_err(|e| e.to_string())?;
            rvllm_fused::ArgmaxLaunch { num_tokens: 1, vocab }
                .launch(
                    br.fused_modules.fn_argmax,
                    logits.device_ptr(),
                    sampled_tokens.device_ptr(),
                    stream,
                ).map_err(|e| e.to_string())?;
        }
        Ok(())
    } };

    // Pinned host buffer for reading back sampled token.
    let mut host_tok = rvllm_mem::PinnedBuf::<i32>::new(1).map_err(|e| e.to_string())?;

    // === PREFILL ===
    // Upload all prompt token IDs.
    let ids_i32: Vec<i32> = prompt_ids.iter().map(|&x| x as i32).collect();
    token_ids_region.copy_from_host(bytemuck_i32(&ids_i32)).map_err(|e| e.to_string())?;

    // Embedding gather: token_ids -> residual (f16 hidden states).
    rvllm_fused::EmbeddingGatherLaunch {
        num_tokens: prompt_len,
        hidden,
        vocab,
    }
    .launch(
        fn_embed,
        residual_ptr,
        br.model.embedding.offset_bytes,
        token_ids_region.device_ptr(),
        stream,
    ).map_err(|e| e.to_string())?;

    // Prefill metadata: positions [0..prompt_len), slots [0..prompt_len),
    // context_lens = [prompt_len].
    {
        let pos: Vec<i32> = (0..prompt_len as i32).collect();
        let slots: Vec<i32> = (0..prompt_len as i32).collect();
        let ctx: Vec<i32> = vec![prompt_len as i32];
        let cu: Vec<i32> = vec![0, prompt_len as i32];
        positions.copy_from_host(bytemuck_i32(&pos)).map_err(|e| e.to_string())?;
        slot_mapping.copy_from_host(bytemuck_i32(&slots)).map_err(|e| e.to_string())?;
        context_lens.copy_from_host(bytemuck_i32(&ctx)).map_err(|e| e.to_string())?;
        cu_seqlens_q.copy_from_host(bytemuck_i32(&cu)).map_err(|e| e.to_string())?;
    }

    let plans_prefill = layer_exec::LayerGemmPlans {
        qkv: plan_qkv_p,
        o: plan_o_p,
        gate_up: plan_gate_up_p,
        down: plan_down_p,
    };
    let prefill_phase = layer_exec::LayerPhase::Prefill {
        cu_seqlens_q: cu_seqlens_q.device_ptr(),
        max_seqlen_q: prompt_len,
    };
    run_forward(prefill_phase, &plans_prefill, prompt_len, k_base_prefill, v_base_prefill)?;
    br.stream.fence().map_err(|e| e.to_string())?;
    let t_prefill = Instant::now();
    eprintln!("prefill done");

    // === DECODE LOOP ===
    let plans_decode = layer_exec::LayerGemmPlans {
        qkv: plan_qkv_d,
        o: plan_o_d,
        gate_up: plan_gate_up_d,
        down: plan_down_d,
    };

    let mut output_ids: Vec<u32> = Vec::with_capacity(max_new as usize);
    // EOS token IDs. RVLLM_EOS overrides; default covers Qwen2.5
    // (151643 = <|endoftext|>, 151645 = <|im_end|>) and Llama (2).
    let eos_ids: Vec<u32> = std::env::var("RVLLM_EOS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 151643, 151645]);
    // After prefill, the last token's hidden state is what we need for
    // the first decode step. The residual buffer currently holds all
    // prompt_len hidden states; we need the LAST one. For N=1 decode the
    // residual buffer is reused from offset 0, so copy last -> first.
    if prompt_len > 1 {
        // Copy the last hidden state (f16, 2*hidden bytes) to position 0.
        use cudarc::driver::sys::*;
        let src = residual_ptr + ((prompt_len - 1) as u64) * (hidden as u64) * 2;
        let dst = residual_ptr;
        let bytes = (hidden * 2) as usize;
        let r = cuMemcpyDtoDAsync_v2(dst, src, bytes, stream as CUstream);
        if r != CUresult::CUDA_SUCCESS {
            return Err("cuMemcpyDtoDAsync residual shuffle".into());
        }
    }

    for step in 0..max_new {
        let cur_pos = prompt_len + step;

        // Metadata for this single decode token.
        let pos = [cur_pos as i32];
        let slot = [cur_pos as i32]; // linear slot = position
        let ctx_len = [cur_pos as i32 + 1]; // context includes this token
        positions.copy_from_host(bytemuck_i32(&pos)).map_err(|e| e.to_string())?;
        slot_mapping.copy_from_host(bytemuck_i32(&slot)).map_err(|e| e.to_string())?;
        context_lens.copy_from_host(bytemuck_i32(&ctx_len)).map_err(|e| e.to_string())?;

        // If this is not the first decode step, we need to embed the
        // previous sampled token into the residual buffer.
        if step > 0 {
            let last_tok = *output_ids.last().ok_or("no token")?;
            let tok_i32 = [last_tok as i32];
            token_ids_region.copy_from_host(bytemuck_i32(&tok_i32)).map_err(|e| e.to_string())?;
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1,
                hidden,
                vocab,
            }
            .launch(
                fn_embed,
                residual_ptr,
                br.model.embedding.offset_bytes,
                token_ids_region.device_ptr(),
                stream,
            ).map_err(|e| e.to_string())?;
        }

        run_forward(
            layer_exec::LayerPhase::Decode,
            &plans_decode,
            1,
            k_base_decode,
            v_base_decode,
        )?;

        // DtoH: read sampled token.
        {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoHAsync_v2(
                host_tok.as_mut_ptr() as *mut _,
                sampled_tokens.device_ptr(),
                4,
                stream as CUstream,
            );
            if r != CUresult::CUDA_SUCCESS {
                return Err("cuMemcpyDtoHAsync sampled token".into());
            }
        }
        br.stream.fence().map_err(|e| e.to_string())?;

        let tok_id = host_tok.as_slice()[0] as u32;
        output_ids.push(tok_id);

        if eos_ids.contains(&tok_id) {
            break;
        }

        // Stream partial output for the first decode step.
        if step == 0 {
            let elapsed = t_prefill.elapsed();
            eprintln!(
                "first token: {} (TTFT {:.1}ms)",
                tok_id,
                elapsed.as_secs_f64() * 1000.0
            );
        }

        // For the first decode step, the residual is already set (from
        // prefill's last hidden state). For subsequent steps, we embed
        // above.
    }

    let total = t_prefill.elapsed();
    let n = output_ids.len();
    if n > 0 {
        eprintln!(
            "decode: {} tokens in {:.2}s ({:.1} tok/s)",
            n,
            total.as_secs_f64(),
            n as f64 / total.as_secs_f64()
        );
    }

    Ok(output_ids)
}

#[cfg(not(feature = "cuda"))]
unsafe fn generate(
    _br: &Bringup,
    _fn_embed: rvllm_kernels::KernelFn,
    _prompt_ids: &[u32],
    _max_new: u32,
) -> Result<Vec<u32>, String> {
    Err("rvllm-eval requires cuda feature".into())
}

fn bytemuck_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
