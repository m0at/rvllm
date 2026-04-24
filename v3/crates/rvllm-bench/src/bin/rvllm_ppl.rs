//! rvllm-ppl: perplexity measurement via full-logits readback.
//!
//! Processes tokens one at a time in decode mode, DtoH-ing one logits
//! row per step. Delegates to Bringup::run_ppl which uses the exact
//! same arena layout as the bench to avoid FA3 workspace issues.
//!
//! Env vars (same as rvllm-bench plus):
//!   RVLLM_MODEL_DIR   = HF snapshot dir (must contain tokenizer.json)
//!   RVLLM_KERNELS_DIR, RVLLM_CUTLASS_SO, RVLLM_FA3_SO, RVLLM_POLICY
//!   RVLLM_ARENA_GB    = arena size in GB (default 32)
//!   RVLLM_PPL_CHUNK   = chunk length in tokens (default 128)
//!   RVLLM_PPL_CHUNKS  = max number of chunks (default 0 = all)
//!   RVLLM_PPL_TEXT    = path to plain-text file to evaluate
//!   RVLLM_PROMPT      = inline text (alternative to file)

use std::path::PathBuf;
use std::time::Instant;

use rvllm_core::{ModelArch as HfModelArch, ModelConfig};
#[cfg(feature = "cuda")]
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};
use rvllm_runtime::{Bringup, EnginePaths};

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

fn env_bool_any(keys: &[&str]) -> Result<Option<bool>, String> {
    for key in keys {
        if let Ok(raw) = std::env::var(key) {
            let value = match raw.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => true,
                "0" | "false" | "no" | "off" => false,
                other => return Err(format!("invalid boolean for {key}: {other}")),
            };
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn env_u32_any(keys: &[&str]) -> Result<Option<u32>, String> {
    for key in keys {
        if let Ok(raw) = std::env::var(key) {
            let value = raw
                .trim()
                .parse::<u32>()
                .map_err(|e| format!("invalid integer for {key}: {e}"))?;
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    if let Err(e) = run() {
        eprintln!("rvllm-ppl: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;

    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| format!("tokenizer load {}: {e}", tok_path.display()))?;

    let text = if let Ok(path) = std::env::var("RVLLM_PPL_TEXT") {
        std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?
    } else if let Ok(p) = std::env::var("RVLLM_PROMPT") {
        p
    } else {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)
            .map_err(|e| format!("stdin: {e}"))?;
        buf
    };
    if text.is_empty() {
        return Err("empty text (set RVLLM_PPL_TEXT or RVLLM_PROMPT)".into());
    }

    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| format!("tokenize: {e}"))?;
    let add_bos = env_bool_any(&["RVLLM_ADD_BOS"])?.unwrap_or(true);
    let bos_token_id = env_u32_any(&["RVLLM_BOS_TOKEN_ID"])?.unwrap_or(2);
    let all_ids: Vec<u32> = if add_bos {
        std::iter::once(bos_token_id)
            .chain(encoding.get_ids().iter().copied())
            .collect()
    } else {
        encoding.get_ids().to_vec()
    };
    eprintln!("total tokens: {}", all_ids.len());

    let chunk_len: usize = std::env::var("RVLLM_PPL_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let max_chunks: usize = std::env::var("RVLLM_PPL_CHUNKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let paths = EnginePaths {
        model_dir: model_dir.clone(),
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path("RVLLM_CUTLASS_SO")?,
        fa3_so: env_path("RVLLM_FA3_SO")?,
        policy_json: env_path("RVLLM_POLICY")?,
    };
    let arena_gb: usize = std::env::var("RVLLM_ARENA_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let arena_bytes: usize = arena_gb * 1024 * 1024 * 1024;
    let t0 = Instant::now();

    if is_gemma4_model_dir(&model_dir)? {
        #[cfg(not(feature = "cuda"))]
        {
            return Err("Gemma4 PPL requires rvllm-bench built with --features cuda".into());
        }
        #[cfg(feature = "cuda")]
        {
            let g4_paths = Gemma4EnginePaths {
                model_dir: paths.model_dir,
                kernels_dir: paths.kernels_dir.clone(),
                cutlass_so: paths.cutlass_so,
                fa3_so: paths.fa3_so,
                policy_json: paths.policy_json,
            };
            let g4 = Gemma4Bringup::load(g4_paths, arena_bytes)
                .map_err(|e| format!("gemma4 bringup: {e}"))?;
            eprintln!("bringup: {:.2}s", t0.elapsed().as_secs_f64());

            let embed_mod = g4
                .kernels
                .load_ptx("embedding_gather_f16")
                .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
            let fn_embed = embed_mod
                .get_function("embedding_gather_f16_kernel")
                .map_err(|e| format!("get embedding_gather_f16_kernel: {e}"))?;

            let mut chunks: Vec<&[u32]> = all_ids.chunks(chunk_len).collect();
            if max_chunks > 0 && chunks.len() > max_chunks {
                chunks.truncate(max_chunks);
            }
            if chunks.is_empty() {
                return Err(format!(
                    "not enough tokens ({}) for chunk_len={chunk_len}",
                    all_ids.len()
                ));
            }
            eprintln!("evaluating {} chunks of {} tokens", chunks.len(), chunk_len);
            if chunks.len() > 1 {
                eprintln!(
                    "warning: chunked PPL resets Gemma4 KV/context for each chunk; \
                     running_ppl is only comparable across identical chunk settings"
                );
            }

            let mut total_nll: f64 = 0.0;
            let mut total_tokens: usize = 0;
            let t_eval = Instant::now();

            for (ci, chunk) in chunks.iter().enumerate() {
                let chunk_t0 = Instant::now();
                let result = unsafe { g4.run_ppl(fn_embed, chunk) }
                    .map_err(|e| format!("run_ppl chunk {ci}: {e}"))?;
                total_nll += result.total_nll;
                total_tokens += result.n_evaluated;
                let running_ppl = (total_nll / total_tokens as f64).exp();
                let chunk_ppl = if result.n_evaluated > 0 {
                    (result.total_nll / result.n_evaluated as f64).exp()
                } else {
                    0.0
                };
                eprintln!(
                    "chunk {}/{}: chunk_ppl={:.4} running_ppl={:.4} ({:.1} tok/s, {:.1}s)",
                    ci + 1,
                    chunks.len(),
                    chunk_ppl,
                    running_ppl,
                    chunk.len() as f64 / chunk_t0.elapsed().as_secs_f64(),
                    chunk_t0.elapsed().as_secs_f64()
                );
            }

            let ppl = (total_nll / total_tokens as f64).exp();
            let elapsed = t_eval.elapsed().as_secs_f64();
            eprintln!("perplexity = {ppl:.4} ({total_tokens} tokens, {elapsed:.1}s)");
            println!("{{\"perplexity\":{ppl:.4},\"tokens\":{total_tokens},\"chunk_len\":{chunk_len},\"elapsed_s\":{elapsed:.1}}}");
            return Ok(());
        }
    }

    let br = Bringup::load(paths, arena_bytes).map_err(|e| format!("bringup: {e}"))?;
    eprintln!("bringup: {:.2}s", t0.elapsed().as_secs_f64());

    let embed_mod = br
        .kernels
        .load_ptx("embedding_gather_f16")
        .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
    let fn_embed = embed_mod
        .get_function("embedding_gather_f16_kernel")
        .map_err(|e| format!("get embedding_gather_f16_kernel: {e}"))?;

    // Chunk the input.
    let mut chunks: Vec<&[u32]> = all_ids.chunks(chunk_len).collect();
    if max_chunks > 0 && chunks.len() > max_chunks {
        chunks.truncate(max_chunks);
    }
    if chunks.is_empty() {
        return Err(format!(
            "not enough tokens ({}) for chunk_len={chunk_len}",
            all_ids.len()
        ));
    }
    eprintln!(
        "evaluating {} chunks of {} tokens ({} total)",
        chunks.len(),
        chunk_len,
        all_ids.len()
    );
    if chunks.len() > 1 {
        eprintln!(
            "warning: chunked PPL resets KV/context for each chunk; \
             running_ppl is only comparable across identical chunk settings"
        );
    }

    let mut total_nll: f64 = 0.0;
    let mut total_tokens: usize = 0;
    let t_eval = Instant::now();

    for (ci, chunk) in chunks.iter().enumerate() {
        let chunk_t0 = Instant::now();
        let result = unsafe { br.run_ppl(fn_embed, chunk) }
            .map_err(|e| format!("run_ppl chunk {ci}: {e}"))?;
        total_nll += result.total_nll;
        total_tokens += result.n_evaluated;
        let running_ppl = (total_nll / total_tokens as f64).exp();
        let chunk_ppl = if result.n_evaluated > 0 {
            (result.total_nll / result.n_evaluated as f64).exp()
        } else {
            0.0
        };
        let chunk_elapsed = chunk_t0.elapsed().as_secs_f64();
        eprintln!(
            "chunk {}/{}: chunk_ppl={:.4} running_ppl={:.4} ({:.1} tok/s, {:.1}s)",
            ci + 1,
            chunks.len(),
            chunk_ppl,
            running_ppl,
            chunk.len() as f64 / chunk_elapsed,
            chunk_elapsed
        );
    }

    let ppl = (total_nll / total_tokens as f64).exp();
    let elapsed = t_eval.elapsed().as_secs_f64();
    eprintln!("perplexity = {ppl:.4} ({total_tokens} tokens, {elapsed:.1}s)");
    println!(
        "{{\"perplexity\":{ppl:.4},\"tokens\":{total_tokens},\"chunk_len\":{chunk_len},\"elapsed_s\":{elapsed:.1}}}"
    );
    Ok(())
}
