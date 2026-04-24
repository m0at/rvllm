use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use base64::{engine::general_purpose, Engine as _};
use rayon::ThreadPoolBuilder;
use rustc_hash::FxHashMap;
use rvllm_loader::K2CpuExpertStore;
use serde_json::Value;
use tiktoken_rs::CoreBPE;

const K2_TIKTOKEN_PAT_STR: &str = concat!(
    r"[\p{Han}]+",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"\p{N}{1,3}",
    "|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    "|",
    r"\s*[\r\n]+",
    "|",
    r"\s+(?!\S)",
    "|",
    r"\s+"
);

fn env_path(k: &str) -> Result<PathBuf, String> {
    std::env::var(k)
        .map_err(|_| format!("missing env var: {k}"))
        .map(PathBuf::from)
}

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_bool(k: &str, default: bool) -> bool {
    std::env::var(k)
        .ok()
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("k2-ppl: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let threads = env_usize("K2_THREADS", 16);
    let preload = env_bool("K2_PRELOAD", false);
    let chunk_len = env_usize("RVLLM_PPL_CHUNK", 128);
    let max_chunks = env_usize("RVLLM_PPL_CHUNKS", 0);
    if chunk_len < 2 {
        return Err("RVLLM_PPL_CHUNK must be at least 2".into());
    }

    let text = if let Ok(path) = std::env::var("RVLLM_PPL_TEXT") {
        std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?
    } else if let Ok(p) = std::env::var("RVLLM_PROMPT") {
        p
    } else {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("stdin: {e}"))?;
        buf
    };
    if text.is_empty() {
        return Err("empty text (set RVLLM_PPL_TEXT or RVLLM_PROMPT)".into());
    }

    let _ = ThreadPoolBuilder::new().num_threads(threads).build_global();

    let tokenizer = load_k2_tokenizer(&model_dir)?;
    let cfg = read_model_config(&model_dir)?;
    let bos_token_id = read_u32(&cfg, "bos_token_id")
        .or_else(|| {
            cfg.get("text_config")
                .and_then(|tc| read_u32(tc, "bos_token_id"))
        })
        .ok_or_else(|| "missing bos_token_id in config.json".to_string())?;

    let mut all_ids = tokenizer.encode_ordinary_as::<usize>(&text);
    all_ids.insert(0, bos_token_id as usize);
    if all_ids.len() < 2 {
        return Err("not enough tokens after tokenization".into());
    }
    eprintln!("total tokens: {}", all_ids.len());

    let load_t0 = Instant::now();
    let store = K2CpuExpertStore::open(&model_dir).map_err(|e| format!("open: {e}"))?;
    let preloaded = if preload {
        store
            .preload_hot_matrices()
            .map_err(|e| format!("preload hot matrices: {e}"))?
    } else {
        0
    };
    let load_s = load_t0.elapsed().as_secs_f64();

    let mut chunks: Vec<&[usize]> = all_ids.chunks(chunk_len).collect();
    if let Some(last) = chunks.last() {
        if last.len() < chunk_len {
            chunks.pop();
        }
    }
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
        "k2-ppl: model={} chunk_len={} chunks={} preload={} preloaded_matrices={} load_s={:.3}",
        model_dir.display(),
        chunk_len,
        chunks.len(),
        preload,
        preloaded,
        load_s
    );

    let t_eval = Instant::now();
    let mut total_nll = 0.0f64;
    let mut total_tokens = 0usize;

    for (ci, chunk) in chunks.iter().enumerate() {
        let chunk_t0 = Instant::now();
        let mut cache = store.new_decode_cache();
        let mut chunk_nll = 0.0f64;
        let mut chunk_tokens = 0usize;

        for pos in 0..(chunk.len() - 1) {
            let logits = store
                .forward_step_cached(chunk[pos], pos, &mut cache)
                .map_err(|e| format!("chunk {ci} pos {pos}: {e}"))?;
            let target = chunk[pos + 1];
            let nll = token_nll(&logits, target)?;
            chunk_nll += nll;
            chunk_tokens += 1;
        }

        total_nll += chunk_nll;
        total_tokens += chunk_tokens;

        let chunk_elapsed = chunk_t0.elapsed().as_secs_f64();
        let chunk_ppl = (chunk_nll / chunk_tokens as f64).exp();
        let running_ppl = (total_nll / total_tokens as f64).exp();
        eprintln!(
            "chunk {}/{}: chunk_ppl={:.4} running_ppl={:.4} ({:.1} tok/s, {:.1}s)",
            ci + 1,
            chunks.len(),
            chunk_ppl,
            running_ppl,
            chunk_tokens as f64 / chunk_elapsed.max(1e-9),
            chunk_elapsed
        );
    }

    if total_tokens == 0 {
        return Err("no tokens evaluated".into());
    }

    let elapsed = t_eval.elapsed().as_secs_f64();
    let avg_nll = total_nll / total_tokens as f64;
    let ppl = avg_nll.exp();
    let tok_s = total_tokens as f64 / elapsed.max(1e-9);

    eprintln!("perplexity = {ppl:.4} ({total_tokens} tokens, {elapsed:.1}s, {tok_s:.1} tok/s)");
    println!(
        "{{\"perplexity\":{ppl:.4},\"nll\":{avg_nll:.6},\"tokens\":{total_tokens},\"chunks\":{},\"chunk_len\":{chunk_len},\"load_s\":{load_s:.4},\"preloaded_matrices\":{},\"elapsed_s\":{elapsed:.4},\"tok_s\":{tok_s:.3}}}",
        chunks.len(),
        preloaded
    );

    Ok(())
}

fn token_nll(logits: &[f32], target: usize) -> Result<f64, String> {
    let Some(&target_logit) = logits.get(target) else {
        return Err(format!(
            "target token {target} out of range for logits len {}",
            logits.len()
        ));
    };
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, x| acc.max(x)) as f64;
    let sum_exp: f64 = logits
        .iter()
        .map(|&logit| ((logit as f64) - max_logit).exp())
        .sum();
    Ok((sum_exp.ln() + max_logit) - (target_logit as f64))
}

fn read_model_config(model_dir: &Path) -> Result<Value, String> {
    let path = model_dir.join("config.json");
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn read_u32(v: &Value, key: &str) -> Option<u32> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|x| u32::try_from(x).ok())
}

fn load_k2_tokenizer(model_dir: &Path) -> Result<CoreBPE, String> {
    let tokenizer_cfg_path = model_dir.join("tokenizer_config.json");
    let tokenizer_cfg_text = std::fs::read_to_string(&tokenizer_cfg_path)
        .map_err(|e| format!("read {}: {e}", tokenizer_cfg_path.display()))?;
    let tokenizer_cfg: Value = serde_json::from_str(&tokenizer_cfg_text)
        .map_err(|e| format!("parse {}: {e}", tokenizer_cfg_path.display()))?;
    let pat_str = tokenizer_cfg
        .get("pat_str")
        .and_then(Value::as_str)
        .unwrap_or(K2_TIKTOKEN_PAT_STR);

    let mut encoder = FxHashMap::default();
    let tiktoken_path = model_dir.join("tiktoken.model");
    let tiktoken_text = std::fs::read_to_string(&tiktoken_path)
        .map_err(|e| format!("read {}: {e}", tiktoken_path.display()))?;
    for (lineno, line) in tiktoken_text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(' ');
        let raw = parts.next().ok_or_else(|| {
            format!(
                "{}:{} missing token bytes",
                tiktoken_path.display(),
                lineno + 1
            )
        })?;
        let rank = parts
            .next()
            .ok_or_else(|| format!("{}:{} missing rank", tiktoken_path.display(), lineno + 1))?;
        let token = general_purpose::STANDARD
            .decode(raw)
            .map_err(|e| format!("{}:{} base64: {e}", tiktoken_path.display(), lineno + 1))?;
        let rank: u32 = rank
            .parse()
            .map_err(|e| format!("{}:{} rank parse: {e}", tiktoken_path.display(), lineno + 1))?;
        encoder.insert(token, rank);
    }

    let mut special_tokens = FxHashMap::default();
    if let Some(obj) = tokenizer_cfg
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
    {
        for (id_str, meta) in obj {
            let id: u32 = id_str
                .parse()
                .map_err(|e| format!("bad added_tokens_decoder id {id_str}: {e}"))?;
            let content = meta
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("added_tokens_decoder[{id_str}] missing content"))?;
            special_tokens.insert(content.to_string(), id);
        }
    }

    CoreBPE::new(encoder, special_tokens, pat_str).map_err(|e| format!("CoreBPE::new: {e}"))
}
