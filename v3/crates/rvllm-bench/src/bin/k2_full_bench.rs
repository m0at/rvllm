use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use base64::{engine::general_purpose, Engine as _};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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

fn env_u64(k: &str, default: u64) -> u64 {
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

fn env_f32(k: &str, default: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("k2-full-bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let max_new = env_usize("RVLLM_MAX_TOKENS", 128);
    let temperature = env_f32("RVLLM_TEMPERATURE", 0.0);
    let top_p = env_f32("RVLLM_TOP_P", 0.95);
    let threads = env_usize("K2_THREADS", 16);
    let seed = env_u64("K2_SEED", 0);
    let preload = env_bool("K2_PRELOAD", false);
    let prompt = if let Ok(p) = std::env::var("RVLLM_PROMPT") {
        p
    } else {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("stdin: {e}"))?;
        buf
    };
    if prompt.trim().is_empty() {
        return Err("empty prompt (set RVLLM_PROMPT or pipe to stdin)".into());
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
    let eos_token_id = read_u32(&cfg, "eos_token_id")
        .or_else(|| {
            cfg.get("text_config")
                .and_then(|tc| read_u32(tc, "eos_token_id"))
        })
        .ok_or_else(|| "missing eos_token_id in config.json".to_string())?;

    let mut prompt_ids: Vec<usize> = tokenizer.encode_ordinary_as::<usize>(&prompt);
    prompt_ids.insert(0, bos_token_id as usize);

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

    eprintln!(
        "k2-full-bench: model={} prompt_tokens={} max_new={} temp={} top_p={} threads={} seed={}",
        model_dir.display(),
        prompt_ids.len(),
        max_new,
        temperature,
        top_p,
        threads,
        seed
    );
    eprintln!(
        "load_s={load_s:.3} preload={} preloaded_matrices={}",
        preload, preloaded
    );

    let mut cache = store.new_decode_cache();
    let mut logits = Vec::new();
    let prefill_t0 = Instant::now();
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        logits = store
            .forward_step_cached(token_id, pos, &mut cache)
            .map_err(|e| format!("prefill token {pos}: {e}"))?;
    }
    let prefill_s = prefill_t0.elapsed().as_secs_f64();
    let ttft_ms = prefill_s * 1000.0;
    let prefill_tok_s = prompt_ids.len() as f64 / prefill_s.max(1e-9);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut generated = Vec::new();
    let mut last_token = sample_token(&logits, temperature, top_p, &mut rng)?;
    generated.push(last_token);

    let decode_t0 = Instant::now();
    let mut finished_eos = last_token == eos_token_id as usize;
    while generated.len() < max_new && !finished_eos {
        let pos = prompt_ids.len() + generated.len() - 1;
        logits = store
            .forward_step_cached(last_token, pos, &mut cache)
            .map_err(|e| format!("decode pos {pos}: {e}"))?;
        last_token = sample_token(&logits, temperature, top_p, &mut rng)?;
        finished_eos = last_token == eos_token_id as usize;
        generated.push(last_token);
    }
    let decode_s = decode_t0.elapsed().as_secs_f64();
    let decode_tokens = generated.len().saturating_sub(1);
    let decode_tok_s = if decode_tokens > 0 {
        decode_tokens as f64 / decode_s.max(1e-9)
    } else {
        0.0
    };
    let total_s = prefill_s + decode_s;
    let total_tok_s = generated.len() as f64 / total_s.max(1e-9);

    let output_ids_u32: Vec<u32> = generated.iter().map(|&x| x as u32).collect();
    let output_text = tokenizer
        .decode(&output_ids_u32)
        .map_err(|e| format!("decode output: {e}"))?;

    eprintln!(
        "prefill: {} tokens in {:.3}s ({:.2} tok/s)",
        prompt_ids.len(),
        prefill_s,
        prefill_tok_s
    );
    eprintln!(
        "decode: {} tokens in {:.3}s ({:.2} tok/s) ttft={:.1}ms total={:.3}s",
        generated.len(),
        decode_s,
        decode_tok_s,
        ttft_ms,
        total_s
    );
    eprintln!("output:\n{output_text}");

    println!(
        "{{\"prompt_tokens\":{},\"generated_tokens\":{},\"load_s\":{:.4},\"preloaded_matrices\":{},\"ttft_ms\":{:.3},\"prefill_s\":{:.4},\"prefill_tok_s\":{:.3},\"decode_s\":{:.4},\"decode_tok_s\":{:.3},\"total_s\":{:.4},\"total_tok_s\":{:.3},\"temperature\":{:.3},\"top_p\":{:.3},\"finished_eos\":{},\"seed\":{},\"output_text\":{}}}",
        prompt_ids.len(),
        generated.len(),
        load_s,
        preloaded,
        ttft_ms,
        prefill_s,
        prefill_tok_s,
        decode_s,
        decode_tok_s,
        total_s,
        total_tok_s,
        temperature,
        top_p,
        if finished_eos { "true" } else { "false" },
        seed,
        serde_json::to_string(&output_text).map_err(|e| format!("json output_text: {e}"))?,
    );
    Ok(())
}

fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    rng: &mut StdRng,
) -> Result<usize, String> {
    if logits.is_empty() {
        return Err("empty logits".into());
    }
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx)
            .ok_or_else(|| "argmax failed".to_string());
    }

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, x| acc.max(x));
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &logit)| (idx, ((logit - max_logit) / temperature).exp()))
        .collect();
    probs.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut kept = Vec::new();
    let mut running = 0.0f32;
    let total = probs.iter().map(|(_, p)| *p).sum::<f32>().max(1e-12);
    for (idx, p) in probs {
        let norm = p / total;
        running += norm;
        kept.push((idx, norm));
        if running >= top_p.max(1e-6) {
            break;
        }
    }
    let kept_sum = kept.iter().map(|(_, p)| *p).sum::<f32>().max(1e-12);
    let target = rng.gen::<f32>();
    let mut cdf = 0.0f32;
    for (idx, p) in kept {
        cdf += p / kept_sum;
        if target <= cdf {
            return Ok(idx);
        }
    }
    Err("sampling fell through".into())
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
