#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::str::FromStr;

use rvllm_core::{ModelArch, ModelConfig};
use rvllm_kernels::{KernelFn, LoadedModule};
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};
use tokenizers::Tokenizer;

use crate::state::Persona;

const DEFAULT_ARENA_BYTES: usize = 32 * 1024 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct RvllmCudaConfig {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
    pub max_new_tokens: usize,
    pub decode_batch_target: usize,
    pub eos_ids: Vec<u32>,
    pub add_bos: bool,
    pub bos_token_id: u32,
    pub arena_bytes: usize,
}

impl RvllmCudaConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let mut config = Self {
            model_dir: required_env_path("RVLLM_MODEL_DIR")?,
            kernels_dir: required_env_path("RVLLM_KERNELS_DIR")?,
            cutlass_so: required_env_path("RVLLM_CUTLASS_SO")?,
            fa3_so: required_env_path("RVLLM_FA3_SO")?,
            policy_json: required_env_path("RVLLM_POLICY")?,
            max_new_tokens: 256,
            decode_batch_target: optional_env_parse::<usize>("RVLLM_DECODE_BATCH_TARGET")?
                .unwrap_or(1),
            eos_ids: vec![1, 2, 107],
            add_bos: true,
            bos_token_id: 2,
            arena_bytes: arena_bytes_from_env().unwrap_or(DEFAULT_ARENA_BYTES),
        };

        if let Some(max_new_tokens) = optional_env_parse::<usize>("RVLLM_MAX_NEW_TOKENS")? {
            config.max_new_tokens = max_new_tokens;
        }
        if let Some(eos_ids) = optional_env_csv_parse::<u32>("RVLLM_EOS_IDS")? {
            config.eos_ids = eos_ids;
        }
        if let Some(add_bos) = optional_env_bool("RVLLM_ADD_BOS")? {
            config.add_bos = add_bos;
        }
        if let Some(bos_token_id) = optional_env_parse::<u32>("RVLLM_BOS_TOKEN_ID")? {
            config.bos_token_id = bos_token_id;
        }

        Ok(config)
    }
}

pub struct RvllmCudaClient {
    tokenizer: Tokenizer,
    backend: RvllmBackend,
    max_new_tokens: usize,
    decode_batch_target: usize,
    eos_ids: Vec<u32>,
    add_bos: bool,
    bos_token_id: u32,
}

enum RvllmBackend {
    Gemma4 {
        bringup: Gemma4Bringup,
        _embed_module: LoadedModule,
        embed_fn: KernelFn,
        _argmax_module: LoadedModule,
        argmax_fn: KernelFn,
    },
}

impl RvllmCudaClient {
    pub fn from_config(config: RvllmCudaConfig) -> anyhow::Result<Self> {
        let model_config = ModelConfig::load_hf(&config.model_dir)
            .map_err(|e| anyhow::anyhow!("load model config: {e}"))?;
        let tokenizer_path = config.model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;

        let backend = match model_config.architecture {
            ModelArch::Gemma4 => {
                let bringup = Gemma4Bringup::load(
                    Gemma4EnginePaths {
                        model_dir: config.model_dir.clone(),
                        kernels_dir: config.kernels_dir.clone(),
                        cutlass_so: config.cutlass_so.clone(),
                        fa3_so: config.fa3_so.clone(),
                        policy_json: config.policy_json.clone(),
                    },
                    config.arena_bytes,
                )
                .map_err(|e| anyhow::anyhow!("gemma4 bringup: {e}"))?;

                let embed_module = bringup
                    .kernels
                    .load_ptx("embedding_gather_f16")
                    .map_err(|e| anyhow::anyhow!("load embedding_gather_f16: {e}"))?;
                let embed_fn = embed_module
                    .get_function("embedding_gather_f16_kernel")
                    .map_err(|e| anyhow::anyhow!("get embedding_gather_f16_kernel: {e}"))?;

                let argmax_module = bringup
                    .kernels
                    .load_ptx("argmax")
                    .map_err(|e| anyhow::anyhow!("load argmax: {e}"))?;
                let argmax_fn = argmax_module
                    .get_function("argmax_f16_kernel")
                    .map_err(|e| anyhow::anyhow!("get argmax_f16_kernel: {e}"))?;

                RvllmBackend::Gemma4 {
                    bringup,
                    _embed_module: embed_module,
                    embed_fn,
                    _argmax_module: argmax_module,
                    argmax_fn,
                }
            }
            other => anyhow::bail!("rvllm swarm supports Gemma4 only, got {other:?}"),
        };

        Ok(Self {
            tokenizer,
            backend,
            max_new_tokens: config.max_new_tokens,
            decode_batch_target: config.decode_batch_target.clamp(1, 512),
            eos_ids: config.eos_ids,
            add_bos: config.add_bos,
            bos_token_id: config.bos_token_id,
        })
    }

    pub fn completion(&mut self, prompt: &str) -> anyhow::Result<String> {
        let prompt_ids = self.prompt_ids(prompt)?;
        let output_ids = match &self.backend {
            RvllmBackend::Gemma4 {
                bringup,
                embed_fn,
                argmax_fn,
                ..
            } => unsafe {
                bringup.run_generate_batched(
                    *embed_fn,
                    *argmax_fn,
                    &prompt_ids,
                    self.max_new_tokens,
                    &self.eos_ids,
                    self.decode_batch_target as u32,
                )
            }
            .map_err(|e| anyhow::anyhow!("gemma4 generate: {e}"))?,
        };

        self.tokenizer
            .decode(&output_ids, true)
            .map_err(|e| anyhow::anyhow!("decode output: {e}"))
    }

    fn prompt_ids(&self, prompt: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?;

        let mut prompt_ids = encoding.get_ids().to_vec();
        if self.add_bos {
            prompt_ids.insert(0, self.bos_token_id);
        }
        Ok(prompt_ids)
    }
}

pub fn build_client(
    persona: Persona,
    max_new_tokens: usize,
    decode_batch_target: usize,
) -> anyhow::Result<RvllmCudaClient> {
    let mut decode = RvllmCudaConfig::from_env()?;
    decode.max_new_tokens = max_new_tokens;
    decode.decode_batch_target = decode_batch_target.clamp(1, 512);
    let _ = persona;
    RvllmCudaClient::from_config(decode)
}

pub fn complete_once(
    persona: Persona,
    prompt: &str,
    max_new_tokens: usize,
    decode_batch_target: usize,
    _max_iterations: u32,
    _max_depth: u32,
) -> anyhow::Result<String> {
    let mut client = build_client(persona, max_new_tokens, decode_batch_target)?;
    let _ = (_max_iterations, _max_depth);
    client.completion(prompt)
}

fn required_env_path(name: &str) -> anyhow::Result<PathBuf> {
    std::env::var(name)
        .map(PathBuf::from)
        .map_err(|err| anyhow::anyhow!("missing required env var {name}: {err}"))
}

fn optional_env_string(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn optional_env_parse<T>(name: &str) -> anyhow::Result<Option<T>>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    let Some(value) = optional_env_string(name) else {
        return Ok(None);
    };
    value
        .parse::<T>()
        .map(Some)
        .map_err(|err| anyhow::anyhow!("failed to parse {name}={value:?}: {err}"))
}

fn optional_env_csv_parse<T>(name: &str) -> anyhow::Result<Option<Vec<T>>>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    let Some(value) = optional_env_string(name) else {
        return Ok(None);
    };

    let mut values = Vec::new();
    for item in value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        values.push(
            item.parse::<T>()
                .map_err(|err| anyhow::anyhow!("failed to parse {name} item {item:?}: {err}"))?,
        );
    }

    if values.is_empty() {
        anyhow::bail!("{name} must contain at least one comma-separated value");
    }

    Ok(Some(values))
}

fn optional_env_bool(name: &str) -> anyhow::Result<Option<bool>> {
    let Some(value) = optional_env_string(name) else {
        return Ok(None);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => anyhow::bail!("failed to parse {name}={value:?} as bool"),
    }
}

fn arena_bytes_from_env() -> anyhow::Result<usize> {
    let Some(gb) = optional_env_parse::<usize>("RVLLM_ARENA_GB")? else {
        return Ok(DEFAULT_ARENA_BYTES);
    };
    Ok(gb * 1024 * 1024 * 1024)
}
