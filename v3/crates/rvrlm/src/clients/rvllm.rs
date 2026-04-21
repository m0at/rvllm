use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use rvllm_core::{ModelArch, ModelConfig};
use rvllm_kernels::{KernelFn, LoadedModule};
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};
use tokenizers::Tokenizer;

use crate::error::{Result, RlmError};
use crate::types::{ModelUsageSummary, PerplexitySummary, Prompt, UsageSummary};

use super::LanguageModel;

const DEFAULT_ARENA_BYTES: usize = 32 * 1024 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct RvllmCudaConfig {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
    pub model_name: Option<String>,
    pub max_new_tokens: usize,
    pub eos_ids: Vec<u32>,
    pub add_bos: bool,
    pub bos_token_id: u32,
    pub arena_bytes: usize,
}

impl RvllmCudaConfig {
    pub fn new(
        model_dir: impl Into<PathBuf>,
        kernels_dir: impl Into<PathBuf>,
        cutlass_so: impl Into<PathBuf>,
        fa3_so: impl Into<PathBuf>,
        policy_json: impl Into<PathBuf>,
    ) -> Self {
        Self {
            model_dir: model_dir.into(),
            kernels_dir: kernels_dir.into(),
            cutlass_so: cutlass_so.into(),
            fa3_so: fa3_so.into(),
            policy_json: policy_json.into(),
            model_name: None,
            max_new_tokens: 256,
            eos_ids: vec![1, 2, 107],
            add_bos: true,
            bos_token_id: 2,
            arena_bytes: arena_bytes_from_env().unwrap_or(DEFAULT_ARENA_BYTES),
        }
    }

    pub fn from_env() -> Result<Self> {
        let mut config = Self::new(
            required_env_path("RVLLM_MODEL_DIR")?,
            required_env_path("RVLLM_KERNELS_DIR")?,
            required_env_path("RVLLM_CUTLASS_SO")?,
            required_env_path("RVLLM_FA3_SO")?,
            required_env_path("RVLLM_POLICY")?,
        );

        config.model_name = optional_env_string("RVRLM_MODEL_NAME");
        if let Some(max_new_tokens) = optional_env_parse::<usize>("RVRLM_MAX_NEW_TOKENS")? {
            config.max_new_tokens = max_new_tokens;
        }
        if let Some(eos_ids) = optional_env_csv_parse::<u32>("RVRLM_EOS_IDS")? {
            config.eos_ids = eos_ids;
        }
        if let Some(add_bos) = optional_env_bool("RVRLM_ADD_BOS")? {
            config.add_bos = add_bos;
        }
        if let Some(bos_token_id) = optional_env_parse::<u32>("RVRLM_BOS_TOKEN_ID")? {
            config.bos_token_id = bos_token_id;
        }

        Ok(config)
    }
}

pub struct RvllmCudaClient {
    config: RvllmCudaConfig,
    model_name: String,
    tokenizer: Tokenizer,
    backend: RvllmBackend,
    max_new_tokens: usize,
    eos_ids: Vec<u32>,
    add_bos: bool,
    bos_token_id: u32,
    total_calls: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    last_usage: Option<ModelUsageSummary>,
}

enum RvllmBackend {
    Gemma4 {
        bringup: Gemma4Bringup,
        embed_module: LoadedModule,
        embed_fn: KernelFn,
        argmax_module: LoadedModule,
        argmax_fn: KernelFn,
    },
}

impl RvllmCudaClient {
    pub fn from_config(config: RvllmCudaConfig) -> Result<Self> {
        let client_config = config.clone();
        let model_config = ModelConfig::load_hf(&config.model_dir)
            .map_err(|e| RlmError::Client(format!("load model config: {e}")))?;
        let tokenizer_path = config.model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            RlmError::Client(format!("load tokenizer {}: {e}", tokenizer_path.display()))
        })?;
        let model_name = config
            .model_name
            .clone()
            .unwrap_or_else(|| infer_model_name(&config.model_dir));

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
                .map_err(|e| RlmError::Client(format!("gemma4 bringup: {e}")))?;

                let embed_module = bringup
                    .kernels
                    .load_ptx("embedding_gather_f16")
                    .map_err(|e| RlmError::Client(format!("load embedding_gather_f16: {e}")))?;
                let embed_fn = embed_module
                    .get_function("embedding_gather_f16_kernel")
                    .map_err(|e| {
                        RlmError::Client(format!("get embedding_gather_f16_kernel: {e}"))
                    })?;

                let argmax_module = bringup
                    .kernels
                    .load_ptx("argmax")
                    .map_err(|e| RlmError::Client(format!("load argmax: {e}")))?;
                let argmax_fn = argmax_module
                    .get_function("argmax_kernel")
                    .map_err(|e| RlmError::Client(format!("get argmax_kernel: {e}")))?;

                RvllmBackend::Gemma4 {
                    bringup,
                    embed_module,
                    embed_fn,
                    argmax_module,
                    argmax_fn,
                }
            }
            other => {
                return Err(RlmError::Unsupported(format!(
                    "rvRLM CUDA client currently supports Gemma4 only, got {other:?}"
                )));
            }
        };

        Ok(Self {
            config: client_config,
            model_name,
            tokenizer,
            backend,
            max_new_tokens: config.max_new_tokens,
            eos_ids: config.eos_ids,
            add_bos: config.add_bos,
            bos_token_id: config.bos_token_id,
            total_calls: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            last_usage: None,
        })
    }
}

impl LanguageModel for RvllmCudaClient {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn completion(&mut self, prompt: &Prompt) -> Result<String> {
        let prompt_ids = self.prompt_ids(prompt)?;

        let output_ids = match &self.backend {
            RvllmBackend::Gemma4 {
                bringup,
                embed_module: _embed_module,
                embed_fn,
                argmax_module: _argmax_module,
                argmax_fn,
            } => unsafe {
                bringup.run_generate(
                    *embed_fn,
                    *argmax_fn,
                    &prompt_ids,
                    self.max_new_tokens,
                    &self.eos_ids,
                )
            }
            .map_err(|e| RlmError::Client(format!("gemma4 generate: {e}")))?,
        };

        let response = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| RlmError::Client(format!("decode output: {e}")))?;

        let input_tokens = prompt_ids.len() as u64;
        let output_tokens = output_ids.len() as u64;
        self.total_calls += 1;
        self.total_input_tokens += input_tokens;
        self.total_output_tokens += output_tokens;
        self.last_usage = Some(ModelUsageSummary {
            total_calls: 1,
            total_input_tokens: input_tokens,
            total_output_tokens: output_tokens,
            total_cost: None,
        });

        Ok(response)
    }

    fn fork(&self) -> Result<Box<dyn LanguageModel>> {
        Ok(Box::new(Self::from_config(self.config.clone())?))
    }

    fn perplexity(&mut self, prompt: &Prompt) -> Result<PerplexitySummary> {
        let prompt_ids = self.prompt_ids(prompt)?;
        let result = match &self.backend {
            RvllmBackend::Gemma4 {
                bringup,
                embed_module: _embed_module,
                embed_fn,
                argmax_module: _argmax_module,
                argmax_fn: _argmax_fn,
            } => unsafe { bringup.run_ppl(*embed_fn, &prompt_ids) }
                .map_err(|e| RlmError::Client(format!("gemma4 perplexity: {e}")))?,
        };

        Ok(PerplexitySummary {
            root_model: self.model_name.clone(),
            prompt: prompt.clone(),
            perplexity: result.ppl,
            evaluated_tokens: result.n_evaluated,
            total_nll: result.total_nll,
        })
    }

    fn usage_summary(&self) -> UsageSummary {
        let mut model_usage_summaries = BTreeMap::new();
        model_usage_summaries.insert(
            self.model_name.clone(),
            ModelUsageSummary {
                total_calls: self.total_calls,
                total_input_tokens: self.total_input_tokens,
                total_output_tokens: self.total_output_tokens,
                total_cost: None,
            },
        );
        UsageSummary {
            model_usage_summaries,
        }
    }

    fn last_usage(&self) -> Option<ModelUsageSummary> {
        self.last_usage.clone()
    }
}

impl RvllmCudaClient {
    fn prompt_ids(&self, prompt: &Prompt) -> Result<Vec<u32>> {
        let prompt_text = prompt.as_text();
        let encoding = self
            .tokenizer
            .encode(prompt_text, false)
            .map_err(|e| RlmError::Client(format!("tokenize prompt: {e}")))?;

        let mut prompt_ids = encoding.get_ids().to_vec();
        if self.add_bos {
            prompt_ids.insert(0, self.bos_token_id);
        }
        Ok(prompt_ids)
    }
}

fn infer_model_name(model_dir: &Path) -> String {
    model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| model_dir.display().to_string())
}

fn required_env_path(name: &str) -> Result<PathBuf> {
    std::env::var(name)
        .map(PathBuf::from)
        .map_err(|err| RlmError::InvalidConfig(format!("missing required env var {name}: {err}")))
}

fn optional_env_string(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn optional_env_parse<T>(name: &str) -> Result<Option<T>>
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
        .map_err(|err| RlmError::InvalidConfig(format!("failed to parse {name}={value:?}: {err}")))
}

fn optional_env_csv_parse<T>(name: &str) -> Result<Option<Vec<T>>>
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
        values.push(item.parse::<T>().map_err(|err| {
            RlmError::InvalidConfig(format!("failed to parse {name} item {item:?}: {err}"))
        })?);
    }

    if values.is_empty() {
        return Err(RlmError::InvalidConfig(format!(
            "{name} must contain at least one comma-separated value"
        )));
    }

    Ok(Some(values))
}

fn optional_env_bool(name: &str) -> Result<Option<bool>> {
    let Some(value) = optional_env_string(name) else {
        return Ok(None);
    };

    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => Err(RlmError::InvalidConfig(format!(
            "failed to parse {name}={value:?} as bool"
        ))),
    }
}

fn arena_bytes_from_env() -> Option<usize> {
    std::env::var("RVLLM_ARENA_GB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|gb| gb * 1024 * 1024 * 1024)
}
