//! Model architecture implementations.

pub mod cohere;
pub mod deepseek;
pub mod embedding;
pub mod gemma;
pub mod gpt_oss;
pub mod gpt_neox;
pub mod llama;
pub mod mistral;
pub mod mixtral;
pub mod phi;
pub mod qwen2;

use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, LLMError, ModelWeights, Result};
use crate::input::ModelInput;
use crate::runner::ModelRunnerConfig;

/// Trait for a causal LM architecture.
pub trait Architecture: Send + Sync {
    /// Run the full forward pass: embed -> layers -> LM head -> logits.
    fn forward(
        &self,
        input: &ModelInput,
        cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>>;
}

/// Factory function to instantiate a model architecture by name.
pub fn create_model(
    architecture: &str,
    weights: ModelWeights,
    config: &ModelRunnerConfig,
) -> Result<Box<dyn Architecture>> {
    match architecture {
        "LlamaForCausalLM" => Ok(Box::new(llama::LlamaForCausalLM::new(weights, config)?)),
        "MistralForCausalLM" => Ok(Box::new(mistral::MistralForCausalLM::new(weights, config)?)),
        "Qwen2ForCausalLM" => Ok(Box::new(qwen2::Qwen2ForCausalLM::new(weights, config)?)),
        "CohereForCausalLM" => Ok(Box::new(cohere::CohereForCausalLM::new(weights, config)?)),
        "GPTNeoXForCausalLM" => Ok(Box::new(gpt_neox::GPTNeoXForCausalLM::new(
            weights, config,
        )?)),
        "StableLmForCausalLM" | "StableLMForCausalLM" => Ok(Box::new(
            gpt_neox::StableLmForCausalLM::new(weights, config)?,
        )),
        "GemmaForCausalLM" => Ok(Box::new(gemma::GemmaForCausalLM::new(weights, config)?)),
        "Gemma2ForCausalLM" => Ok(Box::new(gemma::Gemma2ForCausalLM::new(weights, config)?)),
        "DeepSeekV2ForCausalLM" | "DeepseekV2ForCausalLM" => Ok(Box::new(
            deepseek::DeepSeekV2ForCausalLM::new(weights, config)?,
        )),
        "MixtralForCausalLM" => Ok(Box::new(mixtral::MixtralForCausalLM::new(weights, config)?)),
        "GptOssForCausalLM" => Ok(Box::new(gpt_oss::GptOssForCausalLM::new(
            weights, config,
        )?)),
        "PhiForCausalLM" | "Phi3ForCausalLM" | "Phi3SmallForCausalLM" => {
            Ok(Box::new(phi::PhiForCausalLM::new(weights, config)?))
        }
        // Embedding / encoder-only models (sentence-transformers, E5, GTE, BGE, etc.)
        "BertModel"
        | "RobertaModel"
        | "XLMRobertaModel"
        | "E5Model"
        | "GTEModel"
        | "BGEModel"
        | "EmbeddingModel"
        | "SentenceTransformer" => Ok(Box::new(embedding::EmbeddingModel::new(weights, config)?)),
        other => Err(LLMError::ModelError(format!(
            "unsupported architecture: {}",
            other
        ))),
    }
}
