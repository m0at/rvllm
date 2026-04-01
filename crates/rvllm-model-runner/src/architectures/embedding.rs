//! Embedding model architectures (BERT-style encoder for sentence embeddings).
//!
//! Supports sentence-transformers models (E5, GTE, BGE, etc.) that produce
//! fixed-size embeddings from variable-length input. The encoder runs the
//! full transformer stack and then pools hidden states into a single vector
//! per input sequence.

use half::f16;
use tracing::trace;

use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::norm::{LayerNorm, RMSNorm};
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, embed_tokens, get_or_zeros};
use super::Architecture;

/// Pooling strategy for converting token-level hidden states to a single embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingMode {
    /// Average all token hidden states (most common for sentence-transformers).
    Mean,
    /// Use the hidden state of the first token ([CLS] / BOS).
    Cls,
    /// Use the hidden state of the last token (EOS).
    LastToken,
}

impl PoolingMode {
    /// Parse from a string, defaulting to Mean if unrecognised.
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "cls" | "first" => Self::Cls,
            "last" | "last_token" | "eos" => Self::LastToken,
            _ => Self::Mean,
        }
    }
}

/// A BERT/RoBERTa-style encoder model that produces embeddings.
///
/// Supports sentence-transformers, E5, GTE, BGE and similar encoder models.
/// The forward pass returns f32 hidden states (one vector per input token)
/// which the caller pools and optionally normalises.
pub struct EmbeddingModel {
    hidden_size: usize,
    head_dim: usize,
    #[allow(dead_code)]
    num_heads: usize,
    #[allow(dead_code)]
    num_kv_heads: usize,
    norm_eps: f32,
    pooling: PoolingMode,
    normalize: bool,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<EncoderLayer>,
    final_norm_weight: GpuBuffer<f16>,
    /// Optional final norm bias (LayerNorm-based models).
    final_norm_bias: Option<GpuBuffer<f16>>,
    /// True when the model uses LayerNorm (BERT-style) rather than RMSNorm.
    use_layer_norm: bool,
}

struct EncoderLayer {
    input_norm_weight: GpuBuffer<f16>,
    input_norm_bias: Option<GpuBuffer<f16>>,
    post_attn_norm_weight: GpuBuffer<f16>,
    post_attn_norm_bias: Option<GpuBuffer<f16>>,
    q_proj: GpuBuffer<f16>,
    k_proj: GpuBuffer<f16>,
    v_proj: GpuBuffer<f16>,
    o_proj: GpuBuffer<f16>,
    /// Optional attention biases (BERT has them, some models do not).
    q_bias: Option<GpuBuffer<f16>>,
    k_bias: Option<GpuBuffer<f16>>,
    v_bias: Option<GpuBuffer<f16>>,
    o_bias: Option<GpuBuffer<f16>>,
    /// FFN weights. BERT uses dense (2-layer), while some models use gated MLP.
    intermediate_weight: GpuBuffer<f16>,
    intermediate_bias: Option<GpuBuffer<f16>>,
    output_weight: GpuBuffer<f16>,
    output_bias: Option<GpuBuffer<f16>>,
}

impl EmbeddingModel {
    /// Construct an EmbeddingModel from loaded weights and model config.
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let num_layers = config.num_layers;

        // Detect BERT-style (LayerNorm + biases) vs RMSNorm-based encoders.
        let use_layer_norm = weights
            .get("model.encoder.layer.0.attention.output.LayerNorm.weight")
            .is_ok()
            || weights
                .get("encoder.layer.0.attention.output.LayerNorm.weight")
                .is_ok()
            || weights.get("model.layers.0.input_layernorm.bias").is_ok();

        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .or_else(|_| weights.get_as_buffer("embeddings.word_embeddings.weight"))
            .or_else(|_| weights.get_as_buffer("model.embeddings.word_embeddings.weight"))
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, hidden_size]));

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // Try BERT-style prefix first, then Llama-style.
            let layer = if use_layer_norm {
                Self::load_bert_layer(
                    &weights,
                    i,
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    head_dim,
                )
            } else {
                Self::load_llama_layer(&weights, i, config)
            };
            layers.push(layer);
        }

        let (final_norm_weight, final_norm_bias) = if use_layer_norm {
            let w = weights
                .get_as_buffer("encoder.LayerNorm.weight")
                .or_else(|_| weights.get_as_buffer("model.norm.weight"))
                .unwrap_or_else(|_| GpuBuffer::zeros(&[hidden_size]));
            let b = weights
                .get_as_buffer("encoder.LayerNorm.bias")
                .or_else(|_| weights.get_as_buffer("model.norm.bias"))
                .ok();
            (w, b)
        } else {
            let w = weights
                .get_as_buffer("model.norm.weight")
                .unwrap_or_else(|_| GpuBuffer::zeros(&[hidden_size]));
            (w, None)
        };

        // Default pooling: mean for sentence-transformers, cls for BERT.
        let pooling = if use_layer_norm {
            PoolingMode::Cls
        } else {
            PoolingMode::Mean
        };

        Ok(Self {
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            norm_eps: 1e-5,
            pooling,
            normalize: true,
            embed_tokens,
            layers,
            final_norm_weight,
            final_norm_bias,
            use_layer_norm,
        })
    }

    /// Load a BERT-style encoder layer.
    fn load_bert_layer(
        weights: &ModelWeights,
        idx: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> EncoderLayer {
        let p = format!("encoder.layer.{}", idx);
        let alt = format!("model.encoder.layer.{}", idx);

        let try_get = |suffix: &str, shape: &[usize]| -> GpuBuffer<f16> {
            weights
                .get_as_buffer(&format!("{p}.{suffix}"))
                .or_else(|_| weights.get_as_buffer(&format!("{alt}.{suffix}")))
                .unwrap_or_else(|_| GpuBuffer::zeros(shape))
        };

        let try_get_opt = |suffix: &str, shape: &[usize]| -> Option<GpuBuffer<f16>> {
            weights
                .get_as_buffer(&format!("{p}.{suffix}"))
                .or_else(|_| weights.get_as_buffer(&format!("{alt}.{suffix}")))
                .ok()
                .or_else(|| Some(GpuBuffer::zeros(shape)))
        };

        let attn_out = num_heads * head_dim;

        EncoderLayer {
            input_norm_weight: try_get("attention.output.LayerNorm.weight", &[hidden_size]),
            input_norm_bias: try_get_opt("attention.output.LayerNorm.bias", &[hidden_size]),
            post_attn_norm_weight: try_get("output.LayerNorm.weight", &[hidden_size]),
            post_attn_norm_bias: try_get_opt("output.LayerNorm.bias", &[hidden_size]),
            q_proj: try_get("attention.self.query.weight", &[attn_out, hidden_size]),
            k_proj: try_get("attention.self.key.weight", &[attn_out, hidden_size]),
            v_proj: try_get("attention.self.value.weight", &[attn_out, hidden_size]),
            o_proj: try_get("attention.output.dense.weight", &[hidden_size, attn_out]),
            q_bias: try_get_opt("attention.self.query.bias", &[attn_out]),
            k_bias: try_get_opt("attention.self.key.bias", &[attn_out]),
            v_bias: try_get_opt("attention.self.value.bias", &[attn_out]),
            o_bias: try_get_opt("attention.output.dense.bias", &[hidden_size]),
            intermediate_weight: try_get(
                "intermediate.dense.weight",
                &[intermediate_size, hidden_size],
            ),
            intermediate_bias: try_get_opt("intermediate.dense.bias", &[intermediate_size]),
            output_weight: try_get("output.dense.weight", &[hidden_size, intermediate_size]),
            output_bias: try_get_opt("output.dense.bias", &[hidden_size]),
        }
    }

    /// Load a Llama/Mistral-style encoder layer (for models like E5-mistral).
    fn load_llama_layer(
        weights: &ModelWeights,
        idx: usize,
        config: &ModelRunnerConfig,
    ) -> EncoderLayer {
        let p = format!("model.layers.{}", idx);
        let h = config.hidden_size;
        let qk_size = config.num_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;

        EncoderLayer {
            input_norm_weight: get_or_zeros(weights, &format!("{p}.input_layernorm.weight"), &[h]),
            input_norm_bias: None,
            post_attn_norm_weight: get_or_zeros(
                weights,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
            ),
            post_attn_norm_bias: None,
            q_proj: get_or_zeros(
                weights,
                &format!("{p}.self_attn.q_proj.weight"),
                &[qk_size, h],
            ),
            k_proj: get_or_zeros(
                weights,
                &format!("{p}.self_attn.k_proj.weight"),
                &[kv_size, h],
            ),
            v_proj: get_or_zeros(
                weights,
                &format!("{p}.self_attn.v_proj.weight"),
                &[kv_size, h],
            ),
            o_proj: get_or_zeros(
                weights,
                &format!("{p}.self_attn.o_proj.weight"),
                &[h, qk_size],
            ),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            o_bias: None,
            intermediate_weight: get_or_zeros(
                weights,
                &format!("{p}.mlp.gate_proj.weight"),
                &[config.intermediate_size, h],
            ),
            intermediate_bias: None,
            output_weight: get_or_zeros(
                weights,
                &format!("{p}.mlp.down_proj.weight"),
                &[h, config.intermediate_size],
            ),
            output_bias: None,
        }
    }

    /// Apply the normalization layer (LayerNorm or RMSNorm) to hidden states.
    fn apply_norm(
        &self,
        hidden: &GpuBuffer<f16>,
        weight: &GpuBuffer<f16>,
        bias: Option<&GpuBuffer<f16>>,
    ) -> Result<GpuBuffer<f16>> {
        if self.use_layer_norm {
            let b = bias
                .cloned()
                .unwrap_or_else(|| GpuBuffer::zeros(&[self.hidden_size]));
            LayerNorm::forward(hidden, weight, &b, self.norm_eps)
        } else {
            RMSNorm::forward(hidden, weight, self.norm_eps)
        }
    }

    /// Pool token-level hidden states into a single embedding per sequence.
    ///
    /// For simplicity the current implementation treats the entire batch as one
    /// sequence. Multi-sequence batching with per-sequence boundaries can be
    /// added via attention metadata.
    pub fn pool(&self, hidden_states: &GpuBuffer<f16>, num_tokens: usize) -> Vec<f32> {
        let h = self.hidden_size;
        match self.pooling {
            PoolingMode::Mean => {
                let mut embedding = vec![0.0f32; h];
                for t in 0..num_tokens {
                    let start = t * h;
                    for j in 0..h {
                        embedding[j] += hidden_states.data[start + j].to_f32();
                    }
                }
                let inv_n = 1.0 / num_tokens.max(1) as f32;
                for v in &mut embedding {
                    *v *= inv_n;
                }
                embedding
            }
            PoolingMode::Cls => {
                // First token.
                (0..h).map(|j| hidden_states.data[j].to_f32()).collect()
            }
            PoolingMode::LastToken => {
                let start = (num_tokens.saturating_sub(1)) * h;
                (0..h)
                    .map(|j| hidden_states.data[start + j].to_f32())
                    .collect()
            }
        }
    }

    /// L2-normalize an embedding vector in-place.
    pub fn l2_normalize(embedding: &mut [f32]) {
        let mut sum_sq = 0.0f32;
        for &v in embedding.iter() {
            sum_sq += v * v;
        }
        let inv_norm = (sum_sq + 1e-12).sqrt().recip();
        for v in embedding.iter_mut() {
            *v *= inv_norm;
        }
    }

    /// Return the pooling mode.
    pub fn pooling_mode(&self) -> PoolingMode {
        self.pooling
    }

    /// Return whether normalization is enabled.
    pub fn normalize_embeddings(&self) -> bool {
        self.normalize
    }
}

/// The Architecture impl returns the full hidden-state tensor as "logits"
/// (shape [num_tokens, hidden_size] cast to f32). The API layer is
/// responsible for pooling and normalisation.
impl Architecture for EmbeddingModel {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let h = self.hidden_size;

        // Token embeddings.
        let mut hidden = embed_tokens(&self.embed_tokens, &input.token_ids, h);

        // Encoder layers.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "embedding encoder layer forward");

            // Pre-attention norm.
            let normed = self.apply_norm(
                &hidden,
                &layer.input_norm_weight,
                layer.input_norm_bias.as_ref(),
            )?;

            // QKV projections.
            let q = LinearLayer::forward(&normed, &layer.q_proj, layer.q_bias.as_ref())?;
            let k = LinearLayer::forward(&normed, &layer.k_proj, layer.k_bias.as_ref())?;
            let v = LinearLayer::forward(&normed, &layer.v_proj, layer.v_bias.as_ref())?;

            // RoPE (no-op for BERT, but applied for E5-mistral-style models).
            let (q_rot, k_rot) = if !self.use_layer_norm {
                RotaryEmbedding::forward(&input.position_ids, &q, &k, self.head_dim)?
            } else {
                (q, k)
            };

            // Attention.
            let attn_out =
                attention.forward(&q_rot, &k_rot, &v, &input.attention_metadata, layer_idx)?;

            // Output projection.
            let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, layer.o_bias.as_ref())?;
            add_inplace(&mut hidden, &attn_proj);

            // Post-attention norm + FFN.
            let normed2 = self.apply_norm(
                &hidden,
                &layer.post_attn_norm_weight,
                layer.post_attn_norm_bias.as_ref(),
            )?;

            // FFN: intermediate -> activation -> output.
            let intermediate = LinearLayer::forward(
                &normed2,
                &layer.intermediate_weight,
                layer.intermediate_bias.as_ref(),
            )?;

            // GELU activation (approximation used by BERT / most embedding models).
            let activated = gelu(&intermediate);

            let ffn_out =
                LinearLayer::forward(&activated, &layer.output_weight, layer.output_bias.as_ref())?;
            add_inplace(&mut hidden, &ffn_out);
        }

        // Final normalization.
        let normed_final = self.apply_norm(
            &hidden,
            &self.final_norm_weight,
            self.final_norm_bias.as_ref(),
        )?;

        // Cast to f32 -- the full hidden states, not logits.
        let f32_data: Vec<f32> = normed_final.data.iter().map(|v| v.to_f32()).collect();
        Ok(GpuBuffer::from_vec(f32_data, vec![num_tokens, h]))
    }
}

/// GELU activation (tanh approximation).
fn gelu(input: &GpuBuffer<f16>) -> GpuBuffer<f16> {
    let mut out = Vec::with_capacity(input.data.len());
    for &v in &input.data {
        let x = v.to_f32();
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let inner = 0.7978845608_f32 * (x + 0.044715 * x * x * x);
        let y = 0.5 * x * (1.0 + inner.tanh());
        out.push(f16::from_f32(y));
    }
    GpuBuffer::from_vec(out, input.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{AttentionMetadata, MockAttentionBackend, ModelWeights};

    fn make_config(num_layers: usize, hidden_size: usize) -> ModelRunnerConfig {
        ModelRunnerConfig {
            num_layers,
            hidden_size,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: hidden_size / 2,
            intermediate_size: hidden_size * 4,
            vocab_size: 32,
            max_position: 512,
            dtype: rvllm_core::types::Dtype::Float16,
            rms_norm_eps: 1e-5, rope_theta: 10000.0,
            architecture: "EmbeddingModel".into(),
            layer_types: vec![],
        }
    }

    fn make_input(token_ids: Vec<u32>) -> ModelInput {
        let n = token_ids.len();
        ModelInput {
            position_ids: (0..n as u32).collect(),
            token_ids,
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![],
                context_lens: vec![],
                block_tables: vec![],
                query_lens: vec![1],
                    max_context_len: 0,
            },
            is_prefill: true,
        }
    }

    #[test]
    fn embedding_model_constructs() {
        let config = make_config(1, 8);
        let weights = ModelWeights::default();
        let model = EmbeddingModel::new(weights, &config).unwrap();
        assert_eq!(model.hidden_size, 8);
        assert_eq!(model.layers.len(), 1);
    }

    #[test]
    fn forward_returns_hidden_states() {
        let config = make_config(1, 8);
        let weights = ModelWeights::default();
        let model = EmbeddingModel::new(weights, &config).unwrap();
        let input = make_input(vec![1, 2, 3]);
        let cache = CacheEngine::new(1, 64);
        let attn = MockAttentionBackend;
        let out = model.forward(&input, &cache, &attn).unwrap();
        // Should be [3, 8] -- 3 tokens, hidden_size 8.
        assert_eq!(out.shape, vec![3, 8]);
        assert_eq!(out.data.len(), 24);
    }

    #[test]
    fn mean_pooling() {
        let config = make_config(1, 4);
        let weights = ModelWeights::default();
        let model = EmbeddingModel::new(weights, &config).unwrap();
        let hidden = GpuBuffer::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
                f16::from_f32(7.0),
                f16::from_f32(8.0),
            ],
            vec![2, 4],
        );
        let pooled = model.pool(&hidden, 2);
        assert_eq!(pooled.len(), 4);
        // mean([1,5])=3, mean([2,6])=4, mean([3,7])=5, mean([4,8])=6
        assert!((pooled[0] - 3.0).abs() < 0.01);
        assert!((pooled[1] - 4.0).abs() < 0.01);
        assert!((pooled[2] - 5.0).abs() < 0.01);
        assert!((pooled[3] - 6.0).abs() < 0.01);
    }

    #[test]
    fn cls_pooling() {
        let hidden = GpuBuffer::from_vec(
            vec![
                f16::from_f32(10.0),
                f16::from_f32(20.0),
                f16::from_f32(30.0),
                f16::from_f32(40.0),
            ],
            vec![2, 2],
        );
        let config = make_config(1, 2);
        let weights = ModelWeights::default();
        let mut model = EmbeddingModel::new(weights, &config).unwrap();
        model.pooling = PoolingMode::Cls;
        let pooled = model.pool(&hidden, 2);
        assert!((pooled[0] - 10.0).abs() < 0.01);
        assert!((pooled[1] - 20.0).abs() < 0.01);
    }

    #[test]
    fn last_token_pooling() {
        let hidden = GpuBuffer::from_vec(
            vec![
                f16::from_f32(10.0),
                f16::from_f32(20.0),
                f16::from_f32(30.0),
                f16::from_f32(40.0),
            ],
            vec![2, 2],
        );
        let config = make_config(1, 2);
        let weights = ModelWeights::default();
        let mut model = EmbeddingModel::new(weights, &config).unwrap();
        model.pooling = PoolingMode::LastToken;
        let pooled = model.pool(&hidden, 2);
        assert!((pooled[0] - 30.0).abs() < 0.01);
        assert!((pooled[1] - 40.0).abs() < 0.01);
    }

    #[test]
    fn l2_normalize_works() {
        let mut v = vec![3.0, 4.0];
        EmbeddingModel::l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 0.001);
        assert!((v[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn pooling_mode_from_str() {
        assert_eq!(PoolingMode::from_str_loose("mean"), PoolingMode::Mean);
        assert_eq!(PoolingMode::from_str_loose("cls"), PoolingMode::Cls);
        assert_eq!(PoolingMode::from_str_loose("first"), PoolingMode::Cls);
        assert_eq!(
            PoolingMode::from_str_loose("last_token"),
            PoolingMode::LastToken
        );
        assert_eq!(PoolingMode::from_str_loose("eos"), PoolingMode::LastToken);
        assert_eq!(PoolingMode::from_str_loose("unknown"), PoolingMode::Mean);
    }

    #[test]
    fn gelu_activation() {
        let input = GpuBuffer::from_vec(
            vec![f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(-1.0)],
            vec![3],
        );
        let out = gelu(&input);
        // GELU(0) ~= 0, GELU(1) ~= 0.841, GELU(-1) ~= -0.159
        assert!((out.data[0].to_f32()).abs() < 0.01);
        assert!((out.data[1].to_f32() - 0.841).abs() < 0.02);
        assert!((out.data[2].to_f32() + 0.159).abs() < 0.02);
    }
}
