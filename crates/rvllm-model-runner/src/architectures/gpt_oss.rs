//! GptOssForCausalLM architecture.
//!
//! This mirrors the dense attention and routed-MoE layout used by GPT-OSS.
//! The production CUDA runner handles GPT-OSS sinks, alternating sliding/full
//! attention, and MXFP4 experts. This CPU/mock architecture keeps targeted
//! errors for those paths because the mock `AttentionBackend` shim still lacks
//! the extra controls needed to execute them faithfully.

use half::f16;
use tracing::trace;

use crate::bridge::{
    AttentionBackend, CacheEngine, GpuBuffer, LLMError, ModelWeights, Result,
};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::norm::RMSNorm;
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, embed_tokens, get_or_zeros, lm_head};
use super::Architecture;

const GPT_OSS_SWIGLU_ALPHA: f32 = 1.702;
const GPT_OSS_SWIGLU_LIMIT: f32 = 7.0;
const NONZERO_SINK_EPSILON: f32 = 1e-6;

pub struct GptOssForCausalLM {
    config: GptOssConfig,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<GptOssLayer>,
    norm_weight: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

struct GptOssConfig {
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    attention_bias: bool,
    sliding_window: Option<usize>,
    layer_types: Vec<String>,
    num_local_experts: usize,
    num_experts_per_tok: usize,
}

struct GptOssLayer {
    input_layernorm: GpuBuffer<f16>,
    post_attention_layernorm: GpuBuffer<f16>,
    q_proj: GpuBuffer<f16>,
    q_proj_bias: Option<GpuBuffer<f16>>,
    k_proj: GpuBuffer<f16>,
    k_proj_bias: Option<GpuBuffer<f16>>,
    v_proj: GpuBuffer<f16>,
    v_proj_bias: Option<GpuBuffer<f16>>,
    o_proj: GpuBuffer<f16>,
    o_proj_bias: Option<GpuBuffer<f16>>,
    sinks: GpuBuffer<f16>,
    layer_type: String,
    mlp: GptOssMlp,
}

struct GptOssMlp {
    router_weight: GpuBuffer<f16>,
    router_bias: GpuBuffer<f16>,
    expert_storage: ExpertStorage,
    num_local_experts: usize,
    num_experts_per_tok: usize,
    hidden_size: usize,
}

enum ExpertStorage {
    Unquantized(GptOssExpertWeights),
    QuantizedMxfp4,
    Missing,
}

struct GptOssExpertWeights {
    gate_up_proj: GpuBuffer<f16>,
    gate_up_proj_bias: GpuBuffer<f16>,
    down_proj: GpuBuffer<f16>,
    down_proj_bias: GpuBuffer<f16>,
    hidden_size: usize,
    intermediate_size: usize,
}

impl GptOssForCausalLM {
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        if config.num_local_experts == 0 {
            return Err(LLMError::ModelError(
                "gpt-oss requires num_local_experts in the model runner config".into(),
            ));
        }
        if config.num_experts_per_tok == 0 {
            return Err(LLMError::ModelError(
                "gpt-oss requires num_experts_per_tok in the model runner config".into(),
            ));
        }

        let layer_types = if config.layer_types.is_empty() {
            vec!["full_attention".to_string(); config.num_layers]
        } else {
            config.layer_types.clone()
        };
        if layer_types.len() != config.num_layers {
            return Err(LLMError::ModelError(format!(
                "gpt-oss layer_types length mismatch: expected {}, got {}",
                config.num_layers,
                layer_types.len()
            )));
        }

        let cfg = GptOssConfig {
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            rms_norm_eps: config.rms_norm_eps,
            attention_bias: config.attention_bias,
            sliding_window: config.sliding_window,
            layer_types,
            num_local_experts: config.num_local_experts,
            num_experts_per_tok: config.num_experts_per_tok,
        };

        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, cfg.hidden_size]));

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{}", i);
            let layer_type = cfg.layer_types[i].clone();

            let expert_storage = build_expert_storage(
                &weights,
                &p,
                cfg.num_local_experts,
                cfg.hidden_size,
                cfg.intermediate_size,
            );

            layers.push(GptOssLayer {
                input_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.input_layernorm.weight"),
                    &[cfg.hidden_size],
                ),
                post_attention_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[cfg.hidden_size],
                ),
                q_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.q_proj.weight"),
                    &[cfg.num_heads * cfg.head_dim, cfg.hidden_size],
                ),
                q_proj_bias: load_optional_bias(
                    &weights,
                    &format!("{p}.self_attn.q_proj.bias"),
                    cfg.num_heads * cfg.head_dim,
                    cfg.attention_bias,
                ),
                k_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.k_proj.weight"),
                    &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
                ),
                k_proj_bias: load_optional_bias(
                    &weights,
                    &format!("{p}.self_attn.k_proj.bias"),
                    cfg.num_kv_heads * cfg.head_dim,
                    cfg.attention_bias,
                ),
                v_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
                ),
                v_proj_bias: load_optional_bias(
                    &weights,
                    &format!("{p}.self_attn.v_proj.bias"),
                    cfg.num_kv_heads * cfg.head_dim,
                    cfg.attention_bias,
                ),
                o_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    &[cfg.hidden_size, cfg.num_heads * cfg.head_dim],
                ),
                o_proj_bias: load_optional_bias(
                    &weights,
                    &format!("{p}.self_attn.o_proj.bias"),
                    cfg.hidden_size,
                    cfg.attention_bias,
                ),
                sinks: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.sinks"),
                    &[cfg.num_heads],
                ),
                layer_type,
                mlp: GptOssMlp {
                    router_weight: get_or_zeros(
                        &weights,
                        &format!("{p}.mlp.router.weight"),
                        &[cfg.num_local_experts, cfg.hidden_size],
                    ),
                    router_bias: get_or_zeros(
                        &weights,
                        &format!("{p}.mlp.router.bias"),
                        &[cfg.num_local_experts],
                    ),
                    expert_storage,
                    num_local_experts: cfg.num_local_experts,
                    num_experts_per_tok: cfg.num_experts_per_tok,
                    hidden_size: cfg.hidden_size,
                },
            });
        }

        let norm_weight = weights
            .get_as_buffer("model.norm.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.hidden_size]));

        let lm_head_weight = weights
            .get_as_buffer("lm_head.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.vocab_size, cfg.hidden_size]));

        Ok(Self {
            config: cfg,
            embed_tokens,
            layers,
            norm_weight,
            lm_head_weight,
        })
    }
}

impl Architecture for GptOssForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let mut hidden = embed_tokens(&self.embed_tokens, &input.token_ids, self.config.hidden_size);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "gpt_oss layer forward");

            let normed =
                RMSNorm::forward(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let q = LinearLayer::forward(&normed, &layer.q_proj, layer.q_proj_bias.as_ref())?;
            let k = LinearLayer::forward(&normed, &layer.k_proj, layer.k_proj_bias.as_ref())?;
            let v = LinearLayer::forward(&normed, &layer.v_proj, layer.v_proj_bias.as_ref())?;

            let (q_rot, k_rot) =
                RotaryEmbedding::forward(&input.position_ids, &q, &k, self.config.head_dim)?;

            if layer.layer_type == "sliding_attention" {
                return Err(LLMError::ModelError(format!(
                    "gpt-oss layer {} uses sliding attention (window {:?}), but the CPU/mock attention backend does not expose alternating sliding/full controls",
                    layer_idx,
                    self.config.sliding_window
                )));
            }
            if has_nonzero_sinks(&layer.sinks) {
                return Err(LLMError::ModelError(format!(
                    "gpt-oss layer {} uses attention sinks, but the CPU/mock attention backend does not expose sink logits",
                    layer_idx
                )));
            }

            let attn_out =
                attention.forward(&q_rot, &k_rot, &v, &input.attention_metadata, layer_idx)?;
            let attn_proj =
                LinearLayer::forward(&attn_out, &layer.o_proj, layer.o_proj_bias.as_ref())?;
            add_inplace(&mut hidden, &attn_proj);

            let normed2 =
                RMSNorm::forward(&hidden, &layer.post_attention_layernorm, self.config.rms_norm_eps)?;
            let mlp_out = layer.mlp.forward(&normed2)?;
            add_inplace(&mut hidden, &mlp_out);
        }

        let normed_final = RMSNorm::forward(&hidden, &self.norm_weight, self.config.rms_norm_eps)?;
        lm_head(
            &normed_final,
            &self.lm_head_weight,
            num_tokens,
            self.config.vocab_size,
        )
    }
}

impl GptOssMlp {
    fn forward(&self, input: &GpuBuffer<f16>) -> Result<GpuBuffer<f16>> {
        let experts = match &self.expert_storage {
            ExpertStorage::Unquantized(weights) => weights,
            ExpertStorage::QuantizedMxfp4 => {
                return Err(LLMError::ModelError(
                    "gpt-oss expert tensors are stored in MXFP4 blocks/scales; the CPU/mock architecture does not execute them".into(),
                ))
            }
            ExpertStorage::Missing => {
                return Err(LLMError::ModelError(
                    "gpt-oss expert tensors are missing from the loaded weights".into(),
                ))
            }
        };

        let num_tokens = input.shape.first().copied().unwrap_or(0);
        let router_logits =
            LinearLayer::forward(input, &self.router_weight, Some(&self.router_bias))?;

        let mut output = vec![f16::ZERO; num_tokens * self.hidden_size];
        let mut expert_tokens = vec![Vec::<(usize, f32)>::new(); self.num_local_experts];
        let top_k = self.num_experts_per_tok.min(self.num_local_experts);

        for token_idx in 0..num_tokens {
            let offset = token_idx * self.num_local_experts;
            let logits: Vec<f32> = (0..self.num_local_experts)
                .map(|expert_idx| router_logits.data[offset + expert_idx].to_f32())
                .collect();
            let top_indices = top_k_indices(&logits, top_k);
            let top_logits: Vec<f32> = top_indices.iter().map(|&idx| logits[idx]).collect();
            let route_weights = softmax(&top_logits);

            for (rank, &expert_idx) in top_indices.iter().enumerate() {
                expert_tokens[expert_idx].push((token_idx, route_weights[rank]));
            }
        }

        for (expert_idx, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }

            let mut batch_input = Vec::with_capacity(tokens.len() * self.hidden_size);
            for &(token_idx, _) in tokens {
                let start = token_idx * self.hidden_size;
                batch_input.extend_from_slice(&input.data[start..start + self.hidden_size]);
            }
            let batch = GpuBuffer::from_vec(batch_input, vec![tokens.len(), self.hidden_size]);
            let expert_out = experts.forward(expert_idx, &batch)?;

            for (batch_idx, &(token_idx, weight)) in tokens.iter().enumerate() {
                let src_offset = batch_idx * self.hidden_size;
                let dst_offset = token_idx * self.hidden_size;
                for hidden_idx in 0..self.hidden_size {
                    let cur = output[dst_offset + hidden_idx].to_f32();
                    let val = expert_out.data[src_offset + hidden_idx].to_f32();
                    output[dst_offset + hidden_idx] = f16::from_f32(cur + val * weight);
                }
            }
        }

        Ok(GpuBuffer::from_vec(output, vec![num_tokens, self.hidden_size]))
    }
}

impl GptOssExpertWeights {
    fn forward(&self, expert_idx: usize, input: &GpuBuffer<f16>) -> Result<GpuBuffer<f16>> {
        let num_tokens = input.shape.first().copied().unwrap_or(0);
        let gate_up = project_in_out_with_bias(
            input,
            expert_tensor_slice(
                &self.gate_up_proj,
                expert_idx,
                self.hidden_size,
                self.intermediate_size * 2,
            ),
            expert_vector_slice(
                &self.gate_up_proj_bias,
                expert_idx,
                self.intermediate_size * 2,
            ),
            self.hidden_size,
            self.intermediate_size * 2,
        );
        let gated = apply_gpt_oss_swiglu(&gate_up, num_tokens, self.intermediate_size);
        let down = project_in_out_with_bias(
            &gated,
            expert_tensor_slice(
                &self.down_proj,
                expert_idx,
                self.intermediate_size,
                self.hidden_size,
            ),
            expert_vector_slice(&self.down_proj_bias, expert_idx, self.hidden_size),
            self.intermediate_size,
            self.hidden_size,
        );
        Ok(down)
    }
}

fn build_expert_storage(
    weights: &ModelWeights,
    prefix: &str,
    num_local_experts: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> ExpertStorage {
    let gate_up_name = format!("{prefix}.mlp.experts.gate_up_proj");
    let down_name = format!("{prefix}.mlp.experts.down_proj");

    if let (Ok(gate_up_proj), Ok(gate_up_proj_bias), Ok(down_proj), Ok(down_proj_bias)) = (
        weights.get_as_buffer(&gate_up_name),
        weights.get_as_buffer(&format!("{prefix}.mlp.experts.gate_up_proj_bias")),
        weights.get_as_buffer(&down_name),
        weights.get_as_buffer(&format!("{prefix}.mlp.experts.down_proj_bias")),
    ) {
        return ExpertStorage::Unquantized(GptOssExpertWeights {
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            hidden_size,
            intermediate_size,
        });
    }

    if weights
        .get(&format!("{prefix}.mlp.experts.gate_up_proj_blocks"))
        .is_ok()
        || weights
            .get(&format!("{prefix}.mlp.experts.down_proj_blocks"))
            .is_ok()
    {
        return ExpertStorage::QuantizedMxfp4;
    }

    let _ = num_local_experts;
    ExpertStorage::Missing
}

fn load_optional_bias(
    weights: &ModelWeights,
    name: &str,
    size: usize,
    enabled: bool,
) -> Option<GpuBuffer<f16>> {
    if enabled {
        Some(
            weights
                .get_as_buffer(name)
                .unwrap_or_else(|_| GpuBuffer::zeros(&[size])),
        )
    } else {
        None
    }
}

fn has_nonzero_sinks(sinks: &GpuBuffer<f16>) -> bool {
    sinks
        .data
        .iter()
        .any(|value| value.to_f32().abs() > NONZERO_SINK_EPSILON)
}

fn expert_tensor_slice<'a>(
    tensor: &'a GpuBuffer<f16>,
    expert_idx: usize,
    rows: usize,
    cols: usize,
) -> &'a [f16] {
    let per_expert = rows * cols;
    let offset = expert_idx * per_expert;
    &tensor.data[offset..offset + per_expert]
}

fn expert_vector_slice<'a>(tensor: &'a GpuBuffer<f16>, expert_idx: usize, len: usize) -> &'a [f16] {
    let offset = expert_idx * len;
    &tensor.data[offset..offset + len]
}

fn project_in_out_with_bias(
    input: &GpuBuffer<f16>,
    weight: &[f16],
    bias: &[f16],
    in_features: usize,
    out_features: usize,
) -> GpuBuffer<f16> {
    let num_tokens = input.shape.first().copied().unwrap_or(0);
    let mut out = vec![f16::ZERO; num_tokens * out_features];

    for token_idx in 0..num_tokens {
        let input_offset = token_idx * in_features;
        for out_idx in 0..out_features {
            let mut acc = bias[out_idx].to_f32();
            for in_idx in 0..in_features {
                let input_val = input.data[input_offset + in_idx].to_f32();
                let weight_val = weight[in_idx * out_features + out_idx].to_f32();
                acc += input_val * weight_val;
            }
            out[token_idx * out_features + out_idx] = f16::from_f32(acc);
        }
    }

    GpuBuffer::from_vec(out, vec![num_tokens, out_features])
}

fn apply_gpt_oss_swiglu(
    gate_up: &GpuBuffer<f16>,
    num_tokens: usize,
    intermediate_size: usize,
) -> GpuBuffer<f16> {
    let mut out = vec![f16::ZERO; num_tokens * intermediate_size];
    for token_idx in 0..num_tokens {
        let src_offset = token_idx * intermediate_size * 2;
        let dst_offset = token_idx * intermediate_size;
        for intermediate_idx in 0..intermediate_size {
            let gate = gate_up.data[src_offset + 2 * intermediate_idx]
                .to_f32()
                .min(GPT_OSS_SWIGLU_LIMIT);
            let up = gate_up.data[src_offset + 2 * intermediate_idx + 1]
                .to_f32()
                .clamp(-GPT_OSS_SWIGLU_LIMIT, GPT_OSS_SWIGLU_LIMIT);
            let glu = gate * sigmoid(gate * GPT_OSS_SWIGLU_ALPHA);
            out[dst_offset + intermediate_idx] = f16::from_f32((up + 1.0) * glu);
        }
    }
    GpuBuffer::from_vec(out, vec![num_tokens, intermediate_size])
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn top_k_indices(vals: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = vals.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

fn softmax(vals: &[f32]) -> Vec<f32> {
    if vals.is_empty() {
        return Vec::new();
    }
    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = vals.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::bridge::{AttentionMetadata, MockAttentionBackend, WeightTensor};

    fn tensor(name: &str, vals: &[f32], shape: &[usize]) -> (String, WeightTensor) {
        (
            name.to_string(),
            WeightTensor {
                name: name.to_string(),
                data: vals.iter().map(|&v| f16::from_f32(v)).collect(),
                shape: shape.to_vec(),
            },
        )
    }

    fn test_config(layer_type: &str) -> ModelRunnerConfig {
        ModelRunnerConfig {
            num_layers: 1,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            intermediate_size: 2,
            vocab_size: 8,
            max_position: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
            attn_logit_softcapping: 0.0,
            attention_bias: false,
            sliding_window: Some(128),
            layer_types: vec![layer_type.to_string()],
            num_local_experts: 1,
            num_experts_per_tok: 1,
            dtype: rvllm_core::types::Dtype::Float16,
            architecture: "GptOssForCausalLM".into(),
        }
    }

    fn test_input() -> ModelInput {
        ModelInput {
            token_ids: vec![1, 2],
            position_ids: vec![0, 1],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0, 1],
                context_lens: vec![2],
                block_tables: vec![vec![0]],
                query_lens: vec![1],
                max_context_len: 2,
            },
            is_prefill: true,
        }
    }

    fn test_weights(include_experts: bool) -> ModelWeights {
        let mut tensors = HashMap::new();

        tensors.extend([
            tensor("model.embed_tokens.weight", &[0.0; 32], &[8, 4]),
            tensor("model.layers.0.input_layernorm.weight", &[1.0; 4], &[4]),
            tensor("model.layers.0.post_attention_layernorm.weight", &[1.0; 4], &[4]),
            tensor("model.layers.0.self_attn.q_proj.weight", &[0.0; 16], &[4, 4]),
            tensor("model.layers.0.self_attn.k_proj.weight", &[0.0; 16], &[4, 4]),
            tensor("model.layers.0.self_attn.v_proj.weight", &[0.0; 16], &[4, 4]),
            tensor("model.layers.0.self_attn.o_proj.weight", &[0.0; 16], &[4, 4]),
            tensor("model.layers.0.self_attn.sinks", &[0.0; 2], &[2]),
            tensor("model.layers.0.mlp.router.weight", &[0.0; 4], &[1, 4]),
            tensor("model.layers.0.mlp.router.bias", &[0.0], &[1]),
            tensor("model.norm.weight", &[1.0; 4], &[4]),
            tensor("lm_head.weight", &[0.0; 32], &[8, 4]),
        ]);

        if include_experts {
            tensors.extend([
                tensor(
                    "model.layers.0.mlp.experts.gate_up_proj",
                    &[0.0; 16],
                    &[1, 4, 4],
                ),
                tensor(
                    "model.layers.0.mlp.experts.gate_up_proj_bias",
                    &[0.0; 4],
                    &[1, 4],
                ),
                tensor(
                    "model.layers.0.mlp.experts.down_proj",
                    &[0.0; 8],
                    &[1, 2, 4],
                ),
                tensor(
                    "model.layers.0.mlp.experts.down_proj_bias",
                    &[0.0; 4],
                    &[1, 4],
                ),
            ]);
        }

        ModelWeights { tensors }
    }

    #[test]
    fn gpt_oss_forward_smoke() {
        let model = GptOssForCausalLM::new(test_weights(true), &test_config("full_attention"))
            .unwrap();
        let cache = CacheEngine::new(1, 64);
        let attention = MockAttentionBackend;
        let logits = model.forward(&test_input(), &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![2, 8]);
    }

    #[test]
    fn gpt_oss_sliding_attention_is_rejected() {
        let model =
            GptOssForCausalLM::new(test_weights(true), &test_config("sliding_attention")).unwrap();
        let cache = CacheEngine::new(1, 64);
        let attention = MockAttentionBackend;
        let err = model.forward(&test_input(), &cache, &attention).unwrap_err();
        assert!(err.to_string().contains("sliding attention"));
    }

    #[test]
    fn gpt_oss_missing_expert_weights_is_rejected() {
        let model = GptOssForCausalLM::new(test_weights(false), &test_config("full_attention"))
            .unwrap();
        let cache = CacheEngine::new(1, 64);
        let attention = MockAttentionBackend;
        let err = model.forward(&test_input(), &cache, &attention).unwrap_err();
        assert!(err.to_string().contains("expert tensors"));
    }
}
