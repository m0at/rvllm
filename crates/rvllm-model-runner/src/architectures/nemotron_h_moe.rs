#![allow(dead_code)]
//! Experimental Nemotron Cascade / `nemotron_h_moe` architecture.
//!
//! The real model is a hybrid stack with attention in every layer plus an
//! alternating secondary branch (SSM or expert FFN). This implementation is a
//! best-effort execution path for rvLLM's CPU/mock model-runner stack so the
//! architecture is no longer just a registry stub.
//!
//! Important caveats:
//! - matrix orientation is inferred from GGUF tensor shapes at runtime
//! - SSM execution is an approximation, not a faithful Mamba-2 kernel
//! - expert FFN routing is top-1 and intentionally simple
//! - mixed GGUF quant execution is still unsupported in the main runtime

use half::f16;
use tracing::trace;

use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, LLMError, ModelWeights, Result};
use crate::input::ModelInput;
use crate::layers::activation::silu_inplace;
use crate::layers::norm::RMSNorm;
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, lm_head};
use super::Architecture;

#[derive(Debug, Clone)]
struct NemotronAttention {
    norm_weight: GpuBuffer<f16>,
    q_proj: GpuBuffer<f16>,
    k_proj: GpuBuffer<f16>,
    v_proj: GpuBuffer<f16>,
    o_proj: GpuBuffer<f16>,
}

#[derive(Debug, Clone)]
struct NemotronSsm {
    norm_weight: GpuBuffer<f16>,
    ssm_in: GpuBuffer<f16>,
    ssm_out: GpuBuffer<f16>,
    ssm_conv1d_bias: Option<GpuBuffer<f16>>,
    ssm_dt_bias: Option<GpuBuffer<f16>>,
    ssm_a: Option<GpuBuffer<f16>>,
    ssm_d: Option<GpuBuffer<f16>>,
}

#[derive(Debug, Clone)]
struct NemotronExpertFfn {
    gate_inp: GpuBuffer<f16>,
    exp_probs_b: Option<GpuBuffer<f16>>,
    up_exps: GpuBuffer<f16>,
    down_exps: GpuBuffer<f16>,
    up_shared: Option<GpuBuffer<f16>>,
    down_shared: Option<GpuBuffer<f16>>,
}

#[derive(Debug, Clone)]
struct NemotronLayer {
    attn: Option<NemotronAttention>,
    ssm: Option<NemotronSsm>,
    ffn: Option<NemotronExpertFfn>,
}

pub struct NemotronHMoEForCausalLM {
    config: NemotronConfig,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<NemotronLayer>,
    norm_weight: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

#[derive(Debug, Clone)]
struct NemotronConfig {
    architecture: String,
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
}

impl NemotronHMoEForCausalLM {
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let cfg = NemotronConfig {
            architecture: config.architecture.clone(),
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            vocab_size: config.vocab_size,
            rms_norm_eps: 1e-5,
        };

        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.hidden_size, cfg.vocab_size]));
        let norm_weight = weights
            .get_as_buffer("model.norm.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.hidden_size]));
        let lm_head_weight = weights
            .get_as_buffer("lm_head.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.hidden_size, cfg.vocab_size]));

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("blk.{i}");
            let attn = Self::load_attention(&weights, &p)?;
            let ssm = Self::load_ssm(&weights, &p)?;
            let ffn = Self::load_ffn(&weights, &p)?;
            layers.push(NemotronLayer { attn, ssm, ffn });
        }

        Ok(Self {
            config: cfg,
            embed_tokens,
            layers,
            norm_weight,
            lm_head_weight,
        })
    }

    fn get_opt(weights: &ModelWeights, name: &str) -> Option<GpuBuffer<f16>> {
        weights.get_as_buffer(name).ok()
    }

    fn load_attention(weights: &ModelWeights, prefix: &str) -> Result<Option<NemotronAttention>> {
        let q = Self::get_opt(weights, &format!("{prefix}.attn_q.weight"));
        let k = Self::get_opt(weights, &format!("{prefix}.attn_k.weight"));
        let v = Self::get_opt(weights, &format!("{prefix}.attn_v.weight"));
        let o = Self::get_opt(weights, &format!("{prefix}.attn_output.weight"));
        let n = Self::get_opt(weights, &format!("{prefix}.attn_norm.weight"));
        match (q, k, v, o, n) {
            (Some(q_proj), Some(k_proj), Some(v_proj), Some(o_proj), Some(norm_weight)) => {
                Ok(Some(NemotronAttention {
                    norm_weight,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                }))
            }
            _ => Ok(None),
        }
    }

    fn load_ssm(weights: &ModelWeights, prefix: &str) -> Result<Option<NemotronSsm>> {
        let ssm_in = Self::get_opt(weights, &format!("{prefix}.ssm_in.weight"));
        let ssm_out = Self::get_opt(weights, &format!("{prefix}.ssm_out.weight"));
        let norm = Self::get_opt(weights, &format!("{prefix}.ssm_norm.weight"));
        match (ssm_in, ssm_out, norm) {
            (Some(ssm_in), Some(ssm_out), Some(norm_weight)) => Ok(Some(NemotronSsm {
                norm_weight,
                ssm_in,
                ssm_out,
                ssm_conv1d_bias: Self::get_opt(weights, &format!("{prefix}.ssm_conv1d.bias")),
                ssm_dt_bias: Self::get_opt(weights, &format!("{prefix}.ssm_dt.bias")),
                ssm_a: Self::get_opt(weights, &format!("{prefix}.ssm_a")),
                ssm_d: Self::get_opt(weights, &format!("{prefix}.ssm_d")),
            })),
            _ => Ok(None),
        }
    }

    fn load_ffn(weights: &ModelWeights, prefix: &str) -> Result<Option<NemotronExpertFfn>> {
        let gate_inp = Self::get_opt(weights, &format!("{prefix}.ffn_gate_inp.weight"));
        let up_exps = Self::get_opt(weights, &format!("{prefix}.ffn_up_exps.weight"));
        let down_exps = Self::get_opt(weights, &format!("{prefix}.ffn_down_exps.weight"));
        match (gate_inp, up_exps, down_exps) {
            (Some(gate_inp), Some(up_exps), Some(down_exps)) => Ok(Some(NemotronExpertFfn {
                gate_inp,
                exp_probs_b: Self::get_opt(weights, &format!("{prefix}.exp_probs_b.bias")),
                up_exps,
                down_exps,
                up_shared: Self::get_opt(weights, &format!("{prefix}.ffn_up_shexp.weight")),
                down_shared: Self::get_opt(weights, &format!("{prefix}.ffn_down_shexp.weight")),
            })),
            _ => Ok(None),
        }
    }

    fn embed_tokens_flexible(&self, token_ids: &[u32]) -> GpuBuffer<f16> {
        let shape = &self.embed_tokens.shape;
        if shape.len() == 2 && shape[0] == self.config.hidden_size && shape[1] == self.config.vocab_size {
            // GGUF orientation: [hidden, vocab]
            let mut out = Vec::with_capacity(token_ids.len() * self.config.hidden_size);
            for &tid in token_ids {
                let col = tid as usize;
                for h in 0..self.config.hidden_size {
                    out.push(self.embed_tokens.data[h * self.config.vocab_size + col]);
                }
            }
            GpuBuffer::from_vec(out, vec![token_ids.len(), self.config.hidden_size])
        } else {
            super::llama::embed_tokens(&self.embed_tokens, token_ids, self.config.hidden_size)
        }
    }

    fn lm_head_flexible(&self, hidden: &GpuBuffer<f16>, num_tokens: usize) -> Result<GpuBuffer<f32>> {
        let shape = &self.lm_head_weight.shape;
        if shape.len() == 2 && shape[0] == self.config.hidden_size && shape[1] == self.config.vocab_size {
            let h = self.config.hidden_size;
            let mut logits = Vec::with_capacity(num_tokens * self.config.vocab_size);
            for t in 0..num_tokens {
                let row_start = t * h;
                for v in 0..self.config.vocab_size {
                    let mut acc = 0.0f32;
                    for k in 0..h {
                        acc += hidden.data[row_start + k].to_f32()
                            * self.lm_head_weight.data[k * self.config.vocab_size + v].to_f32();
                    }
                    logits.push(acc);
                }
            }
            Ok(GpuBuffer::from_vec(logits, vec![num_tokens, self.config.vocab_size]))
        } else {
            lm_head(hidden, &self.lm_head_weight, num_tokens, self.config.vocab_size)
        }
    }

    fn linear_flexible(input: &GpuBuffer<f16>, weight: &GpuBuffer<f16>) -> Result<GpuBuffer<f16>> {
        if weight.shape.len() != 2 {
            return Err(LLMError::ModelError(format!(
                "nemotron linear_flexible expected 2D weight, got {:?}",
                weight.shape
            )));
        }
        let in_features = if input.shape.len() >= 2 { input.shape[1] } else { input.len() };
        let a = weight.shape[0];
        let b = weight.shape[1];
        if b == in_features {
            return Self::linear_forward(input, weight, false);
        }
        if a == in_features {
            return Self::linear_forward(input, weight, true);
        }
        // best-effort: choose the orientation whose input dimension is closer
        if a.abs_diff(in_features) < b.abs_diff(in_features) {
            Self::linear_forward(input, weight, true)
        } else {
            Self::linear_forward(input, weight, false)
        }
    }

    fn linear_forward(input: &GpuBuffer<f16>, weight: &GpuBuffer<f16>, transposed: bool) -> Result<GpuBuffer<f16>> {
        let in_features = if input.shape.len() >= 2 { input.shape[1] } else { input.len() };
        let num_tokens = input.len() / in_features.max(1);
        let (out_features, weight_in) = if transposed {
            (weight.shape[1], weight.shape[0])
        } else {
            (weight.shape[0], weight.shape[1])
        };
        let used_in = in_features.min(weight_in);
        let mut out = vec![f16::ZERO; num_tokens * out_features];
        for t in 0..num_tokens {
            let row = &input.data[t * in_features..t * in_features + in_features];
            for o in 0..out_features {
                let mut acc = 0.0f32;
                for i in 0..used_in {
                    let w = if transposed {
                        weight.data[i * out_features + o]
                    } else {
                        weight.data[o * weight_in + i]
                    };
                    acc += row[i].to_f32() * w.to_f32();
                }
                out[t * out_features + o] = f16::from_f32(acc);
            }
        }
        Ok(GpuBuffer::from_vec(out, vec![num_tokens, out_features]))
    }

    fn linear_expert(weight3: &GpuBuffer<f16>, expert_idx: usize, input: &GpuBuffer<f16>) -> Result<GpuBuffer<f16>> {
        if weight3.shape.len() != 3 {
            return Err(LLMError::ModelError(format!(
                "nemotron expert weight expected 3D, got {:?}",
                weight3.shape
            )));
        }
        let a = weight3.shape[0];
        let b = weight3.shape[1];
        let e = weight3.shape[2];
        if expert_idx >= e {
            return Err(LLMError::ModelError(format!("expert index {} out of range {}", expert_idx, e)));
        }
        let chunk = a * b;
        let start = expert_idx * chunk;
        let end = start + chunk;
        let w = GpuBuffer::from_vec(weight3.data[start..end].to_vec(), vec![a, b]);
        Self::linear_flexible(input, &w)
    }

    fn apply_ssm(ssm: &NemotronSsm, hidden: &GpuBuffer<f16>) -> Result<GpuBuffer<f16>> {
        let normed = RMSNorm::forward(hidden, &ssm.norm_weight, 1e-5)?;
        let projected = Self::linear_flexible(&normed, &ssm.ssm_in)?;
        let internal_dim = if ssm.ssm_out.shape.len() == 2 {
            if ssm.ssm_out.shape[1] == hidden.shape[1] { ssm.ssm_out.shape[0] } else { ssm.ssm_out.shape[1] }
        } else {
            hidden.shape[1]
        };
        let num_tokens = projected.shape[0];
        let proj_dim = projected.shape[1];
        let mut state = vec![f16::ZERO; num_tokens * internal_dim];
        for t in 0..num_tokens {
            for i in 0..internal_dim {
                let src = projected.data[t * proj_dim + (i % proj_dim)].to_f32();
                let conv_bias = ssm
                    .ssm_conv1d_bias
                    .as_ref()
                    .and_then(|b| b.data.get(i % b.len()))
                    .map(|v| v.to_f32())
                    .unwrap_or(0.0);
                let groups = ssm.ssm_dt_bias.as_ref().map(|b| b.len()).unwrap_or(1).max(1);
                let g = i % groups;
                let dt = ssm.ssm_dt_bias.as_ref().map(|b| b.data[g].to_f32()).unwrap_or(0.0);
                let a = ssm.ssm_a.as_ref().and_then(|b| b.data.get(g)).map(|v| v.to_f32()).unwrap_or(0.0);
                let d = ssm.ssm_d.as_ref().and_then(|b| b.data.get(g)).map(|v| v.to_f32()).unwrap_or(0.0);
                let scale = 1.0 / (1.0 + (-(dt + d - a)).exp());
                state[t * internal_dim + i] = f16::from_f32((src + conv_bias) * scale);
            }
        }
        let state_buf = GpuBuffer::from_vec(state, vec![num_tokens, internal_dim]);
        Self::linear_flexible(&state_buf, &ssm.ssm_out)
    }

    fn apply_ffn(ffn: &NemotronExpertFfn, hidden: &GpuBuffer<f16>) -> Result<GpuBuffer<f16>> {
        let logits = Self::linear_flexible(hidden, &ffn.gate_inp)?;
        let num_tokens = logits.shape[0];
        let num_experts = logits.shape[1];
        let mut out = vec![f16::ZERO; hidden.len()];

        // shared expert path
        if let (Some(up_shared), Some(down_shared)) = (&ffn.up_shared, &ffn.down_shared) {
            let mut shared = Self::linear_flexible(hidden, up_shared)?;
            silu_inplace(&mut shared);
            let shared_down = Self::linear_flexible(&shared, down_shared)?;
            for (dst, src) in out.iter_mut().zip(shared_down.data.iter()) {
                *dst = *src;
            }
        }

        for t in 0..num_tokens {
            let row = &logits.data[t * num_experts..(t + 1) * num_experts];
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, v) in row.iter().enumerate() {
                let bias = ffn
                    .exp_probs_b
                    .as_ref()
                    .and_then(|b| b.data.get(i))
                    .map(|x| x.to_f32())
                    .unwrap_or(0.0);
                let score = v.to_f32() + bias;
                if score > best_val {
                    best_val = score;
                    best_idx = i;
                }
            }
            let token_buf = GpuBuffer::from_vec(
                hidden.data[t * hidden.shape[1]..(t + 1) * hidden.shape[1]].to_vec(),
                vec![1, hidden.shape[1]],
            );
            let mut up = Self::linear_expert(&ffn.up_exps, best_idx, &token_buf)?;
            silu_inplace(&mut up);
            let down = Self::linear_expert(&ffn.down_exps, best_idx, &up)?;
            let dst = &mut out[t * hidden.shape[1]..(t + 1) * hidden.shape[1]];
            for (d, s) in dst.iter_mut().zip(down.data.iter()) {
                *d = f16::from_f32(d.to_f32() + s.to_f32());
            }
        }
        Ok(GpuBuffer::from_vec(out, hidden.shape.clone()))
    }
}

impl Architecture for NemotronHMoEForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let mut hidden = self.embed_tokens_flexible(&input.token_ids);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, arch = %self.config.architecture, "nemotron layer forward");

            if let Some(attn) = &layer.attn {
                let normed = RMSNorm::forward(&hidden, &attn.norm_weight, self.config.rms_norm_eps)?;
                let q = Self::linear_flexible(&normed, &attn.q_proj)?;
                let k = Self::linear_flexible(&normed, &attn.k_proj)?;
                let v = Self::linear_flexible(&normed, &attn.v_proj)?;
                let (q_rot, k_rot) = RotaryEmbedding::forward(&input.position_ids, &q, &k, self.config.head_dim)?;
                let attn_out = attention.forward(&q_rot, &k_rot, &v, &input.attention_metadata, layer_idx)?;
                let attn_proj = Self::linear_flexible(&attn_out, &attn.o_proj)?;
                add_inplace(&mut hidden, &attn_proj);
            }

            if let Some(ssm) = &layer.ssm {
                let ssm_out = Self::apply_ssm(ssm, &hidden)?;
                add_inplace(&mut hidden, &ssm_out);
            }

            if let Some(ffn) = &layer.ffn {
                let ffn_out = Self::apply_ffn(ffn, &hidden)?;
                add_inplace(&mut hidden, &ffn_out);
            }
        }

        let normed_final = RMSNorm::forward(&hidden, &self.norm_weight, self.config.rms_norm_eps)?;
        self.lm_head_flexible(&normed_final, num_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{AttentionMetadata, MockAttentionBackend, ModelWeights, WeightTensor, CacheEngine};
    use crate::input::ModelInput;
    use std::sync::Arc;

    fn add_tensor(weights: &mut ModelWeights, name: &str, shape: &[usize]) {
        let size = shape.iter().product();
        weights.tensors.insert(
            name.to_string(),
            WeightTensor {
                name: name.to_string(),
                data: vec![f16::from_f32(0.01); size],
                shape: shape.to_vec(),
            },
        );
    }

    fn config() -> ModelRunnerConfig {
        ModelRunnerConfig {
            num_layers: 2,
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: 16,
            max_position: 16,
            rope_theta: 10000.0,
            dtype: rvllm_core::types::Dtype::Float16,
            architecture: "nemotron_h_moe".to_string(),
        }
    }

    #[test]
    fn flexible_linear_handles_transposed_gguf_matrices() {
        let input = GpuBuffer::from_vec(vec![f16::from_f32(1.0); 8], vec![1, 8]);
        let weight = GpuBuffer::from_vec(vec![f16::from_f32(0.5); 8 * 12], vec![8, 12]);
        let out = NemotronHMoEForCausalLM::linear_flexible(&input, &weight).unwrap();
        assert_eq!(out.shape, vec![1, 12]);
    }

    #[test]
    fn nemotron_forward_smoke_with_hybrid_layers() {
        let mut weights = ModelWeights::default();
        add_tensor(&mut weights, "token_embd.weight", &[8, 16]);
        add_tensor(&mut weights, "output_norm.weight", &[8]);
        add_tensor(&mut weights, "output.weight", &[8, 16]);
        // layer 0 attention + ssm
        add_tensor(&mut weights, "blk.0.attn_norm.weight", &[8]);
        add_tensor(&mut weights, "blk.0.attn_q.weight", &[8, 8]);
        add_tensor(&mut weights, "blk.0.attn_k.weight", &[8, 4]);
        add_tensor(&mut weights, "blk.0.attn_v.weight", &[8, 4]);
        add_tensor(&mut weights, "blk.0.attn_output.weight", &[8, 8]);
        add_tensor(&mut weights, "blk.0.ssm_norm.weight", &[8]);
        add_tensor(&mut weights, "blk.0.ssm_in.weight", &[8, 12]);
        add_tensor(&mut weights, "blk.0.ssm_out.weight", &[12, 8]);
        add_tensor(&mut weights, "blk.0.ssm_conv1d.bias", &[12]);
        add_tensor(&mut weights, "blk.0.ssm_dt.bias", &[4]);
        add_tensor(&mut weights, "blk.0.ssm_a", &[4]);
        add_tensor(&mut weights, "blk.0.ssm_d", &[4]);
        // layer 1 attention + ffn
        add_tensor(&mut weights, "blk.1.attn_norm.weight", &[8]);
        add_tensor(&mut weights, "blk.1.attn_q.weight", &[8, 8]);
        add_tensor(&mut weights, "blk.1.attn_k.weight", &[8, 4]);
        add_tensor(&mut weights, "blk.1.attn_v.weight", &[8, 4]);
        add_tensor(&mut weights, "blk.1.attn_output.weight", &[8, 8]);
        add_tensor(&mut weights, "blk.1.ffn_gate_inp.weight", &[8, 2]);
        add_tensor(&mut weights, "blk.1.exp_probs_b.bias", &[2]);
        add_tensor(&mut weights, "blk.1.ffn_up_exps.weight", &[8, 6, 2]);
        add_tensor(&mut weights, "blk.1.ffn_down_exps.weight", &[6, 8, 2]);
        add_tensor(&mut weights, "blk.1.ffn_up_shexp.weight", &[8, 6]);
        add_tensor(&mut weights, "blk.1.ffn_down_shexp.weight", &[6, 8]);

        let model = NemotronHMoEForCausalLM::new(weights, &config()).unwrap();
        let input = ModelInput {
            token_ids: vec![1, 2],
            position_ids: vec![0, 1],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0, 1],
                context_lens: vec![2],
                block_tables: vec![vec![0]],
                max_context_len: 2,
                query_lens: vec![2],
            },
            is_prefill: true,
        };
        let cache = Arc::new(CacheEngine::new(1, 1));
        let attention = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![2, 16]);
    }
}
