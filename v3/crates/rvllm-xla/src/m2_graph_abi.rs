use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_loader::{M2CheckpointIndex, M2Projection};

use crate::ffi::PjrtElementType;

pub const M2_VOCAB: usize = 200_064;
pub const M2_HIDDEN: usize = 3_072;
pub const M2_NUM_LAYERS: usize = 62;
pub const M2_NUM_EXPERTS: usize = 256;
pub const M2_TOP_K: usize = 8;
pub const M2_NUM_Q_HEADS: usize = 48;
pub const M2_NUM_KV_HEADS: usize = 8;
pub const M2_HEAD_DIM: usize = 128;
pub const M2_ROTARY_DIM: usize = 64;
pub const M2_MOE_INTER: usize = 1_536;
pub const M2_NVFP4_GROUP: usize = 16;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum M2GraphPhase {
    Decode,
    Prefill,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum M2KvCacheDType {
    Int8,
    Bf16,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum M2DecodeLayerCustomCall {
    FusedAttentionNvfp4Moe,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2GraphShape {
    pub phase: M2GraphPhase,
    pub batch: usize,
    pub prompt_len: usize,
    pub ctx: usize,
    pub kv_bytes_per_elem: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2GraphTensorSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: PjrtElementType,
    pub nbytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2Nvfp4ProjectionAbi {
    pub base: String,
    pub projection: M2Projection,
    pub rows: usize,
    pub cols: usize,
    pub weight: M2GraphTensorSpec,
    pub weight_scale: M2GraphTensorSpec,
    pub weight_scale_2: M2GraphTensorSpec,
    pub input_scale: M2GraphTensorSpec,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct M2DecodeLayerArenaOffsets {
    pub input_norm: i64,
    pub post_attention_norm: i64,
    pub q_proj: i64,
    pub k_proj: i64,
    pub v_proj: i64,
    pub o_proj: i64,
    pub q_norm: i64,
    pub k_norm: i64,
    pub router: i64,
    pub router_bias: i64,
    pub w1_first_packed: i64,
    pub w1_first_scale: i64,
    pub w1_first_global_scale: i64,
    pub w1_first_input_scale: i64,
    pub w2_first_packed: i64,
    pub w3_first_packed: i64,
    pub w3_last_packed: i64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct M2DecodeLayerCustomCallAbi {
    pub call: M2DecodeLayerCustomCall,
    pub batch: usize,
    pub ctx: usize,
    pub kv_dtype: M2KvCacheDType,
    pub kv_cache_bytes: usize,
    pub hidden: usize,
    pub top_k: usize,
    pub expert_count: usize,
    pub expert_directory_cols: usize,
    pub weight_offsets: M2DecodeLayerArenaOffsets,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2LayerWeightAbi {
    pub layer: usize,
    pub dense: Vec<M2GraphTensorSpec>,
    pub experts_per_layer: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2GraphAbi {
    pub shape: M2GraphShape,
    pub runtime_inputs: Vec<M2GraphTensorSpec>,
    pub runtime_outputs: Vec<M2GraphTensorSpec>,
    pub global_weights: Vec<M2GraphTensorSpec>,
    pub layer_weights: Vec<M2LayerWeightAbi>,
}

impl M2KvCacheDType {
    pub fn from_bytes_per_elem(bytes: usize) -> Result<Self> {
        match bytes {
            1 => Ok(Self::Int8),
            2 => Ok(Self::Bf16),
            _ => Err(invalid("kv_bytes_per_elem", "must be 1 or 2")),
        }
    }

    pub const fn as_mlir_dtype(self) -> &'static str {
        match self {
            Self::Int8 => "i8",
            Self::Bf16 => "bf16",
        }
    }

    pub const fn bytes_per_elem(self) -> usize {
        match self {
            Self::Int8 => 1,
            Self::Bf16 => 2,
        }
    }
}

impl M2DecodeLayerCustomCall {
    pub const fn target(self) -> &'static str {
        match self {
            Self::FusedAttentionNvfp4Moe => "rvllm.m2.decode_layer.fused_attention_nvfp4_moe",
        }
    }

    pub const fn abi(self) -> &'static str {
        match self {
            Self::FusedAttentionNvfp4Moe => "m2_decode_layer_v1",
        }
    }
}

impl M2DecodeLayerArenaOffsets {
    pub const DENSE_ATTRS: &'static str =
        "input_norm,post_attention_norm,q_proj,k_proj,v_proj,o_proj,q_norm,k_norm,router,router_bias";
}

impl M2DecodeLayerCustomCallAbi {
    pub fn new(
        shape: &M2GraphShape,
        expert_count: usize,
        expert_directory_cols: usize,
        weight_offsets: M2DecodeLayerArenaOffsets,
    ) -> Result<Self> {
        shape.validate()?;
        if shape.phase != M2GraphPhase::Decode {
            return Err(invalid("phase", "expected decode graph shape"));
        }
        Ok(Self {
            call: M2DecodeLayerCustomCall::FusedAttentionNvfp4Moe,
            batch: shape.batch,
            ctx: shape.ctx,
            kv_dtype: shape.kv_dtype()?,
            kv_cache_bytes: shape.kv_cache_bytes(),
            hidden: M2_HIDDEN,
            top_k: M2_TOP_K,
            expert_count,
            expert_directory_cols,
            weight_offsets,
        })
    }
}

impl M2GraphShape {
    pub fn decode(batch: usize, ctx: usize, kv_bytes_per_elem: usize) -> Self {
        Self {
            phase: M2GraphPhase::Decode,
            batch,
            prompt_len: 1,
            ctx,
            kv_bytes_per_elem,
        }
    }

    pub fn prefill(batch: usize, prompt_len: usize, ctx: usize, kv_bytes_per_elem: usize) -> Self {
        Self {
            phase: M2GraphPhase::Prefill,
            batch,
            prompt_len,
            ctx,
            kv_bytes_per_elem,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.batch == 0 {
            return Err(invalid("batch", "must be > 0"));
        }
        if self.ctx == 0 {
            return Err(invalid("ctx", "must be > 0"));
        }
        if self.prompt_len == 0 {
            return Err(invalid("prompt_len", "must be > 0"));
        }
        if self.prompt_len > self.ctx {
            return Err(invalid("prompt_len", "must be <= ctx"));
        }
        if !matches!(self.kv_bytes_per_elem, 1 | 2) {
            return Err(invalid("kv_bytes_per_elem", "must be 1 or 2"));
        }
        if self.phase == M2GraphPhase::Decode && self.prompt_len != 1 {
            return Err(invalid("prompt_len", "decode prompt_len must be 1"));
        }
        Ok(())
    }

    pub fn kv_dtype(&self) -> Result<M2KvCacheDType> {
        M2KvCacheDType::from_bytes_per_elem(self.kv_bytes_per_elem)
    }

    pub const fn total_tokens(&self) -> usize {
        self.batch * self.prompt_len
    }

    pub const fn kv_cache_bytes(&self) -> usize {
        2 * M2_NUM_LAYERS
            * self.batch
            * self.ctx
            * M2_NUM_KV_HEADS
            * M2_HEAD_DIM
            * self.kv_bytes_per_elem
    }

    pub fn runtime_inputs(&self) -> Result<Vec<M2GraphTensorSpec>> {
        self.validate()?;
        let mut inputs = match self.phase {
            M2GraphPhase::Decode => vec![
                tensor("token_ids", &[self.batch], PjrtElementType::S32),
                tensor("positions", &[self.batch], PjrtElementType::S32),
            ],
            M2GraphPhase::Prefill => vec![
                tensor(
                    "token_ids",
                    &[self.batch, self.prompt_len],
                    PjrtElementType::S32,
                ),
                tensor("positions", &[self.total_tokens()], PjrtElementType::S32),
                tensor("slot_mapping", &[self.total_tokens()], PjrtElementType::S32),
                tensor("cu_seqlens_q", &[self.batch + 1], PjrtElementType::S32),
                tensor("context_lens", &[self.batch], PjrtElementType::S32),
            ],
        };
        inputs.push(tensor(
            "kv_cache",
            &[self.kv_cache_bytes()],
            PjrtElementType::S8,
        ));
        Ok(inputs)
    }

    pub fn runtime_outputs(&self) -> Result<Vec<M2GraphTensorSpec>> {
        self.validate()?;
        Ok(match self.phase {
            M2GraphPhase::Decode => vec![
                tensor("logits", &[self.batch, M2_VOCAB], PjrtElementType::BF16),
                tensor("next_token", &[self.batch], PjrtElementType::S32),
                tensor("kv_cache", &[self.kv_cache_bytes()], PjrtElementType::S8),
            ],
            M2GraphPhase::Prefill => vec![
                tensor(
                    "last_hidden",
                    &[self.batch, M2_HIDDEN],
                    PjrtElementType::BF16,
                ),
                tensor("kv_cache", &[self.kv_cache_bytes()], PjrtElementType::S8),
            ],
        })
    }
}

impl M2GraphAbi {
    pub fn new(shape: M2GraphShape) -> Result<Self> {
        shape.validate()?;
        let runtime_inputs = shape.runtime_inputs()?;
        let runtime_outputs = shape.runtime_outputs()?;
        Ok(Self {
            shape,
            runtime_inputs,
            runtime_outputs,
            global_weights: global_weight_specs(),
            layer_weights: (0..M2_NUM_LAYERS).map(layer_weight_abi).collect(),
        })
    }

    pub fn validate_checkpoint_index(&self, index: &M2CheckpointIndex) -> Result<()> {
        let summary = index.validate_m2(M2_NUM_LAYERS, M2_NUM_EXPERTS)?;
        if summary.total_tensors != 191_069 {
            return Err(invalid("checkpoint", "unexpected M2 tensor count"));
        }
        for spec in &self.global_weights {
            require(index, &spec.name)?;
        }
        for layer in &self.layer_weights {
            for spec in &layer.dense {
                require(index, &spec.name)?;
            }
            for expert in 0..layer.experts_per_layer {
                for projection in [M2Projection::W1, M2Projection::W2, M2Projection::W3] {
                    let group = M2Nvfp4ProjectionAbi::new(layer.layer, expert, projection);
                    require(index, &group.weight.name)?;
                    require(index, &group.weight_scale.name)?;
                    require(index, &group.weight_scale_2.name)?;
                    if index.contains(&group.input_scale.name) {
                        continue;
                    }
                    if summary.input_scales_missing == 0 {
                        return Err(invalid(
                            "input_scale",
                            "missing optional scale unexpectedly",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

impl M2Nvfp4ProjectionAbi {
    pub fn new(layer: usize, expert: usize, projection: M2Projection) -> Self {
        let (rows, cols) = match projection {
            M2Projection::W1 | M2Projection::W3 => (M2_MOE_INTER, M2_HIDDEN),
            M2Projection::W2 => (M2_HIDDEN, M2_MOE_INTER),
        };
        let base = format!(
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.{}",
            projection.as_str()
        );
        Self {
            projection,
            rows,
            cols,
            weight: tensor(
                format!("{base}.weight"),
                &[rows, cols / 2],
                PjrtElementType::U8,
            ),
            weight_scale: tensor(
                format!("{base}.weight_scale"),
                &[rows, cols / M2_NVFP4_GROUP],
                PjrtElementType::U8,
            ),
            weight_scale_2: tensor(format!("{base}.weight_scale_2"), &[1], PjrtElementType::F32),
            input_scale: tensor(format!("{base}.input_scale"), &[1], PjrtElementType::F32),
            base,
        }
    }
}

fn global_weight_specs() -> Vec<M2GraphTensorSpec> {
    vec![
        tensor(
            "model.embed_tokens.weight",
            &[M2_VOCAB, M2_HIDDEN],
            PjrtElementType::BF16,
        ),
        tensor("model.norm.weight", &[M2_HIDDEN], PjrtElementType::BF16),
        tensor(
            "lm_head.weight",
            &[M2_VOCAB, M2_HIDDEN],
            PjrtElementType::BF16,
        ),
    ]
}

fn layer_weight_abi(layer: usize) -> M2LayerWeightAbi {
    let p = format!("model.layers.{layer}");
    M2LayerWeightAbi {
        layer,
        experts_per_layer: M2_NUM_EXPERTS,
        dense: vec![
            tensor(
                format!("{p}.input_layernorm.weight"),
                &[M2_HIDDEN],
                PjrtElementType::BF16,
            ),
            tensor(
                format!("{p}.post_attention_layernorm.weight"),
                &[M2_HIDDEN],
                PjrtElementType::BF16,
            ),
            tensor(
                format!("{p}.self_attn.q_proj.weight"),
                &[M2_NUM_Q_HEADS * M2_HEAD_DIM, M2_HIDDEN],
                PjrtElementType::BF16,
            ),
            tensor(
                format!("{p}.self_attn.k_proj.weight"),
                &[M2_NUM_KV_HEADS * M2_HEAD_DIM, M2_HIDDEN],
                PjrtElementType::BF16,
            ),
            tensor(
                format!("{p}.self_attn.v_proj.weight"),
                &[M2_NUM_KV_HEADS * M2_HEAD_DIM, M2_HIDDEN],
                PjrtElementType::BF16,
            ),
            tensor(
                format!("{p}.self_attn.o_proj.weight"),
                &[M2_HIDDEN, M2_NUM_Q_HEADS * M2_HEAD_DIM],
                PjrtElementType::BF16,
            ),
            tensor(
                format!("{p}.self_attn.q_norm.weight"),
                &[M2_HEAD_DIM],
                PjrtElementType::BF16,
            ),
            tensor(
                format!("{p}.self_attn.k_norm.weight"),
                &[M2_HEAD_DIM],
                PjrtElementType::BF16,
            ),
            tensor(
                format!("{p}.block_sparse_moe.gate.weight"),
                &[M2_NUM_EXPERTS, M2_HIDDEN],
                PjrtElementType::BF16,
            ),
            tensor(
                format!("{p}.block_sparse_moe.e_score_correction_bias"),
                &[M2_NUM_EXPERTS],
                PjrtElementType::BF16,
            ),
        ],
    }
}

fn tensor(name: impl Into<String>, shape: &[usize], dtype: PjrtElementType) -> M2GraphTensorSpec {
    let nbytes = shape.iter().product::<usize>() * element_size(dtype);
    M2GraphTensorSpec {
        name: name.into(),
        shape: shape.iter().map(|&x| x as i64).collect(),
        dtype,
        nbytes,
    }
}

fn element_size(dtype: PjrtElementType) -> usize {
    match dtype {
        PjrtElementType::PRED | PjrtElementType::S8 | PjrtElementType::U8 => 1,
        PjrtElementType::S16
        | PjrtElementType::U16
        | PjrtElementType::F16
        | PjrtElementType::BF16 => 2,
        PjrtElementType::S32 | PjrtElementType::U32 | PjrtElementType::F32 => 4,
        PjrtElementType::S64 | PjrtElementType::U64 | PjrtElementType::F64 => 8,
        PjrtElementType::C64 => 8,
        PjrtElementType::C128 => 16,
        PjrtElementType::F8E5M2 | PjrtElementType::F8E4M3FN => 1,
        PjrtElementType::INVALID => 0,
    }
}

fn require(index: &M2CheckpointIndex, name: &str) -> Result<()> {
    if index.contains(name) {
        Ok(())
    } else {
        Err(invalid("checkpoint", "ABI tensor missing from checkpoint"))
    }
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_graph_abi",
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn real_index() -> M2CheckpointIndex {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema/model.safetensors.index.json");
        M2CheckpointIndex::from_index_file(path).unwrap()
    }

    #[test]
    fn decode_b8_abi_has_flat_kv_and_logits() {
        let abi = M2GraphAbi::new(M2GraphShape::decode(8, 2048, 1)).unwrap();
        assert_eq!(abi.runtime_inputs[0].name, "token_ids");
        assert_eq!(abi.runtime_inputs[0].shape, vec![8]);
        assert_eq!(abi.runtime_inputs[1].name, "positions");
        assert_eq!(abi.runtime_inputs[2].name, "kv_cache");
        assert_eq!(abi.runtime_inputs[2].dtype, PjrtElementType::S8);
        assert_eq!(abi.runtime_inputs[2].nbytes, 2_080_374_784);
        assert_eq!(abi.runtime_outputs[0].name, "logits");
        assert_eq!(abi.runtime_outputs[0].shape, vec![8, M2_VOCAB as i64]);
        assert_eq!(abi.runtime_outputs[1].name, "next_token");
        assert_eq!(abi.global_weights.len(), 3);
        assert_eq!(abi.layer_weights.len(), M2_NUM_LAYERS);
    }

    #[test]
    fn prefill_b8_t20_abi_has_metadata_and_last_hidden() {
        let abi = M2GraphAbi::new(M2GraphShape::prefill(8, 20, 2048, 1)).unwrap();
        assert_eq!(abi.runtime_inputs[0].name, "token_ids");
        assert_eq!(abi.runtime_inputs[0].shape, vec![8, 20]);
        assert_eq!(abi.runtime_inputs[1].shape, vec![160]);
        assert_eq!(abi.runtime_inputs[3].name, "cu_seqlens_q");
        assert_eq!(abi.runtime_inputs[3].shape, vec![9]);
        assert_eq!(abi.runtime_outputs[0].name, "last_hidden");
        assert_eq!(abi.runtime_outputs[0].shape, vec![8, M2_HIDDEN as i64]);
    }

    #[test]
    fn nvfp4_projection_abi_matches_modelopt_shapes() {
        let w1 = M2Nvfp4ProjectionAbi::new(0, 0, M2Projection::W1);
        assert_eq!(w1.weight.shape, vec![1536, 1536]);
        assert_eq!(w1.weight_scale.shape, vec![1536, 192]);
        assert_eq!(w1.weight_scale_2.shape, vec![1]);
        assert_eq!(w1.input_scale.shape, vec![1]);

        let w2 = M2Nvfp4ProjectionAbi::new(0, 0, M2Projection::W2);
        assert_eq!(w2.weight.shape, vec![3072, 768]);
        assert_eq!(w2.weight_scale.shape, vec![3072, 96]);
    }

    #[test]
    fn abi_tensor_names_are_present_in_real_checkpoint_index() {
        let index = real_index();
        let abi = M2GraphAbi::new(M2GraphShape::decode(8, 2048, 1)).unwrap();
        abi.validate_checkpoint_index(&index).unwrap();
    }

    #[test]
    fn dense_layer_shapes_match_m2_schema() {
        let layer = layer_weight_abi(17);
        assert_eq!(
            layer.dense[2].name,
            "model.layers.17.self_attn.q_proj.weight"
        );
        assert_eq!(layer.dense[2].shape, vec![6144, 3072]);
        assert_eq!(layer.dense[3].shape, vec![1024, 3072]);
        assert_eq!(layer.dense[5].shape, vec![3072, 6144]);
        assert_eq!(layer.dense[8].shape, vec![256, 3072]);
        assert_eq!(
            layer.dense[9].name,
            "model.layers.17.block_sparse_moe.e_score_correction_bias"
        );
    }

    #[test]
    fn decode_layer_custom_call_abi_tracks_runtime_dispatch_shape() {
        let shape = M2GraphShape::decode(4, 1024, 2);
        let offsets = M2DecodeLayerArenaOffsets {
            input_norm: 1,
            post_attention_norm: 2,
            q_proj: 3,
            k_proj: 4,
            v_proj: 5,
            o_proj: 6,
            q_norm: 7,
            k_norm: 8,
            router: 9,
            router_bias: 10,
            w1_first_packed: 11,
            w1_first_scale: 12,
            w1_first_global_scale: 13,
            w1_first_input_scale: -1,
            w2_first_packed: 14,
            w3_first_packed: 15,
            w3_last_packed: 16,
        };
        let abi = M2DecodeLayerCustomCallAbi::new(&shape, M2_NUM_EXPERTS, 13, offsets).unwrap();
        assert_eq!(
            abi.call.target(),
            "rvllm.m2.decode_layer.fused_attention_nvfp4_moe"
        );
        assert_eq!(abi.call.abi(), "m2_decode_layer_v1");
        assert_eq!(abi.batch, 4);
        assert_eq!(abi.ctx, 1024);
        assert_eq!(abi.kv_dtype, M2KvCacheDType::Bf16);
        assert_eq!(abi.kv_dtype.as_mlir_dtype(), "bf16");
        assert_eq!(abi.kv_cache_bytes, shape.kv_cache_bytes());
        assert_eq!(abi.expert_directory_cols, 13);
        assert_eq!(abi.weight_offsets, offsets);
    }
}
