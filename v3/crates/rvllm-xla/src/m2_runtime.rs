use std::path::PathBuf;

use rvllm_core::{
    BatchedPrefillPlan, ConfigError, PrefillRequest, ReqId, Result, RvllmError, TokenId,
};
use rvllm_fused::{M2PrefillKvDType, M2PrefillScanShape};
use rvllm_loader::{M2CheckpointIndex, M2CheckpointSummary};

use crate::{
    m2_decode_graph_mlir, make_m2_prefill_input_specs, M2GraphAbi, M2GraphShape,
    M2PrefillHostInputSpec, M2WeightUploadPlan, PjrtElementType,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RustPrefillConfig {
    pub model_dir: PathBuf,
    pub batch: usize,
    pub prompt_len: usize,
    pub ctx: usize,
    pub block_size: u32,
    pub kv_dtype: M2PrefillKvDType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RustPrefillPlan {
    pub checkpoint: M2CheckpointSummary,
    pub shape: M2PrefillScanShape,
    pub plan: BatchedPrefillPlan,
    pub input_specs: Vec<M2PrefillHostInputSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RustPrefillDecodeConfig {
    pub model_dir: PathBuf,
    pub batch: usize,
    pub prompt_len: usize,
    pub decode_steps: usize,
    pub ctx: usize,
    pub block_size: u32,
    pub kv_dtype: M2PrefillKvDType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2DecodeRuntimeInputSpec {
    pub name: &'static str,
    pub shape: Vec<i64>,
    pub dtype: PjrtElementType,
    pub nbytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RustPrefillDecodePlan {
    pub prefill: M2RustPrefillPlan,
    pub decode_shape: M2GraphShape,
    pub decode_input_specs: Vec<M2DecodeRuntimeInputSpec>,
    pub decode_output_specs: Vec<M2DecodeRuntimeInputSpec>,
    pub seed_decode_token_ids: Vec<i32>,
    pub seed_decode_positions: Vec<i32>,
    pub decode_steps: usize,
    pub weight_arena_bytes: usize,
    pub weight_entries: usize,
    pub decode_mlir: String,
}

impl M2RustPrefillPlan {
    pub fn total_host_input_bytes(&self) -> usize {
        self.input_specs.iter().map(|spec| spec.nbytes).sum()
    }
}

impl M2RustPrefillDecodePlan {
    pub fn total_decode_input_bytes(&self) -> usize {
        self.decode_input_specs.iter().map(|spec| spec.nbytes).sum()
    }
}

pub fn plan_m2_rust_prefill(cfg: &M2RustPrefillConfig) -> Result<M2RustPrefillPlan> {
    cfg.validate()?;
    let index_path = cfg.model_dir.join("model.safetensors.index.json");
    let index = M2CheckpointIndex::from_index_file(index_path)?;
    let checkpoint = index.validate_m2(62, 256)?;

    let max_blocks_per_seq = max_blocks_per_seq(cfg.ctx, cfg.block_size)?;
    let prompts = synthetic_prompts(cfg.batch, cfg.prompt_len)?;
    let requests = prompts
        .iter()
        .enumerate()
        .map(|(i, prompt_tokens)| PrefillRequest {
            req_id: ReqId((i + 1) as u64),
            prompt_tokens,
            max_blocks_per_seq,
            block_size: cfg.block_size,
        })
        .collect::<Vec<_>>();
    let plan = BatchedPrefillPlan::from_requests(&requests)?;
    let shape = M2PrefillScanShape {
        batch: cfg.batch,
        prompt_len: cfg.prompt_len,
        hidden: 3072,
        ctx: cfg.ctx,
        num_layers: 62,
        num_kv_heads: 8,
        head_dim: 128,
        kv_dtype: cfg.kv_dtype,
    };
    let input_specs = make_m2_prefill_input_specs(&plan, shape)?;
    Ok(M2RustPrefillPlan {
        checkpoint,
        shape,
        plan,
        input_specs,
    })
}

pub fn plan_m2_rust_prefill_decode(
    cfg: &M2RustPrefillDecodeConfig,
) -> Result<M2RustPrefillDecodePlan> {
    cfg.validate()?;
    let prefill = plan_m2_rust_prefill(&M2RustPrefillConfig {
        model_dir: cfg.model_dir.clone(),
        batch: cfg.batch,
        prompt_len: cfg.prompt_len,
        ctx: cfg.ctx,
        block_size: cfg.block_size,
        kv_dtype: cfg.kv_dtype,
    })?;
    let kv_bytes_per_elem = match cfg.kv_dtype {
        M2PrefillKvDType::Int8 => 1,
        M2PrefillKvDType::Bf16 => 2,
    };
    let decode_shape = M2GraphShape::decode(cfg.batch, cfg.ctx, kv_bytes_per_elem);
    let abi = M2GraphAbi::new(decode_shape.clone())?;
    let weights = M2WeightUploadPlan::from_index_dir(&cfg.model_dir, &abi)?;
    let arena = weights.flat_arena(128)?;
    let decode_mlir = m2_decode_graph_mlir("rvllm_m2_decode", &decode_shape, &arena)?;
    let decode_input_specs = decode_input_specs(&decode_shape, arena.total_bytes);
    let decode_output_specs = decode_output_specs(&decode_shape);
    Ok(M2RustPrefillDecodePlan {
        seed_decode_token_ids: seed_decode_token_ids(&prefill.plan, cfg.batch, cfg.prompt_len)?,
        seed_decode_positions: vec![cfg.prompt_len as i32; cfg.batch],
        prefill,
        decode_shape,
        decode_input_specs,
        decode_output_specs,
        decode_steps: cfg.decode_steps,
        weight_arena_bytes: arena.total_bytes,
        weight_entries: arena.entries.len(),
        decode_mlir,
    })
}

impl M2RustPrefillConfig {
    fn validate(&self) -> Result<()> {
        if self.batch == 0 {
            return Err(invalid("batch", "must be > 0"));
        }
        if self.prompt_len == 0 {
            return Err(invalid("prompt_len", "must be > 0"));
        }
        if self.ctx == 0 {
            return Err(invalid("ctx", "must be > 0"));
        }
        if self.prompt_len > self.ctx {
            return Err(invalid("prompt_len", "must be <= ctx"));
        }
        if self.block_size == 0 {
            return Err(invalid("block_size", "must be > 0"));
        }
        Ok(())
    }
}

impl M2RustPrefillDecodeConfig {
    fn validate(&self) -> Result<()> {
        if self.decode_steps == 0 {
            return Err(invalid("decode_steps", "must be > 0"));
        }
        M2RustPrefillConfig {
            model_dir: self.model_dir.clone(),
            batch: self.batch,
            prompt_len: self.prompt_len,
            ctx: self.ctx,
            block_size: self.block_size,
            kv_dtype: self.kv_dtype,
        }
        .validate()
    }
}

fn max_blocks_per_seq(ctx: usize, block_size: u32) -> Result<u32> {
    let block_size = block_size as usize;
    let blocks = (ctx + block_size - 1) / block_size;
    u32::try_from(blocks).map_err(|_| invalid("ctx", "too large for u32 block count"))
}

fn synthetic_prompts(batch: usize, prompt_len: usize) -> Result<Vec<Vec<TokenId>>> {
    if prompt_len > u32::MAX as usize - 1024 {
        return Err(invalid("prompt_len", "too large for u32 token ids"));
    }
    let mut prompts = Vec::with_capacity(batch);
    for seq in 0..batch {
        let base = 1024u32 + (seq as u32) * 17;
        let prompt = (0..prompt_len)
            .map(|i| TokenId(base + i as u32))
            .collect::<Vec<_>>();
        prompts.push(prompt);
    }
    Ok(prompts)
}

fn seed_decode_token_ids(
    plan: &BatchedPrefillPlan,
    batch: usize,
    prompt_len: usize,
) -> Result<Vec<i32>> {
    if plan.prompt_tokens_flat.len() != batch * prompt_len {
        return Err(invalid("prompt_tokens", "flat prompt length mismatch"));
    }
    Ok(plan
        .prompt_tokens_flat
        .chunks_exact(prompt_len)
        .map(|seq| seq[prompt_len - 1].raw() as i32)
        .collect())
}

fn decode_input_specs(
    shape: &M2GraphShape,
    weight_arena_bytes: usize,
) -> Vec<M2DecodeRuntimeInputSpec> {
    vec![
        decode_spec("token_ids", &[shape.batch], PjrtElementType::S32),
        decode_spec("positions", &[shape.batch], PjrtElementType::S32),
        M2DecodeRuntimeInputSpec {
            name: "kv_cache",
            shape: vec![shape.kv_cache_bytes() as i64],
            dtype: PjrtElementType::S8,
            nbytes: shape.kv_cache_bytes(),
        },
        M2DecodeRuntimeInputSpec {
            name: "weight_arena",
            shape: vec![weight_arena_bytes as i64],
            dtype: PjrtElementType::S8,
            nbytes: weight_arena_bytes,
        },
    ]
}

fn decode_output_specs(shape: &M2GraphShape) -> Vec<M2DecodeRuntimeInputSpec> {
    vec![
        decode_spec("logits", &[shape.batch, 200_064], PjrtElementType::BF16),
        decode_spec("next_token", &[shape.batch], PjrtElementType::S32),
        M2DecodeRuntimeInputSpec {
            name: "kv_cache",
            shape: vec![shape.kv_cache_bytes() as i64],
            dtype: PjrtElementType::S8,
            nbytes: shape.kv_cache_bytes(),
        },
    ]
}

fn decode_spec(
    name: &'static str,
    shape: &[usize],
    dtype: PjrtElementType,
) -> M2DecodeRuntimeInputSpec {
    let elems = shape.iter().product::<usize>();
    M2DecodeRuntimeInputSpec {
        name,
        shape: shape.iter().map(|&x| x as i64).collect(),
        dtype,
        nbytes: elems * element_size(dtype),
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

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_rust_prefill",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_b8_prefill_from_checked_in_m2_schema_without_python() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema");
        let cfg = M2RustPrefillConfig {
            model_dir,
            batch: 8,
            prompt_len: 4,
            ctx: 8,
            block_size: 8,
            kv_dtype: M2PrefillKvDType::Int8,
        };
        let plan = plan_m2_rust_prefill(&cfg).unwrap();
        assert_eq!(plan.checkpoint.total_tensors, 191_069);
        assert_eq!(plan.checkpoint.nvfp4_groups, 47_616);
        assert_eq!(plan.shape.batch, 8);
        assert_eq!(plan.shape.total_tokens(), 32);
        assert_eq!(
            plan.plan.cu_seqlens_q,
            vec![0, 4, 8, 12, 16, 20, 24, 28, 32]
        );
        assert_eq!(plan.plan.context_lens, vec![4; 8]);
        assert_eq!(plan.input_specs[0].name, "token_ids");
        assert_eq!(plan.input_specs[0].shape, vec![8, 4]);
        assert_eq!(plan.input_specs[5].name, "kv_cache");
        assert_eq!(plan.input_specs[5].nbytes, plan.shape.kv_cache_bytes());
    }

    #[test]
    fn rejects_prompt_longer_than_context() {
        let cfg = M2RustPrefillConfig {
            model_dir: PathBuf::from("unused"),
            batch: 8,
            prompt_len: 9,
            ctx: 8,
            block_size: 8,
            kv_dtype: M2PrefillKvDType::Int8,
        };
        assert!(plan_m2_rust_prefill(&cfg).is_err());
    }

    #[test]
    fn plans_b8_prefill_decode_sequence_without_python() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema");
        let plan = plan_m2_rust_prefill_decode(&M2RustPrefillDecodeConfig {
            model_dir,
            batch: 8,
            prompt_len: 4,
            decode_steps: 32,
            ctx: 2048,
            block_size: 32,
            kv_dtype: M2PrefillKvDType::Int8,
        })
        .unwrap();
        assert_eq!(plan.prefill.shape.total_tokens(), 32);
        assert_eq!(plan.decode_shape.batch, 8);
        assert_eq!(plan.decode_shape.kv_cache_bytes(), 2_080_374_784);
        assert_eq!(plan.decode_steps, 32);
        assert_eq!(plan.weight_entries, 191_069);
        assert_eq!(
            plan.seed_decode_token_ids,
            vec![1027, 1044, 1061, 1078, 1095, 1112, 1129, 1146]
        );
        assert_eq!(plan.seed_decode_positions, vec![4; 8]);
        assert_eq!(plan.decode_input_specs[3].name, "weight_arena");
        assert_eq!(plan.decode_input_specs[3].nbytes, plan.weight_arena_bytes);
        assert!(plan.decode_mlir.contains("\"rvllm.m2.decode_layer\""));
    }
}
