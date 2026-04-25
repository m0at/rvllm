use std::path::PathBuf;

use rvllm_core::{
    BatchedPrefillPlan, ConfigError, PrefillRequest, ReqId, Result, RvllmError, TokenId,
};
use rvllm_fused::{M2PrefillKvDType, M2PrefillScanShape};
use rvllm_loader::{M2CheckpointIndex, M2CheckpointSummary};

use crate::{make_m2_prefill_input_specs, M2PrefillHostInputSpec};

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

impl M2RustPrefillPlan {
    pub fn total_host_input_bytes(&self) -> usize {
        self.input_specs.iter().map(|spec| spec.nbytes).sum()
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
}
