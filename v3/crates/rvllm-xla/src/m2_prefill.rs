use rvllm_core::{BatchedPrefillPlan, ConfigError, Result, RvllmError};
use rvllm_fused::M2PrefillScanShape;

use crate::ffi::PjrtElementType;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2PrefillHostInput {
    pub name: &'static str,
    pub shape: Vec<i64>,
    pub dtype: PjrtElementType,
    pub bytes: Vec<u8>,
}

pub fn make_m2_prefill_inputs(
    plan: &BatchedPrefillPlan,
    shape: M2PrefillScanShape,
) -> Result<Vec<M2PrefillHostInput>> {
    shape.validate()?;
    if plan.num_seqs() as usize != shape.batch {
        return Err(invalid("batch", "plan num_seqs does not match shape batch"));
    }
    if plan.num_tokens() as usize != shape.total_tokens() {
        return Err(invalid(
            "prompt_len",
            "plan token count does not match shape batch*prompt_len",
        ));
    }
    if plan.positions.len() != shape.total_tokens()
        || plan.slot_mapping.len() != shape.total_tokens()
        || plan.cu_seqlens_q.len() != shape.batch + 1
        || plan.context_lens.len() != shape.batch
    {
        return Err(invalid(
            "metadata",
            "plan metadata lengths do not match shape",
        ));
    }

    Ok(vec![
        input_i32(
            "token_ids",
            &[shape.batch, shape.prompt_len],
            plan.prompt_tokens_flat.iter().map(|t| t.raw() as i32),
        ),
        input_i32(
            "positions",
            &[shape.total_tokens()],
            plan.positions.iter().map(|&x| x as i32),
        ),
        input_i32(
            "slot_mapping",
            &[shape.total_tokens()],
            plan.slot_mapping.iter().copied(),
        ),
        input_i32(
            "cu_seqlens_q",
            &[shape.batch + 1],
            plan.cu_seqlens_q.iter().map(|&x| x as i32),
        ),
        input_i32(
            "context_lens",
            &[shape.batch],
            plan.context_lens.iter().map(|&x| x as i32),
        ),
        M2PrefillHostInput {
            name: "kv_cache",
            shape: vec![shape.kv_cache_bytes() as i64],
            dtype: shape.kv_dtype.into(),
            bytes: vec![0u8; shape.kv_cache_bytes()],
        },
    ])
}

fn input_i32(
    name: &'static str,
    shape: &[usize],
    vals: impl Iterator<Item = i32>,
) -> M2PrefillHostInput {
    let mut bytes = Vec::new();
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    M2PrefillHostInput {
        name,
        shape: shape.iter().map(|&x| x as i64).collect(),
        dtype: PjrtElementType::S32,
        bytes,
    }
}

impl From<rvllm_fused::M2PrefillKvDType> for PjrtElementType {
    fn from(value: rvllm_fused::M2PrefillKvDType) -> Self {
        match value {
            rvllm_fused::M2PrefillKvDType::Bf16 => PjrtElementType::BF16,
            rvllm_fused::M2PrefillKvDType::Int8 => PjrtElementType::S8,
        }
    }
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_prefill_inputs",
    )
}

#[cfg(test)]
mod tests {
    use rvllm_core::{BatchedPrefillPlan, PrefillRequest, ReqId, TokenId};
    use rvllm_fused::M2PrefillKvDType;

    use super::*;

    fn read_i32s(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect()
    }

    #[test]
    fn packs_single_prompt_inputs_for_pjrt() {
        let prompt = [TokenId(10), TokenId(11), TokenId(12), TokenId(13)];
        let plan = BatchedPrefillPlan::from_requests(&[PrefillRequest {
            req_id: ReqId(1),
            prompt_tokens: &prompt,
            max_blocks_per_seq: 64,
            block_size: 32,
        }])
        .unwrap();
        let shape = M2PrefillScanShape {
            batch: 1,
            prompt_len: 4,
            hidden: 3072,
            ctx: 2048,
            num_layers: 62,
            num_kv_heads: 8,
            head_dim: 128,
            kv_dtype: M2PrefillKvDType::Int8,
        };
        let inputs = make_m2_prefill_inputs(&plan, shape).unwrap();
        assert_eq!(inputs.len(), 6);
        assert_eq!(inputs[0].name, "token_ids");
        assert_eq!(inputs[0].shape, vec![1, 4]);
        assert_eq!(read_i32s(&inputs[0].bytes), vec![10, 11, 12, 13]);
        assert_eq!(read_i32s(&inputs[1].bytes), vec![0, 1, 2, 3]);
        assert_eq!(read_i32s(&inputs[2].bytes), vec![0, 1, 2, 3]);
        assert_eq!(read_i32s(&inputs[3].bytes), vec![0, 4]);
        assert_eq!(read_i32s(&inputs[4].bytes), vec![4]);
        assert_eq!(inputs[5].name, "kv_cache");
        assert_eq!(inputs[5].dtype, PjrtElementType::S8);
        assert_eq!(inputs[5].bytes.len(), shape.kv_cache_bytes());
    }

    #[test]
    fn rejects_shape_that_does_not_match_plan() {
        let prompt = [TokenId(1), TokenId(2)];
        let plan = BatchedPrefillPlan::from_requests(&[PrefillRequest {
            req_id: ReqId(1),
            prompt_tokens: &prompt,
            max_blocks_per_seq: 1,
            block_size: 32,
        }])
        .unwrap();
        let shape = M2PrefillScanShape {
            batch: 1,
            prompt_len: 3,
            hidden: 3072,
            ctx: 2048,
            num_layers: 62,
            num_kv_heads: 8,
            head_dim: 128,
            kv_dtype: M2PrefillKvDType::Int8,
        };
        assert!(make_m2_prefill_inputs(&plan, shape).is_err());
    }
}
