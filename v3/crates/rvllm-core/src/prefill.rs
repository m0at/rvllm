use crate::{ConfigError, ReqId, Result, RvllmError, TokenId};

#[derive(Copy, Clone, Debug)]
pub struct PrefillRequest<'a> {
    pub req_id: ReqId,
    pub prompt_tokens: &'a [TokenId],
    pub max_blocks_per_seq: u32,
    pub block_size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchedPrefillPlan {
    pub req_ids: Vec<ReqId>,
    pub prompt_tokens_flat: Vec<TokenId>,
    pub cu_seqlens_q: Vec<u32>,
    pub positions: Vec<u32>,
    pub slot_mapping: Vec<i32>,
    pub context_lens: Vec<u32>,
    pub max_seqlen_q: u32,
}

impl BatchedPrefillPlan {
    pub fn from_requests(requests: &[PrefillRequest<'_>]) -> Result<Self> {
        if requests.is_empty() {
            return Err(invalid("requests", "must not be empty"));
        }

        let mut req_ids = Vec::with_capacity(requests.len());
        let mut prompt_tokens_flat = Vec::new();
        let mut cu_seqlens_q = Vec::with_capacity(requests.len() + 1);
        let mut positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut context_lens = Vec::with_capacity(requests.len());
        let mut max_seqlen_q = 0u32;
        cu_seqlens_q.push(0);

        for (seq_idx, req) in requests.iter().enumerate() {
            validate_request(req)?;
            let prompt_len = req.prompt_tokens.len() as u32;
            req_ids.push(req.req_id);
            prompt_tokens_flat.extend_from_slice(req.prompt_tokens);
            cu_seqlens_q.push(prompt_tokens_flat.len() as u32);
            context_lens.push(prompt_len);
            max_seqlen_q = max_seqlen_q.max(prompt_len);

            let base_slot = seq_idx as u32 * req.max_blocks_per_seq * req.block_size;
            for pos in 0..prompt_len {
                positions.push(pos);
                slot_mapping.push((base_slot + pos) as i32);
            }
        }

        Ok(Self {
            req_ids,
            prompt_tokens_flat,
            cu_seqlens_q,
            positions,
            slot_mapping,
            context_lens,
            max_seqlen_q,
        })
    }

    pub fn num_tokens(&self) -> u32 {
        self.prompt_tokens_flat.len() as u32
    }

    pub fn num_seqs(&self) -> u32 {
        self.req_ids.len() as u32
    }
}

pub fn serial_prompt_metadata(
    prompt_len: u32,
    seq_idx: u32,
    max_blocks_per_seq: u32,
    block_size: u32,
) -> Result<(Vec<u32>, Vec<i32>)> {
    if prompt_len == 0 {
        return Err(invalid("prompt_len", "must be > 0"));
    }
    if block_size == 0 || max_blocks_per_seq == 0 {
        return Err(invalid(
            "blocks",
            "block_size and max_blocks_per_seq must be > 0",
        ));
    }
    if prompt_len > max_blocks_per_seq * block_size {
        return Err(invalid("prompt_len", "exceeds allocated KV blocks"));
    }

    let base_slot = seq_idx * max_blocks_per_seq * block_size;
    let positions = (0..prompt_len).collect();
    let slot_mapping = (0..prompt_len)
        .map(|pos| (base_slot + pos) as i32)
        .collect();
    Ok((positions, slot_mapping))
}

fn validate_request(req: &PrefillRequest<'_>) -> Result<()> {
    let prompt_len = req.prompt_tokens.len() as u32;
    if prompt_len == 0 {
        return Err(invalid("prompt_tokens", "must not be empty"));
    }
    if req.block_size == 0 || req.max_blocks_per_seq == 0 {
        return Err(invalid(
            "blocks",
            "block_size and max_blocks_per_seq must be > 0",
        ));
    }
    if prompt_len > req.max_blocks_per_seq * req.block_size {
        return Err(invalid("prompt_tokens", "exceeds allocated KV blocks"));
    }
    Ok(())
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "prefill",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_prompt_plan_matches_serial_metadata() {
        let prompt = [TokenId(10), TokenId(11), TokenId(12), TokenId(13)];
        let req = PrefillRequest {
            req_id: ReqId(7),
            prompt_tokens: &prompt,
            max_blocks_per_seq: 2,
            block_size: 32,
        };
        let plan = BatchedPrefillPlan::from_requests(&[req]).unwrap();
        let (serial_pos, serial_slot) = serial_prompt_metadata(4, 0, 2, 32).unwrap();

        assert_eq!(plan.req_ids, vec![ReqId(7)]);
        assert_eq!(plan.prompt_tokens_flat, prompt);
        assert_eq!(plan.cu_seqlens_q, vec![0, 4]);
        assert_eq!(plan.context_lens, vec![4]);
        assert_eq!(plan.positions, serial_pos);
        assert_eq!(plan.slot_mapping, serial_slot);
        assert_eq!(plan.max_seqlen_q, 4);
        assert_eq!(plan.num_tokens(), 4);
        assert_eq!(plan.num_seqs(), 1);
    }

    #[test]
    fn multi_prompt_plan_offsets_slots_by_sequence() {
        let a = [TokenId(1), TokenId(2)];
        let b = [TokenId(3), TokenId(4), TokenId(5)];
        let reqs = [
            PrefillRequest {
                req_id: ReqId(1),
                prompt_tokens: &a,
                max_blocks_per_seq: 2,
                block_size: 32,
            },
            PrefillRequest {
                req_id: ReqId(2),
                prompt_tokens: &b,
                max_blocks_per_seq: 2,
                block_size: 32,
            },
        ];
        let plan = BatchedPrefillPlan::from_requests(&reqs).unwrap();
        assert_eq!(
            plan.prompt_tokens_flat,
            vec![TokenId(1), TokenId(2), TokenId(3), TokenId(4), TokenId(5)]
        );
        assert_eq!(plan.cu_seqlens_q, vec![0, 2, 5]);
        assert_eq!(plan.positions, vec![0, 1, 0, 1, 2]);
        assert_eq!(plan.slot_mapping, vec![0, 1, 64, 65, 66]);
        assert_eq!(plan.context_lens, vec![2, 3]);
        assert_eq!(plan.max_seqlen_q, 3);
    }

    #[test]
    fn rejects_prompt_that_exceeds_kv_allocation() {
        let prompt = [TokenId(0); 65];
        let req = PrefillRequest {
            req_id: ReqId(1),
            prompt_tokens: &prompt,
            max_blocks_per_seq: 2,
            block_size: 32,
        };
        assert!(BatchedPrefillPlan::from_requests(&[req]).is_err());
    }
}
