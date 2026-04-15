use std::collections::HashMap;

use crate::types::{BlockId, GpuBatchInput, RequestId, ResponseFormat, SamplingParams, WorkerRequest};
#[cfg(test)]
use crate::types::{SequenceId, TokenId};

pub struct InputBuilder {
    input: GpuBatchInput,
    block_tables_raw: Vec<Vec<u32>>,
    prefill_keys: Vec<RequestId>,
    decode_keys: Vec<RequestId>,
    is_all_greedy: bool,
}

impl InputBuilder {
    pub fn new() -> Self {
        Self {
            input: GpuBatchInput {
                num_seqs: 0,
                num_prefill_seqs: 0,
                num_decode_seqs: 0,
                seq_ids: Vec::with_capacity(128),
                token_ids: Vec::with_capacity(128),
                position_ids: Vec::with_capacity(128),
                slot_mapping: Vec::with_capacity(128),
                context_lens: Vec::with_capacity(128),
                query_lens: Vec::with_capacity(128),
                is_all_greedy: true,
                block_tables_flat: Vec::with_capacity(128 * 32),
                max_blocks_per_seq: 0,
                prefill_tokens: Vec::with_capacity(512),
                prefill_positions: Vec::with_capacity(512),
                prefill_slot_mapping: Vec::with_capacity(512),
                is_all_decode: true,
                is_all_prefill: true,
                max_context_len: 0,
            },
            block_tables_raw: Vec::with_capacity(128),
            prefill_keys: Vec::with_capacity(64),
            decode_keys: Vec::with_capacity(64),
            is_all_greedy: true,
        }
    }

    pub fn build(
        &mut self,
        requests: &HashMap<RequestId, WorkerRequest>,
        block_size: usize,
    ) -> &GpuBatchInput {
        self.clear();

        for (id, req) in requests {
            if req.is_prefill {
                self.prefill_keys.push(*id);
            } else {
                self.decode_keys.push(*id);
            }
        }

        self.prefill_keys.sort_unstable_by_key(|id| id.0);
        self.decode_keys.sort_unstable_by_key(|id| id.0);

        for i in 0..self.prefill_keys.len() {
            let id = self.prefill_keys[i];
            let req = &requests[&id];
            self.add_prefill_request(req, block_size);
        }

        for i in 0..self.decode_keys.len() {
            let id = self.decode_keys[i];
            let req = &requests[&id];
            self.add_decode_request(req, block_size);
        }

        let num_prefill = self.prefill_keys.len();
        let num_decode = self.decode_keys.len();

        self.flatten_block_tables();

        self.input.num_seqs = num_prefill + num_decode;
        self.input.num_prefill_seqs = num_prefill;
        self.input.num_decode_seqs = num_decode;
        self.input.is_all_decode = num_prefill == 0;
        self.input.is_all_prefill = num_decode == 0;
        self.input.max_context_len = self.input.context_lens.iter().copied().max().unwrap_or(0);
        self.input.is_all_greedy = self.is_all_greedy;

        &self.input
    }

    fn clear(&mut self) {
        self.input.clear();
        for bt in &mut self.block_tables_raw {
            bt.clear();
        }
        self.prefill_keys.clear();
        self.decode_keys.clear();
        self.is_all_greedy = true;
    }

    fn track_greedy(&mut self, p: &SamplingParams) {
        if self.is_all_greedy
            && (p.temperature != 0.0
                || !matches!(p.response_format, ResponseFormat::Text)
                || p.repetition_penalty != 1.0
                || p.frequency_penalty != 0.0
                || p.presence_penalty != 0.0)
        {
            self.is_all_greedy = false;
        }
    }

    fn add_decode_request(&mut self, req: &WorkerRequest, block_size: usize) {
        let seq_len = req.seq_len();
        let pos = seq_len - 1;
        let block_idx = pos / block_size;
        let block_offset = pos % block_size;
        let slot = req.block_table[block_idx].0 * block_size as u32 + block_offset as u32;

        self.input.seq_ids.push(req.seq_id.0);
        self.input.token_ids.push(req.last_token_id());
        self.input.position_ids.push(pos as u32);
        self.input.slot_mapping.push(slot);
        self.input.context_lens.push(seq_len as u32);
        self.input.query_lens.push(1);
        self.track_greedy(&req.sampling_params);

        self.push_block_table(&req.block_table);
    }

    fn add_prefill_request(&mut self, req: &WorkerRequest, block_size: usize) {
        let chunk = &req.token_chunk;
        let chunk_len = chunk.end - chunk.start;

        self.input.seq_ids.push(req.seq_id.0);
        let last_chunk_token = if chunk_len > 0 {
            req.prompt_token_ids[chunk.end - 1]
        } else {
            req.last_token_id()
        };
        self.input.token_ids.push(last_chunk_token);
        self.input.position_ids.push(if chunk_len > 0 {
            (chunk.end - 1) as u32
        } else {
            0
        });
        if chunk_len > 0 {
            let last_pos = chunk.end - 1;
            let bi = last_pos / block_size;
            let bo = last_pos % block_size;
            self.input
                .slot_mapping
                .push(req.block_table[bi].0 * block_size as u32 + bo as u32);
        } else {
            self.input.slot_mapping.push(0);
        }
        self.input.query_lens.push(chunk_len as u32);
        self.input
            .context_lens
            .push(req.num_computed_tokens as u32 + chunk_len as u32);
        self.track_greedy(&req.sampling_params);

        for pos in chunk.clone() {
            let token_id = req.prompt_token_ids[pos];
            let bi = pos / block_size;
            let bo = pos % block_size;
            let slot = req.block_table[bi].0 * block_size as u32 + bo as u32;

            self.input.prefill_tokens.push(token_id);
            self.input.prefill_positions.push(pos as u32);
            self.input.prefill_slot_mapping.push(slot);
        }

        self.push_block_table(&req.block_table);
    }

    fn push_block_table(&mut self, table: &[BlockId]) {
        let idx = self.input.seq_ids.len() - 1;
        if idx < self.block_tables_raw.len() {
            self.block_tables_raw[idx].extend(table.iter().map(|b| b.0));
        } else {
            self.block_tables_raw
                .push(table.iter().map(|b| b.0).collect());
        }
    }

    fn flatten_block_tables(&mut self) {
        let num_seqs = self.input.seq_ids.len();
        let max_blocks = self
            .block_tables_raw
            .iter()
            .take(num_seqs)
            .map(|bt| bt.len())
            .max()
            .unwrap_or(0);
        self.input.max_blocks_per_seq = max_blocks;

        self.input.block_tables_flat.clear();
        self.input
            .block_tables_flat
            .resize(num_seqs * max_blocks, 0);

        for (seq_idx, row) in self.block_tables_raw.iter().take(num_seqs).enumerate() {
            let base = seq_idx * max_blocks;
            for (blk_idx, &blk) in row.iter().enumerate() {
                self.input.block_tables_flat[base + blk_idx] = blk;
            }
        }
    }
}

impl Default for InputBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBatchInput {
    pub fn all_greedy(&self) -> bool {
        self.is_all_greedy
    }

    pub fn max_context_len(&self) -> u32 {
        self.context_lens.iter().copied().max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decode_request(
        req_id: u64,
        seq_id: u64,
        prompt: &[TokenId],
        output: &[TokenId],
        blocks: &[u32],
    ) -> (RequestId, WorkerRequest) {
        let id = RequestId(req_id);
        let req = WorkerRequest {
            request_id: id,
            seq_id: SequenceId(seq_id),
            prompt_token_ids: prompt.to_vec(),
            output_token_ids: output.to_vec(),
            sampling_params: SamplingParams::default(),
            block_table: blocks.iter().map(|&b| BlockId(b)).collect(),
            is_prefill: false,
            num_computed_tokens: prompt.len() + output.len(),
            token_chunk: 0..0,
        };
        (id, req)
    }

    fn make_prefill_request(
        req_id: u64,
        seq_id: u64,
        prompt: &[TokenId],
        blocks: &[u32],
        chunk: std::ops::Range<usize>,
    ) -> (RequestId, WorkerRequest) {
        let id = RequestId(req_id);
        let req = WorkerRequest {
            request_id: id,
            seq_id: SequenceId(seq_id),
            prompt_token_ids: prompt.to_vec(),
            output_token_ids: Vec::new(),
            sampling_params: SamplingParams::default(),
            block_table: blocks.iter().map(|&b| BlockId(b)).collect(),
            is_prefill: true,
            num_computed_tokens: 0,
            token_chunk: chunk,
        };
        (id, req)
    }

    #[test]
    fn test_decode_only() {
        let block_size = 16;
        let mut requests = HashMap::new();

        let (id1, req1) = make_decode_request(1, 100, &[10, 20, 30], &[40, 50], &[10]);
        requests.insert(id1, req1);

        let (id2, req2) = make_decode_request(2, 200, &[1; 16], &[99], &[20, 21]);
        requests.insert(id2, req2);

        let mut builder = InputBuilder::new();
        let input = builder.build(&requests, block_size);

        assert_eq!(input.num_seqs, 2);
        assert_eq!(input.num_prefill_seqs, 0);
        assert_eq!(input.num_decode_seqs, 2);
        assert!(input.is_all_decode);
        assert!(!input.is_all_prefill);

        assert_eq!(input.seq_ids, vec![100, 200]);
        assert_eq!(input.token_ids, vec![50, 99]);
        assert_eq!(input.position_ids, vec![4, 16]);
        assert_eq!(input.slot_mapping, vec![164, 336]);
        assert_eq!(input.context_lens, vec![5, 17]);
        assert_eq!(input.query_lens, vec![1, 1]);

        assert_eq!(input.max_blocks_per_seq, 2);
        assert_eq!(input.block_tables_flat, vec![10, 0, 20, 21]);

        assert!(input.prefill_tokens.is_empty());
    }

    #[test]
    fn test_prefill_only() {
        let block_size = 4;
        let mut requests = HashMap::new();

        let (id, req) = make_prefill_request(1, 10, &[100, 101, 102, 103], &[5], 0..4);
        requests.insert(id, req);

        let mut builder = InputBuilder::new();
        let input = builder.build(&requests, block_size);

        assert_eq!(input.num_seqs, 1);
        assert_eq!(input.num_prefill_seqs, 1);
        assert_eq!(input.num_decode_seqs, 0);
        assert!(input.is_all_prefill);

        assert_eq!(input.prefill_tokens, vec![100, 101, 102, 103]);
        assert_eq!(input.prefill_positions, vec![0, 1, 2, 3]);
        assert_eq!(input.prefill_slot_mapping, vec![20, 21, 22, 23]);

        assert_eq!(input.query_lens, vec![4]);
        assert_eq!(input.context_lens, vec![4]);
    }

    #[test]
    fn test_mixed_prefill_decode() {
        let block_size = 8;
        let mut requests = HashMap::new();

        let (id_d, req_d) = make_decode_request(2, 200, &[1, 2, 3], &[4], &[10]);
        requests.insert(id_d, req_d);

        let (id_p, req_p) = make_prefill_request(1, 100, &[10, 20, 30], &[5], 0..3);
        requests.insert(id_p, req_p);

        let mut builder = InputBuilder::new();
        let input = builder.build(&requests, block_size);

        assert_eq!(input.num_seqs, 2);
        assert_eq!(input.num_prefill_seqs, 1);
        assert_eq!(input.num_decode_seqs, 1);
        assert!(!input.is_all_decode);
        assert!(!input.is_all_prefill);
        assert_eq!(input.seq_ids, vec![100, 200]);
    }

    #[test]
    fn test_builder_reuse() {
        let block_size = 16;
        let mut builder = InputBuilder::new();

        let mut requests = HashMap::new();
        let (id, req) = make_decode_request(1, 10, &[1, 2], &[3], &[0]);
        requests.insert(id, req);
        let _ = builder.build(&requests, block_size);

        let mut requests2 = HashMap::new();
        let (id2, req2) = make_decode_request(5, 50, &[10, 20, 30, 40], &[50, 60], &[7, 8]);
        requests2.insert(id2, req2);
        let input = builder.build(&requests2, block_size);

        assert_eq!(input.num_seqs, 1);
        assert_eq!(input.seq_ids, vec![50]);
        assert_eq!(input.token_ids, vec![60]);
    }

    #[test]
    fn test_all_greedy() {
        let mut input = GpuBatchInput::new();
        assert!(input.all_greedy());

        input.is_all_greedy = false;
        assert!(!input.all_greedy());
    }

    #[test]
    fn test_empty_requests() {
        let mut builder = InputBuilder::new();
        let requests = HashMap::new();
        let input = builder.build(&requests, 16);

        assert_eq!(input.num_seqs, 0);
        assert!(input.is_all_decode);
        assert!(input.is_all_prefill);
        assert_eq!(input.max_context_len, 0);
    }
}
