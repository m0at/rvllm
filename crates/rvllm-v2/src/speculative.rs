use std::collections::HashMap;

use crate::types::{GpuBatchInput, TokenId};

// =================================================================
// Config
// =================================================================

pub struct SpecDecodeConfig {
    pub max_draft_len: usize,
    pub ngram_order: usize,
    pub max_batch_for_spec: usize,
}

impl Default for SpecDecodeConfig {
    fn default() -> Self {
        Self {
            max_draft_len: 5,
            ngram_order: 3,
            max_batch_for_spec: 32,
        }
    }
}

// =================================================================
// N-gram drafter
// =================================================================

/// N-gram based token drafter for speculative decoding.
/// Builds tables from context (prompt + output), drafts continuations
/// by matching the longest n-gram suffix.
pub struct NgramDrafter {
    order: usize,
    max_draft: usize,
    // tables[n-1] maps an n-gram suffix to the list of tokens that followed it
    tables: Vec<HashMap<Vec<TokenId>, Vec<TokenId>>>,
}

impl NgramDrafter {
    pub fn new(config: &SpecDecodeConfig) -> Self {
        let mut tables = Vec::with_capacity(config.ngram_order);
        for _ in 0..config.ngram_order {
            tables.push(HashMap::new());
        }
        Self {
            order: config.ngram_order,
            max_draft: config.max_draft_len,
            tables,
        }
    }

    /// Rebuild n-gram tables from the full token context.
    pub fn build_table(&mut self, tokens: &[TokenId]) {
        for table in &mut self.tables {
            table.clear();
        }
        for n in 1..=self.order {
            if tokens.len() <= n {
                continue;
            }
            let table = &mut self.tables[n - 1];
            for i in 0..tokens.len() - n {
                let key: Vec<TokenId> = tokens[i..i + n].to_vec();
                let next = tokens[i + n];
                table.entry(key).or_default().push(next);
            }
        }
    }

    /// Draft up to max_draft tokens by repeatedly matching the longest suffix.
    pub fn draft(&self, context: &[TokenId]) -> Vec<TokenId> {
        let mut drafts = Vec::with_capacity(self.max_draft);
        let mut ext = context.to_vec();

        for _ in 0..self.max_draft {
            let mut found = None;
            for n in (1..=self.order).rev() {
                if ext.len() < n {
                    continue;
                }
                let suffix = &ext[ext.len() - n..];
                if let Some(table) = self.tables.get(n - 1) {
                    if let Some(continuations) = table.get(suffix) {
                        if let Some(&next) = continuations.last() {
                            found = Some(next);
                            break;
                        }
                    }
                }
            }
            match found {
                Some(token) => {
                    drafts.push(token);
                    ext.push(token);
                }
                None => break,
            }
        }
        drafts
    }
}

// =================================================================
// Verification
// =================================================================

/// Verify draft tokens against target model argmax outputs.
///
/// `draft_tokens`: K tokens from the drafter.
/// `target_ids`:   K+1 argmax token IDs from the target model verify pass.
///   target_ids[i] = model prediction at verify position i.
///   Positions: [last_real, d0, d1, ..., dK-1]
///   Predictions: [pred_for_next_after_last, pred_for_next_after_d0, ...]
///
/// Returns accepted tokens (length 1..K+1).
pub fn verify_and_accept(draft_tokens: &[TokenId], target_ids: &[TokenId]) -> Vec<TokenId> {
    let k = draft_tokens.len();
    let mut accepted = Vec::with_capacity(k + 1);

    for i in 0..k {
        if i < target_ids.len() && draft_tokens[i] == target_ids[i] {
            accepted.push(draft_tokens[i]);
        } else {
            if i < target_ids.len() {
                accepted.push(target_ids[i]);
            }
            return accepted;
        }
    }
    // All K correct -- bonus token
    if k < target_ids.len() {
        accepted.push(target_ids[k]);
    }
    accepted
}

// =================================================================
// Verify input construction
// =================================================================

/// Per-sequence data needed for the verify pass.
pub struct VerifySequence {
    pub seq_id: u64,
    pub last_token: TokenId,
    pub seq_len: usize,
    pub draft_tokens: Vec<TokenId>,
    pub block_table: Vec<u32>,
}

/// Build a GpuBatchInput for the speculative verify pass.
/// Each sequence contributes K+1 tokens treated as a mini-prefill:
///   tokens  = [last_real, d0, d1, ..., dK-1]
///   positions = [seq_len-1, seq_len, seq_len+1, ..., seq_len+K-1]
///
/// The target model runs causal attention over these K+1 positions using
/// existing KV cache entries for positions 0..seq_len-2.
pub fn build_verify_input(
    sequences: &[VerifySequence],
    block_size: usize,
) -> GpuBatchInput {
    let mut input = GpuBatchInput::new();
    let num_seqs = sequences.len();
    let mut block_tables_raw: Vec<Vec<u32>> = Vec::with_capacity(num_seqs);
    let mut max_blocks = 0usize;

    for seq in sequences {
        let k = seq.draft_tokens.len();
        let k_plus_1 = k + 1;

        input.seq_ids.push(seq.seq_id);
        // Per-seq metadata (last position)
        input.token_ids.push(seq.last_token);
        let last_verify_pos = seq.seq_len + k - 1;
        input.position_ids.push(last_verify_pos as u32);
        let bi = last_verify_pos / block_size;
        let bo = last_verify_pos % block_size;
        input
            .slot_mapping
            .push(seq.block_table[bi] * block_size as u32 + bo as u32);

        input.query_lens.push(k_plus_1 as u32);
        input.context_lens.push((seq.seq_len + k_plus_1 - 1) as u32);
        // all_greedy stays true (SamplingParams::default is greedy)

        // Prefill token list: [last_real, d0, d1, ..., dK-1]
        // Position seq_len-1 for last_real (already has KV but we re-write -- harmless)
        let base_pos = seq.seq_len - 1;
        input.prefill_tokens.push(seq.last_token);
        input.prefill_positions.push(base_pos as u32);
        let bi0 = base_pos / block_size;
        let bo0 = base_pos % block_size;
        input
            .prefill_slot_mapping
            .push(seq.block_table[bi0] * block_size as u32 + bo0 as u32);

        for (di, &draft_tok) in seq.draft_tokens.iter().enumerate() {
            let pos = seq.seq_len + di;
            let bi = pos / block_size;
            let bo = pos % block_size;
            input.prefill_tokens.push(draft_tok);
            input.prefill_positions.push(pos as u32);
            input
                .prefill_slot_mapping
                .push(seq.block_table[bi] * block_size as u32 + bo as u32);
        }

        let bt = seq.block_table.clone();
        max_blocks = max_blocks.max(bt.len());
        block_tables_raw.push(bt);
    }

    // Flatten block tables
    input.block_tables_flat.resize(num_seqs * max_blocks, 0);
    for (i, bt) in block_tables_raw.iter().enumerate() {
        for (j, &b) in bt.iter().enumerate() {
            input.block_tables_flat[i * max_blocks + j] = b;
        }
    }
    input.max_blocks_per_seq = max_blocks;

    input.num_seqs = num_seqs;
    input.num_prefill_seqs = num_seqs;
    input.num_decode_seqs = 0;
    input.is_all_decode = false;
    input.is_all_prefill = true;
    input.max_context_len = input.context_lens.iter().copied().max().unwrap_or(0);

    input
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_all_match() {
        let drafts = vec![10, 20, 30];
        let targets = vec![10, 20, 30, 99];
        let accepted = verify_and_accept(&drafts, &targets);
        assert_eq!(accepted, vec![10, 20, 30, 99]);
    }

    #[test]
    fn verify_first_mismatch() {
        let drafts = vec![10, 20, 30];
        let targets = vec![10, 25, 30, 99];
        let accepted = verify_and_accept(&drafts, &targets);
        assert_eq!(accepted, vec![10, 25]);
    }

    #[test]
    fn verify_immediate_mismatch() {
        let drafts = vec![10, 20, 30];
        let targets = vec![99, 20, 30, 50];
        let accepted = verify_and_accept(&drafts, &targets);
        assert_eq!(accepted, vec![99]);
    }

    #[test]
    fn verify_empty_draft() {
        let drafts: Vec<TokenId> = vec![];
        let targets = vec![42];
        let accepted = verify_and_accept(&drafts, &targets);
        assert_eq!(accepted, vec![42]);
    }

    #[test]
    fn ngram_basic() {
        let config = SpecDecodeConfig {
            max_draft_len: 3,
            ngram_order: 2,
            max_batch_for_spec: 32,
        };
        let mut drafter = NgramDrafter::new(&config);
        // "A B C A B D" -- after "A B" we've seen C (first) and D (last)
        let tokens = vec![1, 2, 3, 1, 2, 4];
        drafter.build_table(&tokens);
        // suffix [1,2] -> last continuation = 4
        let draft = drafter.draft(&[1, 2]);
        assert_eq!(draft[0], 4);
    }

    #[test]
    fn ngram_chain() {
        let config = SpecDecodeConfig {
            max_draft_len: 4,
            ngram_order: 2,
            max_batch_for_spec: 32,
        };
        let mut drafter = NgramDrafter::new(&config);
        // repeating pattern: 1 2 3 1 2 3 1 2 3
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        drafter.build_table(&tokens);
        // from [2, 3]: next should be 1, then from [3,1] -> 2, etc.
        let draft = drafter.draft(&[2, 3]);
        assert_eq!(draft, vec![1, 2, 3, 1]);
    }

    #[test]
    fn verify_input_structure() {
        let seqs = vec![VerifySequence {
            seq_id: 100,
            last_token: 42,
            seq_len: 10,
            draft_tokens: vec![50, 51, 52],
            block_table: vec![0, 1, 2, 3], // 4 blocks covers positions 0..15
        }];
        let input = build_verify_input(&seqs, 4);
        assert_eq!(input.num_seqs, 1);
        assert_eq!(input.num_prefill_seqs, 1);
        assert!(input.is_all_prefill);
        assert_eq!(input.prefill_tokens.len(), 4); // 1 real + 3 draft
        assert_eq!(input.prefill_tokens, vec![42, 50, 51, 52]);
        assert_eq!(input.prefill_positions, vec![9, 10, 11, 12]);
        assert_eq!(input.query_lens, vec![4]);
    }
}
