//! Input preparation: converts sequence metadata into batched model tensors.

use rvllm_core::prelude::{Result, TokenId};
use rvllm_model_runner::input::ModelInput;
use rvllm_model_runner::bridge::AttentionMetadata;

// Wire to real types from rvllm-sequence.
pub use rvllm_sequence::{SequenceData, SequenceGroupMetadata};

/// Prepare batched model input from a list of sequence group metadata.
///
/// Dispatches to `prepare_prefill` or `prepare_decode` based on whether
/// all groups are in prompt phase or decode phase. Mixed batches are not
/// supported (vLLM separates them at the scheduler level).
pub fn prepare_input(metadata: &[SequenceGroupMetadata]) -> Result<ModelInput> {
    if metadata.is_empty() {
        return Ok(ModelInput {
            token_ids: Vec::new(),
            position_ids: Vec::new(),
            attention_metadata: AttentionMetadata {
                slot_mapping: Vec::new(),
                context_lens: Vec::new(),
                block_tables: Vec::new(),
                max_context_len: 0,
            },
            is_prefill: true,
        });
    }

    let is_prefill = metadata[0].is_prompt;
    if is_prefill {
        prepare_prefill(metadata)
    } else {
        prepare_decode(metadata)
    }
}

/// Prepare input for the prefill (prompt processing) phase.
///
/// Each sequence contributes all its prompt tokens. Position IDs are
/// 0..prompt_len for each sequence. Block tables and slot mappings are
/// derived from the provided block tables.
fn prepare_prefill(metadata: &[SequenceGroupMetadata]) -> Result<ModelInput> {
    let mut token_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut slot_mapping = Vec::new();
    let mut context_lens = Vec::new();
    let mut block_tables_out = Vec::new();

    for group in metadata {
        for (seq_id, seq_data) in &group.seq_data {
            let prompt_tokens = &seq_data.prompt_token_ids;
            let seq_len = prompt_tokens.len();

            token_ids.extend_from_slice(prompt_tokens);
            position_ids.extend((0..seq_len as u32).collect::<Vec<_>>());
            context_lens.push(seq_len as u32);

            // Build slot mapping from block table
            let bt = group.block_tables.get(seq_id)
                .map(|t| t.as_slice())
                .unwrap_or(&[]);

            let block_size: usize = 16; // default block size
            for pos in 0..seq_len {
                let block_idx = pos / block_size;
                let block_offset = pos % block_size;
                if block_idx < bt.len() {
                    let physical_block = bt[block_idx].0;
                    slot_mapping.push(physical_block * block_size as u32 + block_offset as u32);
                } else {
                    slot_mapping.push(0);
                }
            }

            block_tables_out.push(bt.iter().map(|b| b.0).collect());
        }
    }

    let max_context_len = context_lens.iter().copied().max().unwrap_or(0);

    Ok(ModelInput {
        token_ids,
        position_ids,
        attention_metadata: AttentionMetadata {
            slot_mapping,
            context_lens,
            block_tables: block_tables_out,
            max_context_len,
        },
        is_prefill: true,
    })
}

/// Prepare input for the decode (autoregressive) phase.
///
/// Each sequence contributes exactly one token (the last generated token).
/// Position ID is the total sequence length - 1. Block tables are passed
/// through for paged attention.
fn prepare_decode(metadata: &[SequenceGroupMetadata]) -> Result<ModelInput> {
    let mut token_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut slot_mapping = Vec::new();
    let mut context_lens = Vec::new();
    let mut block_tables_out = Vec::new();

    for group in metadata {
        for (seq_id, seq_data) in &group.seq_data {
            // Decode: single token per sequence (the last one)
            let all_tokens = all_token_ids(seq_data);
            let last_token = all_tokens.last().copied().unwrap_or(0);
            token_ids.push(last_token);

            let seq_len = all_tokens.len();
            position_ids.push((seq_len - 1) as u32);
            context_lens.push(seq_len as u32);

            let bt = group.block_tables.get(seq_id)
                .map(|t| t.as_slice())
                .unwrap_or(&[]);

            let block_size: usize = 16;
            let block_idx = (seq_len - 1) / block_size;
            let block_offset = (seq_len - 1) % block_size;
            if block_idx < bt.len() {
                let physical_block = bt[block_idx].0;
                slot_mapping.push(physical_block * block_size as u32 + block_offset as u32);
            } else {
                slot_mapping.push(0);
            }

            block_tables_out.push(bt.iter().map(|b| b.0).collect());
        }
    }

    let max_context_len = context_lens.iter().copied().max().unwrap_or(0);

    Ok(ModelInput {
        token_ids,
        position_ids,
        attention_metadata: AttentionMetadata {
            slot_mapping,
            context_lens,
            block_tables: block_tables_out,
            max_context_len,
        },
        is_prefill: false,
    })
}

/// Helper: get all token IDs (prompt + output) from SequenceData.
fn all_token_ids(sd: &SequenceData) -> Vec<TokenId> {
    let mut ids = sd.prompt_token_ids.clone();
    ids.extend_from_slice(&sd.output_token_ids);
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::prelude::{BlockId, SamplingParams, SequenceId};

    fn make_seq_data(prompt: Vec<TokenId>, output: Vec<TokenId>) -> SequenceData {
        SequenceData {
            prompt_token_ids: prompt,
            output_token_ids: output,
            cumulative_logprob: 0.0,
        }
    }

    fn make_group(
        request_id: u64,
        is_prompt: bool,
        seqs: Vec<(SequenceId, SequenceData)>,
        block_tables: Vec<(SequenceId, Vec<BlockId>)>,
    ) -> SequenceGroupMetadata {
        SequenceGroupMetadata {
            request_id: request_id.into(),
            is_prompt,
            seq_data: seqs.into_iter().collect(),
            sampling_params: SamplingParams::default(),
            block_tables: block_tables.into_iter().collect(),
        }
    }

    #[test]
    fn empty_input_produces_empty_model_input() {
        let input = prepare_input(&[]).unwrap();
        assert!(input.token_ids.is_empty());
        assert!(input.position_ids.is_empty());
        assert!(input.is_prefill);
    }

    #[test]
    fn prefill_single_sequence() {
        let sd = make_seq_data(vec![10, 20, 30, 40], vec![]);
        let blocks = vec![BlockId(0), BlockId(1)];
        let group = make_group(1, true, vec![(SequenceId(100), sd)], vec![(SequenceId(100), blocks)]);

        let input = prepare_input(&[group]).unwrap();
        assert!(input.is_prefill);
        assert_eq!(input.token_ids, vec![10, 20, 30, 40]);
        assert_eq!(input.position_ids, vec![0, 1, 2, 3]);
        assert_eq!(input.attention_metadata.context_lens, vec![4]);
        assert_eq!(input.attention_metadata.max_context_len, 4);
        assert_eq!(input.num_tokens(), 4);
    }

    #[test]
    fn prefill_multiple_sequences() {
        let sd1 = make_seq_data(vec![1, 2, 3], vec![]);
        let sd2 = make_seq_data(vec![4, 5], vec![]);
        let g1 = make_group(1, true, vec![(SequenceId(10), sd1)], vec![(SequenceId(10), vec![BlockId(0)])]);
        let g2 = make_group(2, true, vec![(SequenceId(20), sd2)], vec![(SequenceId(20), vec![BlockId(1)])]);

        let input = prepare_input(&[g1, g2]).unwrap();
        assert!(input.is_prefill);
        assert_eq!(input.token_ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(input.position_ids, vec![0, 1, 2, 0, 1]);
        assert_eq!(input.attention_metadata.context_lens, vec![3, 2]);
        assert_eq!(input.attention_metadata.max_context_len, 3);
    }

    #[test]
    fn decode_single_sequence() {
        // Sequence with 4 prompt tokens + 1 generated token
        let sd = make_seq_data(vec![10, 20, 30, 40], vec![50]);
        let blocks = vec![BlockId(0), BlockId(1)];
        let group = make_group(1, false, vec![(SequenceId(100), sd)], vec![(SequenceId(100), blocks)]);

        let input = prepare_input(&[group]).unwrap();
        assert!(!input.is_prefill);
        // Only the last token
        assert_eq!(input.token_ids, vec![50]);
        // Position = total_len - 1 = 4
        assert_eq!(input.position_ids, vec![4]);
        assert_eq!(input.attention_metadata.context_lens, vec![5]);
        assert_eq!(input.num_tokens(), 1);
    }

    #[test]
    fn decode_multiple_sequences() {
        let sd1 = make_seq_data(vec![1, 2, 3], vec![100]);
        let sd2 = make_seq_data(vec![4, 5], vec![200]);
        let g1 = make_group(1, false, vec![(SequenceId(10), sd1)], vec![(SequenceId(10), vec![BlockId(0)])]);
        let g2 = make_group(2, false, vec![(SequenceId(20), sd2)], vec![(SequenceId(20), vec![BlockId(1)])]);

        let input = prepare_input(&[g1, g2]).unwrap();
        assert!(!input.is_prefill);
        assert_eq!(input.token_ids, vec![100, 200]);
        assert_eq!(input.position_ids, vec![3, 2]);
        assert_eq!(input.attention_metadata.context_lens, vec![4, 3]);
        assert_eq!(input.attention_metadata.max_context_len, 4);
    }

    #[test]
    fn sequence_data_properties() {
        let sd = make_seq_data(vec![1, 2, 3], vec![4, 5]);
        let all = all_token_ids(&sd);
        assert_eq!(all.len(), 5);
        assert_eq!(all.last(), Some(&5));
        assert_eq!(sd.output_token_ids.len(), 2);
        assert_eq!(sd.prompt_token_ids.len(), 3);
    }

    #[test]
    fn empty_sequence_data() {
        let sd = make_seq_data(vec![], vec![]);
        assert!(sd.prompt_token_ids.is_empty());
        assert!(sd.output_token_ids.is_empty());
    }

    #[test]
    fn slot_mapping_computed_correctly_prefill() {
        // 3 tokens, block_size=16, one block at physical id 5
        let sd = make_seq_data(vec![1, 2, 3], vec![]);
        let group = make_group(1, true, vec![(SequenceId(10), sd)], vec![(SequenceId(10), vec![BlockId(5)])]);

        let input = prepare_input(&[group]).unwrap();
        // slots: 5*16+0=80, 5*16+1=81, 5*16+2=82
        assert_eq!(input.attention_metadata.slot_mapping, vec![80, 81, 82]);
    }

    #[test]
    fn slot_mapping_computed_correctly_decode() {
        // 5 tokens total (4 prompt + 1 gen), block_size=16, one block at physical id 3
        let sd = make_seq_data(vec![1, 2, 3, 4], vec![5]);
        let group = make_group(1, false, vec![(SequenceId(10), sd)], vec![(SequenceId(10), vec![BlockId(3)])]);

        let input = prepare_input(&[group]).unwrap();
        // position 4, block_idx=4/16=0, offset=4, slot=3*16+4=52
        assert_eq!(input.attention_metadata.slot_mapping, vec![52]);
    }

    #[test]
    fn block_tables_passed_through() {
        let sd = make_seq_data(vec![1, 2, 3], vec![]);
        let blocks = vec![BlockId(10), BlockId(20), BlockId(30)];
        let group = make_group(1, true, vec![(SequenceId(5), sd)], vec![(SequenceId(5), blocks)]);

        let input = prepare_input(&[group]).unwrap();
        assert_eq!(input.attention_metadata.block_tables, vec![vec![10, 20, 30]]);
    }
}
