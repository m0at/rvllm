//! Input preparation: converts sequence metadata into batched model tensors.

use rvllm_core::prelude::{Result, TokenId};
use rvllm_model_runner::bridge::AttentionMetadata;
use rvllm_model_runner::input::ModelInput;

// Wire to real types from rvllm-sequence.
pub use rvllm_sequence::{SequenceData, SequenceGroupMetadata};

/// Prepare batched model input from a list of sequence group metadata.
///
/// Handles pure prefill, pure decode, and mixed prefill+decode batches.
/// For mixed batches, prefill groups are processed first, then decode groups,
/// and the results are merged into a single `ModelInput`.
pub fn prepare_input(metadata: &[SequenceGroupMetadata], block_size: usize) -> Result<ModelInput> {
    if metadata.is_empty() {
        return Ok(ModelInput {
            token_ids: Vec::new(),
            position_ids: Vec::new(),
            attention_metadata: AttentionMetadata {
                slot_mapping: Vec::new(),
                context_lens: Vec::new(),
                block_tables: Vec::new(),
                max_context_len: 0,
                query_lens: Vec::new(),
            },
            is_prefill: true,
        });
    }

    let prefill_groups: Vec<&SequenceGroupMetadata> =
        metadata.iter().filter(|g| g.is_prompt).collect();
    let decode_groups: Vec<&SequenceGroupMetadata> =
        metadata.iter().filter(|g| !g.is_prompt).collect();

    match (prefill_groups.is_empty(), decode_groups.is_empty()) {
        (false, true) => prepare_prefill(metadata, block_size),
        (true, false) => prepare_decode(metadata, block_size),
        (false, false) => {
            let prefill = prepare_prefill_refs(&prefill_groups, block_size)?;
            let decode = prepare_decode_refs(&decode_groups, block_size)?;
            Ok(merge_inputs(prefill, decode))
        }
        // Both empty is unreachable since metadata is non-empty, but handle gracefully.
        (true, true) => unreachable!(),
    }
}

/// Like `prepare_input`, but reuses caller-owned scratch for pure decode batches.
pub fn prepare_input_reuse(
    decode_scratch: &mut DecodeInputScratch,
    metadata: &[SequenceGroupMetadata],
    block_size: usize,
) -> Result<ModelInput> {
    if metadata.is_empty() {
        return Ok(ModelInput {
            token_ids: Vec::new(),
            position_ids: Vec::new(),
            attention_metadata: AttentionMetadata {
                slot_mapping: Vec::new(),
                context_lens: Vec::new(),
                block_tables: Vec::new(),
                max_context_len: 0,
                query_lens: Vec::new(),
            },
            is_prefill: true,
        });
    }

    let all_decode = metadata.iter().all(|g| !g.is_prompt);
    if all_decode {
        return prepare_decode_reuse(decode_scratch, metadata, block_size);
    }

    prepare_input(metadata, block_size)
}

/// Prepare input for the prefill (prompt processing) phase.
///
/// Each sequence contributes all its prompt tokens. Position IDs are
/// 0..prompt_len for each sequence. Block tables and slot mappings are
/// derived from the provided block tables.
fn prepare_prefill(metadata: &[SequenceGroupMetadata], block_size: usize) -> Result<ModelInput> {
    let mut token_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut slot_mapping = Vec::new();
    let mut context_lens = Vec::new();
    let mut block_tables_out = Vec::new();

    for group in metadata {
        for (seq_id, seq_data) in &group.seq_data {
            let prompt_tokens = &seq_data.prompt_token_ids;
            let prior_len = seq_data.output_token_ids.len();
            let seq_len = prior_len + prompt_tokens.len();

            token_ids.extend_from_slice(prompt_tokens);
            position_ids.extend((prior_len as u32)..(seq_len as u32));
            context_lens.push(seq_len as u32);

            // Build slot mapping from block table
            let bt = group
                .block_tables
                .get(seq_id)
                .map(|t| t.as_slice())
                .unwrap_or(&[]);

            for pos in prior_len..seq_len {
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
            query_lens: metadata
                .iter()
                .flat_map(|g| {
                    g.seq_data
                        .values()
                        .map(|sd| sd.prompt_token_ids.len() as u32)
                })
                .collect(),
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
fn prepare_decode(metadata: &[SequenceGroupMetadata], block_size: usize) -> Result<ModelInput> {
    let mut token_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut slot_mapping = Vec::new();
    let mut context_lens = Vec::new();
    let mut block_tables_out = Vec::new();

    for group in metadata {
        for (seq_id, seq_data) in &group.seq_data {
            let last_token = decode_last_token(seq_data);
            token_ids.push(last_token);

            let seq_len = decode_seq_len(seq_data);
            position_ids.push((seq_len - 1) as u32);
            context_lens.push(seq_len as u32);

            let bt = group
                .block_tables
                .get(seq_id)
                .map(|t| t.as_slice())
                .unwrap_or(&[]);

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
            query_lens: vec![1; context_lens.len()],
            context_lens,
            block_tables: block_tables_out,
            max_context_len,
        },
        is_prefill: false,
    })
}

/// Like `prepare_prefill` but accepts `&[&SequenceGroupMetadata]` for mixed-batch use.
fn prepare_prefill_refs(
    metadata: &[&SequenceGroupMetadata],
    block_size: usize,
) -> Result<ModelInput> {
    let mut token_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut slot_mapping = Vec::new();
    let mut context_lens = Vec::new();
    let mut block_tables_out = Vec::new();

    for group in metadata {
        for (seq_id, seq_data) in &group.seq_data {
            let prompt_tokens = &seq_data.prompt_token_ids;
            let prior_len = seq_data.output_token_ids.len();
            let seq_len = prior_len + prompt_tokens.len();

            token_ids.extend_from_slice(prompt_tokens);
            position_ids.extend((prior_len as u32)..(seq_len as u32));
            context_lens.push(seq_len as u32);

            let bt = group
                .block_tables
                .get(seq_id)
                .map(|t| t.as_slice())
                .unwrap_or(&[]);

            for pos in prior_len..seq_len {
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
            query_lens: metadata
                .iter()
                .flat_map(|g| {
                    g.seq_data
                        .values()
                        .map(|sd| sd.prompt_token_ids.len() as u32)
                })
                .collect(),
            context_lens,
            block_tables: block_tables_out,
            max_context_len,
        },
        is_prefill: true,
    })
}

/// Like `prepare_decode` but accepts `&[&SequenceGroupMetadata]` for mixed-batch use.
fn prepare_decode_refs(
    metadata: &[&SequenceGroupMetadata],
    block_size: usize,
) -> Result<ModelInput> {
    let mut token_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut slot_mapping = Vec::new();
    let mut context_lens = Vec::new();
    let mut block_tables_out = Vec::new();

    for group in metadata {
        for (seq_id, seq_data) in &group.seq_data {
            let last_token = decode_last_token(seq_data);
            token_ids.push(last_token);

            let seq_len = decode_seq_len(seq_data);
            position_ids.push((seq_len - 1) as u32);
            context_lens.push(seq_len as u32);

            let bt = group
                .block_tables
                .get(seq_id)
                .map(|t| t.as_slice())
                .unwrap_or(&[]);

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
            query_lens: vec![1; context_lens.len()],
            context_lens,
            block_tables: block_tables_out,
            max_context_len,
        },
        is_prefill: false,
    })
}

/// Merge prefill and decode `ModelInput`s into a single mixed batch.
/// Prefill tokens come first, then decode tokens.
fn merge_inputs(prefill: ModelInput, decode: ModelInput) -> ModelInput {
    let mut token_ids = prefill.token_ids;
    token_ids.extend(decode.token_ids);

    let mut position_ids = prefill.position_ids;
    position_ids.extend(decode.position_ids);

    let mut slot_mapping = prefill.attention_metadata.slot_mapping;
    slot_mapping.extend(decode.attention_metadata.slot_mapping);

    let mut context_lens = prefill.attention_metadata.context_lens;
    context_lens.extend(decode.attention_metadata.context_lens);

    let mut block_tables = prefill.attention_metadata.block_tables;
    block_tables.extend(decode.attention_metadata.block_tables);

    let max_context_len = std::cmp::max(
        prefill.attention_metadata.max_context_len,
        decode.attention_metadata.max_context_len,
    );

    let mut query_lens = prefill.attention_metadata.query_lens;
    query_lens.extend(decode.attention_metadata.query_lens);

    ModelInput {
        token_ids,
        position_ids,
        attention_metadata: AttentionMetadata {
            slot_mapping,
            query_lens,
            context_lens,
            block_tables,
            max_context_len,
        },
        is_prefill: true,
    }
}

/// Reusable scratch buffers for `prepare_decode_reuse`, avoiding per-step heap allocations.
pub struct DecodeInputScratch {
    pub token_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub slot_mapping: Vec<u32>,
    pub context_lens: Vec<u32>,
    pub block_tables: Vec<Vec<u32>>,
    pub query_lens: Vec<u32>,
}

impl DecodeInputScratch {
    pub fn new() -> Self {
        Self {
            token_ids: Vec::with_capacity(64),
            position_ids: Vec::with_capacity(64),
            slot_mapping: Vec::with_capacity(64),
            context_lens: Vec::with_capacity(64),
            block_tables: Vec::with_capacity(64),
            query_lens: Vec::with_capacity(64),
        }
    }

    fn clear(&mut self) {
        self.token_ids.clear();
        self.position_ids.clear();
        self.slot_mapping.clear();
        self.context_lens.clear();
        // Reuse inner Vecs in block_tables by clearing them rather than dropping
        for bt in &mut self.block_tables {
            bt.clear();
        }
        self.query_lens.clear();
    }
}

impl Default for DecodeInputScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Like `prepare_decode` but reuses caller-owned scratch buffers to avoid heap allocations.
pub fn prepare_decode_reuse(
    scratch: &mut DecodeInputScratch,
    metadata: &[SequenceGroupMetadata],
    block_size: usize,
) -> Result<ModelInput> {
    scratch.clear();

    // Count sequences for block_tables reuse
    let mut seq_idx = 0usize;

    for group in metadata {
        for (seq_id, seq_data) in &group.seq_data {
            let seq_len = decode_seq_len(seq_data);
            let last_token = decode_last_token(seq_data);
            scratch.token_ids.push(last_token);

            scratch.position_ids.push((seq_len - 1) as u32);
            scratch.context_lens.push(seq_len as u32);

            let bt = group
                .block_tables
                .get(seq_id)
                .map(|t| t.as_slice())
                .unwrap_or(&[]);

            let block_idx = (seq_len - 1) / block_size;
            let block_offset = (seq_len - 1) % block_size;
            if block_idx < bt.len() {
                let physical_block = bt[block_idx].0;
                scratch
                    .slot_mapping
                    .push(physical_block * block_size as u32 + block_offset as u32);
            } else {
                scratch.slot_mapping.push(0);
            }

            // Reuse existing inner Vec if available, otherwise push a new one
            if seq_idx < scratch.block_tables.len() {
                scratch.block_tables[seq_idx].extend(bt.iter().map(|b| b.0));
            } else {
                scratch.block_tables.push(bt.iter().map(|b| b.0).collect());
            }
            seq_idx += 1;
        }
    }

    // Truncate block_tables to actual sequence count (extras already cleared)
    scratch.block_tables.truncate(seq_idx);

    let num_seqs = scratch.context_lens.len();
    scratch.query_lens.resize(num_seqs, 1);

    let max_context_len = scratch.context_lens.iter().copied().max().unwrap_or(0);

    Ok(ModelInput {
        token_ids: scratch.token_ids.clone(),
        position_ids: scratch.position_ids.clone(),
        attention_metadata: AttentionMetadata {
            slot_mapping: scratch.slot_mapping.clone(),
            query_lens: scratch.query_lens.clone(),
            context_lens: scratch.context_lens.clone(),
            block_tables: scratch.block_tables.clone(),
            max_context_len,
        },
        is_prefill: false,
    })
}

fn decode_seq_len(sd: &SequenceData) -> usize {
    if sd.seq_len != 0 {
        sd.seq_len as usize
    } else {
        sd.prompt_token_ids.len() + sd.output_token_ids.len()
    }
}

fn decode_last_token(sd: &SequenceData) -> TokenId {
    if sd.seq_len != 0 {
        sd.last_token_id
    } else {
        sd.output_token_ids
            .last()
            .or_else(|| sd.prompt_token_ids.last())
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::prelude::{BlockId, SamplingParams, SequenceId};

    const TEST_BLOCK_SIZE: usize = 16;

    fn make_seq_data(prompt: Vec<TokenId>, output: Vec<TokenId>) -> SequenceData {
        SequenceData {
            prompt_token_ids: prompt,
            output_token_ids: output,
            cumulative_logprob: 0.0,
            seq_len: 0,
            last_token_id: 0,
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
        let input = prepare_input(&[], TEST_BLOCK_SIZE).unwrap();
        assert!(input.token_ids.is_empty());
        assert!(input.position_ids.is_empty());
        assert!(input.is_prefill);
    }

    #[test]
    fn prefill_single_sequence() {
        let sd = make_seq_data(vec![10, 20, 30, 40], vec![]);
        let blocks = vec![BlockId(0), BlockId(1)];
        let group = make_group(
            1,
            true,
            vec![(SequenceId(100), sd)],
            vec![(SequenceId(100), blocks)],
        );

        let input = prepare_input(&[group], TEST_BLOCK_SIZE).unwrap();
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
        let g1 = make_group(
            1,
            true,
            vec![(SequenceId(10), sd1)],
            vec![(SequenceId(10), vec![BlockId(0)])],
        );
        let g2 = make_group(
            2,
            true,
            vec![(SequenceId(20), sd2)],
            vec![(SequenceId(20), vec![BlockId(1)])],
        );

        let input = prepare_input(&[g1, g2], TEST_BLOCK_SIZE).unwrap();
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
        let group = make_group(
            1,
            false,
            vec![(SequenceId(100), sd)],
            vec![(SequenceId(100), blocks)],
        );

        let input = prepare_input(&[group], TEST_BLOCK_SIZE).unwrap();
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
        let g1 = make_group(
            1,
            false,
            vec![(SequenceId(10), sd1)],
            vec![(SequenceId(10), vec![BlockId(0)])],
        );
        let g2 = make_group(
            2,
            false,
            vec![(SequenceId(20), sd2)],
            vec![(SequenceId(20), vec![BlockId(1)])],
        );

        let input = prepare_input(&[g1, g2], TEST_BLOCK_SIZE).unwrap();
        assert!(!input.is_prefill);
        assert_eq!(input.token_ids, vec![100, 200]);
        assert_eq!(input.position_ids, vec![3, 2]);
        assert_eq!(input.attention_metadata.context_lens, vec![4, 3]);
        assert_eq!(input.attention_metadata.max_context_len, 4);
    }

    #[test]
    fn sequence_data_properties() {
        let sd = make_seq_data(vec![1, 2, 3], vec![4, 5]);
        assert_eq!(decode_seq_len(&sd), 5);
        assert_eq!(decode_last_token(&sd), 5);
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
        let group = make_group(
            1,
            true,
            vec![(SequenceId(10), sd)],
            vec![(SequenceId(10), vec![BlockId(5)])],
        );

        let input = prepare_input(&[group], TEST_BLOCK_SIZE).unwrap();
        // slots: 5*16+0=80, 5*16+1=81, 5*16+2=82
        assert_eq!(input.attention_metadata.slot_mapping, vec![80, 81, 82]);
    }

    #[test]
    fn slot_mapping_computed_correctly_decode() {
        // 5 tokens total (4 prompt + 1 gen), block_size=16, one block at physical id 3
        let sd = make_seq_data(vec![1, 2, 3, 4], vec![5]);
        let group = make_group(
            1,
            false,
            vec![(SequenceId(10), sd)],
            vec![(SequenceId(10), vec![BlockId(3)])],
        );

        let input = prepare_input(&[group], TEST_BLOCK_SIZE).unwrap();
        // position 4, block_idx=4/16=0, offset=4, slot=3*16+4=52
        assert_eq!(input.attention_metadata.slot_mapping, vec![52]);
    }

    #[test]
    fn block_tables_passed_through() {
        let sd = make_seq_data(vec![1, 2, 3], vec![]);
        let blocks = vec![BlockId(10), BlockId(20), BlockId(30)];
        let group = make_group(
            1,
            true,
            vec![(SequenceId(5), sd)],
            vec![(SequenceId(5), blocks)],
        );

        let input = prepare_input(&[group], TEST_BLOCK_SIZE).unwrap();
        assert_eq!(
            input.attention_metadata.block_tables,
            vec![vec![10, 20, 30]]
        );
    }

    #[test]
    fn mixed_prefill_and_decode_batch() {
        // Prefill group: 3 prompt tokens
        let sd_prefill = make_seq_data(vec![1, 2, 3], vec![]);
        let g_prefill = make_group(
            1,
            true,
            vec![(SequenceId(10), sd_prefill)],
            vec![(SequenceId(10), vec![BlockId(5)])],
        );

        // Decode group: 4 prompt + 1 generated = 5 total tokens, contributes 1 token
        let sd_decode = make_seq_data(vec![10, 20, 30, 40], vec![50]);
        let g_decode = make_group(
            2,
            false,
            vec![(SequenceId(20), sd_decode)],
            vec![(SequenceId(20), vec![BlockId(3)])],
        );

        let input = prepare_input(&[g_prefill, g_decode], TEST_BLOCK_SIZE).unwrap();
        // is_prefill true because batch contains prefill groups
        assert!(input.is_prefill);
        // Prefill tokens first, then decode token
        assert_eq!(input.token_ids, vec![1, 2, 3, 50]);
        // Prefill positions: 0,1,2; Decode position: 4 (total_len - 1)
        assert_eq!(input.position_ids, vec![0, 1, 2, 4]);
        // Prefill context_len: 3; Decode context_len: 5
        assert_eq!(input.attention_metadata.context_lens, vec![3, 5]);
        assert_eq!(input.attention_metadata.max_context_len, 5);
        // 4 total tokens in the batch
        assert_eq!(input.num_tokens(), 4);
        // Slot mapping: prefill slots 80,81,82; decode slot 52
        assert_eq!(input.attention_metadata.slot_mapping, vec![80, 81, 82, 52]);
    }

    #[test]
    fn mixed_batch_decode_before_prefill_in_metadata_order() {
        // Even if decode group appears first in the metadata slice,
        // prefill tokens should still come first in the merged output.
        let sd_decode = make_seq_data(vec![4, 5], vec![200]);
        let g_decode = make_group(
            1,
            false,
            vec![(SequenceId(10), sd_decode)],
            vec![(SequenceId(10), vec![BlockId(1)])],
        );

        let sd_prefill = make_seq_data(vec![1, 2, 3], vec![]);
        let g_prefill = make_group(
            2,
            true,
            vec![(SequenceId(20), sd_prefill)],
            vec![(SequenceId(20), vec![BlockId(5)])],
        );

        // Decode group listed before prefill group in the slice
        let input = prepare_input(&[g_decode, g_prefill], TEST_BLOCK_SIZE).unwrap();
        assert!(input.is_prefill);
        // Prefill tokens come first regardless of metadata order
        assert_eq!(input.token_ids, vec![1, 2, 3, 200]);
        assert_eq!(input.position_ids, vec![0, 1, 2, 2]);
        assert_eq!(input.attention_metadata.context_lens, vec![3, 3]);
    }
}
