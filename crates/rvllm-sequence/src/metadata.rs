use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use rvllm_core::prelude::{BlockId, RequestId, SamplingParams, SequenceId, TokenId};

/// Serializable subset of sequence data sent to workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceData {
    pub prompt_token_ids: Vec<TokenId>,
    pub output_token_ids: Vec<TokenId>,
    pub cumulative_logprob: f32,
}

/// Metadata for a sequence group, serialized and sent to model workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceGroupMetadata {
    pub request_id: RequestId,
    pub is_prompt: bool,
    pub seq_data: HashMap<SequenceId, SequenceData>,
    pub sampling_params: SamplingParams,
    pub block_tables: HashMap<SequenceId, Vec<BlockId>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metadata() -> SequenceGroupMetadata {
        let mut seq_data = HashMap::new();
        seq_data.insert(
            SequenceId(1),
            SequenceData {
                prompt_token_ids: vec![10, 20],
                output_token_ids: vec![30],
                cumulative_logprob: -1.5,
            },
        );
        let mut block_tables = HashMap::new();
        block_tables.insert(SequenceId(1), vec![BlockId(0), BlockId(1)]);

        SequenceGroupMetadata {
            request_id: RequestId(42),
            is_prompt: false,
            seq_data,
            sampling_params: SamplingParams::default(),
            block_tables,
        }
    }

    #[test]
    fn serde_roundtrip() {
        let m = make_metadata();
        let json = serde_json::to_string(&m).unwrap();
        let m2: SequenceGroupMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.request_id, RequestId(42));
        assert!(!m2.is_prompt);
        assert_eq!(m2.seq_data.len(), 1);
        let sd = &m2.seq_data[&SequenceId(1)];
        assert_eq!(sd.prompt_token_ids, vec![10, 20]);
        assert_eq!(sd.output_token_ids, vec![30]);
        assert_eq!(m2.block_tables[&SequenceId(1)].len(), 2);
    }

    #[test]
    fn multiple_sequences() {
        let mut seq_data = HashMap::new();
        seq_data.insert(
            SequenceId(0),
            SequenceData {
                prompt_token_ids: vec![1],
                output_token_ids: vec![],
                cumulative_logprob: 0.0,
            },
        );
        seq_data.insert(
            SequenceId(1),
            SequenceData {
                prompt_token_ids: vec![1],
                output_token_ids: vec![2, 3],
                cumulative_logprob: -2.0,
            },
        );

        let m = SequenceGroupMetadata {
            request_id: RequestId(7),
            is_prompt: true,
            seq_data,
            sampling_params: SamplingParams::default(),
            block_tables: HashMap::new(),
        };

        assert_eq!(m.seq_data.len(), 2);
        assert!(m.is_prompt);
    }
}
