//! Attention metadata passed alongside each forward call.

use crate::buffer::GpuBuffer;

/// Metadata describing the paged-attention layout for a batch of sequences.
pub struct AttentionMetadata {
    /// Mapping from sequence position to physical block id.
    /// Shape: `[num_seqs, max_blocks_per_seq]`
    pub block_tables: GpuBuffer<i32>,

    /// Number of context tokens for each sequence.
    /// Shape: `[num_seqs]`
    pub context_lens: GpuBuffer<i32>,

    /// Maximum context length across all sequences in the batch.
    pub max_context_len: usize,

    /// Whether this batch is a prefill (prompt) pass or decode pass.
    pub is_prefill: bool,

    /// Mapping from each token position to its KV cache slot.
    /// Shape: `[num_tokens]`
    pub slot_mapping: GpuBuffer<i32>,
}

impl AttentionMetadata {
    pub fn new(
        block_tables: GpuBuffer<i32>,
        context_lens: GpuBuffer<i32>,
        max_context_len: usize,
        is_prefill: bool,
        slot_mapping: GpuBuffer<i32>,
    ) -> Self {
        Self {
            block_tables,
            context_lens,
            max_context_len,
            is_prefill,
            slot_mapping,
        }
    }

    /// Number of sequences in the batch.
    pub fn num_seqs(&self) -> usize {
        self.context_lens.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf(data: Vec<i32>, shape: Vec<usize>) -> GpuBuffer<i32> {
        GpuBuffer { data, shape }
    }

    #[test]
    fn metadata_num_seqs() {
        let meta = AttentionMetadata::new(
            make_buf(vec![0, 1, 2, 3], vec![2, 2]),
            make_buf(vec![4, 6], vec![2]),
            6,
            false,
            make_buf(vec![0, 1, 2, 3, 4, 5], vec![6]),
        );
        assert_eq!(meta.num_seqs(), 2);
        assert_eq!(meta.max_context_len, 6);
        assert!(!meta.is_prefill);
    }

    #[test]
    fn metadata_prefill_flag() {
        let meta = AttentionMetadata::new(
            make_buf(vec![0], vec![1, 1]),
            make_buf(vec![10], vec![1]),
            10,
            true,
            make_buf(vec![0; 10], vec![10]),
        );
        assert!(meta.is_prefill);
    }
}
