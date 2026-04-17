//! `MetadataLayout`: the frozen per-bucket packed-buffer layout.
//!
//! Keyed on `(bucket, max_blocks_per_seq)`. Computed once at engine
//! init for every bucket in the graph-capture set, stored as a
//! `BTreeMap<(bucket, max_blocks), MetadataLayout>`. Captured graphs
//! bind the exact device offsets in this struct; replays write into
//! those offsets. There is NO second layout — prefill and decode have
//! separate entry points in `rvllm-runtime` that each produce their
//! own `BatchPlan`, and each plan goes through exactly one upload path
//! (`pack::upload`) keyed by its layout.

use rvllm_core::MetaLayoutHash;
use sha2::{Digest, Sha256};

/// Byte offsets (in i32 elements, not bytes) into the packed metadata
/// buffer. Every field is padded to `bucket` entries; `block_tables`
/// is `bucket * max_blocks`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MetadataLayout {
    pub bucket: u32,
    pub max_blocks: u32,
    /// Offsets into the packed buffer (units = i32 elements).
    pub token_ids_off: u32,
    pub positions_off: u32,
    pub context_lens_off: u32,
    pub block_tables_off: u32,
    pub slot_mapping_off: u32,
    pub seq_start_pos_off: u32,
    /// Total length of the packed buffer (i32 elements).
    pub total_elements: u32,
}

impl MetadataLayout {
    /// Compute the canonical layout for a given bucket + max_blocks.
    pub fn compute(bucket: u32, max_blocks: u32) -> Self {
        let token_ids_off = 0u32;
        let positions_off = token_ids_off + bucket;
        let context_lens_off = positions_off + bucket;
        let block_tables_off = context_lens_off + bucket;
        let slot_mapping_off = block_tables_off + bucket * max_blocks;
        let seq_start_pos_off = slot_mapping_off + bucket;
        // seq_start_pos is bucket+1 entries (prefix sums + total).
        let total_elements = seq_start_pos_off + bucket + 1;
        Self {
            bucket,
            max_blocks,
            token_ids_off,
            positions_off,
            context_lens_off,
            block_tables_off,
            slot_mapping_off,
            seq_start_pos_off,
            total_elements,
        }
    }

    /// sha256 of the layout descriptor. Captured graphs carry this
    /// hash so replay can assert the bucket's layout hasn't drifted.
    pub fn hash(&self) -> MetaLayoutHash {
        let mut h = Sha256::new();
        h.update(self.bucket.to_le_bytes());
        h.update(self.max_blocks.to_le_bytes());
        h.update(self.token_ids_off.to_le_bytes());
        h.update(self.positions_off.to_le_bytes());
        h.update(self.context_lens_off.to_le_bytes());
        h.update(self.block_tables_off.to_le_bytes());
        h.update(self.slot_mapping_off.to_le_bytes());
        h.update(self.seq_start_pos_off.to_le_bytes());
        h.update(self.total_elements.to_le_bytes());
        let digest = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        MetaLayoutHash(out)
    }

    /// Bytes needed to hold the packed buffer.
    pub fn bytes(&self) -> usize {
        (self.total_elements as usize) * core::mem::size_of::<i32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_deterministic_for_bucket_maxblocks() {
        let a = MetadataLayout::compute(128, 129);
        let b = MetadataLayout::compute(128, 129);
        assert_eq!(a, b);
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn layout_differs_across_buckets() {
        let a = MetadataLayout::compute(1, 129);
        let b = MetadataLayout::compute(128, 129);
        assert_ne!(a.hash(), b.hash());
        // block_tables_off scales with bucket
        assert!(b.block_tables_off > a.block_tables_off);
    }

    #[test]
    fn qwen_decode_128_fits_in_under_100kb() {
        let l = MetadataLayout::compute(128, 129);
        assert!(l.bytes() < 100 * 1024);
    }
}
