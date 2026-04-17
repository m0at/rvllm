//! Input plan that feeds `pack::upload`.
//!
//! Decoupled from the scheduler so this crate doesn't depend upstream.
//! The scheduler (`rvllm-runtime::scheduler`) populates one of these and
//! hands it to the metadata layer.

use rvllm_core::{BlockId, TokenId};

/// One decoded step's worth of scheduler output, in the shape the
/// metadata packer expects. All slices are borrowed from scheduler-owned
/// storage; this struct never allocates.
#[derive(Debug)]
pub struct BatchPlan<'s> {
    /// Number of active sequences in this step (≤ bucket).
    pub num_seqs: u32,
    /// Per-seq current token (to be re-embedded). Padded with 0 to bucket.
    pub token_ids: &'s [TokenId],
    /// Per-seq position in its sequence.
    pub positions: &'s [u32],
    /// Per-seq current context length (# of valid KV tokens).
    pub context_lens: &'s [u32],
    /// Per-seq row-major block table, flattened. `block_tables.len() == num_seqs * max_blocks_input`.
    pub block_tables_flat: &'s [BlockId],
    pub max_blocks_input: u32,
    /// Per-seq KV slot for the new token. -1 for padded slots.
    pub slot_mapping: &'s [i32],
    /// `[0, 1, 2, ..., num_seqs]` for decode; scheduler fills in for
    /// prefill with per-seq query lengths.
    pub seq_start_pos: &'s [u32],
}

impl<'s> BatchPlan<'s> {
    /// Sanity-check that the plan fits the layout's bucket.
    pub fn fits_layout(&self, layout: &crate::layout::MetadataLayout) -> bool {
        self.num_seqs <= layout.bucket
            && self.token_ids.len() == self.num_seqs as usize
            && self.positions.len() == self.num_seqs as usize
            && self.context_lens.len() == self.num_seqs as usize
            && self.slot_mapping.len() == self.num_seqs as usize
            && self.max_blocks_input <= layout.max_blocks
    }
}
