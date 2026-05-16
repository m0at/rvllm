//! The ONE metadata upload path.
//!
//! Every `BatchPlan` goes through `upload`. There is no `patch`
//! variant, no non-padded path, no fast-path that skips block_tables.
//! The April-16 correctness bug (CoW-only block-change detection)
//! cannot happen because there is nothing to skip.
//!
//! `upload` fills the caller-provided `pinned_host` buffer, then the
//! runtime issues one async HtoD copy into the device region at the
//! layout's `total_elements * 4` bytes.

use rvllm_core::{Result, RvllmError, SampleCtx, SamplingError};

use crate::layout::MetadataLayout;
use crate::plan::BatchPlan;

/// Fill `pinned_host` (i32 slice, length == layout.total_elements)
/// with the packed metadata for this plan. Returns `Err` if the plan
/// does not fit the layout's bucket (would be a scheduler bug).
pub fn upload(
    layout: &MetadataLayout,
    plan: &BatchPlan<'_>,
    pinned_host: &mut [i32],
) -> Result<()> {
    if pinned_host.len() != layout.total_elements as usize {
        return Err(invalid(
            "pinned_host.len",
            "must equal layout.total_elements",
        ));
    }
    if !plan.fits_layout(layout) {
        return Err(invalid("plan", "does not fit layout bucket/max_blocks"));
    }

    // Zero first — padded slots (bucket > num_seqs) stay at the
    // sentinel value for context_lens (0) and token_ids (0); slot_mapping
    // gets -1 for padded entries below.
    for x in pinned_host.iter_mut() {
        *x = 0;
    }

    let bucket = layout.bucket as usize;
    let max_blocks = layout.max_blocks as usize;
    let actual = plan.num_seqs as usize;

    // token_ids
    let o = layout.token_ids_off as usize;
    for i in 0..actual {
        pinned_host[o + i] = plan.token_ids[i].0 as i32;
    }

    // positions
    let o = layout.positions_off as usize;
    for i in 0..actual {
        pinned_host[o + i] = plan.positions[i] as i32;
    }

    // context_lens
    let o = layout.context_lens_off as usize;
    for i in 0..actual {
        pinned_host[o + i] = plan.context_lens[i] as i32;
    }

    // block_tables — ALWAYS uploaded. No diff, no patch, no skip.
    let max_in = plan.max_blocks_input as usize;
    let copy_len = max_in.min(max_blocks);
    for s in 0..actual {
        let src_start = s * max_in;
        let dst_start = layout.block_tables_off as usize + s * max_blocks;
        let src_end = (src_start + copy_len).min(plan.block_tables_flat.len());
        for (i, b) in plan.block_tables_flat[src_start..src_end]
            .iter()
            .enumerate()
        {
            pinned_host[dst_start + i] = b.0 as i32;
        }
    }

    // slot_mapping: actual entries from plan, -1 for padding.
    let o = layout.slot_mapping_off as usize;
    for i in 0..actual {
        pinned_host[o + i] = plan.slot_mapping[i];
    }
    for i in actual..bucket {
        pinned_host[o + i] = -1;
    }

    // seq_start_pos: prefix sums + total. For decode it's trivially
    // [0, 1, 2, ..., bucket]; caller fills in the exact values.
    let o = layout.seq_start_pos_off as usize;
    for (i, v) in plan.seq_start_pos.iter().enumerate() {
        if o + i < pinned_host.len() {
            pinned_host[o + i] = *v as i32;
        }
    }

    Ok(())
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::Sampling {
        err: SamplingError::InvalidParams {
            reason: format!("{field}: {reason}"),
        },
        ctx: SampleCtx {
            op: "metadata::upload",
            stream: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::{BlockId, TokenId};

    #[test]
    fn upload_fills_fields_and_pads_slot_mapping_with_minus_one() {
        let layout = MetadataLayout::compute(4, 8);
        let mut buf = vec![0i32; layout.total_elements as usize];
        let token_ids = [TokenId(10), TokenId(20)];
        let positions = [5u32, 6];
        let context_lens = [12u32, 13];
        let block_tables_flat = [BlockId(100), BlockId(101), BlockId(200), BlockId(201)];
        let slot_mapping = [42i32, 43];
        let seq_start_pos = [0u32, 1, 2];
        let plan = BatchPlan {
            num_seqs: 2,
            token_ids: &token_ids,
            positions: &positions,
            context_lens: &context_lens,
            block_tables_flat: &block_tables_flat,
            max_blocks_input: 2,
            slot_mapping: &slot_mapping,
            seq_start_pos: &seq_start_pos,
        };
        upload(&layout, &plan, &mut buf).unwrap();

        // token_ids
        assert_eq!(buf[layout.token_ids_off as usize], 10);
        assert_eq!(buf[layout.token_ids_off as usize + 1], 20);
        assert_eq!(buf[layout.token_ids_off as usize + 2], 0); // pad
                                                               // slot_mapping padding uses -1
        let smo = layout.slot_mapping_off as usize;
        assert_eq!(buf[smo], 42);
        assert_eq!(buf[smo + 1], 43);
        assert_eq!(buf[smo + 2], -1);
        assert_eq!(buf[smo + 3], -1);
        // block_tables: stride = layout.max_blocks, not plan's max_blocks_input
        let bto = layout.block_tables_off as usize;
        assert_eq!(buf[bto], 100);
        assert_eq!(buf[bto + 1], 101);
        // second seq starts at stride 8 (layout.max_blocks)
        assert_eq!(buf[bto + 8], 200);
        assert_eq!(buf[bto + 9], 201);
    }

    #[test]
    fn plan_exceeding_bucket_is_err() {
        let layout = MetadataLayout::compute(1, 8);
        let mut buf = vec![0i32; layout.total_elements as usize];
        let tok = [TokenId(1), TokenId(2)]; // 2 > bucket=1
        let plan = BatchPlan {
            num_seqs: 2,
            token_ids: &tok,
            positions: &[0, 0],
            context_lens: &[1, 1],
            block_tables_flat: &[],
            max_blocks_input: 0,
            slot_mapping: &[0, 0],
            seq_start_pos: &[0, 1, 2],
        };
        assert!(upload(&layout, &plan, &mut buf).is_err());
    }
}
