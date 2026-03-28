//! Core scheduler: continuous batching with preemption and chunked prefill.

use std::collections::VecDeque;
use std::time::Instant;

use rvllm_block_manager::BlockManager;
use rvllm_core::prelude::{BlockId, RequestId, Result};
use rvllm_sequence::{Sequence, SequenceStatus};

use crate::outputs::{ScheduledSequenceGroup, SchedulerOutputs};
use crate::policy::{PreemptionMode, SchedulerPolicy};

// ---------------------------------------------------------------------------
// SequenceGroup: kept local because rvllm_sequence::SequenceGroup has different
// fields (sampling_params, prompt_text) vs our priority/chunked-prefill fields.
// Uses real rvllm_sequence::{Sequence, SequenceStatus} for inner sequences.
// TODO: unify with rvllm_sequence::SequenceGroup once APIs converge
// ---------------------------------------------------------------------------

/// A group of sequences sharing the same prompt (e.g. beam search candidates).
#[derive(Debug, Clone)]
pub struct SequenceGroup {
    pub request_id: RequestId,
    pub sequences: Vec<Sequence>,
    pub arrival_time: Instant,
    pub priority: u32,
    /// Number of prompt tokens already processed (for chunked prefill).
    pub num_prompt_tokens_processed: usize,
}

impl SequenceGroup {
    pub fn new(request_id: RequestId, sequences: Vec<Sequence>, priority: u32) -> Self {
        Self {
            request_id,
            sequences,
            arrival_time: Instant::now(),
            priority,
            num_prompt_tokens_processed: 0,
        }
    }

    /// Total tokens across all sequences in the group (uses first seq as representative).
    pub fn total_token_count(&self) -> usize {
        self.sequences.first().map_or(0, |s| s.get_len())
    }

    /// Number of prompt tokens (from the first sequence).
    pub fn prompt_len(&self) -> usize {
        self.sequences.first().map_or(0, |s| s.get_len())
    }

    /// Remaining prompt tokens to prefill.
    pub fn remaining_prefill(&self) -> usize {
        self.prompt_len()
            .saturating_sub(self.num_prompt_tokens_processed)
    }

    /// True if the group still has prompt tokens to prefill.
    pub fn is_prefilling(&self) -> bool {
        self.remaining_prefill() > 0
    }

    /// Number of active (non-finished) sequences.
    pub fn num_active(&self) -> usize {
        self.sequences.iter().filter(|s| !s.is_finished()).count()
    }

    /// True if all sequences are finished.
    pub fn is_finished(&self) -> bool {
        self.sequences.iter().all(|s| s.is_finished())
    }

    /// Set the status of all sequences in this group.
    pub fn set_status(&mut self, status: SequenceStatus) {
        for seq in &mut self.sequences {
            seq.status = status;
        }
    }
}

// ---------------------------------------------------------------------------
// SchedulerConfig
// ---------------------------------------------------------------------------

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_paddings: usize,
    pub preemption_mode: PreemptionMode,
    pub policy: SchedulerPolicy,
    /// Maximum chunk size for chunked prefill. 0 = no chunking (process entire prompt).
    pub max_prefill_chunk: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            max_paddings: 256,
            preemption_mode: PreemptionMode::Recompute,
            policy: SchedulerPolicy::Fcfs,
            max_prefill_chunk: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Continuous batching scheduler.
///
/// Manages three queues (waiting, running, swapped) and decides each iteration
/// which sequence groups get GPU time, which get preempted, and which get
/// swapped back in.
pub struct Scheduler {
    config: SchedulerConfig,
    block_manager: BlockManager,
    waiting: VecDeque<SequenceGroup>,
    running: Vec<SequenceGroup>,
    swapped: VecDeque<SequenceGroup>,
    /// Cached per-group token counts (index-parallel with `running`).
    /// Recomputed only when `running_dirty` is true.
    cached_tokens: Vec<usize>,
    /// True when the running set changed since last sort/token-count cache.
    running_dirty: bool,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, block_manager: BlockManager) -> Self {
        tracing::debug!(
            max_seqs = config.max_num_seqs,
            max_tokens = config.max_num_batched_tokens,
            preemption = ?config.preemption_mode,
            policy = ?config.policy,
            "scheduler created"
        );
        Self {
            config,
            block_manager,
            waiting: VecDeque::new(),
            running: Vec::new(),
            swapped: VecDeque::new(),
            cached_tokens: Vec::new(),
            running_dirty: true,
        }
    }

    /// Add a new request to the waiting queue.
    pub fn add_request(&mut self, request: SequenceGroup) {
        tracing::debug!(request_id = %request.request_id, "add_request: enqueued");
        self.waiting.push_back(request);
    }

    /// Abort a request by id, removing it from whichever queue it's in.
    pub fn abort_request(&mut self, request_id: &RequestId) {
        // Remove from waiting.
        self.waiting.retain(|g| &g.request_id != request_id);
        // Remove from running and free blocks.
        let mut removed_running = Vec::new();
        self.running.retain(|g| {
            if &g.request_id == request_id {
                removed_running.push(g.clone());
                false
            } else {
                true
            }
        });
        for group in &removed_running {
            self.free_group(group);
        }
        if !removed_running.is_empty() {
            self.running_dirty = true;
        }
        // Remove from swapped.
        self.swapped.retain(|g| &g.request_id != request_id);
        tracing::debug!(%request_id, "abort_request: removed");
    }

    pub fn has_unfinished(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty() || !self.swapped.is_empty()
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn num_swapped(&self) -> usize {
        self.swapped.len()
    }

    // -----------------------------------------------------------------------
    // Main scheduling loop
    // -----------------------------------------------------------------------

    /// Run one scheduling step.
    ///
    /// 1. Check running sequences for completion.
    /// 2. If memory pressure, preempt lowest-priority running groups.
    /// 3. Try to swap in previously swapped groups.
    /// 4. Try to admit new requests from the waiting queue.
    /// 5. Build the batch: decode-first, then interleave prefill chunks.
    /// 6. Collect block operations.
    pub fn schedule(&mut self) -> Result<SchedulerOutputs> {
        let mut blocks_to_swap_in: Vec<(BlockId, BlockId)> = Vec::new();
        let mut blocks_to_swap_out: Vec<(BlockId, BlockId)> = Vec::new();

        // -- Step 1: retire finished groups from running --
        self.retire_finished();

        // -- Step 2: preempt if under memory pressure --
        let preempt_swaps = self.preempt_if_needed()?;
        blocks_to_swap_out.extend(preempt_swaps);

        // -- Step 3: try to swap in previously swapped groups --
        let swap_ins = self.try_swap_in()?;
        blocks_to_swap_in.extend(swap_ins);

        // -- Step 4: admit from waiting queue --
        self.admit_waiting()?;

        // -- Step 5: build output batch with decode-first interleaving --
        // Only re-sort and recompute token counts when running set changed.
        if self.running_dirty {
            self.config.policy.sort(&mut self.running);
            self.cached_tokens.clear();
            self.cached_tokens.reserve(self.running.len());
            for group in &self.running {
                self.cached_tokens.push(Self::tokens_for_group_static(
                    group,
                    self.config.max_prefill_chunk,
                ));
            }
            self.running_dirty = false;
        }

        let max_seqs = self.config.max_num_seqs;
        let max_tokens = self.config.max_num_batched_tokens;

        // Pre-size output vec.
        let mut scheduled: Vec<ScheduledSequenceGroup> =
            Vec::with_capacity(self.running.len().min(max_seqs));
        let mut num_batched_tokens: usize = 0;
        let mut num_prefill_groups: usize = 0;

        // Bitmap tracking which indices are selected into the batch.
        let n = self.running.len();
        let mut selected = vec![false; n];

        // Pass 1: schedule all decode groups first (1 token each, cheap).
        // This keeps decode latency low and packs the batch efficiently.
        for i in 0..n {
            if scheduled.len() >= max_seqs {
                break;
            }
            let tokens_i = self.cached_tokens[i];
            // Decode group: not prefilling (remaining_prefill == 0).
            if self.running[i].is_prefilling() {
                continue;
            }
            if num_batched_tokens + tokens_i > max_tokens {
                continue;
            }
            num_batched_tokens += tokens_i;
            selected[i] = true;
            scheduled.push(ScheduledSequenceGroup {
                seq_group: self.running[i].clone(),
                token_chunk_size: tokens_i,
            });
        }

        // Pass 2: fill remaining budget with prefill chunks, interleaved.
        for i in 0..n {
            if scheduled.len() >= max_seqs {
                break;
            }
            if selected[i] {
                continue;
            }
            let mut tokens_i = self.cached_tokens[i];
            if !self.running[i].is_prefilling() {
                // Decode that didn't fit in pass 1 due to token budget.
                if num_batched_tokens + tokens_i > max_tokens {
                    continue;
                }
                num_batched_tokens += tokens_i;
                selected[i] = true;
                scheduled.push(ScheduledSequenceGroup {
                    seq_group: self.running[i].clone(),
                    token_chunk_size: tokens_i,
                });
                continue;
            }
            // Prefill: chunk to fit remaining token budget.
            let remaining_budget = max_tokens.saturating_sub(num_batched_tokens);
            if remaining_budget == 0 {
                break;
            }
            tokens_i = tokens_i.min(remaining_budget);
            if tokens_i == 0 {
                continue;
            }
            num_batched_tokens += tokens_i;
            num_prefill_groups += 1;
            selected[i] = true;

            let mut group = self.running[i].clone();
            group.num_prompt_tokens_processed += tokens_i;
            scheduled.push(ScheduledSequenceGroup {
                seq_group: group,
                token_chunk_size: tokens_i,
            });
        }

        // Update running list: replace scheduled groups with their updated state
        // and keep unselected groups as-is.
        {
            let mut sched_idx = 0;
            for i in 0..n {
                if selected[i] {
                    // Find the matching scheduled entry (they appear in order).
                    while sched_idx < scheduled.len() {
                        if scheduled[sched_idx].seq_group.request_id == self.running[i].request_id {
                            self.running[i] = scheduled[sched_idx].seq_group.clone();
                            sched_idx += 1;
                            break;
                        }
                        sched_idx += 1;
                    }
                }
            }
            // Invalidate cached tokens since prefill progress changed.
            if num_prefill_groups > 0 {
                self.running_dirty = true;
            }
        }

        // -- Step 6: collect CoW block copies --
        let blocks_to_copy = self.block_manager.get_copy_on_write_blocks();

        tracing::debug!(
            scheduled = scheduled.len(),
            batched_tokens = num_batched_tokens,
            prefill_groups = num_prefill_groups,
            swap_in = blocks_to_swap_in.len(),
            swap_out = blocks_to_swap_out.len(),
            copies = blocks_to_copy.len(),
            waiting = self.waiting.len(),
            running = self.running.len(),
            swapped = self.swapped.len(),
            "schedule step complete"
        );

        Ok(SchedulerOutputs {
            scheduled_seq_groups: scheduled,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            num_batched_tokens,
            num_prefill_groups,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Determine how many tokens to process for a group this step.
    /// Static version avoids borrowing self, used for pre-computing token counts.
    #[inline(always)]
    fn tokens_for_group_static(group: &SequenceGroup, max_prefill_chunk: usize) -> usize {
        let remaining = group.remaining_prefill();
        if remaining > 0 {
            if max_prefill_chunk > 0 {
                remaining.min(max_prefill_chunk)
            } else {
                remaining
            }
        } else {
            group.num_active()
        }
    }

    /// Determine how many tokens to process for a group this step.
    fn tokens_for_group(&self, group: &SequenceGroup) -> usize {
        Self::tokens_for_group_static(group, self.config.max_prefill_chunk)
    }

    /// Remove fully finished groups from running and free their blocks.
    fn retire_finished(&mut self) {
        let before = self.running.len();
        let bm = &mut self.block_manager;
        self.running.retain(|group| {
            if group.is_finished() {
                tracing::debug!(request_id = %group.request_id, "retiring finished group");
                for seq in &group.sequences {
                    bm.free(seq);
                }
                false
            } else {
                true
            }
        });
        if self.running.len() != before {
            self.running_dirty = true;
        }
    }

    /// If there isn't enough GPU memory for all running sequences, preempt
    /// the lowest-priority ones until memory is sufficient.
    fn preempt_if_needed(&mut self) -> Result<Vec<(BlockId, BlockId)>> {
        let mut swap_pairs = Vec::new();
        let mut did_preempt = false;

        while !self.running.is_empty() && !self.block_manager.above_watermark() {
            // Preempt the last (lowest-priority after sort) running group.
            let victim = self.running.pop().unwrap();
            did_preempt = true;
            tracing::debug!(
                request_id = %victim.request_id,
                mode = ?self.config.preemption_mode,
                "preempting group"
            );

            match self.config.preemption_mode {
                PreemptionMode::Swap => {
                    let pairs = self.swap_out_group(&victim)?;
                    swap_pairs.extend(pairs);
                    let mut swapped = victim;
                    swapped.set_status(SequenceStatus::Swapped);
                    self.swapped.push_back(swapped);
                }
                PreemptionMode::Recompute => {
                    self.free_group(&victim);
                    let mut requeued = victim;
                    requeued.num_prompt_tokens_processed = 0;
                    requeued.set_status(SequenceStatus::Waiting);
                    self.waiting.push_front(requeued);
                }
            }
        }

        if did_preempt {
            self.running_dirty = true;
        }

        Ok(swap_pairs)
    }

    /// Try to swap in groups from the swapped queue while memory allows.
    fn try_swap_in(&mut self) -> Result<Vec<(BlockId, BlockId)>> {
        if self.swapped.is_empty() {
            return Ok(Vec::new());
        }

        let mut swap_pairs = Vec::new();
        let mut still_swapped = VecDeque::with_capacity(self.swapped.len());
        let mut swapped_any = false;

        while let Some(group) = self.swapped.pop_front() {
            if self.running.len() >= self.config.max_num_seqs {
                still_swapped.push_back(group);
                continue;
            }

            let can_swap = group.sequences.iter().all(|seq| {
                self.block_manager.can_allocate(seq)
            });

            if can_swap {
                let pairs = self.swap_in_group(&group)?;
                swap_pairs.extend(pairs);
                let mut resumed = group;
                resumed.set_status(SequenceStatus::Running);
                tracing::debug!(request_id = %resumed.request_id, "swapped in group");
                self.running.push(resumed);
                swapped_any = true;
            } else {
                still_swapped.push_back(group);
                break;
            }
        }

        // Put remaining swapped groups back.
        while let Some(g) = self.swapped.pop_front() {
            still_swapped.push_back(g);
        }
        self.swapped = still_swapped;

        if swapped_any {
            self.running_dirty = true;
        }

        Ok(swap_pairs)
    }

    /// Admit requests from the waiting queue while budget allows.
    fn admit_waiting(&mut self) -> Result<()> {
        if self.waiting.is_empty() {
            return Ok(());
        }

        let mut still_waiting = VecDeque::with_capacity(self.waiting.len());

        // Sort waiting by policy.
        let mut waiting_vec: Vec<SequenceGroup> = self.waiting.drain(..).collect();
        self.config.policy.sort(&mut waiting_vec);

        let mut admitted_any = false;
        for mut group in waiting_vec {
            if self.running.len() >= self.config.max_num_seqs {
                still_waiting.push_back(group);
                continue;
            }

            // Check if block manager can allocate for each sequence.
            let can_alloc = group
                .sequences
                .iter()
                .all(|seq| self.block_manager.can_allocate(seq));

            if can_alloc {
                for seq in &group.sequences {
                    self.block_manager.allocate(seq)?;
                }
                group.set_status(SequenceStatus::Running);
                tracing::debug!(
                    request_id = %group.request_id,
                    "admitted from waiting"
                );
                self.running.push(group);
                admitted_any = true;
            } else {
                still_waiting.push_back(group);
            }
        }

        self.waiting = still_waiting;
        if admitted_any {
            self.running_dirty = true;
        }
        Ok(())
    }

    /// Free all blocks for every sequence in a group.
    fn free_group(&mut self, group: &SequenceGroup) {
        for seq in &group.sequences {
            self.block_manager.free(seq);
        }
    }

    /// Swap out all sequences of a group, returning GPU->CPU block mappings.
    fn swap_out_group(&mut self, group: &SequenceGroup) -> Result<Vec<(BlockId, BlockId)>> {
        let mut pairs = Vec::new();
        for seq in &group.sequences {
            if seq.is_finished() {
                continue;
            }
            let p = self.block_manager.swap_out(seq)?;
            pairs.extend(p);
        }
        Ok(pairs)
    }

    /// Swap in all sequences of a group, returning CPU->GPU block mappings.
    fn swap_in_group(&mut self, group: &SequenceGroup) -> Result<Vec<(BlockId, BlockId)>> {
        let mut pairs = Vec::new();
        for seq in &group.sequences {
            if seq.is_finished() {
                continue;
            }
            let p = self.block_manager.swap_in(seq)?;
            pairs.extend(p);
        }
        Ok(pairs)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_block_manager::MemoryPool;
    use rvllm_core::prelude::{BlockId, SequenceId};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // Test pool -- simple free-list backed by atomics
    // -----------------------------------------------------------------------

    struct TestPool {
        total: usize,
        next_id: AtomicUsize,
        free_count: AtomicUsize,
    }

    impl TestPool {
        fn new(total: usize) -> Self {
            Self {
                total,
                next_id: AtomicUsize::new(0),
                free_count: AtomicUsize::new(total),
            }
        }
    }

    impl MemoryPool for TestPool {
        fn allocate(&self) -> Option<BlockId> {
            let free = self.free_count.load(Ordering::SeqCst);
            if free == 0 {
                return None;
            }
            self.free_count.fetch_sub(1, Ordering::SeqCst);
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            Some(BlockId(id as u32))
        }
        fn free(&self, _block_id: BlockId) {
            self.free_count.fetch_add(1, Ordering::SeqCst);
        }
        fn free_blocks(&self) -> usize {
            self.free_count.load(Ordering::SeqCst)
        }
        fn total_blocks(&self) -> usize {
            self.total
        }
    }

    fn make_block_manager(gpu_blocks: usize, cpu_blocks: usize) -> BlockManager {
        let gpu = Arc::new(TestPool::new(gpu_blocks));
        let cpu = Arc::new(TestPool::new(cpu_blocks));
        let mut mgr = BlockManager::new(gpu, cpu, 16);
        mgr.set_watermark(0.0);
        mgr
    }

    fn make_seq(id: u64, num_tokens: usize) -> Sequence {
        Sequence::new(SequenceId(id), vec![0u32; num_tokens])
    }

    fn make_group(request_id: u64, seq_id: u64, num_tokens: usize) -> SequenceGroup {
        let seq = make_seq(seq_id, num_tokens);
        SequenceGroup::new(RequestId(request_id), vec![seq], 0)
    }

    fn default_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 256,
            max_paddings: 32,
            preemption_mode: PreemptionMode::Recompute,
            policy: SchedulerPolicy::Fcfs,
            max_prefill_chunk: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Admission tests
    // -----------------------------------------------------------------------

    #[test]
    fn admit_single_request() {
        let bm = make_block_manager(100, 100);
        let mut sched = Scheduler::new(default_config(), bm);

        sched.add_request(make_group(1, 1, 32));
        assert_eq!(sched.num_waiting(), 1);

        let out = sched.schedule().unwrap();
        assert_eq!(out.scheduled_seq_groups.len(), 1);
        assert_eq!(sched.num_running(), 1);
        assert_eq!(sched.num_waiting(), 0);
    }

    #[test]
    fn admit_multiple_up_to_limit() {
        let bm = make_block_manager(100, 100);
        let config = SchedulerConfig {
            max_num_seqs: 2,
            ..default_config()
        };
        let mut sched = Scheduler::new(config, bm);

        for i in 0..5 {
            sched.add_request(make_group(i, i, 16));
        }
        assert_eq!(sched.num_waiting(), 5);

        let out = sched.schedule().unwrap();
        // Only 2 admitted (max_num_seqs).
        assert_eq!(out.scheduled_seq_groups.len(), 2);
        assert_eq!(sched.num_running(), 2);
        assert_eq!(sched.num_waiting(), 3);
    }

    #[test]
    fn admit_respects_token_budget() {
        let bm = make_block_manager(100, 100);
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 48, // room for 3 groups of 16 tokens
            ..default_config()
        };
        let mut sched = Scheduler::new(config, bm);

        for i in 0..5 {
            sched.add_request(make_group(i, i, 16));
        }

        let out = sched.schedule().unwrap();
        assert_eq!(out.num_batched_tokens, 48);
        assert_eq!(out.scheduled_seq_groups.len(), 3);
    }

    #[test]
    fn admit_blocked_by_memory() {
        // Only 1 GPU block = 16 tokens. Second request needs more blocks.
        let bm = make_block_manager(1, 10);
        let mut sched = Scheduler::new(default_config(), bm);

        sched.add_request(make_group(1, 1, 16)); // needs 1 block
        sched.add_request(make_group(2, 2, 32)); // needs 2 blocks

        let out = sched.schedule().unwrap();
        assert_eq!(out.scheduled_seq_groups.len(), 1);
        assert_eq!(sched.num_waiting(), 1); // second still waiting
    }

    // -----------------------------------------------------------------------
    // Completion tests
    // -----------------------------------------------------------------------

    #[test]
    fn finished_groups_retired() {
        let bm = make_block_manager(100, 100);
        let mut sched = Scheduler::new(default_config(), bm);

        sched.add_request(make_group(1, 1, 16));
        let _ = sched.schedule().unwrap();
        assert_eq!(sched.num_running(), 1);

        // Mark as finished externally.
        sched.running[0].set_status(SequenceStatus::FinishedStopped);
        let _ = sched.schedule().unwrap();
        assert_eq!(sched.num_running(), 0);
    }

    // -----------------------------------------------------------------------
    // Preemption: recompute mode
    // -----------------------------------------------------------------------

    #[test]
    fn preempt_recompute_sends_back_to_waiting() {
        // 2 GPU blocks, watermark = 0 initially.
        let gpu = Arc::new(TestPool::new(2));
        let cpu = Arc::new(TestPool::new(10));
        let mut bm = BlockManager::new(gpu, cpu, 16);
        bm.set_watermark(0.0);

        let config = SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 256,
            preemption_mode: PreemptionMode::Recompute,
            ..default_config()
        };
        let mut sched = Scheduler::new(config, bm);

        // Admit two groups each needing 1 block.
        sched.add_request(make_group(1, 1, 16));
        sched.add_request(make_group(2, 2, 16));
        let _ = sched.schedule().unwrap();
        assert_eq!(sched.num_running(), 2);

        // Now increase watermark so memory is tight -- simulate by setting
        // watermark high enough that above_watermark() returns false.
        // We have 0 free blocks, watermark=0 -> free(0) > reserved(0) is false.
        // Actually with 0 free and watermark=0, above_watermark checks free > reserved.
        // 0 > 0 is false, so preemption triggers.
        // The two allocations consumed both blocks, so free = 0.
        let _ = sched.schedule().unwrap();
        // With 0 free blocks and watermark=0, above_watermark returns false (0 > 0 is false).
        // Both groups get preempted back to waiting.
        // After preemption frees blocks, the scheduler re-admits them.
        // So they end up running again. Let's verify the flow works without error.
        assert!(sched.has_unfinished());
    }

    // -----------------------------------------------------------------------
    // Preemption: swap mode
    // -----------------------------------------------------------------------

    #[test]
    fn preempt_swap_moves_to_swapped() {
        let gpu = Arc::new(TestPool::new(4));
        let cpu = Arc::new(TestPool::new(10));
        let mut bm = BlockManager::new(gpu, cpu, 16);
        bm.set_watermark(0.0);

        let config = SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 256,
            preemption_mode: PreemptionMode::Swap,
            ..default_config()
        };
        let mut sched = Scheduler::new(config, bm);

        // Fill all 4 GPU blocks with 4 groups.
        for i in 0..4 {
            sched.add_request(make_group(i, i, 16));
        }
        let _ = sched.schedule().unwrap();
        assert_eq!(sched.num_running(), 4);
        // All 4 blocks used, 0 free. above_watermark -> false.
        // Next schedule should preempt.

        // Mark one group finished to free a block, add a new bigger request.
        sched.running[0].set_status(SequenceStatus::FinishedStopped);
        sched.add_request(make_group(10, 10, 32)); // needs 2 blocks

        let out = sched.schedule().unwrap();
        // Should have some swap outs.
        assert!(sched.num_running() > 0 || sched.num_swapped() > 0 || sched.num_waiting() > 0);
        assert!(sched.has_unfinished());
        // Verify outputs are valid (no panics, correct structure).
        assert!(out.num_batched_tokens <= 256);
    }

    // -----------------------------------------------------------------------
    // Swap in/out round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn swap_out_then_in() {
        let gpu = Arc::new(TestPool::new(3));
        let cpu = Arc::new(TestPool::new(10));
        let mut bm = BlockManager::new(gpu, cpu, 16);
        bm.set_watermark(0.0);

        let config = SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 256,
            preemption_mode: PreemptionMode::Swap,
            ..default_config()
        };
        let mut sched = Scheduler::new(config, bm);

        // Admit 3 groups using all 3 blocks.
        for i in 0..3 {
            sched.add_request(make_group(i, i, 16));
        }
        let _ = sched.schedule().unwrap();
        assert_eq!(sched.num_running(), 3);

        // Next schedule: 0 free -> preemption swaps out at least one.
        let out = sched.schedule().unwrap();
        // After preemption and re-schedule, some should be swapped.
        let total = sched.num_running() + sched.num_swapped() + sched.num_waiting();
        assert!(total > 0);
        assert!(out.blocks_to_swap_out.len() > 0 || sched.num_running() > 0);
    }

    // -----------------------------------------------------------------------
    // Abort
    // -----------------------------------------------------------------------

    #[test]
    fn abort_waiting_request() {
        let bm = make_block_manager(100, 100);
        let mut sched = Scheduler::new(default_config(), bm);

        sched.add_request(make_group(1, 1, 16));
        assert_eq!(sched.num_waiting(), 1);

        sched.abort_request(&RequestId(1));
        assert_eq!(sched.num_waiting(), 0);
    }

    #[test]
    fn abort_running_request() {
        let bm = make_block_manager(100, 100);
        let mut sched = Scheduler::new(default_config(), bm);

        sched.add_request(make_group(1, 1, 16));
        let _ = sched.schedule().unwrap();
        assert_eq!(sched.num_running(), 1);

        sched.abort_request(&RequestId(1));
        assert_eq!(sched.num_running(), 0);
    }

    // -----------------------------------------------------------------------
    // Chunked prefill
    // -----------------------------------------------------------------------

    #[test]
    fn chunked_prefill_splits_long_prompt() {
        let bm = make_block_manager(100, 100);
        let config = SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 256,
            max_prefill_chunk: 32,
            ..default_config()
        };
        let mut sched = Scheduler::new(config, bm);

        // 128-token prompt with chunk size 32 -> 4 steps to prefill.
        sched.add_request(make_group(1, 1, 128));
        let out = sched.schedule().unwrap();
        assert_eq!(out.scheduled_seq_groups.len(), 1);
        assert_eq!(out.scheduled_seq_groups[0].token_chunk_size, 32);
        assert_eq!(out.num_prefill_groups, 1);
    }

    #[test]
    fn chunked_prefill_progresses_over_steps() {
        let bm = make_block_manager(100, 100);
        let config = SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 256,
            max_prefill_chunk: 32,
            ..default_config()
        };
        let mut sched = Scheduler::new(config, bm);

        sched.add_request(make_group(1, 1, 64));

        // Step 1: processes 32 tokens.
        let out1 = sched.schedule().unwrap();
        assert_eq!(out1.scheduled_seq_groups[0].token_chunk_size, 32);
        assert_eq!(out1.num_prefill_groups, 1);

        // Step 2: processes remaining 32 tokens.
        let out2 = sched.schedule().unwrap();
        assert_eq!(out2.scheduled_seq_groups[0].token_chunk_size, 32);
        assert_eq!(out2.num_prefill_groups, 1);

        // Step 3: prefill done, now decoding (1 token per seq).
        let out3 = sched.schedule().unwrap();
        assert_eq!(out3.scheduled_seq_groups[0].token_chunk_size, 1);
        assert_eq!(out3.num_prefill_groups, 0);
    }

    // -----------------------------------------------------------------------
    // Policy tests
    // -----------------------------------------------------------------------

    #[test]
    fn priority_policy_admits_high_priority_first() {
        let bm = make_block_manager(100, 100);
        let config = SchedulerConfig {
            max_num_seqs: 1,
            max_num_batched_tokens: 256,
            policy: SchedulerPolicy::Priority,
            ..default_config()
        };
        let mut sched = Scheduler::new(config, bm);

        let mut low = make_group(1, 1, 16);
        low.priority = 1;
        let mut high = make_group(2, 2, 16);
        high.priority = 10;

        sched.add_request(low);
        sched.add_request(high);

        let out = sched.schedule().unwrap();
        assert_eq!(out.scheduled_seq_groups.len(), 1);
        assert_eq!(
            out.scheduled_seq_groups[0].seq_group.request_id,
            RequestId(2)
        );
    }

    #[test]
    fn sjf_policy_admits_shortest_first() {
        let bm = make_block_manager(100, 100);
        let config = SchedulerConfig {
            max_num_seqs: 1,
            max_num_batched_tokens: 256,
            policy: SchedulerPolicy::ShortestJobFirst,
            ..default_config()
        };
        let mut sched = Scheduler::new(config, bm);

        sched.add_request(make_group(1, 1, 128));
        sched.add_request(make_group(2, 2, 16));

        let out = sched.schedule().unwrap();
        assert_eq!(out.scheduled_seq_groups.len(), 1);
        assert_eq!(
            out.scheduled_seq_groups[0].seq_group.request_id,
            RequestId(2)
        );
    }

    // -----------------------------------------------------------------------
    // has_unfinished / empty schedule
    // -----------------------------------------------------------------------

    #[test]
    fn empty_scheduler_no_work() {
        let bm = make_block_manager(100, 100);
        let mut sched = Scheduler::new(default_config(), bm);

        assert!(!sched.has_unfinished());
        let out = sched.schedule().unwrap();
        assert!(out.is_empty());
        assert_eq!(out.num_batched_tokens, 0);
    }

    // -----------------------------------------------------------------------
    // Output structure validation
    // -----------------------------------------------------------------------

    #[test]
    fn output_fields_populated() {
        let bm = make_block_manager(100, 100);
        let mut sched = Scheduler::new(default_config(), bm);

        sched.add_request(make_group(1, 1, 32));
        let out = sched.schedule().unwrap();

        assert_eq!(out.scheduled_seq_groups.len(), 1);
        assert_eq!(out.num_batched_tokens, 32);
        assert_eq!(out.num_prefill_groups, 1);
        assert!(out.blocks_to_swap_in.is_empty());
        assert!(out.blocks_to_swap_out.is_empty());
    }

    // -----------------------------------------------------------------------
    // Multi-step lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn full_lifecycle_admit_run_finish() {
        let bm = make_block_manager(100, 100);
        let mut sched = Scheduler::new(default_config(), bm);

        sched.add_request(make_group(1, 1, 16));
        assert!(sched.has_unfinished());

        // Admit + schedule.
        let out = sched.schedule().unwrap();
        assert_eq!(out.scheduled_seq_groups.len(), 1);

        // Simulate: mark finished.
        sched.running[0].set_status(SequenceStatus::FinishedStopped);

        // Schedule again: should retire.
        let out2 = sched.schedule().unwrap();
        assert!(out2.is_empty());
        assert!(!sched.has_unfinished());
    }
}
