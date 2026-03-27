//! Scheduling policies and preemption modes.

use crate::scheduler::SequenceGroup;

/// How to handle preempted sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreemptionMode {
    /// Swap KV-cache blocks to CPU; resume later without recomputation.
    Swap,
    /// Discard KV-cache; recompute from prompt when resumed.
    Recompute,
}

impl Default for PreemptionMode {
    fn default() -> Self {
        Self::Recompute
    }
}

/// Policy for ordering the waiting queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulerPolicy {
    /// First-come, first-served (arrival order).
    Fcfs,
    /// Higher priority value runs first.
    Priority,
    /// Shortest prompt first (fewest total tokens).
    ShortestJobFirst,
}

impl Default for SchedulerPolicy {
    fn default() -> Self {
        Self::Fcfs
    }
}

impl SchedulerPolicy {
    /// Sort a mutable slice of sequence groups according to this policy.
    /// After sorting, index 0 is the highest-priority (should be scheduled first).
    pub fn sort(&self, groups: &mut [SequenceGroup]) {
        match self {
            Self::Fcfs => {
                groups.sort_by_key(|g| g.arrival_time);
            }
            Self::Priority => {
                // Higher priority first (descending).
                groups.sort_by(|a, b| b.priority.cmp(&a.priority));
            }
            Self::ShortestJobFirst => {
                groups.sort_by_key(|g| g.total_token_count());
            }
        }
    }
}
