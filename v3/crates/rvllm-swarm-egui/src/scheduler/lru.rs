// Least-Recently-Used slot policy.

use std::time::Instant;

use super::{DenyReason, SchedDecision, SchedulerPolicy};
use crate::ipc::{AgentId, SlotIdx};
use crate::state::{SlotOccupancy, SlotState};

pub struct LruPolicy {
    touched: Vec<Option<Instant>>, // per-slot last touch
}

impl LruPolicy {
    pub fn new() -> Self {
        Self {
            touched: Vec::new(),
        }
    }

    pub fn mark_touched(&mut self, slot: SlotIdx) {
        self.ensure_len(slot.0 + 1);
        self.touched[slot.0] = Some(Instant::now());
    }

    fn ensure_len(&mut self, n: usize) {
        if self.touched.len() < n {
            self.touched.resize(n, None);
        }
    }
}

impl Default for LruPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulerPolicy for LruPolicy {
    fn on_dispatch(&mut self, want: AgentId, slots: &[SlotOccupancy]) -> SchedDecision {
        self.ensure_len(slots.len());

        // 1. Already resident?
        if let Some((i, _)) = slots
            .iter()
            .enumerate()
            .find(|(_, s)| s.occupant == Some(want) && s.state == SlotState::Hot)
        {
            self.mark_touched(SlotIdx(i));
            return SchedDecision::AlreadyResident(SlotIdx(i));
        }

        // 2. Empty slot?
        if let Some((i, _)) = slots
            .iter()
            .enumerate()
            .find(|(_, s)| s.state == SlotState::Empty)
        {
            return SchedDecision::LoadInto(SlotIdx(i));
        }

        // 3. Evict the least-recently-touched Hot slot. The controller is
        //    responsible for pre-filtering pinned occupants before calling
        //    into this policy.
        let mut victim: Option<(SlotIdx, Instant)> = None;
        for (i, s) in slots.iter().enumerate() {
            if s.state == SlotState::Hot {
                let t = self.touched.get(i).copied().flatten()
                    .unwrap_or_else(Instant::now);
                match victim {
                    None => victim = Some((SlotIdx(i), t)),
                    Some((_, best)) if t < best => victim = Some((SlotIdx(i), t)),
                    _ => {}
                }
            }
        }

        match victim {
            Some((idx, _)) => SchedDecision::EvictAndLoad { evict: idx },
            None => SchedDecision::Deny(DenyReason::AllSlotsBusy),
        }
    }

    fn on_task_done(&mut self, slot: SlotIdx) {
        self.mark_touched(slot);
    }
}
