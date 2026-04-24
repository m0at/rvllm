// Live-slot scheduler. See docs/04_SCHEDULER.md.

pub mod lru;
pub use lru::LruPolicy;

use crate::ipc::{AgentId, SlotIdx};
use crate::state::SlotOccupancy;
#[cfg(test)]
use crate::state::SlotState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedDecision {
    AlreadyResident(SlotIdx),
    LoadInto(SlotIdx),
    EvictAndLoad { evict: SlotIdx },
    Deny(DenyReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenyReason {
    AllSlotsBusy,
    AgentPinnedElsewhere,
}

pub trait SchedulerPolicy: Send + 'static {
    fn on_dispatch(&mut self, want: AgentId, slots: &[SlotOccupancy]) -> SchedDecision;
    #[allow(dead_code)]
    fn on_task_done(&mut self, slot: SlotIdx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::AgentId;

    fn empty_slots(n: usize) -> Vec<SlotOccupancy> {
        (0..n).map(SlotOccupancy::empty).collect()
    }

    #[test]
    fn lru_empty_slot_first() {
        let mut p = LruPolicy::new();
        let slots = empty_slots(4);
        let a = AgentId::new();
        assert_eq!(p.on_dispatch(a, &slots), SchedDecision::LoadInto(SlotIdx(0)));
    }

    #[test]
    fn lru_already_resident() {
        let mut p = LruPolicy::new();
        let a = AgentId::new();
        let mut slots = empty_slots(4);
        slots[2].occupant = Some(a);
        slots[2].state = SlotState::Hot;
        p.mark_touched(SlotIdx(2));
        assert_eq!(p.on_dispatch(a, &slots), SchedDecision::AlreadyResident(SlotIdx(2)));
    }

    #[test]
    fn lru_eviction_order() {
        let mut p = LruPolicy::new();
        let a = AgentId::new();
        let b = AgentId::new();
        let c = AgentId::new();
        let want = AgentId::new();
        let mut slots = empty_slots(3);
        slots[0].occupant = Some(a);
        slots[0].state = SlotState::Hot;
        slots[1].occupant = Some(b);
        slots[1].state = SlotState::Hot;
        slots[2].occupant = Some(c);
        slots[2].state = SlotState::Hot;
        p.mark_touched(SlotIdx(0));
        p.mark_touched(SlotIdx(2));
        p.mark_touched(SlotIdx(1));
        // Oldest touch is slot 0
        assert_eq!(
            p.on_dispatch(want, &slots),
            SchedDecision::EvictAndLoad { evict: SlotIdx(0) }
        );
    }
}
