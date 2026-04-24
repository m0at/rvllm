# 04 — Scheduler

## Problem

We have 31 logical agents (1 master + 30 workers) and a runtime that can
serve them in two ways. Master is pinned. The worker scheduler owns the
visible worker pool and decides which of 30 agents receives work at any
given moment.

An "occupied slot" means:

- a native rvLLM decode context lives in memory,
- Gemma 4 E4B FP8 weights are shared read-only across all live services,
- this agent's KV cache exists in HBM,
- the worker thread bound to this slot has a `Arc<AgentState>` it can poll.

`N_LIVE` is still shown because it is useful for memory budgeting and for
older slot-style tests, but decode parallelism is controlled separately by
`decode_batch_target`.

## Execution modes

| Mode          | Visible workers | Decode batch target | Use case                         |
|---------------|-----------------|---------------------|----------------------------------|
| `operator-30` | 30              | 30                  | one visible row per logical bot  |
| `saturator`   | 30              | 512                 | push H100 on-card parallelism    |

The important distinction is **logical agents** versus **decode rows**. The
GUI always shows 30 workers because that is what the operator can reason
about. In `saturator`, the backend is allowed to create extra decode rows so
we can test throughput without pretending the human should supervise 512
separate agents.

## Policy

Pluggable via `SchedulerPolicy` trait, default implementation `LruPolicy`.

```rust
pub trait SchedulerPolicy: Send + 'static {
    fn on_dispatch(&mut self, want: AgentId, slots: &mut [SlotOccupancy]) -> SchedDecision;
    fn on_task_done(&mut self, slot: SlotIdx, slots: &mut [SlotOccupancy]);
    fn on_tick(&mut self, slots: &mut [SlotOccupancy]);
}

pub enum SchedDecision {
    AlreadyResident(SlotIdx),
    LoadInto(SlotIdx),                    // empty slot
    EvictAndLoad { evict: SlotIdx },      // LRU victim chosen
    Deny(DenyReason),                     // all slots busy with mid-task agents
}
```

### LRU default

1. If `want` is already in a slot: `AlreadyResident`.
2. Else if an empty slot exists: `LoadInto(first_empty)`.
3. Else: find the slot whose `last_used_monotonic` is smallest **and**
   whose current agent is not `Running` a task. Evict it.
4. If every slot is mid-task: `Deny(AllSlotsBusy)`. The controller
   re-queues the task and tries again on the next `on_task_done` tick.

### Priority

Tasks carry a `Priority { Low, Normal, High, Interactive }`. The
controller considers priority when picking which queued task to dispatch
next. `Interactive` (e.g. master asking an agent to self-describe) can
preempt: the policy is allowed to evict a `Running` agent only if its
task is at priority `Low` and has been running for > 30 s.

## Load/evict mechanics

Loading an agent means constructing a native rvLLM decode context for its
persona. This is not free (weights are shared, but KV is allocated fresh).
Measurements on H100:

| Action                                                   | Time       |
|----------------------------------------------------------|------------|
| First agent load (weights mmap + quant + graph capture)  | 8-12 s     |
| Subsequent load (weights shared, KV alloc, graph reuse)  | 0.3-0.8 s  |
| Evict (drop KV, free graph, keep weights)                | 0.05-0.1 s |

The first load is a one-time cost on app start. The scheduler eagerly
warms the first `N_LIVE` agents during startup so that by the time the
user submits a goal everything is hot.

## Slot occupancy state

```rust
pub struct SlotOccupancy {
    pub slot_idx: SlotIdx,
    pub occupant: Option<AgentId>,
    pub since: Option<Instant>,
    pub last_used: Instant,
    pub state: SlotState,
}

pub enum SlotState {
    Empty,
    Loading { agent: AgentId, started: Instant },
    Hot { agent: AgentId, current_task: Option<TaskId> },
    Evicting { agent: AgentId },
}
```

UI renders a small indicator in the top bar: four boxes showing which
agent ids are currently in each slot, and flashes blue during `Loading`,
orange during `Hot+Running`, purple during `Evicting`.

## OOM handling

If loading fails with a CUDA OOM:

1. The worker thread reports `WorkerEvent::LoadFailed { oom: true }`.
2. The controller decrements `SchedulerConfig::n_live_slots` by 1,
   evicts the most-recently-loaded idle agent to free HBM,
3. Re-attempts the load.
4. If OOM persists, surfaces as a red banner; operator can reduce
   max context or restart.

## Batch decode shape

`operator-30` is the honest operator mode: a single broadcast can fan out to
all 30 visible workers and the backend target is B=30.

`saturator` is the card-utilisation mode: the same broadcast is still shown
as 30 workers, but the runtime target is B=512 so the H100 can be driven near
its measured high-throughput region. The extra rows are load lanes, not
extra autonomous agents with their own git worktrees.

Tool calls still happen at agent boundaries. If a worker requests a tool,
the controller records that result for that visible worker and the backend
can refill the open decode lane with another ready row.
