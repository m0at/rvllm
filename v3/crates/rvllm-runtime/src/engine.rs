//! v3 Engine: type-state `step_launch` → `PendingStep::collect`.
//!
//! `step_launch` returns a `PendingStep<'e>` that borrows `&mut Engine`.
//! The only way to drain the step is `PendingStep::collect(self)`,
//! which consumes self and releases the borrow. The borrow checker
//! makes "second launch while ticket is live" a compile error; the
//! `#[must_use]` lint catches silent drops; `Drop` debug_asserts so a
//! mis-use panics in tests rather than silently auto-collecting.
//!
//! There is ONE codepath. Graph capture/replay is an implementation
//! detail inside `step_launch`.

use rvllm_core::{ReqId, Result, TokenId};

use crate::scheduler::{BatchPlan, Scheduler};

/// Output of one step: (request id, new token, finished flag).
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub req_id: ReqId,
    pub new_token: TokenId,
    pub finished: bool,
}

pub struct Engine {
    pub scheduler: Scheduler,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
        }
    }

    pub fn has_pending_work(&self) -> bool {
        self.scheduler.num_alive() > 0
    }

    /// Launch one step. Returns a ticket that must be `collect()`ed.
    /// The ticket borrows `&mut self`, so a second `step_launch` cannot
    /// start while it is live.
    pub fn step_launch(&mut self) -> Result<PendingStep<'_>> {
        let plan = self.scheduler.schedule();
        // Phase D wiring: enqueue kernels onto the stream for this plan.
        Ok(PendingStep {
            engine: self,
            plan: Some(plan),
        })
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

/// Consume-once ticket returned by `step_launch`. `#[must_use]` catches
/// silent drops; `Drop` additionally panics in debug if the caller
/// forgets to call `collect`. No auto-collect fallback.
#[must_use = "PendingStep must be collect()-ed; silent drop loses the step's scheduler output"]
pub struct PendingStep<'e> {
    engine: &'e mut Engine,
    /// Holds `Some(plan)` until `collect` takes it.
    /// `Drop` asserts it was taken (i.e. collect ran).
    plan: Option<BatchPlan>,
}

impl<'e> PendingStep<'e> {
    pub fn plan(&self) -> Option<&BatchPlan> {
        self.plan.as_ref()
    }

    /// Drain the launched step. Consumes self so the engine borrow is
    /// released on return. Phase D reads DtoH, commits to scheduler.
    pub fn collect(mut self) -> Result<Vec<StepOutput>> {
        let _plan = self.plan.take().expect("PendingStep::collect called twice");
        let _engine = &mut *self.engine;
        // Phase D: decode DtoH, read tokens, scheduler.commit_decode,
        // return StepOutputs.
        Ok(Vec::new())
    }
}

impl<'e> Drop for PendingStep<'e> {
    fn drop(&mut self) {
        // `collect` sets `plan` to None. Dropping with Some means the
        // caller silently dropped the ticket — programmer error.
        debug_assert!(
            self.plan.is_none(),
            "PendingStep dropped without collect(); scheduler output leaked."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched_state::Request;
    use rvllm_core::{ReqId, TokenId};

    #[test]
    fn empty_engine_has_no_pending_work() {
        let e = Engine::new();
        assert!(!e.has_pending_work());
    }

    #[test]
    fn launch_then_collect_releases_borrow_for_next_launch() {
        let mut e = Engine::new();
        e.scheduler
            .enqueue(Request::new(ReqId(1), vec![TokenId(0)], 1));
        assert!(e.has_pending_work());
        let t = e.step_launch().unwrap();
        let _outputs = t.collect().unwrap();
        // Ticket consumed; engine borrow released; can launch again.
        let t2 = e.step_launch().unwrap();
        let _ = t2.collect().unwrap();
    }
}
