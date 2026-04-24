// Pure-Rust dispatch policy: given a pending task and the current app state,
// pick the best worker. See docs/02_MASTER_AGENT.md §"Dispatch policy".

use crate::ipc::{AgentId, TaskId};
use crate::state::{AgentLifecycle, AppState};

pub fn choose_worker(state: &AppState, task_id: TaskId) -> Option<AgentId> {
    let task = state.tasks.tasks.get(&task_id)?;

    // 1. Persona match + non-failed + non-paused.
    let eligible: Vec<&crate::state::AgentState> = state
        .agents
        .iter()
        .filter(|a| {
            a.persona == task.persona
                && a.lifecycle != AgentLifecycle::Failed
                && a.lifecycle != AgentLifecycle::Paused
                && a.current_task.is_none()
        })
        .collect();

    // 2. Prefer already-resident.
    if let Some(a) = eligible.iter().find(|a| a.metrics.hbm_resident) {
        return Some(a.id);
    }

    // 3. Else: least-loaded (fewest recent iterations).
    if let Some(a) = eligible
        .iter()
        .min_by_key(|a| (a.metrics.tasks_accepted + a.metrics.tasks_rejected, a.display_index))
    {
        return Some(a.id);
    }

    // 4. Fallback: any idle agent (persona mismatch), mark persona was
    //    dynamic-override. (For v0.1 we just pick the first idle any-persona.)
    state
        .agents
        .iter()
        .find(|a| {
            a.lifecycle == AgentLifecycle::Idle
                && a.current_task.is_none()
                && a.lifecycle != AgentLifecycle::Failed
        })
        .map(|a| a.id)
}
