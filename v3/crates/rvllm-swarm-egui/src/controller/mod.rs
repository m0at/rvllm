// The controller thread: single owner of the task queue, the journal,
// and dispatch. See docs/01_ARCHITECTURE.md + docs/07_TASK_GRAPH.md.

pub mod decomposer;
pub mod dispatch;
pub mod journal;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use parking_lot::RwLock;

use crate::ipc::{
    now_ms, AgentId, AppEvent, Cmd, Goal, GoalId, JournalPayload, JournalRecord, LogLevel, LogLine,
    MasterActivity, MasterActivityKind, SlotOccupancyReport, SlotStateKind, Task, TaskId,
    TaskOutcome, TaskResult, PROTOCOL_VERSION,
};
use crate::scheduler::{DenyReason, LruPolicy, SchedDecision, SchedulerPolicy};
use crate::state::history::{HistoryEntry, HistoryKind};
use crate::state::{AgentLifecycle, AppState, BackendKind, GoalStatus, Persona, Priority, SlotState, TaskStatus};
use crate::worker::mock::MockBackend;

pub struct ControllerHandle {
    pub state: Arc<RwLock<AppState>>,
    pub cmd_tx: Sender<Cmd>,
    #[allow(dead_code)]
    pub event_rx: Receiver<AppEvent>,
}

pub fn spawn(repo_root: PathBuf, backend_kind: BackendKind) -> ControllerHandle {
    let state = Arc::new(RwLock::new(AppState::bootstrap(repo_root.clone(), backend_kind)));
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<Cmd>();
    let (event_tx, event_rx) = crossbeam_channel::bounded::<AppEvent>(4096);

    let state_clone = Arc::clone(&state);
    let journal_path = repo_root.join(".swarm").join("tasks.jsonl");
    std::thread::Builder::new()
        .name("swarm-controller".into())
        .spawn(move || {
            match journal::open_for_append(&journal_path) {
                Ok(j) => {
                    let mut ctrl = Controller::new(state_clone, event_tx, cmd_rx, j);
                    ctrl.run();
                }
                Err(e) => {
                    let _ = event_tx_log_err(
                        &state_clone,
                        &format!("journal open failed: {e}"),
                    );
                    tracing::error!("controller failed to start: {e}");
                }
            }
        })
        .expect("failed to spawn controller thread");

    ControllerHandle {
        state,
        cmd_tx,
        event_rx,
    }
}

fn event_tx_log_err(state: &Arc<RwLock<AppState>>, msg: &str) -> anyhow::Result<()> {
    let mut g = state.write();
    g.log.push(LogLine {
        ts_ms: now_ms(),
        level: LogLevel::Error,
        source: "controller".into(),
        message: msg.to_owned(),
    });
    Ok(())
}

// --- Controller inner -------------------------------------------------------

struct Controller {
    state: Arc<RwLock<AppState>>,
    events: Sender<AppEvent>,
    cmds: Receiver<Cmd>,
    journal: journal::Journal,
    policy: LruPolicy,
    backend: MockBackend,
    last_tick: Instant,
}

struct MockBroadcastPlan {
    goal_id: GoalId,
    response: String,
    prompt_tokens: u64,
    response_tokens: u64,
    diff_summary: String,
    agents: Vec<(AgentId, u8, Persona)>,
    tasks: Vec<Task>,
}

impl Controller {
    fn new(
        state: Arc<RwLock<AppState>>,
        events: Sender<AppEvent>,
        cmds: Receiver<Cmd>,
        journal: journal::Journal,
    ) -> Self {
        // Replay journal into the task graph.
        let records = journal.replay_all();
        {
            let mut g = state.write();
            for r in &records {
                apply_journal_to_state(&mut g, r);
            }
            restore_master_state(&mut g);
            g.log.push(LogLine {
                ts_ms: now_ms(),
                level: LogLevel::Info,
                source: "journal".into(),
                message: format!("replayed {} records", records.len()),
            });
        }

        Self {
            state,
            events,
            cmds,
            journal,
            policy: LruPolicy::new(),
            backend: MockBackend::new(),
            last_tick: Instant::now(),
        }
    }

    fn run(&mut self) {
        let tick_every = Duration::from_millis(250);
        let mut decomposer_pending: Option<GoalId> = None;

        loop {
            // Pull at most one command before doing work, or time out.
            let got_cmd = match self.cmds.recv_timeout(tick_every) {
                Ok(cmd) => {
                    self.handle_cmd(cmd, &mut decomposer_pending);
                    true
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => false,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    tracing::info!("controller: cmd channel disconnected, exiting");
                    return;
                }
            };

            // Work: decompose pending goal, dispatch runnable tasks, tick.
            if let Some(g) = decomposer_pending.take() {
                self.run_decomposition(g);
            }
            self.dispatch_one_round();
            if self.last_tick.elapsed() >= Duration::from_millis(500) {
                self.tick();
                self.last_tick = Instant::now();
            }
            // Avoid busy-spinning if idle.
            if !got_cmd {
                std::thread::yield_now();
            }
        }
    }

    // --- Cmd handling -------------------------------------------------------

    fn handle_cmd(&mut self, cmd: Cmd, decomposer_pending: &mut Option<GoalId>) {
        match cmd {
            Cmd::SubmitGoal { text } => {
                let goal = Goal {
                    id: GoalId::new(),
                    submitted_by: "human".into(),
                    text,
                    created_at_ms: now_ms(),
                    plan_summary: None,
                    status: GoalStatus::Decomposing,
                };
                self.journal_write(JournalPayload::GoalCreated { goal: goal.clone() });
                {
                    let mut g = self.state.write();
                    g.tasks.goal_order.push(goal.id);
                    g.tasks.goals.insert(goal.id, goal.clone());
                    g.master.current_goal = Some(goal.id);
                    g.master.decomposition_in_flight = true;
                    g.log.push(LogLine {
                        ts_ms: now_ms(),
                        level: LogLevel::Info,
                        source: "master".into(),
                        message: format!("goal submitted: {}", summarise(&goal.text, 80)),
                    });
                }
                let _ = self.events.try_send(AppEvent::GoalCreated(goal.clone()));
                let _ = self.events.try_send(AppEvent::MasterActivity(MasterActivity {
                    kind: MasterActivityKind::Decomposing,
                    detail: summarise(&goal.text, 80),
                }));
                *decomposer_pending = Some(goal.id);
            }
            Cmd::SubmitBroadcast { text } => {
                self.run_broadcast(text);
            }
            Cmd::CancelGoal { goal } => {
                self.cancel_goal(goal);
            }
            Cmd::AdjustScheduler { n_live } => {
                let mut g = self.state.write();
                g.settings.scheduler.n_live_slots = n_live.clamp(1, 512);
                // Resize slots vector conservatively.
                while g.slots.len() < g.settings.scheduler.n_live_slots {
                    let idx = g.slots.len();
                    g.slots.push(crate::state::SlotOccupancy::empty(idx));
                }
                while g.slots.len() > g.settings.scheduler.n_live_slots {
                    // Best-effort drop of trailing empty slots.
                    if g.slots.last().map(|s| s.state == SlotState::Empty).unwrap_or(false) {
                        g.slots.pop();
                    } else {
                        break;
                    }
                }
                let n = g.settings.scheduler.n_live_slots;
                g.gpu.live_slots = n as u32;
                g.log.push(LogLine {
                    ts_ms: now_ms(),
                    level: LogLevel::Info,
                    source: "scheduler".into(),
                    message: format!("n_live_slots = {}", n),
                });
            }
            Cmd::SetExecutionMode {
                mode,
                decode_batch_target,
            } => {
                let mut g = self.state.write();
                g.settings.scheduler.execution_mode = mode;
                g.settings.scheduler.decode_batch_target = decode_batch_target.clamp(1, 512);
                let active_batch = g.settings.scheduler.decode_batch_target;
                g.log.push(LogLine {
                    ts_ms: now_ms(),
                    level: LogLevel::Info,
                    source: "scheduler".into(),
                    message: format!(
                        "mode = {} decode_batch_target = {}",
                        mode.label(),
                        active_batch
                    ),
                });
            }
            Cmd::PauseAgent { agent } => self.set_agent_lifecycle(agent, AgentLifecycle::Paused),
            Cmd::ResumeAgent { agent } => self.set_agent_lifecycle(agent, AgentLifecycle::Idle),
            Cmd::ResetAgent { agent } => {
                let mut g = self.state.write();
                if let Some(a) = g.agent_mut(agent) {
                    a.lifecycle = AgentLifecycle::Idle;
                    a.current_task = None;
                    a.current_task_summary = None;
                    a.iteration_progress = None;
                    a.flash = Some("reset".into());
                    a.last_error = None;
                }
            }
            Cmd::PinAgent { agent, pinned } => {
                let mut g = self.state.write();
                if let Some(a) = g.agent_mut(agent) {
                    a.pinned = pinned;
                }
            }
            Cmd::ForceEvict { agent } => {
                self.force_evict(agent);
            }
            Cmd::ApproveMerge { goal } => {
                self.set_goal_status(goal, GoalStatus::Done);
            }
            Cmd::Shutdown { .. } => {
                // Best effort: we don't currently stop the thread; the process
                // exit will. Left here for future UI wiring.
            }
            _ => {
                // other Cmds not yet wired in v0.1
            }
        }
    }

    // --- Decomposition (mock in v0.1) --------------------------------------

    fn run_decomposition(&mut self, goal_id: GoalId) {
        let goal_text = {
            let g = self.state.read();
            g.tasks
                .goals
                .get(&goal_id)
                .filter(|goal| goal.status == GoalStatus::Decomposing)
                .map(|goal| goal.text.clone())
        };
        let Some(goal_text) = goal_text else { return; };

        let decomposed = decomposer::mock_decompose(&goal_text);

        let mut created_tasks = Vec::new();
        for (i, sub) in decomposed.subtasks.iter().enumerate() {
            let deps = sub
                .depends_on_indices
                .iter()
                .filter_map(|idx| created_tasks.get(*idx).map(|t: &Task| t.id))
                .collect::<Vec<_>>();

            let task = Task {
                id: TaskId::new(),
                goal: goal_id,
                summary: sub.summary.clone(),
                persona: sub.persona,
                depends_on: deps,
                exit_criteria: sub.exit_criteria.clone(),
                suggested_files: Vec::new(),
                status: TaskStatus::Queued,
                assigned_to: None,
                priority: crate::state::Priority::Normal,
                revisions: 0,
                result: None,
            };
            self.journal_write(JournalPayload::TaskCreated { task: task.clone() });
            let _ = self.events.try_send(AppEvent::TaskCreated(task.clone()));
            created_tasks.push(task);
            let _ = i;
        }

        {
            let mut g = self.state.write();
            if let Some(goal) = g.tasks.goals.get_mut(&goal_id) {
                goal.plan_summary = Some(decomposed.plan_summary.clone());
                goal.status = GoalStatus::Running;
            }
            for t in &created_tasks {
                g.tasks.tasks.insert(t.id, t.clone());
            }
            g.master.decomposition_in_flight = false;
            g.master.last_response = Some(format!(
                "decomposed into {} sub-tasks",
                created_tasks.len()
            ));
            g.log.push(LogLine {
                ts_ms: now_ms(),
                level: LogLevel::Info,
                source: "master".into(),
                message: format!("decomposed '{}' into {} tasks", summarise(&goal_text, 40), created_tasks.len()),
            });
        }
        self.journal_write(JournalPayload::GoalPlanned {
            id: goal_id,
            plan_summary: decomposed.plan_summary.clone(),
        });
        self.journal_write(JournalPayload::GoalStatus {
            id: goal_id,
            status: GoalStatus::Running,
        });
        let _ = self
            .events
            .try_send(AppEvent::GoalStatusChanged { goal: goal_id, status: GoalStatus::Running });
        let _ = self.events.try_send(AppEvent::MasterActivity(MasterActivity {
            kind: MasterActivityKind::Idle,
            detail: String::new(),
        }));
    }

    fn run_broadcast(&mut self, text: String) {
        let fanout = self.state.read().agents.len();
        let plan_summary = format!("single-submit broadcast fanout to {fanout} agents");
        let (max_new_tokens, execution_mode, decode_batch_target) = {
            let g = self.state.read();
            (
                g.settings.max_new_tokens,
                g.settings.scheduler.execution_mode,
                g.settings.scheduler.decode_batch_target,
            )
        };
        let goal = Goal {
            id: GoalId::new(),
            submitted_by: "human".into(),
            text: text.clone(),
            created_at_ms: now_ms(),
            plan_summary: Some(plan_summary.clone()),
            status: GoalStatus::Running,
        };

        self.journal_write(JournalPayload::GoalCreated { goal: goal.clone() });
        {
            let mut g = self.state.write();
            g.tasks.goal_order.push(goal.id);
            g.tasks.goals.insert(goal.id, goal.clone());
            g.master.current_goal = Some(goal.id);
            g.master.decomposition_in_flight = false;
            g.master.last_response = Some(format!(
                "shared generation starting, mode={}, batch={}, max_new_tokens={max_new_tokens}",
                execution_mode.label(),
                decode_batch_target
            ));
            g.log.push(LogLine {
                ts_ms: now_ms(),
                level: LogLevel::Info,
                source: "master".into(),
                message: format!(
                    "broadcast submitted to {} agents: {}",
                    fanout,
                    summarise(&goal.text, 80)
                ),
            });
            g.log.push(LogLine {
                ts_ms: now_ms(),
                level: LogLevel::Info,
                source: "broadcast".into(),
                message: format!(
                    "shared generation starting, mode={}, batch={}, max_new_tokens={max_new_tokens}",
                    execution_mode.label(),
                    decode_batch_target
                ),
            });
        }
        let _ = self.events.try_send(AppEvent::GoalCreated(goal.clone()));
        let _ = self.events.try_send(AppEvent::MasterActivity(MasterActivity {
            kind: MasterActivityKind::Idle,
            detail: format!(
                "generating shared response, mode={}, batch={}, max_new_tokens={max_new_tokens}",
                execution_mode.label(),
                decode_batch_target
            ),
        }));

        let response = match self.generate_broadcast_response(&text, fanout) {
            Ok(response) => response,
            Err(err) => {
                eprintln!("[broadcast] shared generation failed: {err}");
                {
                    let mut g = self.state.write();
                    g.log.push(LogLine {
                        ts_ms: now_ms(),
                        level: LogLevel::Error,
                        source: "broadcast".into(),
                        message: format!("shared generation failed: {err}"),
                    });
                }
                self.set_goal_status(goal.id, GoalStatus::Failed);
                let _ = self.events.try_send(AppEvent::Error {
                    context: "broadcast".into(),
                    message: err.to_string(),
                });
                return;
            }
        };

        let prompt_tokens = text.split_whitespace().count() as u64;
        let response_tokens = response.split_whitespace().count() as u64;
        let diff_summary = format!("single shared response fanned out to {fanout} agents");
        let backend_kind = self.state.read().backend_kind;
        let stamp_ms = now_ms();
        let agents: Vec<(AgentId, u8, Persona)> = {
            let g = self.state.read();
            g.agents
                .iter()
                .map(|agent| (agent.id, agent.display_index, agent.persona))
                .collect()
        };
        let tasks: Vec<Task> = agents
            .iter()
            .map(|(agent_id, display_index, persona)| Task {
                id: TaskId::new(),
                goal: goal.id,
                summary: format!("broadcast reply -> #{display_index:02} {}", persona.label()),
                persona: *persona,
                depends_on: Vec::new(),
                exit_criteria: "shared response delivered".into(),
                suggested_files: Vec::new(),
                status: if backend_kind == BackendKind::Mock {
                    TaskStatus::Queued
                } else {
                    TaskStatus::Accepted
                },
                assigned_to: if backend_kind == BackendKind::Mock {
                    None
                } else {
                    Some(*agent_id)
                },
                priority: Priority::Interactive,
                revisions: 0,
                result: if backend_kind == BackendKind::Mock {
                    None
                } else {
                    Some(TaskResult {
                        outcome: TaskOutcome::Success {
                            response: response.clone(),
                        },
                        tokens_in: prompt_tokens,
                        tokens_out: response_tokens,
                        duration_ms: 0,
                        files_touched: Vec::new(),
                        diff_summary: diff_summary.clone(),
                    })
                },
            })
            .collect();

        {
            let mut g = self.state.write();
            for task in &tasks {
                g.tasks.tasks.insert(task.id, task.clone());
            }
            for (agent_id, _, _) in &agents {
                if let Some(agent) = g.agent_mut(*agent_id) {
                    agent.lifecycle = if backend_kind == BackendKind::Mock {
                        AgentLifecycle::Queued
                    } else {
                        AgentLifecycle::Idle
                    };
                    agent.current_task = None;
                    agent.current_task_summary = None;
                    agent.iteration_progress = None;
                    agent.last_tool = Some(format!("broadcast x{fanout}"));
                    agent.flash = Some(if backend_kind == BackendKind::Mock {
                        "fanout queued".into()
                    } else {
                        "broadcast".into()
                    });
                    agent.history.push(HistoryEntry {
                        ts_ms: stamp_ms,
                        kind: HistoryKind::User,
                        text: text.clone(),
                    });
                    if backend_kind != BackendKind::Mock {
                        agent.history.push(HistoryEntry {
                            ts_ms: stamp_ms,
                            kind: HistoryKind::Assistant,
                            text: response.clone(),
                        });
                        agent.metrics.tokens_in_total += prompt_tokens;
                        agent.metrics.tokens_out_total += response_tokens;
                        agent.metrics.tasks_accepted += 1;
                        agent.metrics.iterations_last = 1;
                        agent.metrics.iterations_mean = 1.0;
                    }
                }
            }
            g.log.push(LogLine {
                ts_ms: stamp_ms,
                level: LogLevel::Info,
                source: "broadcast".into(),
                message: if backend_kind == BackendKind::Mock {
                    format!("shared response queued for animated fanout to {} agents", fanout)
                } else {
                    format!("shared response delivered to {} agents", fanout)
                },
            });
        }
        for task in &tasks {
            self.journal_write(JournalPayload::TaskCreated { task: task.clone() });
        }
        if backend_kind == BackendKind::Mock {
            self.animate_mock_broadcast(MockBroadcastPlan {
                goal_id: goal.id,
                response,
                prompt_tokens,
                response_tokens,
                diff_summary,
                agents,
                tasks,
            });
        } else {
            self.set_goal_status(goal.id, GoalStatus::Done);
        }
    }

    // --- Dispatch + mock worker step ---------------------------------------

    fn dispatch_one_round(&mut self) {
        // Find dispatchable tasks (queued, deps satisfied), sorted by priority then created order.
        let ready: Vec<TaskId> = {
            let g = self.state.read();
            let mut ids: Vec<(TaskId, crate::state::Priority)> = g
                .tasks
                .tasks
                .values()
                .filter(|t| t.status == TaskStatus::Queued)
                .filter(|t| {
                    t.depends_on.iter().all(|d| {
                        matches!(
                            g.tasks.tasks.get(d).map(|t| t.status),
                            Some(TaskStatus::Accepted)
                        )
                    })
                })
                .map(|t| (t.id, t.priority))
                .collect();
            ids.sort_by(|(a_id, a_p), (b_id, b_p)| b_p.cmp(a_p).then(a_id.0.cmp(&b_id.0)));
            ids.into_iter().map(|(id, _)| id).collect()
        };

        for task_id in ready {
            // Find a suitable worker.
            let chosen = dispatch::choose_worker(&self.state.read(), task_id);
            let Some(agent_id) = chosen else { continue; };

            // Ask scheduler for a slot.
            let decision = self.schedule_agent(agent_id);

            let slot_idx = match decision {
                SchedDecision::AlreadyResident(i) => i,
                SchedDecision::LoadInto(i) => {
                    self.load_agent_into(agent_id, i);
                    i
                }
                SchedDecision::EvictAndLoad { evict } => {
                    self.evict_slot(evict);
                    self.load_agent_into(agent_id, evict);
                    evict
                }
                SchedDecision::Deny(_) => {
                    continue;
                }
            };

            // Assign + run (synchronously, mock).
            self.assign_and_run(task_id, agent_id, slot_idx);
        }
    }

    fn load_agent_into(&mut self, agent: AgentId, slot: crate::ipc::SlotIdx) {
        {
            let mut g = self.state.write();
            if let Some(s) = g.slots.get_mut(slot.0) {
                s.occupant = Some(agent);
                s.state = SlotState::Hot;
            }
            if let Some(a) = g.agent_mut(agent) {
                a.lifecycle = AgentLifecycle::Idle;
                a.metrics.hbm_resident = true;
            }
            g.gpu.live_slots_used = g
                .slots
                .iter()
                .filter(|s| s.state == SlotState::Hot)
                .count() as u32;
        }
        self.policy.mark_touched(slot);
        let _ = self.events.try_send(AppEvent::SlotChanged(SlotOccupancyReport {
            slot_idx: slot,
            occupant: Some(agent),
            state: SlotStateKind::Hot,
        }));
    }

    fn evict_slot(&mut self, slot: crate::ipc::SlotIdx) {
        let evicted = {
            let mut g = self.state.write();
            let prev = g.slots.get(slot.0).and_then(|s| s.occupant);
            if let Some(prev) = prev {
                if let Some(a) = g.agent_mut(prev) {
                    a.lifecycle = AgentLifecycle::Swapped;
                    a.metrics.hbm_resident = false;
                }
            }
            if let Some(s) = g.slots.get_mut(slot.0) {
                s.occupant = None;
                s.state = SlotState::Empty;
            }
            prev
        };
        let _ = self.events.try_send(AppEvent::SlotChanged(SlotOccupancyReport {
            slot_idx: slot,
            occupant: evicted,
            state: SlotStateKind::Empty,
        }));
    }

    fn assign_and_run(&mut self, task_id: TaskId, agent: AgentId, slot: crate::ipc::SlotIdx) {
        // Transition task + agent.
        {
            let mut g = self.state.write();
            let task_summary = g
                .tasks
                .tasks
                .get(&task_id)
                .map(|t| t.summary.clone())
                .unwrap_or_default();
            let max_iters = g.settings.max_iterations;
            if let Some(t) = g.tasks.tasks.get_mut(&task_id) {
                t.status = TaskStatus::Running;
                t.assigned_to = Some(agent);
            }
            if let Some(a) = g.agent_mut(agent) {
                a.lifecycle = AgentLifecycle::Running;
                a.current_task = Some(task_id);
                a.current_task_summary = Some(task_summary.clone());
                a.iteration_progress = Some((0, max_iters));
                a.flash = Some("starting".into());
            }
            g.log.push(LogLine {
                ts_ms: now_ms(),
                level: LogLevel::Info,
                source: "dispatch".to_string(),
                message: format!(
                    "{} -> {} (slot {})",
                    task_summary,
                    short_id(agent.0),
                    slot.0
                ),
            });
        }
        self.journal_write(JournalPayload::TaskAssign {
            id: task_id,
            agent,
        });
        self.journal_write(JournalPayload::TaskStatus {
            id: task_id,
            status: TaskStatus::Running,
        });
        let _ = self.events.try_send(AppEvent::TaskAssigned { task: task_id, agent });
        let _ = self.events.try_send(AppEvent::TaskStatusChanged {
            task: task_id,
            status: TaskStatus::Running,
        });

        // Mock worker executes synchronously but streams incremental state
        // via the shared AppState. For v0.1 this is plenty; a real CUDA
        // worker will run on its own thread (see worker::cuda).
        let outcome = self.backend.run_task_streaming(&self.state, agent, task_id, slot);

        // Commit result.
        let tr = TaskResult {
            outcome: outcome.clone(),
            tokens_in: 128,
            tokens_out: 256,
            duration_ms: 700,
            files_touched: vec![],
            diff_summary: String::new(),
        };
        let final_status = match &outcome {
            TaskOutcome::Success { .. } => TaskStatus::Accepted,
            TaskOutcome::NeedsRevision { .. } => TaskStatus::Revising,
            TaskOutcome::TimedOut => TaskStatus::TimedOut,
            TaskOutcome::Failed { .. } => TaskStatus::Failed,
        };

        {
            let mut g = self.state.write();
            if let Some(t) = g.tasks.tasks.get_mut(&task_id) {
                t.status = final_status;
                t.result = Some(tr.clone());
            }
            if let Some(a) = g.agent_mut(agent) {
                a.lifecycle = AgentLifecycle::Idle;
                a.current_task = None;
                a.current_task_summary = None;
                a.iteration_progress = None;
                a.flash = Some(match &outcome {
                    TaskOutcome::Success { .. } => "ok".into(),
                    TaskOutcome::NeedsRevision { reason } => format!("revise: {reason}"),
                    TaskOutcome::TimedOut => "timeout".into(),
                    TaskOutcome::Failed { message } => format!("fail: {message}"),
                });
                if final_status == TaskStatus::Accepted {
                    a.metrics.tasks_accepted += 1;
                }
                a.metrics.tokens_out_total += tr.tokens_out;
                a.metrics.tokens_in_total += tr.tokens_in;
                a.metrics.iterations_last = 3;
                a.metrics.tok_s_decode = 76.0 + (agent.0.as_u128() as u32 as f32 % 8.0);
            }
        }
        self.journal_write(JournalPayload::TaskResult {
            id: task_id,
            result: tr.clone(),
        });
        self.journal_write(JournalPayload::TaskStatus {
            id: task_id,
            status: final_status,
        });
        let _ = self.events.try_send(AppEvent::TaskResult { task: task_id, result: tr });
        let _ = self.events.try_send(AppEvent::TaskStatusChanged {
            task: task_id,
            status: final_status,
        });
        self.policy.on_task_done(slot);
        self.maybe_complete_goal(task_id);
    }

    fn maybe_complete_goal(&mut self, task_id: TaskId) {
        let goal = {
            let g = self.state.read();
            g.tasks.tasks.get(&task_id).map(|t| t.goal)
        };
        let Some(goal) = goal else { return; };
        if matches!(
            self.state.read().tasks.goals.get(&goal).map(|g| g.status),
            Some(GoalStatus::Cancelled)
        ) {
            return;
        }
        let counts = self.state.read().tasks.counts_for(goal);
        if counts.total > 0 && counts.done + counts.failed + counts.cancelled == counts.total {
            let new_status = if counts.failed == 0 {
                GoalStatus::Done
            } else {
                GoalStatus::Failed
            };
            self.set_goal_status(goal, new_status);
        }
    }

    // --- helpers ------------------------------------------------------------

    fn set_goal_status(&mut self, goal: GoalId, status: GoalStatus) {
        {
            let mut g = self.state.write();
            if let Some(gg) = g.tasks.goals.get_mut(&goal) {
                gg.status = status;
            }
            restore_master_state(&mut g);
            g.log.push(LogLine {
                ts_ms: now_ms(),
                level: LogLevel::Info,
                source: "goal".into(),
                message: format!("{} -> {:?}", short_id(goal.0), status),
            });
        }
        self.journal_write(JournalPayload::GoalStatus { id: goal, status });
        let _ = self.events.try_send(AppEvent::GoalStatusChanged { goal, status });
    }

    fn cancel_goal(&mut self, goal: GoalId) {
        let cancelled_tasks = {
            let mut g = self.state.write();
            let mut cancelled_tasks = Vec::new();
            if let Some(gg) = g.tasks.goals.get_mut(&goal) {
                gg.status = GoalStatus::Cancelled;
            }
            let task_ids: Vec<TaskId> = g
                .tasks
                .tasks
                .values()
                .filter(|task| task.goal == goal)
                .map(|task| task.id)
                .collect();
            for task_id in task_ids {
                let should_cancel = matches!(
                    g.tasks.tasks.get(&task_id).map(|task| task.status),
                    Some(TaskStatus::Queued | TaskStatus::Running | TaskStatus::Verifying | TaskStatus::Revising)
                );
                if !should_cancel {
                    continue;
                }
                if let Some(task) = g.tasks.tasks.get_mut(&task_id) {
                    task.status = TaskStatus::Cancelled;
                }
                for agent in &mut g.agents {
                    if agent.current_task == Some(task_id) {
                        agent.lifecycle = AgentLifecycle::Idle;
                        agent.current_task = None;
                        agent.current_task_summary = None;
                        agent.iteration_progress = None;
                        agent.flash = Some("cancelled".into());
                    }
                }
                cancelled_tasks.push(task_id);
            }
            restore_master_state(&mut g);
            g.log.push(LogLine {
                ts_ms: now_ms(),
                level: LogLevel::Warn,
                source: "goal".into(),
                message: format!("{} cancelled", short_id(goal.0)),
            });
            cancelled_tasks
        };
        self.journal_write(JournalPayload::GoalStatus {
            id: goal,
            status: GoalStatus::Cancelled,
        });
        let _ = self.events.try_send(AppEvent::GoalStatusChanged {
            goal,
            status: GoalStatus::Cancelled,
        });
        for task_id in cancelled_tasks {
            self.journal_write(JournalPayload::TaskStatus {
                id: task_id,
                status: TaskStatus::Cancelled,
            });
            let _ = self.events.try_send(AppEvent::TaskStatusChanged {
                task: task_id,
                status: TaskStatus::Cancelled,
            });
        }
    }

    fn force_evict(&mut self, agent: AgentId) {
        let slot = {
            let g = self.state.read();
            if g.agent(agent)
                .and_then(|a| a.current_task)
                .is_some()
            {
                return;
            }
            g.slots
                .iter()
                .position(|slot| slot.occupant == Some(agent))
                .map(crate::ipc::SlotIdx)
        };
        if let Some(slot) = slot {
            self.evict_slot(slot);
            let mut g = self.state.write();
            g.log.push(LogLine {
                ts_ms: now_ms(),
                level: LogLevel::Info,
                source: "scheduler".into(),
                message: format!("force-evicted {}", short_id(agent.0)),
            });
        }
    }

    fn schedule_agent(&mut self, want: AgentId) -> SchedDecision {
        let (filtered_slots, blocked_by_pin) = {
            let g = self.state.read();
            let mut filtered_slots = g.slots.clone();
            let mut blocked_by_pin = false;
            for slot in &mut filtered_slots {
                let Some(occupant) = slot.occupant else { continue; };
                if occupant == want {
                    continue;
                }
                let pinned = g.agent(occupant).map(|agent| agent.pinned).unwrap_or(false);
                if pinned && slot.state == SlotState::Hot {
                    slot.state = SlotState::Loading;
                    blocked_by_pin = true;
                }
            }
            (filtered_slots, blocked_by_pin)
        };
        match self.policy.on_dispatch(want, &filtered_slots) {
            SchedDecision::Deny(DenyReason::AllSlotsBusy) if blocked_by_pin => {
                SchedDecision::Deny(DenyReason::AgentPinnedElsewhere)
            }
            other => other,
        }
    }

    fn set_agent_lifecycle(&mut self, agent: AgentId, lifecycle: AgentLifecycle) {
        let mut g = self.state.write();
        if let Some(a) = g.agent_mut(agent) {
            a.lifecycle = lifecycle;
        }
    }

    fn tick(&mut self) {
        // Update swarm tok/s metric.
        let mut g = self.state.write();
        let swarm_tok_s: f32 = g
            .agents
            .iter()
            .filter(|a| a.lifecycle == AgentLifecycle::Running)
            .map(|a| a.metrics.tok_s_decode)
            .sum();
        g.gpu.swarm_tok_s = swarm_tok_s;
        let live_used = g
            .slots
            .iter()
            .filter(|s| s.state == SlotState::Hot)
            .count() as u32;
        g.gpu.live_slots_used = live_used;
        // KV = model_shared (approx) + kv_per_agent * live_used.
        g.gpu.hbm_used_gb =
            g.settings.scheduler.model_hbm_gb + g.settings.scheduler.kv_per_agent_gb * live_used as f32;
    }

    fn journal_write(&mut self, payload: JournalPayload) {
        let rec = JournalRecord {
            ts_ms: now_ms(),
            protocol_version: PROTOCOL_VERSION,
            payload,
        };
        if let Err(e) = self.journal.append(&rec) {
            tracing::warn!("journal append failed: {e}");
        }
    }

    fn animate_mock_broadcast(&mut self, plan: MockBroadcastPlan) {
        const CHUNK: usize = 5;
        for (chunk_index, (agent_chunk, task_chunk)) in plan
            .agents
            .chunks(CHUNK)
            .zip(plan.tasks.chunks(CHUNK))
            .enumerate()
        {
            {
                let mut g = self.state.write();
                for ((agent_id, display_index, persona), task) in agent_chunk.iter().zip(task_chunk.iter()) {
                    if let Some(t) = g.tasks.tasks.get_mut(&task.id) {
                        t.status = TaskStatus::Running;
                        t.assigned_to = Some(*agent_id);
                    }
                    if let Some(agent) = g.agent_mut(*agent_id) {
                        agent.lifecycle = AgentLifecycle::Running;
                        agent.current_task = Some(task.id);
                        agent.current_task_summary =
                            Some(format!("fanout -> #{display_index:02} {}", persona.label()));
                        agent.iteration_progress = Some((1, 1));
                        agent.flash = Some("broadcasting".into());
                    }
                }
                g.master.last_response =
                    Some(format!("fanning out shared response to {} agents", plan.agents.len()));
            }
            for ((agent_id, _, _), task) in agent_chunk.iter().zip(task_chunk.iter()) {
                self.journal_write(JournalPayload::TaskAssign {
                    id: task.id,
                    agent: *agent_id,
                });
                self.journal_write(JournalPayload::TaskStatus {
                    id: task.id,
                    status: TaskStatus::Running,
                });
            }
            std::thread::sleep(Duration::from_millis(90));

            let results: Vec<(TaskId, TaskResult)> = task_chunk
                .iter()
                .map(|task| {
                    (
                        task.id,
                        TaskResult {
                            outcome: TaskOutcome::Success {
                                response: plan.response.clone(),
                            },
                            tokens_in: plan.prompt_tokens,
                            tokens_out: plan.response_tokens,
                            duration_ms: 90,
                            files_touched: Vec::new(),
                            diff_summary: plan.diff_summary.clone(),
                        },
                    )
                })
                .collect();

            {
                let mut g = self.state.write();
                for (((agent_id, _, _), task), (_, result)) in agent_chunk
                    .iter()
                    .zip(task_chunk.iter())
                    .zip(results.iter())
                {
                    if let Some(t) = g.tasks.tasks.get_mut(&task.id) {
                        t.status = TaskStatus::Accepted;
                        t.result = Some(result.clone());
                    }
                    if let Some(agent) = g.agent_mut(*agent_id) {
                        agent.lifecycle = AgentLifecycle::Idle;
                        agent.current_task = None;
                        agent.current_task_summary = None;
                        agent.iteration_progress = None;
                        agent.flash = Some("broadcast complete".into());
                        agent.history.push(HistoryEntry {
                            ts_ms: now_ms(),
                            kind: HistoryKind::Assistant,
                            text: plan.response.clone(),
                        });
                        agent.metrics.tokens_in_total += plan.prompt_tokens;
                        agent.metrics.tokens_out_total += plan.response_tokens;
                        agent.metrics.tasks_accepted += 1;
                        agent.metrics.iterations_last = 1;
                        agent.metrics.iterations_mean = 1.0;
                        agent.metrics.tok_s_decode = 120.0;
                    }
                }
                g.master.last_response =
                    Some(format!(
                        "shared response delivered to {} / {}",
                        ((chunk_index + 1) * CHUNK).min(plan.agents.len()),
                        plan.agents.len()
                    ));
            }
            for ((agent_id, _, _), (task_id, result)) in agent_chunk.iter().zip(results.iter()) {
                let _ = agent_id;
                self.journal_write(JournalPayload::TaskResult {
                    id: *task_id,
                    result: result.clone(),
                });
                self.journal_write(JournalPayload::TaskStatus {
                    id: *task_id,
                    status: TaskStatus::Accepted,
                });
            }
            std::thread::sleep(Duration::from_millis(40));
        }
        {
            let mut g = self.state.write();
            g.log.push(LogLine {
                ts_ms: now_ms(),
                level: LogLevel::Info,
                source: "broadcast".into(),
                message: format!(
                    "animated fanout complete for {}",
                    short_id(plan.goal_id.0)
                ),
            });
        }
        self.set_goal_status(plan.goal_id, GoalStatus::Done);
    }

    fn generate_broadcast_response(&mut self, text: &str, fanout: usize) -> anyhow::Result<String> {
        let (backend_kind, max_new_tokens, max_iterations, decode_batch_target) = {
            let g = self.state.read();
            (
                g.backend_kind,
                g.settings.max_new_tokens,
                g.settings.max_iterations,
                g.settings.scheduler.decode_batch_target,
            )
        };
        match backend_kind {
            BackendKind::Mock => Ok(self.backend.broadcast_response(text, fanout)),
            BackendKind::Rvllm => {
                #[cfg(feature = "cuda")]
                {
                    return crate::worker::cuda::complete_once(
                        Persona::Runtime,
                        text,
                        max_new_tokens,
                        decode_batch_target,
                        max_iterations,
                        1,
                    );
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = max_new_tokens;
                    let _ = max_iterations;
                    let _ = decode_batch_target;
                    anyhow::bail!("rvllm cuda backend is not compiled in")
                }
            }
        }
    }
}

fn short_id(u: uuid::Uuid) -> String {
    u.simple().to_string().chars().take(8).collect()
}

pub(crate) fn summarise(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

fn apply_journal_to_state(state: &mut AppState, rec: &JournalRecord) {
    match &rec.payload {
        JournalPayload::GoalCreated { goal } => {
            state.tasks.goals.insert(goal.id, goal.clone());
            if !state.tasks.goal_order.contains(&goal.id) {
                state.tasks.goal_order.push(goal.id);
            }
        }
        JournalPayload::GoalPlanned { id, plan_summary } => {
            if let Some(goal) = state.tasks.goals.get_mut(id) {
                goal.plan_summary = Some(plan_summary.clone());
            }
        }
        JournalPayload::GoalStatus { id, status } => {
            if let Some(g) = state.tasks.goals.get_mut(id) {
                g.status = *status;
            }
        }
        JournalPayload::TaskCreated { task } => {
            state.tasks.tasks.insert(task.id, task.clone());
        }
        JournalPayload::TaskStatus { id, status } => {
            if let Some(t) = state.tasks.tasks.get_mut(id) {
                t.status = *status;
            }
        }
        JournalPayload::TaskAssign { id, agent } => {
            if let Some(t) = state.tasks.tasks.get_mut(id) {
                t.assigned_to = Some(*agent);
            }
        }
        JournalPayload::TaskResult { id, result } => {
            if let Some(t) = state.tasks.tasks.get_mut(id) {
                t.result = Some(result.clone());
            }
        }
        JournalPayload::Note { .. } => {}
    }
}

// Suppress unused warnings from the cuda path scaffolding.
#[allow(dead_code)]
fn _touch<T>(_: T) {}

fn restore_master_state(state: &mut AppState) {
    state.master.current_goal = state
        .tasks
        .goal_order
        .iter()
        .rev()
        .copied()
        .find(|goal_id| {
            matches!(
                state.tasks.goals.get(goal_id).map(|goal| goal.status),
                Some(GoalStatus::Decomposing | GoalStatus::Running | GoalStatus::Merging)
            )
        })
        .or_else(|| state.tasks.goal_order.last().copied());
    state.master.decomposition_in_flight = state
        .master
        .current_goal
        .and_then(|goal_id| state.tasks.goals.get(&goal_id))
        .map(|goal| goal.status == GoalStatus::Decomposing)
        .unwrap_or(false);
    state.master.last_response = state
        .master
        .current_goal
        .and_then(|goal_id| state.tasks.goals.get(&goal_id))
        .and_then(|goal| goal.plan_summary.clone());
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::thread;
    use std::time::{Duration, Instant};

    use tempfile::tempdir;

    use super::spawn;
    use crate::ipc::Cmd;
    use crate::state::history::HistoryKind;
    use crate::state::{BackendKind, GoalStatus, TaskStatus};

    #[test]
    fn broadcast_fans_out_one_response_to_all_agents() {
        let tmp = tempdir().expect("tempdir");
        let handle = spawn(tmp.path().to_path_buf(), BackendKind::Mock);
        handle
            .cmd_tx
            .send(Cmd::SubmitBroadcast {
                text: "reply once to everyone".into(),
            })
            .expect("send broadcast");

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let done = {
                let g = handle.state.read();
                g.tasks.goals.len() == 1
                    && g.tasks
                        .goals
                        .values()
                        .all(|goal| goal.status == GoalStatus::Done)
            };
            if done {
                break;
            }
            assert!(Instant::now() < deadline, "broadcast goal did not finish in time");
            thread::sleep(Duration::from_millis(10));
        }

        let g = handle.state.read();
        assert_eq!(g.agents.len(), 30);
        assert_eq!(g.tasks.goals.len(), 1);
        assert_eq!(g.tasks.global_counts().done, 30);

        let goal_id = g.tasks.goal_order[0];
        let goal = g.tasks.goals.get(&goal_id).expect("goal");
        assert_eq!(
            goal.plan_summary.as_deref(),
            Some("single-submit broadcast fanout to 30 agents")
        );

        let tasks = g.tasks.tasks_in_goal(goal_id);
        assert_eq!(tasks.len(), 30);
        assert!(tasks.iter().all(|task| task.status == TaskStatus::Accepted));

        let mut responses = BTreeSet::new();
        for agent in &g.agents {
            let last_user = agent
                .history
                .iter_rev()
                .find(|entry| entry.kind == HistoryKind::User)
                .map(|entry| entry.text.as_str());
            let last_assistant = agent
                .history
                .iter_rev()
                .find(|entry| entry.kind == HistoryKind::Assistant)
                .map(|entry| entry.text.clone())
                .expect("assistant history");
            assert_eq!(last_user, Some("reply once to everyone"));
            responses.insert(last_assistant);
        }
        assert_eq!(responses.len(), 1);
    }
}
