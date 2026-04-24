#![allow(dead_code)]

// In-memory application state. Everything the UI renders comes from here.
//
// See docs/01_ARCHITECTURE.md for the read/write discipline: UI reads,
// controller + workers write. We wrap AppState in Arc<RwLock<_>> at the
// top level in controller/mod.rs.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::ipc::{AgentId, Goal, GoalId, LogLine, SlotIdx, Task, TaskId};

pub mod history;
pub use history::History;

// --- Enums ------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Persona {
    Kernels,
    Runtime,
    Tests,
    Docs,
    Tpu,
    Misc,
}

impl Persona {
    pub fn label(self) -> &'static str {
        match self {
            Persona::Kernels => "kernels",
            Persona::Runtime => "runtime",
            Persona::Tests => "tests",
            Persona::Docs => "docs",
            Persona::Tpu => "tpu",
            Persona::Misc => "misc",
        }
    }

    /// Default distribution across 30 worker slots.
    pub fn default_distribution() -> [Persona; 30] {
        [
            Persona::Kernels, Persona::Kernels, Persona::Kernels,
            Persona::Kernels, Persona::Kernels, Persona::Kernels,
            Persona::Runtime, Persona::Runtime, Persona::Runtime,
            Persona::Runtime, Persona::Runtime, Persona::Runtime,
            Persona::Tests, Persona::Tests, Persona::Tests,
            Persona::Tests, Persona::Tests, Persona::Tests,
            Persona::Docs, Persona::Docs, Persona::Docs, Persona::Docs,
            Persona::Tpu, Persona::Tpu, Persona::Tpu, Persona::Tpu,
            Persona::Misc, Persona::Misc, Persona::Misc, Persona::Misc,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentLifecycle {
    Idle,
    Queued,
    Loading,
    Running,
    Swapped,
    Failed,
    Paused,
}

impl AgentLifecycle {
    pub fn label(self) -> &'static str {
        match self {
            AgentLifecycle::Idle => "IDLE",
            AgentLifecycle::Queued => "QUEUED",
            AgentLifecycle::Loading => "LOADING",
            AgentLifecycle::Running => "RUNNING",
            AgentLifecycle::Swapped => "SWAPPED",
            AgentLifecycle::Failed => "FAILED",
            AgentLifecycle::Paused => "PAUSED",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    Decomposing,
    Running,
    Merging,
    Done,
    Failed,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Queued,
    Running,
    Verifying,
    Revising,
    Accepted,
    Rejected,
    Cancelled,
    TimedOut,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Normal,
    High,
    Interactive,
}

// --- Metrics / budgets ------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub tok_s_decode: f32,
    pub tok_s_prefill: f32,
    pub tokens_out_total: u64,
    pub tokens_in_total: u64,
    pub iterations_last: u32,
    pub iterations_mean: f32,
    pub tool_calls_last: u32,
    pub tool_err_rate: f32,
    pub tasks_accepted: u32,
    pub tasks_rejected: u32,
    pub tasks_timed_out: u32,
    pub hbm_resident: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GpuBudget {
    pub hbm_used_gb: f32,
    pub hbm_total_gb: f32,
    pub live_slots: u32,
    pub live_slots_used: u32,
    pub swarm_tok_s: f32,
}

impl GpuBudget {
    pub fn placeholder(live_slots: u32) -> Self {
        Self {
            hbm_used_gb: 0.0,
            hbm_total_gb: 80.0,
            live_slots,
            live_slots_used: 0,
            swarm_tok_s: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub execution_mode: ExecutionMode,
    pub n_live_slots: usize,
    pub decode_batch_target: usize,
    pub model_hbm_gb: f32,
    pub kv_per_agent_gb: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    Operator30,
    Saturator,
}

impl ExecutionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Operator30 => "operator-30",
            Self::Saturator => "saturator",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Operator30 => "30 visible logical agents, batch target 30",
            Self::Saturator => "30 visible agents backed by 256-512 on-card decode rows",
        }
    }

    pub fn decode_batch_target(self) -> usize {
        match self {
            Self::Operator30 => 30,
            Self::Saturator => 512,
        }
    }

    pub fn from_env() -> Self {
        match std::env::var("RVLLM_SWARM_MODE") {
            Ok(v) if matches!(v.as_str(), "saturator" | "saturate" | "throughput" | "512") => {
                Self::Saturator
            }
            _ => Self::Operator30,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        let execution_mode = ExecutionMode::from_env();
        let decode_batch_target = std::env::var("RVLLM_DECODE_BATCH_TARGET")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| execution_mode.decode_batch_target());
        Self {
            execution_mode,
            n_live_slots: 4,
            decode_batch_target,
            model_hbm_gb: 4.3,
            kv_per_agent_gb: 1.2,
        }
    }
}

// --- AgentState / MasterState -----------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentState {
    pub id: AgentId,
    pub display_index: u8, // 0..30
    pub persona: Persona,
    pub lifecycle: AgentLifecycle,
    pub current_task: Option<TaskId>,
    pub current_task_summary: Option<String>,
    pub iteration_progress: Option<(u32, u32)>, // (current, max)
    pub last_tool: Option<String>,
    pub flash: Option<String>,
    pub worktree_path: PathBuf,
    pub branch: Option<String>,
    pub metrics: AgentMetrics,
    pub history: History,
    pub last_error: Option<String>,
    pub pinned: bool,
}

impl AgentState {
    pub fn new(display_index: u8, persona: Persona, worktree_root: &std::path::Path) -> Self {
        let id = AgentId::new();
        let worktree_path = worktree_root.join(format!("agent-{}", id.short()));
        Self {
            id,
            display_index,
            persona,
            lifecycle: AgentLifecycle::Idle,
            current_task: None,
            current_task_summary: None,
            iteration_progress: None,
            last_tool: None,
            flash: None,
            worktree_path,
            branch: None,
            metrics: AgentMetrics::default(),
            history: History::with_capacity(128),
            last_error: None,
            pinned: false,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MasterState {
    pub agent_id: AgentId,
    pub current_goal: Option<GoalId>,
    pub decomposition_in_flight: bool,
    pub verification_in_flight: Option<TaskId>,
    pub last_response: Option<String>,
    pub calls_today: u64,
    pub total_tokens_today: u64,
}

// --- TaskGraph --------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TaskGraph {
    pub goals: HashMap<GoalId, Goal>,
    pub tasks: HashMap<TaskId, Task>,
    pub goal_order: Vec<GoalId>,
}

impl TaskGraph {
    pub fn tasks_in_goal(&self, goal: GoalId) -> Vec<&Task> {
        let mut out: Vec<&Task> = self.tasks.values().filter(|t| t.goal == goal).collect();
        out.sort_by_key(|t| t.id.0);
        out
    }

    pub fn counts_for(&self, goal: GoalId) -> GoalCounts {
        let mut c = GoalCounts::default();
        for t in self.tasks.values().filter(|t| t.goal == goal) {
            c.total += 1;
            match t.status {
                TaskStatus::Queued => c.queued += 1,
                TaskStatus::Running | TaskStatus::Verifying | TaskStatus::Revising => c.in_flight += 1,
                TaskStatus::Accepted => c.done += 1,
                TaskStatus::Cancelled => c.cancelled += 1,
                TaskStatus::Rejected | TaskStatus::Failed | TaskStatus::TimedOut => c.failed += 1,
            }
        }
        c
    }

    pub fn global_counts(&self) -> GoalCounts {
        let mut c = GoalCounts::default();
        for t in self.tasks.values() {
            c.total += 1;
            match t.status {
                TaskStatus::Queued => c.queued += 1,
                TaskStatus::Running | TaskStatus::Verifying | TaskStatus::Revising => c.in_flight += 1,
                TaskStatus::Accepted => c.done += 1,
                TaskStatus::Cancelled => c.cancelled += 1,
                TaskStatus::Rejected | TaskStatus::Failed | TaskStatus::TimedOut => c.failed += 1,
            }
        }
        c
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct GoalCounts {
    pub total: u32,
    pub queued: u32,
    pub in_flight: u32,
    pub done: u32,
    pub cancelled: u32,
    pub failed: u32,
}

// --- Slot occupancy ---------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotState {
    Empty,
    Loading,
    Hot,
    Evicting,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SlotOccupancy {
    pub slot_idx: SlotIdx,
    pub occupant: Option<AgentId>,
    pub state: SlotState,
}

impl SlotOccupancy {
    pub fn empty(idx: usize) -> Self {
        Self {
            slot_idx: SlotIdx(idx),
            occupant: None,
            state: SlotState::Empty,
        }
    }
}

// --- Settings ---------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub scheduler: SchedulerConfig,
    pub max_new_tokens: usize,
    pub max_iterations: u32,
    pub auto_merge: bool,
    pub ui_scale: f32,
    pub ppl_canary_interval_secs: u64,
}

impl Default for Settings {
    fn default() -> Self {
        let max_new_tokens = std::env::var("RVLLM_MAX_NEW_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1024);
        Self {
            scheduler: SchedulerConfig::default(),
            max_new_tokens,
            max_iterations: 6,
            auto_merge: false,
            ui_scale: 1.0,
            ppl_canary_interval_secs: 300,
        }
    }
}

// --- Log ring buffer --------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogRing {
    inner: VecDeque<LogLine>,
    cap: usize,
}

impl LogRing {
    pub fn new(cap: usize) -> Self {
        Self {
            inner: VecDeque::with_capacity(cap),
            cap,
        }
    }
    pub fn push(&mut self, line: LogLine) {
        if self.inner.len() == self.cap {
            self.inner.pop_front();
        }
        self.inner.push_back(line);
    }
    pub fn iter_rev(&self) -> impl Iterator<Item = &LogLine> {
        self.inner.iter().rev()
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for LogRing {
    fn default() -> Self {
        Self::new(500)
    }
}

// --- AppState ---------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppState {
    pub agents: Vec<AgentState>,
    pub master: MasterState,
    pub tasks: TaskGraph,
    pub slots: Vec<SlotOccupancy>,
    pub gpu: GpuBudget,
    pub git: crate::ipc::GitSummary,
    pub log: LogRing,
    pub settings: Settings,
    pub backend_kind: BackendKind,
    pub repo_root: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    Mock,
    Rvllm,
}

impl AppState {
    pub fn bootstrap(repo_root: PathBuf, backend_kind: BackendKind) -> Self {
        let worktree_root = repo_root.join(".swarm").join("worktrees");
        let personas = Persona::default_distribution();
        let agents: Vec<AgentState> = personas
            .iter()
            .enumerate()
            .map(|(i, p)| AgentState::new(i as u8, *p, &worktree_root))
            .collect();
        let settings = Settings::default();
        let slots = (0..settings.scheduler.n_live_slots)
            .map(SlotOccupancy::empty)
            .collect();
        Self {
            agents,
            master: MasterState::default(),
            tasks: TaskGraph::default(),
            slots,
            gpu: GpuBudget::placeholder(settings.scheduler.n_live_slots as u32),
            git: crate::ipc::GitSummary::placeholder(),
            log: LogRing::default(),
            settings,
            backend_kind,
            repo_root,
        }
    }

    pub fn agent(&self, id: AgentId) -> Option<&AgentState> {
        self.agents.iter().find(|a| a.id == id)
    }

    pub fn agent_mut(&mut self, id: AgentId) -> Option<&mut AgentState> {
        self.agents.iter_mut().find(|a| a.id == id)
    }
}
