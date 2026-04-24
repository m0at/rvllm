#![allow(dead_code)]

// IPC surface shared by UI, controller, and workers.
//
// See docs/09_IPC_PROTOCOL.md for the full rationale. These types are
// also the on-disk journal schema when wrapped by JournalRecord.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::state::{AgentLifecycle, AgentMetrics, GoalStatus, GpuBudget, Persona, Priority, TaskStatus};

// --- Identifiers ------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct AgentId(pub Uuid);

impl AgentId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    pub fn short(&self) -> String {
        self.0.simple().to_string().chars().take(8).collect()
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct GoalId(pub Uuid);

impl GoalId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    pub fn short(&self) -> String {
        self.0.simple().to_string().chars().take(8).collect()
    }
}

impl Default for GoalId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct TaskId(pub Uuid);

impl TaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    pub fn short(&self) -> String {
        self.0.simple().to_string().chars().take(8).collect()
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct SlotIdx(pub usize);

// --- Goal / Task ------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Goal {
    pub id: GoalId,
    pub submitted_by: String,
    pub text: String,
    pub created_at_ms: i64,
    pub plan_summary: Option<String>,
    pub status: GoalStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub goal: GoalId,
    pub summary: String,
    pub persona: Persona,
    pub depends_on: Vec<TaskId>,
    pub exit_criteria: String,
    pub suggested_files: Vec<PathBuf>,
    pub status: TaskStatus,
    pub assigned_to: Option<AgentId>,
    pub priority: Priority,
    pub revisions: u32,
    pub result: Option<TaskResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskResult {
    pub outcome: TaskOutcome,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub duration_ms: u64,
    pub files_touched: Vec<PathBuf>,
    pub diff_summary: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TaskOutcome {
    Success { response: String },
    NeedsRevision { reason: String },
    TimedOut,
    Failed { message: String },
}

// --- Logging ----------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogLine {
    pub ts_ms: i64,
    pub level: LogLevel,
    pub source: String,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub ts_ms: i64,
    pub name: String,
    pub args: String, // compact JSON string
    pub ok: bool,
    pub elapsed_ms: u64,
    pub brief: String, // short human-friendly
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlotOccupancyReport {
    pub slot_idx: SlotIdx,
    pub occupant: Option<AgentId>,
    pub state: SlotStateKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotStateKind {
    Empty,
    Loading,
    Hot,
    Evicting,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MasterActivity {
    pub kind: MasterActivityKind,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MasterActivityKind {
    Decomposing,
    Verifying,
    Merging,
    Idle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GitSummary {
    pub main_head: String,
    pub staging_head: Option<String>,
    pub open_worktrees: u32,
    pub pending_merges: u32,
}

impl GitSummary {
    pub fn placeholder() -> Self {
        Self {
            main_head: "(main HEAD unknown)".into(),
            staging_head: None,
            open_worktrees: 0,
            pending_merges: 0,
        }
    }
}

// --- Command / Event enums --------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum Cmd {
    SubmitGoal { text: String },
    SubmitBroadcast { text: String },
    CancelGoal { goal: GoalId },
    RetryTask { task: TaskId },
    PauseAgent { agent: AgentId },
    ResumeAgent { agent: AgentId },
    ResetAgent { agent: AgentId },
    ForceEvict { agent: AgentId },
    PinAgent { agent: AgentId, pinned: bool },
    AdjustScheduler { n_live: usize },
    SetExecutionMode {
        mode: crate::state::ExecutionMode,
        decode_batch_target: usize,
    },
    ApproveMerge { goal: GoalId },
    SetPersona { agent: AgentId, persona: Persona },
    SendDirectMessage { agent: AgentId, text: String },
    PruneWorktrees,
    Shutdown { clean: bool },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum AppEvent {
    GoalCreated(Goal),
    GoalStatusChanged { goal: GoalId, status: GoalStatus },
    MasterActivity(MasterActivity),
    TaskCreated(Task),
    TaskStatusChanged { task: TaskId, status: TaskStatus },
    TaskAssigned { task: TaskId, agent: AgentId },
    TaskResult { task: TaskId, result: TaskResult },
    AgentLifecycle { agent: AgentId, lifecycle: AgentLifecycle },
    AgentMetrics { agent: AgentId, metrics: AgentMetrics },
    AgentLog { agent: AgentId, line: LogLine },
    SlotChanged(SlotOccupancyReport),
    GpuBudget(GpuBudget),
    GitSummary(GitSummary),
    Log(LogLine),
    Error { context: String, message: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
#[allow(dead_code)]
pub enum WorkerCmd {
    LoadAgent {
        agent: AgentId,
        persona: Persona,
        worktree: PathBuf,
    },
    UnloadAgent,
    RunTask {
        task: Task,
        prompt: String,
    },
    Interrupt,
    Ping,
    Shutdown,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum WorkerEvent {
    Loaded {
        agent: AgentId,
        elapsed: Duration,
    },
    LoadFailed {
        agent: AgentId,
        error: String,
        oom: bool,
    },
    Unloaded {
        agent: AgentId,
    },
    Started {
        task: TaskId,
    },
    Iteration {
        task: TaskId,
        index: u32,
    },
    ToolCall {
        task: TaskId,
        call: ToolCallRecord,
    },
    PartialResponse {
        task: TaskId,
        text: String,
    },
    Finished {
        task: TaskId,
        result: TaskResult,
    },
    Metrics {
        agent: AgentId,
        metrics: AgentMetrics,
    },
    Pong,
    Error {
        task: Option<TaskId>,
        message: String,
    },
}

// --- Journal record envelope (on-disk) --------------------------------------

pub const PROTOCOL_VERSION: u16 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JournalPayload {
    GoalCreated { goal: Goal },
    GoalPlanned { id: GoalId, plan_summary: String },
    GoalStatus { id: GoalId, status: GoalStatus },
    TaskCreated { task: Task },
    TaskStatus { id: TaskId, status: TaskStatus },
    TaskAssign { id: TaskId, agent: AgentId },
    TaskResult { id: TaskId, result: TaskResult },
    Note { text: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JournalRecord {
    pub ts_ms: i64,
    pub protocol_version: u16,
    #[serde(flatten)]
    pub payload: JournalPayload,
}

pub fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
