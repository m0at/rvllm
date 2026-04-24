# 09 — IPC protocol

All messages live in `src/ipc.rs`. They are plain Rust enums, `Debug +
Clone`, with `serde` derives (for journaling, for the `swarm-cli` headless
path, and for future HTTP bridging).

## Top-level channels

```
UI ──Cmd──▶ Controller
Controller ──AppEvent──▶ UI
Controller ──WorkerCmd──▶ Worker[i]
Worker[i] ──WorkerEvent──▶ Controller
```

The UI never talks to a worker directly. This is non-negotiable: it
keeps the state machine tractable.

## `Cmd` (UI → controller)

```rust
pub enum Cmd {
    SubmitGoal { text: String },
    SubmitBroadcast { text: String },    // one shared response fanned out to all agents
    CancelGoal { goal: GoalId },
    RetryTask { task: TaskId },
    PauseAgent { agent: AgentId },
    ResumeAgent { agent: AgentId },
    ResetAgent { agent: AgentId },        // drop & rebuild
    ForceEvict { agent: AgentId },
    PinAgent { agent: AgentId, pinned: bool },
    AdjustScheduler { n_live: usize },
    ApproveMerge { goal: GoalId },        // operator gate on main merge
    SetPersona { agent: AgentId, persona: Persona },
    SendDirectMessage { agent: AgentId, text: String },  // power-user escape
    PruneWorktrees,
    Shutdown { clean: bool },
}
```

## `AppEvent` (controller → UI)

```rust
pub enum AppEvent {
    // Bulk snapshot sent once after UI subscribes (boot or reconnect).
    Snapshot(AppSnapshot),

    // Master / goal lifecycle
    GoalCreated(Goal),
    GoalStatusChanged { goal: GoalId, status: GoalStatus },
    MasterActivity(MasterActivity),

    // Task lifecycle
    TaskCreated(Task),
    TaskStatusChanged { task: TaskId, status: TaskStatus },
    TaskAssigned { task: TaskId, agent: AgentId },
    TaskResult { task: TaskId, result: TaskResult },

    // Agent lifecycle
    AgentLifecycle { agent: AgentId, lifecycle: AgentLifecycle },
    AgentMetrics { agent: AgentId, metrics: AgentMetrics },
    AgentLog { agent: AgentId, line: LogLine },

    // Scheduler / GPU
    SlotChanged(SlotOccupancy),
    GpuBudget(GpuBudget),

    // Git
    GitSummary(GitSummary),

    // Global log
    Log(LogLine),

    // Errors
    Error { context: String, message: String },
}
```

`AppSnapshot` is the whole `AppState` minus the global log, sent once at
subscribe time so the UI can render without flicker. After that, events
are deltas.

## `WorkerCmd` (controller → worker)

```rust
pub enum WorkerCmd {
    LoadAgent { agent: AgentId, persona: Persona, worktree: PathBuf, hbm_budget: HbmBudget },
    UnloadAgent,
    RunTask { task: Task, prompt: String },
    Interrupt,                        // soft stop the current completion
    SetRuntimeSettings(RuntimeSettings),
    Ping,                             // heartbeat
}
```

## `WorkerEvent` (worker → controller)

```rust
pub enum WorkerEvent {
    Loaded { agent: AgentId, elapsed: Duration },
    LoadFailed { agent: AgentId, error: String, oom: bool },
    Unloaded { agent: AgentId },
    Started { task: TaskId, at: Instant },
    Iteration { task: TaskId, index: u32 },
    ToolCall { task: TaskId, call: ToolCallRecord },
    PartialResponse { task: TaskId, text: String },
    Finished { task: TaskId, outcome: TaskOutcome },
    Metrics { agent: AgentId, metrics: AgentMetrics },
    Pong,
    Error { task: Option<TaskId>, message: String },
}
```

## `TaskOutcome`

```rust
pub enum TaskOutcome {
    Success {
        response: String,
        tokens_in: u64,
        tokens_out: u64,
        duration: Duration,
        files_touched: Vec<PathBuf>,
        diff_summary: String,
    },
    NeedsRevision { reason: String },   // worker asking for more context
    TimedOut,
    Failed { message: String },
}
```

## Serde

All structs used in `AppEvent` and `WorkerEvent` derive
`Serialize + Deserialize`. This lets us:

- journal a subset into `tasks.jsonl`,
- replay an entire session for the `swarm-cli` tool (`--replay <path>`),
- later expose a thin `tokio::net::UnixStream` bridge so a second viewer
  process can subscribe (out of scope for v1 but trivial once the types
  are in place).

## Versioning

Every journal record carries `protocol_version: u16`. Today: `1`. Bumps
require a migration in `controller/journal.rs::load()`. The code refuses
to load journals from a newer version with a clear error.

## Backpressure

The event channel from workers back to the controller is bounded at 64.
If a worker tries to push while it's full, the send blocks. This is a
deliberate choice: it rate-limits a runaway tool-call loop.

The event channel from controller to UI is bounded at 4096. A blocked UI
(rare, but possible during paint storms) will briefly block the
controller's emit. That's fine because events the UI doesn't see in time
are irrelevant anyway.
