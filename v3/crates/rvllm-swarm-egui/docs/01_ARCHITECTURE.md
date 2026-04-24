# 01 — Architecture

## Process model

A single OS process. Three classes of threads:

```
+--------------------------------------------------------------+
| OS process: rvllm-swarm-egui                                  |
|                                                              |
|  +---------------+        +----------------+                 |
|  | UI thread     |<------>| Controller     |<--+             |
|  | (eframe)      |  ipc   | thread         |   |             |
|  +---------------+        +----------------+   |             |
|         ^                         |            |             |
|         | events                  | cmds       | events      |
|         |                         v            |             |
|  +------+--------+        +----------------+   |             |
|  | Shared state  |<-------| Worker pool    |---+             |
|  | (Arc<RwLock>) |        | (N_LIVE thds)  |                 |
|  +---------------+        +----------------+                 |
|                                   |                          |
|                                   | cuda / fs / git          |
|                                   v                          |
|                          +-------------------+               |
|                          | rvllm runtime     |               |
|                          | + tool sandbox    |               |
|                          +-------------------+               |
+--------------------------------------------------------------+
```

- **UI thread**: the eframe event loop. Only reads the shared state and
  sends `Cmd` messages. Never blocks on backend calls.
- **Controller thread**: owns the task queue, the journal, the dispatch
  policy, and git `main`. Sends `WorkerCmd` to live workers. Reads
  `WorkerEvent` from them. Publishes `AppEvent` to the UI.
- **Worker threads**: one per live slot (default 4). A given thread is
  *bound to a slot*, not to a logical agent: when the scheduler evicts
  agent A from the slot and loads agent B, the thread's memory for the
  `ServeService` is dropped and replaced. Agent A's `AgentState` (history,
  last task, tools used) lives on in shared state.

## Shared state

```rust
pub struct AppState {
    pub agents: Vec<AgentState>,          // 30 workers, indexed by slot
    pub master: MasterState,              // the 31st agent
    pub tasks: TaskGraph,                 // DAG of tasks + status
    pub slots: Vec<SlotOccupancy>,        // N_LIVE entries, which AgentId is loaded
    pub gpu: GpuBudget,                   // HBM usage, kv usage
    pub log: RingBuffer<LogLine>,         // global log for side panel
    pub settings: Settings,               // n_live, max_new_tokens, etc.
}
```

Wrapped in `Arc<parking_lot::RwLock<AppState>>`. UI acquires read locks in
`App::update`; controller and workers acquire write locks when transitioning
states. Contention is low because work is measured in hundreds of
milliseconds per state change.

`parking_lot` (not `std::sync`) because the UI loop runs at 60 Hz and we
want the cheap unfair lock.

## Channels

Two `crossbeam-channel` channels:

- `Sender<Cmd> -> controller`: one sender cloned into the UI, unbounded.
  `Cmd` variants include `SubmitGoal`, `PauseAgent`, `ResumeAgent`,
  `KillTask`, `AdjustNLive`, `MergeApproved`, `ForceEvict`.
- `Sender<AppEvent> -> ui`: produced by controller and workers, consumed in
  `App::update`. Bounded at 4096 to prevent runaway growth if the UI stalls
  briefly during a frame.

Between controller and each worker thread: a per-slot pair of bounded
channels (`Sender<WorkerCmd>` / `Receiver<WorkerEvent>`), capacity 64.

See `09_IPC_PROTOCOL.md` for the full enum surface.

## HBM budget

Empirically on an H100 SXM 80 GB:

| Item                                   | Size           |
|----------------------------------------|----------------|
| Gemma 4 E4B FP8 weights (shared)       | ~4.3 GB        |
| Activation scratch (shared)            | ~0.5 GB        |
| Per-agent KV cache @ ctx=8k            | ~1.2 GB        |
| CUDA graph pool + misc                 | ~1.5 GB        |
| Headroom / fragmentation               | ~2 GB          |

With these numbers `N_LIVE = 4` leaves well over 60 GB free and lets us
double ctx or add a speculative draft model later. `N_LIVE = 8` is safe at
ctx=4k. `N_LIVE = 16` requires ctx=2k and tight KV accounting and is the
stretch goal.

The budget is a `SchedulerConfig` struct and is **not auto-tuned**; the user
sets it in the side panel. The scheduler refuses to overcommit.

## Feature gates

- No features: mock backend, pure UI + scheduler. `cargo run` works on a
  Macbook.
- `--features cuda`: compiles in the native rvllm backend. Requires all
  `RVLLM_*` env vars to be set at startup; the app
  validates them eagerly and shows a blocking error screen if any are
  missing.

## Crash model

- If a **worker** panics: the slot transitions to `Failed`. The controller
  can re-load the agent (button: "Reset agent"). The UI shows the panic
  message in the agent's tile.
- If the **controller** panics: the whole process exits with an error. On
  relaunch, `.swarm/tasks.jsonl` is replayed to rebuild the task graph up
  to the last durable state.
- **OOM on the GPU**: caught in the worker, reported to the controller,
  which reduces `N_LIVE` by one and re-queues.

## File layout at runtime

```
<repo>/
├── .swarm/
│   ├── tasks.jsonl          # append-only task journal
│   ├── goals.jsonl          # top-level user goals
│   ├── agents/<uuid>/
│   │   ├── persona.md       # worker persona, editable at rest
│   │   ├── history.jsonl    # per-agent message/tool history
│   │   └── last.json        # last task + last result
│   └── worktrees/
│       └── agent-<uuid>/    # git worktree for that worker
└── ... rest of the repo
```

See `10_PERSISTENCE.md` for the exact formats.
