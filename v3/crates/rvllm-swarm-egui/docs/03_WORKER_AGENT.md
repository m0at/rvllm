# 03 — Worker agent

## What a worker is, precisely

A **worker** is a record in `AppState::agents`, plus (when live) a
`ServeService` bound to a live HBM slot, plus a git worktree on disk. It is
*not* a thread; threads are bound to slots, not to agents.

```rust
pub struct AgentState {
    pub id: AgentId,                       // UUID, stable across runs
    pub display_index: u8,                 // 0..30, slot in the 6x5 grid
    pub persona: Persona,                  // one of 6 canonical personas
    pub lifecycle: AgentLifecycle,         // Idle|Queued|Loading|Running|Swapped|Failed
    pub current_task: Option<TaskId>,
    pub worktree_path: PathBuf,
    pub branch: Option<String>,            // agent-<id>/<task-id>
    pub metrics: AgentMetrics,             // tok/s, tokens out, tool calls, ...
    pub history: History,                  // ring buffer of messages + tool calls
    pub last_error: Option<String>,
}
```

## Worker execution loop

When the controller dispatches `TaskId T` to agent `A`, and the scheduler
has loaded `A` into a live slot, the worker thread executes:

```
 1. Pull task record from journal
 2. Build prompt = persona + task.summary + exit_criteria
                 + relevant files from task.suggested_files (fs_read)
                 + any previous attempts' rejection feedback
 3. agent.service.complete(prompt) with streaming tool calls
 4. For each rhai script block:
      run in sandboxed env (see 06_TOOLS.md)
      stream tool-call event to controller + UI
 5. After FINAL_ANSWER sentinel:
      run exit_criteria check (Rust, not LLM)
      if pass  -> emit TaskOutcome::Success { diff_summary, response }
      if fail  -> emit TaskOutcome::NeedsRevision { reason }
 6. Worker returns to Idle; thread waits for next WorkerCmd
```

The worker thread **does not** make its own git commits. On
`TaskOutcome::Success` the controller inspects the worktree, stages the
changes on the agent's branch, and commits with a canonical message
`<agent-id> <task-id>: <task.summary (first line)>`. This keeps commit
authoring uniform and forgery-proof (the worker can't forge a commit that
looks like it came from another agent).

## Personas (canonical)

Defined in `src/worker/persona.rs` as an enum:

```rust
pub enum Persona {
    Kernels,   // CUDA, CUTLASS, PTX, FP8 epilogues
    Runtime,   // rvllm-runtime, scheduler, graph capture
    Tests,     // unit, integration, property tests
    Docs,      // markdown, specs, inline docstrings
    Tpu,       // JAX/XLA paths under tpu/
    Misc,      // anything small; catch-all
}
```

Each persona has:

- A **system prompt** shipped in-crate under `src/worker/personas/<name>.md`
  (loaded at startup).
- A **tool allow-list** (e.g. `Docs` cannot call `cargo_test`, `Kernels` can).
- A **default context window** (docs personas use 4k, kernels uses 8k to
  read header files).

The 30 worker slots are assigned persona distribution by default:

| Persona  | Count |
|----------|-------|
| Kernels  | 6     |
| Runtime  | 6     |
| Tests    | 6     |
| Docs     | 4     |
| Tpu      | 4     |
| Misc     | 4     |

The operator can rebalance from the side panel.

## Persistence per worker

On disk under `.swarm/agents/<uuid>/`:

- `persona.md` — the exact system prompt this agent was built with. If the
  operator edits it, the next load reads the edit.
- `history.jsonl` — an append-only log of every `(role, content)` in this
  agent's conversations, plus every tool call and result.
- `last.json` — a small snapshot of the most recent completed task for
  quick UI rendering without replaying the whole history.

The worktree under `.swarm/worktrees/agent-<uuid>/` is a real `git worktree
add`; see `05_GIT_WORKTREE.md`.

## Termination

A worker marks its current response complete when it emits the sentinel
`FINAL_ANSWER:` followed by its answer text. If no sentinel appears within
`max_iterations` iterations
the controller marks the task `TimedOut` and surfaces it.

## Failure modes and their display

| Failure                                      | Lifecycle     | Tile hint                                          |
|----------------------------------------------|---------------|----------------------------------------------------|
| Tool call returned non-zero exit             | Running       | tile shows last tool in red; worker keeps trying   |
| Exit criteria not met after `max_iterations` | Idle          | last_error = "no FINAL_ANSWER"; controller reqs master verdict |
| `ServeService::complete` error               | Failed        | full error in tile; requires operator reset        |
| Persona file missing                         | Failed        | hard fail at load                                  |
| Worktree dirty at dispatch                   | Failed (pre)  | controller refuses to dispatch                     |

Workers never retry silently; retries are a controller decision driven by
the master's `revise` verdict.
