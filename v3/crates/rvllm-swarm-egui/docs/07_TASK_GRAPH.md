# 07 — Task graph

## Shape

A **goal** is what the human types. A goal decomposes into a DAG of
**tasks**. Each task has zero or more dependencies on other tasks in the
same goal. A task never depends on a task in a different goal.

```
Goal g1 (user: "add FP8 64x64 tile")
 ├── Task t1: "skeleton kernel .cu"           (deps: [])
 ├── Task t2: "autotune entry in policy.json" (deps: [t1])
 ├── Task t3: "unit test for B=1..8"          (deps: [t1])
 └── Task t4: "update kernels README"         (deps: [t1, t2, t3])
```

A task's **status** transitions monotonically through:

```
 queued -> running -> {verifying -> {accepted | revising -> running | rejected}
                      | timed_out | failed}
```

`accepted` is terminal-success. `rejected` is terminal-failure. `revising`
loops back to `running` with added instructions.

## Types

```rust
pub struct GoalId(pub Uuid);
pub struct TaskId(pub Uuid);

pub struct Goal {
    pub id: GoalId,
    pub submitted_by: String,         // "human" or agent id (future: sub-goals)
    pub text: String,
    pub created_at: OffsetDateTime,
    pub plan_summary: Option<String>, // filled after decomposition
    pub status: GoalStatus,           // Decomposing|Running|Merging|Done|Failed
}

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
```

## Storage

In memory: `TaskGraph` inside `AppState`, a plain adjacency structure:
`HashMap<GoalId, Goal>`, `HashMap<TaskId, Task>`, and per-goal
`Vec<TaskId>` in topo order. Indices are maintained lazily on mutation.

On disk: `.swarm/tasks.jsonl` — one record per line, each record is
either a `Goal` mutation or a `Task` mutation:

```jsonl
{"kind":"goal","op":"create","goal":{...}}
{"kind":"task","op":"create","task":{...}}
{"kind":"task","op":"status","id":"<uuid>","status":"running","at":"2026-04-23T10:31:04Z"}
{"kind":"task","op":"assign","id":"<uuid>","agent":"<uuid>"}
{"kind":"goal","op":"status","id":"<uuid>","status":"done"}
```

Replay order is strict: line N is applied before line N+1, with fail-fast
on unknown kinds. This is how we survive controller restarts: on boot,
replay the journal into `TaskGraph`.

## Dispatchability

A task is **dispatchable** iff:

- its status is `queued`,
- every `depends_on` id has status `accepted`,
- its persona has at least one agent not currently `Failed`,
- the scheduler has at least one slot that can become free without
  violating a `Running` task's non-preemption invariant.

The controller wakes up on every `AppEvent::TaskStatusChanged` and runs
`pick_dispatchable()` to find candidates. Candidates are sorted by
`(priority desc, created_at asc)` and dispatched until either all live
slots are working or the candidate list empties.

## Exit criteria evaluation

The `exit_criteria` string is parsed by a tiny hand-written matcher in
`controller/criteria.rs`:

| Prefix                                   | Meaning                                             |
|------------------------------------------|-----------------------------------------------------|
| `cargo check passes`                     | run `cargo check` in worktree, must exit 0          |
| `cargo test <filter> passes`             | run `cargo test <filter>`, must exit 0              |
| `file <path> contains <grep>`            | `rg -n --fixed-strings <grep> <path>` must match    |
| `file <path> exists`                     | path resolvable from worktree root                  |
| `diff <path> touched`                    | path appears in `git diff --name-only` vs base      |
| `human approval`                         | controller flips task to `NeedsReview`              |
| anything else                            | treated as advisory, master does verification only  |

If every automatic criterion passes, the controller still asks the master
for a verdict. Automatic criteria are necessary; they are not sufficient.

## Why journal-first

1. Crash recovery: the UI can be closed and reopened mid-goal without
   losing state.
2. Human diffability: you can `cat .swarm/tasks.jsonl | jq` and see
   exactly what happened. This is the single most-used debugging tool.
3. Reproducibility: a failed run can be turned into a regression fixture
   by copying the relevant journal lines into `tests/fixtures/`.

## Visualisation

The side panel has a small DAG view (see `08_UI_LAYOUT.md`) that draws
the current goal's subtasks as nodes coloured by status, with edges for
dependencies. Above 30 tasks it switches to a dense list.
