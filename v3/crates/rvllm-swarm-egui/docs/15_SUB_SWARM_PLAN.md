# 15 — Sub-swarm plan: 16 agents building this crate

## Premise

The user's ask ends with: *"think it all the way out with a swarm of 16
agents and write md files at length for each part so we can then swarm
each sub area"*. This doc is the dogfood path: **how we would split the
work of building this crate across 16 implementation agents running on
the swarm this crate describes**. It's also the first real workload the
finished app will run.

Each sub-swarm agent gets:

- a persona (from the 6 canonical personas),
- an ownership scope (one doc, one module, or one file),
- a hard dependency on 0–2 other agents,
- an exit criterion phrased for the controller's criteria parser.

## The 16 sub-swarm agents

| # | Agent name     | Persona  | Scope                                      | Depends on | Exit criterion                                                       |
|---|----------------|----------|--------------------------------------------|------------|----------------------------------------------------------------------|
| 1 | `swarm-ipc`    | Runtime  | `src/ipc.rs` (enums + serde)               | —          | `cargo check passes` and `file src/ipc.rs contains pub enum Cmd`     |
| 2 | `swarm-state`  | Runtime  | `src/state/` (AppState, AgentState, ring)  | 1          | `cargo test state:: passes`                                          |
| 3 | `swarm-theme`  | Docs     | `src/theme.rs` (theme tokens, constants)   | —          | `file src/theme.rs contains TILE_BG_IDLE`                            |
| 4 | `swarm-ui-tile`| Runtime  | `src/ui/tile.rs`                           | 2, 3       | `cargo check passes` and `file src/ui/tile.rs contains fn render`    |
| 5 | `swarm-ui-grid`| Runtime  | `src/ui/grid.rs` (6x5 layout)              | 4          | `cargo check passes`                                                 |
| 6 | `swarm-ui-top` | Runtime  | `src/ui/top_bar.rs` (master strip)         | 2, 3       | `cargo check passes`                                                 |
| 7 | `swarm-ui-side`| Runtime  | `src/ui/side_panel.rs` (goal, log, git)    | 2, 3       | `cargo check passes`                                                 |
| 8 | `swarm-ui-dag` | Runtime  | `src/ui/task_graph.rs` (DAG painter)       | 2, 3, 7    | `cargo check passes`                                                 |
| 9 | `swarm-ui-detail` | Runtime | `src/ui/detail_modal.rs`                 | 2, 3       | `cargo check passes`                                                 |
|10 | `swarm-ctrl`   | Runtime  | `src/controller/mod.rs`, `dispatch.rs`     | 1, 2       | `cargo test controller:: passes`                                     |
|11 | `swarm-journal`| Runtime  | `src/controller/journal.rs`                | 1          | `cargo test journal:: passes` and `file controller/journal.rs contains fn replay` |
|12 | `swarm-sched`  | Runtime  | `src/scheduler/` (LRU + slot pool)         | 1, 2       | `cargo test scheduler:: passes`                                      |
|13 | `swarm-worker` | Runtime  | `src/worker/mod.rs`, `agent.rs` (mock + cuda-gated) | 1, 2 | `cargo check passes` and `cargo check --features cuda passes`        |
|14 | `swarm-tools`  | Runtime  | `src/worker/tools.rs` + shell allowlist    | 13         | `cargo test tools:: passes`                                          |
|15 | `swarm-worktree`| Misc    | `src/worker/worktree.rs` (git wrappers)    | 13         | `cargo test worktree:: passes` (uses `tempfile::TempDir`)            |
|16 | `swarm-cli`    | Tests    | `src/bin/swarm_cli.rs` + `bench/` fixtures | 1, 10, 11, 12, 13 | `cargo run --bin swarm-cli -- --agents 2 --bench micro` exits 0 |

The docs writer is **not** in the 16: the docs (this very file, and the
sixteen others) are the input to the sub-swarm, written by the humans
(or by this assistant before swarming begins). The 16 code-writing
agents treat `docs/*.md` as read-only authoritative spec.

## Dependency DAG

```
 (1) ipc ──┬──► (2) state ──┬──► (4) tile ──► (5) grid
          │                 ├──► (6) top
          │                 ├──► (7) side ──► (8) dag
          │                 └──► (9) detail
          │
          ├──► (10) ctrl ──► (16) cli
          ├──► (11) journal
          ├──► (12) sched
          └──► (13) worker ──┬──► (14) tools
                             └──► (15) worktree
```

The three nodes with no incoming deps (1, 3, 11's lone `ipc` dep, 13's
`state` dep) are eligible for dispatch at t=0. With `N_LIVE = 4` live
slots, the first wave is {1, 3, 11}; as soon as (1) lands its
`cargo check passes` verdict, wave 2 opens ({2, 10, 12, 13}).

## Wall-time estimate

Each sub-task is a moderate edit (100–400 lines plus tests). Estimated
at 4B+tools: 5–15 minutes per task on H100, median ~10. Sixteen tasks in
a DAG with depth ≤ 4 and 4 live slots: wall time ~ depth × median = **~40
minutes**, assuming 80 % of tasks accept on first attempt.

## Sub-swarm persona prompts

Each agent gets a system prompt that composes its canonical persona (see
`src/worker/personas/<persona>.md`) with an **agent-specific appendix**
listing:

- its ownership scope (exact file paths),
- the other docs it must read first (always `00_OVERVIEW.md`,
  `01_ARCHITECTURE.md`, and the doc covering its module),
- a reminder that all editing happens in its worktree, not in the repo
  directly,
- the exit criterion verbatim.

## What the human does

Submit a single goal:

> "Build `rvllm-swarm-egui` v0.1 according to `docs/`. Start from the
> empty `src/` directory. Use the 16-agent plan in `15_SUB_SWARM_PLAN.md`.
> Verify each sub-task against its exit criterion."

The controller's decomposition prompt is fed exactly this goal plus the
contents of this doc. The resulting decomposition is expected to
reproduce the 16-task plan above; if the master produces something
materially different, reject and ask again.

## The meta-test

Once the crate can build itself this way, the swarm has passed its first
real-world acceptance test. That moment is the v0.2 milestone.
