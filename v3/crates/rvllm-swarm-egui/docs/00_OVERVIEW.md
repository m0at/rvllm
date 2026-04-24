# 00 — Overview: a 30-agent 4B swarm on one H100

## The one-sentence version

`rvllm-swarm-egui` is a native Rust desktop application that turns a single
H100 into a **workshop of 31 cooperating 4B agents** (1 master + 30 workers),
gives each worker real filesystem/git/shell tools inside its own git
worktree, and visualises the entire swarm on a 45" monitor so a human
operator can steer the work, intervene on individual agents, and keep the
merge of their output coherent. It supports two execution modes: an
`operator-30` mode that maps the visible 30-agent view onto a 30-row decode
batch, and a `saturator` mode that keeps the same 30-agent operator model
while driving a larger on-card decode batch for H100 throughput tests.

## Why this shape

1. **4B is the sweet spot.** Gemma 4 E4B on H100 fits with very small
   per-agent KV overhead; B=1 latency is low, and large decode batches can
   push the card hard. You can run *many* logical 4B agents on one card
   before you saturate it, whereas one 31B agent uses the whole thing.

2. **One strong master, many cheap workers** is the classical swarm topology
   that actually works. The master decomposes goals into concrete subtasks
   and routes them. Workers are narrowly scoped and, critically, **have
   tools**: they can read files, write files, run `cargo check`, run `rg`,
   commit to their own git worktree. A 4B agent with tools massively
   out-performs a 4B agent without tools, because the tool output does the
   reasoning the parameters cannot.

3. **A GUI, not a log file.** At 30 agents the only way a human stays in
   the loop is spatial: tiles, colours, a grid. Watching `stdout` from 30
   processes is impossible. A 45" monitor is enough pixels to see all 30 at
   once.

4. **Git worktrees, not branches in one checkout.** Each agent gets its
   own working directory (`/repo/.swarm/worktrees/agent-<id>/`) on the same
   repository. This gives hard filesystem isolation between agents working
   on the same codebase, no `git checkout` races, and a trivial merge story
   (the master is the only process that touches `main`).

5. **The master is itself a 4B agent.** We treat decomposition as just
   another LLM call. The master has a specialised persona and a much
   smaller tool surface (read-only on worktrees, authoritative on the
   task journal and on merges). This means the whole system is uniform:
   same engine, same prompts, same metrics everywhere.

## What the UI shows, at a glance, on a 45" monitor

```
+-----------------------------------------------------------------------------+
| MASTER STRIP                                                                 |
|  [master-01]  current goal: "wire the CUTLASS 64x64x128 tile"                |
|  sub-tasks in flight: 7   queued: 11   done today: 184   merges: 23          |
|  GPU: 4/4 live slots   KV: 12.3/14 GB   tok/s swarm: 412                     |
+-----------------------+----------------------+----------------------+--------+
| agent-01              | agent-02             | agent-03             | agent-04
| persona: kernels      | persona: tests       | persona: docs        | persona: runtime
| state: RUNNING        | state: QUEUED        | state: RUNNING       | state: SWAPPED
| task: "add 64x64 tile"| task: "cover epilog" | task: "update SPEC"  | task: (none)
| tok/s: 78   iter 3/6  | waiting for slot     | tok/s: 71   iter 2/6 | evicted 00:12 ago
| last tool: cargo_check| -                    | last tool: fs_write  | -
| worktree: agent-01    | worktree: agent-02   | worktree: agent-03   | worktree: agent-04
| flash: "...compiling" |                      | flash: "...saved"    |
+-----------------------+----------------------+----------------------+--------+
...5 more rows of 6 tiles each ... =====================================>
+-----------------------------------------------------------------------------+
| DISPATCH / LOG SIDE PANEL   (global goal input, log stream, git overview)    |
+-----------------------------------------------------------------------------+
```

Colours (defined in `theme.rs`):

| State    | Meaning                                 | Colour           |
|----------|-----------------------------------------|------------------|
| IDLE     | loaded, waiting for work                | dim grey border  |
| QUEUED   | task assigned, awaiting live slot       | yellow border    |
| LOADING  | being paged into HBM                    | blue border      |
| RUNNING  | actively decoding                       | orange accent    |
| SWAPPED  | evicted from HBM, state preserved       | purple border    |
| FAILED   | error, needs human attention            | red border       |

## What happens when the user types a goal

1. Operator types into the top bar: *"port the vLLM FP8 channelscale tile
   autotuner to our CUTLASS kernel; ship it to main"*.
2. UI thread sends `Cmd::SubmitGoal { text }` to controller.
3. Controller appends the goal to `.swarm/tasks.jsonl` with id `goal-<uuid>`
   and status `decomposing`.
4. Controller invokes the **master agent** with the goal plus a
   decomposition prompt. Master returns a JSON DAG of sub-tasks, each with
   `{ id, summary, persona_hint, depends_on[] }`.
5. Controller writes each subtask to the journal with status `queued`.
6. Controller runs the dispatch loop: for each subtask with satisfied deps,
   pick a suitable worker (see `02_MASTER_AGENT.md`). If all live slots are
   full the task stays `queued`.
7. When a slot frees, controller tells the scheduler to load the chosen
   agent's engine (or reuse if already live), writes the subtask prompt into
   the worker's inbox, and flips the tile to `RUNNING`.
8. The worker decodes via `ServeService::complete(...)`, calling tools as
   needed. Each tool call is logged to the journal and streams back to the
   UI as `Event::AgentLogAppended`.
9. On worker success the controller checks the subtask's exit criteria
   (compile? tests pass? diff in range?). If OK, master is asked to verify,
   then the controller cherry-picks the worktree commit onto a staging
   branch.
10. When all subtasks for a goal are verified, the master is asked once more
    to produce a single commit message for the squash-merge. The controller
    performs the merge to `main`. The goal transitions to `done`.

## Execution modes

- **`operator-30`** keeps the runtime and the UI aligned: 30 visible worker
  agents, target decode batch 30. Use this when we want the operator view to
  correspond directly to what the backend is serving.
- **`saturator`** keeps the same 30 visible worker agents but targets a
  larger on-card decode batch, currently 512 by default. Use this for H100
  throughput experiments where the visible agents are the harness and the
  extra rows are load-generation lanes.

Both modes use one rvLLM runtime path. The difference is how aggressively the
controller asks the backend to batch work on-card.

## Non-goals (v1)

- **No independent model replicas per worker.** The visible agents are logical
  workers over a shared runtime and scheduler, not 30 separate GPU-resident
  model copies.
- **No network tools for workers.** No HTTP, no pip, no curl. Reading the
  filesystem and running whitelisted commands is plenty.
- **No autonomous `main` pushes.** The controller cannot push to a remote.
  A human operator does that from the UI after review.
- **No cross-agent chat.** Workers do not talk to each other. All
  coordination flows through the controller and the task journal. This is
  intentional: emergent multi-agent chatter is the single biggest waste of
  tokens in this kind of system.

## Reading order

1. `01_ARCHITECTURE.md` — processes, threads, channels, HBM budget.
2. `08_UI_LAYOUT.md` — how the 45" grid is actually sized.
3. `07_TASK_GRAPH.md` — task IDs and dependencies.
4. `02_MASTER_AGENT.md` and `03_WORKER_AGENT.md` — the two personas.
5. Everything else as reference when you pick up a sub-swarm ticket.

`15_SUB_SWARM_PLAN.md` explains how 16 implementation agents parallelise
building *this very crate* — it is the dogfood path.
