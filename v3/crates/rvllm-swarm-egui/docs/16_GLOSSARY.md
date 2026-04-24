# 16 — Glossary

- **Agent**: a logical 4B worker with a persistent identity, persona,
  history, and git worktree. 30 in the swarm + 1 master = 31.
- **AgentId**: UUID v4, stable across runs, rendered as `agent-<short>`.
- **AppEvent**: a controller → UI message (see `09_IPC_PROTOCOL.md`).
- **AppState**: the single in-memory Rust struct, `Arc<RwLock<_>>`, read
  by the UI and mutated by the controller/workers.
- **Branch**: a per-task git branch created inside an agent's worktree,
  named `agent-<id>/<task-id>`.
- **Cmd**: a UI → controller message.
- **Controller**: the single thread that owns the task queue, the
  journal, and git `main`.
- **Decomposition**: the act of turning a user goal into a subtask DAG,
  performed by the master.
- **E4B**: the Gemma 4 "Effective 4B" model; the model all agents use.
- **Exit criterion**: a string attached to each task describing what
  "done" looks like. A subset is parsed by the controller automatically;
  the rest is advisory for the master.
- **FINAL_ANSWER**: the sentinel used for marking the end of a
  completion.
- **Goal**: a top-level, human-submitted request.
- **HBM**: High Bandwidth Memory — GPU RAM. 80 GB on H100 SXM.
- **History**: per-agent append-only log of all prompts, completions,
  and tool calls. Lives in `.swarm/agents/<id>/history.jsonl`.
- **Journal**: the `.swarm/tasks.jsonl` append-only source of truth.
- **KV cache**: the transformer's key/value cache for a single
  sequence. Per-agent, sized by context length.
- **Live slot**: an HBM slot currently holding an agent's
  `ServeService` + KV. Default `N_LIVE = 4` for workers, +1 pinned for
  the master.
- **LRU**: Least-Recently-Used; default eviction policy.
- **Master**: the 31st agent; the swarm's router and verifier.
- **Mock backend**: a deterministic fake used when the `cuda` feature is
  off. Produces plausible tok/s and toy responses so the UI works on
  any laptop.
- **N_LIVE**: the number of worker slots resident in HBM at any moment.
- **Persona**: one of `Kernels`, `Runtime`, `Tests`, `Docs`, `Tpu`,
  `Misc`; determines system prompt + tool allow-list + default context.
- **Replay**: reconstructing `AppState` by re-applying
  `tasks.jsonl` records in order on startup.
- **Revise**: master verdict that sends the subtask back to the same
  agent with added instructions.
- **SchedulerPolicy**: pluggable trait controlling load/evict. Default:
  LRU. Future: priority-aware, batch-aware.
- **Slot**: one `SlotOccupancy` entry. May be `Empty`, `Loading`, `Hot`,
  or `Evicting`.
- **Subtask (== Task)**: the unit dispatched to a single worker.
- **Swap**: moving an agent out of a live slot. State is preserved; KV
  is discarded (and will be re-prefilled if the agent is re-loaded).
- **Task**: see Subtask.
- **Tile**: one of the 30 cells in the 6x5 UI grid; represents a worker.
- **ToolKit**: the set of rhai-registered tools a worker can call,
  determined by persona.
- **Verdict**: master output for a finished subtask: `accept`, `revise`,
  `reject`.
- **Worktree**: a git `worktree add` directory dedicated to one agent,
  under `.swarm/worktrees/agent-<uuid>/`.
