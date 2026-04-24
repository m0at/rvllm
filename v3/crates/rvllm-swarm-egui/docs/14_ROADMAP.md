# 14 — Roadmap

## v0.1 — "Scaffold + mock" (this crate's first cut)

- [x] Workspace-independent standalone crate under
      `v3/crates/rvllm-swarm-egui/`.
- [x] eframe shell, 6x5 grid, top bar, side panel, theme.
- [x] `AppState` with 30 workers + 1 master.
- [x] Mock backend driving the full lifecycle with fake tok/s.
- [x] Append-only journal with replay.
- [x] `swarm-cli` headless binary.
- [x] Unit tests for journal + scheduler + IPC serde.
- [ ] Screenshots at 4K and 1440p.

## v0.2 — "Real backend on H100"

- [x] Wire the native rvllm backend behind `--features cuda`.
- [ ] Warm all 4 live slots at startup.
- [ ] Per-agent Rhai tool registration with real sandbox paths.
- [ ] Git worktree create/switch/remove wired to real git2 or
      shell-out.
- [ ] Master decomposition prompt and JSON parsing.
- [ ] Master verification prompt and `revise/accept/reject` logic.
- [ ] Cherry-pick + squash-merge pipeline.

## v0.3 — "Usable for a real workday"

- [ ] Persona edit reload.
- [ ] Auto-merge toggle and a "Review mode" dialog that shows the diff
      before the operator approves.
- [ ] NVML telemetry in top bar.
- [ ] Replay mode in the UI: open a prior `.swarm/` and scrub through.
- [ ] `SendDirectMessage` power-user escape wired to a chat drawer on
      the detail modal.

## v0.4 — "Parallel decoding"

- [ ] Batched decode across `N_LIVE` agents: coalesce ready-to-step
      agents into a single forward pass per iteration.
- [ ] Tool-call preempts rebuild batch at next iteration boundary.
- [ ] Speculative decode (draft head) shared across batched agents.

## v0.5 — "Multi-host"

- [ ] Split master onto one GPU, workers onto another (or a cluster).
- [ ] `tokio::net::UnixStream` bridge so a second UI can observe a
      running controller.
- [ ] Optional web UI that consumes the same event stream (re-use the
      in-repo `chat-client/` pattern).

## Out of scope (forever)

- Letting the swarm push to remotes.
- Letting the swarm install packages or hit the network.
- "Agent-to-agent chat." Coordination is through the controller.
- Auto-escalation of allow-listed shell commands.

## Known open questions (file issues against these)

1. **Persona drift.** Once we let operators edit `persona.md` and
   re-dispatch, how do we avoid divergent personas accumulating
   contradictory instructions over a long session? Proposal: persona
   files are append-only logs with a "compact" command that runs the
   master on them.
2. **Verification cost.** Master verification of every subtask is
   expensive. Should we verify only on a sample? Proposal: verify all
   subtasks that change >N lines, sample the rest.
3. **Rollback discipline.** If the master chooses `reject`, what do we
   do with the agent's branch? Keep for postmortems? Proposal: keep
   rejected branches until the goal terminates, then garbage-collect.
4. **Non-reproducible tool calls.** `cargo check` depends on the system.
   Should we snapshot the toolchain per-worktree? Out of scope for v1;
   assume the operator keeps a stable environment.
