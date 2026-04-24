# 10 — Persistence

## Goal

Everything interesting the swarm does is durable. A crash — or a
deliberate Ctrl-C — must not lose state beyond the last in-flight token.

## Directory

Root: `.swarm/` at the repository root. Created on first run.

```
.swarm/
├── tasks.jsonl          # append-only; the source of truth
├── goals.jsonl          # convenience index (also recoverable from tasks.jsonl)
├── agents/
│   └── <agent-uuid>/
│       ├── persona.md
│       ├── history.jsonl
│       └── last.json
├── worktrees/
│   └── agent-<uuid>/    # git worktrees, managed by git
├── snapshots/
│   └── <iso-timestamp>.json   # optional full-state dumps (operator button)
└── version              # "1"
```

The `version` file is a single integer. Bumping it requires a migration
routine, gated in `controller::journal::migrate()`.

## `tasks.jsonl` record schema

Each line is one of the following (see `09_IPC_PROTOCOL.md` for the
typed envelopes). Every record has:

- `ts`: ISO-8601 UTC timestamp, millisecond precision,
- `protocol_version`: `1`,
- `kind`: one of `goal`, `task`, `agent`, `note`,
- `op`: one of `create`, `status`, `assign`, `result`, `error`, `merge`.

Examples:

```jsonl
{"ts":"2026-04-23T10:30:00.123Z","protocol_version":1,"kind":"goal","op":"create","goal":{"id":"...","text":"..."}}
{"ts":"2026-04-23T10:30:01.500Z","protocol_version":1,"kind":"task","op":"create","task":{"id":"...","goal":"...","summary":"..."}}
{"ts":"2026-04-23T10:30:02.111Z","protocol_version":1,"kind":"task","op":"assign","id":"...","agent":"..."}
{"ts":"2026-04-23T10:30:03.002Z","protocol_version":1,"kind":"task","op":"status","id":"...","status":"running"}
{"ts":"2026-04-23T10:31:10.999Z","protocol_version":1,"kind":"task","op":"result","id":"...","outcome":{"tag":"Success","tokens_in":812,"tokens_out":203,"diff_summary":"+12 -3 in src/foo.rs"}}
{"ts":"2026-04-23T10:32:00.000Z","protocol_version":1,"kind":"goal","op":"merge","id":"...","commit":"7d2a91f..."}
```

## Write discipline

- **One writer**: only the controller thread ever opens `tasks.jsonl` for
  write. All other components emit records via `Cmd::Journal(...)` (actually
  via the journal's internal `Sender`).
- **fsync on goal boundaries**: we do not fsync every line (too slow), but
  we fsync on every `goal` record and on `task`+`result`. Worst case on
  crash: lose a few `status` lines, recoverable from worker history.
- **Rotation**: after `10_000` lines the file is rotated to
  `tasks-<iso>.jsonl` and a new file started. Replay walks rotated files
  in order.

## Replay

On startup, if `.swarm/version` exists and matches, the controller
replays (rotated files, then live file) into an empty `TaskGraph`. Unknown
`kind`/`op` combinations hard-fail with a clear error message pointing
at the journal line.

Corrupt last line (half-written due to crash) is tolerated: the replayer
logs it and truncates the file at the last good newline.

## Per-agent history

`.swarm/agents/<uuid>/history.jsonl` has a different, looser schema —
it's for debugging and for giving the UI text to render. Records:

```jsonl
{"ts":"...","role":"system","content":"..."}
{"ts":"...","role":"user","content":"..."}
{"ts":"...","role":"assistant","content":"..."}
{"ts":"...","role":"tool","name":"fs_read","args":"{...}","result":"..."}
```

No fsync. Loss on crash is acceptable.

## `persona.md` round-trip

On agent boot:

1. If `.swarm/agents/<uuid>/persona.md` exists, use it.
2. Else, write the persona template for the persona kind, using the
   default tooling allow-list section, and use that.

The operator can edit this file at any time. The file is re-read on the
next `LoadAgent` for that agent. This is how personas are iterated on.

## Snapshots

A "Snapshot now" button in the UI side panel writes the full in-memory
`AppState` to `.swarm/snapshots/<iso>.json`. Snapshots are inert: they
exist for postmortems and for asking a future version of this app to
"open" a prior session. Not required for correctness.

## Secrets

No secrets are persisted. The GUI does not accept or forward API keys.
If the native rvllm backend needs a HuggingFace token, it is expected in
the environment, not in `.swarm/`.
