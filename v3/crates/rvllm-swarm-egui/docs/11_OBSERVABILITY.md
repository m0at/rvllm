# 11 — Observability

## Metrics we care about

Three scopes: **per-agent**, **per-goal**, **global**.

### Per-agent (`AgentMetrics`)

| Name                     | Unit     | Source                                   |
|--------------------------|----------|------------------------------------------|
| `tok_s_decode`           | tok/s    | native rvllm runtime metrics             |
| `tok_s_prefill`          | tok/s    | native rvllm runtime metrics             |
| `tokens_out_total`       | count    | accumulated across session               |
| `tokens_in_total`        | count    | accumulated across session               |
| `iterations_last`        | count    | last task's iteration count              |
| `iterations_mean`        | count    | running mean over last 20 tasks          |
| `tool_calls_last`        | count    | last task's tool call count              |
| `tool_err_rate`          | ratio    | last 50 tool calls                       |
| `hbm_resident`           | bool     | is the agent currently in a live slot    |
| `hbm_residency_pct_5m`   | %        | fraction of last 5 min spent resident    |
| `tasks_accepted`         | count    | lifetime                                 |
| `tasks_rejected`         | count    | lifetime                                 |
| `tasks_timed_out`        | count    | lifetime                                 |

These render in the tile: `tok/s` and `iterations_last` on the metric
row; the rest are accessible in the detail modal.

### Per-goal (`GoalMetrics`)

| Name                 | Unit    | Source                        |
|----------------------|---------|-------------------------------|
| `subtasks_total`     | count   | from decomposition            |
| `subtasks_done`      | count   | from journal                  |
| `subtasks_failed`    | count   | from journal                  |
| `elapsed_s`          | seconds | `now - goal.created_at`       |
| `tokens_total`       | count   | sum across tasks              |
| `lines_touched`      | count   | from cumulative diff          |
| `merges_attempted`   | count   | controller bookkeeping        |
| `merge_ok`           | bool    |                               |

Rendered in top bar and in the goal-detail area of the side panel.

### Global (`GpuBudget` + `SwarmMetrics`)

| Name                | Unit       | Source                                        |
|---------------------|------------|-----------------------------------------------|
| `hbm_used`          | GB         | sum of resident agents' KV + shared weights   |
| `hbm_total`         | GB         | fixed from `nvml` at startup (or config)      |
| `live_slots`        | count      | `SchedulerConfig::n_live_slots`               |
| `live_slots_used`   | count      | from `SlotOccupancy`                          |
| `swarm_tok_s`       | tok/s      | sum of currently-decoding agents              |
| `queue_depth`       | count      | `queued` tasks                                |
| `in_flight`         | count      | `running` or `verifying` tasks                |
| `ppl_spot`          | ratio      | last perplexity spot-check (see below)        |

## Where metrics come from

- The native rvllm backend returns throughput, tokens in/out, and
  iteration counts. We harvest these on every `Finished` event.
- The controller keeps per-agent running means in
  `AgentMetrics::iterations_mean` etc.
- `GpuBudget` is computed in the scheduler on every slot transition.

## Perplexity spot-check

Optional. On a schedule (operator-configurable, default every 5 minutes
while work is in flight) the master agent is asked to run PPL on a short
canary prompt (`RVLLM_PPL_PROMPT` env var or a built-in default). The
result is pushed as `AppEvent::PplSpot` and rendered in the top bar. A
sudden jump in PPL is a red flag that weight corruption or a kernel
regression crept in; it's cheap insurance.

## Logging

`tracing` with `tracing-subscriber` `EnvFilter`. Default:
`RUST_LOG=rvllm_swarm_egui=info,warn`.

All events that land in `AppState::log` are also emitted at `info`. The
side panel's log is the **in-memory ring**; the persistent record is in
`tasks.jsonl` + per-agent `history.jsonl`. Deliberately two different
things: the UI ring is compact and human-scannable; the journal is
structured and machine-queryable.

## GPU telemetry (optional)

When the `nvml` feature (not shipped in v1; a placeholder in
`observability/nvml.rs`) is enabled, the app also renders:

- SM utilisation %,
- memory bandwidth utilisation,
- power draw (W),
- temperature (°C).

These go in a small always-visible strip under the slot indicators.
Without NVML they're simply omitted.

## Timings we care about debugging

The developer-facing debug overlay (hotkey `F9`) shows:

| Thing                              | Why                                             |
|------------------------------------|-------------------------------------------------|
| UI frame budget (ms)               | catch egui regressions                          |
| Time-to-dispatch (submit → running)| catches scheduler starvation                    |
| Load time (cold / warm)            | spots KV regressions                            |
| Journal fsync time                 | alerts to disk saturation                       |
| Master verification time           | 4B verifying at 8k context, watch for blowups   |

## Export

A "Export metrics" button in the side panel writes a single JSON summary
covering the current session to `.swarm/snapshots/metrics-<iso>.json`.
Good for pasting into a README when benchmarking a new policy.
