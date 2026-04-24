# 02 — Master agent (the controller's brain)

## Role

The master is the 31st 4B agent. It is the only agent that:

- reads the current full task graph,
- decides how to decompose a new goal into a subtask DAG,
- chooses which worker gets which subtask,
- verifies a finished subtask before the controller merges it,
- writes the final squash-merge commit message.

The master **does not** run shell commands, does not touch worktrees beyond
read-only diffs, and does not talk directly to workers. It only produces
structured JSON that the deterministic Rust controller interprets.

This is deliberate: the master is just a policy-producing function. The
controller is the authority. An LLM can hallucinate a file path; a
controller cannot hallucinate a git ref. We want hallucinations to bounce
off a Rust type.

## Prompts (versioned, in `src/controller/prompts.rs`)

### Decomposition prompt

Input:
```
GOAL:
<raw user goal>

REPO SUMMARY:
<auto-extracted: top-level dirs, crate names, last 5 commits>

AVAILABLE PERSONAS:
- kernels: CUDA/CUTLASS, PTX, epilogues
- runtime: Rust engine, scheduler, graph capture
- tests: unit/integration, property tests
- docs: markdown, specs, READMEs
- tpu: JAX/XLA paths
- misc: small one-shots, cleanup
```

Output (strict JSON, parsed by `serde_json` into `DecomposedGoal`):
```json
{
  "plan_summary": "short human summary",
  "subtasks": [
    {
      "id": "s1",
      "summary": "...",
      "persona": "kernels",
      "depends_on": [],
      "exit_criteria": "cargo check passes and file src/foo.rs touched",
      "suggested_files": ["v3/crates/.../foo.rs"]
    }
  ]
}
```

`exit_criteria` is a free-text field; the controller interprets a handful
of well-known prefixes (`cargo check passes`, `cargo test <name> passes`,
`file <path> contains <grep>`, `human approval`) and leaves the rest as
advisory.

### Verification prompt

Input:
```
SUBTASK: <summary + exit_criteria>
DIFF:    <git diff of agent worktree vs main, truncated to 8k tokens>
LOG:     <last 30 lines of agent log, truncated>
```

Output:
```json
{
  "verdict": "accept" | "revise" | "reject",
  "rationale": "...",
  "needed_fixups": ["short imperative sentence", ...]
}
```

On `revise` the controller re-queues the same agent with the fixups
appended to the prompt. On `reject` the controller marks the subtask
failed and surfaces it to the UI for human attention.

### Merge commit prompt

Input: the set of subtask summaries and the aggregated diff summary (files
touched, lines added/removed).

Output: a single-paragraph commit message in the style of the repo's last
10 commits.

## Dispatch policy

When a subtask becomes dispatchable (all deps satisfied) the controller
asks the master — *no, it does not*, and this is important. Dispatch is
**pure Rust**, no LLM call. The reason: dispatch runs hundreds of times
per goal and must be cheap and deterministic. The rules, in priority order:

1. **Persona match.** Prefer an agent whose `persona_hint` equals the
   subtask's `persona`.
2. **Worktree locality.** If an agent is already on the right feature
   branch and has recent commits touching overlapping files, prefer it.
3. **Load.** Among ties, prefer idle agents over queued ones, prefer
   agents with shorter recent history, prefer agents already resident in
   an HBM slot.
4. **Stickiness.** Avoid evicting an agent that is mid-task.

If no good match exists the controller picks the least-loaded agent and
mutates its persona via a one-shot system message ("for this task, act as
X"). A dynamic persona is less effective but keeps throughput high.

## Why not let the master do dispatch?

- It's a hot path; LLM latency would dominate.
- Dispatch decisions are easy to express as code once the persona field
  exists. 4B reasoning cost is not worth the marginal improvement.
- Determinism simplifies debugging; a bad dispatch is always reproducible
  from the task journal.

We *do* let the master do one LLM-heavy job: **replan**. If the controller
detects that >30 % of subtasks under a goal have been rejected/revised, it
interrupts dispatch, sends the full journal state for that goal back to
the master with a `replan` prompt, and replaces the open subtasks with
whatever the master returns. This is the escape hatch when the first
decomposition was wrong.

## Master state in `AppState::master`

```rust
pub struct MasterState {
    pub current_goal: Option<GoalId>,
    pub decomposition_in_flight: bool,
    pub verification_in_flight: Option<TaskId>,
    pub last_response: Option<String>,      // for the master strip in the UI
    pub calls_today: u64,
    pub total_tokens_today: u64,
}
```

The master strip at the top of the UI renders from this struct. When
`decomposition_in_flight` is `true`, the strip shows a subtle pulse and
the submit button is disabled.

## Isolation

The master uses its own `ServeService` instance. It occupies its own
dedicated HBM slot that is not counted against `N_LIVE` for workers. That
slot is always resident; the master is never swapped. This costs us one
agent's worth of HBM but removes a whole class of scheduling deadlocks
(the master needing to verify a subtask but being evicted because it was
idle).
