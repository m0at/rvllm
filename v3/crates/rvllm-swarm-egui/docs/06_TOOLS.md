# 06 — Tools (what the workers can actually do)

## Philosophy

A 4B agent with tools is dramatically more useful than a 4B agent without.
Most "reasoning" a coding agent appears to do is actually **observation +
constrained edit**: read a file, run `cargo check`, read the error, edit,
repeat. We lean into that. Tools are plentiful and precise; the model is
small and cheap.

## Tool registration

The swarm crate registers a **different tool set per persona** at
worker boot, via `ToolKit::for_persona(&persona, sandbox)`. The sandbox
handle keeps all tools inside the agent's worktree.

```rust
pub struct ToolKit {
    entries: Vec<ToolEntry>,
}

pub struct ToolEntry {
    pub name: &'static str,
    pub personas: &'static [Persona],    // empty = all
    pub register: fn(&mut rhai::Engine, &SandboxHandle),
}
```

## The tools

### Read-only / universally safe

| Name            | Signature (rhai)                                  | Notes                                    |
|-----------------|---------------------------------------------------|------------------------------------------|
| `fs_read`       | `fs_read(path) -> string`                         | UTF-8 decode, limit 256 KB               |
| `fs_list`       | `fs_list(path) -> array<string>`                  | Non-recursive                            |
| `fs_tree`       | `fs_tree(path, depth) -> string`                  | ASCII tree, capped at 2000 lines         |
| `rg`            | `rg(pattern, path) -> array<match>`               | Shells out to `rg --json`                |
| `git_status`    | `git_status() -> string`                          | Runs in agent's worktree                 |
| `git_log`       | `git_log(n) -> array<commit>`                     | Porcelain parsed                         |
| `git_diff`      | `git_diff(ref) -> string`                         | vs current HEAD                          |

### Mutating (worktree-only)

| Name             | Signature                                       | Notes                                           |
|------------------|-------------------------------------------------|-------------------------------------------------|
| `fs_write`       | `fs_write(path, content)`                       | overwrite, path must be inside worktree         |
| `fs_append`      | `fs_append(path, content)`                      | append                                          |
| `fs_mkdir`       | `fs_mkdir(path)`                                | mkdir -p, inside worktree                       |
| `fs_patch`       | `fs_patch(path, old, new)`                      | exact-match replace, fails on ambiguity         |
| `fs_rm`          | `fs_rm(path)`                                   | single file only, no dirs                       |

Note there is no `git commit` tool: commits are the controller's job so
that authorship is canonical.

### Build/test (persona-gated)

Gated by `ToolEntry::personas`; e.g. only `Runtime`, `Kernels`, `Tests`,
`Misc` can call these.

| Name            | Signature                                     | Notes                                    |
|-----------------|-----------------------------------------------|------------------------------------------|
| `cargo_check`   | `cargo_check(package?) -> result`             | `cargo check -p <package>`, 120 s cap    |
| `cargo_test`    | `cargo_test(filter?) -> result`               | 300 s cap, captured stdout/stderr        |
| `cargo_clippy`  | `cargo_clippy() -> result`                    | clippy with `-D warnings`                |
| `cargo_fmt`     | `cargo_fmt(check) -> result`                  | `check` true for dry-run                 |

These shell out via `std::process::Command` with a wall-clock timeout and
stream last-N-kb back. On a shared H100 machine `cargo` runs on CPU so
the GPU keeps chewing through decode — this is an important invariant and
not a coincidence.

### Shell (tightly restricted)

| Name            | Signature                                     | Notes                                    |
|-----------------|-----------------------------------------------|------------------------------------------|
| `shell`         | `shell(cmd, argv) -> result`                  | cmd must be in allowlist                 |

Allowlist (in `src/worker/tools/shell_allowlist.rs`):
`cat`, `head`, `tail`, `wc`, `find`, `awk`, `sed`, `diff`, `md5sum`,
`sha256sum`, `python3 -c`, `node -e`, `jq`. Everything else is refused.

No `curl`, no `wget`, no `pip install`, no `apt`. The sandbox enforces
CWD = agent's worktree and strips `HOME` / `PATH` to a minimal pair.

### TPU-only

| Name            | Signature                                     | Notes                                    |
|-----------------|-----------------------------------------------|------------------------------------------|
| `jax_import_check` | `jax_import_check(module) -> result`         | only for Tpu persona                     |

## Results

Every tool returns a `ToolResult { success: bool, stdout: String, stderr: String, elapsed_ms: u64 }`
(rhai sees it as a map). On failure the worker includes stderr in its next
prompt; on success, stdout.

## Tool-call logging

Every tool call is written, with arguments redacted where sensible, to:

- `.swarm/agents/<uuid>/history.jsonl` (append),
- `AppState::log` (ring buffer, visible in side panel),
- `AgentState::history` (ring buffer, visible on tile expand).

The on-disk record is the authoritative one; the in-memory rings are for
display.

## Sandbox boundary

The `SandboxHandle` carries the agent's worktree root as an absolute path.
All path arguments to mutating tools are canonicalised and rejected if they
escape the root. Symlink traversal is blocked by refusing to follow
symlinks whose target is outside the root. See `12_SAFETY_AND_SANDBOX.md`
for the full threat model.
