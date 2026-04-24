# 05 — Git worktrees

## Why worktrees

Branches alone aren't enough: with one checkout and 30 agents writing
concurrently, `git checkout` races would corrupt each other. `git worktree`
gives each agent its own working directory on the same underlying object
store. Commits are shared; working trees are not.

## Layout

```
<repo>/                          # main working tree, untouched by workers
├── .git/
├── .swarm/
│   └── worktrees/
│       ├── agent-<uuid-1>/      # full checkout, own HEAD
│       ├── agent-<uuid-2>/
│       └── ...
```

The `.swarm/worktrees/` directory is listed in the repo's `.gitignore`
indirectly: the worktrees are registered with git but their paths are
outside of any tracked directory and git's own worktree machinery handles
the bookkeeping in `.git/worktrees/`. We don't need to add anything to the
main repo's `.gitignore`.

## Agent lifecycle hooks

At agent first-boot:

```bash
git worktree add -b agent-<uuid>/base .swarm/worktrees/agent-<uuid>
```

At task dispatch:

```bash
cd .swarm/worktrees/agent-<uuid>
git fetch origin main --quiet
git switch -c agent-<uuid>/<task-id> main
```

During task execution the worker may only modify files under its own
worktree. Tool sandbox enforces this (see `12_SAFETY_AND_SANDBOX.md`).

At task completion (controller, not worker):

```bash
cd .swarm/worktrees/agent-<uuid>
git add -A
git -c user.name="rvllm-swarm" \
    -c user.email="swarm@rvllm.local" \
    commit -m "<agent-id> <task-id>: <task.summary>"
```

On verification success the controller cherry-picks the commit onto a
staging branch in the main checkout:

```bash
cd <repo>
git fetch .swarm/worktrees/agent-<uuid> agent-<uuid>/<task-id>
git cherry-pick FETCH_HEAD          # onto staging/<goal-id>
```

On full goal completion the controller squash-merges staging onto main:

```bash
git switch main
git merge --squash staging/<goal-id>
git commit -m "<master-agent composed message>"
```

## Conflict handling

If a cherry-pick fails with a conflict:

1. The controller aborts the cherry-pick (`git cherry-pick --abort`).
2. The subtask goes to `NeedsReview`.
3. The master is shown the diff and the conflict files and asked to
   produce a `fixups` list.
4. The original agent is re-dispatched with the fixups appended.

We never resolve conflicts automatically. Getting a 4B agent to do correct
three-way merges is a research project; we punt.

## Worktree health probe

A background controller tick every 10 s runs `git worktree list --porcelain`
and compares against `AppState::agents`. Any unexpected worktree is
reported as `orphaned_worktree` in the side-panel log and kept until the
operator clicks "Prune" (which runs `git worktree remove`). Workers whose
worktree vanished transition to `Failed`.

## Cleanup

On `AppState::shutdown_clean = true` (checkbox in UI) the controller, on
exit, iterates agents and runs `git worktree remove --force` for each.
Defaults to `false` so you can inspect state across restarts.

## Size considerations

Each worktree is a full checkout. On a 500 MB repo, 30 worktrees is 15 GB
of disk. That's fine for any dev machine but worth flagging. If the repo
grows past a few GB we switch to `git worktree add --no-checkout` and
materialise only what each agent touches via sparse-checkout — there's a
TODO line in `worker/worktree.rs` for this optimisation.
