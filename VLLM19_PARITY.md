# vLLM 0.19 Parity Push

Goal: stop meandering and force `main` toward one boring default execution method that matches `vLLM 0.19` structurally.

Current pushed baseline:
- branch: `main`
- commit: `6da116677`

Rules for this push:
- no new branches
- no extra worktrees
- no benchmark-only hacks in the default path
- no new env-driven hot-path routing in the default batched lane
- every change lands on `main`

Target architectural state:
1. One default batched lane
   - `BatchedV2` becomes the normal batched path
   - old `Batched` no longer owns default execution

2. Centralized execution ownership
   - runner owns path selection
   - runner owns decode graph dispatch policy
   - worker executes the runner plan instead of recomputing policy

3. Persistent decode metadata path
   - pure decode `BatchedV2` uses persistent flat request/block-table state
   - no nested metadata clone path for V2 replay/capture

4. No per-op hot-path env routing in the default lane
   - default `BatchedV2` policy is fixed at init
   - default runtime does not branch per layer on ad hoc env toggles

5. Keep experimental paths fenced
   - persistent/megakernel/special decode paths remain opt-in only
   - they do not interfere with the boring default batched lane

Definition of done for this push:
- local `main` is clean
- `origin/main` matches local `main`
- only `main` exists locally/remotely
- local validation passes
- H100 proof run is kicked from pushed `main`

Out of scope for this push:
- inventing a new architecture
- chasing microbench-only kernel wins
- changing public docs

Next likely bottleneck after parity cleanup:
- compute-side FFN/backend quality, not metadata staging
