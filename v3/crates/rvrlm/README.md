# rvRLM

`rvRLM` is the Rust landing zone for bringing [alexzhang13/rlm](https://github.com/alexzhang13/rlm) into this repo as a real subfeature instead of an external sidecar. The recursive workflow and package shape are attributed to that upstream project; this crate is the repo-local Rust adaptation inside `rvLLM`.

The goal here is not just to port the Python package layout. The goal is to make recursive language-model execution something we can eventually route through this repo's serving and runtime stack.

Current scaffold:

- `src/core`: high-level `Rlm` runner and builder
- `src/clients`: LM client trait plus a stub backend for bring-up
- `src/environments`: REPL/runtime environment traits and a local environment shell
- `src/logger`: trajectory capture for recursive runs
- `src/serve.rs`: a small serving-facing surface so the crate fits the repo's inference direction
- `src/utils`: prompt, parsing, and token-count helpers

Current inference wiring:

- With `--features cuda`, `src/clients/rvllm.rs` reuses an existing repo inference path from `rvllm-runtime`
- It loads `tokenizer.json`, performs runtime bring-up, resolves embedding and argmax kernels, and calls `run_generate(...)`
- `RvllmCudaConfig::from_env()` consumes the repo's standard `RVLLM_*` path variables plus `rvRLM`-specific decode knobs
- `ServeService::from_rvllm_cuda(...)` builds a serving-ready `rvRLM` engine directly from repo model paths

Python-to-Rust map:

- `rlm/core/*` -> `src/core`, `src/types.rs`, `src/config.rs`
- `rlm/clients/*` -> `src/clients`
- `rlm/environments/*` -> `src/environments`
- `rlm/logger/*` -> `src/logger`
- `rlm/utils/*` -> `src/utils`

Near-term next steps:

1. Unify the client path with the generic `Bringup` surface.
2. Implement a real local execution environment instead of the current shell.
3. Add prompt parsing and recursive iteration control that matches the upstream RLM loop.
4. Decide whether `rvllm-serve` should call into `rvrlm` directly or expose it behind a mode flag.
