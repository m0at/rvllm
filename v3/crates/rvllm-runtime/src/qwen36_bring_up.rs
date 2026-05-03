//! Qwen 3.6 35B-A3B bring-up (Phase 0 scaffolding).
//!
//! Phase 0 contract:
//!   - `Qwen36Bringup::load(paths)` reads + validates `config.json`
//!     via [`Qwen36Arch::from_dir`], logs a summary, returns `Ok(...)`.
//!   - All forward methods (`run_generate`, `run_bench`, `run_ppl`,
//!     `init_prefix_cache`) `unimplemented!()` with a phase-pointer
//!     message — the worker will surface that to the client as an
//!     internal error rather than crashing in the Gemma 4 loader.
//!
//! Phase 1+ replace the unimplemented! bodies with the real tensor
//! load + forward implementations. See plan
//! `~/.claude/plans/abundant-meandering-sifakis.md` for the phase-list.

use std::path::PathBuf;

use rvllm_core::Result;

use crate::gemma4_bring_up::Gemma4EnginePaths;
use crate::qwen36_arch::Qwen36Arch;

/// Phase 0 stub. Carries the parsed architecture so callers (and the
/// future Phase-1 loader) have everything they need from `config.json`.
pub struct Qwen36Bringup {
    pub paths: Gemma4EnginePaths,
    pub arena_bytes: usize,
    pub arch: Qwen36Arch,
}

impl Qwen36Bringup {
    /// Load + validate + log the Qwen 3.6 config. Does NOT load
    /// weights yet; that's Phase 1. Returns `Err` if `config.json` is
    /// missing required Qwen-3.6 markers.
    pub fn load(paths: Gemma4EnginePaths, arena_bytes: usize) -> Result<Self> {
        let arch = match Qwen36Arch::from_dir(&paths.model_dir)? {
            Some(a) => a,
            None => {
                // Caller is supposed to have already detected Qwen 3.6
                // and dispatched here only when sure. Fall through with
                // a clear panic so the contract violation is visible.
                panic!(
                    "Qwen36Bringup::load called for model_dir={:?} but \
                     Qwen36Arch::from_dir returned None — caller dispatched \
                     incorrectly",
                    paths.model_dir
                );
            }
        };
        arch.log_summary();
        eprintln!(
            "[qwen36] Phase 0 scaffolding active — forward pass not yet \
             implemented. See ~/.claude/plans/abundant-meandering-sifakis.md \
             for the phase-list. First /v1/chat/completions request will \
             return 500."
        );
        Ok(Self {
            paths,
            arena_bytes,
            arch,
        })
    }

    /// Resolve a kernels directory hint for the worker. Phase 0 just
    /// echoes the configured path so the cuda_worker can surface a
    /// uniform "kernels dir" log line for both backends.
    pub fn kernels_dir(&self) -> &PathBuf {
        &self.paths.kernels_dir
    }

    pub fn run_generate(&self) -> ! {
        unimplemented!(
            "qwen36 phase 1 — forward pass not yet ported (model_dir={:?})",
            self.paths.model_dir
        );
    }

    pub fn run_bench(&self) -> ! {
        unimplemented!("qwen36 phase 1 — bench harness not yet ported");
    }

    pub fn run_ppl(&self) -> ! {
        unimplemented!("qwen36 phase 1 — ppl harness not yet ported");
    }

    pub fn init_prefix_cache(&self) -> ! {
        unimplemented!("qwen36 phase 1 — prefix cache not yet ported");
    }
}
