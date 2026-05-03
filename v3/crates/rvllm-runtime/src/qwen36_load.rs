//! Qwen 3.6 weight loader (Phase 1 placeholder).
//!
//! Phase 0: nothing — `Qwen36Bringup` only needs the architecture
//! summary, no tensors uploaded. Phase 1 adds the full tensor map
//! (Q/K/V/O + Q-Norm + K-Norm + 256 expert G/U/D + shared-expert +
//! router weights + RMSNorm scales + embeddings + lm_head).

#![allow(dead_code)]

use std::path::Path;

use rvllm_core::Result;

use crate::qwen36_arch::Qwen36Arch;

/// Phase 0 stub: just loads the arch summary, no tensors.
pub fn load_config_only(model_dir: &Path) -> Result<Option<Qwen36Arch>> {
    Qwen36Arch::from_dir(model_dir)
}
