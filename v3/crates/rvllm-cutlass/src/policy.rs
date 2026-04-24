//! Autotune policy loader — maps GEMM shape → variant.
//!
//! `policy.json` ships in the build tarball. The runtime never reads
//! `~/.cache`; a missing entry means the engine refuses to start with
//! `CutlassError::AutotuneCacheMiss`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use rvllm_core::{ConfigError, CutlassCtx, CutlassError, DType, IoError, Result, RvllmError};

use crate::variants::{VariantDescriptor, VariantId};

/// A policy entry: given a GEMM shape + dtype, which variant wins.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct PolicyEntry {
    pub variant: VariantId,
    /// Workspace bytes needed for this (variant, shape). Recorded at
    /// autotune time so the runtime allocator sizes the single slab
    /// from the max across all live entries.
    pub workspace_bytes: u64,
}

/// Serialized shape key. We key on (m, n, k, dtype) because different
/// dtypes use different variants.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ShapeKey {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub dtype: DType,
}

/// The full policy file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Policy {
    pub revision: String,
    pub arch: String,
    pub variants: Vec<VariantDescriptor>,
    pub entries: BTreeMap<String, PolicyEntry>,
}

impl Policy {
    pub fn load(path: &Path) -> Result<Self> {
        let body = std::fs::read_to_string(path).map_err(|source| RvllmError::Io {
            err: IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let policy: Policy = serde_json::from_str(&body).map_err(|e| {
            RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!("policy.json not valid JSON: {e}")],
                },
                "policy.json",
            )
        })?;
        // Every listed variant must have a matched schedule pair.
        let mut reasons = Vec::new();
        for v in &policy.variants {
            if !v.validate() {
                reasons.push(format!(
                    "variant {} has mismatched schedules {:?}/{:?}",
                    v.id.0, v.mainloop, v.epilogue
                ));
            }
        }
        if !reasons.is_empty() {
            return Err(RvllmError::config(
                ConfigError::Inconsistent { reasons },
                "policy.json::variants",
            ));
        }
        Ok(policy)
    }

    /// Look up the variant for a shape. Missing entry → typed Err; the
    /// engine's init path refuses to continue.
    pub fn lookup(&self, m: usize, n: usize, k: usize, dtype: DType) -> Result<&PolicyEntry> {
        self.lookup_with_suffix(m, n, k, dtype, "")
    }

    /// Residual-epilogue variant lookup. Distinguishes shapes that
    /// share (m,n,k) between a base GEMM and its residual-fused sibling.
    pub fn lookup_residual(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
    ) -> Result<&PolicyEntry> {
        self.lookup_with_suffix(m, n, k, dtype, "_res")
    }

    fn lookup_with_suffix(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        suffix: &str,
    ) -> Result<&PolicyEntry> {
        let key = format!("{m}_{n}_{k}_{dtype:?}{suffix}");
        self.entries.get(&key).ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::AutotuneCacheMiss { m, n, k, dtype },
                CutlassCtx {
                    kernel: "Policy::lookup",
                    stream: 0,
                },
            )
        })
    }

    pub fn entry_key(m: usize, n: usize, k: usize, dtype: DType) -> String {
        format!("{m}_{n}_{k}_{dtype:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::ScheduleTag;
    use crate::variants::{ClusterShape, TileShape};

    fn policy_with_one_entry() -> Policy {
        let mut entries = BTreeMap::new();
        entries.insert(
            Policy::entry_key(128, 152064, 3584, DType::Fp8E4M3),
            PolicyEntry {
                variant: VariantId(0),
                workspace_bytes: 1 << 20,
            },
        );
        Policy {
            revision: "test".into(),
            arch: "sm_90".into(),
            variants: vec![VariantDescriptor {
                id: VariantId(0),
                tile: TileShape::new(128, 128, 128),
                cluster: ClusterShape::one(),
                mainloop: ScheduleTag::Coop,
                epilogue: ScheduleTag::Coop,
            }],
            entries,
        }
    }

    #[test]
    fn lookup_hit() {
        let p = policy_with_one_entry();
        let e = p.lookup(128, 152064, 3584, DType::Fp8E4M3).unwrap();
        assert_eq!(e.variant, VariantId(0));
    }

    #[test]
    fn lookup_miss_is_typed_err() {
        let p = policy_with_one_entry();
        let err = p.lookup(17, 17, 17, DType::Fp8E4M3).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("AutotuneCacheMiss"));
        assert!(s.contains("m: 17"));
    }

    #[test]
    fn policy_rejects_mismatched_variant_on_load() {
        // Build a file with a WS/Coop variant → load should fail.
        let bad_variant = VariantDescriptor {
            id: VariantId(0),
            tile: TileShape::new(64, 128, 128),
            cluster: ClusterShape::one(),
            mainloop: ScheduleTag::WS,
            epilogue: ScheduleTag::Coop,
        };
        let p = Policy {
            revision: "bad".into(),
            arch: "sm_90".into(),
            variants: vec![bad_variant],
            entries: BTreeMap::new(),
        };
        let tmp = std::env::temp_dir().join(format!("rvllm-policy-{}.json", std::process::id()));
        std::fs::write(&tmp, serde_json::to_string_pretty(&p).unwrap()).unwrap();
        let err = Policy::load(&tmp).unwrap_err();
        std::fs::remove_file(&tmp).ok();
        let s = format!("{err}");
        assert!(s.contains("mismatched"));
    }
}
