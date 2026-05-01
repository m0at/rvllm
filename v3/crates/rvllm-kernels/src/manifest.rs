//! `manifest.json`: the SHA-pinned catalog of every kernel artifact.
//!
//! The deploy tarball ships this file next to `bin/`, `lib/`, and
//! `kernels/`. At engine init, `KernelManifest::load_and_verify` reads
//! `manifest.json`, then recomputes sha256 of every listed file and
//! aborts if any digest drifts. There is no lookup path that bypasses
//! this; `KernelLoader::new` takes a `VerifiedManifest` and refuses to
//! read anything not in it.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use rvllm_core::{ConfigError, IoError, Result, RvllmError};

/// Manifest entry for one artifact.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ArtifactEntry {
    /// Path relative to the manifest file's directory.
    pub path: String,
    /// sha256 hex digest (lowercase, 64 chars).
    pub sha256: String,
    /// Size in bytes.
    pub bytes: u64,
}

/// The full deploy manifest.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct KernelManifest {
    /// Build SHA that produced this manifest. Copied into the binary at
    /// build time; engine init verifies `env!("REVISION") == revision`.
    pub revision: String,
    /// GPU arch the kernels were built for (e.g. `sm_90`).
    pub arch: String,
    /// Entries keyed by logical name (e.g. `libfa3_kernels.so`, `argmax`).
    pub entries: BTreeMap<String, ArtifactEntry>,
}

/// A `KernelManifest` whose on-disk checksums have been re-verified.
/// Only this type unlocks `KernelLoader`.
#[derive(Clone, Debug)]
pub struct VerifiedManifest {
    manifest: KernelManifest,
    root: PathBuf,
}

impl VerifiedManifest {
    pub fn manifest(&self) -> &KernelManifest {
        &self.manifest
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// Resolve a logical name to its on-disk absolute path.
    /// Returns `None` if the name is not in the manifest.
    pub fn path_of(&self, logical_name: &str) -> Option<PathBuf> {
        let rel = &self.manifest.entries.get(logical_name)?.path;
        Some(self.root.join(rel))
    }
    pub fn revision(&self) -> &str {
        &self.manifest.revision
    }

    /// Compile-time git revision of the binary that bundles this crate.
    /// Set by `rvllm-kernels`'s `build.rs` from `git rev-parse --short HEAD`,
    /// `"dev"` when git is absent. Use as the `expected` argument to
    /// `warn_if_revision_drift`.
    pub const BUILD_REVISION: &'static str = env!("RVLLM_BUILD_REVISION");
    pub fn arch(&self) -> &str {
        &self.manifest.arch
    }

    /// Codex23-2: compare the manifest's `revision` against the
    /// expected one (typically `env!("RVLLM_BUILD_REVISION")` baked
    /// in by the kernels crate's `build.rs`) and emit a one-shot
    /// stderr WARN on mismatch. Not a hard-fail: dev rebuilds where
    /// the binary moves ahead of the kernels (or vice versa) are
    /// legitimate; the warn just makes the drift visible so
    /// "wrong-math / launch-error" failures further down trace
    /// back to a stale manifest pairing instead of looking like
    /// fresh kernel bugs.
    pub fn warn_if_revision_drift(&self, expected: &str) {
        if expected == "dev" || self.manifest.revision == "dev" {
            return; // local hack-builds opt out
        }
        if self.manifest.revision != expected {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[kernels] WARN: manifest revision {} does not match binary {} \
                     — wrong-math or launch-error failures may follow. Rebuild \
                     kernels/ or the binary so the pair lines up.",
                    self.manifest.revision, expected
                );
            }
        }
    }
}

impl KernelManifest {
    /// Load `manifest.json` from a deploy directory and verify every
    /// listed artifact's sha256. Returns `VerifiedManifest` on success.
    pub fn load_and_verify(manifest_path: &Path) -> Result<VerifiedManifest> {
        let root = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let body = fs::read_to_string(manifest_path).map_err(|source| RvllmError::Io {
            err: IoError::from(&source),
            path: manifest_path.to_path_buf(),
            source,
        })?;
        let manifest: KernelManifest = serde_json::from_str(&body).map_err(|e| {
            RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!("manifest.json is not valid JSON: {e}")],
                },
                "manifest.json",
            )
        })?;

        let mut mismatches = Vec::new();
        for (name, entry) in &manifest.entries {
            let path = root.join(&entry.path);
            let bytes = fs::read(&path).map_err(|source| RvllmError::Io {
                err: IoError::from(&source),
                path: path.clone(),
                source,
            })?;
            if bytes.len() as u64 != entry.bytes {
                mismatches.push(format!(
                    "{name}: size {} != manifest {}",
                    bytes.len(),
                    entry.bytes
                ));
                continue;
            }
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let got = hex::encode(hasher.finalize());
            if got != entry.sha256 {
                mismatches.push(format!(
                    "{name}: sha256 {got} != manifest {}",
                    entry.sha256
                ));
            }
        }
        if !mismatches.is_empty() {
            return Err(RvllmError::config(
                ConfigError::Inconsistent { reasons: mismatches },
                "manifest.json",
            ));
        }

        Ok(VerifiedManifest { manifest, root })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(body).unwrap();
        p
    }

    #[test]
    fn roundtrip_verify() {
        let tmp = tempdir();
        let artifact = write_tmp(&tmp, "kern.ptx", b"PTX CONTENT");
        let digest = {
            let mut h = Sha256::new();
            h.update(b"PTX CONTENT");
            hex::encode(h.finalize())
        };
        let mut entries = BTreeMap::new();
        entries.insert(
            "argmax".into(),
            ArtifactEntry {
                path: "kern.ptx".into(),
                sha256: digest,
                bytes: 11,
            },
        );
        let manifest = KernelManifest {
            revision: "abcdef".into(),
            arch: "sm_90".into(),
            entries,
        };
        let mp = tmp.join("manifest.json");
        fs::write(&mp, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
        let verified = KernelManifest::load_and_verify(&mp).unwrap();
        assert_eq!(verified.revision(), "abcdef");
        assert_eq!(verified.arch(), "sm_90");
        assert_eq!(verified.path_of("argmax").unwrap(), artifact);
    }

    #[test]
    fn drift_rejected() {
        let tmp = tempdir();
        write_tmp(&tmp, "kern.ptx", b"PTX CONTENT");
        let bogus = "0".repeat(64);
        let mut entries = BTreeMap::new();
        entries.insert(
            "argmax".into(),
            ArtifactEntry {
                path: "kern.ptx".into(),
                sha256: bogus,
                bytes: 11,
            },
        );
        let manifest = KernelManifest {
            revision: "abcdef".into(),
            arch: "sm_90".into(),
            entries,
        };
        let mp = tmp.join("manifest.json");
        fs::write(&mp, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
        let err = KernelManifest::load_and_verify(&mp).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("sha256"));
    }

    fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rvllm-kernels-manifest-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }
}
