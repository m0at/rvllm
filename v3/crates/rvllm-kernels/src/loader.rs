//! `KernelLoader`: the only path to open PTX modules and `.so` handles.
//!
//! Construction takes a `VerifiedManifest`, so the SHA-pinned invariant
//! propagates: no artifact is touched unless its digest matched at
//! verification time. Requests for a logical name that is not in the
//! manifest return `Err(RvllmError::config(MissingField, ...))` — the
//! engine refuses to start rather than fall back.

use rvllm_core::{ConfigError, Result, RvllmError};

use crate::manifest::VerifiedManifest;

pub struct KernelLoader {
    manifest: VerifiedManifest,
    // When `cuda` is on, this is the dlopen handle pool keyed by
    // logical-name → loaded module. Keeping a BTreeMap so dump order
    // is deterministic for CI logs.
    #[cfg(feature = "cuda")]
    modules: std::cell::RefCell<std::collections::BTreeMap<String, ()>>,
}

impl KernelLoader {
    /// Build a loader from a verified manifest. The manifest must
    /// already have passed `KernelManifest::load_and_verify`.
    pub fn new(manifest: VerifiedManifest) -> Self {
        Self {
            manifest,
            #[cfg(feature = "cuda")]
            modules: Default::default(),
        }
    }

    pub fn manifest(&self) -> &VerifiedManifest {
        &self.manifest
    }

    /// Return the absolute path of an artifact by logical name, or
    /// `Err` if the manifest has no such entry. Engine refuses to
    /// start on `Err`.
    pub fn path(&self, logical_name: &str) -> Result<std::path::PathBuf> {
        self.manifest.path_of(logical_name).ok_or_else(|| {
            RvllmError::config(
                ConfigError::MissingField {
                    name: "manifest.entries",
                },
                logical_name_static(logical_name),
            )
        })
    }

    /// Load a PTX module by logical name. Host stub without `cuda`
    /// feature just reads the file and returns a handle containing its
    /// bytes (useful for tests without a GPU).
    #[cfg(not(feature = "cuda"))]
    pub fn load_ptx(&self, logical_name: &str) -> Result<PtxBytes> {
        let path = self.path(logical_name)?;
        let bytes = std::fs::read(&path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path,
            source,
        })?;
        Ok(PtxBytes { bytes })
    }

    /// CUDA impl: cuModuleLoad. Pending wiring.
    #[cfg(feature = "cuda")]
    pub fn load_ptx(&self, _logical_name: &str) -> Result<PtxBytes> {
        Err(RvllmError::cuda(
            "KernelLoader::load_ptx",
            rvllm_core::CudaErrorKind::ModuleLoadFailed,
            rvllm_core::CudaCtx::setup(),
        ))
    }

    /// Return path for dlopen (caller wraps with libloading under
    /// feature `cuda`). Under no-cuda, exposed for inspection only.
    pub fn so_path(&self, logical_name: &str) -> Result<std::path::PathBuf> {
        self.path(logical_name)
    }
}

/// Opaque host-side representation of PTX module bytes. Real runtime
/// replaces with `cudarc::CudaModule` under feature `cuda`.
#[derive(Debug)]
pub struct PtxBytes {
    pub bytes: Vec<u8>,
}

// Interning helper so the error carries a static str that names the
// missing logical key. `&'static str` would be nicer but downstream
// passes arbitrary names; leak-into-static is acceptable for a
// never-happens-in-prod error path.
fn logical_name_static(name: &str) -> &'static str {
    // Leak the string so it lives for 'static. This path is only hit
    // at engine init when a missing artifact aborts startup anyway.
    Box::leak(name.to_owned().into_boxed_str())
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;
    use crate::manifest::{ArtifactEntry, KernelManifest};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;

    fn verified(tmp: &PathBuf, body: &[u8], name: &str) -> VerifiedManifest {
        let file = tmp.join(format!("{name}.ptx"));
        fs::write(&file, body).unwrap();
        let digest = {
            let mut h = Sha256::new();
            h.update(body);
            hex::encode(h.finalize())
        };
        let mut entries = BTreeMap::new();
        entries.insert(
            name.to_string(),
            ArtifactEntry {
                path: format!("{name}.ptx"),
                sha256: digest,
                bytes: body.len() as u64,
            },
        );
        let m = KernelManifest {
            revision: "test".into(),
            arch: "sm_90".into(),
            entries,
        };
        let mp = tmp.join("manifest.json");
        fs::write(&mp, serde_json::to_string_pretty(&m).unwrap()).unwrap();
        KernelManifest::load_and_verify(&mp).unwrap()
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "rvllm-kernels-loader-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn load_ptx_roundtrip() {
        let tmp = tempdir();
        let vm = verified(&tmp, b"HELLO PTX", "argmax");
        let loader = KernelLoader::new(vm);
        let m = loader.load_ptx("argmax").unwrap();
        assert_eq!(m.bytes, b"HELLO PTX");
    }

    #[test]
    fn missing_name_is_err() {
        let tmp = tempdir();
        let vm = verified(&tmp, b"HELLO PTX", "argmax");
        let loader = KernelLoader::new(vm);
        let err = loader.load_ptx("silu").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("MissingField"));
        assert!(s.contains("silu"));
    }
}
