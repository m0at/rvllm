// Tool sandbox scaffold. See docs/06_TOOLS.md + docs/12_SAFETY_AND_SANDBOX.md.
//
// The actual wiring into rhai::Engine happens in worker::cuda when --features
// cuda is enabled. This module owns the path-sandbox helper and the shell
// allowlist, which are used by both the real and mock paths.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct SandboxHandle {
    pub worktree_root: PathBuf,
}

impl SandboxHandle {
    pub fn new(worktree_root: PathBuf) -> Self {
        Self { worktree_root }
    }

    /// Canonicalise `path` (relative to worktree root if not absolute) and
    /// check it stays inside the worktree. Returns a canonicalised absolute
    /// path on success.
    pub fn resolve_inside(&self, path: &Path) -> anyhow::Result<PathBuf> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.worktree_root.join(path)
        };
        // We don't require the target to exist yet (fs_write, fs_mkdir).
        // Walk up until we find an existing ancestor, canonicalise it, then
        // re-attach the tail.
        let (existing, tail) = split_at_existing(&absolute);
        let canon_root = existing
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("canonicalise {}: {e}", existing.display()))?;
        let resolved = canon_root.join(&tail);
        let root_canon = self
            .worktree_root
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("canonicalise worktree root: {e}"))?;
        if !resolved.starts_with(&root_canon) {
            anyhow::bail!(
                "path {} escapes worktree root {}",
                resolved.display(),
                root_canon.display()
            );
        }
        Ok(resolved)
    }
}

fn split_at_existing(p: &Path) -> (PathBuf, PathBuf) {
    let mut cur = p.to_path_buf();
    let mut tail = PathBuf::new();
    loop {
        if cur.exists() {
            return (cur, tail);
        }
        let last = match cur.file_name() {
            Some(s) => s.to_owned(),
            None => break,
        };
        tail = Path::new(&last).join(&tail);
        if !cur.pop() {
            break;
        }
    }
    (cur, tail)
}

pub const SHELL_ALLOWLIST: &[&str] = &[
    "cat", "head", "tail", "wc", "find", "awk", "sed", "diff", "md5sum",
    "sha256sum", "python3", "node", "jq",
];

pub fn shell_is_allowed(cmd: &str) -> bool {
    SHELL_ALLOWLIST.contains(&cmd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_allowlist_rejects_curl() {
        assert!(!shell_is_allowed("curl"));
        assert!(!shell_is_allowed("rm"));
        assert!(shell_is_allowed("cat"));
    }

    #[test]
    fn sandbox_rejects_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let sb = SandboxHandle::new(tmp.path().to_path_buf());
        let inside = sb.resolve_inside(Path::new("foo/bar.txt"));
        assert!(inside.is_ok(), "inside failed: {:?}", inside);
        let outside = sb.resolve_inside(Path::new("/etc/passwd"));
        assert!(outside.is_err(), "escape should fail");
    }
}
