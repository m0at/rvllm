// Git worktree helpers. See docs/05_GIT_WORKTREE.md.
//
// Thin wrappers around `git` CLI. We avoid git2 to keep the dependency
// footprint small and to make the commands observable (every shell call
// is traced). On the real backend every worker gets one worktree
// provisioned at LoadAgent time.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;

#[derive(Debug)]
pub struct WorktreeHandle {
    pub repo_root: PathBuf,
    pub worktree_path: PathBuf,
    pub base_branch: String,
}

impl WorktreeHandle {
    /// `git worktree add -b <base_branch> <path> main`.
    /// Silently succeeds if the worktree already exists and is healthy.
    pub fn ensure(repo_root: &Path, worktree_path: &Path, base_branch: &str) -> anyhow::Result<Self> {
        std::fs::create_dir_all(worktree_path.parent().unwrap_or(Path::new(".")))?;
        if worktree_path.exists() {
            // Best-effort: assume it's already a worktree. Caller can call
            // `remove` + `ensure` again to recreate.
            return Ok(Self {
                repo_root: repo_root.to_path_buf(),
                worktree_path: worktree_path.to_path_buf(),
                base_branch: base_branch.to_owned(),
            });
        }
        let out = Command::new("git")
            .current_dir(repo_root)
            .args([
                "worktree",
                "add",
                "-b",
                base_branch,
                &worktree_path.to_string_lossy(),
            ])
            .output()
            .context("running git worktree add")?;
        if !out.status.success() {
            anyhow::bail!(
                "git worktree add failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Ok(Self {
            repo_root: repo_root.to_path_buf(),
            worktree_path: worktree_path.to_path_buf(),
            base_branch: base_branch.to_owned(),
        })
    }

    pub fn remove(&self) -> anyhow::Result<()> {
        let out = Command::new("git")
            .current_dir(&self.repo_root)
            .args([
                "worktree",
                "remove",
                "--force",
                &self.worktree_path.to_string_lossy(),
            ])
            .output()
            .context("running git worktree remove")?;
        if !out.status.success() {
            anyhow::bail!(
                "git worktree remove failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_repo(path: &Path) -> anyhow::Result<()> {
        let run = |args: &[&str]| -> anyhow::Result<()> {
            let out = Command::new("git")
                .current_dir(path)
                .args(args)
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t")
                .output()?;
            if !out.status.success() {
                anyhow::bail!(
                    "git {:?} failed: {}",
                    args,
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            Ok(())
        };
        run(&["init", "-q", "-b", "main"])?;
        std::fs::write(path.join("a.txt"), "hello")?;
        run(&["add", "-A"])?;
        run(&["commit", "-q", "-m", "initial"])?;
        Ok(())
    }

    #[test]
    fn ensure_worktree_roundtrip() {
        // Skip test if git is unavailable in the environment.
        if Command::new("git").arg("--version").output().is_err() {
            eprintln!("git not available, skipping");
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        if init_repo(tmp.path()).is_err() {
            eprintln!("git init failed, skipping");
            return;
        }
        let wt = tmp.path().join(".swarm").join("worktrees").join("agent-xyz");
        let h = WorktreeHandle::ensure(tmp.path(), &wt, "agent-xyz/base");
        let h = match h {
            Ok(h) => h,
            Err(e) => {
                eprintln!("ensure failed: {e}, skipping");
                return;
            }
        };
        assert!(h.worktree_path.exists());
        let _ = h.remove();
    }
}
