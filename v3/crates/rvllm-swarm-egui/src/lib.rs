pub mod app;
pub mod controller;
pub mod ipc;
pub mod remote;
pub mod scheduler;
pub mod state;
pub mod theme;
pub mod ui;
pub mod worker;

use std::path::PathBuf;

pub fn detect_repo_root() -> PathBuf {
    if let Ok(env) = std::env::var("SWARM_REPO_ROOT") {
        return PathBuf::from(env);
    }
    if let Ok(cwd) = std::env::current_dir() {
        let mut cur = cwd.as_path();
        for _ in 0..8 {
            if cur.join(".git").exists() {
                return cur.to_path_buf();
            }
            if cur.file_name().and_then(|s| s.to_str()) == Some("v3")
                && cur.join("Cargo.toml").exists()
                && cur.join("crates").is_dir()
            {
                return cur.to_path_buf();
            }
            match cur.parent() {
                Some(parent) => cur = parent,
                None => break,
            }
        }
        return cwd;
    }
    PathBuf::from(".")
}
