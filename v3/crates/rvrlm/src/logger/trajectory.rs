use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::Result;
use crate::types::{RlmIteration, RlmMetadata};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectorySnapshot {
    pub run_metadata: Option<RlmMetadata>,
    pub iterations: Vec<RlmIteration>,
}

pub struct TrajectoryLogger {
    log_file_path: Option<PathBuf>,
    run_metadata: Option<RlmMetadata>,
    iterations: Vec<RlmIteration>,
}

impl Default for TrajectoryLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl TrajectoryLogger {
    pub fn new() -> Self {
        Self {
            log_file_path: None,
            run_metadata: None,
            iterations: Vec::new(),
        }
    }

    pub fn with_log_dir<P>(log_dir: P, file_prefix: &str) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        create_dir_all(log_dir.as_ref())?;

        let timestamp = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => duration.as_secs(),
            Err(_) => 0,
        };
        let file_name = format!("{file_prefix}_{timestamp}_{}.jsonl", std::process::id());
        let log_file_path = log_dir.as_ref().join(file_name);

        Ok(Self {
            log_file_path: Some(log_file_path),
            run_metadata: None,
            iterations: Vec::new(),
        })
    }

    pub fn log_metadata(&mut self, metadata: RlmMetadata) -> Result<()> {
        if self.run_metadata.is_none() {
            self.write_entry(json!({
                "type": "metadata",
                "payload": metadata,
            }))?;
            self.run_metadata = Some(metadata);
        }
        Ok(())
    }

    pub fn log_iteration(&mut self, iteration: RlmIteration) -> Result<()> {
        self.write_entry(json!({
            "type": "iteration",
            "iteration": self.iterations.len() + 1,
            "payload": iteration,
        }))?;
        self.iterations.push(iteration);
        Ok(())
    }

    pub fn clear_iterations(&mut self) {
        self.iterations.clear();
    }

    pub fn snapshot(&self) -> TrajectorySnapshot {
        TrajectorySnapshot {
            run_metadata: self.run_metadata.clone(),
            iterations: self.iterations.clone(),
        }
    }

    fn write_entry(&self, entry: Value) -> Result<()> {
        if let Some(path) = &self.log_file_path {
            let mut file = OpenOptions::new().create(true).append(true).open(path)?;
            serde_json::to_writer(&mut file, &entry)?;
            writeln!(file)?;
        }
        Ok(())
    }
}
