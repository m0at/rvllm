// Append-only JSONL journal with replay.
//
// Format: one JournalRecord per line. See docs/10_PERSISTENCE.md.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::ipc::{JournalRecord, PROTOCOL_VERSION};

pub struct Journal {
    path: PathBuf,
    file: File,
}

impl Journal {
    pub fn append(&mut self, rec: &JournalRecord) -> anyhow::Result<()> {
        let line = serde_json::to_string(rec)?;
        self.file.write_all(line.as_bytes())?;
        self.file.write_all(b"\n")?;
        // cheap durability: flush on every write. fsync reserved for goal
        // boundaries; in this scaffold we flush and move on.
        self.file.flush()?;
        Ok(())
    }

    pub fn replay_all(&self) -> Vec<JournalRecord> {
        replay_from(&self.path).unwrap_or_default()
    }
}

pub fn open_for_append(path: &Path) -> anyhow::Result<Journal> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("mkdir -p {}", parent.display()))?;
    }
    // Write a version marker next to the journal if missing.
    if let Some(parent) = path.parent() {
        let version_path = parent.join("version");
        if !version_path.exists() {
            std::fs::write(&version_path, PROTOCOL_VERSION.to_string())?;
        }
    }
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    Ok(Journal {
        path: path.to_path_buf(),
        file,
    })
}

pub fn replay_from(path: &Path) -> anyhow::Result<Vec<JournalRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let f = File::open(path)?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue, // tolerate half-written last line
        };
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<JournalRecord>(&line) {
            Ok(rec) => {
                if rec.protocol_version != PROTOCOL_VERSION {
                    anyhow::bail!(
                        "journal line {} has protocol_version {} but this binary speaks {}",
                        i + 1,
                        rec.protocol_version,
                        PROTOCOL_VERSION,
                    );
                }
                out.push(rec);
            }
            Err(e) => {
                // last-line tolerance: if this is the last line, stop quietly
                tracing::warn!("journal parse error at line {}: {e}", i + 1);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::{now_ms, JournalPayload};

    #[test]
    fn roundtrip_empty_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("tasks.jsonl");
        let recs = replay_from(&p).unwrap();
        assert!(recs.is_empty());
    }

    #[test]
    fn append_and_replay() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("tasks.jsonl");
        let mut j = open_for_append(&p).unwrap();
        let rec = JournalRecord {
            ts_ms: now_ms(),
            protocol_version: PROTOCOL_VERSION,
            payload: JournalPayload::Note {
                text: "hello".into(),
            },
        };
        j.append(&rec).unwrap();
        j.append(&rec).unwrap();
        drop(j);
        let recs = replay_from(&p).unwrap();
        assert_eq!(recs.len(), 2);
        match &recs[0].payload {
            JournalPayload::Note { text } => assert_eq!(text, "hello"),
            _ => panic!("wrong payload"),
        }
    }
}
