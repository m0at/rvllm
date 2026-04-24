// Per-agent bounded history ring.

#![allow(dead_code)]

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub ts_ms: i64,
    pub kind: HistoryKind,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoryKind {
    System,
    User,
    Assistant,
    Tool,
    Info,
    Error,
}

impl HistoryKind {
    pub fn label(self) -> &'static str {
        match self {
            HistoryKind::System => "sys",
            HistoryKind::User => "usr",
            HistoryKind::Assistant => "ast",
            HistoryKind::Tool => "tool",
            HistoryKind::Info => "info",
            HistoryKind::Error => "err",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct History {
    inner: VecDeque<HistoryEntry>,
    cap: usize,
}

impl History {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: VecDeque::with_capacity(cap),
            cap,
        }
    }
    pub fn push(&mut self, entry: HistoryEntry) {
        if self.inner.len() == self.cap {
            self.inner.pop_front();
        }
        self.inner.push_back(entry);
    }
    pub fn iter_rev(&self) -> impl DoubleEndedIterator<Item = &HistoryEntry> {
        self.inner.iter().rev()
    }
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &HistoryEntry> {
        self.inner.iter()
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn last(&self) -> Option<&HistoryEntry> {
        self.inner.back()
    }
}
