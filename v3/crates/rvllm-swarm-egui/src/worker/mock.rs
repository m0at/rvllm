// Deterministic mock backend.
//
// Streams a small number of "iterations" with fake tool calls into the
// shared AppState so the UI can animate without a GPU. Returns a
// TaskOutcome at the end.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use crate::ipc::{now_ms, AgentId, LogLevel, LogLine, SlotIdx, TaskId, TaskOutcome};
use crate::state::history::{HistoryEntry, HistoryKind};
use crate::state::AppState;

pub struct MockBackend {
    tick: u64,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockBackend {
    pub fn new() -> Self {
        Self { tick: 0 }
    }

    pub fn broadcast_response(&mut self, prompt: &str, fanout: usize) -> String {
        self.tick = self.tick.wrapping_add(1);
        format!(
            "(mock broadcast x{fanout}) shared response for '{}'",
            prompt.trim()
        )
    }

    pub fn run_task_streaming(
        &mut self,
        state: &Arc<RwLock<AppState>>,
        agent: AgentId,
        task: TaskId,
        slot: SlotIdx,
    ) -> TaskOutcome {
        self.tick = self.tick.wrapping_add(1);
        let iterations = 3u32;
        let per_iter = Duration::from_millis(180);
        let start = Instant::now();
        let _ = slot;

        let (task_summary, max_iters) = {
            let g = state.read();
            let sum = g
                .tasks
                .tasks
                .get(&task)
                .map(|t| t.summary.clone())
                .unwrap_or_else(|| "<unknown task>".into());
            let max = g.settings.max_iterations;
            (sum, max)
        };

        // Push an initial history line.
        {
            let mut g = state.write();
            if let Some(a) = g.agent_mut(agent) {
                a.history.push(HistoryEntry {
                    ts_ms: now_ms(),
                    kind: HistoryKind::User,
                    text: format!("TASK: {task_summary}"),
                });
            }
        }

        for i in 1..=iterations {
            std::thread::sleep(per_iter);
            let tool = pick_tool(self.tick + i as u64);
            {
                let mut g = state.write();
                if let Some(a) = g.agent_mut(agent) {
                    a.iteration_progress = Some((i, max_iters));
                    a.last_tool = Some(tool.to_string());
                    a.flash = Some(format!("it {i} using {tool}"));
                    a.history.push(HistoryEntry {
                        ts_ms: now_ms(),
                        kind: HistoryKind::Tool,
                        text: format!("{tool}(...) -> ok"),
                    });
                    a.metrics.tool_calls_last = i;
                    a.metrics.tok_s_decode = 70.0 + ((self.tick + i as u64) as f32 % 12.0);
                }
                g.log.push(LogLine {
                    ts_ms: now_ms(),
                    level: LogLevel::Debug,
                    source: format!("agent:{}", short_id(agent.0)),
                    message: format!("iter {i}/{iterations}: {tool}"),
                });
            }
        }

        // Add final assistant message.
        {
            let mut g = state.write();
            if let Some(a) = g.agent_mut(agent) {
                a.history.push(HistoryEntry {
                    ts_ms: now_ms(),
                    kind: HistoryKind::Assistant,
                    text: format!(
                        "FINAL_ANSWER: (mock) completed '{}' in {} iterations ({:.0} ms)",
                        task_summary,
                        iterations,
                        start.elapsed().as_millis()
                    ),
                });
            }
        }

        TaskOutcome::Success {
            response: format!("(mock) completed '{task_summary}'"),
        }
    }
}

fn pick_tool(seed: u64) -> &'static str {
    const TOOLS: &[&str] = &[
        "fs_read", "fs_write", "cargo_check", "rg", "git_diff", "fs_patch",
    ];
    TOOLS[(seed as usize) % TOOLS.len()]
}

fn short_id(u: uuid::Uuid) -> String {
    u.simple().to_string().chars().take(8).collect()
}
