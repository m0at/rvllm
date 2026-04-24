// Compact list of current goal's subtasks. A painter DAG view is
// planned for v0.2; for now this dense list is enough to steer.

use egui::{Color32, RichText};

use crate::app::SwarmApp;
use crate::state::TaskStatus;
use crate::theme;

pub fn render(ui: &mut egui::Ui, app: &mut SwarmApp) {
    let goal = {
        let g = app.state.read();
        g.master.current_goal
    };
    let Some(goal) = goal else {
        ui.label(
            RichText::new("no active goal")
                .size(11.5)
                .color(theme::TEXT_DIM)
                .italics(),
        );
        return;
    };

    let (snapshot, goal_status, master_msg): (Vec<(u128, String, TaskStatus)>, _, _) = {
        let g = app.state.read();
        let mut tasks: Vec<_> = g.tasks.tasks_in_goal(goal);
        tasks.sort_by_key(|t| t.id.0);
        (
            tasks
                .iter()
                .map(|t| (t.id.0.as_u128(), t.summary.clone(), t.status))
                .collect(),
            g.tasks.goals.get(&goal).map(|goal| goal.status),
            g.master.last_response.clone(),
        )
    };

    if snapshot.is_empty() {
        let label = match goal_status {
            Some(crate::state::GoalStatus::Running) => master_msg
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "shared generation running".into()),
            Some(crate::state::GoalStatus::Done) => "(broadcast complete)".into(),
            Some(crate::state::GoalStatus::Failed) => "(generation failed)".into(),
            Some(crate::state::GoalStatus::Cancelled) => "(goal cancelled)".into(),
            Some(crate::state::GoalStatus::Merging) => "(merging...)".into(),
            _ => "(decomposing...)".into(),
        };
        ui.label(
            RichText::new(label)
                .size(11.5)
                .color(theme::TEXT_DIM),
        );
        return;
    }

    for (id, summary, status) in snapshot {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(status_glyph(status))
                    .size(11.0)
                    .color(status_colour(status))
                    .monospace(),
            );
            ui.label(
                RichText::new(format!("{:x}", id & 0xffff))
                    .size(10.5)
                    .color(theme::TEXT_DIM)
                    .monospace(),
            );
            ui.label(
                RichText::new(truncate(&summary, 44))
                    .size(11.0)
                    .color(theme::TEXT_PRIMARY),
            );
        });
    }
    let _ = Color32::BLACK;
}

fn status_glyph(s: TaskStatus) -> &'static str {
    match s {
        TaskStatus::Queued => "◌",
        TaskStatus::Running => "▶",
        TaskStatus::Verifying => "?",
        TaskStatus::Revising => "↻",
        TaskStatus::Accepted => "✓",
        TaskStatus::Rejected => "✗",
        TaskStatus::Cancelled => "⊘",
        TaskStatus::TimedOut => "⏱",
        TaskStatus::Failed => "✗",
    }
}

fn status_colour(s: TaskStatus) -> Color32 {
    match s {
        TaskStatus::Queued => theme::STATE_QUEUED,
        TaskStatus::Running => theme::STATE_RUNNING,
        TaskStatus::Verifying => theme::STATE_LOADING,
        TaskStatus::Revising => theme::STATE_LOADING,
        TaskStatus::Accepted => theme::STATE_IDLE,
        TaskStatus::Cancelled => theme::STATE_SWAPPED,
        TaskStatus::Rejected | TaskStatus::Failed => theme::STATE_FAILED,
        TaskStatus::TimedOut => theme::STATE_SWAPPED,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max).collect();
        t.push('…');
        t
    }
}
