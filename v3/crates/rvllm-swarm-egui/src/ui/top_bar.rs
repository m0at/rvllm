use egui::{Color32, RichText};

use crate::app::SwarmApp;
use crate::state::{AgentLifecycle, GoalStatus, SlotState};
use crate::theme;

pub fn render(ui: &mut egui::Ui, app: &mut SwarmApp) {
    let (
        goal_summary,
        counts,
        gpu,
        master_busy,
        master_msg,
        slots_view,
        agent_count,
        goals_total,
        remote_addr,
        backend_kind,
        max_new_tokens,
        execution_mode,
        decode_batch_target,
    ) = {
        let g = app.state.read();
        let gs = g
            .master
            .current_goal
            .and_then(|id| g.tasks.goals.get(&id))
            .map(|goal| (goal.text.clone(), goal.status, goal.plan_summary.clone()));
        let counts = g.tasks.global_counts();
        let gpu = g.gpu;
        let master_busy = g.master.decomposition_in_flight;
        let master_msg = g.master.last_response.clone().unwrap_or_default();
        let slots_view: Vec<_> = g
            .slots
            .iter()
            .map(|s| {
                (
                    s.state,
                    s.occupant
                        .and_then(|id| g.agent(id).map(|a| (a.display_index, a.persona))),
                )
            })
            .collect();
        (
            gs,
            counts,
            gpu,
            master_busy,
            master_msg,
            slots_view,
            g.agents.len(),
            g.tasks.goals.len(),
            app.remote_addr.clone(),
            g.backend_kind,
            g.settings.max_new_tokens,
            g.settings.scheduler.execution_mode,
            g.settings.scheduler.decode_batch_target,
        )
    };
    let goal_status = goal_summary.as_ref().map(|(_, status, _)| *status);
    let mode_label = match remote_addr.as_deref() {
        Some(addr) => format!("REMOTE {addr}"),
        None => match backend_kind {
            crate::state::BackendKind::Mock => "LOCAL mock".into(),
            crate::state::BackendKind::Rvllm => "LOCAL rvllm".into(),
        },
    };
    let headline = if goals_total == 0 {
        format!("{mode_label}  no goals run yet")
    } else if matches!(goal_status, Some(GoalStatus::Running)) && counts.in_flight == 0 {
        format!("{mode_label}  shared generation running")
    } else if counts.in_flight > 0 || master_busy {
        format!("{mode_label}  running {}/{} tasks", counts.in_flight, counts.total)
    } else {
        format!("{mode_label}  last run done {} tasks", counts.done)
    };
    let master_status = if master_busy {
        "decomposing…"
    } else if matches!(goal_status, Some(GoalStatus::Running)) {
        "generating…"
    } else if matches!(goal_status, Some(GoalStatus::Merging)) {
        "merging…"
    } else {
        "idle"
    };

    ui.horizontal(|ui| {
        ui.label(
            RichText::new("RVLLM SWARM")
                .size(20.0)
                .strong()
                .color(theme::ACCENT),
        );
        ui.add_space(10.0);
        let response = ui.add(
            egui::TextEdit::singleline(&mut app.pending_goal)
                .desired_width(ui.available_width() - 280.0)
                .hint_text("Type a goal, Cmd+Enter dispatch, Cmd+Shift+Enter broadcast x30"),
        );
        let hotkey = response.has_focus()
            && ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Enter));
        let broadcast_hotkey = response.has_focus()
            && ui.input(|i| i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::Enter));
        let clicked = ui
            .add_enabled(!master_busy, egui::Button::new(
                RichText::new("Dispatch").size(14.0).strong().color(Color32::WHITE),
            ).fill(if master_busy { theme::BG_INPUT } else { theme::ACCENT }))
            .clicked();
        let broadcast_clicked = ui
            .add_enabled(
                !master_busy,
                egui::Button::new(
                    RichText::new(format!("Broadcast x{agent_count}"))
                        .size(14.0)
                        .strong()
                        .color(Color32::WHITE),
                )
                .fill(if master_busy { theme::BG_INPUT } else { theme::INFO }),
            )
            .clicked();
        if broadcast_clicked || broadcast_hotkey {
            app.submit_broadcast();
        } else if clicked || hotkey {
            app.submit_goal();
        }
    });

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(RichText::new(headline).size(12.0).strong().color(theme::ACCENT_HOVER));
        ui.add_space(12.0);
        ui.label(
            RichText::new(format!(
                "{} batch {}",
                execution_mode.label(),
                decode_batch_target
            ))
            .size(12.0)
            .strong()
            .color(theme::INFO),
        );
        ui.add_space(12.0);
        ui.label(
            RichText::new(format!(
                "master: {}",
                master_status
            ))
            .size(12.0)
            .color(if master_busy || matches!(goal_status, Some(GoalStatus::Running | GoalStatus::Merging)) {
                theme::INFO
            } else {
                theme::TEXT_SECONDARY
            }),
        );
        ui.add_space(12.0);
        ui.label(RichText::new(format!("in-flight {}", counts.in_flight)).size(12.0).color(theme::TEXT_SECONDARY));
        ui.add_space(10.0);
        ui.label(RichText::new(format!("queued {}", counts.queued)).size(12.0).color(theme::TEXT_SECONDARY));
        ui.add_space(10.0);
        ui.label(RichText::new(format!("done {}", counts.done)).size(12.0).color(theme::TEXT_SECONDARY));
        ui.add_space(10.0);
        ui.label(RichText::new(format!("cancelled {}", counts.cancelled)).size(12.0).color(theme::TEXT_SECONDARY));
        ui.add_space(10.0);
        ui.label(RichText::new(format!("failed {}", counts.failed)).size(12.0).color(theme::TEXT_SECONDARY));
    });
    if !master_msg.is_empty() {
        ui.add_space(2.0);
        ui.label(
            RichText::new(master_msg)
                .size(11.5)
                .color(theme::INFO),
        );
    }

    // Slots row.
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(RichText::new("slots:").size(12.0).color(theme::TEXT_DIM));
        for (i, (state, occ)) in slots_view.iter().enumerate() {
            let (label, col) = match (state, occ) {
                (SlotState::Empty, _) => (format!("[{i} empty]"), theme::STATE_IDLE),
                (SlotState::Loading, _) => (format!("[{i} load…]"), theme::STATE_LOADING),
                (SlotState::Hot, Some((idx, p))) => {
                    (format!("[{i} #{idx:02} {}]", p.label()), theme::STATE_RUNNING)
                }
                (SlotState::Hot, None) => (format!("[{i} hot]"), theme::STATE_RUNNING),
                (SlotState::Evicting, _) => (format!("[{i} evict]"), theme::STATE_SWAPPED),
            };
            ui.label(RichText::new(label).size(11.5).color(col).monospace());
        }
        ui.add_space(12.0);
        ui.label(
            RichText::new(format!(
                "KV {:.1}/{:.0} GB   tok/s {:.0}   max_new {}",
                gpu.hbm_used_gb, gpu.hbm_total_gb, gpu.swarm_tok_s, max_new_tokens
            ))
            .size(11.5)
            .color(theme::TEXT_SECONDARY)
            .monospace(),
        );
    });

    if let Some((text, status, summary)) = goal_summary {
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("goal:").size(11.5).color(theme::TEXT_DIM));
            ui.label(
                RichText::new(status_label(status))
                    .size(11.5)
                    .strong()
                    .color(status_color(status)),
            );
            ui.add_space(8.0);
            ui.label(
                RichText::new(truncate(&text, 100))
                    .size(11.5)
                    .color(theme::TEXT_PRIMARY),
            );
            if let Some(s) = summary {
                ui.add_space(10.0);
                ui.label(RichText::new(format!("plan: {s}")).size(11.0).color(theme::TEXT_DIM));
            }
        });
    }

    let _ = AgentLifecycle::Idle;
}

fn status_label(s: GoalStatus) -> &'static str {
    match s {
        GoalStatus::Decomposing => "DECOMPOSING",
        GoalStatus::Running => "RUNNING",
        GoalStatus::Merging => "MERGING",
        GoalStatus::Done => "DONE",
        GoalStatus::Failed => "FAILED",
        GoalStatus::Cancelled => "CANCELLED",
    }
}

fn status_color(s: GoalStatus) -> Color32 {
    match s {
        GoalStatus::Decomposing => theme::STATE_LOADING,
        GoalStatus::Running => theme::STATE_RUNNING,
        GoalStatus::Merging => theme::STATE_LOADING,
        GoalStatus::Done => theme::STATE_IDLE,
        GoalStatus::Failed => theme::STATE_FAILED,
        GoalStatus::Cancelled => theme::STATE_SWAPPED,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max).collect();
        t.push('…');
        t
    }
}
