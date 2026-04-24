use egui::{Color32, Margin, RichText, ScrollArea};

use crate::app::SwarmApp;
use crate::ipc::Cmd;
use crate::theme;

pub fn render(ctx: &egui::Context, app: &mut SwarmApp) {
    let Some(agent_id) = app.selected_agent else { return; };

    let (ok, details) = {
        let g = app.state.read();
        match g.agent(agent_id) {
            None => (false, None),
            Some(a) => (
                true,
                Some(DetailSnapshot {
                    display_index: a.display_index,
                    persona: a.persona.label().to_string(),
                    lifecycle: a.lifecycle.label().to_string(),
                    worktree: a.worktree_path.display().to_string(),
                    branch: a.branch.clone(),
                    history: a
                        .history
                        .iter()
                        .map(|e| (e.kind.label().to_string(), e.text.clone()))
                        .collect(),
                    metrics: a.metrics,
                    pinned: a.pinned,
                }),
            ),
        }
    };

    let mut keep_open = true;
    egui::Window::new(
        RichText::new(format!(
            "agent #{:02} ({})",
            details.as_ref().map(|d| d.display_index).unwrap_or(0),
            details
                .as_ref()
                .map(|d| d.persona.as_str())
                .unwrap_or("?"),
        ))
        .size(15.0)
        .strong(),
    )
    .default_size([720.0, 540.0])
    .resizable(true)
    .collapsible(true)
    .open(&mut keep_open)
    .show(ctx, |ui| {
        if !ok {
            ui.label(RichText::new("agent not found").color(theme::STATE_FAILED));
            return;
        }
        let d = details.unwrap();
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("state: {}", d.lifecycle)).size(12.5).strong());
            ui.add_space(12.0);
            ui.label(RichText::new(format!("branch: {}", d.branch.as_deref().unwrap_or("-"))).size(12.0));
            ui.add_space(12.0);
            ui.label(RichText::new(format!("worktree: {}", d.worktree)).size(11.0).color(theme::TEXT_DIM));
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.button("Pause").clicked() {
                let _ = app.cmd_tx.send(Cmd::PauseAgent { agent: agent_id });
            }
            if ui.button("Resume").clicked() {
                let _ = app.cmd_tx.send(Cmd::ResumeAgent { agent: agent_id });
            }
            if ui.button("Reset").clicked() {
                let _ = app.cmd_tx.send(Cmd::ResetAgent { agent: agent_id });
            }
            if ui.button("Force evict").clicked() {
                let _ = app.cmd_tx.send(Cmd::ForceEvict { agent: agent_id });
            }
            let pin_label = if d.pinned { "Unpin" } else { "Pin" };
            if ui.button(pin_label).clicked() {
                let _ = app.cmd_tx.send(Cmd::PinAgent { agent: agent_id, pinned: !d.pinned });
            }
        });
        ui.add_space(6.0);

        egui::Frame::new()
            .fill(theme::BG_LOG)
            .inner_margin(Margin::same(8))
            .show(ui, |ui| {
                ui.label(RichText::new("history").size(12.0).strong());
                ui.add_space(4.0);
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .max_height(320.0)
                    .show(ui, |ui| {
                        for (kind, text) in &d.history {
                            ui.label(
                                RichText::new(format!("{kind}  {text}"))
                                    .size(11.5)
                                    .color(theme::TEXT_PRIMARY)
                                    .monospace(),
                            );
                        }
                    });
            });

        ui.add_space(6.0);
        ui.label(
            RichText::new(format!(
                "tok/s {:.1}  tokens_in {}  tokens_out {}  iter_mean {:.1}",
                d.metrics.tok_s_decode,
                d.metrics.tokens_in_total,
                d.metrics.tokens_out_total,
                d.metrics.iterations_mean
            ))
            .size(11.5)
            .color(theme::TEXT_SECONDARY)
            .monospace(),
        );
    });
    let _ = Color32::BLACK;

    if !keep_open {
        app.selected_agent = None;
    }
}

struct DetailSnapshot {
    display_index: u8,
    persona: String,
    lifecycle: String,
    worktree: String,
    branch: Option<String>,
    history: Vec<(String, String)>,
    metrics: crate::state::AgentMetrics,
    pinned: bool,
}
