use egui::{Color32, CornerRadius, Margin, RichText, ScrollArea};

use crate::app::SwarmApp;
use crate::ipc::{Cmd, LogLevel};
use crate::state::{ExecutionMode, GoalStatus};
use crate::theme;
use crate::ui::task_graph;

pub fn render(ui: &mut egui::Ui, app: &mut SwarmApp) {
    ui.vertical(|ui| {
        ui.label(
            RichText::new("SWARM CONSOLE")
                .size(15.0)
                .strong()
                .color(theme::ACCENT),
        );
        ui.add_space(8.0);

        // ---- Goal DAG ----
        section(ui, "Current goal", |ui| {
            task_graph::render(ui, app);
        });
        ui.add_space(8.0);

        // ---- Scheduler ----
        section(ui, "Scheduler", |ui| {
            let (mode, n_live, decode_batch_target, kv_gb) = {
                let g = app.state.read();
                (
                    g.settings.scheduler.execution_mode,
                    g.settings.scheduler.n_live_slots,
                    g.settings.scheduler.decode_batch_target,
                    g.settings.scheduler.kv_per_agent_gb,
                )
            };
            ui.horizontal(|ui| {
                let operator_selected = mode == ExecutionMode::Operator30;
                if ui
                    .selectable_label(operator_selected, "operator-30")
                    .clicked()
                    && !operator_selected
                {
                    let _ = app.cmd_tx.send(Cmd::SetExecutionMode {
                        mode: ExecutionMode::Operator30,
                        decode_batch_target: ExecutionMode::Operator30.decode_batch_target(),
                    });
                }
                let saturator_selected = mode == ExecutionMode::Saturator;
                if ui
                    .selectable_label(saturator_selected, "saturator")
                    .clicked()
                    && !saturator_selected
                {
                    let _ = app.cmd_tx.send(Cmd::SetExecutionMode {
                        mode: ExecutionMode::Saturator,
                        decode_batch_target: ExecutionMode::Saturator.decode_batch_target(),
                    });
                }
            });
            ui.label(
                RichText::new(mode.description())
                    .size(11.0)
                    .color(theme::TEXT_DIM),
            );
            let mut n_live_local = n_live;
            ui.horizontal(|ui| {
                ui.label(RichText::new("N_LIVE").size(12.0).color(theme::TEXT_SECONDARY));
                ui.add(egui::DragValue::new(&mut n_live_local).range(1..=512));
            });
            if n_live_local != n_live {
                let _ = app.cmd_tx.send(Cmd::AdjustScheduler { n_live: n_live_local });
            }
            let mut batch_local = decode_batch_target;
            ui.horizontal(|ui| {
                ui.label(RichText::new("decode batch").size(12.0).color(theme::TEXT_SECONDARY));
                ui.add(egui::DragValue::new(&mut batch_local).range(1..=512));
            });
            if batch_local != decode_batch_target {
                let _ = app.cmd_tx.send(Cmd::SetExecutionMode {
                    mode,
                    decode_batch_target: batch_local,
                });
            }
            ui.label(
                RichText::new(format!(
                    "KV per visible agent: {:.1} GB  (live x KV = {:.1} GB)",
                    kv_gb,
                    kv_gb * n_live_local as f32
                ))
                .size(11.0)
                .color(theme::TEXT_DIM),
            );
        });
        ui.add_space(8.0);

        // ---- Git ----
        section(ui, "Git", |ui| {
            let g = app.state.read().git.clone();
            label_line(ui, "main", &g.main_head);
            label_line(
                ui,
                "staging",
                g.staging_head.as_deref().unwrap_or("(none)"),
            );
            label_line(ui, "worktrees", &g.open_worktrees.to_string());
            label_line(ui, "pending merges", &g.pending_merges.to_string());
        });
        ui.add_space(8.0);

        // ---- Log ----
        section(ui, "Log", |ui| {
            let snapshot: Vec<(LogLevel, String, String, i64)> = {
                let g = app.state.read();
                g.log
                    .iter_rev()
                    .take(80)
                    .map(|l| (l.level, l.source.clone(), l.message.clone(), l.ts_ms))
                    .collect()
            };
            ScrollArea::vertical()
                .max_height(240.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for (lvl, src, msg, _ts) in snapshot {
                        let col = match lvl {
                            LogLevel::Trace | LogLevel::Debug => theme::TEXT_DIM,
                            LogLevel::Info => theme::TEXT_SECONDARY,
                            LogLevel::Warn => theme::STATE_QUEUED,
                            LogLevel::Error => theme::STATE_FAILED,
                        };
                        ui.label(
                            RichText::new(format!("[{src}] {msg}"))
                                .size(11.0)
                                .color(col)
                                .monospace(),
                        );
                    }
                });
        });
        ui.add_space(8.0);

        // ---- Actions ----
        section(ui, "Actions", |ui| {
            let current_goal = {
                let g = app.state.read();
                g.master.current_goal
            };
            if ui
                .button(RichText::new("Approve merge of current goal").size(12.0))
                .clicked()
            {
                if let Some(gid) = current_goal {
                    let _ = app.cmd_tx.send(Cmd::ApproveMerge { goal: gid });
                }
            }
            if ui
                .button(RichText::new("Cancel current goal").size(12.0))
                .clicked()
            {
                if let Some(gid) = current_goal {
                    let _ = app.cmd_tx.send(Cmd::CancelGoal { goal: gid });
                }
            }
        });

        let _ = Color32::BLACK;
        let _ = GoalStatus::Running;
    });
}

fn section<R>(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let mut inner: Option<R> = None;
    egui::Frame::new()
        .fill(theme::BG_CARD)
        .stroke(egui::Stroke::new(1.0, theme::BORDER_SOFT))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(10))
        .show(ui, |ui| {
            ui.label(
                RichText::new(title)
                    .size(12.5)
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(4.0);
            inner = Some(body(ui));
        });
    inner.expect("section body must run")
}

fn label_line(ui: &mut egui::Ui, k: &str, v: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(k).size(11.0).color(theme::TEXT_DIM));
        ui.add_space(6.0);
        ui.label(
            RichText::new(v)
                .size(11.0)
                .color(theme::TEXT_PRIMARY)
                .monospace(),
        );
    });
}
