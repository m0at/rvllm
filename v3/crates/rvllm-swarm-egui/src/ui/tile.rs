// Single agent tile rendering.

use egui::{Color32, CornerRadius, Label, Margin, RichText, Stroke};

use crate::ipc::AgentId;
use crate::state::{AgentLifecycle, AgentState, Persona};
use crate::theme;

#[derive(Clone)]
pub struct TileData {
    pub id: AgentId,
    pub display_index: u8,
    pub persona: Persona,
    pub lifecycle: AgentLifecycle,
    pub task_summary: Option<String>,
    pub iter_progress: Option<(u32, u32)>,
    pub tok_s_decode: f32,
    pub tokens_out: u64,
    pub last_tool: Option<String>,
    pub flash: Option<String>,
    pub last_history: Option<(String, String)>,
    pub last_error: Option<String>,
    pub resident: bool,
    pub pinned: bool,
}

impl TileData {
    pub fn from_agent(a: &AgentState) -> Self {
        let last_history = a
            .history
            .last()
            .map(|e| (e.kind.label().to_string(), e.text.clone()));
        Self {
            id: a.id,
            display_index: a.display_index,
            persona: a.persona,
            lifecycle: a.lifecycle,
            task_summary: a.current_task_summary.clone(),
            iter_progress: a.iteration_progress,
            tok_s_decode: a.metrics.tok_s_decode,
            tokens_out: a.metrics.tokens_out_total,
            last_tool: a.last_tool.clone(),
            flash: a.flash.clone(),
            last_history,
            last_error: a.last_error.clone(),
            resident: a.metrics.hbm_resident,
            pinned: a.pinned,
        }
    }
}

pub struct TileResponse {
    pub clicked: bool,
}

pub fn render(ui: &mut egui::Ui, data: &TileData) -> TileResponse {
    let border_col = state_colour(data.lifecycle);
    let bg = if matches!(data.lifecycle, AgentLifecycle::Running | AgentLifecycle::Loading) {
        theme::BG_TILE_ACTIVE
    } else {
        theme::BG_TILE
    };

    let mut clicked = false;

    let resp = egui::Frame::new()
        .fill(bg)
        .stroke(Stroke::new(1.0, border_col))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::same(3))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(2.0, 1.0);
            let mut badges = String::new();
            if data.resident {
                badges.push_str(" hot");
            }
            if data.pinned {
                badges.push_str(" pin");
            }
            trunc_label(
                ui,
                RichText::new(format!(
                    "#{:02} {} {}{}",
                    data.display_index,
                    data.persona.label(),
                    data.lifecycle.label(),
                    badges
                ))
                .size(9.8)
                .strong()
                .color(theme::TEXT_PRIMARY)
                .monospace(),
            );

            if let Some(s) = &data.task_summary {
                let progress = data
                    .iter_progress
                    .map(|(i, m)| format!(" {i}/{m}"))
                    .unwrap_or_default();
                trunc_label(
                    ui,
                    RichText::new(format!("task {}{}", truncate(s, 46), progress))
                        .size(9.8)
                        .color(theme::TEXT_PRIMARY),
                );
            } else {
                trunc_label(
                    ui,
                    RichText::new("no task")
                        .size(9.4)
                        .color(theme::TEXT_DIM)
                        .italics(),
                );
            }

            trunc_label(
                ui,
                RichText::new(format!(
                    "tok/s {:>5.1} out {:>5} tool {}",
                    data.tok_s_decode,
                    kilo(data.tokens_out),
                    data.last_tool.as_deref().unwrap_or("-")
                ))
                .size(9.4)
                .color(theme::TEXT_SECONDARY)
                .monospace(),
            );

            if let Some((kind, text)) = &data.last_history {
                trunc_label(
                    ui,
                    RichText::new(format!("{} {}", kind, truncate(text, 54)))
                        .size(8.9)
                        .color(theme::TEXT_DIM)
                        .monospace(),
                );
            } else if let Some(f) = &data.flash {
                trunc_label(
                    ui,
                    RichText::new(format!("› {}", truncate(f, 54)))
                        .size(9.2)
                        .color(theme::ACCENT)
                        .italics(),
                );
            }

            if let Some(err) = &data.last_error {
                trunc_label(
                    ui,
                    RichText::new(format!("err {}", truncate(err, 56)))
                        .size(8.9)
                        .color(theme::STATE_FAILED),
                );
            }
        })
        .response
        .interact(egui::Sense::click());

    if resp.clicked() {
        clicked = true;
    }
    let _ = Color32::BLACK;

    TileResponse { clicked }
}

fn trunc_label(ui: &mut egui::Ui, text: RichText) {
    let width = ui.available_width();
    ui.add_sized([width, 0.0], Label::new(text).truncate());
}

fn state_colour(s: AgentLifecycle) -> Color32 {
    match s {
        AgentLifecycle::Idle => theme::STATE_IDLE,
        AgentLifecycle::Queued => theme::STATE_QUEUED,
        AgentLifecycle::Loading => theme::STATE_LOADING,
        AgentLifecycle::Running => theme::STATE_RUNNING,
        AgentLifecycle::Swapped => theme::STATE_SWAPPED,
        AgentLifecycle::Failed => theme::STATE_FAILED,
        AgentLifecycle::Paused => theme::TEXT_DIM,
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

fn kilo(n: u64) -> String {
    if n >= 10_000 {
        format!("{:.1}k", n as f32 / 1000.0)
    } else {
        n.to_string()
    }
}
