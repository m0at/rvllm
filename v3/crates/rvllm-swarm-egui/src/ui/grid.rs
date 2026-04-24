use egui::{Color32, Vec2};

use crate::app::SwarmApp;
use crate::ui::tile;

pub fn render(ui: &mut egui::Ui, app: &mut SwarmApp) {
    let avail = ui.available_size();
    let gutter = 2.0;

    // Snapshot minimal info for rendering.
    let snapshot: Vec<tile::TileData> = {
        let g = app.state.read();
        g.agents.iter().map(tile::TileData::from_agent).collect()
    };
    let count = snapshot.len().max(1);
    let target_tile_w = 250.0;
    let cols = (((avail.x + gutter) / (target_tile_w + gutter)).floor() as usize)
        .clamp(6, 10)
        .min(count);
    let rows = count.div_ceil(cols);
    let tile_w = ((avail.x - gutter * (cols as f32 - 1.0)) / cols as f32).max(150.0);
    let tile_h = ((avail.y - gutter * (rows as f32 - 1.0)) / rows as f32)
        .clamp(74.0, 112.0);

    let mut clicked: Option<crate::ipc::AgentId> = None;

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for row in 0..rows {
                ui.horizontal(|ui| {
                    for col in 0..cols {
                        let idx = row * cols + col;
                        if let Some(data) = snapshot.get(idx) {
                            let (rect, _) =
                                ui.allocate_exact_size(Vec2::new(tile_w, tile_h), egui::Sense::hover());
                            let mut child = ui.new_child(
                                egui::UiBuilder::new()
                                    .max_rect(rect)
                                    .layout(egui::Layout::top_down(egui::Align::Min)),
                            );
                            child.set_clip_rect(rect);
                            if tile::render(&mut child, data).clicked {
                                    clicked = Some(data.id);
                            }
                        } else {
                            ui.allocate_ui(Vec2::new(tile_w, tile_h), |_| {});
                        }
                        if col + 1 < cols {
                            ui.add_space(gutter - ui.spacing().item_spacing.x);
                        }
                    }
                });
                if row + 1 < rows {
                    ui.add_space(gutter - ui.spacing().item_spacing.y);
                }
            }
        });
    let _ = Color32::BLACK;

    if let Some(id) = clicked {
        app.selected_agent = Some(id);
    }
}
