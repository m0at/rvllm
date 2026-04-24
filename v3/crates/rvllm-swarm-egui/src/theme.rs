// Theme tokens for the swarm UI. Palette is the rvLLM orange-accent dark
// theme, extended with state colours for the agent grid.

use egui::{Color32, CornerRadius, FontFamily, FontId, Stroke, Style, TextStyle, Visuals};

// Base palette carried over from the earlier standalone egui experiments.
pub const BG_DARK: Color32 = Color32::from_rgb(15, 15, 18);
pub const BG_PANEL: Color32 = Color32::from_rgb(23, 24, 29);
pub const BG_INPUT: Color32 = Color32::from_rgb(31, 33, 40);
pub const BG_CARD: Color32 = Color32::from_rgb(28, 30, 36);
pub const BG_TILE: Color32 = Color32::from_rgb(26, 28, 34);
pub const BG_TILE_ACTIVE: Color32 = Color32::from_rgb(32, 34, 42);
pub const BORDER: Color32 = Color32::from_rgb(61, 66, 76);
pub const BORDER_SOFT: Color32 = Color32::from_rgb(45, 49, 58);

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(232, 234, 238);
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(163, 169, 180);
pub const TEXT_DIM: Color32 = Color32::from_rgb(110, 117, 129);

pub const ACCENT: Color32 = Color32::from_rgb(214, 109, 55);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(233, 132, 82);
pub const INFO: Color32 = Color32::from_rgb(97, 160, 231);

// Lifecycle state colours (tile border + tile tint).
pub const STATE_IDLE: Color32 = Color32::from_rgb(90, 96, 108);
pub const STATE_QUEUED: Color32 = Color32::from_rgb(222, 186, 65);
pub const STATE_LOADING: Color32 = Color32::from_rgb(97, 160, 231);
pub const STATE_RUNNING: Color32 = Color32::from_rgb(214, 109, 55);
pub const STATE_SWAPPED: Color32 = Color32::from_rgb(156, 112, 199);
pub const STATE_FAILED: Color32 = Color32::from_rgb(210, 78, 78);

pub const BG_LOG: Color32 = Color32::from_rgb(18, 19, 24);

pub fn apply_theme(ctx: &egui::Context) {
    let mut style = Style::default();

    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(19.0, FontFamily::Proportional),
    );
    style
        .text_styles
        .insert(TextStyle::Body, FontId::new(13.5, FontFamily::Proportional));
    style.text_styles.insert(
        TextStyle::Monospace,
        FontId::new(12.0, FontFamily::Monospace),
    );
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(13.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Small,
        FontId::new(11.0, FontFamily::Proportional),
    );

    let mut visuals = Visuals::dark();
    visuals.panel_fill = BG_PANEL;
    visuals.window_fill = BG_PANEL;
    visuals.extreme_bg_color = BG_INPUT;
    visuals.faint_bg_color = BG_DARK;
    visuals.override_text_color = Some(TEXT_PRIMARY);

    visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_SECONDARY);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(6);

    visuals.widgets.inactive.bg_fill = BG_INPUT;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(6);

    visuals.widgets.hovered.bg_fill = ACCENT_HOVER;
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(6);

    visuals.widgets.active.bg_fill = ACCENT;
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.active.corner_radius = CornerRadius::same(6);

    visuals.selection.bg_fill = ACCENT.linear_multiply(0.35);
    visuals.selection.stroke = Stroke::new(1.0, ACCENT_HOVER);

    visuals.window_corner_radius = CornerRadius::same(10);
    visuals.window_stroke = Stroke::new(1.0, BORDER);

    style.visuals = visuals;
    style.spacing.item_spacing = egui::vec2(6.0, 6.0);
    style.spacing.window_margin = egui::Margin::same(10);

    ctx.set_style(style);
}
