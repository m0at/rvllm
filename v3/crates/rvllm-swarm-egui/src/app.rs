// Top-level egui app. Owns the ControllerHandle and forwards user actions
// as Cmd messages.

use crossbeam_channel::{Receiver, Sender};
use egui::{Margin, Vec2};

use crate::controller::{self, ControllerHandle};
use crate::ipc::{AppEvent, Cmd};
use crate::remote;
use crate::state::BackendKind;
use crate::theme;
use crate::ui;

pub struct SwarmApp {
    pub state: std::sync::Arc<parking_lot::RwLock<crate::state::AppState>>,
    pub cmd_tx: Sender<Cmd>,
    pub event_rx: Receiver<AppEvent>,
    pub pending_goal: String,
    pub selected_agent: Option<crate::ipc::AgentId>,
    pub theme_applied: bool,
    pub auto_demo_enabled: bool,
    pub auto_demo_seeded: bool,
    pub remote_addr: Option<String>,
}

impl SwarmApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let repo_root = crate::detect_repo_root();
        let backend_kind = default_backend_kind();
        let remote_addr = remote_addr();
        let ControllerHandle { state, cmd_tx, event_rx } = if let Some(addr) = remote_addr.clone() {
            spawn_remote_client(repo_root, addr)
        } else {
            controller::spawn(repo_root, backend_kind)
        };

        Self {
            state,
            cmd_tx,
            event_rx,
            pending_goal: default_goal_text(),
            selected_agent: None,
            theme_applied: false,
            auto_demo_enabled: auto_demo_enabled(),
            auto_demo_seeded: false,
            remote_addr,
        }
    }

    pub fn submit_goal(&mut self) {
        let text = self.pending_goal.trim().to_owned();
        if text.is_empty() {
            return;
        }
        let _ = self.cmd_tx.send(Cmd::SubmitGoal { text });
        self.pending_goal.clear();
    }

    pub fn submit_broadcast(&mut self) {
        let text = self.pending_goal.trim().to_owned();
        if text.is_empty() {
            return;
        }
        let _ = self.cmd_tx.send(Cmd::SubmitBroadcast { text });
        self.pending_goal.clear();
    }

    fn drain_events(&mut self) {
        // We don't actually need to do anything with most events in the UI
        // because the shared AppState is already updated by the controller.
        // We just drain to keep the channel from filling up, and we force
        // a repaint on any event.
        let mut any = false;
        while self.event_rx.try_recv().is_ok() {
            any = true;
        }
        let _ = any;
    }

    fn maybe_seed_demo(&mut self) {
        if !self.auto_demo_enabled || self.auto_demo_seeded {
            return;
        }
        let should_seed = {
            let g = self.state.read();
            matches!(g.backend_kind, BackendKind::Mock) && g.tasks.goals.is_empty()
        };
        if !should_seed {
            return;
        }
        let _ = self.cmd_tx.send(Cmd::SubmitBroadcast {
            text: default_demo_broadcast_text(),
        });
        self.auto_demo_seeded = true;
    }
}

impl eframe::App for SwarmApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            theme::apply_theme(ctx);
            self.theme_applied = true;
        }

        self.drain_events();
        self.maybe_seed_demo();

        // Top bar (master strip).
        egui::TopBottomPanel::top("top_bar")
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_DARK)
                    .inner_margin(Margin::symmetric(6, 4))
                    .stroke(egui::Stroke::new(1.0, theme::BORDER)),
            )
            .show(ctx, |ui| {
                ui::top_bar::render(ui, self);
            });

        // Right side panel.
        egui::SidePanel::right("side_panel")
            .resizable(true)
            .default_width(280.0)
            .min_width(220.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .inner_margin(Margin::same(4)),
            )
            .show(ctx, |ui| {
                ui::side_panel::render(ui, self);
            });

        // Central: the 6x5 tile grid.
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .inner_margin(Margin::same(2)),
            )
            .show(ctx, |ui| {
                ui.allocate_ui(ui.available_size(), |ui| {
                    ui::grid::render(ui, self);
                });
                let _ = Vec2::new(0.0, 0.0);
            });

        // Detail modal if an agent is selected.
        if self.selected_agent.is_some() {
            ui::detail_modal::render(ctx, self);
        }

        // Keep ticking while we're rendering agent animations.
        ctx.request_repaint_after(std::time::Duration::from_millis(150));
    }
}

fn default_goal_text() -> String {
    "explain angular momentum".into()
}

fn default_demo_broadcast_text() -> String {
    "Broadcast one shared response across all 30 agents so the swarm UI shows live fanout activity."
        .into()
}

fn auto_demo_enabled() -> bool {
    match std::env::var("SWARM_AUTO_DEMO") {
        Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => false,
    }
}

fn default_backend_kind() -> BackendKind {
    if cfg!(feature = "cuda") {
        BackendKind::Rvllm
    } else {
        BackendKind::Mock
    }
}

fn remote_addr() -> Option<String> {
    std::env::var("SWARM_REMOTE_ADDR")
        .ok()
        .map(|addr| addr.trim().to_owned())
        .filter(|addr| !addr.is_empty())
}

fn spawn_remote_client(
    repo_root: std::path::PathBuf,
    addr: String,
) -> ControllerHandle {
    let state = std::sync::Arc::new(parking_lot::RwLock::new(crate::state::AppState::bootstrap(
        repo_root,
        BackendKind::Mock,
    )));
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<Cmd>();
    let (_event_tx, event_rx) = crossbeam_channel::bounded::<AppEvent>(1);
    remote::spawn_client(std::sync::Arc::clone(&state), cmd_rx, addr);
    ControllerHandle {
        state,
        cmd_tx,
        event_rx,
    }
}
