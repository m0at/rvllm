use tracing_subscriber::EnvFilter;
use rvllm_swarm_egui::app::SwarmApp;

fn main() {
    // Tracing. Default to info,rvllm_swarm_egui=debug so we see lifecycle events.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,rvllm_swarm_egui=debug"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    // Determine viewport size. 45" monitor target is 3840x2160; we default
    // to a generous but safe 2200x1400 on boot (fits 1440p laptops too) and
    // let the OS/user maximise.
    let (init_w, init_h) = match std::env::var("SWARM_VIEWPORT").ok().as_deref() {
        Some("4k") => (3800.0, 2100.0),
        Some("ultrawide") => (5000.0, 2100.0),
        Some(s) => {
            // "WIDTHxHEIGHT"
            let mut it = s.split('x');
            let w = it.next().and_then(|s| s.parse().ok()).unwrap_or(2200.0);
            let h = it.next().and_then(|s| s.parse().ok()).unwrap_or(1400.0);
            (w, h)
        }
        None => (2200.0, 1400.0),
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([init_w, init_h])
            .with_min_inner_size([1280.0, 800.0])
            .with_title("rvLLM Swarm Commander"),
        ..Default::default()
    };

    eframe::run_native(
        "rvLLM Swarm Commander",
        options,
        Box::new(|cc| Ok(Box::new(SwarmApp::new(cc)))),
    )
    .expect("failed to run rvllm-swarm-egui");
}
