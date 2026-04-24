use rvllm_swarm_egui::controller::spawn;
use rvllm_swarm_egui::detect_repo_root;
use rvllm_swarm_egui::remote;
use rvllm_swarm_egui::state::BackendKind;
use tracing_subscriber::EnvFilter;

fn main() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,rvllm_swarm_egui=debug"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    let repo_root = detect_repo_root();
    let backend_kind = if cfg!(feature = "cuda") {
        BackendKind::Rvllm
    } else {
        BackendKind::Mock
    };
    let handle = spawn(repo_root, backend_kind);
    if let Some(max_new_tokens) = std::env::var("RVLLM_MAX_NEW_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        handle.state.write().settings.max_new_tokens = max_new_tokens;
    }
    let addr = std::env::var("SWARM_REMOTE_LISTEN").unwrap_or_else(|_| "127.0.0.1:7878".into());

    remote::serve_forever(&addr, handle.state, handle.cmd_tx)
        .expect("failed to run swarmd");
}
