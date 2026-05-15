use std::sync::Arc;

use rvllm_serve::{
    config::ServerConfig,
    http::{self, AppState},
    openai::ChatTemplate,
    worker::{self, WorkerPaths},
};
use tokio::sync::Semaphore;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    if let Err(e) = run() {
        eprintln!("rvllm-server: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = ServerConfig::from_env()?;
    tracing::info!(
        host = %config.host,
        port = config.port,
        max_num_seqs = config.max_num_seqs,
        model = %config.served_model_name,
        dry_run = config.dry_run,
        "rvllm-server starting"
    );

    // Tokenizer + chat template up front so failures abort startup
    // before binding the port.
    let (tokenizer, chat_template) = load_text_artifacts(&config)?;
    let tokenizer = Arc::new(tokenizer);
    let chat_template = Arc::new(chat_template);

    // Spawn engine worker (blocks until engine is ready or fails).
    let worker_handle = worker::spawn(
        WorkerPaths {
            model_dir: config.model_dir.clone(),
            kernels_dir: config.kernels_dir.clone(),
            cutlass_so: config.cutlass_so.clone(),
            fa3_so: config.fa3_so.clone(),
            policy_json: config.policy_json.clone(),
        },
        tokenizer.clone(),
        chat_template.clone(),
        config.dry_run,
    )?;

    let gate = Arc::new(Semaphore::new(config.max_num_seqs));
    let state = AppState {
        config: Arc::new(config),
        worker: worker_handle.clone(),
        tokenizer,
        chat_template,
        gate,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("tokio runtime: {e}"))?;
    rt.block_on(http::serve(state))?;
    Ok(())
}

fn load_text_artifacts(
    cfg: &ServerConfig,
) -> Result<(tokenizers::Tokenizer, ChatTemplate), String> {
    if cfg.dry_run {
        // Dry-run still needs a tokenizer + template if the user
        // wants to hit /v1/chat/completions on the box. If
        // RVLLM_MODEL_DIR isn't a real dir we synthesize a no-op
        // tokenizer + template so that /health and /v1/models work.
        let tok_path = cfg.model_dir.join("tokenizer.json");
        if !tok_path.exists() {
            tracing::warn!(
                "RVLLM_DRY_RUN + no tokenizer at {} — chat completions will 400",
                tok_path.display()
            );
            let tokenizer = build_stub_tokenizer();
            let chat_template = build_stub_template();
            return Ok((tokenizer, chat_template));
        }
    }
    let tok_path = cfg.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| format!("tokenizer load {}: {e}", tok_path.display()))?;
    let chat_template = ChatTemplate::from_model_dir(&cfg.model_dir)?;
    Ok((tokenizer, chat_template))
}

fn build_stub_tokenizer() -> tokenizers::Tokenizer {
    // Minimal whitespace tokenizer so /health + /v1/models work
    // off-GPU. Calls to /v1/chat/completions will likely produce
    // garbage output, but in dry-run we never invoke the engine
    // anyway (the worker replies with Error).
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    let model = WordLevel::builder()
        .unk_token("[UNK]".to_string())
        .build()
        .expect("build stub wordlevel");
    let mut tok = tokenizers::Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(Whitespace));
    tok
}

fn build_stub_template() -> ChatTemplate {
    // A trivial "user: ...\nassistant:" template so the openai shape
    // works off-GPU. Engine still won't fire in dry-run.
    let src = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}assistant:";
    ChatTemplate::from_source(src.to_string(), String::new(), String::new())
        .expect("stub template compile")
}
