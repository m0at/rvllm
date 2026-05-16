use std::sync::mpsc;
use std::thread;

#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use tokenizers::Tokenizer;

use crate::ServeConfig;

#[derive(Clone)]
pub struct WorkerHandle {
    tx: mpsc::Sender<Job>,
}

#[derive(Clone, Debug)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub stop: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct GenerateOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

struct Job {
    req: GenerateRequest,
    tx: mpsc::Sender<Result<GenerateOutput, String>>,
}

impl WorkerHandle {
    pub fn start(config: ServeConfig) -> Result<Self, String> {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();

        thread::Builder::new()
            .name("rvllm-engine".into())
            .spawn(move || worker_loop(config, job_rx, init_tx))
            .map_err(|e| format!("spawn engine worker: {e}"))?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self { tx: job_tx }),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(format!("engine worker exited during init: {e}")),
        }
    }

    pub fn generate(&self, req: GenerateRequest) -> Result<GenerateOutput, String> {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(Job { req, tx })
            .map_err(|e| format!("engine worker is not running: {e}"))?;
        rx.recv()
            .map_err(|e| format!("engine worker dropped response: {e}"))?
    }
}

fn worker_loop(
    config: ServeConfig,
    rx: mpsc::Receiver<Job>,
    init_tx: mpsc::Sender<Result<(), String>>,
) {
    let mut engine = match EngineState::load(config.clone()) {
        Ok(engine) => {
            let _ = init_tx.send(Ok(()));
            engine
        }
        Err(e) => {
            let _ = init_tx.send(Err(e));
            return;
        }
    };

    for job in rx {
        let result = engine.generate(job.req);
        let _ = job.tx.send(result);
    }
}

enum EngineState {
    DryRun(DryRunEngine),
    #[cfg(feature = "cuda")]
    Gemma4(Box<CudaGemma4Engine>),
}

impl EngineState {
    fn load(config: ServeConfig) -> Result<Self, String> {
        if config.dry_run {
            tracing::warn!("RVLLM_DRY_RUN enabled: CUDA engine will not be loaded");
            return Ok(Self::DryRun(DryRunEngine { config }));
        }
        load_cuda_engine(config)
    }

    fn generate(&mut self, req: GenerateRequest) -> Result<GenerateOutput, String> {
        match self {
            EngineState::DryRun(e) => e.generate(req),
            #[cfg(feature = "cuda")]
            EngineState::Gemma4(e) => e.generate(req),
        }
    }
}

struct DryRunEngine {
    config: ServeConfig,
}

impl DryRunEngine {
    fn generate(&self, req: GenerateRequest) -> Result<GenerateOutput, String> {
        Ok(GenerateOutput {
            text: "RVLLM_DRY_RUN".into(),
            prompt_tokens: rough_token_count(&req.prompt).min(self.config.max_model_len),
            completion_tokens: 1,
        })
    }
}

#[cfg(feature = "cuda")]
struct CudaGemma4Engine {
    config: ServeConfig,
    tokenizer: Tokenizer,
    bringup: rvllm_runtime::gemma4_bring_up::Gemma4Bringup,
    _embedding_mod: rvllm_kernels::LoadedModule,
    fn_embed: rvllm_kernels::KernelFn,
    fn_argmax: rvllm_kernels::KernelFn,
    bos_id: Option<u32>,
    stop_token_ids: Vec<u32>,
}

#[cfg(feature = "cuda")]
fn load_cuda_engine(config: ServeConfig) -> Result<EngineState, String> {
    use rvllm_core::{ModelArch, ModelConfig};
    use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};

    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let model_cfg = ModelConfig::load_hf(&model_dir)
        .map_err(|e| format!("config parse {}: {e}", model_dir.display()))?;
    if model_cfg.architecture != ModelArch::Gemma4 {
        return Err(format!(
            "rvllm-server only serves Gemma 4 here; config architecture is {:?}",
            model_cfg.architecture
        ));
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("tokenizer load {}: {e}", tokenizer_path.display()))?;

    let paths = Gemma4EnginePaths {
        model_dir,
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path_or_placeholder("RVLLM_CUTLASS_SO"),
        fa3_so: env_path_or_placeholder("RVLLM_FA3_SO"),
        policy_json: env_path_or_placeholder("RVLLM_POLICY"),
    };
    let arena_bytes = arena_bytes()?;

    tracing::info!(
        arena_gb = (arena_bytes as f64) / 1_073_741_824.0,
        "loading Gemma 4 engine"
    );
    let bringup =
        Gemma4Bringup::load(paths, arena_bytes).map_err(|e| format!("gemma4 bringup: {e}"))?;
    let embedding_mod = bringup
        .kernels
        .load_ptx("embedding_gather_f16")
        .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
    let fn_embed = embedding_mod
        .get_function("embedding_gather_f16_kernel")
        .map_err(|e| format!("get embedding_gather_f16_kernel: {e}"))?;
    let fn_argmax = bringup.fused.fn_argmax;

    let bos_id = tokenizer.token_to_id("<bos>").or(Some(2));
    let stop_token_ids = stop_token_ids(&tokenizer);

    tracing::info!(
        model = %config.served_model_name,
        stop_ids = ?stop_token_ids,
        "Gemma 4 engine ready"
    );
    Ok(EngineState::Gemma4(Box::new(CudaGemma4Engine {
        config,
        tokenizer,
        bringup,
        _embedding_mod: embedding_mod,
        fn_embed,
        fn_argmax,
        bos_id,
        stop_token_ids,
    })))
}

#[cfg(not(feature = "cuda"))]
fn load_cuda_engine(_config: ServeConfig) -> Result<EngineState, String> {
    Err(
        "rvllm-server was built without --features cuda; set RVLLM_DRY_RUN=1 for bind-only checks"
            .into(),
    )
}

#[cfg(feature = "cuda")]
impl CudaGemma4Engine {
    fn generate(&mut self, req: GenerateRequest) -> Result<GenerateOutput, String> {
        let encoding = self
            .tokenizer
            .encode(req.prompt.as_str(), false)
            .map_err(|e| format!("tokenize: {e}"))?;
        let mut prompt_ids = encoding.get_ids().to_vec();
        if let Some(bos) = self.bos_id {
            if prompt_ids.first().copied() != Some(bos) {
                prompt_ids.insert(0, bos);
            }
        }

        if prompt_ids.len() >= self.config.max_model_len {
            return Err(format!(
                "prompt has {} tokens, max_model_len is {}",
                prompt_ids.len(),
                self.config.max_model_len
            ));
        }
        let max_new = req
            .max_tokens
            .min(self.config.max_model_len.saturating_sub(prompt_ids.len()));
        if max_new == 0 {
            return Err("max_tokens leaves no decode room under max_model_len".into());
        }

        let output_ids = unsafe {
            self.bringup.run_generate(
                self.fn_embed,
                self.fn_argmax,
                &prompt_ids,
                max_new,
                &self.stop_token_ids,
            )
        }
        .map_err(|e| format!("gemma4 generate: {e}"))?;

        let mut text = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| format!("detokenize: {e}"))?;
        truncate_on_stop_strings(&mut text, &req.stop);

        Ok(GenerateOutput {
            text,
            prompt_tokens: prompt_ids.len(),
            completion_tokens: output_ids.len(),
        })
    }
}

#[cfg(feature = "cuda")]
fn env_path(name: &str) -> Result<PathBuf, String> {
    std::env::var(name)
        .map(PathBuf::from)
        .map_err(|_| format!("missing env var: {name}"))
}

#[cfg(feature = "cuda")]
fn env_path_or_placeholder(name: &str) -> PathBuf {
    std::env::var(name)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/dev/null"))
}

#[cfg(feature = "cuda")]
fn arena_bytes() -> Result<usize, String> {
    if let Some(gb) = std::env::var("RVLLM_ARENA_GB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        return Ok(gb * 1024 * 1024 * 1024);
    }

    let mut free: usize = 0;
    let mut total: usize = 0;
    let probe_ctx =
        rvllm_mem::context::CudaContextHandle::init(0).map_err(|e| format!("probe ctx: {e}"))?;
    unsafe {
        cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _);
    }
    drop(probe_ctx);

    let reserve = 512 * 1024 * 1024;
    Ok(if free > reserve { free - reserve } else { free })
}

#[cfg(feature = "cuda")]
fn stop_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    if let Ok(raw) = std::env::var("RVLLM_EOS") {
        let ids: Vec<u32> = raw
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect();
        if !ids.is_empty() {
            return ids;
        }
    }

    let mut ids = Vec::new();
    for token in ["<end_of_turn>", "<eos>", "</s>"] {
        if let Some(id) = tokenizer.token_to_id(token) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    if ids.is_empty() {
        ids.push(107);
    }
    ids
}

#[cfg(feature = "cuda")]
fn truncate_on_stop_strings(text: &mut String, stops: &[String]) {
    let mut cut = text.len();
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(idx) = text.find(stop) {
            cut = cut.min(idx);
        }
    }
    text.truncate(cut);
}

fn rough_token_count(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}
