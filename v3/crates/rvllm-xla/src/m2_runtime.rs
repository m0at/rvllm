use std::path::PathBuf;

use rvllm_core::{
    BatchedPrefillPlan, ConfigError, PrefillRequest, ReqId, Result, RvllmError, TokenId,
};
use rvllm_fused::{M2PrefillKvDType, M2PrefillScanShape};
#[cfg(feature = "tpu")]
use rvllm_loader::M2SafetensorsReader;
use rvllm_loader::{M2CheckpointIndex, M2CheckpointSummary};

use crate::{
    m2_decode_graph_mlir, make_m2_prefill_input_specs, M2GraphAbi, M2GraphShape,
    M2PrefillHostInputSpec, M2WeightUploadPlan, PjrtElementType, M2_VOCAB,
};
#[cfg(feature = "tpu")]
use crate::{CompiledExecutable, PjrtBufferHandle, PjrtClientHandle};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RustPrefillConfig {
    pub model_dir: PathBuf,
    pub batch: usize,
    pub prompt_len: usize,
    pub ctx: usize,
    pub block_size: u32,
    pub kv_dtype: M2PrefillKvDType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RustPrefillPlan {
    pub checkpoint: M2CheckpointSummary,
    pub shape: M2PrefillScanShape,
    pub plan: BatchedPrefillPlan,
    pub input_specs: Vec<M2PrefillHostInputSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RustPrefillDecodeConfig {
    pub model_dir: PathBuf,
    pub batch: usize,
    pub prompt_len: usize,
    pub decode_steps: usize,
    pub ctx: usize,
    pub block_size: u32,
    pub kv_dtype: M2PrefillKvDType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2DecodeRuntimeInputSpec {
    pub name: &'static str,
    pub shape: Vec<i64>,
    pub dtype: PjrtElementType,
    pub nbytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RustPrefillDecodePlan {
    pub prefill: M2RustPrefillPlan,
    pub decode_shape: M2GraphShape,
    pub decode_input_specs: Vec<M2DecodeRuntimeInputSpec>,
    pub decode_output_specs: Vec<M2DecodeRuntimeInputSpec>,
    pub seed_decode_token_ids: Vec<i32>,
    pub seed_decode_positions: Vec<i32>,
    pub decode_steps: usize,
    pub weight_arena_bytes: usize,
    pub weight_entries: usize,
    pub decode_mlir: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct M2PplResult {
    pub n_tokens_scored: usize,
    pub avg_nll: f64,
    pub ppl: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2RuntimeConfig {
    pub model_dir: PathBuf,
    pub batch: usize,
    pub prompt_len: usize,
    pub decode_steps: usize,
    pub ctx: usize,
    pub block_size: u32,
    pub kv_dtype: M2PrefillKvDType,
    pub max_weight_arena_bytes: usize,
    pub device_idx: usize,
    pub mode: M2RuntimeMode,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum M2RuntimeMode {
    #[default]
    PlanningOnly,
    CompileOnly,
    Execute,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum M2RuntimeState {
    PlanningOnly,
    CompiledOnly,
    Executable,
}

impl M2RuntimeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PlanningOnly => "planning-only",
            Self::CompileOnly => "compile-only",
            Self::Execute => "execute",
        }
    }
}

impl std::str::FromStr for M2RuntimeMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "planning-only" | "plan-only" | "plan" => Ok(Self::PlanningOnly),
            "compile-only" | "compile" => Ok(Self::CompileOnly),
            "execute" | "exec" => Ok(Self::Execute),
            other => Err(format!(
                "expected planning-only|compile-only|execute, got {other:?}"
            )),
        }
    }
}

impl M2RuntimeState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PlanningOnly => "planning-only",
            Self::CompiledOnly => "compiled-only",
            Self::Executable => "executable",
        }
    }

    pub fn is_executable(&self) -> bool {
        matches!(self, Self::Executable)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2GenerateRequest {
    pub prompt_token_ids: Vec<i32>,
    pub max_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2GenerateOutput {
    pub generated_token_ids: Vec<i32>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

#[cfg(feature = "tpu")]
pub struct M2Runtime {
    plan: M2RustPrefillDecodePlan,
    state: M2RuntimeState,
    client: Option<PjrtClientHandle>,
    exe: Option<CompiledExecutable>,
    weight_arena: Option<PjrtBufferHandle>,
    device_idx: usize,
}

#[cfg(not(feature = "tpu"))]
pub struct M2Runtime {
    plan: M2RustPrefillDecodePlan,
    state: M2RuntimeState,
}

impl M2RustPrefillPlan {
    pub fn total_host_input_bytes(&self) -> usize {
        self.input_specs.iter().map(|spec| spec.nbytes).sum()
    }
}

impl M2RustPrefillDecodePlan {
    pub fn total_decode_input_bytes(&self) -> usize {
        self.decode_input_specs.iter().map(|spec| spec.nbytes).sum()
    }
}

impl M2Runtime {
    pub fn new(cfg: &M2RuntimeConfig) -> Result<Self> {
        let plan = plan_m2_rust_prefill_decode(&M2RustPrefillDecodeConfig {
            model_dir: cfg.model_dir.clone(),
            batch: cfg.batch,
            prompt_len: cfg.prompt_len,
            decode_steps: cfg.decode_steps,
            ctx: cfg.ctx,
            block_size: cfg.block_size,
            kv_dtype: cfg.kv_dtype,
        })?;
        Self::from_plan_with_mode(
            cfg.model_dir.clone(),
            plan,
            cfg.max_weight_arena_bytes,
            cfg.device_idx,
            cfg.mode,
        )
    }

    pub fn plan(&self) -> &M2RustPrefillDecodePlan {
        &self.plan
    }

    pub fn state(&self) -> &M2RuntimeState {
        &self.state
    }

    pub fn is_executable(&self) -> bool {
        self.state.is_executable()
    }

    #[cfg(feature = "tpu")]
    pub fn from_plan(
        model_dir: PathBuf,
        plan: M2RustPrefillDecodePlan,
        max_weight_arena_bytes: usize,
        device_idx: usize,
    ) -> Result<Self> {
        Self::from_plan_with_mode(
            model_dir,
            plan,
            max_weight_arena_bytes,
            device_idx,
            M2RuntimeMode::Execute,
        )
    }

    #[cfg(feature = "tpu")]
    pub fn from_plan_with_mode(
        model_dir: PathBuf,
        plan: M2RustPrefillDecodePlan,
        max_weight_arena_bytes: usize,
        device_idx: usize,
        mode: M2RuntimeMode,
    ) -> Result<Self> {
        match mode {
            M2RuntimeMode::PlanningOnly => {
                return Ok(Self {
                    plan,
                    state: M2RuntimeState::PlanningOnly,
                    client: None,
                    exe: None,
                    weight_arena: None,
                    device_idx,
                });
            }
            M2RuntimeMode::CompileOnly => {
                let client = PjrtClientHandle::new()?;
                let exe = client.compile(&plan.decode_mlir)?;
                return Ok(Self {
                    plan,
                    state: M2RuntimeState::CompiledOnly,
                    client: Some(client),
                    exe: Some(exe),
                    weight_arena: None,
                    device_idx,
                });
            }
            M2RuntimeMode::Execute => {}
        }
        ensure_decode_executable(&plan)?;
        if max_weight_arena_bytes == 0 {
            return Err(invalid(
                "max_weight_arena_bytes",
                "must be > 0 for TPU M2Runtime",
            ));
        }
        let reader = M2SafetensorsReader::open(&model_dir)?;
        let abi = M2GraphAbi::new(plan.decode_shape.clone())?;
        let weights = M2WeightUploadPlan::from_index_dir(&model_dir, &abi)?;
        let arena = weights.flat_arena(128)?;
        let client = PjrtClientHandle::new()?;
        let exe = client.compile(&plan.decode_mlir)?;
        let weight_arena = arena.upload_flat_arena_to_pjrt(
            &reader,
            &client,
            device_idx,
            max_weight_arena_bytes,
        )?;
        Ok(Self {
            plan,
            state: M2RuntimeState::Executable,
            client: Some(client),
            exe: Some(exe),
            weight_arena: Some(weight_arena),
            device_idx,
        })
    }

    #[cfg(not(feature = "tpu"))]
    pub fn from_plan(
        _model_dir: PathBuf,
        plan: M2RustPrefillDecodePlan,
        _max_weight_arena_bytes: usize,
        _device_idx: usize,
    ) -> Result<Self> {
        Self::from_plan_with_mode(
            _model_dir,
            plan,
            _max_weight_arena_bytes,
            _device_idx,
            M2RuntimeMode::Execute,
        )
    }

    #[cfg(not(feature = "tpu"))]
    pub fn from_plan_with_mode(
        _model_dir: PathBuf,
        plan: M2RustPrefillDecodePlan,
        _max_weight_arena_bytes: usize,
        _device_idx: usize,
        mode: M2RuntimeMode,
    ) -> Result<Self> {
        match mode {
            M2RuntimeMode::PlanningOnly => Ok(Self {
                plan,
                state: M2RuntimeState::PlanningOnly,
            }),
            M2RuntimeMode::CompileOnly => Err(invalid(
                "runtime_mode",
                "compile-only requires building rvllm-xla with feature tpu",
            )),
            M2RuntimeMode::Execute => Err(invalid(
                "runtime_mode",
                "execute requires Rust PJRT plus linked M2 custom-call bodies; no Python/JAX fallback is used",
            )),
        }
    }

    #[cfg(feature = "tpu")]
    pub fn generate_token_ids(&self, req: &M2GenerateRequest) -> Result<M2GenerateOutput> {
        if !self.state.is_executable() {
            return Err(runtime_not_executable(&self.state));
        }
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| runtime_not_executable(&self.state))?;
        let exe = self
            .exe
            .as_ref()
            .ok_or_else(|| runtime_not_executable(&self.state))?;
        let weight_arena = self
            .weight_arena
            .as_ref()
            .ok_or_else(|| runtime_not_executable(&self.state))?;
        let mut prepared = prepare_generate_request(&self.plan, req)?;
        let mut generated_token_ids = Vec::with_capacity(prepared.max_tokens);
        let mut kv_buf = client.buffer_from_host(
            &vec![0u8; self.plan.decode_shape.kv_cache_bytes()],
            &[self.plan.decode_shape.kv_cache_bytes() as i64],
            PjrtElementType::S8,
            self.device_idx,
        )?;

        for _ in 0..prepared.max_tokens {
            let token_buf = client.buffer_from_host(
                &i32_bytes(&prepared.token_ids),
                &[self.plan.decode_shape.batch as i64],
                PjrtElementType::S32,
                self.device_idx,
            )?;
            let pos_buf = client.buffer_from_host(
                &i32_bytes(&prepared.positions),
                &[self.plan.decode_shape.batch as i64],
                PjrtElementType::S32,
                self.device_idx,
            )?;
            let inputs = [&token_buf, &pos_buf, &kv_buf, weight_arena];
            let mut outputs = client.execute(exe, &inputs)?;
            if outputs.len() != 3 {
                return Err(invalid_owned(
                    "decode_outputs",
                    format!("got {}, expected 3", outputs.len()),
                ));
            }
            let next_token = outputs.remove(1);
            let new_kv = outputs.remove(1);
            let mut next_bytes = vec![0u8; self.plan.decode_shape.batch * 4];
            client.buffer_to_host(&next_token, &mut next_bytes)?;
            prepared.token_ids = read_i32s(&next_bytes);
            generated_token_ids.push(prepared.token_ids[0]);
            for pos in &mut prepared.positions {
                *pos += 1;
            }
            kv_buf = new_kv;
        }

        Ok(M2GenerateOutput {
            prompt_tokens: req.prompt_token_ids.len(),
            generated_tokens: generated_token_ids.len(),
            generated_token_ids,
        })
    }

    #[cfg(not(feature = "tpu"))]
    pub fn generate_token_ids(&self, req: &M2GenerateRequest) -> Result<M2GenerateOutput> {
        let prepared = prepare_generate_request(&self.plan, req)?;
        let _ = (
            prepared.token_ids.len(),
            prepared.positions.len(),
            prepared.max_tokens,
        );
        Err(runtime_not_executable(&self.state))
    }
}

pub fn m2_decode_execution_blocker(plan: &M2RustPrefillDecodePlan) -> Option<&'static str> {
    m2_decode_mlir_execution_blocker(&plan.decode_mlir)
}

pub fn m2_decode_mlir_execution_blocker(mlir: &str) -> Option<&'static str> {
    if mlir.contains("rvllm.kind = \"m2_decode_graph\"")
        && mlir.contains("rvllm.lowering = \"rust_xla_custom_call\"")
    {
        Some("decode MLIR is a Rust+XLA artifact, but M2 fused decode executable custom-call bodies are not linked yet; execution has no Python/JAX fallback")
    } else {
        None
    }
}

pub fn plan_m2_rust_prefill(cfg: &M2RustPrefillConfig) -> Result<M2RustPrefillPlan> {
    cfg.validate()?;
    let index_path = cfg.model_dir.join("model.safetensors.index.json");
    let index = M2CheckpointIndex::from_index_file(index_path)?;
    let checkpoint = index.validate_m2(62, 256)?;

    let max_blocks_per_seq = max_blocks_per_seq(cfg.ctx, cfg.block_size)?;
    let prompts = synthetic_prompts(cfg.batch, cfg.prompt_len)?;
    let requests = prompts
        .iter()
        .enumerate()
        .map(|(i, prompt_tokens)| PrefillRequest {
            req_id: ReqId((i + 1) as u64),
            prompt_tokens,
            max_blocks_per_seq,
            block_size: cfg.block_size,
        })
        .collect::<Vec<_>>();
    let plan = BatchedPrefillPlan::from_requests(&requests)?;
    let shape = M2PrefillScanShape {
        batch: cfg.batch,
        prompt_len: cfg.prompt_len,
        hidden: 3072,
        ctx: cfg.ctx,
        num_layers: 62,
        num_kv_heads: 8,
        head_dim: 128,
        kv_dtype: cfg.kv_dtype,
    };
    let input_specs = make_m2_prefill_input_specs(&plan, shape)?;
    Ok(M2RustPrefillPlan {
        checkpoint,
        shape,
        plan,
        input_specs,
    })
}

pub fn plan_m2_rust_prefill_decode(
    cfg: &M2RustPrefillDecodeConfig,
) -> Result<M2RustPrefillDecodePlan> {
    cfg.validate()?;
    let prefill = plan_m2_rust_prefill(&M2RustPrefillConfig {
        model_dir: cfg.model_dir.clone(),
        batch: cfg.batch,
        prompt_len: cfg.prompt_len,
        ctx: cfg.ctx,
        block_size: cfg.block_size,
        kv_dtype: cfg.kv_dtype,
    })?;
    let kv_bytes_per_elem = match cfg.kv_dtype {
        M2PrefillKvDType::Int8 => 1,
        M2PrefillKvDType::Bf16 => 2,
    };
    let decode_shape = M2GraphShape::decode(cfg.batch, cfg.ctx, kv_bytes_per_elem);
    let abi = M2GraphAbi::new(decode_shape.clone())?;
    let weights = M2WeightUploadPlan::from_index_dir(&cfg.model_dir, &abi)?;
    let arena = weights.flat_arena(128)?;
    let decode_mlir = m2_decode_graph_mlir("main", &decode_shape, &arena)?;
    let decode_input_specs = decode_input_specs(&decode_shape, arena.total_bytes);
    let decode_output_specs = decode_output_specs(&decode_shape);
    Ok(M2RustPrefillDecodePlan {
        seed_decode_token_ids: seed_decode_token_ids(&prefill.plan, cfg.batch, cfg.prompt_len)?,
        seed_decode_positions: vec![cfg.prompt_len as i32; cfg.batch],
        prefill,
        decode_shape,
        decode_input_specs,
        decode_output_specs,
        decode_steps: cfg.decode_steps,
        weight_arena_bytes: arena.total_bytes,
        weight_entries: arena.entries.len(),
        decode_mlir,
    })
}

pub fn m2_bf16_logits_nll(
    logits_bf16: &[u8],
    target_ids: &[i32],
    batch: usize,
    vocab: usize,
) -> Result<Vec<f64>> {
    if target_ids.len() != batch {
        return Err(invalid("target_ids", "must have one target per batch row"));
    }
    if logits_bf16.len() != batch * vocab * 2 {
        return Err(invalid("logits", "bf16 byte length must be batch*vocab*2"));
    }
    let mut out = Vec::with_capacity(batch);
    for row in 0..batch {
        let target = target_ids[row];
        if target < 0 || target as usize >= vocab {
            return Err(invalid("target_ids", "target id outside vocab"));
        }
        let row_bytes = &logits_bf16[row * vocab * 2..(row + 1) * vocab * 2];
        let mut max_logit = f32::NEG_INFINITY;
        for col in 0..vocab {
            max_logit = max_logit.max(read_bf16(row_bytes, col));
        }
        let mut exp_sum = 0.0f64;
        for col in 0..vocab {
            exp_sum += ((read_bf16(row_bytes, col) - max_logit) as f64).exp();
        }
        let target_logit = read_bf16(row_bytes, target as usize);
        out.push((max_logit as f64 + exp_sum.ln()) - target_logit as f64);
    }
    Ok(out)
}

pub fn m2_ppl_from_nll(nll: &[f64]) -> Result<M2PplResult> {
    if nll.is_empty() {
        return Err(invalid("nll", "must not be empty"));
    }
    let avg_nll = nll.iter().sum::<f64>() / nll.len() as f64;
    Ok(M2PplResult {
        n_tokens_scored: nll.len(),
        avg_nll,
        ppl: avg_nll.exp(),
    })
}

struct M2PreparedGenerate {
    token_ids: Vec<i32>,
    positions: Vec<i32>,
    max_tokens: usize,
}

fn prepare_generate_request(
    plan: &M2RustPrefillDecodePlan,
    req: &M2GenerateRequest,
) -> Result<M2PreparedGenerate> {
    if req.prompt_token_ids.is_empty() {
        return Err(invalid("prompt_token_ids", "must not be empty"));
    }
    if req.max_tokens == 0 {
        return Err(invalid("max_tokens", "must be > 0"));
    }
    if req.max_tokens > plan.decode_steps {
        return Err(invalid("max_tokens", "exceeds compiled decode_steps"));
    }
    if req.prompt_token_ids.len() + req.max_tokens > plan.decode_shape.ctx {
        return Err(invalid(
            "max_tokens",
            "prompt_token_ids + max_tokens exceeds ctx",
        ));
    }
    for token in &req.prompt_token_ids {
        if *token < 0 || *token as usize >= M2_VOCAB {
            return Err(invalid("prompt_token_ids", "token id outside M2 vocab"));
        }
    }
    let last = *req.prompt_token_ids.last().unwrap_or(&0);
    Ok(M2PreparedGenerate {
        token_ids: vec![last; plan.decode_shape.batch],
        positions: vec![req.prompt_token_ids.len() as i32; plan.decode_shape.batch],
        max_tokens: req.max_tokens,
    })
}

impl M2RustPrefillConfig {
    fn validate(&self) -> Result<()> {
        if self.batch == 0 {
            return Err(invalid("batch", "must be > 0"));
        }
        if self.prompt_len == 0 {
            return Err(invalid("prompt_len", "must be > 0"));
        }
        if self.ctx == 0 {
            return Err(invalid("ctx", "must be > 0"));
        }
        if self.prompt_len > self.ctx {
            return Err(invalid("prompt_len", "must be <= ctx"));
        }
        if self.block_size == 0 {
            return Err(invalid("block_size", "must be > 0"));
        }
        Ok(())
    }
}

impl M2RustPrefillDecodeConfig {
    fn validate(&self) -> Result<()> {
        if self.decode_steps == 0 {
            return Err(invalid("decode_steps", "must be > 0"));
        }
        M2RustPrefillConfig {
            model_dir: self.model_dir.clone(),
            batch: self.batch,
            prompt_len: self.prompt_len,
            ctx: self.ctx,
            block_size: self.block_size,
            kv_dtype: self.kv_dtype,
        }
        .validate()
    }
}

fn max_blocks_per_seq(ctx: usize, block_size: u32) -> Result<u32> {
    let block_size = block_size as usize;
    let blocks = (ctx + block_size - 1) / block_size;
    u32::try_from(blocks).map_err(|_| invalid("ctx", "too large for u32 block count"))
}

fn synthetic_prompts(batch: usize, prompt_len: usize) -> Result<Vec<Vec<TokenId>>> {
    if prompt_len > u32::MAX as usize - 1024 {
        return Err(invalid("prompt_len", "too large for u32 token ids"));
    }
    let mut prompts = Vec::with_capacity(batch);
    for seq in 0..batch {
        let base = 1024u32 + (seq as u32) * 17;
        let prompt = (0..prompt_len)
            .map(|i| TokenId(base + i as u32))
            .collect::<Vec<_>>();
        prompts.push(prompt);
    }
    Ok(prompts)
}

fn seed_decode_token_ids(
    plan: &BatchedPrefillPlan,
    batch: usize,
    prompt_len: usize,
) -> Result<Vec<i32>> {
    if plan.prompt_tokens_flat.len() != batch * prompt_len {
        return Err(invalid("prompt_tokens", "flat prompt length mismatch"));
    }
    Ok(plan
        .prompt_tokens_flat
        .chunks_exact(prompt_len)
        .map(|seq| seq[prompt_len - 1].raw() as i32)
        .collect())
}

fn decode_input_specs(
    shape: &M2GraphShape,
    weight_arena_bytes: usize,
) -> Vec<M2DecodeRuntimeInputSpec> {
    vec![
        decode_spec("token_ids", &[shape.batch], PjrtElementType::S32),
        decode_spec("positions", &[shape.batch], PjrtElementType::S32),
        M2DecodeRuntimeInputSpec {
            name: "kv_cache",
            shape: vec![shape.kv_cache_bytes() as i64],
            dtype: PjrtElementType::S8,
            nbytes: shape.kv_cache_bytes(),
        },
        M2DecodeRuntimeInputSpec {
            name: "weight_arena",
            shape: vec![weight_arena_bytes as i64],
            dtype: PjrtElementType::S8,
            nbytes: weight_arena_bytes,
        },
    ]
}

fn decode_output_specs(shape: &M2GraphShape) -> Vec<M2DecodeRuntimeInputSpec> {
    vec![
        decode_spec("logits", &[shape.batch, 200_064], PjrtElementType::BF16),
        decode_spec("next_token", &[shape.batch], PjrtElementType::S32),
        M2DecodeRuntimeInputSpec {
            name: "kv_cache",
            shape: vec![shape.kv_cache_bytes() as i64],
            dtype: PjrtElementType::S8,
            nbytes: shape.kv_cache_bytes(),
        },
    ]
}

fn decode_spec(
    name: &'static str,
    shape: &[usize],
    dtype: PjrtElementType,
) -> M2DecodeRuntimeInputSpec {
    let elems = shape.iter().product::<usize>();
    M2DecodeRuntimeInputSpec {
        name,
        shape: shape.iter().map(|&x| x as i64).collect(),
        dtype,
        nbytes: elems * element_size(dtype),
    }
}

fn element_size(dtype: PjrtElementType) -> usize {
    match dtype {
        PjrtElementType::PRED | PjrtElementType::S8 | PjrtElementType::U8 => 1,
        PjrtElementType::S16
        | PjrtElementType::U16
        | PjrtElementType::F16
        | PjrtElementType::BF16 => 2,
        PjrtElementType::S32 | PjrtElementType::U32 | PjrtElementType::F32 => 4,
        PjrtElementType::S64 | PjrtElementType::U64 | PjrtElementType::F64 => 8,
        PjrtElementType::C64 => 8,
        PjrtElementType::C128 => 16,
        PjrtElementType::F8E5M2 | PjrtElementType::F8E4M3FN => 1,
        PjrtElementType::INVALID => 0,
    }
}

fn read_bf16(bytes: &[u8], idx: usize) -> f32 {
    let lo = bytes[idx * 2];
    let hi = bytes[idx * 2 + 1];
    f32::from_bits((u16::from_le_bytes([lo, hi]) as u32) << 16)
}

#[cfg(feature = "tpu")]
fn i32_bytes(vals: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for val in vals {
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

#[cfg(feature = "tpu")]
fn read_i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

#[cfg(feature = "tpu")]
fn ensure_decode_executable(plan: &M2RustPrefillDecodePlan) -> Result<()> {
    if let Some(reason) = m2_decode_execution_blocker(plan) {
        return Err(invalid_owned("runtime_mode", reason.to_string()));
    }
    Ok(())
}

fn runtime_not_executable(state: &M2RuntimeState) -> RvllmError {
    invalid_owned(
        "runtime_state",
        format!(
            "M2Runtime state is {}; generation requires executable Rust+XLA state and has no Python/JAX fallback",
            state.as_str()
        ),
    )
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    invalid_owned(field, reason.to_string())
}

fn invalid_owned(field: &'static str, reason: String) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason,
        },
        "m2_rust_prefill",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_b8_prefill_from_checked_in_m2_schema_without_python() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema");
        let cfg = M2RustPrefillConfig {
            model_dir,
            batch: 8,
            prompt_len: 4,
            ctx: 8,
            block_size: 8,
            kv_dtype: M2PrefillKvDType::Int8,
        };
        let plan = plan_m2_rust_prefill(&cfg).unwrap();
        assert_eq!(plan.checkpoint.total_tensors, 191_069);
        assert_eq!(plan.checkpoint.nvfp4_groups, 47_616);
        assert_eq!(plan.shape.batch, 8);
        assert_eq!(plan.shape.total_tokens(), 32);
        assert_eq!(
            plan.plan.cu_seqlens_q,
            vec![0, 4, 8, 12, 16, 20, 24, 28, 32]
        );
        assert_eq!(plan.plan.context_lens, vec![4; 8]);
        assert_eq!(plan.input_specs[0].name, "token_ids");
        assert_eq!(plan.input_specs[0].shape, vec![8, 4]);
        assert_eq!(plan.input_specs[5].name, "kv_cache");
        assert_eq!(plan.input_specs[5].nbytes, plan.shape.kv_cache_bytes());
    }

    #[test]
    fn rejects_prompt_longer_than_context() {
        let cfg = M2RustPrefillConfig {
            model_dir: PathBuf::from("unused"),
            batch: 8,
            prompt_len: 9,
            ctx: 8,
            block_size: 8,
            kv_dtype: M2PrefillKvDType::Int8,
        };
        assert!(plan_m2_rust_prefill(&cfg).is_err());
    }

    #[test]
    fn plans_b8_prefill_decode_sequence_without_python() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema");
        let plan = plan_m2_rust_prefill_decode(&M2RustPrefillDecodeConfig {
            model_dir,
            batch: 8,
            prompt_len: 4,
            decode_steps: 32,
            ctx: 2048,
            block_size: 32,
            kv_dtype: M2PrefillKvDType::Int8,
        })
        .unwrap();
        assert_eq!(plan.prefill.shape.total_tokens(), 32);
        assert_eq!(plan.decode_shape.batch, 8);
        assert_eq!(plan.decode_shape.kv_cache_bytes(), 2_080_374_784);
        assert_eq!(plan.decode_steps, 32);
        assert_eq!(plan.weight_entries, 191_069);
        assert_eq!(
            plan.seed_decode_token_ids,
            vec![1027, 1044, 1061, 1078, 1095, 1112, 1129, 1146]
        );
        assert_eq!(plan.seed_decode_positions, vec![4; 8]);
        assert_eq!(plan.decode_input_specs[3].name, "weight_arena");
        assert_eq!(plan.decode_input_specs[3].nbytes, plan.weight_arena_bytes);
        assert!(plan
            .decode_mlir
            .contains("target=rvllm.m2.decode_layer.fused_attention_nvfp4_moe"));
    }

    #[test]
    fn runtime_plan_only_state_is_explicitly_non_executable() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema");
        let plan = plan_m2_rust_prefill_decode(&M2RustPrefillDecodeConfig {
            model_dir: model_dir.clone(),
            batch: 8,
            prompt_len: 4,
            decode_steps: 32,
            ctx: 2048,
            block_size: 32,
            kv_dtype: M2PrefillKvDType::Int8,
        })
        .unwrap();
        let runtime =
            M2Runtime::from_plan_with_mode(model_dir, plan, 0, 0, M2RuntimeMode::PlanningOnly)
                .unwrap();
        assert_eq!(runtime.state(), &M2RuntimeState::PlanningOnly);
        assert!(!runtime.is_executable());
    }

    #[test]
    fn decode_execution_blocker_reports_unlinked_custom_calls() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema");
        let plan = plan_m2_rust_prefill_decode(&M2RustPrefillDecodeConfig {
            model_dir,
            batch: 8,
            prompt_len: 4,
            decode_steps: 32,
            ctx: 2048,
            block_size: 32,
            kv_dtype: M2PrefillKvDType::Int8,
        })
        .unwrap();
        let reason = m2_decode_execution_blocker(&plan).unwrap();
        assert!(reason.contains("custom-call bodies"));
        assert!(reason.contains("no Python/JAX fallback"));
    }

    #[test]
    fn parses_runtime_modes() {
        assert_eq!(
            "planning-only".parse::<M2RuntimeMode>().unwrap(),
            M2RuntimeMode::PlanningOnly
        );
        assert_eq!(
            "compile".parse::<M2RuntimeMode>().unwrap(),
            M2RuntimeMode::CompileOnly
        );
        assert_eq!(
            "execute".parse::<M2RuntimeMode>().unwrap(),
            M2RuntimeMode::Execute
        );
    }

    #[test]
    fn scores_bf16_logits_for_ppl_without_python() {
        fn bf16(v: f32, out: &mut Vec<u8>) {
            let bits = (v.to_bits() >> 16) as u16;
            out.extend_from_slice(&bits.to_le_bytes());
        }

        let mut logits = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 0.0, -1.0] {
            bf16(v, &mut logits);
        }
        let nll = m2_bf16_logits_nll(&logits, &[2, 0], 2, 3).unwrap();
        assert_eq!(nll.len(), 2);
        assert!(nll[0] < 0.5);
        assert!(nll[1] < 0.5);
        let ppl = m2_ppl_from_nll(&nll).unwrap();
        assert_eq!(ppl.n_tokens_scored, 2);
        assert!(ppl.ppl > 1.0);
    }

    #[test]
    fn prepares_server_generation_batch_from_real_prompt_tokens() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema");
        let plan = plan_m2_rust_prefill_decode(&M2RustPrefillDecodeConfig {
            model_dir,
            batch: 8,
            prompt_len: 4,
            decode_steps: 32,
            ctx: 2048,
            block_size: 32,
            kv_dtype: M2PrefillKvDType::Int8,
        })
        .unwrap();
        let prepared = prepare_generate_request(
            &plan,
            &M2GenerateRequest {
                prompt_token_ids: vec![1, 2, 3, 4],
                max_tokens: 16,
            },
        )
        .unwrap();
        assert_eq!(prepared.token_ids, vec![4; 8]);
        assert_eq!(prepared.positions, vec![4; 8]);
        assert_eq!(prepared.max_tokens, 16);
    }
}
