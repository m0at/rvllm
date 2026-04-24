use std::backtrace::Backtrace;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use half::bf16;
use memmap2::Mmap;
use rayon::prelude::*;
use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};

#[derive(Clone, Debug)]
pub struct K2RopeScaling {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position_embeddings: usize,
    pub mscale_all_dim: f32,
}

#[derive(Clone, Debug)]
pub struct K2Arch {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub n_experts_per_tok: usize,
    pub n_routed_experts: usize,
    pub num_attention_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope_theta: f32,
    pub rope_scaling: K2RopeScaling,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub first_k_dense: usize,
}

impl K2Arch {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let bytes = std::fs::read(&config_path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: config_path.clone(),
            source,
        })?;
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!("config.json: {e}"),
                },
                ctx: LoaderCtx {
                    path: config_path.clone(),
                    tensor: None,
                },
                bt: Backtrace::capture(),
            })?;
        let tc = json.get("text_config").unwrap_or(&json);
        let get_u = |name: &'static str| -> Result<usize> {
            tc[name]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| RvllmError::Loader {
                    err: LoaderError::Corrupt {
                        detail: format!("missing or invalid {name}"),
                    },
                    ctx: LoaderCtx {
                        path: config_path.clone(),
                        tensor: None,
                    },
                    bt: Backtrace::capture(),
                })
        };
        let rope_scaling = tc.get("rope_scaling").and_then(|v| v.as_object());
        Ok(Self {
            num_hidden_layers: get_u("num_hidden_layers")?,
            hidden_size: get_u("hidden_size")?,
            moe_intermediate_size: get_u("moe_intermediate_size")?,
            n_experts_per_tok: get_u("num_experts_per_tok")?,
            n_routed_experts: get_u("n_routed_experts")?,
            num_attention_heads: get_u("num_attention_heads")?,
            q_lora_rank: get_u("q_lora_rank")?,
            kv_lora_rank: get_u("kv_lora_rank")?,
            qk_nope_head_dim: get_u("qk_nope_head_dim")?,
            qk_rope_head_dim: get_u("qk_rope_head_dim")?,
            v_head_dim: get_u("v_head_dim")?,
            rope_theta: tc["rope_theta"].as_f64().map(|v| v as f32).ok_or_else(|| {
                RvllmError::Loader {
                    err: LoaderError::Corrupt {
                        detail: "missing or invalid rope_theta".into(),
                    },
                    ctx: LoaderCtx {
                        path: config_path.clone(),
                        tensor: None,
                    },
                    bt: Backtrace::capture(),
                }
            })?,
            rope_scaling: K2RopeScaling {
                factor: rope_scaling
                    .and_then(|o| o.get("factor"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32,
                beta_fast: rope_scaling
                    .and_then(|o| o.get("beta_fast"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(32.0) as f32,
                beta_slow: rope_scaling
                    .and_then(|o| o.get("beta_slow"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32,
                original_max_position_embeddings: rope_scaling
                    .and_then(|o| o.get("original_max_position_embeddings"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4096) as usize,
                mscale_all_dim: rope_scaling
                    .and_then(|o| o.get("mscale_all_dim"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32,
            },
            routed_scaling_factor: tc["routed_scaling_factor"]
                .as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| RvllmError::Loader {
                    err: LoaderError::Corrupt {
                        detail: "missing or invalid routed_scaling_factor".into(),
                    },
                    ctx: LoaderCtx {
                        path: config_path.clone(),
                        tensor: None,
                    },
                    bt: Backtrace::capture(),
                })?,
            rms_norm_eps: tc["rms_norm_eps"]
                .as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| RvllmError::Loader {
                    err: LoaderError::Corrupt {
                        detail: "missing or invalid rms_norm_eps".into(),
                    },
                    ctx: LoaderCtx {
                        path: config_path.clone(),
                        tensor: None,
                    },
                    bt: Backtrace::capture(),
                })?,
            first_k_dense: tc["first_k_dense_replace"].as_u64().unwrap_or(1) as usize,
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum K2DType {
    Bf16,
    F16,
    F32,
    I32,
    I64,
}

impl K2DType {
    fn elem_size(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::I32 => 4,
            Self::I64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
struct K2TensorEntry {
    dtype: K2DType,
    shape: Vec<usize>,
    file_offset: u64,
    nbytes: u64,
}

#[derive(Clone, Debug)]
struct K2ShardHeader {
    tensors: BTreeMap<String, K2TensorEntry>,
}

impl K2ShardHeader {
    fn parse(path: &Path, file_bytes: &[u8]) -> Result<Self> {
        let loader_err = |detail: String| -> RvllmError {
            RvllmError::Loader {
                err: LoaderError::Corrupt { detail },
                ctx: LoaderCtx {
                    path: path.to_path_buf(),
                    tensor: None,
                },
                bt: Backtrace::capture(),
            }
        };

        if file_bytes.len() < 8 {
            return Err(loader_err("shorter than 8-byte header prefix".into()));
        }
        let header_bytes = u64::from_le_bytes(
            file_bytes[..8]
                .try_into()
                .map_err(|_| loader_err("bad header prefix".into()))?,
        ) as usize;
        let payload_start = 8u64 + header_bytes as u64;
        if payload_start as usize > file_bytes.len() {
            return Err(loader_err(format!(
                "header claims {header_bytes} bytes but file is only {}",
                file_bytes.len()
            )));
        }
        let header_str = std::str::from_utf8(&file_bytes[8..8 + header_bytes])
            .map_err(|_| loader_err("header is not valid utf-8".into()))?;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(header_str)
                .map_err(|e| loader_err(format!("header json: {e}")))?;

        let mut tensors = BTreeMap::new();
        for (name, meta) in header {
            if name == "__metadata__" {
                continue;
            }
            let obj = meta
                .as_object()
                .ok_or_else(|| loader_err(format!("{name}: meta not an object")))?;
            let dtype = match obj.get("dtype").and_then(|v| v.as_str()) {
                Some("BF16") => K2DType::Bf16,
                Some("F16") => K2DType::F16,
                Some("F32") => K2DType::F32,
                Some("I32") => K2DType::I32,
                Some("I64") => K2DType::I64,
                Some(other) => {
                    if name.contains(".weight_") {
                        return Err(loader_err(format!(
                            "{name}: unsupported dtype {other} for K2 expert path"
                        )));
                    }
                    continue;
                }
                None => return Err(loader_err(format!("{name}: missing dtype"))),
            };
            let shape: Vec<usize> = obj
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| loader_err(format!("{name}: missing shape")))?
                .iter()
                .map(|v| v.as_u64().map(|n| n as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| loader_err(format!("{name}: bad shape element")))?;
            let offsets = obj
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| loader_err(format!("{name}: missing data_offsets")))?;
            if offsets.len() != 2 {
                return Err(loader_err(format!(
                    "{name}: expected 2 offsets got {}",
                    offsets.len()
                )));
            }
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| loader_err(format!("{name}: bad start offset")))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| loader_err(format!("{name}: bad end offset")))?;
            let nbytes = end
                .checked_sub(start)
                .ok_or_else(|| loader_err(format!("{name}: negative offset range")))?;
            let expected = dtype.elem_size() as u64 * shape.iter().product::<usize>() as u64;
            if expected != nbytes {
                return Err(loader_err(format!(
                    "{name}: offset range {nbytes} != dtype*shape {expected}"
                )));
            }
            tensors.insert(
                name,
                K2TensorEntry {
                    dtype,
                    shape,
                    file_offset: payload_start + start,
                    nbytes,
                },
            );
        }
        Ok(Self { tensors })
    }
}

struct K2ShardMap {
    path: PathBuf,
    _mmap: Mmap,
    header: K2ShardHeader,
}

impl K2ShardMap {
    fn open(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let mmap = unsafe { Mmap::map(&f) }.map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let header = K2ShardHeader::parse(path, &mmap)?;
        Ok(Self {
            path: path.to_path_buf(),
            _mmap: mmap,
            header,
        })
    }

    fn bytes(&self) -> &[u8] {
        &self._mmap
    }
}

#[derive(Clone)]
struct ExpertMatrix {
    rows: usize,
    cols: usize,
    row_major: Arc<Vec<f32>>,
}

#[derive(Clone)]
struct ExpertWeights {
    gate: ExpertMatrix,
    up: ExpertMatrix,
    down: ExpertMatrix,
}

type ExpertKey = (usize, usize);
type TensorKey = String;

#[derive(Clone, Default)]
struct K2LayerCache {
    k_nope: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    k_rope: Vec<Vec<f32>>,
}

pub struct K2DecodeCache {
    layers: Vec<K2LayerCache>,
}

pub struct K2CpuExpertStore {
    pub arch: K2Arch,
    model_dir: PathBuf,
    shards: Vec<K2ShardMap>,
    tensors: BTreeMap<String, (usize, K2TensorEntry)>,
    cache: Mutex<HashMap<ExpertKey, Arc<ExpertWeights>>>,
    bf16_matrix_cache: Mutex<HashMap<TensorKey, Arc<ExpertMatrix>>>,
    bf16_vector_cache: Mutex<HashMap<TensorKey, Arc<Vec<f32>>>>,
    f32_vector_cache: Mutex<HashMap<TensorKey, Arc<Vec<f32>>>>,
    rope_inv_freq: Arc<Vec<f32>>,
    rope_attn_mscale: f32,
}

impl K2CpuExpertStore {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let arch = K2Arch::from_dir(model_dir)?;
        let index = read_weight_map(model_dir)?;
        let mut shard_paths: BTreeMap<PathBuf, usize> = BTreeMap::new();
        for shard in index.values() {
            let next = shard_paths.len();
            shard_paths.entry(shard.clone()).or_insert(next);
        }
        let mut ordered = vec![PathBuf::new(); shard_paths.len()];
        for (path, idx) in &shard_paths {
            ordered[*idx] = path.clone();
        }
        let shards: Vec<K2ShardMap> = ordered
            .iter()
            .map(|path| K2ShardMap::open(path))
            .collect::<Result<Vec<_>>>()?;
        let mut tensors = BTreeMap::new();
        for (name, path) in index {
            let si = *shard_paths.get(&path).ok_or_else(|| RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!("index references unknown shard {}", path.display()),
                },
                ctx: LoaderCtx {
                    path: model_dir.to_path_buf(),
                    tensor: Some(name.clone()),
                },
                bt: Backtrace::capture(),
            })?;
            if let Some(entry) = shards[si].header.tensors.get(&name).cloned() {
                tensors.insert(name, (si, entry));
            }
        }
        Ok(Self {
            rope_inv_freq: Arc::new(build_rope_inv_freq(
                arch.qk_rope_head_dim,
                arch.rope_theta,
                &arch.rope_scaling,
            )),
            rope_attn_mscale: yarn_get_mscale(
                arch.rope_scaling.factor,
                arch.rope_scaling.mscale_all_dim,
            ),
            arch,
            model_dir: model_dir.to_path_buf(),
            shards,
            tensors,
            cache: Mutex::new(HashMap::new()),
            bf16_matrix_cache: Mutex::new(HashMap::new()),
            bf16_vector_cache: Mutex::new(HashMap::new()),
            f32_vector_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn new_decode_cache(&self) -> K2DecodeCache {
        K2DecodeCache {
            layers: vec![K2LayerCache::default(); self.arch.num_hidden_layers],
        }
    }

    pub fn preload_hot_matrices(&self) -> Result<usize> {
        let mut names = Vec::new();
        for layer_idx in 0..self.arch.num_hidden_layers {
            let lp = format!("language_model.model.layers.{layer_idx}");
            names.push(format!("{lp}.self_attn.q_a_proj.weight"));
            names.push(format!("{lp}.self_attn.q_b_proj.weight"));
            names.push(format!("{lp}.self_attn.kv_a_proj_with_mqa.weight"));
            names.push(format!("{lp}.self_attn.kv_b_proj.weight"));
            names.push(format!("{lp}.self_attn.o_proj.weight"));

            if layer_idx < self.arch.first_k_dense {
                names.push(format!("{lp}.mlp.gate_proj.weight"));
                names.push(format!("{lp}.mlp.up_proj.weight"));
                names.push(format!("{lp}.mlp.down_proj.weight"));
            } else {
                names.push(format!("{lp}.mlp.gate.weight"));
                names.push(format!("{lp}.mlp.shared_experts.gate_proj.weight"));
                names.push(format!("{lp}.mlp.shared_experts.up_proj.weight"));
                names.push(format!("{lp}.mlp.shared_experts.down_proj.weight"));
            }
        }
        for name in &names {
            let _ = self.load_bf16_matrix(name)?;
        }
        Ok(names.len())
    }

    pub fn run_expert(&self, layer_idx: usize, expert_idx: usize, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.arch.hidden_size {
            return Err(RvllmError::Loader {
                err: LoaderError::ShapeMismatch {
                    tensor: format!("expert_input_l{layer_idx}_e{expert_idx}"),
                    expected: vec![self.arch.hidden_size],
                    got: vec![x.len()],
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: None,
                },
                bt: Backtrace::capture(),
            });
        }
        let expert = self.load_expert(layer_idx, expert_idx)?;
        Ok(run_expert_mlp(&expert, x))
    }

    pub fn run_routed_topk(
        &self,
        layer_idx: usize,
        expert_ids: &[usize],
        expert_weights: &[f32],
        x: &[f32],
    ) -> Result<Vec<f32>> {
        if expert_ids.len() != expert_weights.len() {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "expert_ids len {} != expert_weights len {}",
                        expert_ids.len(),
                        expert_weights.len()
                    ),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: None,
                },
                bt: Backtrace::capture(),
            });
        }
        let outs: Vec<(f32, Vec<f32>)> = expert_ids
            .par_iter()
            .zip(expert_weights.par_iter())
            .map(|(&expert_idx, &w)| {
                let out = self.run_expert(layer_idx, expert_idx, x)?;
                Ok::<_, RvllmError>((w, out))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut acc = vec![0.0f32; self.arch.hidden_size];
        for (w, out) in outs {
            for (dst, src) in acc.iter_mut().zip(out) {
                *dst += w * src;
            }
        }
        Ok(acc)
    }

    pub fn run_moe_block(&self, layer_idx: usize, x: &[f32]) -> Result<Vec<f32>> {
        if layer_idx < self.arch.first_k_dense {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!("layer {layer_idx} is dense, not MoE"),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: None,
                },
                bt: Backtrace::capture(),
            });
        }
        let norm_gamma = self.load_bf16_vector(&format!(
            "language_model.model.layers.{layer_idx}.post_attention_layernorm.weight"
        ))?;
        let h = rms_norm(x, &norm_gamma, self.arch.rms_norm_eps);
        self.run_moe_from_normed(layer_idx, &h)
    }

    pub fn run_token0_hidden(&self, token_id: usize) -> Result<Vec<f32>> {
        let mut cache = self.new_decode_cache();
        self.forward_hidden_cached(token_id, 0, &mut cache)
    }

    pub fn run_token0_logits(&self, token_id: usize) -> Result<Vec<f32>> {
        let mut cache = self.new_decode_cache();
        self.forward_step_cached(token_id, 0, &mut cache)
    }

    pub fn forward_step_cached(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut K2DecodeCache,
    ) -> Result<Vec<f32>> {
        let h = self.forward_hidden_cached(token_id, pos, cache)?;
        self.gemv_bf16_tensor("language_model.lm_head.weight", &h)
    }

    fn forward_hidden_cached(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut K2DecodeCache,
    ) -> Result<Vec<f32>> {
        let mut x = self.lookup_embedding(token_id)?;
        for layer_idx in 0..self.arch.num_hidden_layers {
            let attn_norm = self.load_bf16_vector(&format!(
                "language_model.model.layers.{layer_idx}.input_layernorm.weight"
            ))?;
            let h_attn = rms_norm(&x, &attn_norm, self.arch.rms_norm_eps);
            let attn_out = self.run_attention_cached(layer_idx, &h_attn, pos, cache)?;
            for (dst, add) in x.iter_mut().zip(attn_out) {
                *dst += add;
            }

            let mlp_norm = self.load_bf16_vector(&format!(
                "language_model.model.layers.{layer_idx}.post_attention_layernorm.weight"
            ))?;
            let h_mlp = rms_norm(&x, &mlp_norm, self.arch.rms_norm_eps);
            let mlp_out = if layer_idx < self.arch.first_k_dense {
                self.run_dense_from_normed(layer_idx, &h_mlp)?
            } else {
                self.run_moe_from_normed(layer_idx, &h_mlp)?
            };
            for (dst, add) in x.iter_mut().zip(mlp_out) {
                *dst += add;
            }
        }
        let final_norm = self.load_bf16_vector("language_model.model.norm.weight")?;
        Ok(rms_norm(&x, &final_norm, self.arch.rms_norm_eps))
    }

    fn run_moe_from_normed(&self, layer_idx: usize, h: &[f32]) -> Result<Vec<f32>> {
        let router_w = self.load_bf16_matrix(&format!(
            "language_model.model.layers.{layer_idx}.mlp.gate.weight"
        ))?;
        let router_bias = self.load_f32_vector(&format!(
            "language_model.model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        ))?;
        let logits = gemv_row_major(&router_w, h);
        let scores: Vec<f32> = logits
            .iter()
            .zip(router_bias.iter())
            .map(|(l, b)| sigmoid(*l) + *b)
            .collect();
        let (expert_ids, expert_weights) = topk_normalized(
            &scores,
            self.arch.n_experts_per_tok,
            self.arch.routed_scaling_factor,
        );

        let shared = self.run_shared_expert(layer_idx, h)?;
        let routed = self.run_routed_topk(layer_idx, &expert_ids, &expert_weights, h)?;

        Ok(shared.into_iter().zip(routed).map(|(s, r)| s + r).collect())
    }

    fn run_attention_token0(&self, layer_idx: usize, h: &[f32]) -> Result<Vec<f32>> {
        let kv_a = self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight"),
            h,
        )?;
        let mut c_kv = kv_a[..self.arch.kv_lora_rank].to_vec();
        let kv_norm = self.load_bf16_vector(&format!(
            "language_model.model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight"
        ))?;
        c_kv = rms_norm(&c_kv, &kv_norm, self.arch.rms_norm_eps);
        let kv_full = self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.self_attn.kv_b_proj.weight"),
            &c_kv,
        )?;
        let per_head = self.arch.qk_nope_head_dim + self.arch.v_head_dim;
        let mut vcat = Vec::with_capacity(self.arch.num_attention_heads * self.arch.v_head_dim);
        for head in 0..self.arch.num_attention_heads {
            let start = head * per_head + self.arch.qk_nope_head_dim;
            let end = start + self.arch.v_head_dim;
            vcat.extend_from_slice(&kv_full[start..end]);
        }
        self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.self_attn.o_proj.weight"),
            &vcat,
        )
    }

    fn run_attention_cached(
        &self,
        layer_idx: usize,
        h: &[f32],
        pos: usize,
        cache: &mut K2DecodeCache,
    ) -> Result<Vec<f32>> {
        let layer_cache = cache
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!("missing decode cache for layer {layer_idx}"),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: None,
                },
                bt: Backtrace::capture(),
            })?;
        if layer_cache.k_nope.len() != pos
            || layer_cache.v.len() != pos
            || layer_cache.k_rope.len() != pos
        {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "decode cache length mismatch at layer {layer_idx}: expected {pos}, got k_nope={} v={} k_rope={}",
                        layer_cache.k_nope.len(),
                        layer_cache.v.len(),
                        layer_cache.k_rope.len(),
                    ),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: None,
                },
                bt: Backtrace::capture(),
            });
        }

        let q_a = self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.self_attn.q_a_proj.weight"),
            h,
        )?;
        let q_a_norm = rms_norm(
            &q_a,
            &self.load_bf16_vector(&format!(
                "language_model.model.layers.{layer_idx}.self_attn.q_a_layernorm.weight"
            ))?,
            self.arch.rms_norm_eps,
        );
        let q_full = self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.self_attn.q_b_proj.weight"),
            &q_a_norm,
        )?;
        let q_per_head = self.arch.qk_nope_head_dim + self.arch.qk_rope_head_dim;
        let expected_q = self.arch.num_attention_heads * q_per_head;
        if q_full.len() != expected_q {
            return Err(corrupt_err(
                &self.model_dir,
                format!(
                    "q_b_proj output len {} != expected {}",
                    q_full.len(),
                    expected_q
                ),
            ));
        }

        let kv_a = self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight"),
            h,
        )?;
        let expected_kv_a = self.arch.kv_lora_rank + self.arch.qk_rope_head_dim;
        if kv_a.len() != expected_kv_a {
            return Err(corrupt_err(
                &self.model_dir,
                format!(
                    "kv_a_proj output len {} != expected {}",
                    kv_a.len(),
                    expected_kv_a
                ),
            ));
        }
        let c_kv = rms_norm(
            &kv_a[..self.arch.kv_lora_rank],
            &self.load_bf16_vector(&format!(
                "language_model.model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight"
            ))?,
            self.arch.rms_norm_eps,
        );
        let k_rope_raw = &kv_a[self.arch.kv_lora_rank..];
        let kv_full = self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.self_attn.kv_b_proj.weight"),
            &c_kv,
        )?;
        let kv_per_head = self.arch.qk_nope_head_dim + self.arch.v_head_dim;
        let expected_kv = self.arch.num_attention_heads * kv_per_head;
        if kv_full.len() != expected_kv {
            return Err(corrupt_err(
                &self.model_dir,
                format!(
                    "kv_b_proj output len {} != expected {}",
                    kv_full.len(),
                    expected_kv
                ),
            ));
        }

        let mut q_nope = vec![0.0f32; self.arch.num_attention_heads * self.arch.qk_nope_head_dim];
        let mut q_rope = vec![0.0f32; self.arch.num_attention_heads * self.arch.qk_rope_head_dim];
        for head in 0..self.arch.num_attention_heads {
            let src = head * q_per_head;
            let qn_dst = head * self.arch.qk_nope_head_dim;
            let qr_dst = head * self.arch.qk_rope_head_dim;
            q_nope[qn_dst..qn_dst + self.arch.qk_nope_head_dim]
                .copy_from_slice(&q_full[src..src + self.arch.qk_nope_head_dim]);
            q_rope[qr_dst..qr_dst + self.arch.qk_rope_head_dim]
                .copy_from_slice(&q_full[src + self.arch.qk_nope_head_dim..src + q_per_head]);
            apply_rope_inplace(
                &mut q_rope[qr_dst..qr_dst + self.arch.qk_rope_head_dim],
                pos,
                &self.rope_inv_freq,
                self.rope_attn_mscale,
            );
        }

        let mut k_nope_cur =
            vec![0.0f32; self.arch.num_attention_heads * self.arch.qk_nope_head_dim];
        let mut v_cur = vec![0.0f32; self.arch.num_attention_heads * self.arch.v_head_dim];
        for head in 0..self.arch.num_attention_heads {
            let src = head * kv_per_head;
            let kn_dst = head * self.arch.qk_nope_head_dim;
            let v_dst = head * self.arch.v_head_dim;
            k_nope_cur[kn_dst..kn_dst + self.arch.qk_nope_head_dim]
                .copy_from_slice(&kv_full[src..src + self.arch.qk_nope_head_dim]);
            v_cur[v_dst..v_dst + self.arch.v_head_dim]
                .copy_from_slice(&kv_full[src + self.arch.qk_nope_head_dim..src + kv_per_head]);
        }
        let mut k_rope_cur = k_rope_raw.to_vec();
        apply_rope_inplace(
            &mut k_rope_cur,
            pos,
            &self.rope_inv_freq,
            self.rope_attn_mscale,
        );
        layer_cache.k_nope.push(k_nope_cur);
        layer_cache.v.push(v_cur);
        layer_cache.k_rope.push(k_rope_cur);

        let seq_len = layer_cache.k_nope.len();
        let scale = ((self.arch.qk_nope_head_dim + self.arch.qk_rope_head_dim) as f32).sqrt();
        let mut out = vec![0.0f32; self.arch.num_attention_heads * self.arch.v_head_dim];
        let mut scores = vec![0.0f32; seq_len];
        for head in 0..self.arch.num_attention_heads {
            let qn =
                &q_nope[head * self.arch.qk_nope_head_dim..(head + 1) * self.arch.qk_nope_head_dim];
            let qr =
                &q_rope[head * self.arch.qk_rope_head_dim..(head + 1) * self.arch.qk_rope_head_dim];
            for (tok, score) in scores.iter_mut().enumerate() {
                let kn = &layer_cache.k_nope[tok]
                    [head * self.arch.qk_nope_head_dim..(head + 1) * self.arch.qk_nope_head_dim];
                let kr = &layer_cache.k_rope[tok];
                *score = (dot(qn, kn) + dot(qr, kr)) / scale;
            }
            softmax_inplace(&mut scores);
            let out_head = &mut out[head * self.arch.v_head_dim..(head + 1) * self.arch.v_head_dim];
            out_head.fill(0.0);
            for (tok, weight) in scores.iter().copied().enumerate() {
                let vv = &layer_cache.v[tok]
                    [head * self.arch.v_head_dim..(head + 1) * self.arch.v_head_dim];
                for (dst, src) in out_head.iter_mut().zip(vv.iter()) {
                    *dst += weight * *src;
                }
            }
        }
        self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.self_attn.o_proj.weight"),
            &out,
        )
    }

    fn run_dense_from_normed(&self, layer_idx: usize, h: &[f32]) -> Result<Vec<f32>> {
        let gate = self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.mlp.gate_proj.weight"),
            h,
        )?;
        let up = self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.mlp.up_proj.weight"),
            h,
        )?;
        let fused: Vec<f32> = gate.into_iter().zip(up).map(|(g, u)| silu(g) * u).collect();
        self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.mlp.down_proj.weight"),
            &fused,
        )
    }

    fn load_expert(&self, layer_idx: usize, expert_idx: usize) -> Result<Arc<ExpertWeights>> {
        let key = (layer_idx, expert_idx);
        if let Some(found) = self
            .cache
            .lock()
            .map_err(|_| lock_err(&self.model_dir))?
            .get(&key)
        {
            return Ok(found.clone());
        }

        let gate = self.load_matrix(layer_idx, expert_idx, "gate_proj")?;
        let up = self.load_matrix(layer_idx, expert_idx, "up_proj")?;
        let down = self.load_matrix(layer_idx, expert_idx, "down_proj")?;
        let weights = Arc::new(ExpertWeights { gate, up, down });

        let mut cache = self.cache.lock().map_err(|_| lock_err(&self.model_dir))?;
        Ok(cache.entry(key).or_insert_with(|| weights.clone()).clone())
    }

    fn load_matrix(&self, layer_idx: usize, expert_idx: usize, proj: &str) -> Result<ExpertMatrix> {
        let prefix =
            format!("language_model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}");
        let packed = self.read_i32_tensor(&format!("{prefix}.weight_packed"))?;
        let scale = self.read_bf16_tensor(&format!("{prefix}.weight_scale"))?;
        let shape = self.read_shape_tensor(&format!("{prefix}.weight_shape"))?;
        if shape.len() != 2 {
            return Err(RvllmError::Loader {
                err: LoaderError::ShapeMismatch {
                    tensor: format!("{prefix}.weight_shape"),
                    expected: vec![2],
                    got: vec![shape.len()],
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: Some(format!("{prefix}.weight_shape")),
                },
                bt: Backtrace::capture(),
            });
        }
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;
        let row_major = Arc::new(dequant_int4_group32(&packed, &scale, rows, cols)?);
        Ok(ExpertMatrix {
            rows,
            cols,
            row_major,
        })
    }

    fn run_shared_expert(&self, layer_idx: usize, x: &[f32]) -> Result<Vec<f32>> {
        let gate_v = self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"),
            x,
        )?;
        let up_v = self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"),
            x,
        )?;
        let fused: Vec<f32> = gate_v
            .into_iter()
            .zip(up_v)
            .map(|(g, u)| silu(g) * u)
            .collect();
        self.gemv_bf16_tensor(
            &format!("language_model.model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"),
            &fused,
        )
    }

    fn load_bf16_matrix(&self, name: &str) -> Result<Arc<ExpertMatrix>> {
        if let Some(found) = self
            .bf16_matrix_cache
            .lock()
            .map_err(|_| lock_err(&self.model_dir))?
            .get(name)
        {
            return Ok(found.clone());
        }
        let (si, entry) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| RvllmError::Loader {
                err: LoaderError::MissingTensor {
                    name: name.to_string(),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            })?;
        if entry.dtype != K2DType::Bf16 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!("{name}: expected BF16 got {:?}", entry.dtype),
                },
                ctx: LoaderCtx {
                    path: self.shards[si].path.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            });
        }
        if entry.shape.len() != 2 {
            return Err(RvllmError::Loader {
                err: LoaderError::ShapeMismatch {
                    tensor: name.to_string(),
                    expected: vec![2],
                    got: vec![entry.shape.len()],
                },
                ctx: LoaderCtx {
                    path: self.shards[si].path.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            });
        }
        let data = self.read_bf16_tensor(name)?;
        let matrix = Arc::new(ExpertMatrix {
            rows: entry.shape[0],
            cols: entry.shape[1],
            row_major: Arc::new(data.into_iter().map(|v| v.to_f32()).collect()),
        });
        let mut cache = self
            .bf16_matrix_cache
            .lock()
            .map_err(|_| lock_err(&self.model_dir))?;
        Ok(cache
            .entry(name.to_string())
            .or_insert_with(|| matrix.clone())
            .clone())
    }

    fn load_bf16_vector(&self, name: &str) -> Result<Arc<Vec<f32>>> {
        if let Some(found) = self
            .bf16_vector_cache
            .lock()
            .map_err(|_| lock_err(&self.model_dir))?
            .get(name)
        {
            return Ok(found.clone());
        }
        let data: Arc<Vec<f32>> = Arc::new(
            self.read_bf16_tensor(name)?
                .into_iter()
                .map(|v| v.to_f32())
                .collect(),
        );
        let mut cache = self
            .bf16_vector_cache
            .lock()
            .map_err(|_| lock_err(&self.model_dir))?;
        Ok(cache
            .entry(name.to_string())
            .or_insert_with(|| data.clone())
            .clone())
    }

    fn load_f32_vector(&self, name: &str) -> Result<Arc<Vec<f32>>> {
        if let Some(found) = self
            .f32_vector_cache
            .lock()
            .map_err(|_| lock_err(&self.model_dir))?
            .get(name)
        {
            return Ok(found.clone());
        }
        let raw = self.tensor_bytes(name, K2DType::F32)?;
        let data: Arc<Vec<f32>> = Arc::new(
            raw.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        );
        let mut cache = self
            .f32_vector_cache
            .lock()
            .map_err(|_| lock_err(&self.model_dir))?;
        Ok(cache
            .entry(name.to_string())
            .or_insert_with(|| data.clone())
            .clone())
    }

    fn lookup_embedding(&self, token_id: usize) -> Result<Vec<f32>> {
        let name = "language_model.model.embed_tokens.weight";
        let (si, entry) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| RvllmError::Loader {
                err: LoaderError::MissingTensor {
                    name: name.to_string(),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            })?;
        if entry.dtype != K2DType::Bf16 || entry.shape.len() != 2 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "{name}: expected BF16 rank-2 got {:?} {:?}",
                        entry.dtype, entry.shape
                    ),
                },
                ctx: LoaderCtx {
                    path: self.shards[si].path.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            });
        }
        if token_id >= entry.shape[0] {
            return Err(RvllmError::Loader {
                err: LoaderError::ShapeMismatch {
                    tensor: format!("embed_row_{token_id}"),
                    expected: vec![entry.shape[0]],
                    got: vec![token_id],
                },
                ctx: LoaderCtx {
                    path: self.shards[si].path.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            });
        }
        let cols = entry.shape[1];
        let bytes = self.shards[si].bytes();
        let row_bytes = cols * 2;
        let start = entry.file_offset as usize + token_id * row_bytes;
        let end = start + row_bytes;
        Ok(bytes[start..end]
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32())
            .collect())
    }

    fn gemv_bf16_tensor(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        if let Some(found) = self
            .bf16_matrix_cache
            .lock()
            .map_err(|_| lock_err(&self.model_dir))?
            .get(name)
            .cloned()
        {
            if x.len() != found.cols {
                return Err(RvllmError::Loader {
                    err: LoaderError::ShapeMismatch {
                        tensor: name.to_string(),
                        expected: vec![found.cols],
                        got: vec![x.len()],
                    },
                    ctx: LoaderCtx {
                        path: self.model_dir.clone(),
                        tensor: Some(name.to_string()),
                    },
                    bt: Backtrace::capture(),
                });
            }
            return Ok(gemv_row_major(&found, x));
        }
        let (si, entry) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| RvllmError::Loader {
                err: LoaderError::MissingTensor {
                    name: name.to_string(),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            })?;
        if entry.dtype != K2DType::Bf16 || entry.shape.len() != 2 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "{name}: expected BF16 rank-2 got {:?} {:?}",
                        entry.dtype, entry.shape
                    ),
                },
                ctx: LoaderCtx {
                    path: self.shards[si].path.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            });
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        if x.len() != cols {
            return Err(RvllmError::Loader {
                err: LoaderError::ShapeMismatch {
                    tensor: name.to_string(),
                    expected: vec![cols],
                    got: vec![x.len()],
                },
                ctx: LoaderCtx {
                    path: self.shards[si].path.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            });
        }
        let bytes = self.shards[si].bytes();
        let start = entry.file_offset as usize;
        let end = start + entry.nbytes as usize;
        let raw = &bytes[start..end];
        let row_bytes = cols * 2;
        Ok((0..rows)
            .into_par_iter()
            .map(|row_idx| {
                let row = &raw[row_idx * row_bytes..(row_idx + 1) * row_bytes];
                row.chunks_exact(2)
                    .zip(x.iter())
                    .map(|(chunk, xv)| {
                        bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32()
                            * *xv
                    })
                    .sum()
            })
            .collect())
    }

    fn read_i32_tensor(&self, name: &str) -> Result<Vec<i32>> {
        let raw = self.tensor_bytes(name, K2DType::I32)?;
        Ok(raw
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn read_shape_tensor(&self, name: &str) -> Result<Vec<i64>> {
        let (si, entry) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| RvllmError::Loader {
                err: LoaderError::MissingTensor {
                    name: name.to_string(),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            })?;
        let bytes = self.shards[si].bytes();
        let start = entry.file_offset as usize;
        let end = start + entry.nbytes as usize;
        let raw = &bytes[start..end];
        match entry.dtype {
            K2DType::I32 => Ok(raw
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i64)
                .collect()),
            K2DType::I64 => Ok(raw
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect()),
            other => Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!("{name}: expected I32/I64 got {:?}", other),
                },
                ctx: LoaderCtx {
                    path: self.shards[si].path.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            }),
        }
    }

    fn read_bf16_tensor(&self, name: &str) -> Result<Vec<bf16>> {
        let raw = self.tensor_bytes(name, K2DType::Bf16)?;
        Ok(raw
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())))
            .collect())
    }

    fn tensor_bytes(&self, name: &str, expected: K2DType) -> Result<&[u8]> {
        let (si, entry) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| RvllmError::Loader {
                err: LoaderError::MissingTensor {
                    name: name.to_string(),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            })?;
        if entry.dtype != expected {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!("{name}: expected {:?} got {:?}", expected, entry.dtype),
                },
                ctx: LoaderCtx {
                    path: self.shards[si].path.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: Backtrace::capture(),
            });
        }
        let bytes = self.shards[si].bytes();
        let start = entry.file_offset as usize;
        let end = start + entry.nbytes as usize;
        Ok(&bytes[start..end])
    }
}

fn read_weight_map(model_dir: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let bytes = std::fs::read(&index_path).map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: index_path.clone(),
        source,
    })?;
    let obj: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!("index json: {e}"),
            },
            ctx: LoaderCtx {
                path: index_path.clone(),
                tensor: None,
            },
            bt: Backtrace::capture(),
        })?;
    let wmap = obj
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: "index missing weight_map".into(),
            },
            ctx: LoaderCtx {
                path: index_path.clone(),
                tensor: None,
            },
            bt: Backtrace::capture(),
        })?;
    let mut out = BTreeMap::new();
    for (name, shard) in wmap {
        let shard = shard.as_str().ok_or_else(|| RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!("{name}: shard value not string"),
            },
            ctx: LoaderCtx {
                path: index_path.clone(),
                tensor: Some(name.clone()),
            },
            bt: Backtrace::capture(),
        })?;
        out.insert(name.clone(), model_dir.join(shard));
    }
    Ok(out)
}

fn dequant_int4_group32(
    packed: &[i32],
    scales: &[bf16],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    if cols % 32 != 0 {
        return Err(corrupt_err(
            Path::new(""),
            format!("dequant expects cols multiple of 32, got {cols}"),
        ));
    }
    let groups = cols / 32;
    let expected_packed = rows
        .checked_mul(cols / 8)
        .ok_or_else(|| corrupt_err(Path::new(""), "packed size overflow".into()))?;
    let expected_scales = rows
        .checked_mul(groups)
        .ok_or_else(|| corrupt_err(Path::new(""), "scale size overflow".into()))?;
    if packed.len() != expected_packed {
        return Err(corrupt_err(
            Path::new(""),
            format!(
                "packed len {} != expected {}",
                packed.len(),
                expected_packed
            ),
        ));
    }
    if scales.len() != expected_scales {
        return Err(corrupt_err(
            Path::new(""),
            format!(
                "scales len {} != expected {}",
                scales.len(),
                expected_scales
            ),
        ));
    }

    let mut out = vec![0.0f32; rows * cols];
    out.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let packed_row = &packed[row_idx * (cols / 8)..(row_idx + 1) * (cols / 8)];
            let scale_row = &scales[row_idx * groups..(row_idx + 1) * groups];
            for group_idx in 0..groups {
                let scale = scale_row[group_idx].to_f32();
                for word_idx in 0..4 {
                    let word = packed_row[group_idx * 4 + word_idx] as u32;
                    let base_col = group_idx * 32 + word_idx * 8;
                    for nibble in 0..8 {
                        let q = ((word >> (nibble * 4)) & 0xF) as i32 - 8;
                        row[base_col + nibble] = q as f32 * scale;
                    }
                }
            }
        });
    Ok(out)
}

fn gemv_row_major(matrix: &ExpertMatrix, x: &[f32]) -> Vec<f32> {
    matrix
        .row_major
        .par_chunks(matrix.cols)
        .map(|row| row.iter().zip(x.iter()).map(|(w, xv)| w * xv).sum())
        .collect()
}

fn run_expert_mlp(expert: &ExpertWeights, x: &[f32]) -> Vec<f32> {
    let gate = gemv_row_major(&expert.gate, x);
    let up = gemv_row_major(&expert.up, x);
    let fused: Vec<f32> = gate.into_iter().zip(up).map(|(g, u)| silu(g) * u).collect();
    gemv_row_major(&expert.down, &fused)
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

fn build_rope_inv_freq(dim: usize, theta: f32, scaling: &K2RopeScaling) -> Vec<f32> {
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
        .collect();
    if scaling.factor > 1.0 {
        let log_theta = theta.ln();
        let low = (((dim as f32
            * ((scaling.original_max_position_embeddings as f32
                / (scaling.beta_fast * 2.0 * std::f32::consts::PI))
                .ln())
            / (2.0 * log_theta))
            .floor()) as isize)
            .max(0) as usize;
        let high = ((((dim as f32
            * ((scaling.original_max_position_embeddings as f32
                / (scaling.beta_slow * 2.0 * std::f32::consts::PI))
                .ln())
            / (2.0 * log_theta))
            .ceil()) as isize)
            .max(0) as usize)
            .min(half.saturating_sub(1));
        for (i, freq) in freqs.iter_mut().enumerate() {
            let smooth = if i < low {
                0.0
            } else if i > high {
                1.0
            } else if high > low {
                (i - low) as f32 / (high - low) as f32
            } else {
                1.0
            };
            *freq = (1.0 - smooth) * (*freq / scaling.factor) + smooth * *freq;
        }
    }
    freqs
}

fn apply_rope_inplace(x: &mut [f32], pos: usize, inv_freq: &[f32], attn_mscale: f32) {
    debug_assert_eq!(x.len() % 2, 0);
    for i in 0..x.len() / 2 {
        let angle = pos as f32 * inv_freq[i];
        let c = angle.cos() * attn_mscale;
        let s = angle.sin() * attn_mscale;
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * c - x1 * s;
        x[2 * i + 1] = x0 * s + x1 * c;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn softmax_inplace(v: &mut [f32]) {
    let max = v
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, x| acc.max(x));
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum.max(1e-12);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(gamma.iter())
        .map(|(v, g)| v * inv * g)
        .collect()
}

fn topk_normalized(scores: &[f32], k: usize, routed_scaling_factor: f32) -> (Vec<usize>, Vec<f32>) {
    let mut pairs: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut top = pairs.into_iter().take(k).collect::<Vec<_>>();
    let sum = top.iter().map(|(_, s)| *s).sum::<f32>().max(1e-12);
    let ids = top.iter().map(|(idx, _)| *idx).collect();
    let weights = top
        .drain(..)
        .map(|(_, s)| (s / sum) * routed_scaling_factor)
        .collect();
    (ids, weights)
}

fn lock_err(model_dir: &Path) -> RvllmError {
    corrupt_err(model_dir, "k2 cache mutex poisoned".into())
}

fn corrupt_err(path: &Path, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx {
            path: path.to_path_buf(),
            tensor: None,
        },
        bt: Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_group32_matches_layout() {
        let packed = vec![
            0x8765_4321i32,
            0x0fed_cba9i32,
            0x7654_3210i32,
            0xfedc_ba98i32,
        ];
        let scales = vec![bf16::from_f32(2.0)];
        let out = dequant_int4_group32(&packed, &scales, 1, 32).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], -14.0);
        assert_eq!(out[1], -12.0);
        assert_eq!(out[7], 0.0);
        assert_eq!(out[8], 2.0);
        assert_eq!(out[15], 16.0);
    }
}
