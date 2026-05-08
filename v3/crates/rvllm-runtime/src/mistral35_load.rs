//! Mistral 3.5 NVFP4 weight upload pass.
//!
//! Splits responsibilities across crates per the rvllm DAG:
//!
//! - `rvllm-loader` owns the struct definitions
//!   (`Mistral35LoadedModel`, `Mistral35LayerLoaded`,
//!   `Nvfp4LinearLoaded`) — see `mistral35_weights.rs`.
//! - This file (`rvllm-runtime`) owns the orchestration: it mmaps
//!   safetensors shards, allocates persistent arena regions for
//!   the device-resident weights, runs the per-projection
//!   `launch_nvfp4_sfb_transform` (CUTLASS — only reachable from
//!   this layer of the DAG), and assembles the
//!   `Mistral35LoadedModel`.
//!
//! Per-projection upload sequence (codex-reviewed strategy):
//!
//! 1. mmap the source shard (kernel pages-in lazily).
//! 2. Persistent device buffer for `weight_packed` (`U8 [N, K/2]`)
//!    via `arena.region(...) + copy_from_host`.
//! 3. Transient scratch region for natural-layout `weight_scale`
//!    (`E4M3 [N, K/16]`); upload, then run
//!    `launch_nvfp4_sfb_transform(natural -> CUTLASS-interleaved)`
//!    into a separate persistent region.
//! 4. Persistent device scalar for `weight_global_scale` (`F32 [1]`).
//! 5. Drop the natural scratch reference (arena bump-allocator
//!    keeps the high-water but the next projection reuses the same
//!    end of the arena via `arena.checkpoint() + restore` the
//!    bring-up does at outer-scope).
//!
//! The orchestration is intentionally synchronous + on-stream: no
//! pipelining of projection uploads today. 88 layers × 7 linears =
//! 616 transform launches at load-time; each is a few µs, total
//! upload dominated by the actual H→D byte copies.

use std::collections::BTreeMap;
use std::path::Path;

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_loader::mistral35_arch::Mistral35Arch;
use rvllm_loader::mistral35_weights::{
    Mistral35LayerLoaded, Mistral35LinearKind, Mistral35LoadedModel, Mistral35Outside,
    Nvfp4LinearLoaded, Nvfp4LinearShape,
};
use rvllm_loader::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use rvllm_loader::weights::F16Weight;
use rvllm_mem::HbmArena;

use crate::mistral35_bring_up::Mistral35EnginePaths;

/// Mmap'd shard pool — keeps every safetensors file mapped in for
/// the duration of the upload pass. `BTreeMap<name, (shard_idx,
/// entry)>` lets us locate any tensor without a per-projection
/// re-scan.
struct ShardPool {
    mmaps: Vec<memmap2::Mmap>,
    paths: Vec<std::path::PathBuf>,
    tensors: BTreeMap<String, (usize, TensorEntry)>,
    model_dir: std::path::PathBuf,
}

impl ShardPool {
    fn open(model_dir: &Path) -> Result<Self> {
        let idx = ShardIndex::resolve(model_dir)?;
        let mut mmaps = Vec::with_capacity(idx.shards.len());
        let mut paths = Vec::with_capacity(idx.shards.len());
        let mut tensors: BTreeMap<String, (usize, TensorEntry)> = BTreeMap::new();
        for (shard_idx, shard_path) in idx.shards.iter().enumerate() {
            let f = std::fs::File::open(shard_path).map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: shard_path.clone(),
                source,
            })?;
            let mmap = unsafe { memmap2::Mmap::map(&f) }
                .map_err(|source| RvllmError::Io {
                    err: rvllm_core::IoError::from(&source),
                    path: shard_path.clone(),
                    source,
                })?;
            let header = ShardHeader::parse(shard_path, &mmap)?;
            for (name, entry) in header.tensors.into_iter() {
                tensors.insert(name, (shard_idx, entry));
            }
            mmaps.push(mmap);
            paths.push(shard_path.clone());
        }
        Ok(Self { mmaps, paths, tensors, model_dir: model_dir.to_path_buf() })
    }

    fn must_get(&self, name: &str) -> Result<(usize, &TensorEntry)> {
        match self.tensors.get(name) {
            Some((si, e)) => Ok((*si, e)),
            None => Err(RvllmError::Loader {
                err: LoaderError::MissingTensor { name: name.to_string() },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: std::backtrace::Backtrace::capture(),
            }),
        }
    }

    fn bytes_of(&self, si: usize, e: &TensorEntry) -> &[u8] {
        let mm = &self.mmaps[si];
        let start = e.file_offset as usize;
        &mm[start..start + e.nbytes as usize]
    }
}

/// Upload a tensor verbatim (bytes-as-stored, no dtype conversion)
/// into the arena and return the resulting `F16Weight`. Despite
/// the name — F16Weight is the workspace's canonical 16-bit-class
/// half holder — this is also the carrier for BF16 tensors. The
/// kernel side reads via the matching dtype'd kernel signature.
fn upload_tensor_verbatim(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    region_name: &'static str,
    tensor_name: &str,
) -> Result<F16Weight> {
    let (si, e) = pool.must_get(tensor_name)?;
    let raw = pool.bytes_of(si, e);
    let region = arena.region(region_name, raw.len(), 16)?;
    unsafe { region.copy_from_host(raw)? };
    Ok(F16Weight {
        offset_bytes: region.device_ptr(),
        shape: e.shape.clone(),
    })
}

/// Upload one Mistral 3.5 NVFP4 linear, run the SFB transform, and
/// produce the device-resident `Nvfp4LinearLoaded` triple.
///
/// Stream + cutlass backend semantics (codex review #2):
/// `tile_atom_to_shape_SFB` consumes only `(N, K, L)`, so the
/// transform runs once per linear and the result is persistent.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn upload_nvfp4_linear(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    backend: &rvllm_cutlass::lib_so::CutlassBackend,
    stream: u64,
    base: &str,
    shape: Nvfp4LinearShape,
) -> Result<Nvfp4LinearLoaded> {
    // (1) weight_packed
    let packed_name = format!("{base}.weight_packed");
    let (psi, pe) = pool.must_get(&packed_name)?;
    let packed_raw = pool.bytes_of(psi, pe);
    let packed_region = arena.region("mistral35_w_packed", packed_raw.len(), 16)?;
    unsafe { packed_region.copy_from_host(packed_raw)? };
    let packed_ptr = packed_region.device_ptr();
    let packed_bytes = packed_raw.len();

    // (2) weight_scale (natural [N, K/16] E4M3) -> transient scratch.
    //     The natural region's lifetime ends after the transform
    //     completes; we keep the arena `Region` alive via the local
    //     binding only until that point.
    let scale_name = format!("{base}.weight_scale");
    let (ssi, se) = pool.must_get(&scale_name)?;
    let scale_raw = pool.bytes_of(ssi, se);
    let nat_bytes = backend.nvfp4_sfb_natural_bytes(shape.n as i32, shape.k as i32);
    if nat_bytes != scale_raw.len() {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "Mistral 3.5 weight_scale {scale_name}: mmap len={} \
                     but cutlass nvfp4_sfb_natural_bytes(n={}, k={}) = {}",
                    scale_raw.len(), shape.n, shape.k, nat_bytes,
                ),
            },
            ctx: LoaderCtx {
                path: pool.model_dir.clone(),
                tensor: Some(scale_name.clone()),
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    let nat_region = arena.region("mistral35_w_sfb_natural", scale_raw.len(), 16)?;
    unsafe { nat_region.copy_from_host(scale_raw)? };

    // (3) Persistent CUTLASS-interleaved SFB target.
    let sfb_bytes = backend.nvfp4_sfb_bytes(shape.n as i32, shape.k as i32);
    if sfb_bytes == 0 {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "cutlass nvfp4_sfb_bytes(n={}, k={}) = 0 — backend not \
                     active or shape rejected",
                    shape.n, shape.k,
                ),
            },
            ctx: LoaderCtx {
                path: pool.model_dir.clone(),
                tensor: Some(scale_name.clone()),
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    let sfb_region = arena.region("mistral35_w_sfb_cutlass", sfb_bytes, 16)?;
    unsafe {
        backend.launch_nvfp4_sfb_transform(
            nat_region.device_ptr(),
            sfb_region.device_ptr(),
            shape.n as i32,
            shape.k as i32,
            stream,
        )?;
    }
    let sfb_cutlass_ptr = sfb_region.device_ptr();

    // (4) weight_global_scale  F32 [1]
    let gs_name = format!("{base}.weight_global_scale");
    let (gsi, ge) = pool.must_get(&gs_name)?;
    let gs_raw = pool.bytes_of(gsi, ge);
    if gs_raw.len() != 4 {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "Mistral 3.5 {gs_name}: expected 4 bytes (F32 [1]), got {}",
                    gs_raw.len(),
                ),
            },
            ctx: LoaderCtx {
                path: pool.model_dir.clone(),
                tensor: Some(gs_name.clone()),
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    // LLMCompressor / compressed-tensors NVFP4 stores weight_global_scale
    // as the encode scale `FP8_E4M3_MAX * FP4_E2M1_MAX / weight_amax`.
    // Its per-block E4M3 scales are generated as `(block_amax / 6) *
    // weight_global_scale`, while dequantization divides that local scale
    // by weight_global_scale.
    //
    // CUTLASS Sm120 blockscaled MMA multiplies by SFA and SFB as supplied,
    // and its epilogue `alpha_ptr` is a plain multiplicative output scalar.
    // Therefore the device scalar passed to CUTLASS must be the decode
    // scale, `1 / weight_global_scale`, not the checkpoint value.
    let gs_f32: f32 = f32::from_le_bytes([gs_raw[0], gs_raw[1], gs_raw[2], gs_raw[3]]);
    let alpha_f32 = if gs_f32.is_finite() && gs_f32 != 0.0 {
        1.0_f32 / gs_f32
    } else {
        gs_f32
    };
    let gs_region = arena.region("mistral35_w_global_scale", 4, 4)?;
    unsafe { gs_region.copy_from_host(&alpha_f32.to_le_bytes())? };
    let global_scale_ptr = gs_region.device_ptr();

    Ok(Nvfp4LinearLoaded {
        shape,
        packed_ptr,
        sfb_cutlass_ptr,
        global_scale_ptr,
        packed_bytes,
        sfb_bytes,
    })
}

/// Synchronously upload all Mistral 3.5 weights into `arena` and
/// run the SFB transform per projection.
///
/// Caller must hold the `cutlass_backend` for the lifetime of the
/// returned `Mistral35LoadedModel` (the SFB pointers are owned by
/// `arena`, but the launch backend handle is needed for forward).
#[cfg(feature = "cuda")]
pub fn load_mistral35_model(
    paths: &Mistral35EnginePaths,
    arch: &Mistral35Arch,
    arena: &HbmArena<'_>,
    cutlass_backend: &rvllm_cutlass::lib_so::CutlassBackend,
    stream: u64,
) -> Result<Mistral35LoadedModel> {
    cutlass_backend.require_nvfp4()?;
    let pool = ShardPool::open(&paths.model_dir)?;

    let prefix = arch.weight_prefix.as_str();
    eprintln!("[mistral35-load] uploading outside tensors (embed + norm + lm_head)…");
    let outside = Mistral35Outside {
        embed_tokens: upload_tensor_verbatim(
            arena, &pool, "mistral35_embed",
            &format!("{prefix}.embed_tokens.weight"),
        )?,
        final_norm: upload_tensor_verbatim(
            arena, &pool, "mistral35_final_norm",
            &format!("{prefix}.norm.weight"),
        )?,
        // Mistral 3.5: tie_word_embeddings = false → separate
        // lm_head tensor at top level.
        lm_head: upload_tensor_verbatim(
            arena, &pool, "mistral35_lm_head", "lm_head.weight",
        )?,
    };
    eprintln!(
        "[mistral35-load]   embed: {:?}  final_norm: {:?}  lm_head: {:?}",
        outside.embed_tokens.shape, outside.final_norm.shape, outside.lm_head.shape,
    );

    eprintln!(
        "[mistral35-load] uploading {} decoder layers (norms + 7 NVFP4 linears each)…",
        arch.text.num_hidden_layers,
    );
    let mut layers = Vec::with_capacity(arch.text.num_hidden_layers);
    let layer_log_step = (arch.text.num_hidden_layers / 8).max(1);
    for li in 0..arch.text.num_hidden_layers {
        let layer = upload_one_layer(arena, &pool, cutlass_backend, stream, prefix, li, arch)?;
        if li % layer_log_step == 0 || li + 1 == arch.text.num_hidden_layers {
            eprintln!("[mistral35-load]   layer {li}/{} done", arch.text.num_hidden_layers);
        }
        layers.push(layer);
    }
    eprintln!("[mistral35-load] all language weights resident on device");

    // Pixtral vision tower (BF16). Optional — controlled by env
    // RVLLM_LOAD_VISION (default on for mistral35; allows skip for
    // text-only smoke).
    let vision = if std::env::var("RVLLM_LOAD_VISION").as_deref() == Ok("0") {
        None
    } else {
        Some(upload_mistral35_vision(arena, &pool, arch)?)
    };

    Ok(Mistral35LoadedModel { outside, layers, vision })
}

#[cfg(feature = "cuda")]
fn upload_mistral35_vision(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    arch: &Mistral35Arch,
) -> Result<rvllm_loader::mistral35_weights::Mistral35Vision> {
    use rvllm_loader::mistral35_weights::{Mistral35Vision, VisionLayerLoaded};

    eprintln!(
        "[mistral35-load] uploading vision tower (Pixtral, {} layers)…",
        arch.vision.num_hidden_layers,
    );

    // The checkpoint stores vision tensors under the *direct*
    // `model.vision_tower.*` prefix (not nested under language_model).
    // Mistral checkpoints sometimes use `model.vision_tower.` and
    // older variants used bare `vision_tower.`; the inventory
    // validator already accepts both. The shard pool's
    // BTreeMap doesn't care — names are exact.
    let vt = "model.vision_tower";
    let mmp = "model.multi_modal_projector";

    let patch_conv = upload_tensor_verbatim(
        arena, pool, "mistral35_v_patch_conv",
        &format!("{vt}.patch_conv.weight"),
    )?;
    let ln_pre = upload_tensor_verbatim(
        arena, pool, "mistral35_v_ln_pre",
        &format!("{vt}.ln_pre.weight"),
    )?;

    let mut vlayers = Vec::with_capacity(arch.vision.num_hidden_layers);
    let log_step = (arch.vision.num_hidden_layers / 6).max(1);
    for li in 0..arch.vision.num_hidden_layers {
        let lb = format!("{vt}.transformer.layers.{li}");
        let layer = VisionLayerLoaded {
            attention_norm: upload_tensor_verbatim(arena, pool, "v_attn_norm",
                &format!("{lb}.attention_norm.weight"))?,
            q_proj: upload_tensor_verbatim(arena, pool, "v_q",
                &format!("{lb}.attention.q_proj.weight"))?,
            k_proj: upload_tensor_verbatim(arena, pool, "v_k",
                &format!("{lb}.attention.k_proj.weight"))?,
            v_proj: upload_tensor_verbatim(arena, pool, "v_v",
                &format!("{lb}.attention.v_proj.weight"))?,
            o_proj: upload_tensor_verbatim(arena, pool, "v_o",
                &format!("{lb}.attention.o_proj.weight"))?,
            ffn_norm: upload_tensor_verbatim(arena, pool, "v_ffn_norm",
                &format!("{lb}.ffn_norm.weight"))?,
            gate_proj: upload_tensor_verbatim(arena, pool, "v_gate",
                &format!("{lb}.feed_forward.gate_proj.weight"))?,
            up_proj: upload_tensor_verbatim(arena, pool, "v_up",
                &format!("{lb}.feed_forward.up_proj.weight"))?,
            down_proj: upload_tensor_verbatim(arena, pool, "v_down",
                &format!("{lb}.feed_forward.down_proj.weight"))?,
        };
        vlayers.push(layer);
        if li % log_step == 0 || li + 1 == arch.vision.num_hidden_layers {
            eprintln!("[mistral35-load]   vision layer {li}/{} done",
                arch.vision.num_hidden_layers);
        }
    }

    let projector_norm = upload_tensor_verbatim(
        arena, pool, "v_proj_norm", &format!("{mmp}.norm.weight"))?;
    let projector_patch_merger = upload_tensor_verbatim(
        arena, pool, "v_patch_merger",
        &format!("{mmp}.patch_merger.merging_layer.weight"))?;
    let projector_linear_1 = upload_tensor_verbatim(
        arena, pool, "v_proj_l1", &format!("{mmp}.linear_1.weight"))?;
    let projector_linear_2 = upload_tensor_verbatim(
        arena, pool, "v_proj_l2", &format!("{mmp}.linear_2.weight"))?;

    eprintln!(
        "[mistral35-load] vision tower uploaded: patch_conv={:?} \
         ln_pre={:?} layers={} projector(norm/merger/l1/l2)={:?}/{:?}/{:?}/{:?}",
        patch_conv.shape, ln_pre.shape, vlayers.len(),
        projector_norm.shape, projector_patch_merger.shape,
        projector_linear_1.shape, projector_linear_2.shape,
    );

    Ok(Mistral35Vision {
        patch_conv, ln_pre, layers: vlayers,
        projector_norm, projector_patch_merger,
        projector_linear_1, projector_linear_2,
    })
}

#[cfg(feature = "cuda")]
fn upload_one_layer(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    backend: &rvllm_cutlass::lib_so::CutlassBackend,
    stream: u64,
    prefix: &str,
    li: usize,
    arch: &Mistral35Arch,
) -> Result<Mistral35LayerLoaded> {
    let layer_base = format!("{prefix}.layers.{li}");
    // Norms — bytes verbatim (BF16 [hidden]).
    let input_layernorm = upload_tensor_verbatim(
        arena, pool, "mistral35_in_norm",
        &format!("{layer_base}.input_layernorm.weight"),
    )?;
    let post_attention_layernorm = upload_tensor_verbatim(
        arena, pool, "mistral35_post_norm",
        &format!("{layer_base}.post_attention_layernorm.weight"),
    )?;

    // Resolve the seven NVFP4 linears.
    let mut loaded: [Option<Nvfp4LinearLoaded>; 7] = Default::default();
    for (idx, kind) in Mistral35LinearKind::ALL.iter().enumerate() {
        let shape = kind.shape_for(arch);
        let base = format!("{layer_base}.{}", kind.name());
        loaded[idx] = Some(upload_nvfp4_linear(arena, pool, backend, stream, &base, shape)?);
    }
    // SAFETY: the loop above populates every slot; we unwrap each
    // exactly once.
    let q_proj = loaded[0].take().unwrap();
    let k_proj = loaded[1].take().unwrap();
    let v_proj = loaded[2].take().unwrap();
    let o_proj = loaded[3].take().unwrap();
    let gate_proj = loaded[4].take().unwrap();
    let up_proj = loaded[5].take().unwrap();
    let down_proj = loaded[6].take().unwrap();

    Ok(Mistral35LayerLoaded {
        input_layernorm,
        post_attention_layernorm,
        q_proj, k_proj, v_proj, o_proj,
        gate_proj, up_proj, down_proj,
    })
}
