//! Qwen 3.6 35B-A3B weight loader.
//!
//! Phase 1 (`load_qwen36_outside`): the three outside-the-stack tensors
//! (embedding, final RMSNorm, lm_head) — bf16 → f16 upload.
//!
//! Phase 2a (`load_qwen36_model`): outside + per-layer attention tensors
//! for the 10 full-attention layers. Per-layer linear-attention state
//! matrices and the 256-expert MoE block are NOT yet uploaded; they
//! land in Phase 2b/3.
//!
//! Tensor naming: Qwen 3.6 uses the `model.language_model.*` prefix.
//! FP8 weights carry a `weight_scale_inv` companion (BF16, shape
//! `[N/128, K/128]`, blockwise [128, 128] scales — same layout the
//! sm_120 CUTLASS blockwise FP8 GEMM consumes for Gemma 4).

use std::collections::BTreeMap;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_mem::HbmArena;

use crate::fp8_quant::{check_clamp_gate, quantize_per_tensor_ref, FP8_E4M3_MAX};

use crate::load::LayerAttnType;
use crate::qwen36_weights::{
    Qwen36FullAttnLayer, Qwen36Layer, Qwen36LayerAttn, Qwen36LinearAttnLayer,
    Qwen36LoadedModel, Qwen36LoadedOutside, Qwen36MoeBlock,
};
use crate::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use crate::weights::{F16Weight, Fp8Weight};

const QWEN36_PREFIX: &str = "model.language_model";

/// Internal mmap wrapper. Mirrors the private `ShardMap` in
/// `gemma4_load.rs` — kept duplicate here to avoid widening the public
/// API surface for a Phase-1/2 loader.
struct ShardMap {
    _mmap: Mmap,
    header: ShardHeader,
}

impl ShardMap {
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
        let header = ShardHeader::parse(path, &mmap)?;
        Ok(Self {
            _mmap: mmap,
            header,
        })
    }

    fn bytes(&self) -> &[u8] {
        &self._mmap
    }
}

/// Per-load context: opened shards + tensor index. Shared between the
/// outside-tensor pass and the per-layer pass so we mmap each shard
/// exactly once.
struct LoadCtx<'a> {
    shards: Vec<ShardMap>,
    tensors: BTreeMap<String, (usize, TensorEntry)>,
    model_dir: &'a Path,
    arena: &'a HbmArena<'a>,
}

impl<'a> LoadCtx<'a> {
    fn new(model_dir: &'a Path, arena: &'a HbmArena<'a>) -> Result<Self> {
        let idx = ShardIndex::resolve(model_dir)?;
        let mut shards = Vec::with_capacity(idx.shards.len());
        for p in &idx.shards {
            shards.push(ShardMap::open(p)?);
        }
        let mut tensors: BTreeMap<String, (usize, TensorEntry)> = BTreeMap::new();
        for (si, sm) in shards.iter().enumerate() {
            for (name, entry) in &sm.header.tensors {
                tensors.insert(name.clone(), (si, entry.clone()));
            }
        }
        Ok(Self {
            shards,
            tensors,
            model_dir,
            arena,
        })
    }

    fn bytes_of(&self, si: usize, e: &TensorEntry) -> &[u8] {
        let s = self.shards[si].bytes();
        let start = e.file_offset as usize;
        &s[start..start + e.nbytes as usize]
    }

    fn must_get(&self, name: &str) -> Result<(usize, TensorEntry)> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| RvllmError::Loader {
                err: LoaderError::MissingTensor {
                    name: name.to_string(),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.to_path_buf(),
                    tensor: Some(name.to_string()),
                },
                bt: std::backtrace::Backtrace::capture(),
            })
    }

    fn upload_f16(&self, region_name: &'static str, hf_name: &str) -> Result<(F16Weight, u64)> {
        self.upload_f16_with_bias(region_name, hf_name, 0.0)
    }

    /// Upload f16 weight, optionally adding `bias` to every element
    /// before uploading. Used for Qwen 3.6 RMSNorm gammas which are
    /// stored centered at 0 and apply as `out = x * rsqrt(...) *
    /// (1.0 + gamma)` (Gemma-style RMSNorm via vLLM's GemmaRMSNorm
    /// alias). Our kernel multiplies directly by gamma so we need to
    /// add 1.0 at load time. Linear-attn `norm.weight` (RMSNormGated)
    /// is centered at 1 already and uses bias=0.
    fn upload_f16_with_bias(
        &self,
        region_name: &'static str,
        hf_name: &str,
        bias: f32,
    ) -> Result<(F16Weight, u64)> {
        let (si, e) = self.must_get(hf_name)?;
        let mut buf = tensor_to_f16_bytes(&e, self.bytes_of(si, &e), self.model_dir)?;
        if bias != 0.0 {
            // f16 bytes; in-place add `bias` to every element.
            let n = buf.len() / 2;
            for i in 0..n {
                let lo = buf[i * 2];
                let hi = buf[i * 2 + 1];
                let bits = u16::from_le_bytes([lo, hi]);
                let v = f16::from_bits(bits).to_f32() + bias;
                let new_bits = f16::from_f32(v).to_bits();
                let nb = new_bits.to_le_bytes();
                buf[i * 2] = nb[0];
                buf[i * 2 + 1] = nb[1];
            }
        }
        let region = self.arena.region(region_name, buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        Ok((
            F16Weight {
                offset_bytes: region.device_ptr(),
                shape: e.shape.clone(),
            },
            buf.len() as u64,
        ))
    }

    /// Upload an FP8 weight + its blockwise `weight_scale_inv` companion.
    /// Qwen 3.6 ships FP8 e4m3 weights with bf16 [N/128, K/128] scales.
    /// We:
    ///   1. Upload the FP8 bytes verbatim.
    ///   2. Convert the bf16 scale tensor to f32 and upload as the
    ///      blockscale arena region (the sm_120 CUTLASS path expects
    ///      f32 scales here, mirroring the Gemma 4 fp8 blockwise pipe).
    ///   3. Set `scale = 1.0` (per-tensor scale unused on the blockwise
    ///      path) + `channelscale_ptr = None` (Qwen has no per-row
    ///      channelscale).
    fn upload_fp8_blockwise(
        &self,
        region_name: &'static str,
        weight_name: &str,
    ) -> Result<Fp8Weight> {
        let (wsi, we) = self.must_get(weight_name)?;
        if we.dtype != DType::Fp8E4M3 {
            return Err(RvllmError::Loader {
                err: LoaderError::DtypeMismatch {
                    tensor: we.name.clone(),
                    expected: DType::Fp8E4M3,
                    got: we.dtype,
                },
                ctx: LoaderCtx {
                    path: self.model_dir.to_path_buf(),
                    tensor: Some(we.name.clone()),
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let raw = self.bytes_of(wsi, &we);
        let region = self.arena.region(region_name, raw.len(), 16)?;
        unsafe { region.copy_from_host(raw)? };

        let scale_name = format!("{weight_name}_scale_inv");
        let (ssi, se) = self.must_get(&scale_name)?;
        // Qwen ships bf16 scales — convert to f32 for the kernel.
        let scale_bytes = match se.dtype {
            DType::Bf16 => bf16_bytes_to_f32_bytes(self.bytes_of(ssi, &se)),
            DType::F32 => self.bytes_of(ssi, &se).to_vec(),
            _ => {
                return Err(RvllmError::Loader {
                    err: LoaderError::DtypeMismatch {
                        tensor: se.name.clone(),
                        expected: DType::F32,
                        got: se.dtype,
                    },
                    ctx: LoaderCtx {
                        path: self.model_dir.to_path_buf(),
                        tensor: Some(se.name.clone()),
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
        };
        let bs_region = self.arena.region("qwen36_fp8_blockscale", scale_bytes.len(), 16)?;
        unsafe { bs_region.copy_from_host(&scale_bytes)? };

        // Per-tensor scalar `1.0` placeholder (the blockwise path drives
        // dequant via the 2-D `blockscale_ptr`; per-tensor `scale` is
        // unused but the Fp8Weight contract requires a non-null pointer).
        let one = 1.0f32;
        let one_r = self.arena.region("qwen36_fp8_scale", 4, 4)?;
        unsafe { one_r.copy_from_host(&one.to_le_bytes())? };

        let weight_n = we.shape[0];
        let weight_k = if we.shape.len() >= 2 { we.shape[1] } else { 0 };
        let n_blocks = (weight_n + 127) / 128;
        let k_blocks = (weight_k + 127) / 128;

        Ok(Fp8Weight {
            offset_bytes: region.device_ptr(),
            scale_ptr: one_r.device_ptr(),
            shape: we.shape.clone(),
            scale: 1.0,
            clamp_ppm: 0.0,
            dtype: DType::Fp8E4M3,
            channelscale_ptr: None,
            blockscale_ptr: Some(bs_region.device_ptr()),
            blockscale_n_blocks: n_blocks as u32,
            blockscale_k_blocks: k_blocks as u32,
        })
    }

    /// CPU-quantize a bf16/f16 tensor to FP8 with per-tensor scale,
    /// clamp-gate, and arena upload. Used for the lm_head FP8 mirror
    /// (Phase 3d) — the same idea Gemma 4 uses for the tied-embedding
    /// lm_head, inlined here to avoid widening the loader pub API.
    fn upload_lm_head_fp8(
        &self,
        f16_weight: &F16Weight,
        f16_bytes_len: u64,
        tensor_name: &str,
    ) -> Result<(Fp8Weight, u64)> {
        // Re-read the source tensor from the safetensors mmap and run
        // the same bf16→f16 conversion the f16 upload used. Cheaper
        // than DtoH on the just-uploaded f16 region.
        let (si, e) = self.must_get(tensor_name)?;
        let f16_bytes = tensor_to_f16_bytes(&e, self.bytes_of(si, &e), self.model_dir)?;
        debug_assert_eq!(f16_bytes.len() as u64, f16_bytes_len);
        // f16 → f32 (rayon-parallel)
        use rayon::prelude::*;
        let f32_vals: Vec<f32> = f16_bytes
            .par_chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        let q = quantize_per_tensor_ref(&f32_vals);
        check_clamp_gate(tensor_name, q.clamp_ppm, self.model_dir)?;
        let fp8: Vec<u8> = f32_vals
            .par_iter()
            .map(|v| {
                fp8_e4m3_encode((*v / q.scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX))
            })
            .collect();

        let region_name: &'static str = "qwen36_lm_head_fp8";
        let region = self.arena.region(region_name, fp8.len(), 16)?;
        unsafe { region.copy_from_host(&fp8)? };
        let scale_region = self.arena.region("qwen36_lm_head_fp8_scale", 4, 4)?;
        unsafe { scale_region.copy_from_host(&q.scale.to_le_bytes())? };

        Ok((
            Fp8Weight {
                offset_bytes: region.device_ptr(),
                scale_ptr: scale_region.device_ptr(),
                shape: f16_weight.shape.clone(),
                scale: q.scale,
                clamp_ppm: q.clamp_ppm,
                dtype: DType::Fp8E4M3,
                channelscale_ptr: None,
                blockscale_ptr: None,
                blockscale_n_blocks: 0,
                blockscale_k_blocks: 0,
            },
            fp8.len() as u64,
        ))
    }

    /// Upload `num_experts` expert weight tensors (all sharing one
    /// projection role: `gate_proj`, `up_proj`, or `down_proj`) as a
    /// single fused FP8 arena region. Per-expert blockwise scales are
    /// stacked into one f32 region the same way.
    ///
    /// Layout in the fused region: expert `e` starts at byte offset
    /// `e * per_expert_fp8_bytes`. The shape recorded in the returned
    /// `Fp8Weight` is `[num_experts, N, K]` (3-D), making it explicit
    /// to downstream MoE-GEMM kernels that the leading axis is the
    /// expert index rather than a row of a 2-D matrix.
    fn upload_experts_fused(
        &self,
        region_name: &'static str,
        layer_idx: usize,
        num_experts: usize,
        projection: &str,
    ) -> Result<Fp8Weight> {
        let first_w_name = format!(
            "{QWEN36_PREFIX}.layers.{layer_idx}.mlp.experts.0.{projection}.weight"
        );
        let first_s_name = format!(
            "{QWEN36_PREFIX}.layers.{layer_idx}.mlp.experts.0.{projection}.weight_scale_inv"
        );
        let (_, w0) = self.must_get(&first_w_name)?;
        let (_, s0) = self.must_get(&first_s_name)?;
        if w0.dtype != DType::Fp8E4M3 {
            return Err(RvllmError::Loader {
                err: LoaderError::DtypeMismatch {
                    tensor: w0.name.clone(),
                    expected: DType::Fp8E4M3,
                    got: w0.dtype,
                },
                ctx: LoaderCtx {
                    path: self.model_dir.to_path_buf(),
                    tensor: Some(w0.name.clone()),
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let per_w_bytes = w0.nbytes as usize;
        let per_scale_elems = (s0.nbytes as usize) / 2; // bf16 → 2 bytes/elem
        let total_w_bytes = per_w_bytes * num_experts;
        let total_scale_bytes = per_scale_elems * 4 * num_experts; // bf16 → f32

        // Stage host-side, then one HtoD per fused region.
        let mut fused_w: Vec<u8> = Vec::with_capacity(total_w_bytes);
        let mut fused_s: Vec<u8> = Vec::with_capacity(total_scale_bytes);
        let n0 = w0.shape[0];
        let k0 = if w0.shape.len() >= 2 { w0.shape[1] } else { 0 };

        for e in 0..num_experts {
            let wn = format!(
                "{QWEN36_PREFIX}.layers.{layer_idx}.mlp.experts.{e}.{projection}.weight"
            );
            let sn = format!(
                "{QWEN36_PREFIX}.layers.{layer_idx}.mlp.experts.{e}.{projection}.weight_scale_inv"
            );
            let (wsi, we) = self.must_get(&wn)?;
            if we.nbytes as usize != per_w_bytes
                || we.shape.first().copied() != Some(n0)
                || we.shape.get(1).copied() != Some(k0)
            {
                return Err(RvllmError::Loader {
                    err: LoaderError::Corrupt {
                        detail: format!(
                            "expert {e} {projection} shape {:?} mismatches expert 0 shape {:?}",
                            we.shape, w0.shape
                        ),
                    },
                    ctx: LoaderCtx {
                        path: self.model_dir.to_path_buf(),
                        tensor: Some(wn),
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
            fused_w.extend_from_slice(self.bytes_of(wsi, &we));

            let (ssi, se) = self.must_get(&sn)?;
            let scale_f32 = match se.dtype {
                DType::Bf16 => bf16_bytes_to_f32_bytes(self.bytes_of(ssi, &se)),
                DType::F32 => self.bytes_of(ssi, &se).to_vec(),
                _ => {
                    return Err(RvllmError::Loader {
                        err: LoaderError::DtypeMismatch {
                            tensor: se.name.clone(),
                            expected: DType::F32,
                            got: se.dtype,
                        },
                        ctx: LoaderCtx {
                            path: self.model_dir.to_path_buf(),
                            tensor: Some(sn),
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    });
                }
            };
            fused_s.extend_from_slice(&scale_f32);
        }

        let region = self.arena.region(region_name, fused_w.len(), 16)?;
        unsafe { region.copy_from_host(&fused_w)? };
        let bs_region = self.arena.region("qwen36_moe_blockscale", fused_s.len(), 16)?;
        unsafe { bs_region.copy_from_host(&fused_s)? };

        let one = 1.0f32;
        let one_r = self.arena.region("qwen36_fp8_scale", 4, 4)?;
        unsafe { one_r.copy_from_host(&one.to_le_bytes())? };

        let n_blocks = (n0 + 127) / 128;
        let k_blocks = (k0 + 127) / 128;

        Ok(Fp8Weight {
            offset_bytes: region.device_ptr(),
            scale_ptr: one_r.device_ptr(),
            shape: vec![num_experts, n0, k0],
            scale: 1.0,
            clamp_ppm: 0.0,
            dtype: DType::Fp8E4M3,
            channelscale_ptr: None,
            blockscale_ptr: Some(bs_region.device_ptr()),
            blockscale_n_blocks: n_blocks as u32,
            blockscale_k_blocks: k_blocks as u32,
        })
    }
}

/// Phase 1 entry point: outside tensors only.
pub fn load_qwen36_outside(
    model_dir: &Path,
    arena: &HbmArena,
) -> Result<Qwen36LoadedOutside> {
    let ctx = LoadCtx::new(model_dir, arena)?;
    load_outside_via_ctx(&ctx)
}

/// Phase 2b entry point: outside tensors + every per-layer block
/// (linear-attn or full-attn) + 256-expert MoE per layer.
///
/// `num_experts` must match `Qwen36Arch::num_experts` (typically 256
/// for the 35B-A3B variant).
pub fn load_qwen36_model(
    model_dir: &Path,
    arena: &HbmArena,
    layer_types: &[LayerAttnType],
    num_experts: usize,
) -> Result<Qwen36LoadedModel> {
    let ctx = LoadCtx::new(model_dir, arena)?;
    let outside = load_outside_via_ctx(&ctx)?;

    let mut layers: Vec<Qwen36Layer> = Vec::with_capacity(layer_types.len());
    let mut n_full = 0usize;
    let mut n_linear = 0usize;
    let load_started = std::time::Instant::now();

    for (l, ty) in layer_types.iter().enumerate() {
        let attn = match ty {
            LayerAttnType::Full => {
                n_full += 1;
                Qwen36LayerAttn::Full(load_full_attn_layer(&ctx, l)?)
            }
            LayerAttnType::Linear | LayerAttnType::SlidingAttention => {
                n_linear += 1;
                Qwen36LayerAttn::Linear(load_linear_attn_layer(&ctx, l)?)
            }
        };
        let moe = load_moe_block(&ctx, l, num_experts)?;
        let elapsed = load_started.elapsed().as_secs_f64();
        if l == 0 || l == layer_types.len() - 1 || (l + 1) % 10 == 0 {
            eprintln!(
                "[qwen36-loader] layer {l}/{total} loaded ({attn_kind}) \
                 [{elapsed:.1}s elapsed, arena.used={:.2} GiB]",
                arena.used() as f64 / (1024.0 * 1024.0 * 1024.0),
                total = layer_types.len(),
                attn_kind = match ty {
                    LayerAttnType::Full => "full",
                    LayerAttnType::Linear => "linear",
                    LayerAttnType::SlidingAttention => "sliding",
                },
                elapsed = elapsed,
            );
        }
        layers.push(Qwen36Layer { attn, moe });
    }

    eprintln!(
        "[qwen36-loader] full per-layer upload complete: \
         {n_full} full-attn + {n_linear} linear-attn layers, \
         {num_experts} experts/layer + shared expert. \
         Total wall: {:.1}s.",
        load_started.elapsed().as_secs_f64(),
    );

    Ok(Qwen36LoadedModel { outside, layers })
}

fn load_outside_via_ctx(ctx: &LoadCtx) -> Result<Qwen36LoadedOutside> {
    let embed_name = format!("{QWEN36_PREFIX}.embed_tokens.weight");
    let norm_name = format!("{QWEN36_PREFIX}.norm.weight");
    let lm_head_name = "lm_head.weight";

    let (embed_tokens, embed_tokens_bytes) = ctx.upload_f16("qwen36_embedding", &embed_name)?;
    // Qwen3-Next's final norm uses GemmaRMSNorm (out = x * rsqrt(...) *
    // (1 + gamma)); checkpoint stores gamma centered at 0 → add +1.
    let (final_norm, final_norm_bytes) =
        ctx.upload_f16_with_bias("qwen36_final_norm", &norm_name, 1.0)?;
    let (lm_head, lm_head_bytes) = ctx.upload_f16("qwen36_lm_head", lm_head_name)?;

    // Phase 3d: CPU-quantize lm_head bf16 → FP8 per-tensor for the
    // fp8_gemv kernel path. Mirrors Gemma 4's tied-embedding case
    // (gemma4_load.rs:198-211) — same `upload_fp8` shape used there,
    // inlined here to avoid widening the rvllm-loader pub API.
    let (lm_head_fp8, lm_head_fp8_bytes) =
        ctx.upload_lm_head_fp8(&lm_head, lm_head_bytes, lm_head_name)?;

    eprintln!(
        "[qwen36-loader] outside tensors uploaded: \
         embed_tokens {:?} ({:.1} MiB), \
         final_norm {:?} ({:.2} KiB), \
         lm_head_f16 {:?} ({:.1} MiB), \
         lm_head_fp8 {:?} ({:.1} MiB, scale={:.6e}, clamp_ppm={:.3})",
        embed_tokens.shape,
        embed_tokens_bytes as f64 / (1024.0 * 1024.0),
        final_norm.shape,
        final_norm_bytes as f64 / 1024.0,
        lm_head.shape,
        lm_head_bytes as f64 / (1024.0 * 1024.0),
        lm_head_fp8.shape,
        lm_head_fp8_bytes as f64 / (1024.0 * 1024.0),
        lm_head_fp8.scale,
        lm_head_fp8.clamp_ppm,
    );

    Ok(Qwen36LoadedOutside {
        embed_tokens,
        final_norm,
        lm_head,
        lm_head_fp8,
        embed_tokens_bytes,
        final_norm_bytes,
        lm_head_bytes,
        lm_head_fp8_bytes,
    })
}

fn load_linear_attn_layer(ctx: &LoadCtx, layer_idx: usize) -> Result<Qwen36LinearAttnLayer> {
    let ln = |s: &str| format!("{QWEN36_PREFIX}.layers.{layer_idx}.{s}");

    // GemmaRMSNorm-style: gamma centered at 0, add +1 at load.
    let (input_layernorm, _) = ctx.upload_f16_with_bias(
        "qwen36_input_ln", &ln("input_layernorm.weight"), 1.0)?;
    let (post_attention_layernorm, _) = ctx.upload_f16_with_bias(
        "qwen36_post_attn_ln", &ln("post_attention_layernorm.weight"), 1.0)?;
    let (a_log, _) = ctx.upload_f16("qwen36_a_log", &ln("linear_attn.A_log"))?;
    let (dt_bias, _) = ctx.upload_f16("qwen36_dt_bias", &ln("linear_attn.dt_bias"))?;
    let (conv1d, _) = ctx.upload_f16("qwen36_conv1d", &ln("linear_attn.conv1d.weight"))?;
    let (in_proj_a, _) = ctx.upload_f16("qwen36_in_proj_a", &ln("linear_attn.in_proj_a.weight"))?;
    let (in_proj_b, _) = ctx.upload_f16("qwen36_in_proj_b", &ln("linear_attn.in_proj_b.weight"))?;
    let (norm, _) = ctx.upload_f16("qwen36_la_norm", &ln("linear_attn.norm.weight"))?;

    let in_proj_qkv =
        ctx.upload_fp8_blockwise("qwen36_la_in_qkv", &ln("linear_attn.in_proj_qkv.weight"))?;
    let in_proj_z =
        ctx.upload_fp8_blockwise("qwen36_la_in_z", &ln("linear_attn.in_proj_z.weight"))?;
    let out_proj =
        ctx.upload_fp8_blockwise("qwen36_la_out", &ln("linear_attn.out_proj.weight"))?;

    if layer_idx <= 2 {
        eprintln!(
            "[qwen36-loader] layer {layer_idx} linear-attn: \
             qkv={:?} z={:?} out={:?} a_log={:?} conv1d={:?}",
            in_proj_qkv.shape, in_proj_z.shape, out_proj.shape,
            a_log.shape, conv1d.shape,
        );
    }

    Ok(Qwen36LinearAttnLayer {
        input_layernorm,
        post_attention_layernorm,
        a_log,
        dt_bias,
        conv1d,
        in_proj_a,
        in_proj_b,
        in_proj_qkv,
        in_proj_z,
        norm,
        out_proj,
    })
}

/// Load the per-layer MoE block. Routes 256 experts into 3 fused FP8
/// arena regions (gate / up / down) — each holds 256 expert weight
/// matrices stacked along the leading axis. Per-expert blockwise
/// scales are likewise stacked into a single f32 region per role.
fn load_moe_block(ctx: &LoadCtx, layer_idx: usize, num_experts: usize) -> Result<Qwen36MoeBlock> {
    let ln = |s: &str| format!("{QWEN36_PREFIX}.layers.{layer_idx}.mlp.{s}");

    let (router, _) = ctx.upload_f16("qwen36_moe_router", &ln("gate.weight"))?;
    let (shared_expert_gate_logit, _) =
        ctx.upload_f16("qwen36_moe_shared_gate", &ln("shared_expert_gate.weight"))?;

    let shared_expert_gate_proj = ctx
        .upload_fp8_blockwise("qwen36_moe_sh_gp", &ln("shared_expert.gate_proj.weight"))?;
    let shared_expert_up_proj =
        ctx.upload_fp8_blockwise("qwen36_moe_sh_up", &ln("shared_expert.up_proj.weight"))?;
    let shared_expert_down_proj = ctx
        .upload_fp8_blockwise("qwen36_moe_sh_dn", &ln("shared_expert.down_proj.weight"))?;

    let experts_gate_proj_fused = ctx.upload_experts_fused(
        "qwen36_moe_exp_gp",
        layer_idx,
        num_experts,
        "gate_proj",
    )?;
    let experts_up_proj_fused = ctx.upload_experts_fused(
        "qwen36_moe_exp_up",
        layer_idx,
        num_experts,
        "up_proj",
    )?;
    let experts_down_proj_fused = ctx.upload_experts_fused(
        "qwen36_moe_exp_dn",
        layer_idx,
        num_experts,
        "down_proj",
    )?;

    Ok(Qwen36MoeBlock {
        router,
        shared_expert_gate_logit,
        experts_gate_proj_fused,
        experts_up_proj_fused,
        experts_down_proj_fused,
        shared_expert_gate_proj,
        shared_expert_up_proj,
        shared_expert_down_proj,
    })
}

fn load_full_attn_layer(ctx: &LoadCtx, layer_idx: usize) -> Result<Qwen36FullAttnLayer> {
    let ln = |s: &str| format!("{QWEN36_PREFIX}.layers.{layer_idx}.{s}");

    // All four are GemmaRMSNorm-style; gamma centered at 0, add +1.
    let (input_layernorm, _) = ctx.upload_f16_with_bias(
        "qwen36_input_ln", &ln("input_layernorm.weight"), 1.0)?;
    let (post_attention_layernorm, _) = ctx.upload_f16_with_bias(
        "qwen36_post_attn_ln", &ln("post_attention_layernorm.weight"), 1.0)?;
    let (q_norm, _) = ctx.upload_f16_with_bias(
        "qwen36_q_norm", &ln("self_attn.q_norm.weight"), 1.0)?;
    let (k_norm, _) = ctx.upload_f16_with_bias(
        "qwen36_k_norm", &ln("self_attn.k_norm.weight"), 1.0)?;

    let q_proj = ctx.upload_fp8_blockwise("qwen36_q_proj", &ln("self_attn.q_proj.weight"))?;
    let k_proj = ctx.upload_fp8_blockwise("qwen36_k_proj", &ln("self_attn.k_proj.weight"))?;
    let v_proj = ctx.upload_fp8_blockwise("qwen36_v_proj", &ln("self_attn.v_proj.weight"))?;
    let o_proj = ctx.upload_fp8_blockwise("qwen36_o_proj", &ln("self_attn.o_proj.weight"))?;

    if layer_idx <= 7 {
        eprintln!(
            "[qwen36-loader] layer {layer_idx} full-attn: q={:?} k={:?} v={:?} o={:?}",
            q_proj.shape, k_proj.shape, v_proj.shape, o_proj.shape,
        );
    }

    Ok(Qwen36FullAttnLayer {
        input_layernorm,
        post_attention_layernorm,
        q_norm,
        k_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
    })
}

fn tensor_to_f16_bytes(e: &TensorEntry, raw: &[u8], model_dir: &Path) -> Result<Vec<u8>> {
    match e.dtype {
        DType::F16 => Ok(raw.to_vec()),
        DType::Bf16 => Ok(bf16_bytes_to_f16_bytes(raw)),
        DType::F32 => Ok(f32_bytes_to_f16_bytes(raw)),
        _ => Err(RvllmError::Loader {
            err: LoaderError::DtypeMismatch {
                tensor: e.name.clone(),
                expected: DType::F16,
                got: e.dtype,
            },
            ctx: LoaderCtx {
                path: model_dir.to_path_buf(),
                tensor: Some(e.name.clone()),
            },
            bt: std::backtrace::Backtrace::capture(),
        }),
    }
}

fn bf16_bytes_to_f16_bytes(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let lo = raw[2 * i];
        let hi = raw[2 * i + 1];
        let as_f32 = f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]));
        out.extend_from_slice(&f16::from_f32(as_f32).to_le_bytes());
    }
    out
}

fn bf16_bytes_to_f32_bytes(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let lo = raw[2 * i];
        let hi = raw[2 * i + 1];
        let as_f32 = f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]));
        out.extend_from_slice(&as_f32.to_le_bytes());
    }
    out
}

/// FP8 E4M3FN encoder with round-to-nearest-even. Mirrors the private
/// helper in `load.rs` — kept duplicate here so the qwen36 loader
/// stays self-contained without widening the loader pub API. Finite
/// range [-448, 448]; `+/-0` canonicalises to 0x00.
fn fp8_e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let s: u8 = if v.to_bits() >> 31 != 0 { 0x80 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return 0;
    }
    if a > FP8_E4M3_MAX {
        return s | 0x7e;
    }
    let bits = a.to_bits();
    let exp32 = ((bits >> 23) & 0xff) as i32 - 127;
    let mant32 = bits & 0x7f_ffff;
    let mut exp8 = exp32 + 7;
    if exp8 <= 0 {
        let shift = 1 - exp8;
        let full = mant32 | (1 << 23);
        let rshift = (20 + shift) as u32;
        let mut m = full >> rshift;
        let round_bit = if rshift > 0 { (full >> (rshift - 1)) & 1 } else { 0 };
        let sticky = if rshift > 1 {
            (full & ((1 << (rshift - 1)) - 1) != 0) as u32
        } else {
            0
        };
        m += round_bit & (sticky | (m & 1));
        if m >= 8 {
            return s | 0x08;
        }
        return s | (m as u8 & 0x07);
    }
    let trunc = mant32 >> 20;
    let round_bit = (mant32 >> 19) & 1;
    let sticky = (mant32 & 0x7_ffff) != 0;
    let m = trunc + (round_bit & (sticky as u32 | (trunc & 1)));
    if m >= 8 {
        exp8 += 1;
        if exp8 > 15 {
            return s | 0x7e;
        }
        return s | ((exp8 as u8 & 0x0f) << 3);
    }
    if exp8 > 15 {
        return s | 0x7e;
    }
    s | ((exp8 as u8 & 0x0f) << 3) | (m as u8 & 0x07)
}

fn f32_bytes_to_f16_bytes(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}
