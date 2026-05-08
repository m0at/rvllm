//! NVFP4-packed weight types for Mistral Medium 3.5.
//!
//! The compressed-tensors NVFP4 layout (Mistral / NVIDIA spec):
//!
//! - **`<linear>.weight_packed`** — `U8`, shape `[N, K / 2]`. Two
//!   E2M1 (FP4) values are packed per byte: low nibble = element 2i,
//!   high nibble = element 2i+1.
//! - **`<linear>.weight_scale`** — `Fp8E4M3`, shape `[N, K / 16]`.
//!   One per-block scale per 16-element block of K.
//! - **`<linear>.weight_global_scale`** — `F32`, shape `[1]`. Outer
//!   scalar that the per-block scales multiply against; reuses the
//!   FP8-style `s_global * s_block * fp4_value` reconstruction.
//!
//! This is **not** the existing FP8-blockscale layout
//! (`[N/128, K/128]` 2-D). NVFP4 ships a **per-row × per-16-K**
//! grid, so the CUTLASS Sm120 ABI used for Mistral cannot reuse the
//! Gemma 4 / Qwen 3.6 FP8-block symbol set — it gets its own
//! function-pointer family in `rvllm-cutlass` (Step 4).
//!
//! The types here are loader-side only: shapes, expected tensor
//! names, validated entries. Device-upload + CUTLASS dispatch land
//! when the GPU backend is wired (Step 4 / 5).

use std::collections::BTreeMap;

use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};

use crate::mistral35_arch::Mistral35Arch;
use crate::safetensors::TensorEntry;
use crate::weights::F16Weight;

/// Logical shape of one NVFP4-packed linear, plus the validated
/// dtypes/shapes of its three companion tensors. Returned by
/// [`Mistral35WeightInventory::resolve_linear`] so the loader and the
/// CUTLASS backend share a single source of truth for shape math.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Nvfp4LinearShape {
    /// Output dimension (rows). For Mistral 3.5: 12288 for q/o,
    /// 1024 for k/v, 28672 for gate/up, 12288 for down.
    pub n: usize,
    /// Input dimension (cols, in *elements* not bytes). For Mistral
    /// 3.5: 12288 for q/k/v/o/gate/up, 28672 for down.
    pub k: usize,
}

impl Nvfp4LinearShape {
    /// `K / 2` — packed-byte width of `weight_packed`.
    pub fn packed_cols(&self) -> usize {
        self.k / 2
    }
    /// `K / 16` — number of per-row blocks in `weight_scale`.
    pub fn scale_cols(&self) -> usize {
        self.k / 16
    }
    /// Bytes occupied by the packed weight on disk + device.
    pub fn packed_bytes(&self) -> usize {
        self.n * self.packed_cols()
    }
    /// Bytes occupied by the FP8 per-block scales.
    pub fn scale_bytes(&self) -> usize {
        self.n * self.scale_cols() // 1 byte per E4M3 value
    }
}

/// One validated NVFP4 linear (weight + per-block scale + global
/// scale), as parsed out of the safetensors index. No device pointer
/// yet — that's set when the loader uploads.
#[derive(Clone, Debug)]
pub struct Nvfp4LinearWeight {
    pub shape: Nvfp4LinearShape,
    pub packed: TensorEntry,
    pub scale: TensorEntry,
    pub global_scale: TensorEntry,
}

/// Result of validating every Mistral 3.5 NVFP4 tensor in a model
/// directory's safetensors index. Does not load any bytes; produced
/// by [`validate_mistral35_inventory`] and consumed by the device
/// upload pass.
#[derive(Clone, Debug)]
pub struct Mistral35WeightInventory {
    pub num_layers: usize,
    /// Resolved per-layer NVFP4 linears, in `[layer_idx][proj_kind]`
    /// order. `proj_kind` is one of [`Mistral35LinearKind`] (in the
    /// fixed order Q, K, V, O, gate, up, down).
    pub layers: Vec<[Nvfp4LinearWeight; 7]>,
    /// Counts kept for cheap CI / diagnostics.
    pub counts: Mistral35TensorCounts,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Mistral35LinearKind {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

impl Mistral35LinearKind {
    pub const ALL: [Self; 7] = [
        Self::QProj,
        Self::KProj,
        Self::VProj,
        Self::OProj,
        Self::GateProj,
        Self::UpProj,
        Self::DownProj,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::QProj => "self_attn.q_proj",
            Self::KProj => "self_attn.k_proj",
            Self::VProj => "self_attn.v_proj",
            Self::OProj => "self_attn.o_proj",
            Self::GateProj => "mlp.gate_proj",
            Self::UpProj => "mlp.up_proj",
            Self::DownProj => "mlp.down_proj",
        }
    }

    /// Logical (N, K) shape for the given Mistral 3.5 architecture.
    /// Returns `(n, k)` in element units.
    pub fn shape_for(self, arch: &Mistral35Arch) -> Nvfp4LinearShape {
        let h = arch.text.hidden_size;
        let i = arch.text.intermediate_size;
        let q = arch.q_rows();
        let kv = arch.kv_rows();
        let (n, k) = match self {
            Self::QProj => (q, h),
            Self::KProj => (kv, h),
            Self::VProj => (kv, h),
            Self::OProj => (q, h), // o_proj projects q_dim → hidden, but with N=q,K=h convention
            Self::GateProj => (i, h),
            Self::UpProj => (i, h),
            Self::DownProj => (h, i),
        };
        Nvfp4LinearShape { n, k }
    }
}

/// Tensor-count summary used as a cheap loader smoke-check. Mirrors
/// the integration spec's "616 packed / 616 scale / 616 global / 434
/// vision / 4 projector" expectations exactly.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Mistral35TensorCounts {
    pub packed: usize,        // expected: 88 * 7 = 616
    pub scale: usize,         // expected: 616
    pub global_scale: usize,  // expected: 616
    pub vision_bf16: usize,   // expected: 434
    pub projector_bf16: usize, // expected: 4
}

impl Mistral35TensorCounts {
    pub fn expected(num_layers: usize) -> Self {
        let per_layer = Mistral35LinearKind::ALL.len();
        Self {
            packed: num_layers * per_layer,
            scale: num_layers * per_layer,
            global_scale: num_layers * per_layer,
            vision_bf16: 434,
            projector_bf16: 4,
        }
    }
}

/// Walk the index and validate every Mistral 3.5 NVFP4 tensor. Pure
/// metadata pass — does not read bytes. Returns a typed inventory
/// the loader can hand straight to the device-upload pass.
pub fn validate_mistral35_inventory(
    arch: &Mistral35Arch,
    tensors: &BTreeMap<String, TensorEntry>,
) -> Result<Mistral35WeightInventory> {
    let prefix = arch.weight_prefix.clone();
    let mut layers: Vec<[Option<Nvfp4LinearWeight>; 7]> =
        (0..arch.text.num_hidden_layers).map(|_| Default::default()).collect();

    let mut counts = Mistral35TensorCounts::default();

    for layer_idx in 0..arch.text.num_hidden_layers {
        for (kind_idx, kind) in Mistral35LinearKind::ALL.iter().enumerate() {
            let shape = kind.shape_for(arch);
            // Reject indivisible K up-front so a caller misparsing
            // the head_dim doesn't quietly succeed at validation but
            // explode later inside the GEMM.
            if shape.k % 16 != 0 {
                return Err(corrupt(format!(
                    "Mistral 3.5 layer {layer_idx} {kind:?}: K={} not divisible by 16",
                    shape.k
                )));
            }
            let base = format!("{prefix}.layers.{layer_idx}.{}", kind.name());
            let packed = require_tensor(
                tensors,
                &format!("{base}.weight_packed"),
                DType::U8,
                &[shape.n, shape.packed_cols()],
            )?;
            let scale = require_tensor(
                tensors,
                &format!("{base}.weight_scale"),
                DType::Fp8E4M3,
                &[shape.n, shape.scale_cols()],
            )?;
            let global_scale = require_tensor(
                tensors,
                &format!("{base}.weight_global_scale"),
                DType::F32,
                &[1],
            )?;
            counts.packed += 1;
            counts.scale += 1;
            counts.global_scale += 1;
            layers[layer_idx][kind_idx] = Some(Nvfp4LinearWeight {
                shape,
                packed,
                scale,
                global_scale,
            });
        }
    }

    // Vision + projector tensor counts: cheap smoke check. We don't
    // resolve every name here (those land with the Pixtral forward
    // in Step 7) — just count what's present so an obviously-wrong
    // checkpoint trips before bring-up.
    for (name, entry) in tensors.iter() {
        // Real Mistral 3.5 NVFP4 checkpoint prefixes (verified
        // against the public zdy1995love/Mistral-Medium-3.5-128B-NVFP4
        // index.json on 2026-05-08): vision tensors live under
        // `model.vision_tower.*` (NOT bare `vision_tower.*`), and
        // the projector under `model.multi_modal_projector.*`. We
        // accept both spellings so a future repackage that drops
        // the `model.` outer namespace still validates.
        if name.starts_with("model.vision_tower.") || name.starts_with("vision_tower.") {
            if entry.dtype == DType::Bf16 {
                counts.vision_bf16 += 1;
            }
        } else if name.starts_with("model.multi_modal_projector.")
            || name.starts_with("multi_modal_projector.") {
            if entry.dtype == DType::Bf16 {
                counts.projector_bf16 += 1;
            }
        }
    }

    let resolved: Vec<[Nvfp4LinearWeight; 7]> = layers
        .into_iter()
        .map(|opts| {
            // Each slot was set in the loop above; unwrap is safe but
            // we structure-preserve via Option::expect with a clear
            // message in case future refactors drop a kind.
            let arr: [Nvfp4LinearWeight; 7] = std::array::from_fn(|i| {
                opts[i].clone().expect("every linear kind populated above")
            });
            arr
        })
        .collect();

    Ok(Mistral35WeightInventory {
        num_layers: arch.text.num_hidden_layers,
        layers: resolved,
        counts,
    })
}

fn require_tensor(
    tensors: &BTreeMap<String, TensorEntry>,
    name: &str,
    expected_dtype: DType,
    expected_shape: &[usize],
) -> Result<TensorEntry> {
    let e = tensors
        .get(name)
        .cloned()
        .ok_or_else(|| missing(name))?;
    if e.dtype != expected_dtype {
        return Err(corrupt(format!(
            "Mistral 3.5 tensor {name}: dtype mismatch (got {:?}, expected {:?})",
            e.dtype, expected_dtype
        )));
    }
    if e.shape != expected_shape {
        return Err(corrupt(format!(
            "Mistral 3.5 tensor {name}: shape mismatch (got {:?}, expected {:?})",
            e.shape, expected_shape
        )));
    }
    Ok(e)
}

fn missing(name: &str) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::MissingTensor {
            name: name.to_string(),
        },
        ctx: LoaderCtx {
            path: std::path::PathBuf::from("(mistral35 inventory)"),
            tensor: Some(name.to_string()),
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

fn corrupt(detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx {
            path: std::path::PathBuf::from("(mistral35 inventory)"),
            tensor: None,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

// ─── Device-resident loaded model (post-upload) ────────────────────
//
// Populated by the rvllm-runtime side `mistral35_load`, since the
// SFB-transform step requires the CUTLASS backend and the DAG keeps
// rvllm-loader at a lower layer than rvllm-cutlass. The structs
// themselves stay here so the loader can construct the natural-
// layout intermediates without crossing the DAG line.

/// One per-layer NVFP4 linear after weight upload + SFB layout
/// transform. All pointers are absolute device addresses; the
/// caller (the bring-up) owns the arena that backs them.
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4LinearLoaded {
    pub shape: Nvfp4LinearShape,
    /// `[N, K/2]` U8 NVFP4-packed weight bytes.
    pub packed_ptr: u64,
    /// CUTLASS-interleaved E4M3 SFB scratch (sized via
    /// `cutlass_nvfp4_gemm_sm120_sfb_bytes`). Persistent for the
    /// life of the model.
    pub sfb_cutlass_ptr: u64,
    /// `[1]` F32 device scalar; passed to the GEMM epilogue's
    /// `alpha_ptr` (no host stall).
    pub global_scale_ptr: u64,
    pub packed_bytes: usize,
    pub sfb_bytes: usize,
}

/// "Outside-the-stack" weights — embedding table, final RMSNorm,
/// lm_head. Per Mistral 3.5 (`tie_word_embeddings = false`),
/// `lm_head.weight` is a separate tensor, not aliased to embed.
#[derive(Debug)]
pub struct Mistral35Outside {
    /// `[vocab=131072, hidden=12288]` BF16 (uploaded as 2-byte
    /// half-class memory; the kernel-side loader interprets it as
    /// BF16 via the matching kernel signature).
    pub embed_tokens: F16Weight,
    /// `[hidden=12288]` BF16.
    pub final_norm: F16Weight,
    /// `[vocab=131072, hidden=12288]` BF16. Untied from
    /// `embed_tokens`.
    pub lm_head: F16Weight,
}

/// Per-decoder-layer weights, post-upload + post-SFB-transform.
/// Mirrors the per-layer structure in
/// `rvllm-runtime::mistral35_layer_ref::LayerWeightsF32` but with
/// device pointers + NVFP4 representation.
#[derive(Debug)]
pub struct Mistral35LayerLoaded {
    pub input_layernorm: F16Weight,           // BF16 [hidden]
    pub post_attention_layernorm: F16Weight,  // BF16 [hidden]
    pub q_proj: Nvfp4LinearLoaded,
    pub k_proj: Nvfp4LinearLoaded,
    pub v_proj: Nvfp4LinearLoaded,
    pub o_proj: Nvfp4LinearLoaded,
    pub gate_proj: Nvfp4LinearLoaded,
    pub up_proj: Nvfp4LinearLoaded,
    pub down_proj: Nvfp4LinearLoaded,
}

impl Mistral35LayerLoaded {
    /// Iterate the per-layer NVFP4 linears in canonical
    /// `Mistral35LinearKind` order: q, k, v, o, gate, up, down.
    /// Used by the forward path to thread per-projection device
    /// pointers in step with the layer flow documented in
    /// `mistral35_layer_ref::mistral_layer_step`.
    pub fn linears(&self) -> [&Nvfp4LinearLoaded; 7] {
        [
            &self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj,
            &self.gate_proj, &self.up_proj, &self.down_proj,
        ]
    }
}

/// Fully-loaded Mistral 3.5 NVFP4 model. The CUDA forward path
/// reads device pointers directly off this struct; nothing in here
/// owns the arena, so the bring-up that built this MUST keep the
/// arena alive for the lifetime of `Mistral35LoadedModel`.
#[derive(Debug)]
pub struct Mistral35LoadedModel {
    pub outside: Mistral35Outside,
    pub layers: Vec<Mistral35LayerLoaded>,
    /// Pixtral vision tower (BF16). Optional — non-multimodal
    /// requests can run with `None`. When present, the bring-up
    /// has uploaded all 434 vision + 4 projector tensors.
    pub vision: Option<Mistral35Vision>,
}

/// One ViT block in the Pixtral vision tower (Llama-style: pre-norm,
/// SwiGLU MLP, RMSNorm). All weights BF16.
#[derive(Debug)]
pub struct VisionLayerLoaded {
    pub attention_norm: F16Weight,         // [v_hidden=1664]
    pub q_proj: F16Weight,                  // [v_hidden, v_hidden]
    pub k_proj: F16Weight,
    pub v_proj: F16Weight,
    pub o_proj: F16Weight,
    pub ffn_norm: F16Weight,                // [v_hidden]
    pub gate_proj: F16Weight,               // [v_intermediate=8192, v_hidden]
    pub up_proj: F16Weight,
    pub down_proj: F16Weight,               // [v_hidden, v_intermediate]
}

/// Pixtral vision tower + projector to text-embedding space.
/// All BF16 row-major. Patch embedding is a Conv2D (`patch_conv`)
/// of stride=patch_size=14, in_channels=3, out_channels=v_hidden.
#[derive(Debug)]
pub struct Mistral35Vision {
    /// `[v_hidden, 3, patch_size, patch_size]` BF16 conv kernel.
    pub patch_conv: F16Weight,
    /// `[v_hidden]` BF16 — pre-transformer RMSNorm.
    pub ln_pre: F16Weight,
    /// 48 transformer blocks.
    pub layers: Vec<VisionLayerLoaded>,
    /// Projector: norm → linear_1 → SiLU·linear_2-like activation
    /// → text_hidden. Exact ordering set by the projector forward
    /// kernel (TBD).
    pub projector_norm: F16Weight,
    /// `[merge_dim, v_hidden * spatial_merge_size^2]` —
    /// patch-merger linear layer.
    pub projector_patch_merger: F16Weight,
    /// `[text_hidden, merge_dim]` — first projector linear.
    pub projector_linear_1: F16Weight,
    /// `[text_hidden, text_hidden]` — second projector linear.
    pub projector_linear_2: F16Weight,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mistral35_arch::{
        Mistral35Arch, Mistral35TextArch, Mistral35VisionArch, YarnRopeConfig,
    };

    fn arch_fixture(num_layers: usize) -> Mistral35Arch {
        Mistral35Arch {
            text: Mistral35TextArch {
                num_hidden_layers: num_layers,
                hidden_size: 12288,
                intermediate_size: 28672,
                num_attention_heads: 96,
                num_key_value_heads: 8,
                head_dim: 128,
                vocab_size: 131072,
                max_position_embeddings: 262144,
                rms_norm_eps: 1e-5,
                hidden_act_silu: true,
                tie_word_embeddings: false,
                yarn: YarnRopeConfig {
                    rope_theta: 1_000_000.0,
                    original_max_position_embeddings: 4096,
                    factor: 64.0,
                    beta_fast: 4.0,
                    beta_slow: 1.0,
                    mscale: 1.0,
                    mscale_all_dim: 0.0,
                },
            },
            vision: Mistral35VisionArch {
                model_type_pixtral: true,
                hidden_size: 1664,
                num_hidden_layers: 48,
                num_attention_heads: 16,
                head_dim: 104,
                intermediate_size: 8192,
                patch_size: 14,
                image_size: 1540,
                num_channels: 3,
                rope_theta: 10_000.0,
                spatial_merge_size: 2,
            },
            image_token_index: 10,
            weight_prefix: "model.language_model".into(),
        }
    }

    fn fake_entry(name: &str, dtype: DType, shape: &[usize]) -> TensorEntry {
        TensorEntry {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            file_offset: 0,
            nbytes: 0,
        }
    }

    fn populate_layers(
        tensors: &mut BTreeMap<String, TensorEntry>,
        arch: &Mistral35Arch,
    ) {
        for layer_idx in 0..arch.text.num_hidden_layers {
            for kind in Mistral35LinearKind::ALL {
                let shape = kind.shape_for(arch);
                let base = format!("{}.layers.{}.{}", arch.weight_prefix, layer_idx, kind.name());
                tensors.insert(
                    format!("{base}.weight_packed"),
                    fake_entry(
                        &format!("{base}.weight_packed"),
                        DType::U8,
                        &[shape.n, shape.packed_cols()],
                    ),
                );
                tensors.insert(
                    format!("{base}.weight_scale"),
                    fake_entry(
                        &format!("{base}.weight_scale"),
                        DType::Fp8E4M3,
                        &[shape.n, shape.scale_cols()],
                    ),
                );
                tensors.insert(
                    format!("{base}.weight_global_scale"),
                    fake_entry(
                        &format!("{base}.weight_global_scale"),
                        DType::F32,
                        &[1],
                    ),
                );
            }
        }
    }

    #[test]
    fn shape_helpers_match_spec() {
        let arch = arch_fixture(88);
        let q = Mistral35LinearKind::QProj.shape_for(&arch);
        assert_eq!(q, Nvfp4LinearShape { n: 12288, k: 12288 });
        assert_eq!(q.packed_cols(), 6144); // K/2
        assert_eq!(q.scale_cols(), 768);   // K/16
        let k = Mistral35LinearKind::KProj.shape_for(&arch);
        assert_eq!(k, Nvfp4LinearShape { n: 1024, k: 12288 });
        let down = Mistral35LinearKind::DownProj.shape_for(&arch);
        assert_eq!(down, Nvfp4LinearShape { n: 12288, k: 28672 });
        assert_eq!(down.scale_cols(), 1792); // 28672 / 16
    }

    #[test]
    fn full_88_layer_inventory_validates() {
        let arch = arch_fixture(88);
        let mut tensors = BTreeMap::new();
        populate_layers(&mut tensors, &arch);
        // 434 vision + 4 projector dummies — enough for the cheap
        // count check, no shape validation on these here.
        for i in 0..434 {
            tensors.insert(
                format!("vision_tower.t{i}"),
                fake_entry(&format!("vision_tower.t{i}"), DType::Bf16, &[1]),
            );
        }
        for i in 0..4 {
            tensors.insert(
                format!("multi_modal_projector.p{i}"),
                fake_entry(&format!("multi_modal_projector.p{i}"), DType::Bf16, &[1]),
            );
        }

        let inv = validate_mistral35_inventory(&arch, &tensors).expect("valid");
        assert_eq!(inv.num_layers, 88);
        assert_eq!(inv.layers.len(), 88);
        assert_eq!(inv.counts, Mistral35TensorCounts::expected(88));
    }

    #[test]
    fn missing_packed_tensor_reports_named_error() {
        let arch = arch_fixture(2);
        let mut tensors = BTreeMap::new();
        populate_layers(&mut tensors, &arch);
        let key = format!(
            "{}.layers.1.{}.weight_packed",
            arch.weight_prefix,
            Mistral35LinearKind::QProj.name()
        );
        tensors.remove(&key);
        let err = validate_mistral35_inventory(&arch, &tensors).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("weight_packed"));
        assert!(msg.contains("layers.1"));
    }

    #[test]
    fn shape_mismatch_rejects() {
        let arch = arch_fixture(1);
        let mut tensors = BTreeMap::new();
        populate_layers(&mut tensors, &arch);
        let key = format!(
            "{}.layers.0.{}.weight_scale",
            arch.weight_prefix,
            Mistral35LinearKind::QProj.name()
        );
        let bad = fake_entry(&key, DType::Fp8E4M3, &[12288, 100]); // wrong K/16
        tensors.insert(key, bad);
        let err = validate_mistral35_inventory(&arch, &tensors).unwrap_err();
        assert!(format!("{err:?}").contains("shape mismatch"));
    }

    #[test]
    fn dtype_mismatch_rejects() {
        let arch = arch_fixture(1);
        let mut tensors = BTreeMap::new();
        populate_layers(&mut tensors, &arch);
        let key = format!(
            "{}.layers.0.{}.weight_packed",
            arch.weight_prefix,
            Mistral35LinearKind::DownProj.name()
        );
        // Replace with wrong dtype (F16 instead of U8).
        let n = 12288;
        let k_packed = 28672 / 2;
        tensors.insert(
            key.clone(),
            fake_entry(&key, DType::F16, &[n, k_packed]),
        );
        let err = validate_mistral35_inventory(&arch, &tensors).unwrap_err();
        assert!(format!("{err:?}").contains("dtype mismatch"));
    }

    #[test]
    fn expected_counts_for_88_layers() {
        let c = Mistral35TensorCounts::expected(88);
        assert_eq!(c.packed, 616);
        assert_eq!(c.scale, 616);
        assert_eq!(c.global_scale, 616);
        assert_eq!(c.vision_bf16, 434);
        assert_eq!(c.projector_bf16, 4);
    }

    #[test]
    fn inventory_counts_partial_vision_dummies() {
        let arch = arch_fixture(1);
        let mut tensors = BTreeMap::new();
        populate_layers(&mut tensors, &arch);
        for i in 0..3 {
            tensors.insert(
                format!("vision_tower.x{i}"),
                fake_entry(&format!("vision_tower.x{i}"), DType::Bf16, &[1]),
            );
        }
        let inv = validate_mistral35_inventory(&arch, &tensors).expect("valid");
        assert_eq!(inv.counts.vision_bf16, 3);
        assert_eq!(inv.counts.projector_bf16, 0);
        assert_eq!(inv.counts.packed, 7);
    }
}
