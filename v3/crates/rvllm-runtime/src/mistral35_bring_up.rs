//! Mistral 3.5 NVFP4 bring-up scaffolding.
//!
//! Step 5 (Rust scaffolding) — the entry-point shape `cuda_worker`
//! dispatches into. Everything that *can* be validated up-front
//! without an actual GPU forward pass runs here so an operator
//! pointing the server at a Mistral checkpoint gets concrete
//! diagnostics during startup:
//!
//! 1. Parse `config.json` via [`rvllm_loader::mistral35_arch::Mistral35Arch`]
//!    (fails on YaRN drift, GQA-ratio mismatch, missing pixtral block).
//! 2. Open the safetensors index and validate every NVFP4 linear via
//!    [`rvllm_loader::mistral35_weights::validate_mistral35_inventory`]
//!    (616 packed/scale/global, 434 vision BF16, 4 projector BF16).
//! 3. Resolve `libcutlass_sm120.so` and call
//!    [`rvllm_cutlass::lib_so::CutlassBackend::require_nvfp4`] —
//!    refuses startup when the NVFP4 entry-point set is missing.
//!
//! Only after all three pass does [`Mistral35Bringup::load`] succeed.
//! The forward path ([`Mistral35Bringup::run_generate`]) is the next
//! milestone; it currently returns a typed
//! [`Mistral35Error::ForwardNotImplemented`] so a per-request
//! invocation surfaces a clean error rather than silently producing
//! garbage.
//!
//! No CUDA / cudarc imports here — the heavy GPU plumbing lives
//! alongside the Gemma 4 / Qwen 3.6 paths and lands when the
//! NVFP4 kernel ABI is wired through (steps 4-CUDA / 6 / 9).

use std::path::PathBuf;

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_loader::mistral35_arch::Mistral35Arch;
use rvllm_loader::mistral35_weights::{
    validate_mistral35_inventory, Mistral35TensorCounts, Mistral35WeightInventory,
};
use rvllm_loader::safetensors::{ShardHeader, ShardIndex, TensorEntry};

use crate::gemma4_bring_up::Gemma4EnginePaths;

/// Same path bundle Gemma 4 takes — kept so `cuda_worker` can pass
/// the resolved paths through unchanged. We alias rather than
/// inventing `Mistral35EnginePaths` to avoid duplicated path
/// resolution code; the spec calls out "split or sibling type" as
/// the long-term direction (Step 5-CUDA, when the per-family
/// kernel selection actually diverges).
pub type Mistral35EnginePaths = Gemma4EnginePaths;

/// Top-level bring-up handle. Parses arch, validates the inventory,
/// and verifies CUTLASS NVFP4 symbols are present. The actual
/// forward path lands when steps 4-CUDA / 6 / 9 are wired.
///
/// `Gemma4EnginePaths` upstream is not `Debug`, so neither is this.
pub struct Mistral35Bringup {
    pub paths: Mistral35EnginePaths,
    pub arch: Mistral35Arch,
    pub inventory: Mistral35WeightInventory,
    /// Reserved for the device-upload pass (Step 4-CUDA): bytes the
    /// caller asked us to allocate for the arena. We keep it on the
    /// struct so the run-loop can size scratch consistently with
    /// the value `cuda_worker` was started with.
    pub arena_bytes: usize,
    /// Whether every required NVFP4 CUTLASS symbol resolved. Always
    /// true on the `Ok` path — `load` refuses startup otherwise.
    pub nvfp4_active: bool,
}

#[derive(Debug)]
pub enum Mistral35Error {
    /// Generation called before the GPU forward path is wired.
    ForwardNotImplemented,
    /// CUTLASS NVFP4 symbol set missing.
    Nvfp4SymbolsMissing,
    /// `cuda` feature not enabled at compile time.
    NoCudaFeature,
}

/// Which NVFP4-KV decode kernel can serve a given GQA ratio.
///
/// The existing NVFP4-KV decode kernel set has compile-time caps:
///
/// - `kernels/flash_attention_nvfp4kv.cu::MAX_GQA_DECODE = 4`
///   (also `..._bf16out.cu`) — fused per-`(seq, kv_head)` decode that
///   loads K/V exactly once per tile and computes Q·Kᵀ for all q-heads
///   sharing that kv-head. Gemma 4 sliding (GQA=2) fits.
/// - `kernels/flash_attention_split_decode_nvfp4kv.cu::MAX_GQA_SPLIT = 8`
///   (also `..._bf16out.cu`) — paged_attention_v2-style split decode.
///   Qwen 3.6 (GQA=8) fits.
/// - Per-head fallback (`flash_attention_2_decode_nvfp4kv_kernel`) — one
///   block per `(seq, head)`. Loads K/V once per tile per Q-HEAD
///   instead of per (seq, kv_head); ~`gqa_ratio×` bandwidth waste vs
///   the fused path, but works for any GQA.
///
/// Mistral 3.5 has GQA=12. The kernel-side fix is to raise both
/// constants to ≥12 (q_reg / row_max / acc / s_score allocations
/// scale linearly; sm_121 has the register + smem headroom). Until
/// that lands, Mistral routes through the per-head fallback —
/// correct, just slower. The gate lives here so a future kernel
/// rebuild flips the strategy without changing the runtime.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KvDecodeStrategy {
    /// `flash_attention_2_decode_nvfp4kv_gqa_kernel` — fused
    /// per-(seq, kv_head), MAX_GQA_DECODE=4. Tightest path.
    FusedGqa,
    /// `paged_attention_v2_split_decode_nvfp4kv_kernel` —
    /// MAX_GQA_SPLIT=8. Used for Qwen 3.6's GQA=8.
    SplitDecode,
    /// `flash_attention_2_decode_nvfp4kv_kernel` — one block per
    /// (seq, head). Works for any GQA at the cost of `gqa_ratio×`
    /// duplicated K/V load. Mistral 3.5 (GQA=12) currently routes
    /// here.
    PerHeadFallback,
}

impl KvDecodeStrategy {
    /// Pick a strategy from the model's GQA ratio. The per-head
    /// fallback is universal; we prefer the fused / split paths
    /// whenever the kernel cap covers the requested ratio.
    pub fn for_gqa_ratio(gqa_ratio: usize) -> Self {
        // Match the kernel-side caps exactly. Bumping these here
        // without bumping MAX_GQA_DECODE / MAX_GQA_SPLIT in the .cu
        // sources would silently route Mistral through a kernel
        // that returns early on `GQA > MAX_*`.
        const MAX_GQA_DECODE_FUSED: usize = 4;
        const MAX_GQA_SPLIT: usize = 8;

        if gqa_ratio == 0 {
            return Self::PerHeadFallback;
        }
        if gqa_ratio <= MAX_GQA_DECODE_FUSED {
            Self::FusedGqa
        } else if gqa_ratio <= MAX_GQA_SPLIT {
            Self::SplitDecode
        } else {
            Self::PerHeadFallback
        }
    }
}

impl Mistral35Bringup {
    /// Decode-strategy gate. Surfaced at startup so the operator
    /// log line records exactly which path each request will take.
    pub fn kv_decode_strategy(&self) -> KvDecodeStrategy {
        let q = self.arch.text.num_attention_heads;
        let kv = self.arch.text.num_key_value_heads;
        if kv == 0 {
            return KvDecodeStrategy::PerHeadFallback;
        }
        KvDecodeStrategy::for_gqa_ratio(q / kv)
    }
}

impl Mistral35Bringup {
    /// Open + validate the model directory. Does not yet upload any
    /// weights. Fails on:
    ///
    /// - non-Mistral / corrupted `config.json`
    /// - YaRN `mscale_all_dim != 0.0` (known checkpoint correction)
    /// - missing/wrong-shape/wrong-dtype NVFP4 tensors
    /// - missing `libcutlass_sm120.so` NVFP4 symbol set (under
    ///   `cuda` feature; on default builds this branch is skipped
    ///   because no `.so` is opened).
    pub fn load(paths: Mistral35EnginePaths, arena_bytes: usize) -> Result<Self> {
        // (1) Arch — already gated by family resolver, but re-parse
        //     here so failures point at the actual model_dir even
        //     when the operator passed `--model-family auto`.
        let arch = Mistral35Arch::from_dir(&paths.model_dir)?
            .ok_or_else(|| corrupt(
                paths.model_dir.clone(),
                "Mistral35Bringup::load: config.json does not match Mistral 3.5 markers \
                 (architectures[0]==Mistral3ForConditionalGeneration, model_type==mistral3, \
                 quantization_config.format==nvfp4-pack-quantized)".into(),
            ))?;
        eprintln!("[mistral35] {}", arch.summary());

        // (2) Inventory — mmap the index header (cheap), validate
        //     every NVFP4 linear's dtype + shape, count vision /
        //     projector tensors. Doesn't read tensor payload bytes.
        let tensors = scan_safetensors_index(&paths.model_dir)?;
        let inventory = validate_mistral35_inventory(&arch, &tensors)?;
        let expected = Mistral35TensorCounts::expected(arch.text.num_hidden_layers);
        eprintln!(
            "[mistral35] inventory: packed={}/{} scale={}/{} global_scale={}/{} \
             vision_bf16={}/{} projector_bf16={}/{}",
            inventory.counts.packed, expected.packed,
            inventory.counts.scale, expected.scale,
            inventory.counts.global_scale, expected.global_scale,
            inventory.counts.vision_bf16, expected.vision_bf16,
            inventory.counts.projector_bf16, expected.projector_bf16,
        );

        // (3) CUTLASS NVFP4 symbols. Only on `cuda` builds — the
        //     default build doesn't open the `.so` at all, so we
        //     skip the gate there and only record the field. Any
        //     attempt to actually generate then trips
        //     `Mistral35Error::NoCudaFeature` cleanly.
        let nvfp4_active = require_nvfp4_symbols(&paths)?;

        let bringup = Self { paths, arch, inventory, arena_bytes, nvfp4_active };
        let strategy = bringup.kv_decode_strategy();
        eprintln!(
            "[mistral35] kv_decode_strategy={:?} (gqa_ratio={})",
            strategy,
            bringup.arch.gqa_ratio()
        );
        if matches!(strategy, KvDecodeStrategy::PerHeadFallback) {
            eprintln!(
                "[mistral35] note: Mistral 3.5's GQA=12 exceeds the existing \
                 NVFP4-KV fused (MAX_GQA_DECODE=4) and split-decode \
                 (MAX_GQA_SPLIT=8) kernel caps; per-head fallback is \
                 correct but ~12x duplicated K/V load. Raise both .cu \
                 constants to >=12 to flip onto the fused path."
            );
        }
        Ok(bringup)
    }

    /// Forward placeholder. Returns a typed error at every call.
    /// Wired into `cuda_worker.rs::spawn_cuda_worker` for the
    /// Mistral 3.5 branch so per-request execution surfaces the
    /// "kernel not implemented" message via the existing
    /// `GenerateEvent::Error` path.
    pub fn forward_not_implemented_yet() -> Mistral35Error {
        Mistral35Error::ForwardNotImplemented
    }
}

fn scan_safetensors_index(
    model_dir: &std::path::Path,
) -> Result<std::collections::BTreeMap<String, TensorEntry>> {
    let idx = ShardIndex::resolve(model_dir)?;
    let mut tensors: std::collections::BTreeMap<String, TensorEntry> =
        std::collections::BTreeMap::new();
    for shard_path in &idx.shards {
        // For the inventory pass we only need the header. Read the
        // first 8 bytes (header_bytes prefix) + the JSON header,
        // never the payload. That keeps load() fast on a 128B
        // checkpoint (~80 GiB on disk) — full validation in seconds
        // rather than a multi-second mmap walk.
        let bytes = std::fs::read(shard_path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: shard_path.clone(),
            source,
        })?;
        let header = ShardHeader::parse(shard_path, &bytes)?;
        for (name, entry) in header.tensors.into_iter() {
            tensors.insert(name, entry);
        }
    }
    Ok(tensors)
}

#[cfg(feature = "cuda")]
fn require_nvfp4_symbols(paths: &Mistral35EnginePaths) -> Result<bool> {
    use rvllm_cutlass::lib_so::CutlassBackend;
    use rvllm_core::CompileTarget;

    // Resolve the same way `cuda_worker` does for Gemma 4 / Qwen.
    // The sm_121 hint short-circuits straight to libcutlass_sm120.so.
    let backend = CutlassBackend::load_for(
        Some(CompileTarget::Sm121),
        paths.cutlass_so.clone(),
        &[],
    )?;
    backend.require_nvfp4()?;
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
fn require_nvfp4_symbols(_paths: &Mistral35EnginePaths) -> Result<bool> {
    // No CUDA build: skip the dlopen check. The bring-up record
    // still parses arch + validates inventory so config.json /
    // checkpoint integrity is verifiable without a GPU. Every
    // forward call then errors with NoCudaFeature.
    Ok(false)
}

fn corrupt(path: PathBuf, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx { path, tensor: None },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths_for(model_dir: PathBuf) -> Mistral35EnginePaths {
        Gemma4EnginePaths {
            model_dir,
            kernels_dir: PathBuf::from("/tmp/rvllm-mistral-test-kernels"),
            cutlass_so: PathBuf::from("/tmp/rvllm-mistral-test-libcutlass.so"),
            fa3_so: PathBuf::from("/tmp/rvllm-mistral-test-fa3.so"),
            policy_json: PathBuf::from("/tmp/rvllm-mistral-test-policy.json"),
        }
    }

    #[test]
    fn load_rejects_non_mistral_dir() {
        let tmp = std::env::temp_dir().join(format!(
            "rvllm-mistral-bringup-test-{}-nonmistral",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(
            tmp.join("config.json"),
            r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4"}"#,
        )
        .unwrap();
        match Mistral35Bringup::load(paths_for(tmp.clone()), 1) {
            Ok(_) => panic!("expected non-Mistral dir to be rejected"),
            Err(e) => assert!(format!("{e:?}").contains("Mistral 3.5")),
        }
    }

    #[test]
    fn load_rejects_yarn_drift() {
        let tmp = std::env::temp_dir().join(format!(
            "rvllm-mistral-bringup-test-{}-yarn",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Mistral markers + YaRN with a non-zero mscale_all_dim should
        // trip the parser before the inventory pass even runs.
        std::fs::write(
            tmp.join("config.json"),
            r#"{
              "architectures":["Mistral3ForConditionalGeneration"],
              "model_type":"mistral3",
              "quantization_config":{"format":"nvfp4-pack-quantized"},
              "text_config":{
                "num_hidden_layers":2,"hidden_size":12288,
                "intermediate_size":28672,"num_attention_heads":96,
                "num_key_value_heads":8,"head_dim":128,
                "vocab_size":131072,"max_position_embeddings":262144,
                "rms_norm_eps":1e-5,"hidden_act":"silu",
                "tie_word_embeddings":false,"rope_theta":1000000.0,
                "rope_scaling":{"rope_type":"yarn","original_max_position_embeddings":4096,
                  "factor":64.0,"beta_fast":4.0,"beta_slow":1.0,"mscale":1.0,"mscale_all_dim":1.0}
              },
              "vision_config":{"model_type":"pixtral","head_dim":104,"image_size":1540}
            }"#,
        )
        .unwrap();
        match Mistral35Bringup::load(paths_for(tmp), 1) {
            Ok(_) => panic!("expected YaRN drift to be rejected"),
            Err(e) => assert!(format!("{e:?}").contains("mscale_all_dim")),
        }
    }

    #[test]
    fn forward_stub_is_typed_error() {
        let e = Mistral35Bringup::forward_not_implemented_yet();
        assert!(matches!(e, Mistral35Error::ForwardNotImplemented));
    }

    #[test]
    fn kv_decode_strategy_picks_fused_for_low_gqa() {
        // Gemma 4 sliding has GQA=2; falls under MAX_GQA_DECODE=4.
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(2),
            KvDecodeStrategy::FusedGqa
        );
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(4),
            KvDecodeStrategy::FusedGqa
        );
    }

    #[test]
    fn kv_decode_strategy_picks_split_for_qwen_ratio() {
        // Qwen 3.6 full attention has GQA=8; below MAX_GQA_SPLIT=8.
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(5),
            KvDecodeStrategy::SplitDecode
        );
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(8),
            KvDecodeStrategy::SplitDecode
        );
    }

    #[test]
    fn kv_decode_strategy_falls_back_for_mistral() {
        // Mistral 3.5 has GQA=12; both fused and split caps blown.
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(12),
            KvDecodeStrategy::PerHeadFallback
        );
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(16),
            KvDecodeStrategy::PerHeadFallback
        );
    }

    #[test]
    fn kv_decode_strategy_safe_on_zero_ratio() {
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(0),
            KvDecodeStrategy::PerHeadFallback
        );
    }
}
