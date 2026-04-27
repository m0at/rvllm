//! Gemma 4 layer forward -- 14 kernel launches per layer.
//!
//! Differs from the Llama/Qwen path (layer_exec.rs) in:
//!   - 4 norms per layer (input, post_attn, pre_ff, post_ff)
//!   - QK-norm (RMSNorm on Q and K heads before RoPE)
//!   - v_norm (parameter-free RMS norm on V after projection)
//!   - GELU(tanh) activation instead of SiLU
//!   - Partial RoPE (only rotate first rotary_dim dims per head)
//!   - Per-layer KV head count (sliding vs global)
//!   - head_dim = 256 (requires FA3 .so compiled for 256)
//!   - Per-layer learnable scalar (applied ONCE after both sub-blocks)
//!
//! Launch sequence:
//!   1.  fused_rmsnorm_fp8_quant          input_layernorm
//!   2.  fp8_gemm (cuBLASLt)             Q||K||V projection
//!  2b.  vnorm_f16                       parameter-free RMS norm on V
//!   3.  fused_qk_rmsnorm                QK-norm on Q and K heads
//!   4.  fused_rope_partial_fp8kv        partial RoPE + FP8 Q + paged KV
//!   5.  paged_decode / paged_prefill    FA3 attention (head_dim=256)
//!   6.  quantize_fp8_per_token          attn_out -> fp8
//!   7.  fp8_gemm_residual (cuBLASLt)    O proj += residual
//!   8.  fused_rmsnorm                   post_attention_layernorm (norm only)
//!   9.  fused_rmsnorm_fp8_quant         pre_feedforward_layernorm
//!  10.  fp8_gemm (cuBLASLt)             gate||up projection
//!  11.  fused_gelu_mul_fp8_quant        GELU(tanh)(gate) * up -> FP8
//!  12.  fp8_gemm_residual (cuBLASLt)    down proj += residual
//!  13.  fused_rmsnorm                   post_feedforward_layernorm (norm only)
//!  14.  residual_scale_f16              residual *= layer_scalar (once)

use rvllm_core::Result;
use rvllm_cutlass::{CublasLt, CutlassBackend, Fp8GemmPlan};
use rvllm_fused::gemma4_launcher;
use rvllm_fused::FusedRmsnormFp8QuantLaunch;
use rvllm_kernels::KernelFn;

use rvllm_attention::{AttentionBackend, PagedDecodeFp8Launcher, PagedDecodeParams};

use rvllm_loader::gemma4_arch::Gemma4LayerType;

/// Storage dtype of the paged KV cache. Picked once at bring-up time
/// per the `RVLLM_NVFP4_KV` / `RVLLM_F16_KV` env-flag priority and
/// threaded through on every attention launch.
///
///   * `F16`  — 2 bytes per element, dense. CPU-host-visible path,
///              still the engine default when no env flag is set.
///   * `Fp8`  — 1 byte per element, per-tensor E4M3 descale. Current
///              sm_121 hot path (`RVLLM_F16_KV=0`).
///   * `Nvfp4`— 4 bits per element packed two-per-byte + 1 E4M3
///              microscale per 16 elements. `RVLLM_NVFP4_KV=1`.
///              Storage = elements × 0.5 bytes, scale region =
///              elements × 0.0625 bytes, total 4.5 effective bits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KvDtype {
    F16,
    Fp8,
    Nvfp4,
}

impl KvDtype {
    /// Main-cache bytes per element (scale region is separate).
    #[must_use]
    pub fn cache_bytes_per_elem(&self) -> u32 {
        match self {
            KvDtype::F16 => 2,
            KvDtype::Fp8 => 1,
            // Packed 4-bit → 0.5 byte per element. Callers that need
            // an integer byte count should compute on multiples of 16
            // elements (which NVFP4 requires anyway for the scale
            // block). Returned here as `0` to force the caller to
            // explicitly use `nvfp4_total_bytes` below.
            KvDtype::Nvfp4 => 0,
        }
    }

    /// Scale-region bytes per element. 0 for F16/Fp8 (per-tensor
    /// descale lives in a separate 4-byte f32). For NVFP4, 1 E4M3
    /// scale per 16 elements → 1/16 byte/elem.
    #[must_use]
    pub fn scale_bytes_per_elem_x16(&self) -> u32 {
        match self {
            KvDtype::Nvfp4 => 1,
            _ => 0,
        }
    }

    /// Is this an NVFP4 packed layout?
    #[must_use]
    pub fn is_nvfp4(&self) -> bool {
        matches!(self, KvDtype::Nvfp4)
    }

    /// Is the cache dense F16 (no descale)?
    #[must_use]
    pub fn is_f16(&self) -> bool {
        matches!(self, KvDtype::F16)
    }

    /// Resolve from the engine env flags. Precedence (post-Phase-2b
    /// of task aa01001nvf4f16mma, commit 8e8f517 — NVFP4 + f16-MMA
    /// unified prefill is 13× faster than the per-qi kernel and
    /// beats FP8 unified prefill at 1082-tok prompts, so NVFP4 is
    /// now the default KV dtype):
    ///
    ///   1. `RVLLM_NVFP4_KV=1`    → Nvfp4 (explicit opt-in).
    ///   2. `RVLLM_F16_KV=1`      → F16.
    ///   3. `RVLLM_FP8_KV=1`      → Fp8.
    ///   4. `RVLLM_NVFP4_KV=0`    → F16 (explicit opt-out of the
    ///                              NVFP4 default — matches the
    ///                              pre-2b engine default for
    ///                              legacy tests / probes).
    ///   5. unset → Nvfp4.
    ///
    /// `f16_only` short-circuits to F16 — used by the bench path
    /// when the caller explicitly asks for f16 regardless of env.
    #[must_use]
    pub fn from_env(f16_only: bool) -> Self {
        if f16_only {
            return KvDtype::F16;
        }
        if crate::gemma4_bring_up::parse_truthy_env("RVLLM_NVFP4_KV").unwrap_or(false) {
            return KvDtype::Nvfp4;
        }
        if crate::gemma4_bring_up::parse_truthy_env("RVLLM_F16_KV").unwrap_or(false) {
            return KvDtype::F16;
        }
        if crate::gemma4_bring_up::parse_truthy_env("RVLLM_FP8_KV").unwrap_or(false) {
            return KvDtype::Fp8;
        }
        if crate::gemma4_bring_up::parse_truthy_env("RVLLM_NVFP4_KV") == Some(false) {
            return KvDtype::F16;
        }
        KvDtype::Nvfp4
    }

    /// Hybrid per-layer KV dtype for Gemma 4. When
    /// `RVLLM_NVFP4_HYBRID_GLOBAL_FP8=1` is set, global-attention layers
    /// (head_dim=512, see-everything) use FP8 while sliding-window
    /// layers stay on the env-default (typically NVFP4). Hypothesis H1
    /// for the long-context Rusty-persona garbage: outlier-channel
    /// pressure hits the 10 global layers harder than the 50 sliding
    /// layers; FP8 on globals + NVFP4 on slidings should rescue
    /// quality while still saving most of the KV memory.
    ///
    /// When the env flag is absent or `0`, returns `from_env(f16_only)`
    /// for every layer (uniform mode, default behaviour).
    #[must_use]
    pub fn for_layer_or_env(
        layer_type: rvllm_loader::gemma4_arch::Gemma4LayerType,
        f16_only: bool,
    ) -> Self {
        Self::for_layer_index_or_env(layer_type, /*layer_idx=*/usize::MAX, f16_only)
    }

    /// Layer-index-aware variant. Adds a comma-separated list env var
    /// `RVLLM_FP8_KV_LAYERS=0,30,59` that forces specific layer indices
    /// to FP8 KV when the default would have been NVFP4. Strictly
    /// generalizes `RVLLM_NVFP4_HYBRID_GLOBAL_FP8` (which forces the
    /// 10 global layers to FP8). Cycle 24: targeted-FP8 attack on
    /// high-divergence layers identified empirically.
    ///
    /// `layer_idx == usize::MAX` is the "unknown index" sentinel — only
    /// the global-fp8 hybrid env applies, list env is ignored. Call
    /// sites with the index available should pass the real index.
    #[must_use]
    pub fn for_layer_index_or_env(
        layer_type: rvllm_loader::gemma4_arch::Gemma4LayerType,
        layer_idx: usize,
        f16_only: bool,
    ) -> Self {
        let default = Self::from_env(f16_only);
        // Env-list override fires first. Most targeted, lets operators
        // pin specific layer indices to FP8 even if hybrid is off.
        if default == KvDtype::Nvfp4 && layer_idx != usize::MAX {
            if let Ok(list) = std::env::var("RVLLM_FP8_KV_LAYERS") {
                if list.split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .any(|i| i == layer_idx)
                {
                    return KvDtype::Fp8;
                }
            }
        }
        let hybrid = crate::gemma4_bring_up::parse_truthy_env("RVLLM_NVFP4_HYBRID_GLOBAL_FP8")
            .unwrap_or(false);
        if hybrid && default == KvDtype::Nvfp4
            && layer_type == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention
        {
            return KvDtype::Fp8;
        }
        // Inverse hybrid (cycle 25 research): all SLIDING layers FP8,
        // globals stay NVFP4. Tests whether sliding-layer cumulative
        // noise dominates over global-layer noise, given that the prior
        // global-FP8 hybrid did not close the WEATHER cliff.
        let hybrid_sliding =
            crate::gemma4_bring_up::parse_truthy_env("RVLLM_NVFP4_HYBRID_SLIDING_FP8")
                .unwrap_or(false);
        if hybrid_sliding && default == KvDtype::Nvfp4
            && layer_type == rvllm_loader::gemma4_arch::Gemma4LayerType::SlidingAttention
        {
            return KvDtype::Fp8;
        }
        default
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerDims {
    pub num_tokens: u32,
    pub hidden: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub intermediate: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    /// Current-step upper bound on KV context length across the active
    /// batch, when the caller can compute it cheaply (e.g. decode step
    /// knows `step + 1`). Used by the split-KV decode dispatch to skip
    /// the split path when the current context is short enough that
    /// the single-CTA kernel wins, without conflating it with the
    /// bucket max that still sizes workspace. `None` falls back to
    /// `max_blocks_per_seq * block_size` — correct but conservative.
    pub current_max_context_len: Option<u32>,
    pub attn_scale: f32,
    pub rms_eps: f32,
    pub layer_type: Gemma4LayerType,
    pub sliding_window: u32,
    /// Retained for existing branches that still key on the bool
    /// (rope_f16kv vs rope_fp8kv). Equivalent to `kv_dtype.is_f16()`.
    pub f16_kv: bool,
    pub kv_dtype: KvDtype,
    /// Cycle 54 Stage 1: when true, the residual buffer is interpreted
    /// as bf16 (still 2 bytes per element, same allocation). Set by
    /// the engine when `RVLLM_RESIDUAL_BF16=1`. The math inside each
    /// rmsnorm + residual-add is unchanged (f32 internally); only the
    /// inter-layer storage dtype differs. f16 vs bf16 trade 3 mantissa
    /// bits for 3 exponent bits — bf16 keeps long-context cumulative
    /// residual values from saturating/underflowing across 60 layers.
    /// Matches vLLM's bf16-activation production setup.
    pub bf16_residual: bool,
}

impl Gemma4LayerDims {
    /// Cycle 53 hardening: catch plumbing misconfigurations before they
    /// reach the kernel layer. The attention launchers already validate
    /// their own params, but several invariants live at the *layer*
    /// level — across attn + RoPE + cache layout — and previously had
    /// to be caught either by visual review of the call sites or by
    /// kernel-side OOB reads (silent garbage on long context). This
    /// runs once per `gemma4_forward_phase` call; cost is negligible
    /// vs the kernel launches that follow.
    ///
    /// Failure converts to `AttentionError::FeatureNotAvailable` with
    /// a descriptive `op` so the source of the misconfiguration is
    /// obvious in logs.
    pub fn validate(&self) -> Result<()> {
        let mk_err = |op: &'static str| -> rvllm_core::RvllmError {
            rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    backend: "Gemma4LayerDims",
                    op,
                },
                ctx: rvllm_core::AttnCtx {
                    op: "gemma4_forward_phase.validate",
                    stream: 0,
                    num_seqs: self.num_tokens,
                    head_dim: self.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            }
        };
        if self.num_tokens == 0 {
            return Err(mk_err("Gemma4LayerDims: num_tokens == 0"));
        }
        if self.hidden == 0 || self.num_heads == 0 || self.num_kv_heads == 0 {
            return Err(mk_err(
                "Gemma4LayerDims: hidden/num_heads/num_kv_heads must be > 0",
            ));
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(mk_err(
                "Gemma4LayerDims: num_heads not divisible by num_kv_heads",
            ));
        }
        if self.head_dim == 0 || self.head_dim > 1024 {
            // 1024 is well above any real Gemma 4 head_dim (256 / 512);
            // catches u32 underflow / wild values.
            return Err(mk_err("Gemma4LayerDims: head_dim out of range"));
        }
        if self.rotary_dim > self.head_dim {
            return Err(mk_err("Gemma4LayerDims: rotary_dim > head_dim"));
        }
        if self.rotary_dim % 2 != 0 {
            // RoPE pairs (cos, sin) over even/odd channel pairs.
            return Err(mk_err("Gemma4LayerDims: rotary_dim must be even"));
        }
        if self.block_size == 0 || (self.block_size & (self.block_size - 1)) != 0 {
            // Power-of-two block_size is what the page-table arithmetic
            // assumes; non-PoT silently mis-indexes pages on long ctx.
            return Err(mk_err(
                "Gemma4LayerDims: block_size must be a non-zero power of two",
            ));
        }
        if self.max_blocks_per_seq == 0 || self.num_blocks_total == 0 {
            return Err(mk_err(
                "Gemma4LayerDims: max_blocks_per_seq / num_blocks_total must be > 0",
            ));
        }
        if let Some(cur) = self.current_max_context_len {
            let bucket_max = (self.max_blocks_per_seq as u64)
                .saturating_mul(self.block_size as u64);
            if (cur as u64) > bucket_max {
                // Decode reads block_tables[seq * max_blocks_per_seq +
                // (cur-1)/block_size]. If cur exceeds the bucket,
                // that's an OOB read on the page-table that would
                // silently feed garbage block IDs to the kernel.
                return Err(mk_err(
                    "Gemma4LayerDims: current_max_context_len exceeds max_blocks_per_seq*block_size",
                ));
            }
        }
        if self.intermediate == 0 {
            return Err(mk_err("Gemma4LayerDims: intermediate (MLP dim) == 0"));
        }
        if !self.attn_scale.is_finite() || self.attn_scale <= 0.0 {
            return Err(mk_err("Gemma4LayerDims: attn_scale must be finite and > 0"));
        }
        if !self.rms_eps.is_finite() || self.rms_eps <= 0.0 {
            return Err(mk_err("Gemma4LayerDims: rms_eps must be finite and > 0"));
        }
        // f16_kv must agree with kv_dtype — drift between these two
        // routes the wrong RoPE kernel (rope_f16kv vs rope_fp8kv vs
        // rope_nvfp4kv) and KV cache writes go to the wrong layout.
        if self.f16_kv != self.kv_dtype.is_f16() {
            return Err(mk_err(
                "Gemma4LayerDims: f16_kv flag disagrees with kv_dtype.is_f16()",
            ));
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerWeightPtrs {
    pub attn_norm_gamma: u64,
    pub post_attn_norm_gamma: u64,
    pub pre_ff_norm_gamma: u64,
    pub post_ff_norm_gamma: u64,
    pub q_norm_gamma: u64,
    pub k_norm_gamma: u64,
    pub qkv_fp8: u64,
    pub qkv_scale: u64,
    pub o_fp8: u64,
    pub o_scale: u64,
    pub gate_up_fp8: u64,
    pub gate_up_scale: u64,
    pub down_fp8: u64,
    pub down_scale: u64,
    pub layer_scalar_ptr: u64, // [1] f16, per-layer residual multiplier
    pub qkv_f16: u64,          // 0 = use FP8, nonzero = use F16 GEMM
    pub o_f16: u64,
    pub gate_up_f16: u64,
    pub down_f16: u64,
    pub qkv_chscale: u64, // 0 = scalar scale, nonzero = per-channel f32 vec
    pub o_chscale: u64,
    pub gate_up_chscale: u64,
    pub down_chscale: u64,
    /// 2-D blockscale tensor `[N_blocks, K_blocks]` f32 on device.
    /// `0` when the weight's source scale was per-row (or for
    /// synthesized fused qkv/gate_up — their per-part block
    /// alignments don't compose cleanly into a single 2-D tensor).
    /// Only consumed by kernels whose ABI expects the full 2-D
    /// shape (`Fp8GemvF16InLaunch`, CUTLASS SFB).  When `0`, any
    /// such caller MUST fall back to the channelscale-preserving
    /// path — reading `*_chscale` as 2-D produces garbage (walks
    /// off the end of the per-row vec).
    pub qkv_blockscale: u64,
    pub o_blockscale: u64,
    pub gate_up_blockscale: u64,
    pub down_blockscale: u64,
    /// Cycle 45 step 4.5b: per-layer AWQ INT4 W4A16 weights. All-zero
    /// fields = AWQ inactive for this linear, fall back to the FP8 path
    /// using the `*_fp8` / `*_scale` / `*_blockscale` fields above.
    /// Non-zero `*_packed` = AWQ active for that linear; the dispatch
    /// in `exec_layer` reads `awq` and routes to `AwqInt4GemvF16Launch`
    /// instead of `Fp8GemvF16InLaunch`.
    ///
    /// Compressed-tensors AWQ stores Q/K/V un-fused, so QKV is split
    /// into three (q/k/v) independent launches that write into the
    /// existing Q|K|V scratch buffer at q_dim / kv_dim offsets;
    /// gate/up are similarly split into (gate, up). The down + o
    /// projections stay 1-launch.
    pub awq: Gemma4AwqLayerPtrs,
}

/// Per-layer AWQ device pointer set. All-zero `*_packed` = AWQ
/// inactive for that linear; each tuple of (packed, scale, zero) for
/// one linear must be all-zero or all-non-zero — partial fills are a
/// programming error caught by `exec_layer`'s dispatch gate.
///
/// Layout matches what `compressed_tensors::upload_gemma4_awq_layer`
/// produces (one `AwqLinearWeight` per of the seven Gemma 4 linears).
#[derive(Copy, Clone, Debug, Default)]
pub struct Gemma4AwqLayerPtrs {
    pub q_packed: u64,    pub q_scale: u64,    pub q_zero: u64,
    pub k_packed: u64,    pub k_scale: u64,    pub k_zero: u64,
    pub v_packed: u64,    pub v_scale: u64,    pub v_zero: u64,
    pub o_packed: u64,    pub o_scale: u64,    pub o_zero: u64,
    pub gate_packed: u64, pub gate_scale: u64, pub gate_zero: u64,
    pub up_packed: u64,   pub up_scale: u64,   pub up_zero: u64,
    pub down_packed: u64, pub down_scale: u64, pub down_zero: u64,
    /// AWQ block-scale group size along K. Typical 128. Same value
    /// shared across every linear in the layer.
    pub group_size: u32,
}

impl Gemma4AwqLayerPtrs {
    /// `true` if any linear in this layer has AWQ weights bound.
    pub fn any_active(&self) -> bool {
        self.q_packed != 0 || self.k_packed != 0 || self.v_packed != 0
            || self.o_packed != 0
            || self.gate_packed != 0 || self.up_packed != 0
            || self.down_packed != 0
    }

    /// `true` if this specific linear is AWQ-active (only the packed
    /// pointer is checked — scale/zero must be set in tandem).
    pub fn linear_active(&self, kind: AwqLinearKind) -> bool {
        match kind {
            AwqLinearKind::Q    => self.q_packed != 0,
            AwqLinearKind::K    => self.k_packed != 0,
            AwqLinearKind::V    => self.v_packed != 0,
            AwqLinearKind::O    => self.o_packed != 0,
            AwqLinearKind::Gate => self.gate_packed != 0,
            AwqLinearKind::Up   => self.up_packed != 0,
            AwqLinearKind::Down => self.down_packed != 0,
        }
    }
}

/// Identifies one of the seven AWQ-quantizable Gemma 4 linears.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum AwqLinearKind {
    Q, K, V, O, Gate, Up, Down,
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerScratch {
    pub hidden_fp8: u64,
    pub hidden_scale: u64,
    pub q_out: u64,
    pub k_out: u64,
    pub v_out: u64,
    pub q_normed: u64,
    pub k_normed: u64,
    /// V after RmsNorm, compact `[num_tokens, num_kv_heads, head_dim]`.
    /// Used by rope to read V before paged-cache write. Was previously
    /// in-place on `v_out` (inside the interleaved QKV buffer), which
    /// silently broke for `num_tokens > 1` because downstream kernels
    /// index as compact.
    pub v_normed: u64,
    pub q_fp8: u64,
    pub k_cache: u64,
    pub v_cache: u64,
    /// NVFP4 scale regions. 0 when `kv_dtype != Nvfp4`. Sized at
    /// `layer_elems / 16 * 1` bytes per layer; offsets mirror the
    /// main `k_cache` / `v_cache` layer layout.
    pub k_cache_scale: u64,
    pub v_cache_scale: u64,
    pub q_scale_ptr: u64,
    pub kv_scale_ptr: u64,
    /// Per-slot-per-head f32 K scale cache, shape
    /// `[num_blocks * block_size * num_kv_heads]`. Written by the
    /// rope kernel (amax/448 per slot) and read by the attention
    /// kernel during FP8→f32 dequant. Eliminates the per-tensor
    /// calibration guess.
    pub k_scale_cache: u64,
    /// Companion to `k_scale_cache` for V.
    pub v_scale_cache: u64,
    /// Per-(token, head) f32 Q scale scratch for this layer, shape
    /// `[num_tokens * num_heads]`. Written by the rope kernel when
    /// non-null; read by decode attention on load. Unlike the K/V
    /// caches this is transient — Q is consumed by THIS step's
    /// attention only, so the same region can be reused across
    /// layers and across per-token-decode iterations.
    pub q_scale_cache: u64,
    pub attn_out: u64,
    pub attn_out_fp8: u64,
    pub attn_out_scale: u64,
    pub delta_f16: u64,
    pub gate_up_out: u64,
    pub gate_up_fp8: u64,
    pub gate_up_scale: u64,
    pub mlp_out_fp8: u64,
    pub mlp_out_scale: u64,
    pub gemm_f32_tmp: u64,
    pub cutlass_workspace: u64,
    pub cutlass_workspace_bytes: usize,
    pub fa3_workspace: u64,
    // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
    // When non-zero on an NVFP4 layer in the instrumented set, the layer
    // performs a pre-write `rope_f16kv` into these pointers BEFORE the
    // primary `rope_nvfp4kv`. The shadow region is sized for f16 KV of
    // that layer only; the primary NVFP4 write is untouched. 0 = no
    // shadow for this layer. See `parse_shadow_layers` and
    // `NvFp4ShadowDumper` in gemma4_bring_up.rs.
    pub shadow_k_cache: u64,
    pub shadow_v_cache: u64,
    /// When non-zero, post-RoPE f16 Q snapshot dst (device buffer),
    /// size `num_tokens * num_heads * head_dim * 2` bytes. Populated
    /// on decode step 0 only; the Rust-side gate in `run_one_token`
    /// zeros this for every other step. Snapshot is taken AFTER
    /// `rope_f16kv_shadow` writes post-RoPE Q into `scratch.q_normed`
    /// and BEFORE `rope_nvfp4kv` clobbers it with another RoPE pass.
    pub shadow_q_cache: u64,
    // === END NVFP4 SHADOW DIAGNOSTIC ===
    // === HADAMARD ROTATION ===
    /// Device pointer to this layer's ±1 sign vector (i8 storage,
    /// length `head_dim`) used as `D` in the signed Walsh-Hadamard
    /// rotation `R = H * diag(D)` applied to Q post-RoPE pre-FP8-
    /// quantize. `0` when rotation is disabled
    /// (`RVLLM_NVFP4_HADAMARD` unset or kv_dtype != Nvfp4). The
    /// rope kernel treats either pointer being null as "disabled"
    /// and runs byte-identical to the pre-Hadamard path.
    pub hadamard_signs_q: u64,
    /// Companion to `hadamard_signs_q` for K. Production paths set
    /// both to the same per-layer vector — kept as separate fields
    /// so future revisions can experiment with asymmetric Q/K
    /// rotation structures without touching the kernel ABI again.
    pub hadamard_signs_k: u64,
    // === END HADAMARD ROTATION ===
}

// === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
/// Parse `RVLLM_NVFP4_SHADOW_LAYERS` (comma-separated layer indices).
/// Returns `None` when the master gate `RVLLM_NVFP4_SHADOW_F16` is off
/// or the list parses empty. Default set applied by caller when the
/// gate is on but the list env var is unset.
pub fn parse_shadow_layers() -> Option<Vec<u32>> {
    let gate = crate::gemma4_bring_up::parse_truthy_env("RVLLM_NVFP4_SHADOW_F16")
        .unwrap_or(false);
    if !gate {
        return None;
    }
    let default_set: Vec<u32> = vec![0, 10, 20, 30, 40, 50, 59];
    let raw = match std::env::var("RVLLM_NVFP4_SHADOW_LAYERS") {
        Ok(s) if !s.trim().is_empty() => s,
        _ => return Some(default_set),
    };
    let parsed: Vec<u32> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();
    if parsed.is_empty() {
        Some(default_set)
    } else {
        Some(parsed)
    }
}
// === END NVFP4 SHADOW DIAGNOSTIC ===

#[derive(Clone, Debug)]
pub struct Gemma4GemmPlans {
    pub qkv: Fp8GemmPlan,
    pub o: Fp8GemmPlan,
    pub gate_up: Fp8GemmPlan,
    pub down: Fp8GemmPlan,
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4MetadataPtrs {
    pub positions: u64,
    pub slot_mapping: u64,
    pub cos: u64,
    pub sin: u64,
    pub block_tables: u64,
    pub context_lens: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerKernels {
    pub fused_rmsnorm: KernelFn,
    pub fused_rmsnorm_fp8_quant: KernelFn,
    pub fused_qk_rmsnorm: KernelFn,
    pub fused_rope_partial_fp8kv: KernelFn,
    /// RoPE + NVFP4-packed KV write. `None` when the NVFP4 PTX
    /// module isn't built into `$KERNELS_DIR` (pre-NVFP4 branches
    /// or TPU-focused kernel trees). Populated iff
    /// `Fa2PtxKernels::fn_rope_nvfp4kv` is `Some`. Layer-exec
    /// dispatch guards on this when `kv_dtype == Nvfp4`.
    pub fused_rope_partial_nvfp4kv: Option<KernelFn>,
    pub fused_gelu_mul: KernelFn,
    pub quantize_fp8_per_token: KernelFn,
    pub residual_scale_f16: KernelFn,
    pub vnorm_f16: KernelFn,
    pub vector_add_f16: KernelFn,
    pub bf16_to_f16_sat: KernelFn,
    pub rmsnorm_inplace_bf16: KernelFn,
    pub vector_add_bf16_to_f16: KernelFn,
    pub f32_to_bf16: KernelFn,
    pub f32_to_f16_sat: KernelFn,
    pub scale_cols_f32: KernelFn,
    /// Post-GEMM per-row scale RATIO correction for FP8 GEMM at M>1.
    /// cuBLASLt on sm_121 only supports SCALAR B_SCALE mode (OUTER_VEC
    /// loses the heuristic), so a per-token-scaled activation comes
    /// out of the GEMM with `scale[0]` applied uniformly. This
    /// kernel multiplies row m by `scale[m] / scale[0]` to recover
    /// the per-token scaling.
    pub scale_rows_f32_ratio: KernelFn,
    pub compute_qkv_scales: KernelFn,
    pub fused_gelu_mul_f16: KernelFn,
    pub fused_rope_partial_f16kv: KernelFn,
    pub fused_norm_add_residual: KernelFn,
    pub fused_norm_add_residual_f16: KernelFn,
    /// F16-input variant of `fused_norm_add_residual_f16`. Reads f16
    /// gemm output directly (no channelscale broadcast), applies
    /// rmsnorm + residual add + optional layer_scalar. Used by the
    /// Sm121 O-proj and down-proj fast paths after the f16-input
    /// fp8_gemv has already baked the per-channel weight scale into
    /// its output.
    pub fused_norm_add_residual_f16in: KernelFn,
    // Cycle 54 Stage 1: BF16 residual chain. Mirror the f16 set; only
    // the residual storage dtype differs (math stays f32 internally).
    // Used when `dims.bf16_residual = true` (RVLLM_RESIDUAL_BF16=1).
    pub f16_to_bf16: KernelFn,
    pub fused_norm_add_residual_bf16: KernelFn,
    pub fused_norm_add_residual_bf16_f16in: KernelFn,
    pub fused_rmsnorm_fp8_quant_bf16in: KernelFn,
    pub fused_qkv_rmsnorm: KernelFn,
    pub scale_cols_f16: KernelFn,
    /// F16-input fp8_gemv kernel (`fp8_gemv_blockwise_wpr_native_f16in_kernel`).
    /// `None` on non-Blackwell targets — the kernel is gated on
    /// `__CUDA_ARCH__ >= 1000` in `kernels/fp8_gemv.cu`. When `Some` and
    /// the decode batch size is 1, the QKV projection skips the
    /// activation FP8-quant step and runs this kernel directly on the
    /// f16 rmsnorm output.
    pub fp8_gemv_wpr_native_f16in: Option<KernelFn>,
    /// Cycle 45 step 4.5c: AWQ INT4 W4A16 GEMV kernel handle. `None`
    /// when the PTX wasn't built into `$KERNELS_DIR`; the AWQ
    /// dispatch path checks this and rejects load if an AwqConfig
    /// is present without the kernel available. Companion to
    /// `Gemma4AwqLayerPtrs` on `Gemma4LayerWeightPtrs`.
    pub awq_int4_gemv_f16: Option<KernelFn>,
    /// Cycle 51 step 10d.4: AWQ INT4 W4A16 GEMM kernel (M>1 prefill).
    /// Companion to `awq_int4_gemv_f16` for prefill paths.
    /// When `Some` and `dims.num_tokens > 1`, the AWQ dispatch sites
    /// prefer this over the per-token GEMV loop. `None` falls back to
    /// the loop (or `RVLLM_AWQ_PREFILL_LOOP=1` debug fallback).
    pub awq_int4_gemm_sm120_wmma: Option<KernelFn>,
    /// Companion to the V-rotation arm of the NVFP4 RoPE kernel.
    /// When `RVLLM_NVFP4_HADAMARD_V=1` and the rope kernel rotated V
    /// before NVFP4-packing it, this kernel right-multiplies the
    /// attention output by R^T per (token, head) so the downstream
    /// O-projection sees un-rotated P·V. `None` when the PTX wasn't
    /// built into `$KERNELS_DIR`; dispatch site treats absent kernel
    /// as "V rotation disabled" and runs unchanged.
    pub hadamard_unrotate_f16: Option<KernelFn>,
}

#[derive(Copy, Clone, Debug)]
pub enum Gemma4Phase {
    Decode,
    Prefill {
        cu_seqlens_q: u64,
        max_seqlen_q: u32,
        num_seqs: u32,
    },
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma4_forward(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    weights: &Gemma4LayerWeightPtrs,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    cublaslt: &CublasLt,
    cutlass: &CutlassBackend,
    sliding_attention: &AttentionBackend,
    global_attention: &AttentionBackend,
    residual: u64,
    stream: u64,
) -> Result<()> {
    gemma4_forward_phase(dims, kernels, weights, scratch, meta, cublaslt, cutlass,
        sliding_attention, global_attention, residual, stream, Gemma4Phase::Decode)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma4_forward_phase(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    weights: &Gemma4LayerWeightPtrs,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    cublaslt: &CublasLt,
    cutlass: &CutlassBackend,
    sliding_attention: &AttentionBackend,
    global_attention: &AttentionBackend,
    residual: u64,
    stream: u64,
    phase: Gemma4Phase,
) -> Result<()> {
    // Route O / gate_up / down through the f16-in GEMV fast path when
    // `num_tokens <= FAST_PATH_M_MAX`. That kernel is a per-row GEMV
    // (grid.y = M, each block reloads the weight tile) and preserves
    // the full 2-D `[N/128, K/128]` weight blockscale.
    //
    // The other option — `fp8_gemm_channelscale_or_fallback` — routes
    // M>=128 through CUTLASS SM120 (full blockscale preserved) and
    // smaller M through a cuBLASLt-scalar + `scale_cols_f32` path
    // that applies only the 1-D per-row `b_chscale` approximation.
    // For Gemma 4 fp8-block weights, the K-axis blockscale has
    // meaningful variation, so the 1-D approximation measurably
    // degrades output (RVLLM_DIAG_COMPARE against the per-token
    // decode reference: 38% row-0 / 93% row-(N-1) rel_err at
    // prompt_len=18 on fp8-block weights over 60 layers).
    //
    // Cap at 127 so the bad zone (16 < M < 128, blockscale-present)
    // takes the GEMV fast path instead of the lossy fallback.
    // CUTLASS SM120 continues to cover M>=128.
    const FAST_PATH_M_MAX: u32 = 127;
    // Cycle 53 hardening: cross-cutting layer-dims invariants. Catches
    // plumbing misuse (mis-sized buckets, dtype/flag drift, non-PoT
    // block_size, OOB current_max_context_len) before any kernel
    // launches consume the values. Errors out cleanly with a typed
    // attention error pointing at the failing invariant. Called once
    // per phase; cost is negligible.
    dims.validate()?;
    // Cycle 54 Stage 1: BF16 residual chain dispatch. When the engine
    // sets `dims.bf16_residual = true` (RVLLM_RESIDUAL_BF16=1), every
    // residual touchpoint reads/writes bf16 instead of f16. The bf16
    // sibling kernels share the same launch ABI as their f16
    // counterparts; only the storage dtype changes. Math stays f32.
    let rmsnorm_quant_kernel = if dims.bf16_residual {
        kernels.fused_rmsnorm_fp8_quant_bf16in
    } else {
        kernels.fused_rmsnorm_fp8_quant
    };
    let norm_add_residual_kernel = if dims.bf16_residual {
        kernels.fused_norm_add_residual_bf16
    } else {
        kernels.fused_norm_add_residual
    };
    let norm_add_residual_f16in_kernel = if dims.bf16_residual {
        kernels.fused_norm_add_residual_bf16_f16in
    } else {
        kernels.fused_norm_add_residual_f16in
    };
    let q_dim = dims.num_heads * dims.head_dim;
    let _kv_dim = dims.num_kv_heads * dims.head_dim;
    let qkv_rows = (dims.num_heads + 2 * dims.num_kv_heads) * dims.head_dim;

    #[cfg(feature = "cuda")]
    let dbg_layer: i32 = {
        use std::sync::atomic::{AtomicU32, Ordering};
        static DBG_CTR: AtomicU32 = AtomicU32::new(0);
        let cnt = DBG_CTR.fetch_add(1, Ordering::Relaxed);
        if cnt < 2 && std::env::var("RVLLM_DBG_LAYER").is_ok() {
            cnt as i32
        } else {
            -1
        }
    };
    #[cfg(feature = "cuda")]
    macro_rules! probe {
        ($label:expr, $ptr:expr, $n:expr) => {
            if dbg_layer >= 0 {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, $ptr, 8);
                let v: Vec<f32> = s.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
                eprintln!("    [L{} {}] first4={:.4?}", dbg_layer, $label, v);
            }
        };
    }
    #[cfg(feature = "cuda")]
    macro_rules! probe_f32 {
        ($label:expr, $ptr:expr) => {
            // Cycle 49 step 8d: skip when ptr is null (AWQ-only layers
            // have weights.qkv_scale / qkv_fp8 = 0 — copying from
            // device address 0 is invalid).
            if dbg_layer >= 0 && $ptr != 0 {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut v = [0.0f32; 1];
                cudarc::driver::sys::cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut _, $ptr, 4);
                eprintln!("    [L{} {}] = {:.6e}", dbg_layer, $label, v[0]);
            }
        };
    }
    // 1. input_layernorm -> FP8 quant
    // Sm121 fast path for QKV writes f16 into delta_f16 via its
    // own rmsnorm — `scratch.hidden_fp8`/`hidden_scale` go unused. Skip
    // the quant-rmsnorm in that case to avoid the duplicate work.
    #[cfg(feature = "cuda")]
    // Must match the fast-path gate below: we can only skip the FP8
    // quant when `Fp8GemvF16InLaunch` will actually take over and
    // consume `delta_f16`. The fast path additionally requires
    // `qkv_blockscale != 0` — for fused QKV weights (blockscale == 0)
    // the kernel falls back to `fp8_gemm_channelscale_or_fallback`,
    // which reads `scratch.hidden_fp8` and needs the quant to have
    // produced it. Dropping `blockscale != 0` here silently zeroed
    // `hidden_fp8` and propagated zero logits through the LM head.
    let skip_attn_quant = dims.num_tokens == 1
        && weights.qkv_chscale != 0
        && weights.qkv_blockscale != 0
        && weights.qkv_f16 == 0
        && kernels.fp8_gemv_wpr_native_f16in.is_some();
    #[cfg(not(feature = "cuda"))]
    let skip_attn_quant = false;
    if !skip_attn_quant {
        FusedRmsnormFp8QuantLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            rmsnorm_quant_kernel,
            scratch.hidden_fp8,
            scratch.hidden_scale,
            residual,
            weights.attn_norm_gamma,
            stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe!("after_step1_residual", residual, dims.hidden);
    #[cfg(feature = "cuda")]
    probe_f32!("step1_hidden_scale", scratch.hidden_scale);
    #[cfg(feature = "cuda")]
    probe_f32!("step1_qkv_wscale", weights.qkv_scale);
    #[cfg(feature = "cuda")]
    {
        // Cycle 49 step 8d: AWQ-only layers have weights.qkv_scale = 0
        // (FP8 path absent). Copying from null device addr is invalid.
        if dbg_layer >= 0 && weights.qkv_scale != 0 && scratch.hidden_scale != 0 {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut hs = [0.0f32; 1];
            let mut ws = [0.0f32; 1];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                hs.as_mut_ptr() as *mut _,
                scratch.hidden_scale,
                4,
            );
            cudarc::driver::sys::cuMemcpyDtoH_v2(ws.as_mut_ptr() as *mut _, weights.qkv_scale, 4);
            eprintln!(
                "    [L{} step1_scale_product] hidden*qkv = {:.6e} * {:.6e} = {:.6e}",
                dbg_layer,
                hs[0],
                ws[0],
                hs[0] * ws[0]
            );
        }
    }

    #[cfg(feature = "cuda")]
    {
        // Same null-guard for the qkv_fp8 / hidden_fp8 byte probe.
        if dbg_layer >= 0 && weights.qkv_fp8 != 0 && scratch.hidden_fp8 != 0 {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut wb = [0u8; 8];
            cudarc::driver::sys::cuMemcpyDtoH_v2(wb.as_mut_ptr() as *mut _, weights.qkv_fp8, 8);
            eprintln!("    [L{} step2_qkv_fp8_bytes] first8={:?}", dbg_layer, wb);
            let mut hb = [0u8; 8];
            cudarc::driver::sys::cuMemcpyDtoH_v2(hb.as_mut_ptr() as *mut _, scratch.hidden_fp8, 8);
            eprintln!(
                "    [L{} step2_hidden_fp8_bytes] first8={:?}",
                dbg_layer, hb
            );
        }
    }

    // 2. Q||K||V projection
    #[cfg(feature = "cuda")]
    if weights.awq.linear_active(AwqLinearKind::Q) {
        // Cycle 46: AWQ INT4 W4A16 path. Three sequential GEMV launches
        // (Q, K, V) compose into the existing Q|K|V scratch via offset
        // pointer arithmetic — preserves the fused QKV RMSNorm /
        // attention path that consumes that buffer downstream.
        debug_assert!(
            weights.awq.linear_active(AwqLinearKind::K)
                && weights.awq.linear_active(AwqLinearKind::V),
            "AWQ QKV must be all-or-nothing (Q/K/V flags inconsistent)"
        );
        // Prefill (M>1) preference order:
        //   1. WMMA AWQ GEMM kernel (cycle 51d) when its PTX is present.
        //   2. Per-token GEMV loop opt-in via RVLLM_AWQ_PREFILL_LOOP=1
        //      (debug fallback, ~50ms/token).
        //   3. Otherwise FeatureNotAvailable — the kernel tree predates
        //      the GEMM and the user hasn't opted into the slow loop.
        let prefill_loop_enabled = std::env::var("RVLLM_AWQ_PREFILL_LOOP").is_ok();
        let gemm_available = kernels.awq_int4_gemm_sm120_wmma.is_some();
        if dims.num_tokens != 1 && !gemm_available && !prefill_loop_enabled {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    backend: "AwqInt4GemvF16",
                    op: "awq qkv prefill (M>1): GEMM kernel PTX absent and RVLLM_AWQ_PREFILL_LOOP not set",
                },
                ctx: rvllm_core::AttnCtx {
                    op: "awq_qkv",
                    stream,
                    num_seqs: dims.num_tokens,
                    head_dim: dims.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        // GEMV kernel handle — only required for M=1 decode or the
        // M>1 loop fallback. Pure-GEMM-prefill builds can ship without
        // the GEMV PTX (codex review of cycle 51d.4c flagged the
        // earlier unconditional ok_or_else as forcing GEMV-presence).
        let fn_awq_opt = kernels.awq_int4_gemv_f16;
        let need_gemv = dims.num_tokens == 1
            || (dims.num_tokens > 1 && (!gemm_available || prefill_loop_enabled));
        if need_gemv && fn_awq_opt.is_none() {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    backend: "AwqInt4GemvF16",
                    op: "awq_int4_gemv_f16 PTX missing from $KERNELS_DIR",
                },
                ctx: rvllm_core::AttnCtx {
                    op: "awq_qkv", stream,
                    num_seqs: dims.num_tokens, head_dim: dims.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        // fn_awq is only consumed by GEMV branches (M=1 or loop fallback);
        // need_gemv guarantees fn_awq_opt.is_some() in those branches.
        let fn_awq = fn_awq_opt;

        // Same prelude as the qkv_f16 branch: copy residual to delta
        // scratch then in-place rmsnorm.
        // Cycle 54 Stage 2.1: when the residual chain runs in bf16
        // (RVLLM_RESIDUAL_BF16=1), narrow bf16→f16-sat into the scratch
        // the f16-typed rmsnorm + downstream f16-input AWQ GEMV/GEMM
        // expect; otherwise keep the byte-identical memcpy. Both leave
        // f16 in `scratch.delta_f16`.
        if dims.bf16_residual {
            gemma4_launcher::Bf16ToF16SatLaunch {
                n: dims.num_tokens * dims.hidden,
            }
            .launch(kernels.bf16_to_f16_sat, scratch.delta_f16, residual, stream)?;
        } else {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                scratch.delta_f16,
                residual,
                (dims.num_tokens * dims.hidden * 2) as _,
                stream as _,
            );
        }
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.delta_f16,
            weights.attn_norm_gamma,
            stream,
        )?;

        let kv_dim = (dims.num_kv_heads * dims.head_dim) as u32;
        let q_dim_u = q_dim as u32;
        let g = weights.awq.group_size;
        let k = dims.hidden as u32;
        let qkv_rows_u = qkv_rows as u32;

        // Composes Q/K/V into the contiguous Q|K|V scratch via offset
        // pointers. For decode (M=1), three M=1 GEMV launches. For
        // prefill (M>1), three per-token-loop launches with
        // out_stride=qkv_rows so each token's output lands at the
        // right row.
        // Cycle 51d.4c: dispatch precedence is M=1 → GEMV; M>1 +
        // GEMM-kernel-available + NOT loop-opt-out → GEMM (~6 TFLOPS,
        // 30ms M=2048 vs 100s loop); else per-token GEMV loop fallback.
        let force_loop = std::env::var("RVLLM_AWQ_PREFILL_LOOP").is_ok();
        let use_gemm = dims.num_tokens > 1
            && !force_loop
            && kernels.awq_int4_gemm_sm120_wmma.is_some();
        unsafe {
            if dims.num_tokens == 1 {
                gemma4_launcher::AwqInt4GemvF16Launch {
                    n: q_dim_u, k, group_size: g,
                }.launch(fn_awq.unwrap(), scratch.delta_f16,
                    weights.awq.q_packed, weights.awq.q_scale, weights.awq.q_zero,
                    scratch.q_out, stream)?;
                gemma4_launcher::AwqInt4GemvF16Launch {
                    n: kv_dim, k, group_size: g,
                }.launch(fn_awq.unwrap(), scratch.delta_f16,
                    weights.awq.k_packed, weights.awq.k_scale, weights.awq.k_zero,
                    scratch.q_out + (q_dim_u as u64) * 2, stream)?;
                gemma4_launcher::AwqInt4GemvF16Launch {
                    n: kv_dim, k, group_size: g,
                }.launch(fn_awq.unwrap(), scratch.delta_f16,
                    weights.awq.v_packed, weights.awq.v_scale, weights.awq.v_zero,
                    scratch.q_out + ((q_dim_u + kv_dim) as u64) * 2, stream)?;
            } else if use_gemm {
                // Prefill via WMMA GEMM. 3 launches compose into one
                // [num_tokens, qkv_rows] scratch via ld_d=qkv_rows +
                // column-offset destination pointers.
                let fn_gemm = kernels.awq_int4_gemm_sm120_wmma.unwrap();
                let mk_gemm = |n: u32| gemma4_launcher::AwqInt4GemmSm120WmmaLaunch {
                    m: dims.num_tokens, n, k, group_size: g, ld_d: qkv_rows_u,
                };
                mk_gemm(q_dim_u).launch(fn_gemm, scratch.q_out, scratch.delta_f16,
                    weights.awq.q_packed, weights.awq.q_scale, weights.awq.q_zero,
                    stream)?;
                mk_gemm(kv_dim).launch(fn_gemm,
                    scratch.q_out + (q_dim_u as u64) * 2, scratch.delta_f16,
                    weights.awq.k_packed, weights.awq.k_scale, weights.awq.k_zero,
                    stream)?;
                mk_gemm(kv_dim).launch(fn_gemm,
                    scratch.q_out + ((q_dim_u + kv_dim) as u64) * 2, scratch.delta_f16,
                    weights.awq.v_packed, weights.awq.v_scale, weights.awq.v_zero,
                    stream)?;
            } else {
                // Per-token GEMV loop fallback (debug-only, slow).
                let mk_loop = |n: u32| gemma4_launcher::AwqInt4GemvF16PrefillLoop {
                    num_tokens: dims.num_tokens, n, k, group_size: g,
                    in_stride_elems:  k,
                    out_stride_elems: qkv_rows_u,
                };
                let f = fn_awq.unwrap();
                mk_loop(q_dim_u).launch(f, scratch.delta_f16,
                    weights.awq.q_packed, weights.awq.q_scale, weights.awq.q_zero,
                    scratch.q_out, stream)?;
                mk_loop(kv_dim).launch(f, scratch.delta_f16,
                    weights.awq.k_packed, weights.awq.k_scale, weights.awq.k_zero,
                    scratch.q_out + (q_dim_u as u64) * 2, stream)?;
                mk_loop(kv_dim).launch(f, scratch.delta_f16,
                    weights.awq.v_packed, weights.awq.v_scale, weights.awq.v_zero,
                    scratch.q_out + ((q_dim_u + kv_dim) as u64) * 2, stream)?;
            }
        }
    } else if weights.qkv_f16 != 0 {
        // F16 path: copy residual to delta_f16 scratch, apply rmsnorm in-place, use as GEMM input
        // Cycle 54 Stage 2.1: bf16-residual narrow at branch entry.
        if dims.bf16_residual {
            gemma4_launcher::Bf16ToF16SatLaunch {
                n: dims.num_tokens * dims.hidden,
            }
            .launch(kernels.bf16_to_f16_sat, scratch.delta_f16, residual, stream)?;
        } else {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                scratch.delta_f16,
                residual,
                (dims.num_tokens * dims.hidden * 2) as _,
                stream as _,
            );
        }
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.delta_f16,
            weights.attn_norm_gamma,
            stream,
        )?;
        cublaslt.f16_gemm_f32(
            scratch.delta_f16,
            weights.qkv_f16,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            qkv_rows as i32,
            dims.hidden as i32,
            stream,
        )?;
        gemma4_launcher::Bf16ToF16SatLaunch {
            n: dims.num_tokens * qkv_rows,
        }
        .launch(
            kernels.f32_to_f16_sat,
            scratch.q_out,
            scratch.gemm_f32_tmp,
            stream,
        )?;
    } else if let (true, Some(fn_gemv)) = (
        // Blockscale gate: `Fp8GemvF16InLaunch` reads a 2-D
        // `[N/128, K/128]` tensor. Only enable it when the loader has
        // actually uploaded one (`*_blockscale != 0`). Weights whose
        // scale was per-row or synthesized have `blockscale == 0` and
        // stay on the channelscale-preserving fallback below.
        weights.qkv_blockscale != 0 && dims.num_tokens == 1,
        kernels.fp8_gemv_wpr_native_f16in,
    ) {
        // sm_121 fast path: skip the activation FP8-quant entirely
        // and run `fp8_gemv_blockwise_wpr_native_f16in_kernel` directly
        // against the f16 rmsnorm output. Wins over the
        // `fp8_gemm_channelscale_or_fallback` path on two axes:
        //
        //   * Quality: preserves the per-channel weight block-scale that
        //     the cuBLASLt fallback drops (the cuBLASLt FP8 channelscale
        //     heuristic `LaunchFailed`s on Blackwell consumer, so the
        //     fallback currently collapses to a scalar weight scale).
        //   * Speed: one kernel (f16 GEMV) instead of two (FP8 quant +
        //     cuBLASLt FP8 GEMM), no scratch round-trip through f32.
        //
        // The extra memcpy + rmsnorm-inplace here duplicates the work
        // already done by `fused_rmsnorm_fp8_quant` in step 1 — at M=1
        // that's ~5 KiB of rmsnorm work against a >30 MiB weight GEMV,
        // well below the noise floor.
        // Cycle 54 Stage 2.1: this is the M=1 decode QKV hot path.
        // Narrow bf16→f16-sat at branch entry when the residual chain
        // is bf16 so the f16-typed rmsnorm + Fp8GemvF16InLaunch consume
        // valid f16 (was a silent dtype mismatch in Stage 2.0).
        if dims.bf16_residual {
            gemma4_launcher::Bf16ToF16SatLaunch {
                n: dims.num_tokens * dims.hidden,
            }
            .launch(kernels.bf16_to_f16_sat, scratch.delta_f16, residual, stream)?;
        } else {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                scratch.delta_f16,
                residual,
                (dims.num_tokens * dims.hidden * 2) as _,
                stream as _,
            );
        }
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.delta_f16,
            weights.attn_norm_gamma,
            stream,
        )?;
        gemma4_launcher::Fp8GemvF16InLaunch {
            m: dims.num_tokens,
            n: qkv_rows,
            k: dims.hidden,
        }
        .launch(
            fn_gemv,
            scratch.q_out,
            weights.qkv_fp8,
            weights.qkv_blockscale,
            scratch.delta_f16,
            stream,
        )?;
    } else if weights.qkv_chscale != 0 {
        fp8_gemm_channelscale_or_fallback(
            cutlass, cublaslt, kernels.f32_to_f16_sat, kernels.scale_cols_f32, kernels.scale_rows_f32_ratio,
            scratch.q_out, scratch.hidden_fp8, weights.qkv_fp8,
            scratch.hidden_scale, weights.qkv_chscale, weights.qkv_blockscale, weights.qkv_scale,
            dims.num_tokens as i32, qkv_rows as i32, dims.hidden as i32,
            scratch.gemm_f32_tmp,
            scratch.cutlass_workspace, scratch.cutlass_workspace_bytes,
            stream,
        )?;
    } else {
        cublaslt.fp8_gemm(
            scratch.hidden_fp8, weights.qkv_fp8, scratch.q_out,
            dims.num_tokens as i32, qkv_rows as i32, dims.hidden as i32,
            scratch.hidden_scale, weights.qkv_scale, stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe!("step2_q_proj", scratch.q_out, dims.hidden);
    #[cfg(feature = "cuda")]
    {
        if dbg_layer >= 0 {
            let k_offset = q_dim as u64 * 2;
            let v_offset = (q_dim + dims.num_kv_heads * dims.head_dim) as u64 * 2;
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut sk = [0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                sk.as_mut_ptr() as *mut _,
                scratch.q_out + k_offset,
                8,
            );
            let kv: Vec<f32> = sk.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
            eprintln!("    [L{} step2_k_proj] first4={:.4?}", dbg_layer, kv);
            let mut sv = [0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                sv.as_mut_ptr() as *mut _,
                scratch.q_out + v_offset,
                8,
            );
            let vv: Vec<f32> = sv.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
            eprintln!("    [L{} step2_v_proj] first4={:.4?}", dbg_layer, vv);
        }
    }

    // 2b+3. Fused QKV-norm: Q/K with learned gamma, V parameter-free.
    // Src pointers (q_out/k_out/v_out) index into the shared interleaved
    // QKV GEMM output with row stride `qkv_rows`. Dsts are compact
    // scratch buffers so the downstream rope kernel sees a uniform
    // `[num_tokens, n_heads, head_dim]` layout across all three.
    let qkv_rows = q_dim + 2 * dims.num_kv_heads * dims.head_dim;
    gemma4_launcher::FusedQkvRmsnormLaunch {
        num_tokens: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        eps: dims.rms_eps,
        src_row_stride: qkv_rows,
    }
    .launch(
        kernels.fused_qkv_rmsnorm,
        scratch.q_out,
        scratch.k_out,
        scratch.v_out,
        scratch.q_normed,
        scratch.k_normed,
        scratch.v_normed,
        weights.q_norm_gamma,
        weights.k_norm_gamma,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    probe!("step3_q_norm", scratch.q_normed, dims.hidden);
    #[cfg(feature = "cuda")]
    probe!("step3_k_norm", scratch.k_normed, dims.hidden);

    // 4-5. RoPE + attention (F16 or FP8 KV cache, decode or prefill)
    let attention = match dims.layer_type {
        Gemma4LayerType::SlidingAttention => sliding_attention,
        Gemma4LayerType::GlobalAttention => global_attention,
    };
    let window_size_left: i32 = match dims.layer_type {
        Gemma4LayerType::SlidingAttention => (dims.sliding_window as i32) - 1,
        Gemma4LayerType::GlobalAttention => -1,
    };

    #[cfg(feature = "cuda")]
    match phase {
        Gemma4Phase::Decode => {
            let decode_params = PagedDecodeParams {
                num_seqs: dims.num_tokens,
                num_heads: dims.num_heads,
                num_kv_heads: dims.num_kv_heads,
                head_dim: dims.head_dim,
                block_size: dims.block_size,
                max_blocks_per_seq: dims.max_blocks_per_seq,
                num_blocks_total: dims.num_blocks_total,
                scale: dims.attn_scale,
                window_size_left,
            };
            match dims.kv_dtype {
                KvDtype::F16 => {
                    // F16 KV cache path: RoPE outputs F16 Q and F16 KV cache.
                    rope_f16kv(dims, kernels, scratch, meta, stream)?;
                    let decode = rvllm_attention::PagedDecodeLauncher::new(attention);
                    decode.launch(
                        decode_params,
                        scratch.attn_out,
                        scratch.q_normed,
                        scratch.k_cache,
                        scratch.v_cache,
                        meta.block_tables,
                        meta.context_lens,
                        scratch.fa3_workspace,
                        stream,
                    )?;
                }
                KvDtype::Fp8 => {
                    // FP8 path with F-series per-slot scale signature.
                    rope_fp8kv(dims, kernels, scratch, meta, stream)?;
                    let decode = PagedDecodeFp8Launcher::new(attention);
                    decode.launch(
                        decode_params,
                        scratch.attn_out,
                        scratch.q_fp8,
                        scratch.k_cache,
                        scratch.v_cache,
                        scratch.k_scale_cache,
                        scratch.v_scale_cache,
                        scratch.q_scale_cache,
                        0, // k_descale_fallback (unused when per-slot populated)
                        0, // v_descale_fallback
                        meta.block_tables,
                        meta.context_lens,
                        scratch.fa3_workspace,
                        scratch.q_scale_ptr,
                        stream,
                    )?;
                }
                KvDtype::Nvfp4 => {
                    // NVFP4 path: dedicated RoPE + dedicated decode launcher.
                    // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
                    // Shadow rope writes f16 K/V to `shadow_k_cache` /
                    // `shadow_v_cache` and rotated Q to
                    // `shadow_q_cache` (per-layer slot on step-0
                    // capture, shared throwaway otherwise). Must NOT
                    // touch `q_normed` — the primary rope below reads
                    // it as pre-RoPE and rotates exactly once.
                    if scratch.shadow_k_cache != 0 {
                        rope_f16kv_shadow(dims, kernels, scratch, meta, stream)?;
                    }
                    // === END NVFP4 SHADOW DIAGNOSTIC ===
                    rope_nvfp4kv(dims, kernels, scratch, meta, stream)?;
                    let decode = rvllm_attention::PagedDecodeNvfp4Launcher::new(attention);

                    // Split-KV decode (paged_attention_v2-style) —
                    // default ON. Opt-out via `RVLLM_NVFP4_SPLIT_KV=0`
                    // for A/B comparisons. Engages only when the split
                    // kernels exist AND the context is long enough to
                    // benefit (single-partition would waste phase-2
                    // overhead).
                    //
                    // Measured +75% on 15k-ctx bs=1 decode vs the
                    // single-CTA path (2.0 → 3.5 tok/s on GB10); was
                    // the gating change that got zeroclaw's 14795-tok
                    // prompt under the 600s gateway timeout.
                    //
                    // Partition size follows vLLM's default (512).
                    // `max_num_parts` is sized for the engine's max
                    // context window so the workspace layout is
                    // stable across calls — running at a shorter ctx
                    // just writes sentinels to the tail partitions.
                    let split_env_on =
                        crate::gemma4_bring_up::parse_truthy_env("RVLLM_NVFP4_SPLIT_KV")
                            .unwrap_or(true);
                    // Cycle 23 sweep winner: 1024 — only setting with both
                    // (a) zero garbage on long-ctx WEATHER cliff and
                    // (b) clean short-ctx simple tool-call (no regression).
                    // Sweep on V=ON + cycle21 f32-tmp_out base, 5 trials each:
                    //   512:  1 partial-clean, 1 garbage, 3 empty.
                    //   1024: 1 fully-clean, 0 garbage, 4 empty. SHORT: clean.
                    //   2048: 2 fully-clean, 1 partial, 1 garbage, 1 empty.
                    //   256:  2 fully-clean, 2 partial, 1 garbage, 0 empty
                    //         BUT regresses short-ctx tool-call to text refusal
                    //         (same regression as SPLIT_KV=0 in cycle 20).
                    const PARTITION_SIZE: u32 = 1024;
                    // Bucket max ctx = max_blocks_per_seq * block_size.
                    // Workspace is sized off this so the layout is
                    // stable across iterations regardless of current
                    // context growth.
                    let bucket_ctx = (dims.max_blocks_per_seq as u32)
                        * (dims.block_size as u32);
                    let max_num_parts = bucket_ctx.div_ceil(PARTITION_SIZE).max(1);
                    // Current-step max ctx — used only for the dispatch
                    // gate, NOT for workspace sizing. Earlier revisions
                    // used `bucket_ctx` here, which over-fired the split
                    // path on short-context turns (each empty partition
                    // still launches a CTA, just writing sentinels).
                    // Fall back to bucket_ctx when caller didn't supply.
                    let current_ctx = dims
                        .current_max_context_len
                        .unwrap_or(bucket_ctx);
                    let current_num_parts =
                        current_ctx.div_ceil(PARTITION_SIZE).max(1);
                    // Skip split path if: env gate off, split kernels
                    // missing, current ctx short enough that one CTA
                    // per (seq, head) fits comfortably, or workspace
                    // can't fit the scratch layout.
                    let slots = (decode_params.num_seqs as u64)
                        * (dims.num_heads as u64)
                        * (max_num_parts as u64);
                    // tmp_out widened to f32 in cycle21 (codex precision fix):
                    // 4 bytes/elem instead of 2. Metadata still f32 (max_logits + exp_sums = 2*4 bytes/slot).
                    let ws_need = slots * (dims.head_dim as u64) * 4
                        + slots * 4 * 2;
                    let use_split = split_env_on
                        && decode.has_split_kernels()
                        && current_num_parts > 1
                        && ws_need <= 16 * 1024 * 1024; // fa3_ws is 16MiB
                    // === DYNAMIC NVFP4 Q SCALE ===
                    // When `RVLLM_PER_TOKEN_Q_SCALE=1`, pass the per-
                    // (token, head) Q descale table populated by
                    // `rope_nvfp4kv`. Required for `RVLLM_NVFP4_HADAMARD=1`
                    // where rotated Q saturates the static scalar.
                    let nvfp4_per_token_q =
                        crate::gemma4_bring_up::per_token_q_scale_enabled(/*default_on=*/false);
                    let nvfp4_q_scale_cache: u64 =
                        if nvfp4_per_token_q { scratch.q_scale_cache } else { 0 };
                    // === END DYNAMIC NVFP4 Q SCALE ===
                    if use_split {
                        decode.launch_split(
                            decode_params,
                            scratch.attn_out,
                            scratch.q_fp8,
                            scratch.k_cache,
                            scratch.v_cache,
                            scratch.k_cache_scale,
                            scratch.v_cache_scale,
                            // === DYNAMIC NVFP4 Q SCALE ===
                            nvfp4_q_scale_cache,
                            // === END DYNAMIC NVFP4 Q SCALE ===
                            meta.block_tables,
                            meta.context_lens,
                            scratch.q_scale_ptr,
                            scratch.fa3_workspace,
                            PARTITION_SIZE,
                            max_num_parts,
                            stream,
                        )?;
                    } else {
                        decode.launch(
                            decode_params,
                            scratch.attn_out,
                            scratch.q_fp8,
                            scratch.k_cache,
                            scratch.v_cache,
                            scratch.k_cache_scale,
                            scratch.v_cache_scale,
                            // === DYNAMIC NVFP4 Q SCALE ===
                            nvfp4_q_scale_cache,
                            // === END DYNAMIC NVFP4 Q SCALE ===
                            meta.block_tables,
                            meta.context_lens,
                            scratch.q_scale_ptr,
                            stream,
                        )?;
                    }
                    // Strip V's R rotation from attn_out before the
                    // O-projection sees it. No-op unless RVLLM_NVFP4_HADAMARD_V=1.
                    unrotate_attn_out_v_if_enabled(dims, kernels, scratch, stream)?;
                }
            }
        }
        Gemma4Phase::Prefill { cu_seqlens_q, max_seqlen_q, num_seqs: _ } => {
            // NVFP4 prefill: dedicated RoPE + attention launcher. Must
            // fall through to the shared post-attention block below
            // (O proj + post-attn norm + residual add + MLP + post-FF
            // norm + residual add), same as the FP8 path does after
            // its unified/decode-per-qi attention. A previous revision
            // `return Ok(())`'d here, which turned every NVFP4-prefill
            // layer into a no-op on the residual — end-to-end output
            // was the raw embedding, manifesting as word-salad tokens
            // at the LM head regardless of prompt length.
            if dims.kv_dtype == KvDtype::Nvfp4 {
                // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
                // Shadow rope writes f16 K/V + rotated Q to the
                // shadow slots without touching q_normed; primary rope
                // below gets clean pre-RoPE Q.
                if scratch.shadow_k_cache != 0 {
                    rope_f16kv_shadow(dims, kernels, scratch, meta, stream)?;
                }
                // === END NVFP4 SHADOW DIAGNOSTIC ===
                rope_nvfp4kv(dims, kernels, scratch, meta, stream)?;
                let prefill_params = rvllm_attention::PagedPrefillParams {
                    num_tokens: dims.num_tokens,
                    num_seqs: 1,
                    num_heads: dims.num_heads,
                    num_kv_heads: dims.num_kv_heads,
                    head_dim: dims.head_dim,
                    block_size: dims.block_size,
                    max_blocks_per_seq: dims.max_blocks_per_seq,
                    num_blocks_total: dims.num_blocks_total,
                    scale: dims.attn_scale,
                    window_size_left,
                };
                let prefill = rvllm_attention::PagedPrefillNvfp4Launcher::new(attention);

                // Phase 2b of aa01001nvf4f16mma: route multi-query
                // prefill through the f16-MMA unified kernel when it
                // loaded into Fa2PtxKernels. Matches the FP8 unified
                // dispatch gate above — opt-out via
                // `RVLLM_UNIFIED_PREFILL=0` for bisect bisect/quality
                // comparisons. Falls back to the per-qi dedicated
                // kernel when the unified PTX is missing (old kernel
                // trees) or disabled via env.
                let unified_enabled_nvfp4 =
                    crate::gemma4_bring_up::parse_truthy_env("RVLLM_UNIFIED_PREFILL")
                        .unwrap_or(true);
                let have_nvfp4_unified = match attention {
                    rvllm_attention::AttentionBackend::Fa2Ptx(k)
                        if k.has_unified_prefill_nvfp4() => true,
                    _ => false,
                };
                if unified_enabled_nvfp4 && have_nvfp4_unified && dims.num_tokens > 1 {
                    // Tile size matches the FP8 unified heuristic —
                    // head_dim ≤ 256 gets tile_size=32 (sliding), else
                    // 16 (global, smem-tight at head_dim=512).
                    let tile_size = if dims.head_dim <= 256 { 32u32 } else { 16u32 };
                    let num_queries_per_kv = dims.num_heads / dims.num_kv_heads;
                    let block_q = rvllm_attention::UNIFIED_PREFILL_BLOCK_M
                        / num_queries_per_kv.max(1);
                    let unified = rvllm_attention::UnifiedPrefillParams {
                        num_queries_per_kv,
                        tile_size,
                        block_q,
                        use_mma: true,
                    };
                    // === DYNAMIC NVFP4 Q SCALE ===
                    // When `RVLLM_PER_TOKEN_Q_SCALE=1`, rope_nvfp4kv
                    // writes a fresh per-(token, head) FP8 scale into
                    // scratch.q_scale_cache (mirroring the FP8 sibling).
                    // Pass that cache through so the attention kernel
                    // dequants Q with the same scale rope used at
                    // quantize time. Required for RVLLM_NVFP4_HADAMARD
                    // where rotated Q can saturate the static scalar.
                    let nvfp4_per_token_q =
                        crate::gemma4_bring_up::per_token_q_scale_enabled(/*default_on=*/false);
                    let nvfp4_q_scale_cache: u64 =
                        if nvfp4_per_token_q { scratch.q_scale_cache } else { 0 };
                    // === END DYNAMIC NVFP4 Q SCALE ===
                    prefill.launch_nvfp4kv_unified_sm121(
                        prefill_params,
                        unified,
                        scratch.attn_out,
                        scratch.q_fp8,
                        scratch.k_cache,
                        scratch.v_cache,
                        scratch.k_cache_scale,
                        scratch.v_cache_scale,
                        nvfp4_q_scale_cache,
                        meta.block_tables,
                        cu_seqlens_q,
                        meta.context_lens,
                        scratch.q_scale_ptr,
                        stream,
                    )?;
                } else {
                    // === DYNAMIC NVFP4 Q SCALE ===
                    let nvfp4_per_token_q =
                        crate::gemma4_bring_up::per_token_q_scale_enabled(/*default_on=*/false);
                    let nvfp4_q_scale_cache: u64 =
                        if nvfp4_per_token_q { scratch.q_scale_cache } else { 0 };
                    // === END DYNAMIC NVFP4 Q SCALE ===
                    prefill.launch(
                        prefill_params,
                        scratch.attn_out,
                        scratch.q_fp8,
                        scratch.k_cache,
                        scratch.v_cache,
                        scratch.k_cache_scale,
                        scratch.v_cache_scale,
                        // === DYNAMIC NVFP4 Q SCALE ===
                        nvfp4_q_scale_cache,
                        // === END DYNAMIC NVFP4 Q SCALE ===
                        meta.block_tables,
                        meta.context_lens,
                        cu_seqlens_q,
                        scratch.q_scale_ptr,
                        max_seqlen_q,
                        stream,
                    )?;
                }
                // Strip V's R rotation from attn_out before O-proj sees it.
                // No-op unless RVLLM_NVFP4_HADAMARD_V=1. Both prefill paths
                // (unified + per-qi) above wrote into scratch.attn_out, so
                // a single call here covers both.
                unrotate_attn_out_v_if_enabled(dims, kernels, scratch, stream)?;
            } else {
            // FP8 / F16 prefill: share the FP8 write path (no F16 prefill
            // kernel on sm_121). F-series unified-or-fallback structure:
            // Prefill always uses FP8 KV path (no F16 prefill kernel).
            rope_fp8kv(dims, kernels, scratch, meta, stream)?;

            // --- Unified multi-Q prefill fast path (sm_121) -----------
            // `flash_attention_2_prefill_fp8kv_unified_kernel` handles
            // all prompt tokens of one sequence in ONE kernel launch
            // per layer. See v3/UNIFIED_PREFILL_SPEC.md. We route
            // through it whenever:
            //   * the attention backend is Fa2Ptx (sm_121),
            //   * the PTX module + symbol loaded (`has_unified_prefill`),
            //   * `RVLLM_UNIFIED_PREFILL` isn't explicitly set to "0".
            // Otherwise we fall back to the decode-per-qi loop below —
            // retained both as a bisect tool and as the reference path
            // rvllm-ppl validates bit-for-bit.
            let unified_enabled =
                crate::gemma4_bring_up::parse_truthy_env("RVLLM_UNIFIED_PREFILL")
                    .unwrap_or(true);
            let fa2_unified = if unified_enabled {
                match attention {
                    rvllm_attention::AttentionBackend::Fa2Ptx(k) if k.has_unified_prefill() => true,
                    _ => false,
                }
            } else {
                false
            };
            if fa2_unified && dims.num_tokens > 1 {
                let prefill_params = rvllm_attention::PagedPrefillParams {
                    num_tokens: dims.num_tokens,
                    num_seqs: 1,
                    num_heads: dims.num_heads,
                    num_kv_heads: dims.num_kv_heads,
                    head_dim: dims.head_dim,
                    block_size: dims.block_size,
                    max_blocks_per_seq: dims.max_blocks_per_seq,
                    num_blocks_total: dims.num_blocks_total,
                    scale: dims.attn_scale,
                    window_size_left,
                };
                // Tile size picked so smem stays inside the sm_121
                // 99 KB opt-in cap (see UNIFIED_PREFILL_SPEC §params).
                let tile_size = if dims.head_dim <= 256 { 32u32 } else { 16u32 };
                let num_queries_per_kv = dims.num_heads / dims.num_kv_heads;
                let block_q = rvllm_attention::UNIFIED_PREFILL_BLOCK_M / num_queries_per_kv.max(1);
                // Q·Kᵀ MMA is opt-in during bring-up (Phase F3).
                // `RVLLM_UNIFIED_PREFILL_MMA=1` flips on the
                // sm_121a `mma.sync.kind::f8f6f4` tensor-core path;
                // unset / `0` keeps the scalar FMA reference.
                let use_mma =
                    crate::gemma4_bring_up::parse_truthy_env("RVLLM_UNIFIED_PREFILL_MMA")
                        .unwrap_or(false);
                let unified = rvllm_attention::UnifiedPrefillParams {
                    num_queries_per_kv,
                    tile_size,
                    block_q,
                    use_mma,
                };
                let prefill = rvllm_attention::PagedPrefillFp8Launcher::new(attention);
                prefill.launch_fp8kv_unified_sm121(
                    prefill_params,
                    unified,
                    scratch.attn_out,
                    scratch.q_fp8,
                    scratch.k_cache,
                    scratch.v_cache,
                    scratch.k_scale_cache,
                    scratch.v_scale_cache,
                    scratch.q_scale_cache,
                    0, // k_descale_fallback (unused when per-slot populated)
                    0, // v_descale_fallback
                    meta.block_tables,
                    cu_seqlens_q,
                    meta.context_lens,
                    scratch.q_scale_ptr,
                    stream,
                )?;
            } else {
            // --- Decode-per-qi fallback -------------------------------
            // Replaces batch prefill with a loop of single-query decode
            // kernel calls — one per prompt position qi. Per-qi
            // context_lens value is NOT rewritten via HtoD each
            // iteration (that races against the non-default stream);
            // instead we reuse the `cu_seqlens_q` scratch region as a
            // pre-populated device array `[1, 2, ..., num_tokens]`
            // and let decode read ctx = (qi+1) by pointing into it at
            // offset qi. By construction this is bit-identical to the
            // per-token decode path rvllm-ppl validates.
            //
            // Cost: prompt_len extra kernel launches per layer — the
            // ≈30× TTFT gap against vLLM that motivated the unified
            // kernel above.
            let decode_params = PagedDecodeParams {
                num_seqs: 1,
                num_heads: dims.num_heads,
                num_kv_heads: dims.num_kv_heads,
                head_dim: dims.head_dim,
                block_size: dims.block_size,
                max_blocks_per_seq: dims.max_blocks_per_seq,
                num_blocks_total: dims.num_blocks_total,
                scale: dims.attn_scale,
                window_size_left,
            };
            let decode = PagedDecodeFp8Launcher::new(attention);
            let o_stride_bytes =
                (dims.num_heads as u64) * (dims.head_dim as u64) * 2; // f16
            let q_fp8_stride_bytes = (dims.num_heads as u64) * (dims.head_dim as u64); // fp8
            // Per-(token, head) Q scale cache stride. Rope writes at
            // `q_scale_cache[token_idx * num_heads + head_idx]` (see
            // fused_rope_partial_fp8kv.cu). Per-qi decode reads at
            // `q_scale_cache[seq_idx * num_heads + head_idx]` with
            // seq_idx=0 (num_seqs=1 per launch), so we must advance
            // the pointer by `qi * num_heads * sizeof::<f32>()` just
            // like the Q FP8 pointer — otherwise token qi gets token 0's
            // scale and prefill logits diverge from the per-token
            // decode reference.
            let q_scale_stride_bytes = (dims.num_heads as u64) * 4;

            // Pre-populate cu_seqlens_q region with [1, 2, ..., num_tokens]
            // via a small host→device copy on THIS stream (async with
            // pageable src; cudarc routes through the stream handle).
            // We reuse cu_seqlens_q because it's already sized
            // `(num_tokens + 1) * 4 bytes` and otherwise unused once
            // the unified attention replaces the FA2 prefill kernel.
            let ctx_host: Vec<i32> =
                (1..=dims.num_tokens as i32).collect();
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                cu_seqlens_q,
                ctx_host.as_ptr() as *const _,
                (ctx_host.len() * 4) as _,
                stream as _,
            );

            for qi in 0..dims.num_tokens {
                let q_scale_cache_qi = if scratch.q_scale_cache != 0 {
                    scratch.q_scale_cache + (qi as u64) * q_scale_stride_bytes
                } else {
                    0
                };
                decode.launch(
                    decode_params,
                    scratch.attn_out + (qi as u64) * o_stride_bytes,
                    scratch.q_fp8 + (qi as u64) * q_fp8_stride_bytes,
                    scratch.k_cache,
                    scratch.v_cache,
                    scratch.k_scale_cache,
                    scratch.v_scale_cache,
                    q_scale_cache_qi,
                    0, // k_descale_fallback (unused when per-slot populated)
                    0, // v_descale_fallback
                    meta.block_tables,
                    cu_seqlens_q + (qi as u64) * 4,
                    scratch.fa3_workspace,
                    scratch.q_scale_ptr,
                    stream,
                )?;
            }
            // Ensure ctx_host lives until the stream has consumed it.
            std::hint::black_box(&ctx_host);
            } // end of decode-per-qi fallback
            } // end of FP8/F16 prefill (else of NVFP4 branch)
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = phase;

    #[cfg(feature = "cuda")]
    probe!("step5_attn_out", scratch.attn_out, q_dim);

    // 6. quantize attn_out -> fp8 per-token (skip when F16 KV + F16 O-proj,
    // or when the Sm121 fast path will read `scratch.attn_out`
    // as f16 directly in step 7).
    #[cfg(feature = "cuda")]
    let skip_o_quant = dims.num_tokens <= FAST_PATH_M_MAX
        && weights.o_f16 == 0
        && weights.o_chscale != 0
        && weights.o_blockscale != 0
        && kernels.fp8_gemv_wpr_native_f16in.is_some();
    #[cfg(not(feature = "cuda"))]
    let skip_o_quant = false;
    if (!dims.f16_kv || weights.o_f16 == 0) && !skip_o_quant {
        rvllm_fused::QuantizeFp8PerTokenLaunch {
            num_tokens: dims.num_tokens,
            dim: q_dim,
        }
        .launch(
            kernels.quantize_fp8_per_token,
            scratch.attn_out_fp8,
            scratch.attn_out_scale,
            scratch.attn_out,
            stream,
        )?;
    }

    // 7-8. O proj + channelscale + post_attn norm + residual add
    #[cfg(feature = "cuda")]
    if weights.awq.linear_active(AwqLinearKind::O) {
        // Cycle 46 step 5b: AWQ O-proj. Reads f16 attn_out directly,
        // writes f16 staging in gemm_f32_tmp, then the standard f16-in
        // post-attn-norm + residual epilogue rolls it into `residual`.
        let prefill_loop_enabled = std::env::var("RVLLM_AWQ_PREFILL_LOOP").is_ok();
        let gemm_available = kernels.awq_int4_gemm_sm120_wmma.is_some();
        if dims.num_tokens != 1 && !gemm_available && !prefill_loop_enabled {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    backend: "AwqInt4GemvF16",
                    op: "awq o_proj prefill (M>1): GEMM kernel PTX absent and RVLLM_AWQ_PREFILL_LOOP not set",
                },
                ctx: rvllm_core::AttnCtx {
                    op: "awq_o", stream,
                    num_seqs: dims.num_tokens, head_dim: dims.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        // O proj: GEMV needed only for decode or loop-fallback path.
        let need_gemv = dims.num_tokens == 1
            || (dims.num_tokens > 1 && (!gemm_available || prefill_loop_enabled));
        if need_gemv && kernels.awq_int4_gemv_f16.is_none() {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    backend: "AwqInt4GemvF16", op: "awq_int4_gemv_f16 PTX missing",
                },
                ctx: rvllm_core::AttnCtx { op: "awq_o", stream,
                    num_seqs: dims.num_tokens, head_dim: dims.head_dim },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let fn_awq = kernels.awq_int4_gemv_f16;
        let o_out_f16 = scratch.gemm_f32_tmp;
        let use_gemm = dims.num_tokens > 1 && !prefill_loop_enabled && gemm_available;
        unsafe {
            if dims.num_tokens == 1 {
                gemma4_launcher::AwqInt4GemvF16Launch {
                    n: dims.hidden as u32,
                    k: q_dim as u32,
                    group_size: weights.awq.group_size,
                }.launch(fn_awq.unwrap(), scratch.attn_out,
                    weights.awq.o_packed, weights.awq.o_scale, weights.awq.o_zero,
                    o_out_f16, stream)?;
            } else if use_gemm {
                // O proj prefill via GEMM: ld_d=hidden (output is
                // [num_tokens, hidden] contiguous in gemm_f32_tmp).
                gemma4_launcher::AwqInt4GemmSm120WmmaLaunch {
                    m: dims.num_tokens,
                    n: dims.hidden as u32,
                    k: q_dim as u32,
                    group_size: weights.awq.group_size,
                    ld_d: dims.hidden as u32,
                }.launch(kernels.awq_int4_gemm_sm120_wmma.unwrap(),
                    o_out_f16, scratch.attn_out,
                    weights.awq.o_packed, weights.awq.o_scale, weights.awq.o_zero,
                    stream)?;
            } else {
                // O proj prefill loop fallback.
                gemma4_launcher::AwqInt4GemvF16PrefillLoop {
                    num_tokens: dims.num_tokens,
                    n: dims.hidden as u32,
                    k: q_dim as u32,
                    group_size: weights.awq.group_size,
                    in_stride_elems:  q_dim as u32,
                    out_stride_elems: dims.hidden as u32,
                }.launch(fn_awq.unwrap(), scratch.attn_out,
                    weights.awq.o_packed, weights.awq.o_scale, weights.awq.o_zero,
                    o_out_f16, stream)?;
            }
        }
        gemma4_launcher::FusedNormAddResidualF16InLaunch {
            num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
        }.launch(
            norm_add_residual_f16in_kernel, o_out_f16,
            weights.post_attn_norm_gamma, residual, 0, stream,
        )?;
    } else if weights.o_f16 != 0 {
        cublaslt.f16_gemm_f32(
            scratch.attn_out, weights.o_f16, scratch.gemm_f32_tmp,
            dims.num_tokens as i32, dims.hidden as i32, q_dim as i32, stream,
        )?;
        gemma4_launcher::FusedNormAddResidualLaunch {
            num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
        }.launch(norm_add_residual_kernel, scratch.gemm_f32_tmp,
            weights.post_attn_norm_gamma, residual, 0, stream)?;
    } else if let (true, Some(fn_gemv)) = (
        weights.o_blockscale != 0 && dims.num_tokens <= FAST_PATH_M_MAX,
        kernels.fp8_gemv_wpr_native_f16in,
    ) {
        // sm_121 fast path for O projection.
        // `scratch.attn_out` is already f16 (attention output), no
        // pre-rmsnorm needed — post-attn-norm runs in the epilogue via
        // `fused_norm_add_residual_f16in`. We write the GEMV result
        // into `gemm_f32_tmp` (reused as f16 scratch: we only need
        // num_tokens*hidden*2 bytes, well under gemm_f32_tmp's capacity).
        gemma4_launcher::Fp8GemvF16InLaunch {
            m: dims.num_tokens,
            n: dims.hidden,
            k: q_dim,
        }
        .launch(
            fn_gemv,
            scratch.gemm_f32_tmp,
            weights.o_fp8,
            weights.o_blockscale,
            scratch.attn_out,
            stream,
        )?;
        gemma4_launcher::FusedNormAddResidualF16InLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            norm_add_residual_f16in_kernel,
            scratch.gemm_f32_tmp,
            weights.post_attn_norm_gamma,
            residual,
            0,
            stream,
        )?;
    } else if weights.o_chscale != 0 {
        // O-proj batch path (num_tokens > FAST_PATH_M_MAX).
        //
        // The old path was `cublaslt.fp8_gemm_f32` into f32 scratch
        // + `FusedNormAddResidualF16Launch` applying per-row o_chscale.
        // That applies the CHANNELSCALE approximation (per-row only,
        // collapsing K-block variation) whereas the per-token fast path
        // applies the full 2-D blockscale via Fp8GemvF16InLaunch. On
        // real Gemma 4 31B the scale tensor has meaningful K-block
        // variation, so the approximation diverges from per-token —
        // root cause of aa01001pftrope0 step-8 divergence.
        //
        // Route batch through `fp8_gemm_channelscale_or_fallback`
        // which, after the b_blockscale gate fix, picks CUTLASS
        // SM120 + full 2-D blockscale at M>=128, and the
        // f32-scratch channelscale fallback otherwise (still lossy
        // in the 16 < M < 128 range, tracked as a follow-up).
        let o_out_f16 = scratch.gemm_f32_tmp; // reused as f16 staging, plenty big
        fp8_gemm_channelscale_or_fallback(
            cutlass, cublaslt, kernels.f32_to_f16_sat, kernels.scale_cols_f32, kernels.scale_rows_f32_ratio,
            o_out_f16, scratch.attn_out_fp8, weights.o_fp8,
            scratch.attn_out_scale, weights.o_chscale, weights.o_blockscale, weights.o_scale,
            dims.num_tokens as i32, dims.hidden as i32, q_dim as i32,
            scratch.gemm_f32_tmp + (dims.num_tokens * dims.hidden * 2) as u64,
            scratch.cutlass_workspace, scratch.cutlass_workspace_bytes,
            stream,
        )?;
        gemma4_launcher::FusedNormAddResidualF16InLaunch {
            num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
        }.launch(norm_add_residual_f16in_kernel, o_out_f16,
            weights.post_attn_norm_gamma, residual, 0, stream)?;
    } else {
        cublaslt.fp8_gemm_f32(
            scratch.attn_out_fp8, weights.o_fp8, scratch.gemm_f32_tmp,
            dims.num_tokens as i32, dims.hidden as i32, q_dim as i32,
            scratch.attn_out_scale, weights.o_scale, stream,
        )?;
        gemma4_launcher::FusedNormAddResidualLaunch {
            num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
        }.launch(norm_add_residual_kernel, scratch.gemm_f32_tmp,
            weights.post_attn_norm_gamma, residual, 0, stream)?;
    }

    #[cfg(feature = "cuda")]
    probe!("after_step8_residual", residual, dims.hidden);

    // 9. pre_feedforward_layernorm -> FP8 quant
    // Same fast-path skip as step 1: gate_up fast path does its own
    // f16 rmsnorm into delta_f16, leaving hidden_fp8/hidden_scale
    // unused.
    #[cfg(feature = "cuda")]
    let skip_ff_quant = dims.num_tokens <= FAST_PATH_M_MAX
        && weights.gate_up_chscale != 0
        && weights.gate_up_blockscale != 0
        && weights.gate_up_f16 == 0
        && kernels.fp8_gemv_wpr_native_f16in.is_some();
    #[cfg(not(feature = "cuda"))]
    let skip_ff_quant = false;
    if !skip_ff_quant {
        FusedRmsnormFp8QuantLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            rmsnorm_quant_kernel,
            scratch.hidden_fp8,
            scratch.hidden_scale,
            residual,
            weights.pre_ff_norm_gamma,
            stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe_f32!("step9_hidden_scale", scratch.hidden_scale);
    #[cfg(feature = "cuda")]
    probe_f32!("step9_gate_up_wscale", weights.gate_up_scale);

    // 10. gate||up projection
    #[cfg(feature = "cuda")]
    if weights.awq.linear_active(AwqLinearKind::Gate) {
        // Cycle 46 step 5b: AWQ gate||up. Two sequential launches into
        // scratch.gate_up_out at offsets 0 and `intermediate * 2`,
        // composing the same fused [gate || up] layout downstream
        // GELU-mul expects.
        debug_assert!(weights.awq.linear_active(AwqLinearKind::Up),
            "AWQ gate/up must be all-or-nothing");
        let prefill_loop_enabled = std::env::var("RVLLM_AWQ_PREFILL_LOOP").is_ok();
        let gemm_available = kernels.awq_int4_gemm_sm120_wmma.is_some();
        if dims.num_tokens != 1 && !gemm_available && !prefill_loop_enabled {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    backend: "AwqInt4GemvF16",
                    op: "awq gate_up prefill (M>1): GEMM kernel PTX absent and RVLLM_AWQ_PREFILL_LOOP not set",
                },
                ctx: rvllm_core::AttnCtx { op: "awq_gate_up", stream,
                    num_seqs: dims.num_tokens, head_dim: dims.head_dim },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let need_gemv = dims.num_tokens == 1
            || (dims.num_tokens > 1 && (!gemm_available || prefill_loop_enabled));
        if need_gemv && kernels.awq_int4_gemv_f16.is_none() {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    backend: "AwqInt4GemvF16", op: "awq_int4_gemv_f16 PTX missing",
                },
                ctx: rvllm_core::AttnCtx { op: "awq_gate_up", stream,
                    num_seqs: dims.num_tokens, head_dim: dims.head_dim },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let fn_awq = kernels.awq_int4_gemv_f16;
        // Same prelude as gate_up FP8 fast path: residual → delta_f16 +
        // pre-FF rmsnorm in place.
        // Cycle 54 Stage 2.1: bf16-residual narrow at branch entry.
        if dims.bf16_residual {
            gemma4_launcher::Bf16ToF16SatLaunch {
                n: dims.num_tokens * dims.hidden,
            }
            .launch(kernels.bf16_to_f16_sat, scratch.delta_f16, residual, stream)?;
        } else {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                scratch.delta_f16, residual,
                (dims.num_tokens * dims.hidden * 2) as _,
                stream as _,
            );
        }
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
        }.launch(
            kernels.fused_rmsnorm, scratch.delta_f16,
            weights.pre_ff_norm_gamma, stream,
        )?;
        let inter = dims.intermediate as u32;
        let k = dims.hidden as u32;
        let g = weights.awq.group_size;
        let use_gemm = dims.num_tokens > 1 && !prefill_loop_enabled && gemm_available;
        unsafe {
            if dims.num_tokens == 1 {
                let f = fn_awq.unwrap();
                // gate -> scratch.gate_up_out + 0
                gemma4_launcher::AwqInt4GemvF16Launch { n: inter, k, group_size: g }
                    .launch(f, scratch.delta_f16,
                        weights.awq.gate_packed, weights.awq.gate_scale, weights.awq.gate_zero,
                        scratch.gate_up_out, stream)?;
                // up -> scratch.gate_up_out + intermediate*2 (f16 = 2B)
                gemma4_launcher::AwqInt4GemvF16Launch { n: inter, k, group_size: g }
                    .launch(f, scratch.delta_f16,
                        weights.awq.up_packed, weights.awq.up_scale, weights.awq.up_zero,
                        scratch.gate_up_out + (inter as u64) * 2, stream)?;
            } else if use_gemm {
                // gate||up prefill via GEMM: ld_d=2*intermediate
                // (fused [gate || up] layout), 2 launches at column
                // offsets 0 / intermediate.
                let fn_gemm = kernels.awq_int4_gemm_sm120_wmma.unwrap();
                let mk_gemm = || gemma4_launcher::AwqInt4GemmSm120WmmaLaunch {
                    m: dims.num_tokens, n: inter, k, group_size: g,
                    ld_d: 2 * inter,
                };
                mk_gemm().launch(fn_gemm, scratch.gate_up_out, scratch.delta_f16,
                    weights.awq.gate_packed, weights.awq.gate_scale, weights.awq.gate_zero,
                    stream)?;
                mk_gemm().launch(fn_gemm,
                    scratch.gate_up_out + (inter as u64) * 2, scratch.delta_f16,
                    weights.awq.up_packed, weights.awq.up_scale, weights.awq.up_zero,
                    stream)?;
            } else {
                // gate||up prefill loop fallback. in_stride=hidden,
                // out_stride=2*intermediate (fused [gate || up] layout
                // downstream GELU-mul expects).
                let mk_loop = |n: u32| gemma4_launcher::AwqInt4GemvF16PrefillLoop {
                    num_tokens: dims.num_tokens, n, k, group_size: g,
                    in_stride_elems: k,
                    out_stride_elems: 2 * inter,
                };
                let f = fn_awq.unwrap();
                mk_loop(inter).launch(f, scratch.delta_f16,
                    weights.awq.gate_packed, weights.awq.gate_scale, weights.awq.gate_zero,
                    scratch.gate_up_out, stream)?;
                mk_loop(inter).launch(f, scratch.delta_f16,
                    weights.awq.up_packed, weights.awq.up_scale, weights.awq.up_zero,
                    scratch.gate_up_out + (inter as u64) * 2, stream)?;
            }
        }
    } else if weights.gate_up_f16 != 0 {
        // F16 path: norm residual into gate_up_out scratch, then F16 GEMM
        // Cycle 54 Stage 2.1: bf16-residual narrow at branch entry. Note
        // the scratch here is `scratch.gate_up_out` (re-used as the
        // pre-norm staging buffer for this branch), not `delta_f16`.
        if dims.bf16_residual {
            gemma4_launcher::Bf16ToF16SatLaunch {
                n: dims.num_tokens * dims.hidden,
            }
            .launch(kernels.bf16_to_f16_sat, scratch.gate_up_out, residual, stream)?;
        } else {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                scratch.gate_up_out,
                residual,
                (dims.num_tokens * dims.hidden * 2) as _,
                stream as _,
            );
        }
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.gate_up_out,
            weights.pre_ff_norm_gamma,
            stream,
        )?;
        cublaslt.f16_gemm_f32(
            scratch.gate_up_out,
            weights.gate_up_f16,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            (2 * dims.intermediate) as i32,
            dims.hidden as i32,
            stream,
        )?;
        gemma4_launcher::Bf16ToF16SatLaunch {
            n: dims.num_tokens * 2 * dims.intermediate,
        }
        .launch(
            kernels.f32_to_f16_sat,
            scratch.gate_up_out,
            scratch.gemm_f32_tmp,
            stream,
        )?;
    } else if let (true, Some(fn_gemv)) = (
        weights.gate_up_blockscale != 0 && dims.num_tokens <= FAST_PATH_M_MAX,
        kernels.fp8_gemv_wpr_native_f16in,
    ) {
        // sm_121 fast path for gate||up projection. Mirrors
        // the QKV fast path: f16 rmsnorm into delta_f16 (pre-FF norm
        // gamma this time), then f16-input fp8_gemv direct to
        // gate_up_out. Downstream `fused_gelu_mul` reads gate_up_out
        // as f16 so no epilogue change is needed.
        // Cycle 54 Stage 2.1: bf16-residual narrow at branch entry.
        if dims.bf16_residual {
            gemma4_launcher::Bf16ToF16SatLaunch {
                n: dims.num_tokens * dims.hidden,
            }
            .launch(kernels.bf16_to_f16_sat, scratch.delta_f16, residual, stream)?;
        } else {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                scratch.delta_f16,
                residual,
                (dims.num_tokens * dims.hidden * 2) as _,
                stream as _,
            );
        }
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.delta_f16,
            weights.pre_ff_norm_gamma,
            stream,
        )?;
        gemma4_launcher::Fp8GemvF16InLaunch {
            m: dims.num_tokens,
            n: 2 * dims.intermediate,
            k: dims.hidden,
        }
        .launch(
            fn_gemv,
            scratch.gate_up_out,
            weights.gate_up_fp8,
            weights.gate_up_blockscale,
            scratch.delta_f16,
            stream,
        )?;
    } else if weights.gate_up_chscale != 0 {
        fp8_gemm_channelscale_or_fallback(
            cutlass, cublaslt, kernels.f32_to_f16_sat, kernels.scale_cols_f32, kernels.scale_rows_f32_ratio,
            scratch.gate_up_out, scratch.hidden_fp8, weights.gate_up_fp8,
            scratch.hidden_scale, weights.gate_up_chscale, weights.gate_up_blockscale, weights.gate_up_scale,
            dims.num_tokens as i32, (2 * dims.intermediate) as i32, dims.hidden as i32,
            scratch.gemm_f32_tmp,
            scratch.cutlass_workspace, scratch.cutlass_workspace_bytes,
            stream,
        )?;
    } else {
        cublaslt.fp8_gemm(
            scratch.hidden_fp8, weights.gate_up_fp8, scratch.gate_up_out,
            dims.num_tokens as i32, (2 * dims.intermediate) as i32, dims.hidden as i32,
            scratch.hidden_scale, weights.gate_up_scale, stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe!("step10_gate_up_out", scratch.gate_up_out, dims.intermediate);

    // 11-12. GELU*up + down_proj
    #[cfg(feature = "cuda")]
    if weights.awq.linear_active(AwqLinearKind::Down) {
        // Cycle 46 step 5b: AWQ down. F16 GELU output staged into
        // gate_up_fp8 (alias trick — same as the f16-weights and
        // f16-gemv-fast-path branches), then 1 AWQ launch into
        // gemm_f32_tmp (used as f16 staging), then the standard
        // f16-in post-FF-norm + residual epilogue.
        let prefill_loop_enabled = std::env::var("RVLLM_AWQ_PREFILL_LOOP").is_ok();
        let gemm_available = kernels.awq_int4_gemm_sm120_wmma.is_some();
        if dims.num_tokens != 1 && !gemm_available && !prefill_loop_enabled {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    backend: "AwqInt4GemvF16",
                    op: "awq down_proj prefill (M>1): GEMM kernel PTX absent and RVLLM_AWQ_PREFILL_LOOP not set",
                },
                ctx: rvllm_core::AttnCtx { op: "awq_down", stream,
                    num_seqs: dims.num_tokens, head_dim: dims.head_dim },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let need_gemv = dims.num_tokens == 1
            || (dims.num_tokens > 1 && (!gemm_available || prefill_loop_enabled));
        if need_gemv && kernels.awq_int4_gemv_f16.is_none() {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    backend: "AwqInt4GemvF16", op: "awq_int4_gemv_f16 PTX missing",
                },
                ctx: rvllm_core::AttnCtx { op: "awq_down", stream,
                    num_seqs: dims.num_tokens, head_dim: dims.head_dim },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let fn_awq = kernels.awq_int4_gemv_f16;
        // f16 GELU(gate)*up into gate_up_fp8 alias.
        {
            let mut out = scratch.gate_up_fp8;
            let mut inp = scratch.gate_up_out;
            let mut inter = dims.intermediate as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                (&mut inter) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block = (dims.intermediate.min(1024), 1, 1);
            let grid = (dims.num_tokens, 1, 1);
            rvllm_fused::launch_raw(kernels.fused_gelu_mul_f16, grid, block, 0, stream, &args)?;
        }
        let use_gemm = dims.num_tokens > 1 && !prefill_loop_enabled && gemm_available;
        unsafe {
            if dims.num_tokens == 1 {
                gemma4_launcher::AwqInt4GemvF16Launch {
                    n: dims.hidden as u32,
                    k: dims.intermediate as u32,
                    group_size: weights.awq.group_size,
                }.launch(fn_awq.unwrap(), scratch.gate_up_fp8,
                    weights.awq.down_packed, weights.awq.down_scale, weights.awq.down_zero,
                    scratch.gemm_f32_tmp, stream)?;
            } else if use_gemm {
                // down prefill via GEMM: ld_d=hidden (output is
                // [num_tokens, hidden] contiguous in gemm_f32_tmp).
                gemma4_launcher::AwqInt4GemmSm120WmmaLaunch {
                    m: dims.num_tokens,
                    n: dims.hidden as u32,
                    k: dims.intermediate as u32,
                    group_size: weights.awq.group_size,
                    ld_d: dims.hidden as u32,
                }.launch(kernels.awq_int4_gemm_sm120_wmma.unwrap(),
                    scratch.gemm_f32_tmp, scratch.gate_up_fp8,
                    weights.awq.down_packed, weights.awq.down_scale, weights.awq.down_zero,
                    stream)?;
            } else {
                // down prefill loop fallback. in_stride=intermediate
                // (gate_up_fp8 alias holds [num_tokens, intermediate]
                // f16 GELU output), out_stride=hidden (gemm_f32_tmp
                // staged).
                gemma4_launcher::AwqInt4GemvF16PrefillLoop {
                    num_tokens: dims.num_tokens,
                    n: dims.hidden as u32,
                    k: dims.intermediate as u32,
                    group_size: weights.awq.group_size,
                    in_stride_elems:  dims.intermediate as u32,
                    out_stride_elems: dims.hidden as u32,
                }.launch(fn_awq.unwrap(), scratch.gate_up_fp8,
                    weights.awq.down_packed, weights.awq.down_scale, weights.awq.down_zero,
                    scratch.gemm_f32_tmp, stream)?;
            }
        }
        gemma4_launcher::FusedNormAddResidualF16InLaunch {
            num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
        }.launch(
            norm_add_residual_f16in_kernel, scratch.gemm_f32_tmp,
            weights.post_ff_norm_gamma, residual,
            weights.layer_scalar_ptr, stream,
        )?;
    } else if weights.down_f16 != 0 {
        // F16 path: GELU output to separate buffer (can't alias gate_up_out)
        {
            let mut out = scratch.gate_up_fp8; // use gate_up_fp8 scratch as f16 gelu output
            let mut inp = scratch.gate_up_out;
            let mut inter = dims.intermediate as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                (&mut inter) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block = (dims.intermediate.min(1024), 1, 1);
            let grid = (dims.num_tokens, 1, 1);
            rvllm_fused::launch_raw(kernels.fused_gelu_mul_f16, grid, block, 0, stream, &args)?;
        }
        // F16 GEMM for down_proj (reads from gate_up_fp8 where GELU output was stored)
        cublaslt.f16_gemm_f32(
            scratch.gate_up_fp8,
            weights.down_f16,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            dims.hidden as i32,
            dims.intermediate as i32,
            stream,
        )?;
    } else if let (true, Some(fn_gemv)) = (
        weights.down_blockscale != 0 && dims.num_tokens <= FAST_PATH_M_MAX,
        kernels.fp8_gemv_wpr_native_f16in,
    ) {
        // sm_121 fast path for down projection.
        // Skip FP8 GELU-quant — run f16 GELU into `gate_up_fp8`
        // scratch (same aliasing trick as the f16-weights branch),
        // then f16-input fp8_gemv writes f16 directly to
        // `gemm_f32_tmp` (reused as f16 scratch), and
        // `fused_norm_add_residual_f16in` rolls the residual add +
        // post-FF norm in one pass.
        {
            let mut out = scratch.gate_up_fp8;
            let mut inp = scratch.gate_up_out;
            let mut inter = dims.intermediate as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                (&mut inter) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block = (dims.intermediate.min(1024), 1, 1);
            let grid = (dims.num_tokens, 1, 1);
            rvllm_fused::launch_raw(kernels.fused_gelu_mul_f16, grid, block, 0, stream, &args)?;
        }
        gemma4_launcher::Fp8GemvF16InLaunch {
            m: dims.num_tokens,
            n: dims.hidden,
            k: dims.intermediate,
        }
        .launch(
            fn_gemv,
            scratch.gemm_f32_tmp,
            weights.down_fp8,
            weights.down_blockscale,
            scratch.gate_up_fp8,
            stream,
        )?;
        gemma4_launcher::FusedNormAddResidualF16InLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            norm_add_residual_f16in_kernel,
            scratch.gemm_f32_tmp,
            weights.post_ff_norm_gamma,
            residual,
            weights.layer_scalar_ptr,
            stream,
        )?;
    } else {
        // FP8 path
        gemma4_launcher::FusedGeluMulFp8QuantLaunch {
            num_tokens: dims.num_tokens,
            intermediate: dims.intermediate,
        }
        .launch(
            kernels.fused_gelu_mul,
            scratch.mlp_out_fp8,
            scratch.mlp_out_scale,
            scratch.gate_up_out,
            stream,
        )?;
        if weights.down_chscale != 0 {
            // Same reroute as O-proj: use fp8_gemm_channelscale_or_fallback
            // so the M>=128 batch path picks CUTLASS + full 2-D blockscale
            // (when available) instead of the cuBLASLt + channelscale
            // approximation that collapses K-block variation.
            let down_out_f16 = scratch.gemm_f32_tmp;
            fp8_gemm_channelscale_or_fallback(
                cutlass, cublaslt, kernels.f32_to_f16_sat, kernels.scale_cols_f32, kernels.scale_rows_f32_ratio,
                down_out_f16, scratch.mlp_out_fp8, weights.down_fp8,
                scratch.mlp_out_scale, weights.down_chscale, weights.down_blockscale, weights.down_scale,
                dims.num_tokens as i32, dims.hidden as i32, dims.intermediate as i32,
                scratch.gemm_f32_tmp + (dims.num_tokens * dims.hidden * 2) as u64,
                scratch.cutlass_workspace, scratch.cutlass_workspace_bytes,
                stream,
            )?;
            gemma4_launcher::FusedNormAddResidualF16InLaunch {
                num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
            }.launch(norm_add_residual_f16in_kernel, down_out_f16,
                weights.post_ff_norm_gamma, residual,
                weights.layer_scalar_ptr, stream)?;
        } else {
            cublaslt.fp8_gemm_f32(
                scratch.mlp_out_fp8, weights.down_fp8, scratch.gemm_f32_tmp,
                dims.num_tokens as i32, dims.hidden as i32, dims.intermediate as i32,
                scratch.mlp_out_scale, weights.down_scale, stream,
            )?;
            gemma4_launcher::FusedNormAddResidualLaunch {
                num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
            }.launch(norm_add_residual_kernel, scratch.gemm_f32_tmp,
                weights.post_ff_norm_gamma, residual,
                weights.layer_scalar_ptr, stream)?;
        }
    }

    #[cfg(feature = "cuda")]
    probe!("after_step14_residual", residual, dims.hidden);

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (cublaslt, qkv_rows, _kv_dim);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
unsafe fn launch_scale_cols_f32(
    kernel: KernelFn,
    data: u64,
    scale: u64,
    m: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    let total = m * n;
    let mut data = data;
    let mut scale = scale;
    let mut m_i = m as i32;
    let mut n_i = n as i32;
    let args = [
        (&mut data) as *mut u64 as *mut core::ffi::c_void,
        (&mut scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
        (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
    ];
    let block = (256u32, 1, 1);
    let grid = ((total + 255) / 256, 1, 1);
    rvllm_fused::launch_raw(kernel, grid, block, 0, stream, &args)
}

/// Post-GEMM per-row scale RATIO correction: `data[m, n] *=
/// scale[m] / scale[0]` for an MxN f32 row-major buffer. Used to
/// fix up cuBLASLt FP8 GEMM output when the B_SCALE was a per-token
/// array but cuBLASLt ran in SCALAR mode (sm_121's only option).
#[cfg(feature = "cuda")]
unsafe fn launch_scale_rows_f32_ratio(
    kernel: KernelFn,
    data: u64,
    scale: u64,
    m: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    let total = m * n;
    let mut data = data;
    let mut scale = scale;
    let mut m_i = m as i32;
    let mut n_i = n as i32;
    let args = [
        (&mut data) as *mut u64 as *mut core::ffi::c_void,
        (&mut scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
        (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
    ];
    let block = (256u32, 1, 1);
    let grid = ((total + 255) / 256, 1, 1);
    rvllm_fused::launch_raw(kernel, grid, block, 0, stream, &args)
}

/// CUTLASS FP8 GEMM with row×col scale epilogue + sm_121 fallback.
///
/// On SM80/89/90 this delegates straight to the CUTLASS `.so`
/// wrapper (`cutlass_fp8_gemm_channelscale`), which applies the full
/// per-token row scale × per-channel column scale epilogue.
///
/// On sm_121 (`CutlassBackend::Absent`) cuBLASLt's FP8
/// `OUTER_VEC_32F` channelscale path doesn't produce a matmul
/// heuristic for Blackwell consumer (`heuristic LaunchFailed`), so
/// we fall back to the **scalar-scaled** FP8 matmul:
/// `cublaslt.fp8_gemm(..., a_scale, b_scale_scalar)`. The
/// per-channel b_chscale vector is intentionally dropped — this
/// incurs a known PPL regression vs the CUTLASS path, documented
/// as a follow-up. A proper channelscale kernel for sm_121 (native
/// mma.sync + column-scale epilogue) is the next GB10 perf PR.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn fp8_gemm_channelscale_or_fallback(
    cutlass: &CutlassBackend,
    cublaslt: &CublasLt,
    fn_f32_to_f16: KernelFn,
    fn_scale_cols_f32: KernelFn,
    fn_scale_rows_f32_ratio: KernelFn,
    output_f16: u64,
    a_fp8: u64,
    b_fp8: u64,
    a_scale: u64,
    b_chscale: u64,
    // Optional per-(N/128, K/128) blockscale, row-major, passed DIRECTLY
    // to CUTLASS prep_sfb. `0` when weights were fused (e.g. QKV /
    // gate_up where multi-part blockwise reconstruction is undefined);
    // in that case the CUTLASS path is skipped and we fall through to
    // the channelscale-preserving fallback. Not interchangeable with
    // `b_chscale` — the latter is [rows] per-row scale, the former is
    // [n_blocks, k_blocks] 2-D block scale.
    b_blockscale: u64,
    b_scale_scalar: u64,
    m: i32,
    n: i32,
    k: i32,
    scratch_f32: u64,
    cutlass_workspace: u64,
    cutlass_workspace_bytes: usize,
    stream: u64,
) -> Result<()> {
    // Safety rail from PR review: `b_chscale != 0` means the real
    // per-channel scale lives in the vector; `b_scale_scalar` is a
    // sentinel 1.0 set by the loader. Any fallback arm that ignores
    // `b_chscale` would multiply raw FP8 bytes by 1.0 and emit
    // garbage. Every arm below must either
    //   (a) consume `b_chscale` (the full row×col channelscale path),
    //   (b) be guaranteed unreachable when `b_chscale != 0`, or
    //   (c) fall through to the channelscale-preserving f32+scale_cols
    //       path at the tail of this function.
    // The cuBLASLt scalar `fp8_gemm(..., b_scale_scalar)` shortcut is
    // reserved for the `b_chscale == 0` case (true scalar-scale
    // weights).
    debug_assert!(
        b_chscale != 0 || b_scale_scalar != 0,
        "fp8_gemm_channelscale_or_fallback called with both b_chscale \
         and b_scale_scalar == 0 — no scale source",
    );
    // CUTLASS SM120 blockwise path — default ON when a `SoSm120`
    // backend is loaded with all four prep symbols present (SFA bytes /
    // SFB bytes / prep SFA / prep SFB). Older `libcutlass_sm120.so`
    // builds without the prep helpers fall through to cuBLASLt.
    // Measured on GB10 Gemma 4 31B decode: batch=128 461→555 tok/s
    // (+20%), batch=256 665→909 tok/s (+37%). Opt-OUT via
    // `RVLLM_FP8_GEMM_CUTLASS_SM120=0` for regression diagnosis.
    //
    // The CUTLASS cooperative blockwise kernel is built with a 128×128
    // MMA tile and hard-asserts `M >= 128` (sm90_gemm_tma_warpspecialized_
    // cooperative.hpp:371). Gate the dispatch so small-batch decode
    // (M=num_seqs < 128) still gets routed through the cuBLASLt
    // fallback below.
    // CUTLASS blockwise path requires a proper 2-D b_blockscale. When
    // the loader fused a weight (QKV / gate_up) it cannot reconstruct
    // a consistent blockscale across parts (different shards ship
    // different block alignments), so blockscale_ptr is `None` →
    // b_blockscale == 0 here. In that case fall through to cuBLASLt
    // channelscale fallback — feeding the per-row channelscale to
    // CUTLASS's prep_sfb produces silently-wrong output (the kernel
    // interprets it as [n_blocks, k_blocks] and reads completely
    // unrelated row entries for the K/V output regions, which is
    // exactly how aa01001pftrope0 manifested).
    let cutlass_sm120_enabled = m >= 128
        && b_blockscale != 0
        && crate::gemma4_bring_up::parse_truthy_env("RVLLM_FP8_GEMM_CUTLASS_SM120")
            .unwrap_or(true);
    if cutlass_sm120_enabled {
        if let CutlassBackend::SoSm120(lib) = cutlass {
            if lib.prep_sfa.is_some() && lib.prep_sfb.is_some() {
                let sfa_bytes = lib.sfa_bytes(m, k);
                let _sfb_bytes = lib.sfb_bytes(n, k);
                // 16-byte-align the SFB offset inside scratch_f32.
                let sfa_aligned = (sfa_bytes + 15) & !15;
                let sfa_ptr = scratch_f32;
                let sfb_ptr = scratch_f32 + sfa_aligned as u64;
                lib.launch_prep_sfa(a_scale, sfa_ptr, m, k, stream)?;
                lib.launch_prep_sfb(b_blockscale, sfb_ptr, n, k, stream)?;
                return cutlass.launch_fp8_gemm_blockscale_sm120(
                    output_f16,
                    a_fp8,
                    b_fp8,
                    sfa_ptr,
                    sfb_ptr,
                    m,
                    n,
                    k,
                    cutlass_workspace,
                    cutlass_workspace_bytes,
                    stream,
                );
            }
        }
    }
    // Channelscale-preserving fallback for the sm_121 / SoSm120
    // paths. cuBLASLt's scalar `fp8_gemm(..., b_scale_scalar)` is
    // the shortcut when there's no per-channel scale to apply; when
    // `b_chscale != 0` we can NOT use that shortcut (see the
    // safety rail at the top of this fn — raw FP8 × 1.0 would
    // land in the output). Instead route through:
    //   1. FP8 GEMM into the f32 scratch region, no per-channel
    //      scale yet → result has baked-in a_scale × 1.0.
    //   2. scale_cols_f32 multiplies each column by `b_chscale[n]`.
    //   3. cast f32 → f16 into the actual output buffer.
    // Slower than a fused CUTLASS channelscale GEMM but correct.
    // `scratch_f32` must be sized for `m * n * sizeof(f32)` — the
    // caller guarantees this (same `gemm_f32_tmp` region the
    // existing fp8_gemm_f32 path uses).
    let fallback_f32_scale_cast = || -> Result<()> {
        cublaslt.fp8_gemm_f32(
            a_fp8, b_fp8, scratch_f32,
            m, n, k,
            a_scale, b_scale_scalar,
            stream,
        )?;
        // cuBLASLt on sm_121 only supports SCALAR B_SCALE mode; the
        // FP8 GEMM above applied `a_scale[0]` uniformly to every
        // output row. For M>1 with per-token activation scales this
        // under/over-scales rows 1..M-1 by `a_scale[m] / a_scale[0]`.
        // Correct it with the ratio kernel. At M=1 the kernel is a
        // no-op (row 0 stays put) so decode stays bit-identical.
        if m > 1 {
            launch_scale_rows_f32_ratio(
                fn_scale_rows_f32_ratio,
                scratch_f32, a_scale,
                m as u32, n as u32, stream,
            )?;
        }
        if b_chscale != 0 {
            launch_scale_cols_f32(
                fn_scale_cols_f32,
                scratch_f32, b_chscale,
                m as u32, n as u32, stream,
            )?;
        }
        // `Bf16ToF16SatLaunch` has the (dst, src, n) ABI we need;
        // name refers to the historical caller, the launched kernel
        // is `f32_to_f16_sat`.
        gemma4_launcher::Bf16ToF16SatLaunch {
            n: (m * n) as u32,
        }
        .launch(fn_f32_to_f16, output_f16, scratch_f32, stream)
    };

    match cutlass {
        CutlassBackend::So(_) => cutlass.launch_fp8_gemm_channelscale(
            output_f16,
            a_fp8,
            b_fp8,
            a_scale,
            b_chscale,
            m,
            n,
            k,
            cutlass_workspace,
            cutlass_workspace_bytes,
            stream,
        ),
        CutlassBackend::Absent => {
            if b_chscale != 0 {
                fallback_f32_scale_cast()
            } else {
                cublaslt.fp8_gemm(
                    a_fp8, b_fp8, output_f16,
                    m, n, k,
                    a_scale, b_scale_scalar, stream,
                )
            }
        }
        // CUTLASS SM120 blockwise kernel — the .so is loaded and
        // dispatchable, but its SFA ABI wants a [ceil(M/128), K/128] f32
        // tensor, not the per-M a_scale vector we have here. That
        // broadcast has to be staged by the caller (future: prefill
        // scratch region + a K/128-wide broadcast kernel), so for now
        // we route through the f32+scale_cols path which preserves
        // channelscale correctly. The `RVLLM_FP8_GEMM_CUTLASS_SM120`
        // opt-in at the top takes the SoSm120 fast path for M>=128;
        // below that we land here.
        CutlassBackend::SoSm120(_) => {
            if b_chscale != 0 {
                fallback_f32_scale_cast()
            } else {
                cublaslt.fp8_gemm(
                    a_fp8, b_fp8, output_f16,
                    m, n, k,
                    a_scale, b_scale_scalar, stream,
                )
            }
        }
        // Exhaustiveness for #[non_exhaustive] CutlassBackend — a
        // future variant lands here with a typed error rather than
        // aborting the process, so adding a new backend can never
        // silently panic in prod. The explicit arms above stay the
        // source of truth; this is the default-deny for unknowns.
        _ => Err(rvllm_core::RvllmError::cutlass(
            rvllm_core::CutlassError::FeatureNotAvailable {
                op: "fp8_gemm_channelscale (unknown CutlassBackend variant)",
            },
            rvllm_core::CutlassCtx {
                kernel: "fp8_gemm_channelscale",
                stream,
            },
        )),
    }
}

#[cfg(feature = "cuda")]
unsafe fn rope_f16kv(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    stream: u64,
) -> Result<()> {
    let mut q_in = scratch.q_normed;
    let mut k_in = scratch.k_normed;
    let mut v_in = scratch.v_normed;
    let mut q_out = scratch.q_normed;
    let mut k_cache = scratch.k_cache;
    let mut v_cache = scratch.v_cache;
    let mut cos = meta.cos;
    let mut sin = meta.sin;
    let mut positions = meta.positions;
    let mut slot_mapping = meta.slot_mapping;
    let mut nt = dims.num_tokens as i32;
    let mut nh = dims.num_heads as i32;
    let mut nkvh = dims.num_kv_heads as i32;
    let mut hd = dims.head_dim as i32;
    let mut rd = dims.rotary_dim as i32;
    let args = [
        (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
        (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
        (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
        (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
        (&mut k_cache) as *mut u64 as *mut core::ffi::c_void,
        (&mut v_cache) as *mut u64 as *mut core::ffi::c_void,
        (&mut cos) as *mut u64 as *mut core::ffi::c_void,
        (&mut sin) as *mut u64 as *mut core::ffi::c_void,
        (&mut positions) as *mut u64 as *mut core::ffi::c_void,
        (&mut slot_mapping) as *mut u64 as *mut core::ffi::c_void,
        (&mut nt) as *mut i32 as *mut core::ffi::c_void,
        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
        (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
        (&mut hd) as *mut i32 as *mut core::ffi::c_void,
        (&mut rd) as *mut i32 as *mut core::ffi::c_void,
    ];
    let max_heads = dims.num_heads.max(dims.num_kv_heads);
    let grid = (dims.num_tokens, max_heads, 1);
    let block = ((dims.head_dim / 2).max(32), 1, 1);
    rvllm_fused::launch_raw(kernels.fused_rope_partial_f16kv, grid, block, 0, stream, &args)
}

// === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
/// Ground-truth f16 KV write for a shadow region. Identical to
/// `rope_f16kv` except K/V pointers are overridden to the shadow
/// region and Q output goes to `scratch.shadow_q_cache` (per-layer
/// slot on step-0 capture, shared throwaway otherwise).
///
/// CRITICAL: `q_out` must NOT alias `scratch.q_normed`. An earlier
/// revision aliased them, which caused the subsequent primary
/// `rope_nvfp4kv` to apply RoPE a second time on an already-rotated
/// Q. Result: `q_fp8` was double-RoPE'd and every forward pass
/// silently produced wrong attention whenever the shadow gate was on.
/// Caller must guarantee `scratch.shadow_q_cache` is non-zero and
/// points to a valid f16 buffer of at least
/// `num_tokens * num_heads * head_dim * 2` bytes.
#[cfg(feature = "cuda")]
unsafe fn rope_f16kv_shadow(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    stream: u64,
) -> Result<()> {
    debug_assert!(scratch.shadow_q_cache != 0,
        "rope_f16kv_shadow called with shadow_q_cache=0; caller must route to \
         per-layer slot or throwaway to avoid aliasing q_normed");
    let mut q_in = scratch.q_normed;
    let mut k_in = scratch.k_normed;
    let mut v_in = scratch.v_normed;
    let mut q_out = scratch.shadow_q_cache;
    let mut k_cache = scratch.shadow_k_cache;
    let mut v_cache = scratch.shadow_v_cache;
    let mut cos = meta.cos;
    let mut sin = meta.sin;
    let mut positions = meta.positions;
    let mut slot_mapping = meta.slot_mapping;
    let mut nt = dims.num_tokens as i32;
    let mut nh = dims.num_heads as i32;
    let mut nkvh = dims.num_kv_heads as i32;
    let mut hd = dims.head_dim as i32;
    let mut rd = dims.rotary_dim as i32;
    let args = [
        (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
        (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
        (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
        (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
        (&mut k_cache) as *mut u64 as *mut core::ffi::c_void,
        (&mut v_cache) as *mut u64 as *mut core::ffi::c_void,
        (&mut cos) as *mut u64 as *mut core::ffi::c_void,
        (&mut sin) as *mut u64 as *mut core::ffi::c_void,
        (&mut positions) as *mut u64 as *mut core::ffi::c_void,
        (&mut slot_mapping) as *mut u64 as *mut core::ffi::c_void,
        (&mut nt) as *mut i32 as *mut core::ffi::c_void,
        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
        (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
        (&mut hd) as *mut i32 as *mut core::ffi::c_void,
        (&mut rd) as *mut i32 as *mut core::ffi::c_void,
    ];
    let max_heads = dims.num_heads.max(dims.num_kv_heads);
    let grid = (dims.num_tokens, max_heads, 1);
    let block = ((dims.head_dim / 2).max(32), 1, 1);
    rvllm_fused::launch_raw(kernels.fused_rope_partial_f16kv, grid, block, 0, stream, &args)
}
// === END NVFP4 SHADOW DIAGNOSTIC ===

#[cfg(feature = "cuda")]
unsafe fn rope_fp8kv(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    stream: u64,
) -> Result<()> {
    gemma4_launcher::FusedRopePartialFp8KvLaunch {
        num_tokens: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        rotary_dim: dims.rotary_dim,
    }
    .launch(
        kernels.fused_rope_partial_fp8kv,
        scratch.q_normed,
        scratch.k_normed,
        scratch.v_normed,
        scratch.q_fp8,
        scratch.k_cache,
        scratch.v_cache,
        scratch.k_scale_cache,
        scratch.v_scale_cache,
        scratch.q_scale_cache,
        meta.cos,
        meta.sin,
        meta.positions,
        meta.slot_mapping,
        scratch.q_scale_ptr,
        stream,
    )
}

/// RoPE + NVFP4 paged-KV write. Mirrors `rope_fp8kv` but writes the
/// K/V cache as packed 4-bit bytes + per-16-element E4M3 microscale
/// into the separate `k_cache_scale` / `v_cache_scale` regions. Q
/// still lands in `q_fp8` per-tensor (the FA2 NVFP4 decode/prefill
/// kernels read Q as FP8).
#[cfg(feature = "cuda")]
unsafe fn rope_nvfp4kv(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    stream: u64,
) -> Result<()> {
    // `fused_rope_partial_nvfp4kv` is `None` when the NVFP4 PTX isn't
    // built into $KERNELS_DIR. That means the operator asked for
    // `RVLLM_NVFP4_KV=1` but their kernel tree predates the NVFP4
    // branch — fail clean with a typed attention error (closest
    // match in rvllm-core; we don't have a KernelError variant).
    let kernel = kernels.fused_rope_partial_nvfp4kv.ok_or_else(|| {
        rvllm_core::RvllmError::Attention {
            err: rvllm_core::AttentionError::FeatureNotAvailable {
                backend: "Fa2Ptx",
                op: "rope_nvfp4kv (fused_rope_partial_nvfp4kv.ptx not in $KERNELS_DIR)",
            },
            ctx: rvllm_core::AttnCtx {
                op: "rope_nvfp4kv",
                stream,
                num_seqs: dims.num_tokens,
                head_dim: dims.head_dim,
            },
            bt: std::backtrace::Backtrace::capture(),
        }
    })?;
    let mut q_in = scratch.q_normed;
    let mut k_in = scratch.k_normed;
    // Gemma 4 applies a parameter-free v-norm BEFORE writing V to the
    // KV cache — the FP8 rope reads `scratch.v_normed` for the same
    // reason. Reading pre-norm `v_out` here put raw QKV-GEMM output
    // into the NVFP4 V cache with a different magnitude distribution
    // than the attention kernel expected; that's where the
    // end-to-end PPL blow-up came from (71 FP8 → 6.4M NVFP4).
    let mut v_in = scratch.v_normed;
    let mut q_out = scratch.q_fp8;
    let mut k_packed = scratch.k_cache;
    let mut v_packed = scratch.v_cache;
    let mut k_scale = scratch.k_cache_scale;
    let mut v_scale = scratch.v_cache_scale;
    let mut cos = meta.cos;
    let mut sin = meta.sin;
    let mut positions = meta.positions;
    let mut slot_mapping = meta.slot_mapping;
    let mut q_scale_ptr = scratch.q_scale_ptr;
    // === DYNAMIC NVFP4 Q SCALE ===
    // When `RVLLM_PER_TOKEN_Q_SCALE=1`, pass `scratch.q_scale_cache`
    // so the rope kernel computes a fresh per-(token, head) FP8 Q
    // scale. Required for `RVLLM_NVFP4_HADAMARD=1` because rotated Q
    // values can saturate the static scalar (default 0.1). When 0,
    // the kernel reads the static `q_scale_ptr` (pre-Hadamard
    // behaviour, byte-identical to the prior path).
    let per_token_q_scale =
        crate::gemma4_bring_up::per_token_q_scale_enabled(/*default_on=*/false);
    let mut q_scale_cache_arg: u64 =
        if per_token_q_scale { scratch.q_scale_cache } else { 0 };
    // === END DYNAMIC NVFP4 Q SCALE ===
    // === HADAMARD ROTATION ===
    // Per-layer ±1 sign vectors for signed Walsh-Hadamard rotation
    // of Q and K post-RoPE (NVFP4 path only). Both are 0 when
    // disabled (master env gate `RVLLM_NVFP4_HADAMARD` off, OR
    // sign tables not allocated). Rope kernel's `hadamard_on`
    // requires BOTH non-null, so passing 0 cleanly disables.
    let mut hadamard_signs_q = scratch.hadamard_signs_q;
    let mut hadamard_signs_k = scratch.hadamard_signs_k;
    // V rotation gate (companion: post-attention hadamard_unrotate_f16
    // multiplies attn_out by R^T before O-proj). Default OFF until the
    // dispatch is wired AND quality-validated. RVLLM_NVFP4_HADAMARD_V=1
    // turns it on. The rope kernel reads a separate i32 param so we
    // can A/B without touching the Q+K rotation.
    let mut rotate_v: i32 =
        if crate::gemma4_bring_up::parse_truthy_env("RVLLM_NVFP4_HADAMARD_V")
            .unwrap_or(false) { 1 } else { 0 };
    // === END HADAMARD ROTATION ===
    // === CYCLE 28 PRE-QUANT K SIDECAR ===
    // Reuse the shadow K region: when shadow is enabled, primary
    // overwrites what rope_f16kv_shadow wrote with the EXACT f32 K
    // value (cast to f16) the NVFP4 quantizer is about to consume,
    // post-RoPE + post-optional-Hadamard. Layout matches
    // (slot * num_kv_heads + head) * head_dim. Analyzer then compares
    // this f16 prequant against its Python dequant of the packed
    // nibbles + E4M3 scales — apples-to-apples within the same path.
    // Layout is identical to shadow's so files dumped by ShadowDumper
    // (`layer_{L}_k_shadow.bin`) now contain pre-quant K under this
    // gate. Shadow K vs shadow comparison is lost in this mode but
    // that comparison was uncorrelated anyway (codex review).
    let mut debug_k_prequant: u64 = scratch.shadow_k_cache;
    // === END CYCLE 28 ===
    // === CYCLE 29 V PRE-QUANT SIDECAR ===
    // Mirror the cycle-28 K patch for V. Reuses scratch.shadow_v_cache —
    // shadow region holds primary's pre-quant V (post-optional-Hadamard,
    // V never gets RoPE) instead of shadow_rope's f16 V output.
    let mut debug_v_prequant: u64 = scratch.shadow_v_cache;
    // === END CYCLE 29 ===
    // === CYCLE 31 STOCHASTIC V ROUNDING ===
    // Opt-in via RVLLM_NVFP4_STOCH_ROUND_V=1. Default OFF (deterministic
    // round-to-nearest preserves byte-identical legacy behavior). Seed
    // is per-(slot,head,lane) so prefix-cache reuse stays consistent.
    let mut stoch_round_v: i32 =
        if crate::gemma4_bring_up::parse_truthy_env("RVLLM_NVFP4_STOCH_ROUND_V")
            .unwrap_or(false) { 1 } else { 0 };
    // === END CYCLE 31 ===
    let mut nt = dims.num_tokens as i32;
    let mut nh = dims.num_heads as i32;
    let mut nkvh = dims.num_kv_heads as i32;
    let mut hd = dims.head_dim as i32;
    let mut rd = dims.rotary_dim as i32;
    // NVFP4 scale policy — read env lazily. Stable across a run once
    // chosen. `amax6` (0) = OCP baseline (range-preserving, outlier-
    // insensitive — produces garbage on Gemma 4 at 15k). `mse` (1) =
    // blockwise MSE search over 4 outlier-aware candidates (see
    // fused_rope_partial_nvfp4kv.cu for the candidate set).
    fn parse_policy(v: &str) -> Option<i32> {
        match v {
            "amax6" | "0" => Some(0),
            "mse" | "1" => Some(1),
            _ => None,
        }
    }
    let global_policy: i32 = std::env::var("RVLLM_NVFP4_SCALE_POLICY")
        .ok()
        .and_then(|s| parse_policy(&s))
        .unwrap_or(0);
    let mut scale_policy: i32 = std::env::var("RVLLM_NVFP4_K_SCALE_POLICY")
        .ok()
        .and_then(|s| parse_policy(&s))
        .unwrap_or(global_policy);
    let mut v_scale_policy: i32 = std::env::var("RVLLM_NVFP4_V_SCALE_POLICY")
        .ok()
        .and_then(|s| parse_policy(&s))
        .unwrap_or(global_policy);
    let args = [
        (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
        (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
        (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
        (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
        (&mut k_packed) as *mut u64 as *mut core::ffi::c_void,
        (&mut v_packed) as *mut u64 as *mut core::ffi::c_void,
        (&mut k_scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut v_scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut cos) as *mut u64 as *mut core::ffi::c_void,
        (&mut sin) as *mut u64 as *mut core::ffi::c_void,
        (&mut positions) as *mut u64 as *mut core::ffi::c_void,
        (&mut slot_mapping) as *mut u64 as *mut core::ffi::c_void,
        (&mut q_scale_ptr) as *mut u64 as *mut core::ffi::c_void,
        // === DYNAMIC NVFP4 Q SCALE ===
        (&mut q_scale_cache_arg) as *mut u64 as *mut core::ffi::c_void,
        // === END DYNAMIC NVFP4 Q SCALE ===
        (&mut nt) as *mut i32 as *mut core::ffi::c_void,
        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
        (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
        (&mut hd) as *mut i32 as *mut core::ffi::c_void,
        (&mut rd) as *mut i32 as *mut core::ffi::c_void,
        (&mut scale_policy) as *mut i32 as *mut core::ffi::c_void,
        (&mut v_scale_policy) as *mut i32 as *mut core::ffi::c_void,
        // === HADAMARD ROTATION ===
        (&mut hadamard_signs_q) as *mut u64 as *mut core::ffi::c_void,
        (&mut hadamard_signs_k) as *mut u64 as *mut core::ffi::c_void,
        (&mut rotate_v) as *mut i32 as *mut core::ffi::c_void,
        // === END HADAMARD ROTATION ===
        // === CYCLE 28 PRE-QUANT SIDECAR ===
        (&mut debug_k_prequant) as *mut u64 as *mut core::ffi::c_void,
        // === END CYCLE 28 ===
        // === CYCLE 29 V PRE-QUANT SIDECAR ===
        (&mut debug_v_prequant) as *mut u64 as *mut core::ffi::c_void,
        // === END CYCLE 29 ===
        // === CYCLE 31 STOCHASTIC V ROUNDING ===
        (&mut stoch_round_v) as *mut i32 as *mut core::ffi::c_void,
        // === END CYCLE 31 ===
    ];
    let max_heads = dims.num_heads.max(dims.num_kv_heads);
    let grid = (dims.num_tokens, max_heads, 1);
    let block = (dims.head_dim, 1, 1);
    rvllm_fused::launch_raw(kernel, grid, block, 0, stream, &args)
}

/// Apply R^T = diag(D)·H to attn_out per (token, head) in place. Companion
/// to the V-rotation in `rope_nvfp4kv`: V_cache = V·R, attn_out = P·V·R,
/// this strips the trailing R so the O-projection sees raw P·V.
///
/// No-op (returns Ok) when:
///   * `RVLLM_NVFP4_HADAMARD_V` is not truthy, OR
///   * the un-rotation kernel isn't loaded (PTX absent), OR
///   * `hadamard_signs_k` is null (rotation tables not allocated).
///
/// Each branch must match the rope kernel's V-rotate gate or attn_out
/// gets mangled. Reuses `signs_k` because rope's V-arm shares the same
/// sign vector as K (single per-layer D, both Q and V/K share when
/// rotate_v is on).
#[cfg(feature = "cuda")]
unsafe fn unrotate_attn_out_v_if_enabled(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    scratch: &Gemma4LayerScratch,
    stream: u64,
) -> Result<()> {
    let rotate_v = crate::gemma4_bring_up::parse_truthy_env("RVLLM_NVFP4_HADAMARD_V")
        .unwrap_or(false);
    if !rotate_v { return Ok(()); }
    let kernel = match kernels.hadamard_unrotate_f16 {
        Some(k) => k,
        None => return Ok(()),
    };
    if scratch.hadamard_signs_k == 0 { return Ok(()); }
    let mut x = scratch.attn_out;
    let mut signs = scratch.hadamard_signs_k;
    let mut nt = dims.num_tokens as i32;
    let mut nh = dims.num_heads as i32;
    let mut hd = dims.head_dim as i32;
    let args = [
        (&mut x) as *mut u64 as *mut core::ffi::c_void,
        (&mut signs) as *mut u64 as *mut core::ffi::c_void,
        (&mut nt) as *mut i32 as *mut core::ffi::c_void,
        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
        (&mut hd) as *mut i32 as *mut core::ffi::c_void,
    ];
    let grid = (dims.num_tokens, dims.num_heads, 1);
    let block = (dims.head_dim, 1, 1);
    rvllm_fused::launch_raw(kernel, grid, block, 0, stream, &args)
}

pub unsafe fn logit_softcap(
    kernel: KernelFn,
    logits_ptr: u64,
    num_tokens: u32,
    vocab: u32,
    cap: f32,
    stream: u64,
) -> Result<()> {
    gemma4_launcher::LogitSoftcapLaunch {
        num_tokens,
        vocab,
        cap,
    }
    .launch(kernel, logits_ptr, stream)
}

#[cfg(test)]
mod validate_tests {
    use super::*;

    fn good() -> Gemma4LayerDims {
        Gemma4LayerDims {
            num_tokens: 1,
            hidden: 5376,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 256,
            rotary_dim: 256,
            intermediate: 14336,
            block_size: 16,
            max_blocks_per_seq: 1024,
            num_blocks_total: 1024,
            current_max_context_len: Some(1024),
            attn_scale: 0.0625,
            rms_eps: 1e-6,
            layer_type: Gemma4LayerType::SlidingAttention,
            sliding_window: 1024,
            f16_kv: false,
            kv_dtype: KvDtype::Fp8,
            bf16_residual: false,
        }
    }

    #[test]
    fn good_dims_pass() {
        good().validate().unwrap();
    }

    #[test]
    fn rejects_zero_num_tokens() {
        let mut d = good(); d.num_tokens = 0;
        assert!(d.validate().is_err());
    }

    #[test]
    fn rejects_gqa_ratio_not_divisible() {
        let mut d = good(); d.num_kv_heads = 5;
        assert!(d.validate().is_err());
    }

    #[test]
    fn rejects_rotary_dim_exceeds_head_dim() {
        let mut d = good(); d.rotary_dim = 512;
        assert!(d.validate().is_err());
    }

    #[test]
    fn rejects_odd_rotary_dim() {
        let mut d = good(); d.rotary_dim = 255;
        assert!(d.validate().is_err());
    }

    #[test]
    fn rejects_non_power_of_two_block_size() {
        let mut d = good(); d.block_size = 12;
        assert!(d.validate().is_err());
    }

    #[test]
    fn rejects_zero_block_size() {
        let mut d = good(); d.block_size = 0;
        assert!(d.validate().is_err());
    }

    #[test]
    fn rejects_current_ctx_exceeding_bucket() {
        let mut d = good();
        // bucket_max = 1024 * 16 = 16384; one past = 16385
        d.current_max_context_len = Some(16385);
        assert!(d.validate().is_err());
    }

    #[test]
    fn accepts_current_ctx_at_bucket_boundary() {
        let mut d = good();
        d.current_max_context_len = Some(16384);
        d.validate().unwrap();
    }

    #[test]
    fn rejects_nonfinite_attn_scale() {
        let mut d = good(); d.attn_scale = f32::NAN;
        assert!(d.validate().is_err());
    }

    #[test]
    fn rejects_zero_attn_scale() {
        let mut d = good(); d.attn_scale = 0.0;
        assert!(d.validate().is_err());
    }

    #[test]
    fn rejects_f16_flag_disagreeing_with_dtype() {
        // Fp8 dtype but f16_kv=true → mismatch
        let mut d = good(); d.f16_kv = true;
        assert!(d.validate().is_err());
        // Conversely: F16 dtype but f16_kv=false
        let mut d = good();
        d.kv_dtype = KvDtype::F16;
        d.f16_kv = false;
        assert!(d.validate().is_err());
    }

    #[test]
    fn accepts_global_head_dim_512() {
        let mut d = good();
        d.head_dim = 512;
        d.rotary_dim = 512;
        d.layer_type = Gemma4LayerType::GlobalAttention;
        d.validate().unwrap();
    }
}
