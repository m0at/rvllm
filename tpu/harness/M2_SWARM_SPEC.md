# MiniMax-M2.7-NVFP4 on TPU v6e-8 — 16-agent swarm spec

This is the single source of truth. Every agent works only from this file.
No agent reads memory, prior chat, or other agents' output mid-run.

## Mission

Deploy `lukealonso/MiniMax-M2.7-NVFP4` (230B total / 10B active, NVFP4-quantized MoE)
on a single TPU v6e-8 slice, using the existing v6e-4 Gemma 4 harness as the template.

Approach: **NVFP4 weights stay packed in HBM (130 GB). On-the-fly dequant fuses into
the GEMM epilogue in JAX. Path B fallback (upcast to int8 at load via Zig CPU) is
optional but specified.**

Host-side heavy work (tokenization, optional dequant-at-load, metadata pack) uses
Zig + SIMD + thread pool sized to all TPU host cores.

## Target model — verified from config.json

```
architectures:         ["MiniMaxM2ForCausalLM"]
model_type:            minimax_m2
num_hidden_layers:     62
hidden_size:           3072
intermediate_size:     1536           # per-expert FFN inner dim
vocab_size:            200064
max_position_embeddings: 196608        # 192K
num_attention_heads:   48
num_key_value_heads:   8               # GQA, kv_heads=8
head_dim:              128
rotary_dim:            64
partial_rotary_factor: 0.5
rope_theta:            5_000_000
use_qk_norm:           true
qk_norm_type:          per_layer
num_local_experts:     256
num_experts_per_tok:   8
shared_intermediate_size: 0            # no shared expert
scoring_func:          sigmoid
use_routing_bias:      true            # aux-loss-free, per-expert bias
norm_topk_prob:        true            # renormalize top-8 weights to sum=1
router_jitter_noise:   0.0
use_mtp:               true
num_mtp_modules:       3
mtp_transformer_layers: 1
tie_word_embeddings:   false
sliding_window:        null            # all layers are full-attention
hidden_act:            "silu"          # SwiGLU
bos_token_id:          1
eos_token_id:          2

quantization_config:
  quant_algo:          "NVFP4"
  quant_method:        "modelopt"
  weights:
    num_bits: 4, type: float, group_size: 16
  input_activations:
    num_bits: 4, type: float, group_size: 16   # NVFP4 activations too
  ignore: ["lm_head", "model.layers.*.block_sparse_moe.gate",
           "model.layers.*.self_attn*"]
```

**On-disk shape:** experts packed NVFP4 (= 0.5 bytes/param) + FP8 E4M3 scales per 16
elements (= 1 byte / 16 weight values = 0.0625 bytes/param scale). Attention, router
gates, embeddings, lm_head, MTP all stay BF16. Total ~130 GB.

NVFP4 raw encoding (per HuggingFace `modelopt` producer):
- Weights stored as packed `uint8` (two FP4 values per byte), little-endian nibbles
  (low nibble = even index, high nibble = odd index).
- FP4 = E2M1, signed, 16 discrete values: `{+/-{0, 0.5, 1, 1.5, 2, 3, 4, 6}}`.
- Each 16-element group has one FP8 E4M3 scale stored separately (tensor name
  suffix `.weight_scale`). Effective value = `fp4_decoded * fp8_scale_decoded`.

## Target hardware

- Single TPU v6e-8 slice, one host (`us-east5-b`).
- 8 chips × 32 GB HBM = 256 GB HBM total.
- Host: ~180 vCPUs, ~1.4 TB RAM (assumption — verify with `nproc` at deploy time).
- Boot disk: **300 GB** (130 GB model + 30 GB OS/packages + 140 GB scratch).

## Sharding: expert-parallel 8-way

- Mesh axes: `('expert', 'tp')` with shape `(8, 1)` to start. Pure EP — 256 experts
  sharded 32 experts per chip, attention replicated (heads are small: 48 Q / 8 KV).
- If attention becomes a bottleneck, re-shape to `(8,)` along `'tp'` for attention
  and shard_map experts separately.
- All-to-all dispatch: token→expert after top-8 selection, all-gather combine after
  expert FFN. JAX `shard_map` + `jax.lax.all_to_all`.

## File ownership — disjoint per agent

Every agent writes to its listed files. Agents do NOT modify files owned by other
agents. Read-only references are common (the two anchors below).

### Read-only anchors (reference only)

- `tpu/harness/gemma4_tpu_infer.py` — template for JAX model, mesh, loader, forward.
  Copy layout, adapt for M2 specifics.
- `tpu/harness/api_server.py` — template for OpenAI server.
- `v3/crates/rvllm-core/src/config/model.rs` — model registry (agent 1 edits).
- `v3/crates/rvllm-loader/src/k2_cpu.rs` — reference for DeepSeek/MoE loader shape.
- `zig/src/weight_convert.zig` — reference for Zig SIMD conversion pattern.
- `zig/build.zig` — edited only by agent 5.

---

## Agents 1–16 — exact scopes

### Agent 1 — M2 arch registry (Rust)

**Owns:**
- `v3/crates/rvllm-core/src/config/model.rs` — add `MiniMaxM2` variant to `ModelArch`,
  extend `ModelConfig` with M2-specific fields (num_experts, experts_per_tok,
  rotary_dim, partial_rotary_factor, n_kv_heads already present, scoring_func,
  use_routing_bias, use_mtp, num_mtp_modules, moe_intermediate_size).
- `v3/crates/rvllm-core/src/config/minimax_m2.rs` — new file, the M2-specific parser
  that reads the config.json fields listed above and constructs the extended struct.

**API contract (exposed symbols):**
```rust
pub enum ModelArch { /* existing */ MiniMaxM2 }

pub struct MiniMaxM2Extras {
    pub num_local_experts: usize,        // 256
    pub num_experts_per_tok: usize,      // 8
    pub moe_intermediate_size: usize,    // 1536
    pub shared_intermediate_size: usize, // 0
    pub rotary_dim: usize,               // 64
    pub partial_rotary_factor: f32,      // 0.5
    pub use_qk_norm: bool,               // true
    pub use_routing_bias: bool,          // true
    pub scoring_func: String,            // "sigmoid"
    pub norm_topk_prob: bool,            // true
    pub use_mtp: bool,                   // true
    pub num_mtp_modules: usize,          // 3
    pub mtp_transformer_layers: usize,   // 1
    pub nvfp4: Option<NvFp4Config>,
}

pub struct NvFp4Config {
    pub weight_num_bits: u8,             // 4
    pub activation_num_bits: u8,         // 4
    pub group_size: usize,               // 16
    pub scale_dtype: String,             // "e4m3"
    pub ignore_patterns: Vec<String>,    // lm_head, router gates, self_attn*
}
```

**Don't touch:** any other file. No Cargo.toml edits.
**Validation:** `cargo check -p rvllm-core` must succeed; add one unit test that
parses the config.json fields verbatim from a fixture string (embed minimal json
inline in the test).

### Agent 2 — Zig NVFP4 → int8 SIMD dequant

**Owns:**
- `zig/src/nvfp4_to_int8.zig` — new file.

**API contract:**
```zig
pub const Nvfp4Block = extern struct {
    packed_vals: [8]u8,  // 16 FP4 values packed 2/byte
    scale_e4m3: u8,      // one FP8 E4M3 scale per group of 16
};

/// Dequant one tensor: n_elements FP4 weights + (n/16) FP8 scales -> n int8 values.
/// output_scale_per_channel filled with per-row int8 rescale (max(abs)/127).
/// Uses SIMD (AVX-512 VBMI on x86 if available, NEON/generic otherwise).
pub fn nvfp4ToInt8Row(
    packed: []const u8,           // n/2 bytes
    scales_e4m3: []const u8,      // n/16 bytes
    out_i8: []i8,                 // n bytes
    out_row_scale: *f32,
) void;

/// Batched over rows. Caller provides thread pool; internal function is
/// embarrassingly parallel over rows.
pub fn nvfp4ToInt8Matrix(
    packed: []const u8,
    scales_e4m3: []const u8,
    rows: usize,
    cols: usize,
    out_i8: []i8,                 // row-major, rows*cols
    out_row_scales: []f32,        // rows
) void;

test "fp4 decode table correctness" { ... }
test "roundtrip known blocks" { ... }
```

**FP4 E2M1 decode table (use exactly these values):**
```
idx  nibble  value       idx  nibble  value
0    0000    +0.0        8    1000    -0.0
1    0001    +0.5        9    1001    -0.5
2    0010    +1.0        10   1010    -1.0
3    0011    +1.5        11   1011    -1.5
4    0100    +2.0        12   1100    -2.0
5    0101    +3.0        13   1101    -3.0
6    0110    +4.0        14   1110    -4.0
7    0111    +6.0        15   1111    -6.0
```
**FP8 E4M3 decode:** standard IEEE-like, sign(1) + exp(4, bias=7) + mantissa(3),
special: all-1s = NaN, 0x00 = +0.0, 0x80 = -0.0. Use `std.math` or manual bitfield.

**Target throughput:** ≥ 20 GB/s single-threaded on M5 or Skylake-X, ≥ 500 GB/s with
180-thread pool on x86 host.

**Don't touch:** build.zig (agent 5), other zig files.
**Validation:** `zig build test` must pass; included unit tests must produce known
values for all 16 FP4 codes and at least 3 FP8 scale cases.

### Agent 3 — Zig NVFP4 → BF16 SIMD dequant

**Owns:**
- `zig/src/nvfp4_to_bf16.zig` — new file.

**API contract:**
```zig
pub fn nvfp4ToBf16Row(
    packed: []const u8,
    scales_e4m3: []const u8,
    out_bf16: []u16,              // raw bf16 bits, n elements
) void;

pub fn nvfp4ToBf16Matrix(
    packed: []const u8,
    scales_e4m3: []const u8,
    rows: usize,
    cols: usize,
    out_bf16: []u16,
) void;

test "bf16 roundtrip" { ... }
```

**Same FP4/FP8 decode tables as agent 2.** Intermediate compute in f32, then
f32→bf16 via top-16 bits truncation with round-to-nearest-even.

**Don't touch:** build.zig, other files.
**Validation:** unit tests; output must be bit-equivalent to a reference f32
implementation rounded to bf16.

### Agent 4 — Zig BPE tokenizer (200K tiktoken-style)

**Owns:**
- `zig/src/tiktoken_bpe.zig` — new file.

**API contract:**
```zig
pub const Bpe = struct {
    allocator: std.mem.Allocator,
    vocab: VocabMap,              // token bytes -> id (HAMT or robin-hood hash)
    merges: MergeMap,             // (left_id, right_id) -> merged_id
    pattern: Regex,               // tiktoken pre-tokenize pattern

    pub fn init(
        allocator: std.mem.Allocator,
        tokenizer_json_path: []const u8,
    ) !Bpe;

    pub fn deinit(self: *Bpe) void;

    /// Thread-safe: encode a single text buffer to token IDs.
    /// Caller owns returned slice.
    pub fn encode(self: *const Bpe, text: []const u8,
                  out: *std.ArrayList(u32)) !void;

    /// Thread-safe: decode token IDs to bytes.
    pub fn decode(self: *const Bpe, ids: []const u32,
                  out: *std.ArrayList(u8)) !void;
};

/// Batch-encode N prompts across a thread pool. Each prompt independent.
pub fn encodeBatchParallel(
    bpe: *const Bpe,
    prompts: []const []const u8,
    out_arrays: []std.ArrayList(u32),  // len == prompts.len
    n_threads: usize,
) !void;
```

**Tokenizer file format:** HuggingFace `tokenizer.json` (loads via
`std.json.parseFromSlice`). Look for `model.vocab`, `model.merges`,
`pre_tokenizer.pattern` (GPT-4 style regex — implement as compiled NFA or
byte-level regex subset).

**Don't touch:** build.zig, other files.
**Validation:** include test vectors (3 known string→id sequences) that match
HuggingFace `tokenizers` output exactly for at least "Hello, world!",
"the quick brown fox", and a UTF-8 multibyte string.

### Agent 5 — Zig build system + C FFI + binary driver

**Owns:**
- `zig/build.zig` — add new modules and the binary (edit existing file).
- `zig/src/rvllm_dequant_bin.zig` — new file, `main()` entry point for the
  end-to-end dequant tool.
- `zig/src/c_abi.zig` — new file, exported C-ABI functions for CPython ctypes
  binding (used by `nvfp4_loader.py` in agent 6).

**Binary behavior (`rvllm_dequant`):**
```
rvllm_dequant --in  <nvfp4_model_dir> \
              --out <dequantized_model_dir> \
              --dtype {int8|bf16} \
              --threads <N>
```
- Reads every `*.safetensors` shard in `--in` (with its `.weight_scale` tensor for
  NVFP4 tensors).
- For each NVFP4-quantized tensor, spawns a task in a fixed-size thread pool
  (`--threads`, default = `std.Thread.getCpuCount()`).
- For tensors in the ignore list (bf16 already), copies through unchanged.
- Writes output safetensors in row-major layout with new header that reflects new
  dtype.
- Progress to stderr every 1 GB processed.

**C ABI (for Python ctypes in agent 6):**
```zig
// exported: extern "C" fn rvllm_nvfp4_to_int8(...) callconv(.C) i32;
export fn rvllm_nvfp4_to_int8(
    packed_ptr: [*]const u8, packed_len: usize,
    scales_ptr: [*]const u8, scales_len: usize,
    rows: usize, cols: usize,
    out_i8_ptr: [*]i8,
    out_row_scales_ptr: [*]f32,
    n_threads: usize,
) i32;  // 0 = ok, non-zero = error code

export fn rvllm_nvfp4_to_bf16(...) i32;

export fn rvllm_bpe_encode(
    bpe_handle: *anyopaque, text_ptr: [*]const u8, text_len: usize,
    out_ids_ptr: [*]u32, out_ids_cap: usize, out_ids_len: *usize,
) i32;
```

**build.zig changes:**
- Add `nvfp4_to_int8.zig`, `nvfp4_to_bf16.zig`, `tiktoken_bpe.zig`, `c_abi.zig` to
  the library root module.
- Add executable target `rvllm_dequant` built from `rvllm_dequant_bin.zig` linking
  the library.
- Add shared library target `librvllm_zig.dylib`/`.so` with c_abi exports.
- Add test runners for all new modules.
- Ensure `-O3` + `-fno-omit-frame-pointer` for ReleaseFast, `-mcpu=native` default.

**Don't touch:** any file not listed above (do NOT edit agent 2/3/4's source files).
**Validation:** `zig build` and `zig build test` both pass; `zig build
rvllm_dequant` produces a runnable binary that at minimum prints a `--help`
message.

### Agent 6 — NVFP4 safetensors loader (Python)

**Owns:**
- `tpu/harness/nvfp4_loader.py` — new file.

**API contract:**
```python
class NvFp4Tensor:
    name: str
    shape: tuple[int, ...]         # logical (in dequantized units)
    packed: np.ndarray             # uint8, shape = (rows, cols/2)
    scales: np.ndarray             # uint8 (FP8 E4M3 bits), shape = (rows, cols/16)
    group_size: int                # 16

class ModeloptSafetensorsReader:
    def __init__(self, model_dir: str): ...
    def list_tensors(self) -> list[str]: ...
    def is_nvfp4(self, name: str) -> bool: ...
    def read_nvfp4(self, name: str) -> NvFp4Tensor: ...
    def read_bf16(self, name: str) -> np.ndarray: ...        # for lm_head, attn, router
    def iter_shards(self): ...

def dequant_nvfp4_to_int8_cpu(t: NvFp4Tensor, n_threads: int = 0):
    """Calls into librvllm_zig via ctypes for SIMD dequant.
    Returns (int8_matrix, row_scales_f32).
    """

def dequant_nvfp4_to_bf16_cpu(t: NvFp4Tensor, n_threads: int = 0):
    """Calls into librvllm_zig via ctypes.
    Returns bf16_matrix (np.ndarray with ml_dtypes.bfloat16).
    """
```

**Implementation notes:**
- Find the `librvllm_zig.so` / `.dylib` via env var `RVLLM_ZIG_LIB` with a sensible
  default `../zig/zig-out/lib/librvllm_zig.so`.
- For NVFP4 tensors: the safetensors header has the packed tensor at
  `<name>.weight_packed` (uint8) and scale at `<name>.weight_scale` (uint8 E4M3),
  per modelopt v0.39 layout. If the layout turns out to use a different suffix,
  detect by probing the header dtype and name patterns.
- `ignore` patterns from config determine which tensors stay bf16.

**Don't touch:** any JAX model code.
**Validation:** a `__main__` block that opens a model dir, prints shape + dtype of
the first NVFP4 tensor and the first bf16 tensor. Do not load the whole model in
the smoke test.

### Agent 7 — NVFP4 on-the-fly JAX dequant kernel (path A)

**Owns:**
- `tpu/harness/nvfp4_jax_ops.py` — new file.

**API contract:**
```python
def nvfp4_to_bf16_jax(packed: jax.Array, scales: jax.Array,
                     out_shape: tuple[int, ...]) -> jax.Array:
    """JAX-only op: packed uint8 + FP8 scales -> bf16 values.
    Fully traceable, jit-compatible. No custom call — pure XLA.
    `out_shape` is the dequantized shape (rows, cols)."""

def nvfp4_matmul(x_bf16: jax.Array,
                 w_packed: jax.Array,       # uint8 (out_features, in_features/2)
                 w_scales: jax.Array,       # uint8 (out_features, in_features/16)
                 out_features: int,
                 in_features: int) -> jax.Array:
    """Fused: dequant W on the fly, then x @ W^T.
    Uses `jax.lax.dot_general` after producing bf16 W via `nvfp4_to_bf16_jax`.
    Relies on XLA fusion — tests confirm no materialization of full W in HBM."""
```

**Implementation strategy:**
- Unpack uint8 to two FP4 nibbles: `hi = packed >> 4; lo = packed & 0x0F`.
- LUT-based FP4 decode: construct a `jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
  6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=jnp.bfloat16)` and
  gather via `jnp.take`.
- FP8 E4M3 decode: bit manipulate `uint8 -> f32 -> bf16`. Build scale_bf16 tensor
  of shape `(rows, cols/16)`, broadcast along cols direction by repeating 16×.
- Final value: `fp4_bf16 * scale_bf16`.
- For the fused matmul variant, use `jax.jit` + `jax.checkpoint` around a closure
  that reshapes packed to `(rows, cols)` bf16 on the fly. Profile with
  `jax.make_jaxpr` to confirm no intermediate materialization (the XLA rematerial
  pass should handle it).

**Validation:** `python3 nvfp4_jax_ops.py` runs a standalone test on CPU with
small random tensors, comparing against agent 6's CPU dequant to cosine ≥ 0.9999.

### Agent 8 — M2 JAX attention (GQA, QK-norm, partial RoPE)

**Owns:**
- `tpu/harness/m2_attention.py` — new file.

**API contract:**
```python
# Exported JAX ops. All callables are jit-friendly, no mutable global state.

def precompute_rope_m2(theta: float, rotary_dim: int, max_ctx: int):
    """theta=5_000_000, rotary_dim=64. Returns (cos, sin) of shape (max_ctx, rotary_dim//2)."""

def rope_partial_apply(q: jax.Array, cos: jax.Array, sin: jax.Array,
                      rotary_dim: int) -> jax.Array:
    """Apply RoPE to first `rotary_dim` dims of head, leave remaining head_dim-rotary_dim pass-through.
    q shape: (..., head_dim)"""

def qk_rmsnorm(x: jax.Array, g: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm per head: x shape (..., head_dim), g shape (head_dim,)."""

def gqa_attention_decode(q: jax.Array,          # (B, num_q_heads, head_dim)
                         k_cache: jax.Array,     # (B, max_ctx, num_kv_heads, head_dim)
                         v_cache: jax.Array,     # (B, max_ctx, num_kv_heads, head_dim)
                         pos: jax.Array,         # (B,) scalar current-pos
                         cache_len: jax.Array,   # (B,) valid kv len
                         num_q_heads: int = 48,
                         num_kv_heads: int = 8,
                         head_dim: int = 128) -> jax.Array:
    """Softmax attention over full (non-sliding) context. Returns (B, num_q_heads, head_dim).
    No attention mask beyond causal + cache_len."""

def attention_layer(x: jax.Array,                # (B, H) — H=3072
                    weights: dict,                # qw, kw, vw, ow (bf16), qn, kn (per-layer qk-norm)
                    ln_w: jax.Array,              # input layernorm weight
                    k_cache: jax.Array,
                    v_cache: jax.Array,
                    pos: jax.Array,
                    cos: jax.Array, sin: jax.Array,
                    cache_len: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single attention block decode step. Returns (x_out, k_cache_new, v_cache_new).
    x_out shape: (B, H)."""
```

**Constants (hard-code from verified config):**
```python
NUM_Q_HEADS = 48
NUM_KV_HEADS = 8
HEAD_DIM = 128
ROTARY_DIM = 64
PARTIAL_ROTARY_FACTOR = 0.5  # rotary_dim / head_dim == 0.5
ROPE_THETA = 5_000_000.0
H = 3072
RMS_EPS = 1e-6
```

**Don't touch:** anything outside this file.
**Validation:** a `__main__` block with small random weights that runs one attention
step on CPU with `jax.config.update('jax_platforms','cpu')` and checks output
shape/finiteness. No numerical-parity test against HF here — that lives in agent 16.

### Agent 9 — M2 JAX MoE layer (sigmoid + bias router, top-8, 256 experts)

**Owns:**
- `tpu/harness/m2_moe.py` — new file.

**API contract:**
```python
NUM_EXPERTS = 256
TOP_K = 8
MOE_INTER = 1536  # per-expert intermediate size
H = 3072

def router_sigmoid_topk(x: jax.Array,
                        router_w: jax.Array,      # (NUM_EXPERTS, H)
                        router_bias: jax.Array,   # (NUM_EXPERTS,) — aux-loss-free bias
                        ) -> tuple[jax.Array, jax.Array]:
    """Returns (topk_weights, topk_indices). topk_weights shape (B, TOP_K) sums to 1
    (per norm_topk_prob=True). Uses sigmoid(logits + bias), selects top-K, renormalizes.

    Rule (from MiniMax reference):
      scores = sigmoid(x @ router_w.T)           # (B, E)
      biased = scores + router_bias              # used ONLY for selection
      topk_idx = argmax_top8(biased)
      topk_sel = scores[topk_idx]                # actual (unbiased) weights
      topk_w = topk_sel / topk_sel.sum(axis=-1, keepdims=True)
    """

def expert_ffn(x: jax.Array,                     # (B, H) pre-normed input
               gate_up_w: jax.Array,             # (2*MOE_INTER, H) fused gate||up
               down_w: jax.Array,                # (H, MOE_INTER)
               ) -> jax.Array:
    """Single expert SwiGLU FFN. Returns (B, H)."""

def moe_block(x_normed: jax.Array,               # (B, H)
              router_w: jax.Array, router_bias: jax.Array,
              expert_gate_up: jax.Array,         # (NUM_EXPERTS, 2*MOE_INTER, H)
              expert_down: jax.Array,            # (NUM_EXPERTS, H, MOE_INTER)
              mesh) -> jax.Array:
    """Top-8 routing + expert dispatch + weighted combine.
    If `mesh` has 'expert' axis: use shard_map with all-to-all dispatch.
    Else: gather-expert variant (reference / unit tests).
    Returns (B, H)."""
```

**Shard_map variant:** experts are sharded across mesh `'expert'` axis. For each
token, we know the 8 expert indices. Gather into a dense (B*TOP_K, H) send buffer,
all-to-all, local expert compute, all-to-all back, scatter+combine.

**Don't touch:** anything outside this file.
**Validation:** `__main__` with 4 experts (cut-down) on CPU, confirms that gather
and shard_map variants produce cosine ≥ 0.9999.

### Agent 10 — M2 JAX MTP heads (speculative decode)

**Owns:**
- `tpu/harness/m2_mtp.py` — new file.

**API contract:**
```python
NUM_MTP_MODULES = 3
MTP_TRANSFORMER_LAYERS = 1

def mtp_forward(h_last: jax.Array,                # (B, H) final hidden state pre-lm_head
                mtp_weights: list[dict],           # 3 modules, each a dict of transformer + lm head
                embed: jax.Array,                  # token embedding, (V, H) for input-token embed
                prev_token_id: jax.Array,          # (B,) the base-model's predicted token
                ) -> jax.Array:
    """Returns (B, NUM_MTP_MODULES, V) draft logits for speculative decode.
    Each module predicts the i-th next token given h_last and the chain of previous drafts.
    Implementation follows MiniMax reference (similar to DeepSeek-V3 MTP):
      for i in 0..NUM_MTP_MODULES:
        h = norm(concat([h_{i}, embed(prev_i)]))   # (B, 2H)
        h = eh_proj(h)                              # (B, H)
        h = transformer_layer(h)                   # 1 layer
        logits_i = h @ lm_head.T                   # shared with main lm_head
        prev_{i+1} = argmax(logits_i)
    """

def mtp_load_weights(all_tensors: dict, prefix: str) -> list[dict]:
    """Load MTP weights from safetensors dict. Prefix like 'model.mtp_modules.'."""
```

**Don't touch:** anything outside this file.
**Validation:** smoke `__main__` that constructs random MTP weights and runs a
single step on CPU.

### Agent 11 — M2 main inference script (integration)

**Owns:**
- `tpu/harness/m2_tpu_infer.py` — new file.

**This is the integrator.** It imports from agents 8, 9, 10, 12, 13, 14 and
produces a working `main()` that matches the `gemma4_tpu_infer.py` CLI.

**CLI (must match):**
```
python3 m2_tpu_infer.py --model-dir PATH [--max-tokens N] [--max-ctx N] \
                       [--prompt S] [--batch N] [--fused] [--perplexity]
```

**Entry points imported:**
- `m2_attention.attention_layer`, `precompute_rope_m2`
- `m2_moe.moe_block`, `router_sigmoid_topk`
- `m2_mtp.mtp_forward` (optional — behind `--speculate` flag, default off for v0)
- `m2_mesh.make_mesh_v6e8`
- `m2_kv_cache.make_kv_caches`, `M2_KV_LAYOUT`
- `m2_chat.apply_chat_template`, `load_tokenizer_m2`
- `nvfp4_loader.ModeloptSafetensorsReader`
- `nvfp4_jax_ops.nvfp4_matmul` (for path A) OR agent 6's int8 path (for path B)

**Structure:**
```python
def load_config_m2(model_dir): ...   # mirrors gemma4_tpu_infer.load_config, sets globals
def load_model_m2(model_dir, mesh, max_ctx, path: str = "A"): ...
def forward_step_m2(...): ...
def run_generate_m2(args, ...): ...
def run_perplexity_m2(args, ...): ...
def main(): ...
```

**Globals at module scope (set by `load_config_m2`):**
```python
H, NH, NKV, HEAD_DIM, NL, VOCAB, ROTARY_DIM, ROPE_THETA
NUM_EXPERTS, TOP_K, MOE_INTER
USE_MTP, NUM_MTP
PATH  # "A" or "B"
```

**Don't touch:** any files owned by other agents. This one has lots of imports —
if import fails because another agent is late, stub with `try/except ImportError`
and raise a clear error naming the missing agent + file.
**Validation:** `python3 -c "from tpu.harness.m2_tpu_infer import main"` must import
cleanly. No model load required for smoke — validation is import-cleanliness only.

### Agent 12 — v6e-8 expert-parallel mesh + shard_map topology

**Owns:**
- `tpu/harness/m2_mesh.py` — new file.

**API contract:**
```python
def make_mesh_v6e8():
    """Returns (mesh, axes_dict). Mesh shape = (8,) axis='expert'.
    Future: (2, 4) with 'tp' × 'expert' if attention sharding becomes needed."""

def expert_shard_spec(num_experts: int = 256) -> P:
    """PartitionSpec for expert weight tensor. Experts sharded along axis 0."""
    return P('expert', None, None)

def replicate_spec() -> P:
    return P(None, None)  # or P() for scalar

def expert_all_to_all_dispatch(
    x_per_token: jax.Array,          # (B, TOP_K, H) — each token's top-K copies
    expert_idx: jax.Array,           # (B, TOP_K) target expert id
    num_experts: int,
    num_shards: int,
) -> tuple[jax.Array, jax.Array]:
    """Sort tokens by expert, pack into (num_experts_per_shard, capacity, H),
    all-to-all exchange across 'expert' axis. Returns (local_tokens, local_idx_map)."""

def expert_all_to_all_combine(
    local_out: jax.Array,            # output of local expert FFN
    idx_map: jax.Array,              # reverse mapping
    weights: jax.Array,              # (B, TOP_K) routing weights
    out_shape: tuple,
) -> jax.Array:
    """All-to-all back, weighted sum, scatter-add to original token positions."""
```

**Notes:**
- Start with capacity = `ceil(B * TOP_K / num_experts) * 2` (overflow factor 2).
  If capacity overflow, drop tokens (log a warning) for v0; fix in v1.
- Use `jax.lax.all_to_all(..., split_axis=0, concat_axis=0, axis_name='expert')`.

**Don't touch:** any other file.
**Validation:** `__main__` fake 4-chip mesh on CPU, dispatches 32 tokens across
16 experts with top-2, checks cosine ≥ 0.9999 vs. naive gather reference.

### Agent 13 — M2 KV cache layout (paged, GQA, 196K-capable)

**Owns:**
- `tpu/harness/m2_kv_cache.py` — new file.

**API contract:**
```python
M2_KV_LAYOUT = {
    'num_kv_heads': 8,
    'head_dim': 128,
    'num_layers': 62,
    'dtype': 'bf16',
    'block_size': 256,               # tokens per page
}

def kv_bytes_per_token() -> int:
    # 8 * 128 * 2 (K,V) * 2 bytes * 62 layers = 260_096 bytes
    ...

def make_kv_caches(B: int, max_ctx: int, mesh):
    """Returns dict with 'k' and 'v' arrays of shape (62, B, max_ctx, 8, 128),
    bf16, sharded replicated across 'expert' axis."""

def allocate_paged_kv(max_pages: int, block_size: int, mesh):
    """For long-context (196K) operation. Returns (paged_k, paged_v, block_table).
    Block table shape: (B, max_pages)."""

def compute_max_ctx_for_hbm(available_gb: int, B: int) -> int:
    """Helper: given leftover HBM after weights, return max ctx len at given B."""
```

**Don't touch:** any other file.
**Validation:** `__main__` prints the KV bytes for B=1 ctx=196K (should be ~50 GB)
and for B=8 ctx=32K.

### Agent 14 — Chat template + tokenizer config + PY tokenizer wrapper

**Owns:**
- `tpu/harness/m2_chat.py` — new file.

**API contract:**
```python
def load_tokenizer_m2(model_dir: str):
    """Loads the tokenizer from tokenizer.json in model_dir.
    Prefers the Zig BPE via ctypes if RVLLM_ZIG_LIB is set; falls back to
    `tokenizers` library otherwise. Returns an object with:
      .encode(str) -> list[int]
      .decode(list[int]) -> str
      .bos_token_id, .eos_token_id
    """

def apply_chat_template(messages: list[dict]) -> str:
    """MiniMax chat template.
    Per the official MiniMax-M2 format, tokens used are:
      <|im_start|>system\n...<|im_end|>
      <|im_start|>user\n...<|im_end|>
      <|im_start|>assistant\n
    If the official template from tokenizer_config.json differs, use that
    template via jinja2 rendering.
    """
```

**Don't touch:** anything else.
**Validation:** `__main__` block encodes "Hello, world!" and prints the token IDs;
also prints the chat template output for a 2-turn conversation.

### Agent 15 — Deploy script + runbook for v6e-8

**Owns:**
- `tpu/harness/deploy_m2_tpu.sh` — new file, executable bash.
- `tpu/harness/M2_DEPLOY.md` — new file, runbook.

**deploy_m2_tpu.sh behavior:**
1. Check prerequisites: `gcloud`, `huggingface-cli` installed; `PROJECT` and
   `ZONE` env vars set (default zone=us-east5-b).
2. Create TPU VM: `gcloud compute tpus tpu-vm create rvllm-m2 --zone=$ZONE
   --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e --boot-disk-size=300`.
3. Wait for READY.
4. SSH install script (heredoc):
   ```
   pip3 install 'jax[tpu]' safetensors huggingface_hub tokenizers ml_dtypes
   # Zig toolchain
   curl -L https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz \
     | tar xJ -C $HOME/
   export PATH=$HOME/zig-linux-x86_64-0.13.0:$PATH
   ```
5. Transfer the repo via tarball (SHA-pinned, per user's global rule): local
   `git archive --format=tar.gz --prefix=rvllm-$SHA/ HEAD | ssh tpu-vm "tar xz
   -C /workspace/runs/$SHA"`.
6. On remote: `cd /workspace/runs/$SHA/zig && zig build -Doptimize=ReleaseFast`.
7. Model download: `huggingface-cli download lukealonso/MiniMax-M2.7-NVFP4
   --local-dir /workspace/models/m2-nvfp4 --max-workers 32`.
8. (Path A only) Smoke-test: `python3 tpu/harness/m2_tpu_infer.py --model-dir
   /workspace/models/m2-nvfp4 --max-tokens 16 --prompt "Hello"`.

**M2_DEPLOY.md contents:**
- Prerequisites (gcloud auth, HF token, project, quota).
- Cost estimate table (on-demand + spot for v6e-8).
- Step-by-step commands.
- Common failures and fixes.
- Verification checklist.

**Don't touch:** anything else.
**Validation:** `bash -n tpu/harness/deploy_m2_tpu.sh` syntax-checks cleanly.

### Agent 16 — Benchmark + PPL harness for M2

**Owns:**
- `tpu/harness/m2_bench.py` — new file.

**API contract:**
```
python3 m2_bench.py --model-dir PATH \
                    [--batch 1|8|32|64|128] \
                    [--ctx 2048|32768|196608] \
                    [--iters 30] \
                    [--warmup 5] \
                    [--ppl] [--ppl-file wikitext.txt]
```

**Behavior:**
- Imports `m2_tpu_infer` symbols: `load_config_m2`, `load_model_m2`, `forward_step_m2`.
- Warmup: N steps discarded.
- Measure: M steps, print min/mean/max ms/step and tok/s.
- Perplexity mode: chunked over `--ppl-file` with 2048-token windows, matches the
  existing `bench_ppl.py` methodology.
- Writes results JSON to `tpu/out/m2_bench_<batch>_<ctx>.json`.

**Don't touch:** anything else.
**Validation:** `python3 -c "from tpu.harness.m2_bench import main"` imports cleanly.

---

## Cross-agent API contracts (append-only)

Any agent that needs a symbol from another agent uses the exact signature above.
If an agent discovers a signature must change, they MUST:
1. NOT change it.
2. Add a note in their agent's output saying "requested change to agent X's
   signature: <old> -> <new>, reason: <why>".

## Integration phase (coordinator runs after all 16 finish)

1. `cargo check --manifest-path v3/Cargo.toml -p rvllm-core` — agent 1 validation.
2. `cd zig && zig build && zig build test` — agents 2–5 validation.
3. `python3 -c "from tpu.harness.m2_tpu_infer import main"` — agent 11 integration.
4. Dry-run smoke: `python3 tpu/harness/m2_tpu_infer.py --help`.
5. Syntax-check deploy: `bash -n tpu/harness/deploy_m2_tpu.sh`.

## What NOT to do (for all agents)

- Do not add emoji to any code or doc.
- Do not write Co-Authored-By trailers.
- Do not add docstrings to unchanged code.
- Do not introduce fallback-on-failure paths. Panic / raise with a clear message.
- Do not use Python type annotations beyond what's needed for clarity.
- Do not fetch from the internet at runtime (model download is explicit in deploy
  script only).
- Do not modify files outside your listed owned set.
- No Cargo dependency changes except agent 1 may touch `rvllm-core/Cargo.toml` ONLY
  if absolutely required for serde types.
