//! Compressed-tensors AWQ checkpoint metadata.
//!
//! Cycle 41 step 3b: parse `quantization_config` from the model's
//! `config.json` so the loader knows which linears are AWQ-quantized
//! and what their group structure is.
//!
//! Format reference: ebircak/gemma-4-31B-it-4bit-W4A16-AWQ ships
//! `config.json` with:
//!
//! ```json
//! "quantization_config": {
//!   "config_groups": {
//!     "group_0": {
//!       "format": "pack-quantized",
//!       "targets": ["Linear"],
//!       "weights": {
//!         "num_bits": 4,
//!         "group_size": 128,
//!         "strategy": "group",
//!         "symmetric": false,
//!         "type": "int",
//!         "zp_dtype": "torch.int8"
//!       }
//!     }
//!   },
//!   "format": "pack-quantized",
//!   "ignore": ["lm_head", "re:model.vision_tower.*"],
//!   "quantization_status": "compressed"
//! }
//! ```
//!
//! Per-tensor layout (verified on the Gemma 4 31B AWQ checkpoint):
//!
//! ```text
//! <linear>.weight_packed     : I32 [N, K/8]   (8 INT4 per int32 along K)
//! <linear>.weight_scale      : BF16 [N, K/g]
//! <linear>.weight_zero_point : I32 [N/8, K/g] (8 INT4 per int32 along N)
//! <linear>.weight_shape      : I64 [2]         (original [N, K] dense shape)
//! ```

// No `serde` dep on this crate — parse the JSON via serde_json::Value
// directly to avoid pulling a new dep just for one struct.

/// AWQ / compressed-tensors weight quantization scheme. Captured from
/// `quantization_config.config_groups[*].weights` in `config.json`.
#[derive(Clone, Debug)]
pub struct AwqWeightScheme {
    /// Bits per weight element. Always 4 for the AWQ W4A16 path.
    pub num_bits: u32,
    /// Block-scale group size along K. Typical AWQ value: 128.
    pub group_size: u32,
    /// `true` = symmetric quant (zero_point fixed at 2^(num_bits-1) = 8 for 4-bit);
    /// `false` = asymmetric (per-group, per-row INT4 zero stored separately).
    pub symmetric: bool,
}

/// AWQ pack format names we accept. compressed-tensors uses
/// `"pack-quantized"`. Other strings are rejected so the loader fails
/// loudly on unknown variants instead of silently mis-loading.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AwqPackFormat {
    /// 4-bit nibbles packed 8-per-int32 along K, zero-points 8-per-int32
    /// along N (the format ebircak / cyankiwi / QuantTrio Gemma 4
    /// checkpoints use).
    PackQuantized,
}

/// Top-level AWQ config plumbed in from `config.json`.
#[derive(Clone, Debug)]
pub struct AwqConfig {
    /// Format of weight packing.
    pub format: AwqPackFormat,
    /// Quant scheme — single shared scheme across all targeted linears
    /// (compressed-tensors supports multiple `config_groups` but every
    /// real AWQ Gemma 4 checkpoint uses one).
    pub scheme: AwqWeightScheme,
    /// Linear-name patterns to skip (kept dense / unquantized).
    /// Compressed-tensors `ignore` lists raw names AND `re:<regex>`
    /// patterns. We store the raw strings; runtime matching uses
    /// [`Self::is_ignored`].
    pub ignore: Vec<String>,
}

impl AwqConfig {
    /// Parse `quantization_config` JSON value. Returns `Ok(None)` when
    /// the model is unquantized (no `quantization_config` field), an
    /// `Err` when the field is present but malformed.
    pub fn from_json(qc: Option<&serde_json::Value>) -> Result<Option<Self>, String> {
        let Some(qc) = qc else { return Ok(None) };

        let format = match qc.get("format").and_then(|v| v.as_str()) {
            Some("pack-quantized") => AwqPackFormat::PackQuantized,
            Some(other) => {
                return Err(format!(
                    "quantization_config.format = {other:?} unsupported; \
                     only \"pack-quantized\" recognized"
                ));
            }
            None => return Err("quantization_config.format missing".into()),
        };

        let groups = qc
            .get("config_groups")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "quantization_config.config_groups missing".to_string())?;

        // Find the (one) group that targets Linear.
        let mut linear_groups = groups.values().filter(|g| {
            g.get("targets")
                .and_then(|t| t.as_array())
                .map(|arr| arr.iter().any(|s| s.as_str() == Some("Linear")))
                .unwrap_or(false)
        });
        let group = linear_groups
            .next()
            .ok_or_else(|| "no config_group targets Linear".to_string())?;
        if linear_groups.next().is_some() {
            return Err(
                "multiple config_groups target Linear — heterogeneous quant \
                 schemes not yet supported"
                    .into(),
            );
        }

        let weights = group
            .get("weights")
            .ok_or_else(|| "config_group.weights missing".to_string())?;

        let r#type = weights.get("type").and_then(|v| v.as_str()).unwrap_or("int");
        if r#type != "int" {
            return Err(format!(
                "weights.type = {type:?} not supported; expected \"int\"",
                type = r#type
            ));
        }
        let strategy = weights.get("strategy").and_then(|v| v.as_str()).unwrap_or("group");
        if strategy != "group" {
            return Err(format!(
                "weights.strategy = {strategy:?} not supported; expected \"group\""
            ));
        }
        let num_bits = weights
            .get("num_bits")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "weights.num_bits missing or not a number".to_string())?
            as u32;
        let group_size = weights
            .get("group_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "weights.group_size missing or not a number".to_string())?
            as u32;
        let symmetric = weights
            .get("symmetric")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let scheme = AwqWeightScheme {
            num_bits,
            group_size,
            symmetric,
        };

        let ignore = qc
            .get("ignore")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Some(Self {
            format,
            scheme,
            ignore,
        }))
    }

    /// True when `linear_name` matches one of the `ignore` entries
    /// (literal name OR `re:<regex>`). For now the regex side does a
    /// simple substring fallback if the entry starts with `re:` —
    /// proper regex would require pulling in `regex` as a loader dep,
    /// which we defer until a real ignore pattern bites us.
    pub fn is_ignored(&self, linear_name: &str) -> bool {
        for entry in &self.ignore {
            if let Some(pat) = entry.strip_prefix("re:") {
                // Crude regex fallback (avoids pulling `regex` as a
                // loader dep). Strip a trailing `.*` (the dominant
                // pattern in compressed-tensors `ignore` lists) and
                // substring-match the prefix. If anchored or more
                // complex metacharacters appear, conservatively skip.
                let pat = pat.strip_suffix(".*").unwrap_or(pat);
                if !pat.contains(['^', '$', '|', '+', '?', '(', ')', '[', ']'])
                    && !pat.contains("\\")
                    && linear_name.contains(pat)
                {
                    return true;
                }
            } else if linear_name == entry {
                return true;
            }
        }
        false
    }
}

/// Names of the four safetensors entries that compose one AWQ-quantized
/// linear under the `pack-quantized` format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AwqTensorNames {
    /// `<linear>.weight_packed`  — I32 [N, K/8]
    pub packed: String,
    /// `<linear>.weight_scale`   — BF16 [N, K/group_size]
    pub scale: String,
    /// `<linear>.weight_zero_point` — I32 [N/8, K/group_size]
    pub zero_point: String,
    /// `<linear>.weight_shape`   — I64 [2] metadata
    pub shape: String,
}

impl AwqTensorNames {
    /// Compose the four expected entry names from a `<linear>` prefix
    /// (e.g. `model.language_model.layers.0.self_attn.q_proj`).
    pub fn for_linear(linear: &str) -> Self {
        Self {
            packed:     format!("{linear}.weight_packed"),
            scale:      format!("{linear}.weight_scale"),
            zero_point: format!("{linear}.weight_zero_point"),
            shape:      format!("{linear}.weight_shape"),
        }
    }
}

/// Per-linear AWQ shape header. Captures what the safetensors entries
/// MUST be for the `pack-quantized` format given a dense `[N, K]` linear
/// shape and the global group_size from [`AwqConfig`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AwqExpectedShapes {
    /// `[N, K/8]`
    pub packed_shape: [usize; 2],
    /// `[N, K/group_size]`
    pub scale_shape: [usize; 2],
    /// `[N/8, K/group_size]`
    pub zero_point_shape: [usize; 2],
    /// Always `[2]` (stores `[N, K]` as i64 metadata).
    pub shape_shape: [usize; 1],
}

impl AwqExpectedShapes {
    /// Compute expected entry shapes from the dense linear shape `[N, K]`
    /// and the AWQ group size. Returns Err if N or K are not divisible
    /// by the required factors (N % 8, K % 8, K % group_size, K %
    /// (2*group_size) for the per-group quantities).
    pub fn from_dense(n: usize, k: usize, group_size: u32) -> Result<Self, String> {
        let g = group_size as usize;
        if g == 0 {
            return Err("group_size must be > 0".into());
        }
        if k % 8 != 0 {
            return Err(format!("K={k} must be divisible by 8 (8 INT4 per int32 along K)"));
        }
        if k % g != 0 {
            return Err(format!("K={k} must be divisible by group_size={g}"));
        }
        if n % 8 != 0 {
            return Err(format!("N={n} must be divisible by 8 (8 INT4 per int32 along N for zero_point)"));
        }
        Ok(Self {
            packed_shape:     [n,     k / 8],
            scale_shape:      [n,     k / g],
            zero_point_shape: [n / 8, k / g],
            shape_shape:      [2],
        })
    }
}

/// One linear's AWQ tensor metadata, validated against expected shapes
/// and dtypes. Created via [`validate_awq_linear`].
#[derive(Clone, Debug)]
pub struct AwqLinearLayout {
    pub names:    AwqTensorNames,
    pub expected: AwqExpectedShapes,
    /// Dense `[N, K]` of the linear; reproduced here for callers that
    /// do not already track it.
    pub dense:    [usize; 2],
}

/// CPU-side bytes for one AWQ-quantized linear, ready to ship to GPU.
/// The four byte buffers correspond to the four safetensors entries.
///
/// Cycle 43 step 3d: kept CPU-only on purpose — GPU buffer allocation +
/// memcpy is a thin wrapper in cycle 44 that takes one of these and a
/// device pool. Splitting like this makes the staging path unit-testable
/// without a CUDA dep.
#[derive(Clone, Debug)]
pub struct AwqLinearStaged {
    pub layout: AwqLinearLayout,
    /// Raw bytes of `weight_packed` (I32 LE, [N, K/8]).
    pub packed: Vec<u8>,
    /// Raw bytes of `weight_scale` (BF16 LE, [N, K/g]).
    pub scale: Vec<u8>,
    /// Raw bytes of `weight_zero_point` (I32 LE, [N/8, K/g]).
    pub zero_point: Vec<u8>,
    /// Raw bytes of `weight_shape` (I64 LE, [2]).
    pub shape: Vec<u8>,
}

impl AwqLinearStaged {
    /// Total bytes pulled into host memory for this one linear.
    pub fn host_bytes(&self) -> usize {
        self.packed.len() + self.scale.len() + self.zero_point.len() + self.shape.len()
    }
}

/// Pull the four AWQ tensor byte slices for `linear_name` from a
/// generic `byte_lookup` that resolves an entry name to its raw bytes
/// (e.g. an mmap'd safetensors slice). Validates dtype + shape, then
/// stages the bytes into `AwqLinearStaged`.
///
/// The lookup signature matches what an `mmap` view of the safetensors
/// header naturally supports: given an entry name, return its dtype,
/// dense shape, and byte slice. Caller is responsible for keeping the
/// slice alive long enough; we copy into owned `Vec<u8>` so the staged
/// struct outlives the mmap.
pub fn stage_awq_linear(
    linear_name: &str,
    dense_shape: [usize; 2],
    scheme: &AwqWeightScheme,
    byte_lookup: &dyn Fn(&str) -> Option<(rvllm_core::DType, Vec<usize>, Vec<u8>)>,
) -> Result<AwqLinearStaged, String> {
    // First validate the layout — uses the dtype/shape pair, ignores bytes.
    let dtype_shape_lookup = |k: &str| -> Option<(rvllm_core::DType, Vec<usize>)> {
        byte_lookup(k).map(|(d, s, _)| (d, s))
    };
    let layout = validate_awq_linear(linear_name, dense_shape, scheme, &dtype_shape_lookup)?;

    // Then pull bytes for each, re-asserting dtype against the expected
    // value (closes a TOCTOU gap if `byte_lookup` is non-pure — e.g. a
    // racy mmap source).
    let need_bytes = |key: &str, want: rvllm_core::DType| -> Result<Vec<u8>, String> {
        let (dtype, _, bytes) = byte_lookup(key).ok_or_else(|| {
            format!("AWQ entry vanished between validate and stage: {key}")
        })?;
        if dtype != want {
            return Err(format!(
                "AWQ entry {key} dtype changed mid-stage: {dtype:?} != {want:?}"
            ));
        }
        Ok(bytes)
    };

    let packed     = need_bytes(&layout.names.packed,     rvllm_core::DType::I32)?;
    let scale      = need_bytes(&layout.names.scale,      rvllm_core::DType::Bf16)?;
    let zero_point = need_bytes(&layout.names.zero_point, rvllm_core::DType::I32)?;
    let shape      = need_bytes(&layout.names.shape,      rvllm_core::DType::I64)?;

    // Sanity-check byte sizes against expected shapes to catch
    // truncated/over-long mmap slices.
    let n = dense_shape[0];
    let k = dense_shape[1];
    let g = scheme.group_size as usize;
    let expected_packed = n * (k / 8) * 4;       // I32
    let expected_scale  = n * (k / g) * 2;       // BF16
    let expected_zero   = (n / 8) * (k / g) * 4; // I32
    let expected_shape  = 2 * 8;                  // I64 [2]
    if packed.len() != expected_packed {
        return Err(format!(
            "weight_packed bytes = {} != expected {}", packed.len(), expected_packed
        ));
    }
    if scale.len() != expected_scale {
        return Err(format!(
            "weight_scale bytes = {} != expected {}", scale.len(), expected_scale
        ));
    }
    if zero_point.len() != expected_zero {
        return Err(format!(
            "weight_zero_point bytes = {} != expected {}", zero_point.len(), expected_zero
        ));
    }
    if shape.len() != expected_shape {
        return Err(format!(
            "weight_shape bytes = {} != expected {}", shape.len(), expected_shape
        ));
    }
    // Cross-check the I64 [2] payload against the dense shape we were
    // told. Catches a checkpoint that drifted from its config.json
    // (or a misclassified linear) before any GPU memcpy happens.
    let n_meta = i64::from_le_bytes(shape[0..8].try_into().unwrap());
    let k_meta = i64::from_le_bytes(shape[8..16].try_into().unwrap());
    if n_meta as usize != n || k_meta as usize != k {
        return Err(format!(
            "weight_shape contents = [{n_meta}, {k_meta}] != dense [{n}, {k}]"
        ));
    }

    Ok(AwqLinearStaged { layout, packed, scale, zero_point, shape })
}

/// Cycle 43 step 3d (GPU side): one AWQ-quantized linear's three resident
/// device tensors, mirroring how `Fp8Weight` stores arena-relative offsets.
///
/// `weight_shape` from the source checkpoint is **not** uploaded — it's
/// I64 [2] metadata redundant with `dense`, validated host-side at stage
/// time, and the GEMV kernel does not consume it.
#[derive(Debug, Clone)]
pub struct AwqLinearWeight {
    /// Arena-relative byte offset of `weight_packed` (I32 [N, K/8]).
    pub packed_offset_bytes: u64,
    /// Arena-relative byte offset of `weight_scale` (BF16 [N, K/g]).
    pub scale_offset_bytes: u64,
    /// Arena-relative byte offset of `weight_zero_point` (I32 [N/8, K/g]).
    pub zero_point_offset_bytes: u64,
    /// Original dense `[N, K]` shape of the linear.
    pub dense: [usize; 2],
    /// AWQ block-scale group size along K. `K % group_size == 0`.
    pub group_size: u32,
}

/// Upload one staged AWQ linear into the HBM arena. Mirrors the
/// `upload_fp8_from` pattern in `load.rs`: three sync `cuMemcpyHtoD`s,
/// arena-relative offsets handed back so layer-major callers can store
/// them next to existing FP8 offsets without leaking raw device pointers.
///
/// Region tags use static names because [`rvllm_mem::HbmArena::region`]
/// requires `&'static str` — the layer index is not encoded; tags are
/// debug aids only. Per-region size is the staged byte buffer's length,
/// 16-byte aligned (matches the FP8 path).
///
/// Note on the `*_offset_bytes` naming: matches `F16Weight` /
/// `Fp8Weight` convention in [`crate::weights`]. `load.rs::arena_base()`
/// currently returns 0, so the field stores the absolute device
/// pointer; if a future change moves the arena to a non-zero base the
/// FP8 path and this path will both pick it up uniformly.
pub fn upload_awq_linear<'a>(
    arena: &'a rvllm_mem::HbmArena<'a>,
    staged: &AwqLinearStaged,
) -> rvllm_core::Result<AwqLinearWeight> {
    let pr = arena.region("awq_packed", staged.packed.len(), 16)?;
    unsafe { pr.copy_from_host(&staged.packed)? };
    let packed_offset_bytes = pr.device_ptr();

    let sr = arena.region("awq_scale", staged.scale.len(), 16)?;
    unsafe { sr.copy_from_host(&staged.scale)? };
    let scale_offset_bytes = sr.device_ptr();

    let zr = arena.region("awq_zero_point", staged.zero_point.len(), 16)?;
    unsafe { zr.copy_from_host(&staged.zero_point)? };
    let zero_point_offset_bytes = zr.device_ptr();

    let group_size =
        (staged.layout.dense[1] / staged.layout.expected.scale_shape[1]) as u32;

    Ok(AwqLinearWeight {
        packed_offset_bytes,
        scale_offset_bytes,
        zero_point_offset_bytes,
        dense: staged.layout.dense,
        group_size,
    })
}

/// Geometry of one Gemma 4 transformer layer needed to drive the AWQ
/// load path: dense `[N, K]` for each of the seven linears that
/// compressed-tensors AWQ stores un-fused. K is `hidden_size` for QKV
/// and gate/up, the head-multiple for O, and `intermediate_size` for
/// down_proj.
///
/// Caller derives this from `gemma4_arch::ModelArch` + the per-layer
/// attention type (sliding heads at half-width, full heads at full).
/// We keep it as plain data here so the loader doesn't reach into the
/// arch crate for layer geometry — that mirrors how `validate_awq_linear`
/// already takes `dense_shape` rather than an arch handle.
#[derive(Clone, Debug)]
pub struct AwqLayerShapes {
    /// Output dim of `q_proj` weight (= num_heads * head_dim). Hidden
    /// is shared via `hidden`.
    pub q_out:            usize,
    /// Output dim of `k_proj` and `v_proj` weight (= num_kv_heads * head_dim).
    pub kv_out:           usize,
    /// Output dim of `o_proj` weight (= hidden_size).
    pub o_out:            usize,
    /// Input dim of `o_proj` weight (= num_heads * head_dim).
    pub o_in:             usize,
    /// Output dim of `gate_proj` and `up_proj` weight (= intermediate_size).
    pub mlp_intermediate: usize,
    /// Hidden size — the K of QKV / gate-up, also the N of down_proj.
    pub hidden:           usize,
}

/// Stage + upload all 7 AWQ linears of one Gemma 4 transformer layer.
///
/// `byte_lookup` is the mmap-resolving callback that
/// [`stage_awq_linear`] documents — same shape, same caller contract.
/// `prefix` is the layer's HF weight prefix, e.g.
/// `model.language_model.layers.0` (without trailing dot).
///
/// Cycle 44 step 4: leaves QKV un-fused on purpose. AWQ checkpoints
/// store Q/K/V separately and the GEMV kernel handles M=1 well per
/// linear. A future cycle can fuse along N if profiling shows the
/// per-launch overhead is non-trivial; the call sites here would then
/// pre-concatenate `weight_packed` / `weight_scale` and update the
/// `weight_zero_point` packing.
#[cfg(feature = "cuda")]
pub fn upload_gemma4_awq_layer<'a>(
    arena: &'a rvllm_mem::HbmArena<'a>,
    prefix: &str,
    geom: &AwqLayerShapes,
    scheme: &AwqWeightScheme,
    byte_lookup: &dyn Fn(&str) -> Option<(rvllm_core::DType, Vec<usize>, Vec<u8>)>,
) -> Result<crate::weights::AwqLayerWeights, String> {
    let stage_and_upload = |linear: &str, dense: [usize; 2]|
        -> Result<AwqLinearWeight, String>
    {
        let full = format!("{prefix}.{linear}");
        let staged = stage_awq_linear(&full, dense, scheme, byte_lookup)?;
        upload_awq_linear(arena, &staged)
            .map_err(|e| format!("upload {full}: {e:?}"))
    };

    Ok(crate::weights::AwqLayerWeights {
        q_proj:    stage_and_upload("self_attn.q_proj",  [geom.q_out,            geom.hidden])?,
        k_proj:    stage_and_upload("self_attn.k_proj",  [geom.kv_out,           geom.hidden])?,
        v_proj:    stage_and_upload("self_attn.v_proj",  [geom.kv_out,           geom.hidden])?,
        o_proj:    stage_and_upload("self_attn.o_proj",  [geom.o_out,            geom.o_in])?,
        gate_proj: stage_and_upload("mlp.gate_proj",     [geom.mlp_intermediate, geom.hidden])?,
        up_proj:   stage_and_upload("mlp.up_proj",       [geom.mlp_intermediate, geom.hidden])?,
        down_proj: stage_and_upload("mlp.down_proj",     [geom.hidden,           geom.mlp_intermediate])?,
    })
}

/// Result of validating one AWQ linear against the safetensors entries
/// the shard index resolved.
pub fn validate_awq_linear(
    linear_name: &str,
    dense_shape: [usize; 2],
    scheme: &AwqWeightScheme,
    lookup: &dyn Fn(&str) -> Option<(rvllm_core::DType, Vec<usize>)>,
) -> Result<AwqLinearLayout, String> {
    let names    = AwqTensorNames::for_linear(linear_name);
    let expected = AwqExpectedShapes::from_dense(
        dense_shape[0], dense_shape[1], scheme.group_size,
    )?;

    // Helper: lookup, error path bakes in the linear name + entry kind.
    let need = |key: &str, expected_dtype: rvllm_core::DType, expected_shape: &[usize]| -> Result<(), String> {
        let (dtype, shape) = lookup(key)
            .ok_or_else(|| format!("AWQ entry missing: {key}"))?;
        if dtype != expected_dtype {
            return Err(format!(
                "AWQ entry {key} dtype = {dtype:?}, expected {expected_dtype:?}"
            ));
        }
        if shape != expected_shape {
            return Err(format!(
                "AWQ entry {key} shape = {shape:?}, expected {expected_shape:?}"
            ));
        }
        Ok(())
    };

    // Cycle 43.1: only W4 AWQ is wired today. Reject other widths loudly
    // so a future W8 / W2 checkpoint can't silently fall through with W4
    // assumptions baked into the staging byte-size formulas + GEMV kernel.
    if scheme.num_bits != 4 {
        return Err(format!(
            "AWQ scheme.num_bits = {} unsupported; only 4 wired",
            scheme.num_bits
        ));
    }

    need(&names.packed,     rvllm_core::DType::I32,  &expected.packed_shape)?;
    need(&names.scale,      rvllm_core::DType::Bf16, &expected.scale_shape)?;
    need(&names.zero_point, rvllm_core::DType::I32,  &expected.zero_point_shape)?;
    need(&names.shape,      rvllm_core::DType::I64,  &expected.shape_shape)?;

    Ok(AwqLinearLayout { names, expected, dense: dense_shape })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> serde_json::Value {
        // Mirrors the real ebircak/gemma-4-31B-it-4bit-W4A16-AWQ
        // quantization_config block.
        json!({
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "group_size": 128,
                        "strategy": "group",
                        "symmetric": false,
                        "type": "int",
                        "zp_dtype": "torch.int8"
                    }
                }
            },
            "format": "pack-quantized",
            "ignore": [
                "lm_head",
                "re:model.vision_tower.*",
                "re:model.multi_modal_projector.*"
            ]
        })
    }

    #[test]
    fn parses_real_gemma4_awq_config() {
        let parsed = AwqConfig::from_json(Some(&cfg())).expect("parse").expect("Some");
        assert_eq!(parsed.format, AwqPackFormat::PackQuantized);
        assert_eq!(parsed.scheme.num_bits, 4);
        assert_eq!(parsed.scheme.group_size, 128);
        assert!(!parsed.scheme.symmetric);
        assert_eq!(parsed.ignore.len(), 3);
    }

    #[test]
    fn ignores_lm_head_literal() {
        let parsed = AwqConfig::from_json(Some(&cfg())).unwrap().unwrap();
        assert!(parsed.is_ignored("lm_head"));
        assert!(!parsed.is_ignored("model.language_model.layers.0.self_attn.q_proj"));
    }

    #[test]
    fn ignores_vision_tower_substring() {
        let parsed = AwqConfig::from_json(Some(&cfg())).unwrap().unwrap();
        assert!(parsed.is_ignored("model.vision_tower.layer_0.weight"));
        // multi_modal_projector has metacharacter-free regex too.
        assert!(parsed.is_ignored("foo.model.multi_modal_projector.bar"));
    }

    #[test]
    fn unquantized_returns_none() {
        let r = AwqConfig::from_json(None).expect("ok");
        assert!(r.is_none());
    }

    #[test]
    fn rejects_unknown_format() {
        let bad = json!({
            "config_groups": {
                "g": {"targets": ["Linear"], "weights": {"num_bits": 4, "group_size": 128, "strategy": "group", "type": "int"}}
            },
            "format": "marlin"
        });
        let err = AwqConfig::from_json(Some(&bad)).expect_err("must reject");
        assert!(err.contains("marlin") || err.contains("unsupported"));
    }

    // === Cycle 42 step 3c tensor classifier + shape validator tests ===

    #[test]
    fn tensor_names_compose_correctly() {
        let names = AwqTensorNames::for_linear(
            "model.language_model.layers.0.self_attn.q_proj"
        );
        assert_eq!(names.packed,
            "model.language_model.layers.0.self_attn.q_proj.weight_packed");
        assert_eq!(names.scale,
            "model.language_model.layers.0.self_attn.q_proj.weight_scale");
        assert_eq!(names.zero_point,
            "model.language_model.layers.0.self_attn.q_proj.weight_zero_point");
        assert_eq!(names.shape,
            "model.language_model.layers.0.self_attn.q_proj.weight_shape");
    }

    #[test]
    fn expected_shapes_match_real_gemma4_q_proj() {
        // Real ebircak/gemma-4-31B-it-4bit-W4A16-AWQ q_proj header:
        //   weight_packed:    I32 [8192, 672]
        //   weight_scale:     BF16 [8192, 42]
        //   weight_zero_point: I32 [1024, 42]
        //   weight_shape:     I64 [2]
        // dense shape [N=8192, K=5376], group_size=128
        let exp = AwqExpectedShapes::from_dense(8192, 5376, 128).expect("ok");
        assert_eq!(exp.packed_shape,     [8192, 672]);   // K/8
        assert_eq!(exp.scale_shape,      [8192, 42]);    // K/g
        assert_eq!(exp.zero_point_shape, [1024, 42]);    // N/8, K/g
        assert_eq!(exp.shape_shape,      [2]);
    }

    #[test]
    fn expected_shapes_match_real_gemma4_down_proj() {
        // down_proj: dense [N=5376, K=21504], group_size=128
        //   weight_packed:    I32 [5376, 2688]   K/8
        //   weight_scale:     BF16 [5376, 168]   K/g
        //   weight_zero_point: I32 [672, 168]    N/8, K/g
        let exp = AwqExpectedShapes::from_dense(5376, 21504, 128).expect("ok");
        assert_eq!(exp.packed_shape,     [5376, 2688]);
        assert_eq!(exp.scale_shape,      [5376, 168]);
        assert_eq!(exp.zero_point_shape, [672, 168]);
    }

    #[test]
    fn expected_shapes_reject_unaligned_n() {
        // N=5377 not divisible by 8 → INT4-along-N zero_point can't be
        // packed cleanly → reject.
        assert!(AwqExpectedShapes::from_dense(5377, 5376, 128).is_err());
    }

    #[test]
    fn expected_shapes_reject_unaligned_k() {
        // K=5377 not divisible by 8 → INT4-along-K weight_packed can't.
        assert!(AwqExpectedShapes::from_dense(5376, 5377, 128).is_err());
    }

    #[test]
    fn expected_shapes_reject_k_not_multiple_of_group() {
        // K=128 OK with g=128 but K=200 with g=128 is not → reject.
        assert!(AwqExpectedShapes::from_dense(8, 200, 128).is_err());
    }

    #[test]
    fn validate_awq_linear_happy_path() {
        let scheme = AwqWeightScheme {
            num_bits: 4, group_size: 128, symmetric: false,
        };
        // Mock the safetensors lookup: returns the dtypes/shapes the
        // real ebircak Gemma 4 31B q_proj would have.
        let entries: std::collections::HashMap<&str, (rvllm_core::DType, Vec<usize>)> = [
            ("model.language_model.layers.0.self_attn.q_proj.weight_packed",
              (rvllm_core::DType::I32,  vec![8192, 672])),
            ("model.language_model.layers.0.self_attn.q_proj.weight_scale",
              (rvllm_core::DType::Bf16, vec![8192, 42])),
            ("model.language_model.layers.0.self_attn.q_proj.weight_zero_point",
              (rvllm_core::DType::I32,  vec![1024, 42])),
            ("model.language_model.layers.0.self_attn.q_proj.weight_shape",
              (rvllm_core::DType::I64,  vec![2])),
        ].into_iter().collect();
        let lookup = |k: &str| entries.get(k).cloned();

        let layout = validate_awq_linear(
            "model.language_model.layers.0.self_attn.q_proj",
            [8192, 5376],
            &scheme,
            &lookup,
        ).expect("validates");
        assert_eq!(layout.dense, [8192, 5376]);
        assert_eq!(layout.expected.packed_shape, [8192, 672]);
    }

    #[test]
    fn validate_awq_linear_missing_entry_errors() {
        let scheme = AwqWeightScheme {
            num_bits: 4, group_size: 128, symmetric: false,
        };
        // Provide only 3 of 4 entries (missing weight_zero_point).
        let entries: std::collections::HashMap<&str, (rvllm_core::DType, Vec<usize>)> = [
            ("layer.q_proj.weight_packed", (rvllm_core::DType::I32,  vec![8192, 672])),
            ("layer.q_proj.weight_scale",  (rvllm_core::DType::Bf16, vec![8192, 42])),
            ("layer.q_proj.weight_shape",  (rvllm_core::DType::I64,  vec![2])),
        ].into_iter().collect();
        let lookup = |k: &str| entries.get(k).cloned();
        let err = validate_awq_linear(
            "layer.q_proj", [8192, 5376], &scheme, &lookup
        ).expect_err("must fail");
        assert!(err.contains("weight_zero_point"), "got: {err}");
    }

    #[test]
    fn validate_awq_linear_wrong_dtype_errors() {
        let scheme = AwqWeightScheme {
            num_bits: 4, group_size: 128, symmetric: false,
        };
        // Pass U8 instead of expected I32 for weight_packed (wrong format).
        let entries: std::collections::HashMap<&str, (rvllm_core::DType, Vec<usize>)> = [
            ("l.q.weight_packed",     (rvllm_core::DType::U8,   vec![8192, 672])),
            ("l.q.weight_scale",      (rvllm_core::DType::Bf16, vec![8192, 42])),
            ("l.q.weight_zero_point", (rvllm_core::DType::I32,  vec![1024, 42])),
            ("l.q.weight_shape",      (rvllm_core::DType::I64,  vec![2])),
        ].into_iter().collect();
        let lookup = |k: &str| entries.get(k).cloned();
        let err = validate_awq_linear(
            "l.q", [8192, 5376], &scheme, &lookup
        ).expect_err("must fail");
        assert!(err.contains("dtype"), "got: {err}");
    }

    #[test]
    fn validate_awq_linear_wrong_shape_errors() {
        let scheme = AwqWeightScheme {
            num_bits: 4, group_size: 128, symmetric: false,
        };
        // Provide weight_packed with N=4096 but dense [8192, 5376] →
        // expected [8192, 672], got [4096, 672] — should reject.
        let entries: std::collections::HashMap<&str, (rvllm_core::DType, Vec<usize>)> = [
            ("l.q.weight_packed",     (rvllm_core::DType::I32,  vec![4096, 672])),
            ("l.q.weight_scale",      (rvllm_core::DType::Bf16, vec![8192, 42])),
            ("l.q.weight_zero_point", (rvllm_core::DType::I32,  vec![1024, 42])),
            ("l.q.weight_shape",      (rvllm_core::DType::I64,  vec![2])),
        ].into_iter().collect();
        let lookup = |k: &str| entries.get(k).cloned();
        let err = validate_awq_linear(
            "l.q", [8192, 5376], &scheme, &lookup
        ).expect_err("must fail");
        assert!(err.contains("shape"), "got: {err}");
    }

    // === Cycle 43 step 3d stage_awq_linear tests ===

    /// Helper: builds a byte-lookup that returns dtype/shape/bytes of
    /// the expected size for the given dense [N, K] + group_size,
    /// optionally overriding one entry's byte length.
    fn mock_byte_lookup(
        prefix: &str,
        n: usize,
        k: usize,
        g: usize,
        override_packed_len: Option<usize>,
    ) -> std::collections::HashMap<String, (rvllm_core::DType, Vec<usize>, Vec<u8>)> {
        let names = AwqTensorNames::for_linear(prefix);
        let packed_bytes = override_packed_len.unwrap_or(n * (k / 8) * 4);
        let scale_bytes  = n * (k / g) * 2;
        let zero_bytes   = (n / 8) * (k / g) * 4;
        let mut shape_payload = Vec::with_capacity(16);
        shape_payload.extend_from_slice(&(n as i64).to_le_bytes());
        shape_payload.extend_from_slice(&(k as i64).to_le_bytes());
        [
            (names.packed,     (rvllm_core::DType::I32,  vec![n, k / 8],     vec![0u8; packed_bytes])),
            (names.scale,      (rvllm_core::DType::Bf16, vec![n, k / g],     vec![0u8; scale_bytes])),
            (names.zero_point, (rvllm_core::DType::I32,  vec![n / 8, k / g], vec![0u8; zero_bytes])),
            (names.shape,      (rvllm_core::DType::I64,  vec![2],            shape_payload)),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn stage_awq_linear_happy_path() {
        let scheme = AwqWeightScheme { num_bits: 4, group_size: 128, symmetric: false };
        let prefix = "model.language_model.layers.0.self_attn.q_proj";
        let map = mock_byte_lookup(prefix, 8192, 5376, 128, None);
        let lookup = |k: &str| map.get(k).cloned();

        let staged = stage_awq_linear(prefix, [8192, 5376], &scheme, &lookup)
            .expect("stages");
        assert_eq!(staged.layout.dense, [8192, 5376]);
        assert_eq!(staged.packed.len(),     8192 * (5376 / 8) * 4);
        assert_eq!(staged.scale.len(),      8192 * (5376 / 128) * 2);
        assert_eq!(staged.zero_point.len(), (8192 / 8) * (5376 / 128) * 4);
        assert_eq!(staged.shape.len(),      16);
        assert_eq!(staged.host_bytes(),
            staged.packed.len() + staged.scale.len()
            + staged.zero_point.len() + staged.shape.len());
    }

    #[test]
    fn stage_awq_linear_missing_entry_errors() {
        let scheme = AwqWeightScheme { num_bits: 4, group_size: 128, symmetric: false };
        // No entries at all → validate fails first.
        let lookup = |_k: &str| None;
        let err = stage_awq_linear("l.q", [8192, 5376], &scheme, &lookup)
            .expect_err("must fail");
        assert!(err.contains("missing") || err.contains("AWQ entry"), "got: {err}");
    }

    #[test]
    fn stage_awq_linear_byte_size_mismatch_errors() {
        let scheme = AwqWeightScheme { num_bits: 4, group_size: 128, symmetric: false };
        // Truncate weight_packed bytes by 1 — validate passes (dtype + shape OK)
        // but stage's byte-size sanity check rejects.
        let prefix = "l.q";
        let correct = 8192 * (5376 / 8) * 4;
        let map = mock_byte_lookup(prefix, 8192, 5376, 128, Some(correct - 1));
        let lookup = |k: &str| map.get(k).cloned();
        let err = stage_awq_linear(prefix, [8192, 5376], &scheme, &lookup)
            .expect_err("must fail");
        assert!(err.contains("weight_packed bytes"), "got: {err}");
    }

    #[test]
    fn validate_awq_linear_rejects_non_w4() {
        let scheme = AwqWeightScheme {
            num_bits: 8, group_size: 128, symmetric: false,
        };
        let lookup = |_k: &str| None;
        let err = validate_awq_linear("l.q", [8192, 5376], &scheme, &lookup)
            .expect_err("must reject non-W4");
        assert!(err.contains("num_bits"), "got: {err}");
    }

    #[test]
    fn stage_awq_linear_rejects_shape_payload_mismatch() {
        let scheme = AwqWeightScheme { num_bits: 4, group_size: 128, symmetric: false };
        let prefix = "l.q";
        let mut map = mock_byte_lookup(prefix, 8192, 5376, 128, None);
        // Corrupt weight_shape payload: [N, K] = [9999, 5376] disagrees with dense.
        let bad_shape = {
            let mut v = Vec::with_capacity(16);
            v.extend_from_slice(&(9999_i64).to_le_bytes());
            v.extend_from_slice(&(5376_i64).to_le_bytes());
            v
        };
        let names = AwqTensorNames::for_linear(prefix);
        map.insert(names.shape, (rvllm_core::DType::I64, vec![2], bad_shape));
        let lookup = |k: &str| map.get(k).cloned();
        let err = stage_awq_linear(prefix, [8192, 5376], &scheme, &lookup)
            .expect_err("must reject");
        assert!(err.contains("weight_shape contents"), "got: {err}");
    }

    #[test]
    fn rejects_non_int_type() {
        let bad = json!({
            "config_groups": {
                "g": {"targets": ["Linear"], "weights": {"num_bits": 4, "group_size": 128, "strategy": "group", "type": "float"}}
            },
            "format": "pack-quantized"
        });
        assert!(AwqConfig::from_json(Some(&bad)).is_err());
    }
}
