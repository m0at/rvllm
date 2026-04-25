use rvllm_core::{ConfigError, Result, RvllmError};

use crate::{M2GraphPhase, M2GraphShape, M2WeightArenaPlan, M2_HIDDEN, M2_NUM_LAYERS, M2_VOCAB};

pub fn m2_decode_graph_mlir(
    kernel_name: &str,
    shape: &M2GraphShape,
    arena: &M2WeightArenaPlan,
) -> Result<String> {
    shape.validate()?;
    if shape.phase != M2GraphPhase::Decode {
        return Err(invalid("phase", "expected decode graph shape"));
    }
    if !is_mlir_symbol(kernel_name) {
        return Err(invalid("kernel_name", "must be an MLIR symbol"));
    }
    Ok(format!(
        r#"module attributes {{rvllm.kind = "m2_decode_graph"}} {{
  func.func @{kernel_name}(
      %token_ids: memref<{batch}xi32>,
      %positions: memref<{batch}xi32>,
      %kv_cache: memref<{kv_bytes}xi8>,
      %weight_arena: memref<{weight_bytes}xi8>)
      -> (memref<{batch}x{vocab}xbf16>, memref<{batch}xi32>, memref<{kv_bytes}xi8>)
      attributes {{
        rvllm.signature = "token_ids,positions,kv_cache,weight_arena -> logits,next_token,kv_cache",
        rvllm.phase = "decode",
        rvllm.batch = {batch} : i64,
        rvllm.ctx = {ctx} : i64,
        rvllm.layers = {layers} : i64,
        rvllm.hidden = {hidden} : i64,
        rvllm.vocab = {vocab} : i64,
        rvllm.kv_cache_bytes = {kv_bytes} : i64,
        rvllm.weight_arena_bytes = {weight_bytes} : i64,
        rvllm.weight_entries = {weight_entries} : i64,
        rvllm.weight_alignment = {weight_alignment} : i64,
        rvllm.weight_metadata = "compile_time_offsets_from_M2WeightArenaPlan",
        rvllm.lowering = "rust_mlir_custom_call",
        rvllm.lowering_plan = "embed -> 62 layer scan -> flat-arena dense loads -> flat-arena NVFP4 expert custom calls -> final norm -> lm_head -> argmax"
      }} {{
    // Contract body placeholder. The next slice replaces this with real
    // region bodies/custom-calls that consume offsets from M2WeightArenaPlan.
    // No Python/JAX graph emission belongs on this path.
    %logits = memref.alloc() : memref<{batch}x{vocab}xbf16>
    %next_token = memref.alloc() : memref<{batch}xi32>
    return %logits, %next_token, %kv_cache : memref<{batch}x{vocab}xbf16>, memref<{batch}xi32>, memref<{kv_bytes}xi8>
  }}
}}
"#,
        kernel_name = kernel_name,
        batch = shape.batch,
        ctx = shape.ctx,
        layers = M2_NUM_LAYERS,
        hidden = M2_HIDDEN,
        vocab = M2_VOCAB,
        kv_bytes = shape.kv_cache_bytes(),
        weight_bytes = arena.total_bytes,
        weight_entries = arena.entries.len(),
        weight_alignment = arena.alignment,
    ))
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_decode_graph",
    )
}

fn is_mlir_symbol(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{M2GraphAbi, M2WeightUploadPlan};

    use super::*;

    fn arena() -> M2WeightArenaPlan {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema");
        let shape = M2GraphShape::decode(8, 2048, 1);
        let abi = M2GraphAbi::new(shape).unwrap();
        M2WeightUploadPlan::from_index_dir(model_dir, &abi)
            .unwrap()
            .flat_arena(128)
            .unwrap()
    }

    #[test]
    fn emits_decode_graph_contract_over_flat_weight_arena() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        let arena = arena();
        let mlir = m2_decode_graph_mlir("rvllm_m2_decode", &shape, &arena).unwrap();
        assert!(mlir.contains("rvllm.kind = \"m2_decode_graph\""));
        assert!(mlir.contains("memref<8xi32>"));
        assert!(mlir.contains("memref<2080374784xi8>"));
        assert!(mlir.contains("weight_arena"));
        assert!(mlir.contains("rvllm.weight_entries = 191069 : i64"));
        assert!(mlir.contains("rvllm.lowering = \"rust_mlir_custom_call\""));
    }

    #[test]
    fn rejects_prefill_shape_for_decode_graph() {
        let shape = M2GraphShape::prefill(8, 20, 2048, 1);
        let arena = arena();
        assert!(m2_decode_graph_mlir("rvllm_m2_decode", &shape, &arena).is_err());
    }
}
