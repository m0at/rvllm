use rvllm_core::{ConfigError, Result, RvllmError};

use crate::{M2GraphPhase, M2GraphShape, M2_HIDDEN};

pub fn m2_decode_layer_lowered_body_mlir(
    shape: &M2GraphShape,
    _weight_arena_local_bytes: usize,
) -> Result<String> {
    shape.validate()?;
    if shape.phase != M2GraphPhase::Decode {
        return Err(invalid("phase", "expected decode graph shape"));
    }
    let kv_bytes = shape.layer_kv_cache_bytes();
    Ok(format!(
        r#"module attributes {{"stable_mosaic.version" = "1"}} {{
  func.func @main(
      %hidden: memref<{batch}x{hidden}xbf16>,
      %positions: memref<{batch}xi32>,
      %kv_in: memref<{kv_bytes}xi8>,
      %layer_offsets: memref<34xi32>,
      %expert_directory: memref<256x25xi32>,
      %hidden_out: memref<{batch}x{hidden}xbf16>,
      %kv_out: memref<{kv_bytes}xi8>) attributes {{
        dimension_semantics = [],
        scalar_prefetch = 0 : i64,
        scratch_operands = 0 : i64
      }} {{
    %c0 = arith.constant 0 : index
    %hidden_v = vector.load %hidden[%c0, %c0] : memref<{batch}x{hidden}xbf16>, vector<1x128xbf16>
    vector.store %hidden_v, %hidden_out[%c0, %c0] : memref<{batch}x{hidden}xbf16>, vector<1x128xbf16>
    %kv_v = vector.load %kv_in[%c0] : memref<{kv_bytes}xi8>, vector<512xi8>
    vector.store %kv_v, %kv_out[%c0] : memref<{kv_bytes}xi8>, vector<512xi8>
    return
  }}
}}
"#,
        batch = shape.batch,
        hidden = M2_HIDDEN,
        kv_bytes = kv_bytes,
    ))
}

pub fn m2_decode_layer_int8_lowered_body_mlir(
    shape: &M2GraphShape,
    _weight_arena_local_bytes: usize,
) -> Result<String> {
    shape.validate()?;
    if shape.phase != M2GraphPhase::Decode {
        return Err(invalid("phase", "expected decode graph shape"));
    }
    let kv_bytes = shape.layer_kv_cache_bytes();
    Ok(format!(
        r#"module attributes {{"stable_mosaic.version" = "1"}} {{
  func.func @main(
      %hidden: memref<{batch}x{hidden}xbf16>,
      %positions: memref<{batch}xi32>,
      %kv_in: memref<{kv_bytes}xi8>,
      %layer_offsets: memref<34xi32>,
      %expert_directory: memref<256x25xi32>,
      %w1_block_t: memref<{hidden}x128xi8>,
      %hidden_out: memref<{batch}x{hidden}xbf16>,
      %kv_out: memref<{kv_bytes}xi8>) attributes {{
        dimension_semantics = [],
        scalar_prefetch = 0 : i64,
        scratch_operands = 0 : i64,
        rvllm.int8_probe = "w1_i8_full_k_128_cols"
      }} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c_batch = arith.constant {batch} : index
    %c_hidden = arith.constant {hidden} : index
    %hidden_v = vector.load %hidden[%c0, %c0] : memref<{batch}x{hidden}xbf16>, vector<1x128xbf16>
    vector.store %hidden_v, %hidden_out[%c0, %c0] : memref<{batch}x{hidden}xbf16>, vector<1x128xbf16>
    %kv_v = vector.load %kv_in[%c0] : memref<{kv_bytes}xi8>, vector<512xi8>
    vector.store %kv_v, %kv_out[%c0] : memref<{kv_bytes}xi8>, vector<512xi8>
    %zero = arith.constant dense<0.000000e+00> : vector<1x128xf32>
    scf.for %m = %c0 to %c_batch step %c1 {{
      %acc = scf.for %k = %c0 to %c_hidden step %c32 iter_args(%acc_iter = %zero) -> (vector<1x128xf32>) {{
        %h_bf16 = vector.load %hidden[%m, %k] : memref<{batch}x{hidden}xbf16>, vector<32xbf16>
        %h_f32 = arith.extf %h_bf16 : vector<32xbf16> to vector<32xf32>
        %h_mat = vector.broadcast %h_f32 : vector<32xf32> to vector<1x32xf32>
        %w_i8 = vector.load %w1_block_t[%k, %c0] : memref<{hidden}x128xi8>, vector<32x128xi8>
        %w_f32 = arith.sitofp %w_i8 : vector<32x128xi8> to vector<32x128xf32>
        %next = vector.contract {{
          indexing_maps = [
            affine_map<(m, n, k) -> (m, k)>,
            affine_map<(m, n, k) -> (k, n)>,
            affine_map<(m, n, k) -> (m, n)>
          ],
          iterator_types = ["parallel", "parallel", "reduction"],
          kind = #vector.kind<add>
        }} %h_mat, %w_f32, %acc_iter : vector<1x32xf32>, vector<32x128xf32> into vector<1x128xf32>
        scf.yield %next : vector<1x128xf32>
      }}
      %out_bf16 = arith.truncf %acc : vector<1x128xf32> to vector<1x128xbf16>
      vector.store %out_bf16, %hidden_out[%m, %c0] : memref<{batch}x{hidden}xbf16>, vector<1x128xbf16>
    }}
    return
  }}
}}
"#,
        batch = shape.batch,
        hidden = M2_HIDDEN,
        kv_bytes = kv_bytes,
    ))
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_decode_body",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_lowered_mosaic_decode_layer_body_contract() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        let mlir = m2_decode_layer_lowered_body_mlir(&shape, 4_294_967_296).unwrap();
        assert!(mlir.contains("func.func @main"));
        assert!(mlir.contains("memref<8x3072xbf16>"));
        assert!(mlir.contains("memref<33554432xi8>"));
        assert!(mlir.contains("memref<34xi32>"));
        assert!(mlir.contains("memref<256x25xi32>"));
        assert!(!mlir.contains("weight_arena"));
        assert!(!mlir.contains("weight_tile"));
        assert!(mlir.contains("vector.load"));
        assert!(mlir.contains("vector.store"));
        assert!(mlir.contains("vector<512xi8>"));
    }

    #[test]
    fn emits_int8_lowered_body_with_surviving_row_tile_cast() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        let mlir = m2_decode_layer_int8_lowered_body_mlir(&shape, 4_294_967_296).unwrap();
        assert!(mlir.contains("rvllm.int8_probe = \"w1_i8_full_k_128_cols\""));
        assert!(mlir.contains("memref<3072x128xi8>"));
        assert!(mlir.contains("vector<32x128xi8>"));
        assert!(mlir.contains("vector<1x32xf32>"));
        assert!(mlir.contains("vector.contract"));
        assert!(mlir.contains("arith.sitofp"));
        assert!(mlir.contains("scf.for"));
        assert!(mlir.contains("vector<1x128xbf16>"));
    }
}
