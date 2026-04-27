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
    %hidden_v = vector.load %hidden[%c0, %c0] : memref<{batch}x{hidden}xbf16>, vector<{batch}x{hidden}xbf16>
    vector.store %hidden_v, %hidden_out[%c0, %c0] : memref<{batch}x{hidden}xbf16>, vector<{batch}x{hidden}xbf16>
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
    let mut k_constants = String::new();
    let mut k_tiles = String::new();
    let mut acc_prev = "%zero".to_string();
    const K_TILE: usize = 512;
    for (tile_idx, k) in (0..M2_HIDDEN).step_by(K_TILE).enumerate() {
        let k_idx = if k == 0 {
            "%c0".to_string()
        } else {
            k_constants.push_str(&format!("    %c{k} = arith.constant {k} : index\n"));
            format!("%c{k}")
        };
        let acc_next = format!("%acc_{tile_idx}");
        k_tiles.push_str(&format!(
            r#"    %h_bf16_{tile_idx} = vector.load %hidden[%c0, {k_idx}] : memref<{batch}x{hidden}xbf16>, vector<{batch}x{k_tile}xbf16>
    %h_mat_{tile_idx} = arith.extf %h_bf16_{tile_idx} : vector<{batch}x{k_tile}xbf16> to vector<{batch}x{k_tile}xf32>
    %w_i8_{tile_idx} = vector.load %w1_block_t[{k_idx}, %c0] : memref<{hidden}x128xi8>, vector<{k_tile}x128xi8>
    %w_f32_{tile_idx} = arith.sitofp %w_i8_{tile_idx} : vector<{k_tile}x128xi8> to vector<{k_tile}x128xf32>
    %scale_b_{tile_idx} = vector.broadcast %scale_v : vector<128xf32> to vector<{k_tile}x128xf32>
    %w_scaled_{tile_idx} = arith.mulf %w_f32_{tile_idx}, %scale_b_{tile_idx} : vector<{k_tile}x128xf32>
    {acc_next} = vector.contract {{
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    }} %h_mat_{tile_idx}, %w_scaled_{tile_idx}, {acc_prev} : vector<{batch}x{k_tile}xf32>, vector<{k_tile}x128xf32> into vector<{batch}x128xf32>
"#,
            batch = shape.batch,
            hidden = M2_HIDDEN,
            k_tile = K_TILE,
        ));
        acc_prev = acc_next;
    }
    Ok(format!(
        r#"module attributes {{"stable_mosaic.version" = "1"}} {{
  func.func @main(
      %hidden: memref<{batch}x{hidden}xbf16>,
      %w1_block_t: memref<{hidden}x128xi8>,
      %w1_row_scales: memref<128xf32>,
      %hidden_out: memref<{batch}x128xbf16>) attributes {{
        dimension_semantics = [],
        scalar_prefetch = 0 : i64,
        scratch_operands = 0 : i64,
        rvllm.int8_probe = "w1_i8_full_k_128_cols"
      }} {{
    %c0 = arith.constant 0 : index
{k_constants}
    %zero = arith.constant dense<0.000000e+00> : vector<{batch}x128xf32>
    %scale_v = vector.load %w1_row_scales[%c0] : memref<128xf32>, vector<128xf32>
{k_tiles}
    %out_bf16 = arith.truncf {acc_prev} : vector<{batch}x128xf32> to vector<{batch}x128xbf16>
    vector.store %out_bf16, %hidden_out[%c0, %c0] : memref<{batch}x128xbf16>, vector<{batch}x128xbf16>
    return
  }}
}}
"#,
        batch = shape.batch,
        hidden = M2_HIDDEN,
        k_constants = k_constants,
        k_tiles = k_tiles,
        acc_prev = acc_prev,
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
        assert!(mlir.contains("memref<128xf32>"));
        assert!(mlir.contains("vector<8x128xbf16>"));
        assert!(mlir.contains("vector<512x128xi8>"));
        assert!(mlir.contains("vector<8x512xf32>"));
        assert!(mlir.contains("vector<128xf32>"));
        assert!(mlir.contains("vector.contract"));
        assert!(mlir.contains("arith.sitofp"));
        assert!(mlir.contains("arith.mulf"));
        assert!(!mlir.contains("scf.for"));
        assert!(mlir.contains("%c2560 = arith.constant 2560 : index"));
        assert!(mlir.contains("vector<8x128xbf16>"));
    }
}
