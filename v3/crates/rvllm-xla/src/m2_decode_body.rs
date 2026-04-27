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
    let batch = shape.batch;
    let kv_bytes = shape.layer_kv_cache_bytes();
    let n_cols = 128usize;
    let k_total = M2_HIDDEN;
    let k_step = 32usize;
    let k_tiles = std::env::var("RVLLM_M2_W1_K_TILES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4);
    if k_total % k_step != 0 {
        return Err(invalid(
            "k_total",
            "K must be divisible by 32 sublane stride",
        ));
    }
    let mut k_constants = String::new();
    let mut tile_blocks = String::new();
    let mut acc_prev = "%acc_init".to_string();
    for tile_idx in 0..k_tiles {
        let k_off = tile_idx * k_step;
        let k_idx = if k_off == 0 {
            "%c0".to_string()
        } else {
            k_constants.push_str(&format!(
                "    %ck_{tile_idx} = arith.constant {k_off} : index\n"
            ));
            format!("%ck_{tile_idx}")
        };
        let acc_next = format!("%acc_{tile_idx}");
        tile_blocks.push_str(&format!(
            r#"    %w_i8_{tile_idx} = vector.load %w1_block_t[{k_idx}, %c0] : memref<{k_total}x{n_cols}xi8>, vector<{k_step}x{n_cols}xi8>
    %w_f32_{tile_idx} = arith.sitofp %w_i8_{tile_idx} : vector<{k_step}x{n_cols}xi8> to vector<{k_step}x{n_cols}xf32>
    %w_scaled_{tile_idx} = arith.mulf %w_f32_{tile_idx}, %scale_b : vector<{k_step}x{n_cols}xf32>
    %h_bf16_{tile_idx} = vector.load %hidden[%c0, {k_idx}] : memref<{batch}x{k_total}xbf16>, vector<{batch}x{k_step}xbf16>
    %h_f32_{tile_idx} = arith.extf %h_bf16_{tile_idx} : vector<{batch}x{k_step}xbf16> to vector<{batch}x{k_step}xf32>
    {acc_next} = vector.contract #w1_dot_trait %h_f32_{tile_idx}, %w_scaled_{tile_idx}, {acc_prev} : vector<{batch}x{k_step}xf32>, vector<{k_step}x{n_cols}xf32> into vector<{batch}x{n_cols}xf32>
"#,
            tile_idx = tile_idx,
            k_idx = k_idx,
            k_total = k_total,
            n_cols = n_cols,
            k_step = k_step,
            batch = batch,
            acc_prev = acc_prev,
            acc_next = acc_next,
        ));
        acc_prev = acc_next;
    }
    Ok(format!(
        r#"#w1_dot_trait = {{
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}}
module attributes {{"stable_mosaic.version" = "1"}} {{
  func.func @main(
      %hidden: memref<{batch}x{k_total}xbf16>,
      %positions: memref<{batch}xi32>,
      %kv_in: memref<{kv_bytes}xi8>,
      %layer_offsets: memref<34xi32>,
      %expert_directory: memref<256x25xi32>,
      %w1_block_t: memref<{k_total}x{n_cols}xi8>,
      %w1_row_scales: memref<{n_cols}xf32>,
      %hidden_out: memref<{batch}x{k_total}xbf16>,
      %kv_out: memref<{kv_bytes}xi8>) attributes {{
        dimension_semantics = [],
        scalar_prefetch = 0 : i64,
        scratch_operands = 0 : i64,
        rvllm.int8_probe = "w1_observable_unrolled_bk32x{k_tiles}"
      }} {{
    %c0 = arith.constant 0 : index
{k_constants}    %scale_v = vector.load %w1_row_scales[%c0] : memref<{n_cols}xf32>, vector<{n_cols}xf32>
    %scale_2d = vector.shape_cast %scale_v : vector<{n_cols}xf32> to vector<1x{n_cols}xf32>
    %scale_b = vector.broadcast %scale_2d : vector<1x{n_cols}xf32> to vector<{k_step}x{n_cols}xf32>
    %acc_init = arith.constant dense<0.000000e+00> : vector<{batch}x{n_cols}xf32>
    %hidden_v = vector.load %hidden[%c0, %c0] : memref<{batch}x{k_total}xbf16>, vector<{batch}x{k_total}xbf16>
    vector.store %hidden_v, %hidden_out[%c0, %c0] : memref<{batch}x{k_total}xbf16>, vector<{batch}x{k_total}xbf16>
    %kv_v = vector.load %kv_in[%c0] : memref<{kv_bytes}xi8>, vector<512xi8>
    vector.store %kv_v, %kv_out[%c0] : memref<{kv_bytes}xi8>, vector<512xi8>
{tile_blocks}    %out_bf16 = arith.truncf {acc_prev} : vector<{batch}x{n_cols}xf32> to vector<{batch}x{n_cols}xbf16>
    vector.store %out_bf16, %hidden_out[%c0, %c0] : memref<{batch}x{k_total}xbf16>, vector<{batch}x{n_cols}xbf16>
    return
  }}
}}
"#,
        batch = batch,
        k_total = k_total,
        n_cols = n_cols,
        kv_bytes = kv_bytes,
        k_step = k_step,
        k_tiles = k_tiles,
        k_constants = k_constants,
        tile_blocks = tile_blocks,
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
    fn emits_int8_lowered_body_observable_row_tile_w1() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        let mlir = m2_decode_layer_int8_lowered_body_mlir(&shape, 4_294_967_296).unwrap();
        assert!(mlir.contains("rvllm.int8_probe = \"w1_observable_unrolled_bk32x4\""));
        assert!(mlir.contains("memref<3072x128xi8>"));
        assert!(mlir.contains("memref<128xf32>"));
        assert!(mlir.contains("memref<33554432xi8>"));
        assert!(mlir.contains("memref<34xi32>"));
        assert!(mlir.contains("memref<256x25xi32>"));
        assert!(mlir.contains("memref<8x3072xbf16>"));
        assert!(
            !mlir.contains("affine.for"),
            "affine.for is unsupported by Mosaic infer-vector-layout"
        );
        assert!(
            !mlir.contains("scf.for"),
            "scf.for cannot prove K stride alignment for i8; use unrolled K tiles"
        );
        assert!(mlir.contains("vector<32x128xi8>"));
        assert!(mlir.contains("vector<8x32xbf16>"));
        assert!(mlir.contains("arith.sitofp"));
        assert!(mlir.contains("arith.truncf"));
        assert_eq!(
            mlir.matches("vector.contract").count(),
            4,
            "expected 4 unrolled K-tile contractions"
        );
        assert!(
            mlir.starts_with("#w1_dot_trait"),
            "vector.contract trait alias must precede module per Mosaic contract"
        );
        assert!(
            mlir.contains(
                "vector.store %out_bf16, %hidden_out[%c0, %c0] : memref<8x3072xbf16>, vector<8x128xbf16>"
            ),
            "must observably store W1 result to hidden_out[:, 0:128]"
        );
        assert!(
            mlir.contains("vector.store %hidden_v, %hidden_out"),
            "must pass through full hidden before overlay"
        );
    }
}
