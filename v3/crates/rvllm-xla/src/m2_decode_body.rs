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
    let n_step = 128usize;
    let n_total = std::env::var("RVLLM_M2_W1_N_COLS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(128);
    if n_total % n_step != 0 {
        return Err(invalid(
            "n_total",
            "N must be divisible by 128 sublane stride",
        ));
    }
    let n_tiles = n_total / n_step;
    let n_loop = n_tiles > 1;
    let k_total = M2_HIDDEN;
    // K_step controls the inner contract shape that feeds the MXU. v6e Trillium
    // has a 256x256 MXU, so vector<batch x K_step xf32> @ vector<K_step x N_step xf32>
    // wants K_step >= 128 to actually use a full systolic column. Default stays
    // at 32 (the proven path) but can be bumped via RVLLM_M2_W1_K_STEP.
    let k_step = std::env::var("RVLLM_M2_W1_K_STEP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(32);
    if k_step == 0 || (k_step % 32) != 0 {
        return Err(invalid(
            "k_step",
            "K_step must be a positive multiple of 32 (TPU sublane stride for i8)",
        ));
    }
    if k_total % k_step != 0 {
        return Err(invalid(
            "k_total",
            "K_total (M2_HIDDEN=3072) must be divisible by K_step",
        ));
    }
    let k_tiles = std::env::var("RVLLM_M2_W1_K_TILES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4);

    // K offset constants (compile-time literal indices into %w1_block_t and %hidden).
    let mut k_constants = String::new();
    // Hoisted hidden loads -- independent of N loop variable, so we compute the
    // bf16->f32 conversion once and reuse inside the N loop (or outside if no loop).
    let mut hidden_loads = String::new();
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
        hidden_loads.push_str(&format!(
            "    %h_bf16_{tile_idx} = vector.load %hidden[%c0, {k_idx}] : memref<{batch}x{k_total}xbf16>, vector<{batch}x{k_step}xbf16>\n    %h_f32_{tile_idx} = arith.extf %h_bf16_{tile_idx} : vector<{batch}x{k_step}xbf16> to vector<{batch}x{k_step}xf32>\n",
        ));
    }

    // Operand contracts:
    //   single-tile (n_loop == false): 2D memrefs
    //     %w1_block_t: memref<{k_total}x128xi8>
    //     %w1_row_scales: memref<128xf32>
    //   multi-tile (n_loop == true): 3D weight memref + 2D scale memref so
    //   Mosaic can prove sublane alignment for vector<32x1x128xi8> and
    //   vector<1x128xf32> loads inside scf.for over the N-tile index.
    //     %w1_block_t: memref<{k_total}x{n_tiles}x128xi8>
    //     %w1_row_scales: memref<{n_tiles}x128xf32>
    let (w1_block_ty, w1_row_scales_ty) = if n_loop {
        (
            format!("memref<{k_total}x{n_tiles}x{n_step}xi8>"),
            format!("memref<{n_tiles}x{n_step}xf32>"),
        )
    } else {
        (
            format!("memref<{k_total}x{n_step}xi8>"),
            format!("memref<{n_step}xf32>"),
        )
    };

    // Per-K-tile inner work: load weight tile, scale, contract. Weight load
    // index depends on whether we're in single-tile or multi-tile mode.
    let mut tile_blocks = String::new();
    let mut acc_prev = "%acc_init_n".to_string();
    for tile_idx in 0..k_tiles {
        let k_off = tile_idx * k_step;
        let k_idx = if k_off == 0 {
            "%c0".to_string()
        } else {
            format!("%ck_{tile_idx}")
        };
        let acc_next = format!("%acc_n_{tile_idx}");
        if n_loop {
            tile_blocks.push_str(&format!(
                r#"      %w_i8_3d_{tile_idx} = vector.load %w1_block_t[{k_idx}, %ti, %c0] : memref<{k_total}x{n_tiles}x{n_step}xi8>, vector<{k_step}x1x{n_step}xi8>
      %w_i8_{tile_idx} = vector.shape_cast %w_i8_3d_{tile_idx} : vector<{k_step}x1x{n_step}xi8> to vector<{k_step}x{n_step}xi8>
      %w_f32_{tile_idx} = arith.sitofp %w_i8_{tile_idx} : vector<{k_step}x{n_step}xi8> to vector<{k_step}x{n_step}xf32>
      %w_scaled_{tile_idx} = arith.mulf %w_f32_{tile_idx}, %scale_b : vector<{k_step}x{n_step}xf32>
      {acc_next} = vector.contract #w1_dot_trait %h_f32_{tile_idx}, %w_scaled_{tile_idx}, {acc_prev} : vector<{batch}x{k_step}xf32>, vector<{k_step}x{n_step}xf32> into vector<{batch}x{n_step}xf32>
"#,
                tile_idx = tile_idx,
                k_idx = k_idx,
                k_total = k_total,
                n_tiles = n_tiles,
                n_step = n_step,
                k_step = k_step,
                batch = batch,
                acc_prev = acc_prev,
                acc_next = acc_next,
            ));
        } else {
            tile_blocks.push_str(&format!(
                r#"      %w_i8_{tile_idx} = vector.load %w1_block_t[{k_idx}, %c0] : memref<{k_total}x{n_step}xi8>, vector<{k_step}x{n_step}xi8>
      %w_f32_{tile_idx} = arith.sitofp %w_i8_{tile_idx} : vector<{k_step}x{n_step}xi8> to vector<{k_step}x{n_step}xf32>
      %w_scaled_{tile_idx} = arith.mulf %w_f32_{tile_idx}, %scale_b : vector<{k_step}x{n_step}xf32>
      {acc_next} = vector.contract #w1_dot_trait %h_f32_{tile_idx}, %w_scaled_{tile_idx}, {acc_prev} : vector<{batch}x{k_step}xf32>, vector<{k_step}x{n_step}xf32> into vector<{batch}x{n_step}xf32>
"#,
                tile_idx = tile_idx,
                k_idx = k_idx,
                k_total = k_total,
                n_step = n_step,
                k_step = k_step,
                batch = batch,
                acc_prev = acc_prev,
                acc_next = acc_next,
            ));
        }
        acc_prev = acc_next;
    }

    // Inner work executed once per N tile (loop body when n_loop, top-level
    // otherwise). Loads scale, runs K contractions, truncs+stores W1 result.
    let n_inner = if n_loop {
        format!(
            r#"      %scale_v_3d_n = vector.load %w1_row_scales[%ti, %c0] : memref<{n_tiles}x{n_step}xf32>, vector<1x{n_step}xf32>
      %scale_v_n = vector.shape_cast %scale_v_3d_n : vector<1x{n_step}xf32> to vector<{n_step}xf32>
      %scale_2d_n = vector.shape_cast %scale_v_n : vector<{n_step}xf32> to vector<1x{n_step}xf32>
      %scale_b = vector.broadcast %scale_2d_n : vector<1x{n_step}xf32> to vector<{k_step}x{n_step}xf32>
      %acc_init_n = arith.constant dense<0.000000e+00> : vector<{batch}x{n_step}xf32>
{tile_blocks}      %out_bf16_n = arith.truncf {acc_prev} : vector<{batch}x{n_step}xf32> to vector<{batch}x{n_step}xbf16>
      %n_idx = arith.muli %ti, %c_n_step : index
      vector.store %out_bf16_n, %hidden_out[%c0, %n_idx] : memref<{batch}x{k_total}xbf16>, vector<{batch}x{n_step}xbf16>
"#,
            n_tiles = n_tiles,
            n_step = n_step,
            k_step = k_step,
            batch = batch,
            k_total = k_total,
            tile_blocks = tile_blocks,
            acc_prev = acc_prev,
        )
    } else {
        format!(
            r#"      %scale_v_n = vector.load %w1_row_scales[%c0] : memref<{n_step}xf32>, vector<{n_step}xf32>
      %scale_2d_n = vector.shape_cast %scale_v_n : vector<{n_step}xf32> to vector<1x{n_step}xf32>
      %scale_b = vector.broadcast %scale_2d_n : vector<1x{n_step}xf32> to vector<{k_step}x{n_step}xf32>
      %acc_init_n = arith.constant dense<0.000000e+00> : vector<{batch}x{n_step}xf32>
{tile_blocks}      %out_bf16_n = arith.truncf {acc_prev} : vector<{batch}x{n_step}xf32> to vector<{batch}x{n_step}xbf16>
      vector.store %out_bf16_n, %hidden_out[%c0, %c0] : memref<{batch}x{k_total}xbf16>, vector<{batch}x{n_step}xbf16>
"#,
            n_step = n_step,
            k_step = k_step,
            batch = batch,
            k_total = k_total,
            tile_blocks = tile_blocks,
            acc_prev = acc_prev,
        )
    };

    let n_dispatch = if n_loop {
        let mut body = format!(
            "    %c_n_step = arith.constant {n_step} : index\n    %c_n_tiles = arith.constant {n_tiles} : index\n    %c1 = arith.constant 1 : index\n    scf.for %ti = %c0 to %c_n_tiles step %c1 {{\n",
        );
        body.push_str(&n_inner);
        body.push_str("    }\n");
        body
    } else {
        n_inner
    };

    let probe_label = if n_loop {
        format!("w1_observable_n_loop_bk32x{k_tiles}_n{n_total}_multidim")
    } else {
        format!("w1_observable_unrolled_bk32x{k_tiles}_n{n_total}")
    };

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
      %w1_block_t: {w1_block_ty},
      %w1_row_scales: {w1_row_scales_ty},
      %hidden_out: memref<{batch}x{k_total}xbf16>,
      %kv_out: memref<{kv_bytes}xi8>) attributes {{
        dimension_semantics = [],
        scalar_prefetch = 0 : i64,
        scratch_operands = 0 : i64,
        rvllm.int8_probe = "{probe_label}"
      }} {{
    %c0 = arith.constant 0 : index
{k_constants}{hidden_loads}    %hidden_v = vector.load %hidden[%c0, %c0] : memref<{batch}x{k_total}xbf16>, vector<{batch}x{k_total}xbf16>
    vector.store %hidden_v, %hidden_out[%c0, %c0] : memref<{batch}x{k_total}xbf16>, vector<{batch}x{k_total}xbf16>
    %kv_v = vector.load %kv_in[%c0] : memref<{kv_bytes}xi8>, vector<512xi8>
    vector.store %kv_v, %kv_out[%c0] : memref<{kv_bytes}xi8>, vector<512xi8>
{n_dispatch}    return
  }}
}}
"#,
        batch = batch,
        k_total = k_total,
        kv_bytes = kv_bytes,
        k_constants = k_constants,
        hidden_loads = hidden_loads,
        n_dispatch = n_dispatch,
        probe_label = probe_label,
        w1_block_ty = w1_block_ty,
        w1_row_scales_ty = w1_row_scales_ty,
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
    fn default_single_n_tile_skips_scf_for() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        // No env var = default 128 N cols, no loop, 2D operand contract.
        let mlir = m2_decode_layer_int8_lowered_body_mlir(&shape, 4_294_967_296).unwrap();
        assert!(mlir.contains("rvllm.int8_probe = \"w1_observable_unrolled_bk32x4_n128\""));
        assert!(!mlir.contains("scf.for"));
        assert!(mlir.contains("memref<3072x128xi8>"));
        assert!(mlir.contains("memref<128xf32>"));
        assert!(
            mlir.contains(
                "vector.store %out_bf16_n, %hidden_out[%c0, %c0] : memref<8x3072xbf16>, vector<8x128xbf16>"
            ),
            "single-tile mode stores at static [%c0, %c0]"
        );
    }

    #[test]
    fn emits_int8_lowered_body_full_n_w1_with_n_loop() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        std::env::set_var("RVLLM_M2_W1_N_COLS", "1536");
        let mlir = m2_decode_layer_int8_lowered_body_mlir(&shape, 4_294_967_296).unwrap();
        std::env::remove_var("RVLLM_M2_W1_N_COLS");
        assert!(mlir.contains("rvllm.int8_probe = \"w1_observable_n_loop_bk32x4_n1536_multidim\""));
        // multi-dim operand contract: 3D weight memref + 2D scale memref so
        // Mosaic can prove sublane alignment without per-element offset proofs.
        assert!(mlir.contains("memref<3072x12x128xi8>"));
        assert!(mlir.contains("memref<12x128xf32>"));
        assert!(mlir.contains("memref<33554432xi8>"));
        assert!(mlir.contains("memref<34xi32>"));
        assert!(mlir.contains("memref<256x25xi32>"));
        assert!(mlir.contains("memref<8x3072xbf16>"));
        assert!(
            mlir.contains("scf.for %ti = %c0 to %c_n_tiles step %c1"),
            "must iterate over N-tile index 0..12 with sublane-aligned multi-dim memrefs",
        );
        assert!(
            !mlir.contains("affine.for"),
            "affine.for is unsupported by Mosaic infer-vector-layout"
        );
        assert!(mlir.contains("vector<32x1x128xi8>"));
        assert!(mlir.contains("vector<1x128xf32>"));
        assert!(mlir.contains("arith.muli %ti"));
        assert!(mlir.contains("arith.sitofp"));
        assert!(mlir.contains("arith.truncf"));
        assert_eq!(
            mlir.matches("vector.contract").count(),
            4,
            "expected 4 unrolled K-tile contractions inside N loop"
        );
        assert!(
            mlir.starts_with("#w1_dot_trait"),
            "vector.contract trait alias must precede module per Mosaic contract"
        );
        assert!(
            mlir.contains(
                "vector.store %out_bf16_n, %hidden_out[%c0, %n_idx] : memref<8x3072xbf16>, vector<8x128xbf16>"
            ),
            "must observably store W1 result to hidden_out[:, ti*128:(ti+1)*128]"
        );
        assert!(
            mlir.contains("vector.store %hidden_v, %hidden_out"),
            "must pass through full hidden before overlay"
        );
    }
}
