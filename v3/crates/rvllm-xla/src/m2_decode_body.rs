use rvllm_core::{ConfigError, Result, RvllmError};

use crate::{M2GraphPhase, M2GraphShape, M2_HIDDEN, M2_MOE_INTER};

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
    // Default to a single N=128 tile. Full N=M2_MOE_INTER (1536) requires
    // multi-dim memrefs (graph-side reshape to 3D W1 and 2D scales) so Mosaic
    // can prove sublane alignment of vector<128xf32> loads inside an scf.for
    // over %n; that refactor is tracked separately. Opt in here for
    // experimentation but expect compile failure until the multi-dim path
    // lands.
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
    let n_loop = n_total > n_step;
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

    // K offset constants (compile-time literal indices into %w1_block_t and %hidden).
    let mut k_constants = String::new();
    // Hoisted hidden loads — independent of N loop variable, so we compute the
    // bf16->f32 conversion once and reuse inside the N loop.
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

    // Per-N-tile inner work: load weight tile, scale, contract.
    // %n_idx is whichever SSA value names the N column offset (loop var or %c0).
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
        tile_blocks.push_str(&format!(
            r#"      %w_i8_{tile_idx} = vector.load %w1_block_t[{k_idx}, %n_idx] : memref<{k_total}x{n_total}xi8>, vector<{k_step}x{n_step}xi8>
      %w_f32_{tile_idx} = arith.sitofp %w_i8_{tile_idx} : vector<{k_step}x{n_step}xi8> to vector<{k_step}x{n_step}xf32>
      %w_scaled_{tile_idx} = arith.mulf %w_f32_{tile_idx}, %scale_b : vector<{k_step}x{n_step}xf32>
      {acc_next} = vector.contract #w1_dot_trait %h_f32_{tile_idx}, %w_scaled_{tile_idx}, {acc_prev} : vector<{batch}x{k_step}xf32>, vector<{k_step}x{n_step}xf32> into vector<{batch}x{n_step}xf32>
"#,
            tile_idx = tile_idx,
            k_idx = k_idx,
            k_total = k_total,
            n_total = n_total,
            n_step = n_step,
            k_step = k_step,
            batch = batch,
            acc_prev = acc_prev,
            acc_next = acc_next,
        ));
        acc_prev = acc_next;
    }

    let n_inner = format!(
        r#"      %scale_v_n = vector.load %w1_row_scales[%n_idx] : memref<{n_total}xf32>, vector<{n_step}xf32>
      %scale_2d_n = vector.shape_cast %scale_v_n : vector<{n_step}xf32> to vector<1x{n_step}xf32>
      %scale_b = vector.broadcast %scale_2d_n : vector<1x{n_step}xf32> to vector<{k_step}x{n_step}xf32>
      %acc_init_n = arith.constant dense<0.000000e+00> : vector<{batch}x{n_step}xf32>
{tile_blocks}      %out_bf16_n = arith.truncf {acc_prev} : vector<{batch}x{n_step}xf32> to vector<{batch}x{n_step}xbf16>
      vector.store %out_bf16_n, %hidden_out[%c0, %n_idx] : memref<{batch}x{k_total}xbf16>, vector<{batch}x{n_step}xbf16>
"#,
        n_total = n_total,
        n_step = n_step,
        k_step = k_step,
        batch = batch,
        k_total = k_total,
        tile_blocks = tile_blocks,
        acc_prev = acc_prev,
    );
    let n_dispatch = if n_loop {
        let mut body = format!(
            "    %c_n_step = arith.constant {n_step} : index\n    %c_n_total = arith.constant {n_total} : index\n    scf.for %n_idx = %c0 to %c_n_total step %c_n_step {{\n",
        );
        body.push_str(&n_inner);
        body.push_str("    }\n");
        body
    } else {
        let mut body = String::from("    %n_idx = arith.constant 0 : index\n");
        body.push_str(&n_inner);
        body
    };

    let probe_label = if n_loop {
        format!("w1_observable_n_loop_bk32x{k_tiles}_n{n_total}")
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
      %w1_block_t: memref<{k_total}x{n_total}xi8>,
      %w1_row_scales: memref<{n_total}xf32>,
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
        n_total = n_total,
        kv_bytes = kv_bytes,
        k_constants = k_constants,
        hidden_loads = hidden_loads,
        n_dispatch = n_dispatch,
        probe_label = probe_label,
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
    fn emits_int8_lowered_body_full_n_w1_with_n_loop() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        std::env::set_var("RVLLM_M2_W1_N_COLS", "1536");
        let mlir = m2_decode_layer_int8_lowered_body_mlir(&shape, 4_294_967_296).unwrap();
        std::env::remove_var("RVLLM_M2_W1_N_COLS");
        assert!(mlir.contains("rvllm.int8_probe = \"w1_observable_n_loop_bk32x4_n1536\""));
        // operand contract widens to full M2_MOE_INTER N dim
        assert!(mlir.contains("memref<3072x1536xi8>"));
        assert!(mlir.contains("memref<1536xf32>"));
        assert!(mlir.contains("memref<33554432xi8>"));
        assert!(mlir.contains("memref<34xi32>"));
        assert!(mlir.contains("memref<256x25xi32>"));
        assert!(mlir.contains("memref<8x3072xbf16>"));
        assert!(
            mlir.contains("scf.for %n_idx = %c0 to %c_n_total step %c_n_step"),
            "must iterate over N tiles in stride 128 to fit Mosaic alignment",
        );
        assert!(
            !mlir.contains("affine.for"),
            "affine.for is unsupported by Mosaic infer-vector-layout"
        );
        assert!(mlir.contains("vector<32x128xi8>"));
        assert!(mlir.contains("vector<8x32xbf16>"));
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
            "must observably store W1 result to hidden_out[:, %n_idx:%n_idx+128]"
        );
        assert!(
            mlir.contains("vector.store %hidden_v, %hidden_out"),
            "must pass through full hidden before overlay"
        );
    }

    #[test]
    fn default_single_n_tile_skips_scf_for() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        // No env var = default 128 N cols, no loop.
        let mlir = m2_decode_layer_int8_lowered_body_mlir(&shape, 4_294_967_296).unwrap();
        assert!(mlir.contains("rvllm.int8_probe = \"w1_observable_unrolled_bk32x4_n128\""));
        assert!(!mlir.contains("scf.for"));
        assert!(mlir.contains("memref<3072x128xi8>"));
        assert!(mlir.contains("memref<128xf32>"));
    }
}
