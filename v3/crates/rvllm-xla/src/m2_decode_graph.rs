use std::collections::BTreeMap;

use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_loader::M2Projection;

use crate::m2_graph_abi::{split_i64_row, M2DecodeLayerArenaOffsets, M2DecodeLayerCustomCallAbi};
use crate::m2_tpu_custom_call::{
    tpu_custom_call_backend_config, tpu_custom_call_backend_config_for_body,
    TpuMosaicSerializedBody, TPU_CUSTOM_CALL_TARGET,
};
use crate::{
    M2GraphPhase, M2GraphShape, M2Nvfp4ProjectionAbi, M2WeightArenaEntry, M2WeightArenaPlan,
    M2WeightRole, PjrtElementType, M2_HEAD_DIM, M2_HIDDEN, M2_NUM_EXPERTS, M2_NUM_KV_HEADS,
    M2_NUM_LAYERS, M2_NUM_Q_HEADS, M2_ROTARY_DIM, M2_VOCAB,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2ArenaTensor {
    pub name: String,
    pub offset: usize,
    pub nbytes: usize,
    pub shape: Vec<i64>,
    pub dtype: PjrtElementType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2Nvfp4ProjectionPlan {
    pub projection: M2Projection,
    pub rows: usize,
    pub cols: usize,
    pub packed: M2ArenaTensor,
    pub scale: M2ArenaTensor,
    pub global_scale: M2ArenaTensor,
    pub input_scale: Option<M2ArenaTensor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2ExpertPlan {
    pub expert: usize,
    pub w1: M2Nvfp4ProjectionPlan,
    pub w2: M2Nvfp4ProjectionPlan,
    pub w3: M2Nvfp4ProjectionPlan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2ExpertDirectoryEntry {
    pub expert: usize,
    pub w1_packed_offset: i64,
    pub w1_scale_offset: i64,
    pub w1_global_scale_offset: i64,
    pub w1_input_scale_offset: i64,
    pub w2_packed_offset: i64,
    pub w2_scale_offset: i64,
    pub w2_global_scale_offset: i64,
    pub w2_input_scale_offset: i64,
    pub w3_packed_offset: i64,
    pub w3_scale_offset: i64,
    pub w3_global_scale_offset: i64,
    pub w3_input_scale_offset: i64,
}

impl M2ExpertDirectoryEntry {
    pub const COLS: usize = 13;
    pub const I32_COLS: usize = 1 + (Self::COLS - 1) * 2;

    pub fn as_i64_row(&self) -> [i64; Self::COLS] {
        [
            self.expert as i64,
            self.w1_packed_offset,
            self.w1_scale_offset,
            self.w1_global_scale_offset,
            self.w1_input_scale_offset,
            self.w2_packed_offset,
            self.w2_scale_offset,
            self.w2_global_scale_offset,
            self.w2_input_scale_offset,
            self.w3_packed_offset,
            self.w3_scale_offset,
            self.w3_global_scale_offset,
            self.w3_input_scale_offset,
        ]
    }

    pub fn as_i32_split_row(&self) -> [i32; Self::I32_COLS] {
        let offsets = split_i64_row::<12, 24>(&[
            self.w1_packed_offset,
            self.w1_scale_offset,
            self.w1_global_scale_offset,
            self.w1_input_scale_offset,
            self.w2_packed_offset,
            self.w2_scale_offset,
            self.w2_global_scale_offset,
            self.w2_input_scale_offset,
            self.w3_packed_offset,
            self.w3_scale_offset,
            self.w3_global_scale_offset,
            self.w3_input_scale_offset,
        ]);
        let mut row = [0; Self::I32_COLS];
        row[0] = self.expert as i32;
        row[1..].copy_from_slice(&offsets);
        row
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2DecodeLayerPlan {
    pub layer: usize,
    pub input_norm: M2ArenaTensor,
    pub post_attention_norm: M2ArenaTensor,
    pub q_proj: M2ArenaTensor,
    pub k_proj: M2ArenaTensor,
    pub v_proj: M2ArenaTensor,
    pub o_proj: M2ArenaTensor,
    pub q_norm: M2ArenaTensor,
    pub k_norm: M2ArenaTensor,
    pub router: M2ArenaTensor,
    pub router_bias: M2ArenaTensor,
    pub experts: Vec<M2ExpertPlan>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2DecodeGraphPlan {
    pub embed: M2ArenaTensor,
    pub final_norm: M2ArenaTensor,
    pub lm_head: M2ArenaTensor,
    pub layers: Vec<M2DecodeLayerPlan>,
}

impl M2DecodeGraphPlan {
    pub fn from_arena(arena: &M2WeightArenaPlan) -> Result<Self> {
        let lookup = ArenaLookup::new(arena);
        Ok(Self {
            embed: arena_tensor(lookup.entry("model.embed_tokens.weight")?),
            final_norm: arena_tensor(lookup.entry("model.norm.weight")?),
            lm_head: arena_tensor(lookup.entry("lm_head.weight")?),
            layers: (0..M2_NUM_LAYERS)
                .map(|layer| M2DecodeLayerPlan::from_lookup(layer, &lookup))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    pub fn input_scale_missing_count(&self) -> usize {
        self.layers
            .iter()
            .map(M2DecodeLayerPlan::input_scale_missing_count)
            .sum()
    }
}

impl M2DecodeLayerPlan {
    fn from_lookup(layer: usize, lookup: &ArenaLookup<'_>) -> Result<Self> {
        Ok(Self {
            layer,
            input_norm: dense(lookup, layer, "input_layernorm.weight")?,
            post_attention_norm: dense(lookup, layer, "post_attention_layernorm.weight")?,
            q_proj: dense(lookup, layer, "self_attn.q_proj.weight")?,
            k_proj: dense(lookup, layer, "self_attn.k_proj.weight")?,
            v_proj: dense(lookup, layer, "self_attn.v_proj.weight")?,
            o_proj: dense(lookup, layer, "self_attn.o_proj.weight")?,
            q_norm: dense(lookup, layer, "self_attn.q_norm.weight")?,
            k_norm: dense(lookup, layer, "self_attn.k_norm.weight")?,
            router: dense(lookup, layer, "block_sparse_moe.gate.weight")?,
            router_bias: dense(lookup, layer, "block_sparse_moe.e_score_correction_bias")?,
            experts: (0..M2_NUM_EXPERTS)
                .map(|expert| M2ExpertPlan::from_lookup(layer, expert, lookup))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    pub fn input_scale_missing_count(&self) -> usize {
        self.experts
            .iter()
            .map(|expert| {
                usize::from(expert.w1.input_scale.is_none())
                    + usize::from(expert.w2.input_scale.is_none())
                    + usize::from(expert.w3.input_scale.is_none())
            })
            .sum()
    }

    pub fn expert_directory(&self) -> Vec<M2ExpertDirectoryEntry> {
        self.experts
            .iter()
            .map(M2ExpertPlan::directory_entry)
            .collect()
    }

    pub fn arena_offsets(&self) -> M2DecodeLayerArenaOffsets {
        let first = &self.experts[0];
        let last = self.experts.last().expect("M2 layer has experts");
        M2DecodeLayerArenaOffsets {
            input_norm: self.input_norm.offset as i64,
            post_attention_norm: self.post_attention_norm.offset as i64,
            q_proj: self.q_proj.offset as i64,
            k_proj: self.k_proj.offset as i64,
            v_proj: self.v_proj.offset as i64,
            o_proj: self.o_proj.offset as i64,
            q_norm: self.q_norm.offset as i64,
            k_norm: self.k_norm.offset as i64,
            router: self.router.offset as i64,
            router_bias: self.router_bias.offset as i64,
            w1_first_packed: first.w1.packed.offset as i64,
            w1_first_scale: first.w1.scale.offset as i64,
            w1_first_global_scale: first.w1.global_scale.offset as i64,
            w1_first_input_scale: input_scale_offset(&first.w1),
            w2_first_packed: first.w2.packed.offset as i64,
            w3_first_packed: first.w3.packed.offset as i64,
            w3_last_packed: last.w3.packed.offset as i64,
        }
    }

    pub fn custom_call_abi(&self, shape: &M2GraphShape) -> Result<M2DecodeLayerCustomCallAbi> {
        M2DecodeLayerCustomCallAbi::new(
            shape,
            self.experts.len(),
            M2ExpertDirectoryEntry::COLS,
            self.arena_offsets(),
        )
    }
}

impl M2ExpertPlan {
    fn from_lookup(layer: usize, expert: usize, lookup: &ArenaLookup<'_>) -> Result<Self> {
        Ok(Self {
            expert,
            w1: nvfp4_projection(lookup, layer, expert, M2Projection::W1)?,
            w2: nvfp4_projection(lookup, layer, expert, M2Projection::W2)?,
            w3: nvfp4_projection(lookup, layer, expert, M2Projection::W3)?,
        })
    }

    fn directory_entry(&self) -> M2ExpertDirectoryEntry {
        M2ExpertDirectoryEntry {
            expert: self.expert,
            w1_packed_offset: self.w1.packed.offset as i64,
            w1_scale_offset: self.w1.scale.offset as i64,
            w1_global_scale_offset: self.w1.global_scale.offset as i64,
            w1_input_scale_offset: input_scale_offset(&self.w1),
            w2_packed_offset: self.w2.packed.offset as i64,
            w2_scale_offset: self.w2.scale.offset as i64,
            w2_global_scale_offset: self.w2.global_scale.offset as i64,
            w2_input_scale_offset: input_scale_offset(&self.w2),
            w3_packed_offset: self.w3.packed.offset as i64,
            w3_scale_offset: self.w3.scale.offset as i64,
            w3_global_scale_offset: self.w3.global_scale.offset as i64,
            w3_input_scale_offset: input_scale_offset(&self.w3),
        }
    }
}

pub fn m2_decode_graph_mlir(
    kernel_name: &str,
    shape: &M2GraphShape,
    arena: &M2WeightArenaPlan,
) -> Result<String> {
    m2_decode_graph_mlir_with_mosaic_body(kernel_name, shape, arena, None)
}

pub fn m2_decode_graph_mlir_with_mosaic_body(
    kernel_name: &str,
    shape: &M2GraphShape,
    arena: &M2WeightArenaPlan,
    decode_layer_body: Option<&TpuMosaicSerializedBody>,
) -> Result<String> {
    shape.validate()?;
    if shape.phase != M2GraphPhase::Decode {
        return Err(invalid("phase", "expected decode graph shape"));
    }
    if !is_mlir_symbol(kernel_name) {
        return Err(invalid("kernel_name", "must be an MLIR symbol"));
    }
    let plan = M2DecodeGraphPlan::from_arena(arena)?;
    let dense_bytes_per_layer = dense_weight_bytes_per_layer();
    let dense_arena_bytes = dense_bytes_per_layer * M2_NUM_LAYERS;
    let dense_arena_bf16 = dense_arena_bytes / 2;
    let body = emit_decode_body(shape, arena, &plan, decode_layer_body, dense_arena_bytes)?;
    let stablehlo_layer =
        std::env::var("RVLLM_M2_LAYER_MODE").unwrap_or_default() != "mosaic"
            && std::env::var("RVLLM_M2_LAYER_MODE").is_ok();
    let weight_arena_local_bytes = if stablehlo_layer {
        1
    } else {
        arena.total_bytes.div_ceil(8)
    };
    let int8_row_scale_arena_len = M2_NUM_LAYERS * crate::m2_int8_w1_n_total();
    Ok(format!(
        r###"module attributes {{mhlo.frontend_attributes = {{xla.sdy.meshes = "{{mesh = #sdy.mesh<[\22expert\22=8]>}}"}}, mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, rvllm.kind = "m2_decode_graph"}} {{
  func.func @{kernel_name}(
      %token_ids: tensor<{batch}xi32>,
      %positions: tensor<{batch}xi32>,
      %kv_cache: tensor<{kv_bytes}xi8>,
      %weight_arena: tensor<{weight_arena_local_bytes}xi8>,
      %int8_row_scales: tensor<{int8_row_scale_arena_len}xf32>,
      %input_hidden: tensor<{batch}x{hidden}xbf16>,
      %final_norm: tensor<{hidden}xbf16>,
      %lm_head: tensor<{vocab}x{hidden}xbf16>,
      %dense_weights: tensor<{dense_arena_bf16}xbf16>)
      -> (tensor<{batch}x{vocab}xbf16>, tensor<{batch}xi32>, tensor<{kv_bytes}xi8>)
      attributes {{
        rvllm.signature = "token_ids,positions,kv_cache,weight_arena,int8_row_scales,input_hidden,final_norm,lm_head,dense_weights -> logits,next_token,kv_cache",
        rvllm.dense_arena_bytes = {dense_arena_bytes} : i64,
        rvllm.dense_arena_bf16 = {dense_arena_bf16} : i64,
        rvllm.phase = "decode",
        rvllm.batch = {batch} : i64,
        rvllm.ctx = {ctx} : i64,
        rvllm.layers = {layers} : i64,
        rvllm.hidden = {hidden} : i64,
        rvllm.vocab = {vocab} : i64,
        rvllm.kv_cache_bytes = {kv_bytes} : i64,
        rvllm.weight_arena_bytes = {weight_arena_local_bytes} : i64,
        rvllm.weight_arena_total_bytes = {weight_bytes} : i64,
        rvllm.int8_row_scale_arena_len = {int8_row_scale_arena_len} : i64,
        rvllm.weight_entries = {weight_entries} : i64,
        rvllm.weight_alignment = {weight_alignment} : i64,
        rvllm.weight_input_scales_missing = {missing_input_scales} : i64,
        rvllm.weight_metadata = "compile_time_offsets_from_M2WeightArenaPlan",
        rvllm.lowering = "rust_xla_custom_call",
        rvllm.lowering_plan = "host BF16 embedding gather -> native StableHLO final logits -> 62 fused decode-layer custom calls -> flat-arena dense loads -> fused attention -> top-k router -> flat-arena NVFP4 experts"
      }} {{
{body}
  }}
}}
"###,
        kernel_name = kernel_name,
        batch = shape.batch,
        ctx = shape.ctx,
        layers = M2_NUM_LAYERS,
        hidden = M2_HIDDEN,
        vocab = M2_VOCAB,
        kv_bytes = shape.kv_cache_bytes(),
        weight_bytes = arena.total_bytes,
        weight_arena_local_bytes = weight_arena_local_bytes,
        int8_row_scale_arena_len = int8_row_scale_arena_len,
        weight_entries = arena.entries.len(),
        weight_alignment = arena.alignment,
        missing_input_scales = plan.input_scale_missing_count(),
        body = body,
    ))
}

pub fn m2_decode_smoke_mlir(kernel_name: &str, shape: &M2GraphShape) -> Result<String> {
    shape.validate()?;
    if shape.phase != M2GraphPhase::Decode {
        return Err(invalid("phase", "expected decode graph shape"));
    }
    if !is_mlir_symbol(kernel_name) {
        return Err(invalid("kernel_name", "must be an MLIR symbol"));
    }
    Ok(format!(
        r#"module @rvllm_m2_decode_smoke attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, rvllm.kind = "m2_decode_smoke"}} {{
  func.func public @{kernel_name}(
      %token_ids: tensor<{batch}xi32>,
      %positions: tensor<{batch}xi32>,
      %kv_cache: tensor<{kv_bytes}xi8>,
      %weight_arena: tensor<1xi8>)
      -> (tensor<{batch}x{vocab}xbf16> {{jax.result_info = "logits"}},
          tensor<{batch}xi32> {{jax.result_info = "next_token"}},
          tensor<{kv_bytes}xi8> {{jax.result_info = "kv_cache"}}) {{
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %logits = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<bf16>) -> tensor<{batch}x{vocab}xbf16>
    %zero_i32 = stablehlo.constant dense<0> : tensor<i32>
    %zero_vec = stablehlo.broadcast_in_dim %zero_i32, dims = [] : (tensor<i32>) -> tensor<{batch}xi32>
    %positions_zero = stablehlo.multiply %positions, %zero_vec : tensor<{batch}xi32>
    %next_token = stablehlo.add %token_ids, %positions_zero : tensor<{batch}xi32>
    return %logits, %next_token, %kv_cache : tensor<{batch}x{vocab}xbf16>, tensor<{batch}xi32>, tensor<{kv_bytes}xi8>
  }}
}}
"#,
        kernel_name = kernel_name,
        batch = shape.batch,
        vocab = M2_VOCAB,
        kv_bytes = shape.kv_cache_bytes(),
    ))
}

fn emit_decode_body(
    shape: &M2GraphShape,
    arena: &M2WeightArenaPlan,
    plan: &M2DecodeGraphPlan,
    decode_layer_body: Option<&TpuMosaicSerializedBody>,
    dense_arena_bytes: usize,
) -> Result<String> {
    let hidden_ty = format!("tensor<{}x{}xbf16>", shape.batch, M2_HIDDEN);
    let kv_ty = format!("tensor<{}xi8>", shape.kv_cache_bytes());
    let layer_kv_ty = format!("tensor<{}xi8>", shape.layer_kv_cache_bytes());
    let token_ty = format!("tensor<{}xi32>", shape.batch);
    let logits_ty = format!("tensor<{}x{}xbf16>", shape.batch, M2_VOCAB);
    let mut out = String::new();
    let uses_int8_tile_probe = decode_layer_body.is_some()
        && arena
            .entries
            .iter()
            .any(|entry| entry.role == M2WeightRole::Int8Weight);

    let mut hidden = "%input_hidden".to_string();
    let mut kv = "%kv_cache".to_string();

    // Layer body mode: "mosaic" (default, single tpu_custom_call per layer with
    // a Mosaic-typed body) or "stablehlo" (no custom_call, pure StableHLO ops
    // inline). The stablehlo path is the route to a real PPL because it sidesteps
    // Mosaic body validation; cost is no Mosaic-specific perf optimization.
    // Stages within stablehlo mode:
    //   identity      -- pass through hidden + kv unchanged (Phase 1)
    //   rmsnorm       -- + RMSNorm input + residual stub (Phase 2, future)
    //   attention     -- + full attention (Phase 3, future)
    //   moe           -- + MoE router and experts (Phase 4, future)
    //   full          -- complete M2 layer (target)
    let layer_mode = std::env::var("RVLLM_M2_LAYER_MODE").unwrap_or_else(|_| "mosaic".to_string());
    let stablehlo_mode = layer_mode != "mosaic";

    // W1 N tile per layer custom_call. 128 is the proven single-tile path.
    // M2_MOE_INTER (1536, full N) requires multi-dim operand types so Mosaic
    // can prove sublane alignment of scale loads inside scf.for over %ti.
    let w1_block_cols = crate::m2_int8_w1_n_total();
    let n_step = 128usize;
    if w1_block_cols % n_step != 0 {
        return Err(invalid(
            "w1_block_cols",
            "RVLLM_M2_W1_N_COLS must be a multiple of 128",
        ));
    }
    let n_tiles = w1_block_cols / n_step;
    let n_loop = n_tiles > 1;
    let int8_tile_elems = M2_HIDDEN * w1_block_cols;
    for layer in &plan.layers {
        let next_hidden = format!("%h_l{}", layer.layer);
        let layer_kv = format!("%kv_layer_{}", layer.layer);
        let next_layer_kv = format!("%kv_layer_out_{}", layer.layer);
        let next_kv = format!("%kv_after_l{}", layer.layer);
        let layer_kv_offset = shape.layer_kv_cache_offset(layer.layer);
        let layer_kv_start = format!("%kv_layer_start_{}", layer.layer);

        if stablehlo_mode {
            let (new_hidden, new_kv) = emit_stablehlo_layer(
                &mut out,
                shape,
                layer,
                dense_arena_bytes,
                &hidden,
                &kv,
            )?;
            hidden = new_hidden;
            kv = new_kv;
            continue;
        }
        out.push_str(&format!(
            r#"    {layer_kv_start} = stablehlo.constant dense<{layer_kv_offset}> : tensor<i32>
    {layer_kv} = "stablehlo.dynamic_slice"({kv}, {layer_kv_start}) {{
      slice_sizes = array<i64: {layer_kv_bytes}>
    }} : ({kv_ty}, tensor<i32>) -> {layer_kv_ty}
"#,
            layer_kv_start = layer_kv_start,
            layer_kv_offset = layer_kv_offset,
            layer_kv = layer_kv,
            kv = kv,
            layer_kv_bytes = shape.layer_kv_cache_bytes(),
            kv_ty = kv_ty,
            layer_kv_ty = layer_kv_ty,
        ));
        let (int8_tile_name, int8_row_scale_name) = if uses_int8_tile_probe {
            let w1_tile_start = format!("%w1_int8_tile_start_{}", layer.layer);
            let w1_tile_flat = format!("%w1_int8_tile_flat_{}", layer.layer);
            let w1_tile_rows = format!("%w1_int8_tile_rows_{}", layer.layer);
            let w1_tile_3d = format!("%w1_int8_tile_3d_{}", layer.layer);
            let w1_tile = format!("%w1_int8_tile_t_{}", layer.layer);
            let w1_scale_start = format!("%w1_int8_row_scale_start_{}", layer.layer);
            let w1_scales_flat = format!("%w1_int8_row_scales_flat_{}", layer.layer);
            let w1_scales = format!("%w1_int8_row_scales_{}", layer.layer);
            // Slice + reshape + transpose to produce the body's expected shape.
            // Single-tile (n_loop=false): tensor<{K}x{n_total}xi8> + tensor<{n_total}xf32>
            // Multi-tile (n_loop=true):  tensor<{K}x{n_tiles}x128xi8> + tensor<{n_tiles}x128xf32>
            out.push_str(&format!(
                r#"    {w1_tile_start} = stablehlo.constant dense<{w1_offset}> : tensor<i64>
    {w1_tile_flat} = "stablehlo.dynamic_slice"(%weight_arena, {w1_tile_start}) {{
      slice_sizes = array<i64: {int8_tile_elems}>
    }} : (tensor<{weight_arena_local_bytes}xi8>, tensor<i64>) -> tensor<{int8_tile_elems}xi8>
    {w1_tile_rows} = stablehlo.reshape {w1_tile_flat} : (tensor<{int8_tile_elems}xi8>) -> tensor<{w1_block_cols}x{hidden_size}xi8>
"#,
                w1_tile_start = w1_tile_start,
                w1_tile_flat = w1_tile_flat,
                w1_tile_rows = w1_tile_rows,
                w1_offset = layer.arena_offsets().w1_first_packed,
                int8_tile_elems = int8_tile_elems,
                weight_arena_local_bytes = arena.total_bytes.div_ceil(8),
                hidden_size = M2_HIDDEN,
                w1_block_cols = w1_block_cols,
            ));
            if n_loop {
                out.push_str(&format!(
                    r#"    {w1_tile_3d} = stablehlo.reshape {w1_tile_rows} : (tensor<{w1_block_cols}x{hidden_size}xi8>) -> tensor<{n_tiles}x{n_step}x{hidden_size}xi8>
    {w1_tile} = stablehlo.transpose {w1_tile_3d}, dims = [2, 0, 1] : (tensor<{n_tiles}x{n_step}x{hidden_size}xi8>) -> tensor<{hidden_size}x{n_tiles}x{n_step}xi8>
"#,
                    w1_tile_3d = w1_tile_3d,
                    w1_tile_rows = w1_tile_rows,
                    w1_tile = w1_tile,
                    w1_block_cols = w1_block_cols,
                    hidden_size = M2_HIDDEN,
                    n_tiles = n_tiles,
                    n_step = n_step,
                ));
            } else {
                out.push_str(&format!(
                    r#"    {w1_tile} = stablehlo.transpose {w1_tile_rows}, dims = [1, 0] : (tensor<{w1_block_cols}x{hidden_size}xi8>) -> tensor<{hidden_size}x{w1_block_cols}xi8>
"#,
                    w1_tile = w1_tile,
                    w1_tile_rows = w1_tile_rows,
                    w1_block_cols = w1_block_cols,
                    hidden_size = M2_HIDDEN,
                ));
            }
            // Scales: 1D slice -> optionally reshape to 2D for n_loop.
            let scales_flat_var = if n_loop {
                w1_scales_flat.clone()
            } else {
                w1_scales.clone()
            };
            out.push_str(&format!(
                r#"    {w1_scale_start} = stablehlo.constant dense<{w1_scale_offset}> : tensor<i64>
    {scales_flat_var} = "stablehlo.dynamic_slice"(%int8_row_scales, {w1_scale_start}) {{
      slice_sizes = array<i64: {w1_block_cols}>
    }} : (tensor<{int8_row_scale_arena_len}xf32>, tensor<i64>) -> tensor<{w1_block_cols}xf32>
"#,
                w1_scale_start = w1_scale_start,
                scales_flat_var = scales_flat_var,
                w1_scale_offset = layer.layer * w1_block_cols,
                int8_row_scale_arena_len = M2_NUM_LAYERS * w1_block_cols,
                w1_block_cols = w1_block_cols,
            ));
            if n_loop {
                out.push_str(&format!(
                    r#"    {w1_scales} = stablehlo.reshape {w1_scales_flat} : (tensor<{w1_block_cols}xf32>) -> tensor<{n_tiles}x{n_step}xf32>
"#,
                    w1_scales = w1_scales,
                    w1_scales_flat = w1_scales_flat,
                    w1_block_cols = w1_block_cols,
                    n_tiles = n_tiles,
                    n_step = n_step,
                ));
            }
            (Some(w1_tile), Some(w1_scales))
        } else {
            (None, None)
        };
        out.push_str(&emit_layer_call(
            shape,
            layer,
            &hidden,
            &layer_kv,
            &next_hidden,
            &next_layer_kv,
            &hidden_ty,
            &token_ty,
            &layer_kv_ty,
            decode_layer_body,
            int8_tile_name.as_deref(),
            int8_row_scale_name.as_deref(),
        )?);
        out.push_str(&format!(
            r#"    {next_kv} = "stablehlo.dynamic_update_slice"({kv}, {next_layer_kv}, {layer_kv_start}) : ({kv_ty}, {layer_kv_ty}, tensor<i32>) -> {kv_ty}
"#,
            next_kv = next_kv,
            kv = kv,
            next_layer_kv = next_layer_kv,
            layer_kv_start = layer_kv_start,
            kv_ty = kv_ty,
            layer_kv_ty = layer_kv_ty,
        ));
        hidden = next_hidden;
        kv = next_kv;
    }

    out.push_str(&format!(
        r#"    %final_norm_b = stablehlo.broadcast_in_dim %final_norm, dims = [1] : (tensor<{hidden_size}xbf16>) -> {hidden_ty}
    %h_final = stablehlo.multiply {hidden}, %final_norm_b : {hidden_ty}
    %lm_head_t = stablehlo.transpose %lm_head, dims = [1, 0] : (tensor<{vocab}x{hidden_size}xbf16>) -> tensor<{hidden_size}x{vocab}xbf16>
    %logits = "stablehlo.dot_general"(%h_final, %lm_head_t) {{
      dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
    }} : ({hidden_ty}, tensor<{hidden_size}x{vocab}xbf16>) -> {logits_ty}
    %next_token = stablehlo.add %token_ids, %positions : {token_ty}
    func.return %logits, %next_token, {final_kv} : {logits_ty}, {token_ty}, {kv_ty}
"#,
        hidden = hidden,
        hidden_size = M2_HIDDEN,
        vocab = M2_VOCAB,
        final_kv = kv,
    ));
    Ok(out)
}

fn dense_weight_bytes_per_layer() -> usize {
    let h = M2_HIDDEN;
    let qd = M2_NUM_Q_HEADS * M2_HEAD_DIM;
    let kvd = M2_NUM_KV_HEADS * M2_HEAD_DIM;
    (h + h + qd * h + kvd * h + kvd * h + h * qd + qd + kvd + 256 * h + 256) * 2
}

fn dense_layer_field_offset(field_index: usize) -> usize {
    let h = M2_HIDDEN;
    let qd = M2_NUM_Q_HEADS * M2_HEAD_DIM;
    let kvd = M2_NUM_KV_HEADS * M2_HEAD_DIM;
    let sizes = [
        h,         // 0: input_norm
        h,         // 1: post_attention_norm
        qd * h,    // 2: q_proj
        kvd * h,   // 3: k_proj
        kvd * h,   // 4: v_proj
        h * qd,    // 5: o_proj
        qd,        // 6: q_norm
        kvd,       // 7: k_norm
        256 * h,   // 8: router
        256,       // 9: router_bias
    ];
    sizes[..field_index].iter().sum::<usize>() * 2
}

fn emit_dense_bf16(
    out: &mut String,
    dense_arena_bytes: usize,
    layer: usize,
    field_index: usize,
    prefix: &str,
    shape: &[usize],
) -> String {
    let per_layer = dense_weight_bytes_per_layer();
    let byte_offset = layer * per_layer + dense_layer_field_offset(field_index);
    let n_bf16: usize = shape.iter().product();
    let bf16_offset = byte_offset / 2;
    let bf16_end = bf16_offset + n_bf16;
    let dense_bf16_total = dense_arena_bytes / 2;
    let flat_name = format!("%{}_flat", prefix);
    let result_name = format!("%{}", prefix);
    // Dense weights are uploaded as bf16 -- just slice and reshape, no bitcast
    out.push_str(&format!(
        "    {flat_name} = stablehlo.slice %dense_weights [{bf16_offset}:{bf16_end}] : (tensor<{dense_bf16_total}xbf16>) -> tensor<{n_bf16}xbf16>\n",
    ));
    if shape.len() == 1 {
        out.push_str(&format!(
            "    {result_name} = stablehlo.optimization_barrier {flat_name} : tensor<{n_bf16}xbf16>\n",
        ));
    } else {
        let shape_str = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
        out.push_str(&format!(
            "    {result_name} = stablehlo.reshape {flat_name} : (tensor<{n_bf16}xbf16>) -> tensor<{shape_str}xbf16>\n",
        ));
    }
    result_name
}

fn emit_arena_bf16(
    out: &mut String,
    arena_bytes: usize,
    prefix: &str,
    offset: i64,
    shape: &[usize],
) -> String {
    let nbytes: usize = shape.iter().product::<usize>() * 2;
    let n_bf16: usize = shape.iter().product();
    let offset_name = format!("%{}_off", prefix);
    let raw_name = format!("%{}_raw", prefix);
    let raw2d_name = format!("%{}_raw2d", prefix);
    let flat_name = format!("%{}_flat", prefix);
    let result_name = format!("%{}", prefix);
    // bitcast_convert requires src rank = dst rank + 1 when element sizes differ.
    // Reshape i8 [nbytes] -> [n_bf16, 2] then bitcast to [n_bf16] bf16.
    out.push_str(&format!(
        "    {offset_name} = stablehlo.constant dense<{offset}> : tensor<i64>\n\
         \x20\x20\x20\x20{raw_name} = \"stablehlo.dynamic_slice\"(%weight_arena, {offset_name}) {{\n\
         \x20\x20\x20\x20\x20\x20slice_sizes = array<i64: {nbytes}>\n\
         \x20\x20\x20\x20}} : (tensor<{arena_bytes}xi8>, tensor<i64>) -> tensor<{nbytes}xi8>\n\
         \x20\x20\x20\x20{raw2d_name} = stablehlo.reshape {raw_name} : (tensor<{nbytes}xi8>) -> tensor<{n_bf16}x2xi8>\n\
         \x20\x20\x20\x20{flat_name} = stablehlo.bitcast_convert {raw2d_name} : (tensor<{n_bf16}x2xi8>) -> tensor<{n_bf16}xbf16>\n",
    ));
    if shape.len() == 1 {
        out.push_str(&format!(
            "    {result_name} = stablehlo.optimization_barrier {flat_name} : tensor<{n_bf16}xbf16>\n",
        ));
    } else {
        let shape_str = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
        out.push_str(&format!(
            "    {result_name} = stablehlo.reshape {flat_name} : (tensor<{n_bf16}xbf16>) -> tensor<{shape_str}xbf16>\n",
        ));
    }
    result_name
}

fn emit_rmsnorm(
    out: &mut String,
    input: &str,
    weight: &str,
    batch: usize,
    hidden: usize,
    prefix: &str,
) -> String {
    let h_ty = format!("tensor<{}x{}xbf16>", batch, hidden);
    let sq = format!("%{}_sq", prefix);
    let sum = format!("%{}_sum", prefix);
    let mean = format!("%{}_mean", prefix);
    let eps = format!("%{}_eps", prefix);
    let me = format!("%{}_me", prefix);
    let rs = format!("%{}_rs", prefix);
    let rs_b = format!("%{}_rs_b", prefix);
    let normed = format!("%{}_normed", prefix);
    let w_b = format!("%{}_w_b", prefix);
    let result = format!("%{}_out", prefix);
    let h_f = hidden as f64;
    let init = format!("%{}_init", prefix);
    out.push_str(&format!(
        "    {sq} = stablehlo.multiply {input}, {input} : {h_ty}\n\
         \x20\x20\x20\x20{init} = stablehlo.constant dense<0.0> : tensor<bf16>\n\
         \x20\x20\x20\x20{sum} = \"stablehlo.reduce\"({sq}, {init}) ({{\n\
         \x20\x20\x20\x20    ^bb0(%a_{prefix}: tensor<bf16>, %b_{prefix}: tensor<bf16>):\n\
         \x20\x20\x20\x20      %s_{prefix} = stablehlo.add %a_{prefix}, %b_{prefix} : tensor<bf16>\n\
         \x20\x20\x20\x20      stablehlo.return %s_{prefix} : tensor<bf16>\n\
         \x20\x20\x20\x20    }}) {{dimensions = array<i64: 1>}} : ({h_ty}, tensor<bf16>) -> tensor<{batch}xbf16>\n\
         \x20\x20\x20\x20%eps_s_{prefix} = stablehlo.constant dense<{eps_val}> : tensor<bf16>\n\
         \x20\x20\x20\x20{eps} = stablehlo.broadcast_in_dim %eps_s_{prefix}, dims = [] : (tensor<bf16>) -> tensor<{batch}xbf16>\n\
         \x20\x20\x20\x20%h_inv_s_{prefix} = stablehlo.constant dense<{h_inv}> : tensor<bf16>\n\
         \x20\x20\x20\x20%h_inv_{prefix} = stablehlo.broadcast_in_dim %h_inv_s_{prefix}, dims = [] : (tensor<bf16>) -> tensor<{batch}xbf16>\n\
         \x20\x20\x20\x20{mean} = stablehlo.multiply {sum}, %h_inv_{prefix} : tensor<{batch}xbf16>\n\
         \x20\x20\x20\x20{me} = stablehlo.add {mean}, {eps} : tensor<{batch}xbf16>\n\
         \x20\x20\x20\x20{rs} = stablehlo.rsqrt {me} : tensor<{batch}xbf16>\n\
         \x20\x20\x20\x20{rs_b} = stablehlo.broadcast_in_dim {rs}, dims = [0] : (tensor<{batch}xbf16>) -> {h_ty}\n\
         \x20\x20\x20\x20{normed} = stablehlo.multiply {input}, {rs_b} : {h_ty}\n\
         \x20\x20\x20\x20{w_b} = stablehlo.broadcast_in_dim {weight}, dims = [1] : (tensor<{hidden}xbf16>) -> {h_ty}\n\
         \x20\x20\x20\x20{result} = stablehlo.multiply {normed}, {w_b} : {h_ty}\n",
        eps_val = 1e-6_f64,
        h_inv = 1.0 / h_f,
    ));
    result
}

fn emit_stablehlo_layer(
    out: &mut String,
    shape: &M2GraphShape,
    layer: &M2DecodeLayerPlan,
    dense_arena_bytes: usize,
    hidden_in: &str,
    kv_in: &str,
) -> Result<(String, String)> {
    let b = shape.batch;
    let h = M2_HIDDEN;
    let qh = M2_NUM_Q_HEADS;
    let kvh = M2_NUM_KV_HEADS;
    let hd = M2_HEAD_DIM;
    let rd = M2_ROTARY_DIM;
    let ctx = shape.ctx;
    let qd = qh * hd;
    let kvd = kvh * hd;
    let gqa = qh / kvh;
    let l = layer.layer;

    let kv_bytes = shape.kv_cache_bytes();
    let lkv_bytes = shape.layer_kv_cache_bytes();
    let lkv_offset = shape.layer_kv_cache_offset(l);
    let kv_bpe = shape.kv_bytes_per_elem;
    let p = |s: &str| format!("%l{}_{}", l, s);
    macro_rules! w {
        ($($arg:tt)*) => { out.push_str("    "); out.push_str(&format!($($arg)*)); out.push('\n'); };
    }

    w!("// === Layer {} ===", l);

    // --- Extract dense bf16 weights from dense_weights arena (< INT32_MAX bytes) ---
    let da = dense_arena_bytes;
    let norm1 = emit_dense_bf16(out, da, l, 0, &format!("l{}_norm1", l), &[h]);
    let wq = emit_dense_bf16(out, da, l, 2, &format!("l{}_wq", l), &[qd, h]);
    let wk = emit_dense_bf16(out, da, l, 3, &format!("l{}_wk", l), &[kvd, h]);
    let wv = emit_dense_bf16(out, da, l, 4, &format!("l{}_wv", l), &[kvd, h]);
    let wo = emit_dense_bf16(out, da, l, 5, &format!("l{}_wo", l), &[h, qd]);
    let norm2 = emit_dense_bf16(out, da, l, 1, &format!("l{}_norm2", l), &[h]);

    // --- RMSNorm (pre-attention) ---
    let normed = emit_rmsnorm(out, hidden_in, &norm1, b, h, &format!("l{}_n1", l));

    // --- QKV projections: hidden @ W.T ---
    w!("{} = \"stablehlo.dot_general\"({}, {}) {{", p("qf"), normed, wq);
    w!("  dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>");
    w!("}} : (tensor<{}x{}xbf16>, tensor<{}x{}xbf16>) -> tensor<{}x{}xbf16>", b, h, qd, h, b, qd);
    w!("{} = \"stablehlo.dot_general\"({}, {}) {{", p("kf"), normed, wk);
    w!("  dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>");
    w!("}} : (tensor<{}x{}xbf16>, tensor<{}x{}xbf16>) -> tensor<{}x{}xbf16>", b, h, kvd, h, b, kvd);
    w!("{} = \"stablehlo.dot_general\"({}, {}) {{", p("vf"), normed, wv);
    w!("  dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>");
    w!("}} : (tensor<{}x{}xbf16>, tensor<{}x{}xbf16>) -> tensor<{}x{}xbf16>", b, h, kvd, h, b, kvd);

    // --- Reshape to [B, heads, head_dim] ---
    w!("{} = stablehlo.reshape {} : (tensor<{}x{}xbf16>) -> tensor<{}x{}x{}xbf16>", p("q3"), p("qf"), b, qd, b, qh, hd);
    w!("{} = stablehlo.reshape {} : (tensor<{}x{}xbf16>) -> tensor<{}x{}x{}xbf16>", p("k3"), p("kf"), b, kvd, b, kvh, hd);
    w!("{} = stablehlo.reshape {} : (tensor<{}x{}xbf16>) -> tensor<{}x{}x{}xbf16>", p("v3"), p("vf"), b, kvd, b, kvh, hd);

    // --- RoPE: rotate first rd dims of each head ---
    // Split head_dim into [0:rd] (rotated) and [rd:hd] (passthrough)
    let non_rot = hd - rd;
    w!("{} = stablehlo.slice {} [0:{}, 0:{}, 0:{}] : (tensor<{}x{}x{}xbf16>) -> tensor<{}x{}x{}xbf16>",
        p("q_rot_in"), p("q3"), b, qh, rd, b, qh, hd, b, qh, rd);
    w!("{} = stablehlo.slice {} [0:{}, 0:{}, {}:{}] : (tensor<{}x{}x{}xbf16>) -> tensor<{}x{}x{}xbf16>",
        p("q_pass"), p("q3"), b, qh, rd, hd, b, qh, hd, b, qh, non_rot);
    w!("{} = stablehlo.slice {} [0:{}, 0:{}, 0:{}] : (tensor<{}x{}x{}xbf16>) -> tensor<{}x{}x{}xbf16>",
        p("k_rot_in"), p("k3"), b, kvh, rd, b, kvh, hd, b, kvh, rd);
    w!("{} = stablehlo.slice {} [0:{}, 0:{}, {}:{}] : (tensor<{}x{}x{}xbf16>) -> tensor<{}x{}x{}xbf16>",
        p("k_pass"), p("k3"), b, kvh, rd, hd, b, kvh, hd, b, kvh, non_rot);

    // RoPE: x_rot = x[:,:,0::2]*cos - x[:,:,1::2]*sin, x[:,:,1::2]*cos + x[:,:,0::2]*sin
    // Simplified: split into even/odd pairs, apply rotation matrix
    // For decode, cos/sin come from %positions (dynamic). We need cos[pos] and sin[pos].
    // cos/sin are precomputed as [ctx, rd/2] tensors passed as graph inputs... but they
    // aren't in the current graph signature. For now, skip RoPE and pass Q/K through.
    // TODO: add cos/sin as graph inputs and apply rotation.
    w!("{} = stablehlo.concatenate {}, {}, dim = 2 : (tensor<{}x{}x{}xbf16>, tensor<{}x{}x{}xbf16>) -> tensor<{}x{}x{}xbf16>",
        p("q_roped"), p("q_rot_in"), p("q_pass"), b, qh, rd, b, qh, non_rot, b, qh, hd);
    w!("{} = stablehlo.concatenate {}, {}, dim = 2 : (tensor<{}x{}x{}xbf16>, tensor<{}x{}x{}xbf16>) -> tensor<{}x{}x{}xbf16>",
        p("k_roped"), p("k_rot_in"), p("k_pass"), b, kvh, rd, b, kvh, non_rot, b, kvh, hd);

    // --- KV cache: extract layer slice, write new K/V, read full history ---
    w!("{} = stablehlo.constant dense<{}> : tensor<i32>", p("lkv_off"), lkv_offset);
    w!("{} = \"stablehlo.dynamic_slice\"({}, {}) {{", p("lkv"), kv_in, p("lkv_off"));
    w!("  slice_sizes = array<i64: {}>", lkv_bytes);
    w!("}} : (tensor<{}xi8>, tensor<i32>) -> tensor<{}xi8>", kv_bytes, lkv_bytes);

    // Reshape layer KV to [2, B, ctx, kvh, hd] for bf16 or [2, B, ctx, kvh, hd] for i8
    if kv_bpe == 2 {
        w!("{} = stablehlo.reshape {} : (tensor<{}xi8>) -> tensor<{}x2xi8>",
            p("lkv_2d"), p("lkv"), lkv_bytes, lkv_bytes / 2);
        w!("{} = stablehlo.bitcast_convert {} : (tensor<{}x2xi8>) -> tensor<{}xbf16>",
            p("lkv_bf16"), p("lkv_2d"), lkv_bytes / 2, lkv_bytes / 2);
        w!("{} = stablehlo.reshape {} : (tensor<{}xbf16>) -> tensor<2x{}x{}x{}x{}xbf16>",
            p("lkv5"), p("lkv_bf16"), lkv_bytes / 2, b, ctx, kvh, hd);
    } else {
        w!("{} = stablehlo.reshape {} : (tensor<{}xi8>) -> tensor<2x{}x{}x{}x{}xi8>",
            p("lkv5"), p("lkv"), lkv_bytes, b, ctx, kvh, hd);
    }

    // Write new K at [0, :, pos, :, :] and V at [1, :, pos, :, :]
    // For decode, pos is a scalar broadcast to batch. Use positions[0] as the write slot.
    w!("{} = stablehlo.constant dense<0> : tensor<i32>", p("c0"));
    w!("{} = stablehlo.constant dense<1> : tensor<i32>", p("c1"));
    w!("{} = \"stablehlo.dynamic_slice\"(%positions, {}) {{", p("pos0"), p("c0"));
    w!("  slice_sizes = array<i64: 1>");
    w!("}} : (tensor<{}xi32>, tensor<i32>) -> tensor<1xi32>", b);
    w!("{} = stablehlo.reshape {} : (tensor<1xi32>) -> tensor<i32>", p("pos_s"), p("pos0"));

    if kv_bpe == 2 {
        // K update: reshape k_roped [B,kvh,hd] -> [1,B,1,kvh,hd], write at [0,:,pos,:,:]
        w!("{} = stablehlo.reshape {} : (tensor<{}x{}x{}xbf16>) -> tensor<1x{}x1x{}x{}xbf16>",
            p("k_ins"), p("k_roped"), b, kvh, hd, b, kvh, hd);
        w!("{} = \"stablehlo.dynamic_update_slice\"({}, {}, {}, {}, {}, {}, {}) : (tensor<2x{}x{}x{}x{}xbf16>, tensor<1x{}x1x{}x{}xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x{}x{}x{}x{}xbf16>",
            p("lkv5_k"), p("lkv5"), p("k_ins"), p("c0"), p("c0"), p("pos_s"), p("c0"), p("c0"),
            b, ctx, kvh, hd, b, kvh, hd, b, ctx, kvh, hd);
        w!("{} = stablehlo.reshape {} : (tensor<{}x{}x{}xbf16>) -> tensor<1x{}x1x{}x{}xbf16>",
            p("v_ins"), p("v3"), b, kvh, hd, b, kvh, hd);
        w!("{} = \"stablehlo.dynamic_update_slice\"({}, {}, {}, {}, {}, {}, {}) : (tensor<2x{}x{}x{}x{}xbf16>, tensor<1x{}x1x{}x{}xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x{}x{}x{}x{}xbf16>",
            p("lkv5_kv"), p("lkv5_k"), p("v_ins"), p("c1"), p("c0"), p("pos_s"), p("c0"), p("c0"),
            b, ctx, kvh, hd, b, kvh, hd, b, ctx, kvh, hd);

        // Read full K and V history
        w!("{} = stablehlo.slice {} [0:1, 0:{}, 0:{}, 0:{}, 0:{}] : (tensor<2x{}x{}x{}x{}xbf16>) -> tensor<1x{}x{}x{}x{}xbf16>",
            p("k_hist_5"), p("lkv5_kv"), b, ctx, kvh, hd, b, ctx, kvh, hd, b, ctx, kvh, hd);
        w!("{} = stablehlo.reshape {} : (tensor<1x{}x{}x{}x{}xbf16>) -> tensor<{}x{}x{}x{}xbf16>",
            p("k_hist"), p("k_hist_5"), b, ctx, kvh, hd, b, ctx, kvh, hd);
        w!("{} = stablehlo.slice {} [1:2, 0:{}, 0:{}, 0:{}, 0:{}] : (tensor<2x{}x{}x{}x{}xbf16>) -> tensor<1x{}x{}x{}x{}xbf16>",
            p("v_hist_5"), p("lkv5_kv"), b, ctx, kvh, hd, b, ctx, kvh, hd, b, ctx, kvh, hd);
        w!("{} = stablehlo.reshape {} : (tensor<1x{}x{}x{}x{}xbf16>) -> tensor<{}x{}x{}x{}xbf16>",
            p("v_hist"), p("v_hist_5"), b, ctx, kvh, hd, b, ctx, kvh, hd);

        // Flatten back to i8 for kv_out
        w!("{} = stablehlo.reshape {} : (tensor<2x{}x{}x{}x{}xbf16>) -> tensor<{}xbf16>",
            p("lkv_flat_bf16"), p("lkv5_kv"), b, ctx, kvh, hd, lkv_bytes / 2);
        w!("{} = stablehlo.bitcast_convert {} : (tensor<{}xbf16>) -> tensor<{}x2xi8>",
            p("lkv_out_2d"), p("lkv_flat_bf16"), lkv_bytes / 2, lkv_bytes / 2);
        w!("{} = stablehlo.reshape {} : (tensor<{}x2xi8>) -> tensor<{}xi8>",
            p("lkv_out"), p("lkv_out_2d"), lkv_bytes / 2, lkv_bytes);
    } else {
        // Int8 KV cache: convert bf16 K/V to i8 before write, i8 to bf16 on read
        w!("{} = stablehlo.convert {} : (tensor<{}x{}x{}xbf16>) -> tensor<{}x{}x{}xi8>",
            p("k_i8"), p("k_roped"), b, kvh, hd, b, kvh, hd);
        w!("{} = stablehlo.reshape {} : (tensor<{}x{}x{}xi8>) -> tensor<1x{}x1x{}x{}xi8>",
            p("k_ins"), p("k_i8"), b, kvh, hd, b, kvh, hd);
        w!("{} = \"stablehlo.dynamic_update_slice\"({}, {}, {}, {}, {}, {}, {}) : (tensor<2x{}x{}x{}x{}xi8>, tensor<1x{}x1x{}x{}xi8>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x{}x{}x{}x{}xi8>",
            p("lkv5_k"), p("lkv5"), p("k_ins"), p("c0"), p("c0"), p("pos_s"), p("c0"), p("c0"),
            b, ctx, kvh, hd, b, kvh, hd, b, ctx, kvh, hd);
        w!("{} = stablehlo.convert {} : (tensor<{}x{}x{}xbf16>) -> tensor<{}x{}x{}xi8>",
            p("v_i8"), p("v3"), b, kvh, hd, b, kvh, hd);
        w!("{} = stablehlo.reshape {} : (tensor<{}x{}x{}xi8>) -> tensor<1x{}x1x{}x{}xi8>",
            p("v_ins"), p("v_i8"), b, kvh, hd, b, kvh, hd);
        w!("{} = \"stablehlo.dynamic_update_slice\"({}, {}, {}, {}, {}, {}, {}) : (tensor<2x{}x{}x{}x{}xi8>, tensor<1x{}x1x{}x{}xi8>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x{}x{}x{}x{}xi8>",
            p("lkv5_kv"), p("lkv5_k"), p("v_ins"), p("c1"), p("c0"), p("pos_s"), p("c0"), p("c0"),
            b, ctx, kvh, hd, b, kvh, hd, b, ctx, kvh, hd);

        // Read K/V history as i8, convert to bf16
        w!("{} = stablehlo.slice {} [0:1, 0:{}, 0:{}, 0:{}, 0:{}] : (tensor<2x{}x{}x{}x{}xi8>) -> tensor<1x{}x{}x{}x{}xi8>",
            p("k_hist_i8_5"), p("lkv5_kv"), b, ctx, kvh, hd, b, ctx, kvh, hd, b, ctx, kvh, hd);
        w!("{} = stablehlo.reshape {} : (tensor<1x{}x{}x{}x{}xi8>) -> tensor<{}x{}x{}x{}xi8>",
            p("k_hist_i8"), p("k_hist_i8_5"), b, ctx, kvh, hd, b, ctx, kvh, hd);
        w!("{} = stablehlo.convert {} : (tensor<{}x{}x{}x{}xi8>) -> tensor<{}x{}x{}x{}xbf16>",
            p("k_hist"), p("k_hist_i8"), b, ctx, kvh, hd, b, ctx, kvh, hd);
        w!("{} = stablehlo.slice {} [1:2, 0:{}, 0:{}, 0:{}, 0:{}] : (tensor<2x{}x{}x{}x{}xi8>) -> tensor<1x{}x{}x{}x{}xi8>",
            p("v_hist_i8_5"), p("lkv5_kv"), b, ctx, kvh, hd, b, ctx, kvh, hd, b, ctx, kvh, hd);
        w!("{} = stablehlo.reshape {} : (tensor<1x{}x{}x{}x{}xi8>) -> tensor<{}x{}x{}x{}xi8>",
            p("v_hist_i8"), p("v_hist_i8_5"), b, ctx, kvh, hd, b, ctx, kvh, hd);
        w!("{} = stablehlo.convert {} : (tensor<{}x{}x{}x{}xi8>) -> tensor<{}x{}x{}x{}xbf16>",
            p("v_hist"), p("v_hist_i8"), b, ctx, kvh, hd, b, ctx, kvh, hd);

        // Flatten updated KV back to i8
        w!("{} = stablehlo.reshape {} : (tensor<2x{}x{}x{}x{}xi8>) -> tensor<{}xi8>",
            p("lkv_out"), p("lkv5_kv"), b, ctx, kvh, hd, lkv_bytes);
    }

    // --- GQA: broadcast KV heads to match Q heads ---
    // k_hist: [B, ctx, kvh, hd] -> transpose to [B, kvh, ctx, hd]
    w!("{} = stablehlo.transpose {}, dims = [0, 2, 1, 3] : (tensor<{}x{}x{}x{}xbf16>) -> tensor<{}x{}x{}x{}xbf16>",
        p("kt"), p("k_hist"), b, ctx, kvh, hd, b, kvh, ctx, hd);
    w!("{} = stablehlo.transpose {}, dims = [0, 2, 1, 3] : (tensor<{}x{}x{}x{}xbf16>) -> tensor<{}x{}x{}x{}xbf16>",
        p("vt"), p("v_hist"), b, ctx, kvh, hd, b, kvh, ctx, hd);
    // Broadcast kvh -> qh by reshaping [B, kvh, ctx, hd] -> [B, kvh, 1, ctx, hd] -> broadcast -> [B, kvh, gqa, ctx, hd] -> reshape [B, qh, ctx, hd]
    w!("{} = stablehlo.reshape {} : (tensor<{}x{}x{}x{}xbf16>) -> tensor<{}x{}x1x{}x{}xbf16>",
        p("kt5"), p("kt"), b, kvh, ctx, hd, b, kvh, ctx, hd);
    w!("{} = stablehlo.broadcast_in_dim {}, dims = [0, 1, 2, 3, 4] : (tensor<{}x{}x1x{}x{}xbf16>) -> tensor<{}x{}x{}x{}x{}xbf16>",
        p("kt_bc"), p("kt5"), b, kvh, ctx, hd, b, kvh, gqa, ctx, hd);
    w!("{} = stablehlo.reshape {} : (tensor<{}x{}x{}x{}x{}xbf16>) -> tensor<{}x{}x{}x{}xbf16>",
        p("k_full"), p("kt_bc"), b, kvh, gqa, ctx, hd, b, qh, ctx, hd);
    w!("{} = stablehlo.reshape {} : (tensor<{}x{}x{}x{}xbf16>) -> tensor<{}x{}x1x{}x{}xbf16>",
        p("vt5"), p("vt"), b, kvh, ctx, hd, b, kvh, ctx, hd);
    w!("{} = stablehlo.broadcast_in_dim {}, dims = [0, 1, 2, 3, 4] : (tensor<{}x{}x1x{}x{}xbf16>) -> tensor<{}x{}x{}x{}x{}xbf16>",
        p("vt_bc"), p("vt5"), b, kvh, ctx, hd, b, kvh, gqa, ctx, hd);
    w!("{} = stablehlo.reshape {} : (tensor<{}x{}x{}x{}x{}xbf16>) -> tensor<{}x{}x{}x{}xbf16>",
        p("v_full"), p("vt_bc"), b, kvh, gqa, ctx, hd, b, qh, ctx, hd);

    // --- Attention scores: Q @ K.T / sqrt(hd) ---
    // q_roped: [B, qh, hd] -> add seq dim -> [B, qh, 1, hd]
    w!("{} = stablehlo.reshape {} : (tensor<{}x{}x{}xbf16>) -> tensor<{}x{}x1x{}xbf16>",
        p("q4"), p("q_roped"), b, qh, hd, b, qh, hd);
    // k_full: [B, qh, ctx, hd] -> transpose last two -> [B, qh, hd, ctx]
    w!("{} = stablehlo.transpose {}, dims = [0, 1, 3, 2] : (tensor<{}x{}x{}x{}xbf16>) -> tensor<{}x{}x{}x{}xbf16>",
        p("k_t"), p("k_full"), b, qh, ctx, hd, b, qh, hd, ctx);
    // scores = Q @ K.T -> [B, qh, 1, ctx]
    w!("{} = \"stablehlo.dot_general\"({}, {}) {{", p("scores_raw"), p("q4"), p("k_t"));
    w!("  dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>");
    w!("}} : (tensor<{}x{}x1x{}xbf16>, tensor<{}x{}x{}x{}xbf16>) -> tensor<{}x{}x1x{}xbf16>", b, qh, hd, b, qh, hd, ctx, b, qh, ctx);
    // Scale by 1/sqrt(hd)
    let scale = 1.0_f64 / (hd as f64).sqrt();
    w!("{} = stablehlo.constant dense<{}> : tensor<bf16>", p("scale"), scale);
    w!("{} = stablehlo.broadcast_in_dim {}, dims = [] : (tensor<bf16>) -> tensor<{}x{}x1x{}xbf16>",
        p("scale_b"), p("scale"), b, qh, ctx);
    w!("{} = stablehlo.multiply {}, {} : tensor<{}x{}x1x{}xbf16>",
        p("scores"), p("scores_raw"), p("scale_b"), b, qh, ctx);

    // --- Softmax (numerically stable) ---
    w!("{} = stablehlo.constant dense<0xFF80> : tensor<bf16>", p("neg_inf"));
    w!("{} = \"stablehlo.reduce\"({}, {}) ({{", p("s_max"), p("scores"), p("neg_inf"));
    w!("    ^smbb{}(%sma_{}: tensor<bf16>, %smb_{}: tensor<bf16>):", l, l, l);
    w!("      %smr_{} = stablehlo.maximum %sma_{}, %smb_{} : tensor<bf16>", l, l, l);
    w!("      stablehlo.return %smr_{} : tensor<bf16>", l);
    w!("    }}) {{dimensions = array<i64: 3>}} : (tensor<{}x{}x1x{}xbf16>, tensor<bf16>) -> tensor<{}x{}x1xbf16>",
        b, qh, ctx, b, qh);
    w!("{} = stablehlo.broadcast_in_dim {}, dims = [0, 1, 2] : (tensor<{}x{}x1xbf16>) -> tensor<{}x{}x1x{}xbf16>",
        p("s_max_b"), p("s_max"), b, qh, b, qh, ctx);
    w!("{} = stablehlo.subtract {}, {} : tensor<{}x{}x1x{}xbf16>",
        p("s_shifted"), p("scores"), p("s_max_b"), b, qh, ctx);
    w!("{} = stablehlo.exponential {} : tensor<{}x{}x1x{}xbf16>",
        p("s_exp"), p("s_shifted"), b, qh, ctx);
    w!("{} = stablehlo.constant dense<0.0> : tensor<bf16>", p("zero"));
    w!("{} = \"stablehlo.reduce\"({}, {}) ({{", p("s_sum"), p("s_exp"), p("zero"));
    w!("    ^ssbb{}(%ssa_{}: tensor<bf16>, %ssb_{}: tensor<bf16>):", l, l, l);
    w!("      %ssr_{} = stablehlo.add %ssa_{}, %ssb_{} : tensor<bf16>", l, l, l);
    w!("      stablehlo.return %ssr_{} : tensor<bf16>", l);
    w!("    }}) {{dimensions = array<i64: 3>}} : (tensor<{}x{}x1x{}xbf16>, tensor<bf16>) -> tensor<{}x{}x1xbf16>",
        b, qh, ctx, b, qh);
    w!("{} = stablehlo.broadcast_in_dim {}, dims = [0, 1, 2] : (tensor<{}x{}x1xbf16>) -> tensor<{}x{}x1x{}xbf16>",
        p("s_sum_b"), p("s_sum"), b, qh, b, qh, ctx);
    w!("{} = stablehlo.divide {}, {} : tensor<{}x{}x1x{}xbf16>",
        p("attn_w"), p("s_exp"), p("s_sum_b"), b, qh, ctx);

    // --- Attention output: attn_weights @ V ---
    // attn_w: [B, qh, 1, ctx], v_full: [B, qh, ctx, hd] -> [B, qh, 1, hd]
    w!("{} = \"stablehlo.dot_general\"({}, {}) {{", p("attn_out4"), p("attn_w"), p("v_full"));
    w!("  dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>");
    w!("}} : (tensor<{}x{}x1x{}xbf16>, tensor<{}x{}x{}x{}xbf16>) -> tensor<{}x{}x1x{}xbf16>", b, qh, ctx, b, qh, ctx, hd, b, qh, hd);
    // Reshape to [B, qd] and project
    w!("{} = stablehlo.reshape {} : (tensor<{}x{}x1x{}xbf16>) -> tensor<{}x{}xbf16>",
        p("attn_flat"), p("attn_out4"), b, qh, hd, b, qd);
    // O projection: [B, qd] @ [h, qd].T -> [B, h]
    w!("{} = \"stablehlo.dot_general\"({}, {}) {{", p("attn_proj"), p("attn_flat"), wo);
    w!("  dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>");
    w!("}} : (tensor<{}x{}xbf16>, tensor<{}x{}xbf16>) -> tensor<{}x{}xbf16>", b, qd, h, qd, b, h);

    // --- Residual add (attention) ---
    w!("{} = stablehlo.add {}, {} : tensor<{}x{}xbf16>",
        p("h_attn"), hidden_in, p("attn_proj"), b, h);

    // --- RMSNorm (pre-MLP / pre-MoE) ---
    let _normed2 = emit_rmsnorm(out, &p("h_attn"), &norm2, b, h, &format!("l{}_n2", l));

    // --- MoE: for now, pass through (no expert MLP) ---
    // TODO: implement MoE router + expert dispatch
    // The hidden state after attention is the output for this layer until MoE is wired.
    let next_hidden = format!("%h_l{}", l);
    w!("{} = stablehlo.optimization_barrier {} : tensor<{}x{}xbf16>",
        next_hidden, p("h_attn"), b, h);

    // --- Write updated KV back to global cache ---
    let next_kv = format!("%kv_after_l{}", l);
    w!("{} = stablehlo.constant dense<{}> : tensor<i32>", p("lkv_off_out"), lkv_offset);
    w!("{} = \"stablehlo.dynamic_update_slice\"({}, {}, {}) : (tensor<{}xi8>, tensor<{}xi8>, tensor<i32>) -> tensor<{}xi8>",
        next_kv, kv_in, p("lkv_out"), p("lkv_off_out"), kv_bytes, lkv_bytes, kv_bytes);

    Ok((next_hidden, next_kv))
}

fn emit_layer_call(
    shape: &M2GraphShape,
    layer: &M2DecodeLayerPlan,
    src: &str,
    kv_src: &str,
    dst: &str,
    kv_dst: &str,
    hidden_ty: &str,
    token_ty: &str,
    kv_ty: &str,
    decode_layer_body: Option<&TpuMosaicSerializedBody>,
    int8_tile: Option<&str>,
    int8_row_scales: Option<&str>,
) -> Result<String> {
    let target = layer.custom_call_abi(shape)?.call.target();
    let backend_config = match decode_layer_body {
        Some(body) => tpu_custom_call_backend_config_for_body(body),
        None => tpu_custom_call_backend_config(""),
    };
    let layer_offsets_name = format!("%layer_offsets_{}", layer.layer);
    let expert_directory_name = format!("%expert_directory_{}", layer.layer);
    let layer_offsets = compact_i32_row(&layer.arena_offsets().as_i32_split_row());
    let expert_directory = dense_i32_matrix(
        &layer
            .expert_directory()
            .iter()
            .map(M2ExpertDirectoryEntry::as_i32_split_row)
            .collect::<Vec<_>>(),
    );
    if let (Some(int8_tile), Some(int8_row_scales)) = (int8_tile, int8_row_scales) {
        let w1_block_cols = crate::m2_int8_w1_n_total();
        let n_step = 128usize;
        let n_tiles = w1_block_cols / n_step;
        let n_loop = n_tiles > 1;
        let (weight_tensor_ty, scale_tensor_ty, weight_layout, scale_layout) = if n_loop {
            (
                format!("tensor<{}x{}x{}xi8>", M2_HIDDEN, n_tiles, n_step),
                format!("tensor<{}x{}xf32>", n_tiles, n_step),
                "dense<[2, 1, 0]> : tensor<3xindex>".to_string(),
                "dense<[1, 0]> : tensor<2xindex>".to_string(),
            )
        } else {
            (
                format!("tensor<{}x{}xi8>", M2_HIDDEN, w1_block_cols),
                format!("tensor<{}xf32>", w1_block_cols),
                "dense<[1, 0]> : tensor<2xindex>".to_string(),
                "dense<[0]> : tensor<1xindex>".to_string(),
            )
        };
        return Ok(format!(
            r#"    {layer_offsets_name} = stablehlo.constant dense<{layer_offsets}> : tensor<{offset_cols}xi32>
    {expert_directory_name} = stablehlo.constant dense<{expert_directory}> : tensor<{expert_count}x{expert_cols}xi32>
    {dst}, {kv_dst} = "stablehlo.custom_call"({src}, %positions, {kv_src}, {layer_offsets_name}, {expert_directory_name}, {int8_tile}, {int8_row_scales}) {{
      call_target_name = "{tpu_custom_call}",
      backend_config = "{backend_config}",
      called_computations = [],
      has_side_effect = false,
      api_version = 1 : i32,
      kernel_name = "{target}",
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, {weight_layout}, {scale_layout}],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>]
    }} : ({hidden_ty}, {token_ty}, {kv_ty}, tensor<{offset_cols}xi32>, tensor<{expert_count}x{expert_cols}xi32>, {weight_tensor_ty}, {scale_tensor_ty}) -> ({hidden_ty}, {kv_ty})
"#,
            tpu_custom_call = TPU_CUSTOM_CALL_TARGET,
            backend_config = backend_config,
            target = rvllm_fused::M2_INT8_CUSTOM_CALL_TARGET,
            src = src,
            kv_src = kv_src,
            dst = dst,
            kv_dst = kv_dst,
            int8_tile = int8_tile,
            int8_row_scales = int8_row_scales,
            hidden_ty = hidden_ty,
            token_ty = token_ty,
            kv_ty = kv_ty,
            layer_offsets_name = layer_offsets_name,
            expert_directory_name = expert_directory_name,
            layer_offsets = layer_offsets,
            expert_directory = expert_directory,
            offset_cols = M2DecodeLayerArenaOffsets::I32_COLS,
            expert_count = M2_NUM_EXPERTS,
            expert_cols = M2ExpertDirectoryEntry::I32_COLS,
            weight_tensor_ty = weight_tensor_ty,
            scale_tensor_ty = scale_tensor_ty,
            weight_layout = weight_layout,
            scale_layout = scale_layout,
        ));
    }

    Ok(format!(
        r#"    {layer_offsets_name} = stablehlo.constant dense<{layer_offsets}> : tensor<{offset_cols}xi32>
    {expert_directory_name} = stablehlo.constant dense<{expert_directory}> : tensor<{expert_count}x{expert_cols}xi32>
    {dst}, {kv_dst} = "stablehlo.custom_call"({src}, %positions, {kv_src}, {layer_offsets_name}, {expert_directory_name}) {{
      call_target_name = "{tpu_custom_call}",
      backend_config = "{backend_config}",
      called_computations = [],
      has_side_effect = false,
      api_version = 1 : i32,
      kernel_name = "{target}",
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>]
    }} : ({hidden_ty}, {token_ty}, {kv_ty}, tensor<{offset_cols}xi32>, tensor<{expert_count}x{expert_cols}xi32>) -> ({hidden_ty}, {kv_ty})
"#,
        tpu_custom_call = TPU_CUSTOM_CALL_TARGET,
        backend_config = backend_config,
        target = target,
        src = src,
        kv_src = kv_src,
        dst = dst,
        kv_dst = kv_dst,
        layer_offsets_name = layer_offsets_name,
        expert_directory_name = expert_directory_name,
        layer_offsets = layer_offsets,
        expert_directory = expert_directory,
        offset_cols = M2DecodeLayerArenaOffsets::I32_COLS,
        expert_count = M2_NUM_EXPERTS,
        expert_cols = M2ExpertDirectoryEntry::I32_COLS,
    ))
}

fn compact_i32_row<const N: usize>(row: &[i32; N]) -> String {
    let mut out = String::from("[");
    for (idx, value) in row.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

fn dense_i32_matrix<const N: usize>(rows: &[[i32; N]]) -> String {
    let mut out = String::from("[");
    for (idx, row) in rows.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&compact_i32_row(row));
    }
    out.push(']');
    out
}

fn dense(lookup: &ArenaLookup<'_>, layer: usize, suffix: &str) -> Result<M2ArenaTensor> {
    lookup
        .entry(&format!("model.layers.{layer}.{suffix}"))
        .map(arena_tensor)
}

fn nvfp4_projection(
    lookup: &ArenaLookup<'_>,
    layer: usize,
    expert: usize,
    projection: M2Projection,
) -> Result<M2Nvfp4ProjectionPlan> {
    let abi = M2Nvfp4ProjectionAbi::new(layer, expert, projection);
    Ok(M2Nvfp4ProjectionPlan {
        projection,
        rows: abi.rows,
        cols: abi.cols,
        packed: arena_tensor(lookup.entry(&abi.weight.name)?),
        scale: arena_tensor(lookup.entry(&abi.weight_scale.name)?),
        global_scale: arena_tensor(lookup.entry(&abi.weight_scale_2.name)?),
        input_scale: lookup.get(&abi.input_scale.name).map(arena_tensor),
    })
}

fn input_scale_offset(proj: &M2Nvfp4ProjectionPlan) -> i64 {
    proj.input_scale
        .as_ref()
        .map(|scale| scale.offset as i64)
        .unwrap_or(-1)
}

fn arena_tensor(entry: &M2WeightArenaEntry) -> M2ArenaTensor {
    M2ArenaTensor {
        name: entry.name.clone(),
        offset: entry.offset,
        nbytes: entry.nbytes,
        shape: entry.shape.clone(),
        dtype: entry.dtype,
    }
}

struct ArenaLookup<'a> {
    by_name: BTreeMap<&'a str, &'a M2WeightArenaEntry>,
}

impl<'a> ArenaLookup<'a> {
    fn new(arena: &'a M2WeightArenaPlan) -> Self {
        Self {
            by_name: arena
                .entries
                .iter()
                .map(|entry| (entry.name.as_str(), entry))
                .collect(),
        }
    }

    fn get(&self, name: &str) -> Option<&'a M2WeightArenaEntry> {
        self.by_name.get(name).copied()
    }

    fn entry(&self, name: &str) -> Result<&'a M2WeightArenaEntry> {
        self.get(name)
            .ok_or_else(|| invalid_owned("tensor", format!("missing arena tensor: {name}")))
    }
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

fn invalid_owned(field: &'static str, reason: String) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason,
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

    use crate::{M2GraphAbi, M2Int8ProjectionSet, M2WeightUploadPlan};

    use super::*;

    fn arena() -> M2WeightArenaPlan {
        weight_plan().flat_arena(128).unwrap()
    }

    fn int8_w1_arena() -> M2WeightArenaPlan {
        weight_plan()
            .int8_flat_arena_for(128, M2Int8ProjectionSet::w1())
            .unwrap()
    }

    fn weight_plan() -> M2WeightUploadPlan {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema");
        let shape = M2GraphShape::decode(8, 2048, 1);
        let abi = M2GraphAbi::new(shape).unwrap();
        M2WeightUploadPlan::from_index_dir(model_dir, &abi).unwrap()
    }

    #[test]
    fn decode_graph_plan_resolves_real_layer_offsets() {
        let arena = arena();
        let plan = M2DecodeGraphPlan::from_arena(&arena).unwrap();
        assert_eq!(plan.layers.len(), M2_NUM_LAYERS);
        assert_eq!(plan.layers[0].experts.len(), M2_NUM_EXPERTS);
        assert_eq!(plan.input_scale_missing_count(), 18);
        assert_eq!(plan.embed.name, "model.embed_tokens.weight");
        assert_eq!(
            plan.layers[0].q_proj.name,
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(plan.layers[0].q_proj.shape, vec![6_144, 3_072]);
        assert_eq!(
            plan.layers[0].experts[0].w1.packed.shape,
            vec![1_536, 1_536]
        );
        assert_eq!(plan.layers[0].experts[0].w1.scale.shape, vec![1_536, 192]);
        assert_eq!(plan.layers[61].experts[255].w3.global_scale.nbytes, 4);
    }

    #[test]
    fn expert_directory_pins_all_nvfp4_offsets_for_fused_moe() {
        let arena = arena();
        let plan = M2DecodeGraphPlan::from_arena(&arena).unwrap();
        let layer0 = &plan.layers[0];
        let dir = layer0.expert_directory();
        assert_eq!(dir.len(), M2_NUM_EXPERTS);
        assert_eq!(M2ExpertDirectoryEntry::COLS, 13);
        assert_eq!(dir[0].expert, 0);
        assert_eq!(
            dir[0].w1_packed_offset,
            layer0.experts[0].w1.packed.offset as i64
        );
        assert_eq!(
            dir[0].w1_scale_offset,
            layer0.experts[0].w1.scale.offset as i64
        );
        assert_eq!(
            dir[0].w1_global_scale_offset,
            layer0.experts[0].w1.global_scale.offset as i64
        );
        assert_eq!(
            dir[0].w2_packed_offset,
            layer0.experts[0].w2.packed.offset as i64
        );
        assert_eq!(
            dir[0].w3_packed_offset,
            layer0.experts[0].w3.packed.offset as i64
        );
        assert_eq!(dir[255].expert, 255);
        assert_eq!(
            dir[255].w3_global_scale_offset,
            layer0.experts[255].w3.global_scale.offset as i64
        );
        assert_eq!(dir[0].as_i64_row().len(), M2ExpertDirectoryEntry::COLS);
    }

    #[test]
    fn emits_decode_graph_contract_over_flat_weight_arena() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        let arena = arena();
        let mlir = m2_decode_graph_mlir("rvllm_m2_decode", &shape, &arena).unwrap();
        assert!(mlir.contains("rvllm.kind = \"m2_decode_graph\""));
        assert!(mlir.contains("tensor<8xi32>"));
        assert!(mlir.contains("tensor<2080374784xi8>"));
        assert!(mlir.contains("tensor<33554432xi8>"));
        assert!(!mlir.contains("memref."));
        assert!(mlir.contains("weight_arena"));
        assert!(mlir.contains("rvllm.weight_entries = 191069 : i64"));
        assert!(mlir.contains("rvllm.weight_input_scales_missing = 18 : i64"));
        assert!(mlir.contains("rvllm.lowering = \"rust_xla_custom_call\""));
        assert_eq!(
            mlir.matches("call_target_name = \"tpu_custom_call\"")
                .count(),
            M2_NUM_LAYERS
        );
        assert_eq!(
            mlir.matches("kernel_name = \"rvllm.m2.decode_layer.fused_attention_nvfp4_moe\"")
                .count(),
            M2_NUM_LAYERS
        );
        assert!(!mlir.contains("kernel_name = \"rvllm.m2.embed\""));
        assert!(!mlir.contains("kernel_name = \"rvllm.m2.final_logits\""));
        assert!(mlir.contains("host BF16 embedding gather"));
        assert!(mlir.contains("input_hidden"));
        assert!(mlir.contains("final_norm"));
        assert!(mlir.contains("lm_head"));
        assert!(mlir.contains("\"stablehlo.dot_general\""));
        assert!(!mlir.contains("native StableHLO embed/final placeholders"));
        assert!(!mlir.contains("embed_tokens"));
        assert!(!mlir.contains("\"stablehlo.gather\""));
        assert!(!mlir.contains("%h_embed = stablehlo.broadcast_in_dim"));
        assert!(!mlir.contains("%logits = stablehlo.broadcast_in_dim"));
        assert!(mlir.contains("\\22custom_call_config\\22"));
        assert!(mlir.contains("\\22serialization_format\\22: 1"));
        assert!(mlir.contains("\\22needs_layout_passes\\22: true"));
        assert!(mlir.contains("\\22implicit_sharding\\22"));
        assert!(mlir.contains("rvllm.batch = 8 : i64"));
        assert!(mlir.contains("rvllm.ctx = 2048 : i64"));
        assert!(!mlir.contains("target=rvllm.m2.decode_layer.fused_attention_nvfp4_moe"));
        assert!(!mlir.contains("Contract body placeholder"));
    }

    #[test]
    fn can_link_serialized_decode_layer_body_into_all_layer_calls() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        let arena = arena();
        let body =
            TpuMosaicSerializedBody::from_serialized_bytecode(&[0x4d, 0x4c, 0xef, 0x52]).unwrap();
        let mlir =
            m2_decode_graph_mlir_with_mosaic_body("rvllm_m2_decode", &shape, &arena, Some(&body))
                .unwrap();
        assert_eq!(
            mlir.matches("\\22body\\22: \\22TUzvUg==\\22").count(),
            M2_NUM_LAYERS
        );
        assert!(!mlir.contains("\\22body\\22: \\22\\22"));
    }

    #[test]
    fn can_link_lowered_decode_layer_body_without_serde_format() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        let arena = int8_w1_arena();
        let body = TpuMosaicSerializedBody::from_lowered_mlir(b"module { }").unwrap();
        let mlir =
            m2_decode_graph_mlir_with_mosaic_body("rvllm_m2_decode", &shape, &arena, Some(&body))
                .unwrap();
        assert_eq!(
            mlir.matches("\\22body\\22: \\22bW9kdWxlIHsgfQ==\\22")
                .count(),
            M2_NUM_LAYERS
        );
        assert!(mlir.contains("%h_l0, %kv_layer_out_0 = \"stablehlo.custom_call\""));
        assert!(!mlir.contains("output_operand_aliases = [#stablehlo.output_operand_alias"));
        assert!(mlir.contains(
            "tensor<3072x128xi8>, tensor<128xf32>) -> (tensor<8x3072xbf16>, tensor<33554432xi8>)"
        ));
        assert!(!mlir.contains("_w1_head"));
        assert!(!mlir.contains("_w1_tail"));
        assert!(!mlir.contains("\"stablehlo.concatenate\""));
        assert!(mlir.contains("%kv_after_l0"));
        assert!(!mlir.contains("\\22serialization_format\\22"));
    }

    #[test]
    fn emits_native_stablehlo_smoke_without_custom_calls() {
        let shape = M2GraphShape::decode(8, 2048, 1);
        let mlir = m2_decode_smoke_mlir("main", &shape).unwrap();
        assert!(mlir.contains("rvllm.kind = \"m2_decode_smoke\""));
        assert!(mlir.contains("func.func public @main"));
        assert!(mlir.contains("mhlo.num_partitions = 1 : i32"));
        assert!(mlir.contains("jax.result_info = \"logits\""));
        assert!(mlir.contains("tensor<8x200064xbf16>"));
        assert!(mlir.contains("tensor<2080374784xi8>"));
        assert!(!mlir.contains("stablehlo.custom_call"));
        assert!(!mlir.contains("tpu_custom_call"));
    }

    #[test]
    fn rejects_prefill_shape_for_decode_graph() {
        let shape = M2GraphShape::prefill(8, 20, 2048, 1);
        let arena = arena();
        assert!(m2_decode_graph_mlir("rvllm_m2_decode", &shape, &arena).is_err());
    }
}
