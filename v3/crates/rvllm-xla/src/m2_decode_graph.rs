use std::collections::BTreeMap;

use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_loader::M2Projection;

use crate::m2_graph_abi::{M2DecodeLayerArenaOffsets, M2DecodeLayerCustomCallAbi};
use crate::{
    M2GraphPhase, M2GraphShape, M2Nvfp4ProjectionAbi, M2WeightArenaEntry, M2WeightArenaPlan,
    PjrtElementType, M2_HEAD_DIM, M2_HIDDEN, M2_NUM_EXPERTS, M2_NUM_KV_HEADS, M2_NUM_LAYERS,
    M2_NUM_Q_HEADS, M2_NVFP4_GROUP, M2_ROTARY_DIM, M2_VOCAB,
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
    shape.validate()?;
    if shape.phase != M2GraphPhase::Decode {
        return Err(invalid("phase", "expected decode graph shape"));
    }
    if !is_mlir_symbol(kernel_name) {
        return Err(invalid("kernel_name", "must be an MLIR symbol"));
    }
    let plan = M2DecodeGraphPlan::from_arena(arena)?;
    let body = emit_decode_body(shape, arena, &plan)?;
    Ok(format!(
        r#"module attributes {{rvllm.kind = "m2_decode_graph"}} {{
  func.func @{kernel_name}(
      %token_ids: tensor<{batch}xi32>,
      %positions: tensor<{batch}xi32>,
      %kv_cache: tensor<{kv_bytes}xi8>,
      %weight_arena: tensor<{weight_bytes}xi8>)
      -> (tensor<{batch}x{vocab}xbf16>, tensor<{batch}xi32>, tensor<{kv_bytes}xi8>)
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
        rvllm.weight_input_scales_missing = {missing_input_scales} : i64,
        rvllm.weight_metadata = "compile_time_offsets_from_M2WeightArenaPlan",
        rvllm.lowering = "rust_xla_custom_call",
        rvllm.lowering_plan = "embed -> 62 fused decode-layer custom calls -> flat-arena dense loads -> fused attention -> top-k router -> flat-arena NVFP4 experts -> final norm -> lm_head -> argmax"
      }} {{
{body}
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
        missing_input_scales = plan.input_scale_missing_count(),
        body = body,
    ))
}

fn emit_decode_body(
    shape: &M2GraphShape,
    arena: &M2WeightArenaPlan,
    plan: &M2DecodeGraphPlan,
) -> Result<String> {
    let hidden_ty = format!("tensor<{}x{}xbf16>", shape.batch, M2_HIDDEN);
    let kv_ty = format!("tensor<{}xi8>", shape.kv_cache_bytes());
    let arena_ty = format!("tensor<{}xi8>", arena.total_bytes);
    let token_ty = format!("tensor<{}xi32>", shape.batch);
    let logits_ty = format!("tensor<{}x{}xbf16>", shape.batch, M2_VOCAB);
    let mut out = String::new();

    out.push_str(&format!(
        r#"    %h_embed = "stablehlo.custom_call"(%token_ids, %weight_arena) {{
      call_target_name = "rvllm.m2.embed",
      backend_config = "target=rvllm.m2.embed;embed_offset={embed_offset};embed_nbytes={embed_nbytes}",
      called_computations = [],
      has_side_effect = false,
      api_version = 2 : i32
    }} : ({token_ty}, {arena_ty}) -> {hidden_ty}
"#,
        embed_offset = plan.embed.offset,
        embed_nbytes = plan.embed.nbytes,
    ));

    let mut hidden = "%h_embed".to_string();
    let mut kv = "%kv_cache".to_string();
    for layer in &plan.layers {
        let next_hidden = format!("%h_l{}", layer.layer);
        let next_kv = format!("%kv_l{}", layer.layer);
        out.push_str(&emit_layer_call(
            shape,
            arena,
            layer,
            &hidden,
            &kv,
            &next_hidden,
            &next_kv,
            &hidden_ty,
            &token_ty,
            &kv_ty,
            &arena_ty,
        )?);
        hidden = next_hidden;
        kv = next_kv;
    }

    out.push_str(&format!(
        r#"    %logits, %next_token = "stablehlo.custom_call"({final_hidden}, %weight_arena) {{
      call_target_name = "rvllm.m2.final_logits",
      backend_config = "target=rvllm.m2.final_logits;norm_offset={norm_offset};lm_head_offset={lm_head_offset};vocab={vocab}",
      called_computations = [],
      has_side_effect = false,
      api_version = 2 : i32
    }} : ({hidden_ty}, {arena_ty}) -> ({logits_ty}, {token_ty})
    return %logits, %next_token, {final_kv} : {logits_ty}, {token_ty}, {kv_ty}
"#,
        final_hidden = hidden,
        final_kv = kv,
        norm_offset = plan.final_norm.offset,
        lm_head_offset = plan.lm_head.offset,
        vocab = M2_VOCAB,
    ));
    Ok(out)
}

fn emit_layer_call(
    shape: &M2GraphShape,
    arena: &M2WeightArenaPlan,
    layer: &M2DecodeLayerPlan,
    src: &str,
    kv_src: &str,
    dst: &str,
    kv_dst: &str,
    hidden_ty: &str,
    token_ty: &str,
    kv_ty: &str,
    arena_ty: &str,
) -> Result<String> {
    let abi = layer.custom_call_abi(shape)?;
    let target = abi.call.target();
    let offsets = abi.weight_offsets;
    let expert_directory = expert_directory_attr(layer);
    Ok(format!(
        r#"    {dst}, {kv_dst} = "stablehlo.custom_call"({src}, %positions, {kv_src}, %weight_arena) {{
      call_target_name = "rvllm.m2.decode_layer.fused_attention_nvfp4_moe",
      backend_config = "target={target};custom_call_abi={custom_call_abi};layer={layer_idx};batch={batch};ctx={ctx};kv_dtype={kv_dtype};kv_cache_bytes={kv_cache_bytes};hidden={hidden};num_q_heads={num_q_heads};num_kv_heads={num_kv_heads};head_dim={head_dim};rotary_dim={rotary_dim};top_k={top_k};expert_count={expert_count};nvfp4_group={nvfp4_group};weight_arena=flat_i8_offsets;weight_arena_bytes={weight_arena_bytes};weight_arena_alignment={weight_arena_alignment};weight_arena_dense_offsets={dense_offsets};input_norm_offset={input_norm};post_attention_norm_offset={post_attention_norm};q_proj_offset={q_proj};k_proj_offset={k_proj};v_proj_offset={v_proj};o_proj_offset={o_proj};q_norm_offset={q_norm};k_norm_offset={k_norm};router_offset={router};router_bias_offset={router_bias};w1_first_packed_offset={w1_first_packed};w1_first_scale_offset={w1_first_scale};w1_first_global_scale_offset={w1_first_global};w1_first_input_scale_offset={w1_first_input};w2_first_packed_offset={w2_first_packed};w3_first_packed_offset={w3_first_packed};w3_last_packed_offset={w3_last_packed};input_scales_missing={missing_input_scales};expert_directory=packed_i64_offsets;expert_directory_cols={expert_directory_cols};expert_directory_i64={expert_directory_compact};dispatch=fused_attention_topk_nvfp4_moe;lowering=rust_xla_custom_call",
      called_computations = [],
      has_side_effect = false,
      api_version = 2 : i32
    }} : ({hidden_ty}, {token_ty}, {kv_ty}, {arena_ty}) -> ({hidden_ty}, {kv_ty})
"#,
        target = target,
        custom_call_abi = abi.call.abi(),
        src = src,
        kv_src = kv_src,
        dst = dst,
        kv_dst = kv_dst,
        layer_idx = layer.layer,
        batch = abi.batch,
        ctx = abi.ctx,
        kv_dtype = abi.kv_dtype.as_mlir_dtype(),
        kv_cache_bytes = abi.kv_cache_bytes,
        hidden = abi.hidden,
        num_q_heads = M2_NUM_Q_HEADS,
        num_kv_heads = M2_NUM_KV_HEADS,
        head_dim = M2_HEAD_DIM,
        rotary_dim = M2_ROTARY_DIM,
        top_k = abi.top_k,
        expert_count = abi.expert_count,
        nvfp4_group = M2_NVFP4_GROUP,
        weight_arena_bytes = arena.total_bytes,
        weight_arena_alignment = arena.alignment,
        dense_offsets = M2DecodeLayerArenaOffsets::DENSE_ATTRS,
        input_norm = offsets.input_norm,
        post_attention_norm = offsets.post_attention_norm,
        q_proj = offsets.q_proj,
        k_proj = offsets.k_proj,
        v_proj = offsets.v_proj,
        o_proj = offsets.o_proj,
        q_norm = offsets.q_norm,
        k_norm = offsets.k_norm,
        router = offsets.router,
        router_bias = offsets.router_bias,
        w1_first_packed = offsets.w1_first_packed,
        w1_first_scale = offsets.w1_first_scale,
        w1_first_global = offsets.w1_first_global_scale,
        w1_first_input = offsets.w1_first_input_scale,
        w2_first_packed = offsets.w2_first_packed,
        w3_first_packed = offsets.w3_first_packed,
        w3_last_packed = offsets.w3_last_packed,
        missing_input_scales = layer.input_scale_missing_count(),
        expert_directory_cols = abi.expert_directory_cols,
        expert_directory_compact = compact_backend_config_value(&expert_directory),
    ))
}

fn compact_backend_config_value(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("")
        .replace('"', "'")
}

fn expert_directory_attr(layer: &M2DecodeLayerPlan) -> String {
    let rows = layer.expert_directory();
    let mut out = String::new();
    out.push_str("dense<[");
    for (row_idx, row) in rows.iter().enumerate() {
        if row_idx > 0 {
            out.push_str(", ");
        }
        out.push('[');
        for (col_idx, value) in row.as_i64_row().iter().enumerate() {
            if col_idx > 0 {
                out.push_str(", ");
            }
            out.push_str(&value.to_string());
        }
        out.push(']');
    }
    out.push_str(&format!(
        "]> : tensor<{}x{}xi64>",
        rows.len(),
        M2ExpertDirectoryEntry::COLS
    ));
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
        assert!(!mlir.contains("memref."));
        assert!(mlir.contains("weight_arena"));
        assert!(mlir.contains("rvllm.weight_entries = 191069 : i64"));
        assert!(mlir.contains("rvllm.weight_input_scales_missing = 18 : i64"));
        assert!(mlir.contains("rvllm.lowering = \"rust_xla_custom_call\""));
        assert_eq!(
            mlir.matches("call_target_name = \"rvllm.m2.decode_layer.fused_attention_nvfp4_moe\"")
                .count(),
            M2_NUM_LAYERS
        );
        assert_eq!(
            mlir.matches("target=rvllm.m2.decode_layer.fused_attention_nvfp4_moe")
                .count(),
            M2_NUM_LAYERS
        );
        assert!(mlir.contains("custom_call_abi=m2_decode_layer_v1"));
        assert!(mlir.contains("rvllm.batch = 8 : i64"));
        assert!(mlir.contains("rvllm.ctx = 2048 : i64"));
        assert!(mlir.contains("kv_dtype=i8"));
        assert!(mlir.contains("kv_cache_bytes=2080374784"));
        assert!(mlir.contains("num_q_heads=48"));
        assert!(mlir.contains("num_kv_heads=8"));
        assert!(mlir.contains("head_dim=128"));
        assert!(mlir.contains("rotary_dim=64"));
        assert!(mlir.contains("nvfp4_group=16"));
        assert!(mlir.contains("weight_arena=flat_i8_offsets"));
        assert!(mlir.contains(&format!("weight_arena_bytes={}", arena.total_bytes)));
        assert!(mlir.contains("weight_arena_alignment=128"));
        assert!(mlir.contains("weight_arena_dense_offsets=input_norm,post_attention_norm,q_proj,k_proj,v_proj,o_proj,q_norm,k_norm,router,router_bias"));
        assert!(mlir.contains("q_proj_offset="));
        assert!(mlir.contains("w1_first_packed_offset="));
        assert!(mlir.contains("expert_directory=packed_i64_offsets"));
        assert!(mlir.contains("expert_directory_cols=13"));
        assert!(mlir.contains("expert_directory_i64=dense<[[0,"));
        assert!(mlir.contains("]>:tensor<256x13xi64>"));
        assert!(mlir.contains("dispatch=fused_attention_topk_nvfp4_moe"));
        assert!(!mlir.contains("Contract body placeholder"));
    }

    #[test]
    fn rejects_prefill_shape_for_decode_graph() {
        let shape = M2GraphShape::prefill(8, 20, 2048, 1);
        let arena = arena();
        assert!(m2_decode_graph_mlir("rvllm_m2_decode", &shape, &arena).is_err());
    }
}
