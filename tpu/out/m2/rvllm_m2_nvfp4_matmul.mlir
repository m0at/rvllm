module attributes {rvllm.kind = "m2_nvfp4_mosaic"} {
  func.func @rvllm_m2_nvfp4_matmul(
      %x: memref<8x3072xbf16>,
      %packed: memref<1536x1536xi8>,
      %scales: memref<1536x192xi8>,
      %global_scale: memref<f32>,
      %out: memref<8x1536xbf16>) attributes {
        rvllm.signature = "x_bf16,packed_u8,scale_fp8,global_f32,out_bf16",
        rvllm.layout = "row_major",
        rvllm.nvfp4_group = 16 : i64,
        rvllm.tile = "BM=8,BN=512,BK=1024",
        rvllm.tiles = "M=1,N=3,K=3",
        rvllm.vmem_working_set_bytes = 1384448 : i64,
        rvllm.lowering = "mosaic_custom_call",
        rvllm.lowering_plan = "tpu.load packed/scales -> unpack nibbles -> fp8/fp4 decode in VMEM/registers -> bf16 RHS tile -> tpu.matmul -> tpu.store"
      } {
    // Tile contract:
    //   x tile bf16 bytes: 16384
    //   packed NVFP4 RHS tile bytes: 262144
    //   FP8 scale tile bytes: 32768
    //   decoded bf16 RHS tile bytes: 1048576
    //   f32 accumulator bytes: 16384
    //   output tile bf16 bytes: 8192
    //
    // Lowering outline for the real Mosaic body:
    //   for m_tile, n_tile:
    //     acc = f32[BM, BN]
    //     for k_tile:
    //       tpu.load x[BM, BK] as bf16
    //       tpu.load packed[BN, BK/2] and scales[BN, BK/16]
    //       tpu.unpack_subelements packed uint8 -> low/high FP4 E2M1 nibbles
    //       decode FP8 E4M3 scales and multiply by global_scale
    //       materialize only this RHS bf16[BN, BK] tile in VMEM/registers
    //       tpu.matmul x[BM, BK] * rhs[BN, BK]^T into acc
    //     tpu.store acc cast bf16 to out[BM, BN]
    return
  }
}
