module attributes {"stable_mosaic.version" = "1"} {
  func.func @main(
      %hidden: memref<8x3072xbf16>,
      %positions: memref<8xi32>,
      %kv_in: memref<33554432xi8>,
      %layer_offsets: memref<34xi32>,
      %expert_directory: memref<256x25xi32>,
      %hidden_out: memref<8x3072xbf16>,
      %kv_out: memref<33554432xi8>) attributes {
        dimension_semantics = [],
        scalar_prefetch = 0 : i64,
        scratch_operands = 0 : i64
      } {
    %c0 = arith.constant 0 : index
    %hidden_v = vector.load %hidden[%c0, %c0] : memref<8x3072xbf16>, vector<1x128xbf16>
    vector.store %hidden_v, %hidden_out[%c0, %c0] : memref<8x3072xbf16>, vector<1x128xbf16>
    %kv_v = vector.load %kv_in[%c0] : memref<33554432xi8>, vector<512xi8>
    vector.store %kv_v, %kv_out[%c0] : memref<33554432xi8>, vector<512xi8>
    return
  }
}
