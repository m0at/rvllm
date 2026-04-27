module attributes {"stable_mosaic.version" = "1"} {
  func.func @main(
      %hidden: memref<8x3072xbf16>,
      %positions: memref<8xi32>,
      %kv_in: memref<33554432xi8>,
      %layer_offsets: memref<34xi32>,
      %expert_directory: memref<256x25xi32>,
      %w1_block_t: memref<3072x128xi8>,
      %w1_row_scales: memref<128xf32>,
      %hidden_out: memref<8x3072xbf16>,
      %kv_out: memref<33554432xi8>) attributes {
        dimension_semantics = [],
        scalar_prefetch = 0 : i64,
        scratch_operands = 0 : i64,
        rvllm.int8_probe = "w1_i8_full_k_128_cols"
      } {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c384 = arith.constant 384 : index
    %c512 = arith.constant 512 : index
    %c640 = arith.constant 640 : index
    %c768 = arith.constant 768 : index
    %c896 = arith.constant 896 : index
    %c1024 = arith.constant 1024 : index
    %c1152 = arith.constant 1152 : index
    %c1280 = arith.constant 1280 : index
    %c1408 = arith.constant 1408 : index
    %c1536 = arith.constant 1536 : index
    %c1664 = arith.constant 1664 : index
    %c1792 = arith.constant 1792 : index
    %c1920 = arith.constant 1920 : index
    %c2048 = arith.constant 2048 : index
    %c2176 = arith.constant 2176 : index
    %c2304 = arith.constant 2304 : index
    %c2432 = arith.constant 2432 : index
    %c2560 = arith.constant 2560 : index
    %c2688 = arith.constant 2688 : index
    %c2816 = arith.constant 2816 : index
    %c2944 = arith.constant 2944 : index

    %hidden_v = vector.load %hidden[%c0, %c0] : memref<8x3072xbf16>, vector<1x128xbf16>
    vector.store %hidden_v, %hidden_out[%c0, %c0] : memref<8x3072xbf16>, vector<1x128xbf16>
    %kv_v = vector.load %kv_in[%c0] : memref<33554432xi8>, vector<512xi8>
    vector.store %kv_v, %kv_out[%c0] : memref<33554432xi8>, vector<512xi8>
    %zero = arith.constant dense<0.000000e+00> : vector<8x128xf32>
    %h_bf16_0 = vector.load %hidden[%c0, %c0] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_0 = arith.extf %h_bf16_0 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_0 = vector.load %w1_block_t[%c0, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_0 = arith.sitofp %w_i8_0 : vector<128x128xi8> to vector<128x128xf32>
    %acc_0 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_0, %w_f32_0, %zero : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_1 = vector.load %hidden[%c0, %c128] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_1 = arith.extf %h_bf16_1 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_1 = vector.load %w1_block_t[%c128, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_1 = arith.sitofp %w_i8_1 : vector<128x128xi8> to vector<128x128xf32>
    %acc_1 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_1, %w_f32_1, %acc_0 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_2 = vector.load %hidden[%c0, %c256] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_2 = arith.extf %h_bf16_2 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_2 = vector.load %w1_block_t[%c256, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_2 = arith.sitofp %w_i8_2 : vector<128x128xi8> to vector<128x128xf32>
    %acc_2 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_2, %w_f32_2, %acc_1 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_3 = vector.load %hidden[%c0, %c384] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_3 = arith.extf %h_bf16_3 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_3 = vector.load %w1_block_t[%c384, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_3 = arith.sitofp %w_i8_3 : vector<128x128xi8> to vector<128x128xf32>
    %acc_3 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_3, %w_f32_3, %acc_2 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_4 = vector.load %hidden[%c0, %c512] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_4 = arith.extf %h_bf16_4 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_4 = vector.load %w1_block_t[%c512, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_4 = arith.sitofp %w_i8_4 : vector<128x128xi8> to vector<128x128xf32>
    %acc_4 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_4, %w_f32_4, %acc_3 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_5 = vector.load %hidden[%c0, %c640] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_5 = arith.extf %h_bf16_5 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_5 = vector.load %w1_block_t[%c640, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_5 = arith.sitofp %w_i8_5 : vector<128x128xi8> to vector<128x128xf32>
    %acc_5 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_5, %w_f32_5, %acc_4 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_6 = vector.load %hidden[%c0, %c768] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_6 = arith.extf %h_bf16_6 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_6 = vector.load %w1_block_t[%c768, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_6 = arith.sitofp %w_i8_6 : vector<128x128xi8> to vector<128x128xf32>
    %acc_6 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_6, %w_f32_6, %acc_5 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_7 = vector.load %hidden[%c0, %c896] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_7 = arith.extf %h_bf16_7 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_7 = vector.load %w1_block_t[%c896, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_7 = arith.sitofp %w_i8_7 : vector<128x128xi8> to vector<128x128xf32>
    %acc_7 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_7, %w_f32_7, %acc_6 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_8 = vector.load %hidden[%c0, %c1024] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_8 = arith.extf %h_bf16_8 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_8 = vector.load %w1_block_t[%c1024, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_8 = arith.sitofp %w_i8_8 : vector<128x128xi8> to vector<128x128xf32>
    %acc_8 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_8, %w_f32_8, %acc_7 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_9 = vector.load %hidden[%c0, %c1152] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_9 = arith.extf %h_bf16_9 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_9 = vector.load %w1_block_t[%c1152, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_9 = arith.sitofp %w_i8_9 : vector<128x128xi8> to vector<128x128xf32>
    %acc_9 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_9, %w_f32_9, %acc_8 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_10 = vector.load %hidden[%c0, %c1280] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_10 = arith.extf %h_bf16_10 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_10 = vector.load %w1_block_t[%c1280, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_10 = arith.sitofp %w_i8_10 : vector<128x128xi8> to vector<128x128xf32>
    %acc_10 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_10, %w_f32_10, %acc_9 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_11 = vector.load %hidden[%c0, %c1408] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_11 = arith.extf %h_bf16_11 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_11 = vector.load %w1_block_t[%c1408, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_11 = arith.sitofp %w_i8_11 : vector<128x128xi8> to vector<128x128xf32>
    %acc_11 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_11, %w_f32_11, %acc_10 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_12 = vector.load %hidden[%c0, %c1536] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_12 = arith.extf %h_bf16_12 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_12 = vector.load %w1_block_t[%c1536, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_12 = arith.sitofp %w_i8_12 : vector<128x128xi8> to vector<128x128xf32>
    %acc_12 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_12, %w_f32_12, %acc_11 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_13 = vector.load %hidden[%c0, %c1664] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_13 = arith.extf %h_bf16_13 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_13 = vector.load %w1_block_t[%c1664, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_13 = arith.sitofp %w_i8_13 : vector<128x128xi8> to vector<128x128xf32>
    %acc_13 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_13, %w_f32_13, %acc_12 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_14 = vector.load %hidden[%c0, %c1792] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_14 = arith.extf %h_bf16_14 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_14 = vector.load %w1_block_t[%c1792, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_14 = arith.sitofp %w_i8_14 : vector<128x128xi8> to vector<128x128xf32>
    %acc_14 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_14, %w_f32_14, %acc_13 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_15 = vector.load %hidden[%c0, %c1920] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_15 = arith.extf %h_bf16_15 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_15 = vector.load %w1_block_t[%c1920, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_15 = arith.sitofp %w_i8_15 : vector<128x128xi8> to vector<128x128xf32>
    %acc_15 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_15, %w_f32_15, %acc_14 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_16 = vector.load %hidden[%c0, %c2048] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_16 = arith.extf %h_bf16_16 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_16 = vector.load %w1_block_t[%c2048, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_16 = arith.sitofp %w_i8_16 : vector<128x128xi8> to vector<128x128xf32>
    %acc_16 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_16, %w_f32_16, %acc_15 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_17 = vector.load %hidden[%c0, %c2176] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_17 = arith.extf %h_bf16_17 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_17 = vector.load %w1_block_t[%c2176, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_17 = arith.sitofp %w_i8_17 : vector<128x128xi8> to vector<128x128xf32>
    %acc_17 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_17, %w_f32_17, %acc_16 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_18 = vector.load %hidden[%c0, %c2304] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_18 = arith.extf %h_bf16_18 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_18 = vector.load %w1_block_t[%c2304, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_18 = arith.sitofp %w_i8_18 : vector<128x128xi8> to vector<128x128xf32>
    %acc_18 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_18, %w_f32_18, %acc_17 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_19 = vector.load %hidden[%c0, %c2432] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_19 = arith.extf %h_bf16_19 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_19 = vector.load %w1_block_t[%c2432, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_19 = arith.sitofp %w_i8_19 : vector<128x128xi8> to vector<128x128xf32>
    %acc_19 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_19, %w_f32_19, %acc_18 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_20 = vector.load %hidden[%c0, %c2560] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_20 = arith.extf %h_bf16_20 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_20 = vector.load %w1_block_t[%c2560, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_20 = arith.sitofp %w_i8_20 : vector<128x128xi8> to vector<128x128xf32>
    %acc_20 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_20, %w_f32_20, %acc_19 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_21 = vector.load %hidden[%c0, %c2688] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_21 = arith.extf %h_bf16_21 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_21 = vector.load %w1_block_t[%c2688, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_21 = arith.sitofp %w_i8_21 : vector<128x128xi8> to vector<128x128xf32>
    %acc_21 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_21, %w_f32_21, %acc_20 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_22 = vector.load %hidden[%c0, %c2816] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_22 = arith.extf %h_bf16_22 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_22 = vector.load %w1_block_t[%c2816, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_22 = arith.sitofp %w_i8_22 : vector<128x128xi8> to vector<128x128xf32>
    %acc_22 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_22, %w_f32_22, %acc_21 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>
    %h_bf16_23 = vector.load %hidden[%c0, %c2944] : memref<8x3072xbf16>, vector<8x128xbf16>
    %h_mat_23 = arith.extf %h_bf16_23 : vector<8x128xbf16> to vector<8x128xf32>
    %w_i8_23 = vector.load %w1_block_t[%c2944, %c0] : memref<3072x128xi8>, vector<128x128xi8>
    %w_f32_23 = arith.sitofp %w_i8_23 : vector<128x128xi8> to vector<128x128xf32>
    %acc_23 = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %h_mat_23, %w_f32_23, %acc_22 : vector<8x128xf32>, vector<128x128xf32> into vector<8x128xf32>

    %scale_v = vector.load %w1_row_scales[%c0] : memref<128xf32>, vector<128xf32>
    %scale_b = vector.broadcast %scale_v : vector<128xf32> to vector<8x128xf32>
    %scaled = arith.mulf %acc_23, %scale_b : vector<8x128xf32>
    %out_bf16 = arith.truncf %scaled : vector<8x128xf32> to vector<8x128xbf16>
    vector.store %out_bf16, %hidden_out[%c0, %c0] : memref<8x3072xbf16>, vector<8x128xbf16>
    return
  }
}
