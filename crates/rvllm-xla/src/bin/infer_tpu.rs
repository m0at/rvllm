#[cfg(not(feature = "tpu"))]
fn main() {
    eprintln!("ERROR: infer-tpu requires the 'tpu' feature.");
    eprintln!("Build with: cargo run --release -p rvllm-xla --features tpu --bin infer-tpu");
    std::process::exit(1);
}

#[cfg(feature = "tpu")]
fn main() {
    tpu_main::run();
}

#[cfg(feature = "tpu")]
mod tpu_main {
    use std::path::PathBuf;
    use std::time::Instant;

    use clap::Parser;
    use rvllm_xla::client::PjrtClientHandle;
    use rvllm_xla::ffi::PjrtElementType;

    #[derive(Parser)]
    #[command(name = "infer-tpu", about = "Run rvLLM inference on TPU via PJRT")]
    struct Args {
        #[arg(long, default_value = "tpu/out/")]
        mlir_dir: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        #[arg(long, default_value = "Hello, world")]
        prompt: String,
    }

    // Hardcoded for the MLIR modules we have (Llama 3.1 8B shapes)
    const HIDDEN: usize = 4096;
    const NUM_HEADS: usize = 32;
    const NUM_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const INTERMEDIATE: usize = 14336;
    const NUM_LAYERS: usize = 32;
    const VOCAB: usize = 128256;
    const BLOCK_SIZE: usize = 16;
    const NUM_BLOCKS: usize = 1024;
    const MAX_BLOCKS_PER_SEQ: usize = 4; // 4 blocks * 16 = 64 token context
    const BATCH: usize = 8; // padded batch from layer_decode MLIR
    const EMBED_BATCH: usize = 128; // padded batch from embedding MLIR

    pub fn run() {
        let args = Args::parse();

        // Load compile options
        let opts_path = args.mlir_dir.join("compile_options.pb");
        let compile_opts = if opts_path.exists() {
            Some(std::fs::read(&opts_path).unwrap())
        } else {
            None
        };

        let mut client = PjrtClientHandle::new().unwrap_or_else(|e| {
            eprintln!("FATAL: PJRT init failed: {e}");
            std::process::exit(1);
        });
        eprintln!("PJRT client: {} device(s)", client.num_devices());

        if let Some(opts) = compile_opts {
            client.set_compile_options(opts);
        }

        // Compile the fused full-step module (embed + 32 layers + head + argmax)
        eprintln!("compiling fused decode step...");
        let t0 = Instant::now();
        let step_mlir = std::fs::read_to_string(args.mlir_dir.join("full_decode_step.mlir"))
            .expect("full_decode_step.mlir");
        let step_exe = client.compile(&step_mlir).expect("compile full_decode_step");
        eprintln!("compiled in {:.1}s", t0.elapsed().as_secs_f32());

        // Load model weights from safetensors
        eprintln!("loading weights from {:?}...", args.model_dir);
        let t0 = Instant::now();
        let weights = load_weights(&client, &args.model_dir);
        eprintln!("loaded {NUM_LAYERS} layer weight sets + embed/head in {:.1}s",
            t0.elapsed().as_secs_f32());

        // If prompt looks like comma-separated numbers, use as token IDs directly
        let prompt_ids: Vec<i32> = if args.prompt.contains(',') && args.prompt.chars().all(|c| c.is_ascii_digit() || c == ',' || c == ' ') {
            args.prompt.split(',').filter_map(|s| s.trim().parse().ok()).collect()
        } else {
            args.prompt.bytes().map(|b| b as i32).collect()
        };
        eprintln!("prompt: {} tokens {:?}", prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(10)]);

        // Stacked KV caches: [NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
        let kv_stacked_bytes = NUM_LAYERS * NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * 2;
        let kv_stacked_shape: Vec<i64> = vec![NUM_LAYERS as i64, NUM_BLOCKS as i64,
            BLOCK_SIZE as i64, NUM_KV_HEADS as i64, HEAD_DIM as i64];
        let zero_kv = vec![0u8; kv_stacked_bytes];
        let mut k_cache_stacked = client.buffer_from_host(
            &zero_kv, &kv_stacked_shape, PjrtElementType::BF16, 0).unwrap();
        let mut v_cache_stacked = client.buffer_from_host(
            &zero_kv, &kv_stacked_shape, PjrtElementType::BF16, 0).unwrap();

        // Block tables
        let mut bt_host = vec![0i32; BATCH * MAX_BLOCKS_PER_SEQ];
        for seq in 0..BATCH {
            for b in 0..MAX_BLOCKS_PER_SEQ {
                bt_host[seq * MAX_BLOCKS_PER_SEQ + b] = (seq * MAX_BLOCKS_PER_SEQ + b) as i32;
            }
        }
        let bt_buf = client.buffer_from_host(
            bytemuck::cast_slice(&bt_host),
            &[BATCH as i64, MAX_BLOCKS_PER_SEQ as i64],
            PjrtElementType::S32, 0,
        ).unwrap();

        let mut generated = Vec::new();
        let mut context_len: i32 = 0;
        let mut ttft_ns: Option<u128> = None;
        let mut decode_start: Option<Instant> = None;

        eprintln!("--- inference ---");
        let prompt_start = Instant::now();
        let total_steps = prompt_ids.len() + args.max_tokens;
        for step in 0..total_steps {
            let token_id = if step < prompt_ids.len() {
                prompt_ids[step]
            } else {
                *generated.last().unwrap_or(&0)
            };

            let step_start = Instant::now();

            // Metadata buffers (small HtoD, unavoidable)
            let mut tok_batch = vec![0i32; BATCH];
            tok_batch[0] = token_id;
            let tok_buf = client.buffer_from_host(
                bytemuck::cast_slice(&tok_batch), &[BATCH as i64],
                PjrtElementType::S32, 0).unwrap();

            let pos_host: Vec<i32> = (0..BATCH).map(|_| context_len).collect();
            // slot = block_index * block_size + offset_within_block
            // For seq i: block = context_len / BLOCK_SIZE, offset = context_len % BLOCK_SIZE
            // Slot = block_tables[i][block] * BLOCK_SIZE + offset
            // Simplified: each seq gets its own block range starting at i * MAX_BLOCKS_PER_SEQ
            let slot_host: Vec<i32> = (0..BATCH).map(|i| {
                let block = context_len / BLOCK_SIZE as i32;
                let offset = context_len % BLOCK_SIZE as i32;
                let block_idx = (i as i32) * MAX_BLOCKS_PER_SEQ as i32 + block;
                block_idx * BLOCK_SIZE as i32 + offset
            }).collect();
            let ctx_host: Vec<i32> = (0..BATCH).map(|_| context_len + 1).collect();
            let pos_buf = client.buffer_from_host(
                bytemuck::cast_slice(&pos_host), &[BATCH as i64],
                PjrtElementType::S32, 0).unwrap();
            let slot_buf = client.buffer_from_host(
                bytemuck::cast_slice(&slot_host), &[BATCH as i64],
                PjrtElementType::S32, 0).unwrap();
            let ctx_buf = client.buffer_from_host(
                bytemuck::cast_slice(&ctx_host), &[BATCH as i64],
                PjrtElementType::S32, 0).unwrap();

            // 18 inputs: 10 globals + 6 stacked weights + 2 stacked KV caches
            let t_exec = Instant::now();
            let inputs: Vec<&rvllm_xla::client::PjrtBufferHandle> = vec![
                &tok_buf,
                &weights.embedding,
                &weights.final_norm,
                &weights.lm_head_bf16,
                &weights.rope_cos,
                &weights.rope_sin,
                &pos_buf,
                &slot_buf,
                &bt_buf,
                &ctx_buf,
                &weights.all_n1g,
                &weights.all_qkv,
                &weights.all_o,
                &weights.all_n2g,
                &weights.all_gu,
                &weights.all_down,
                &k_cache_stacked,
                &v_cache_stacked,
            ];

            // ONE execute: embed + scan(32 layers) + head + argmax
            let step_out = client.execute(&step_exe, &inputs).unwrap();
            let exec_us = t_exec.elapsed().as_micros();

            // Output: (token_ids, stacked_k_caches, stacked_v_caches)
            let mut outs = step_out.into_iter();
            let token_buf = outs.next().unwrap();
            k_cache_stacked = outs.next().unwrap();
            v_cache_stacked = outs.next().unwrap();

            // Only DtoH: 32 bytes of token IDs
            let t_dtoh = Instant::now();
            let mut token_bytes = vec![0u8; BATCH * 4];
            client.buffer_to_host(&token_buf, &mut token_bytes).unwrap();
            let dtoh_us = t_dtoh.elapsed().as_micros();

            let tokens: &[i32] = bytemuck::cast_slice(&token_bytes);
            let sampled = tokens[0];
            let step_us = step_start.elapsed().as_micros();

            if step >= prompt_ids.len() && step < prompt_ids.len() + 3 {
                let ds = step - prompt_ids.len();
                eprintln!("[PROFILE] decode step {} total={}us exec={}us dtoh={}us",
                    ds, step_us, exec_us, dtoh_us);
            }

            context_len += 1;

            if step < prompt_ids.len() {
                eprint!(".");
                if step == prompt_ids.len() - 1 {
                    use std::io::Write;
                    std::io::stderr().flush().ok();
                    ttft_ns = Some(prompt_start.elapsed().as_nanos());
                    decode_start = Some(Instant::now());
                    eprintln!("\nTTFT: {:.2}ms ({} prompt tokens)",
                        ttft_ns.unwrap() as f64 / 1_000_000.0, prompt_ids.len());
                    std::io::stderr().flush().ok();
                }
            } else {
                generated.push(sampled);
                if sampled == 1 || sampled == 2 || sampled == 128001 { // EOS
                    eprintln!("[EOS at step {}]", step);
                    break;
                }
                eprint!("[{}]", sampled);
            }
        }

        let total_elapsed = prompt_start.elapsed();
        let decode_elapsed = decode_start.map(|s| s.elapsed());

        eprintln!();
        eprintln!("=== Results ===");
        eprintln!("prompt tokens:    {}", prompt_ids.len());
        eprintln!("generated tokens: {}", generated.len());
        if let Some(ttft) = ttft_ns {
            eprintln!("TTFT:             {:.2}ms", ttft as f64 / 1_000_000.0);
        }
        if let Some(dt) = decode_elapsed {
            let toks = generated.len();
            if toks > 0 {
                let tok_s = toks as f64 / dt.as_secs_f64();
                let ms_per_tok = dt.as_secs_f64() * 1000.0 / toks as f64;
                eprintln!("decode tok/s:     {:.1}", tok_s);
                eprintln!("ms/token:         {:.1}", ms_per_tok);
            }
        }
        eprintln!("total time:       {:.2}s", total_elapsed.as_secs_f64());
        eprintln!("generated:        {:?}", &generated[..generated.len().min(20)]);
    }

    struct ModelWeights {
        embedding: rvllm_xla::client::PjrtBufferHandle,
        final_norm: rvllm_xla::client::PjrtBufferHandle,
        lm_head_bf16: rvllm_xla::client::PjrtBufferHandle,
        rope_cos: rvllm_xla::client::PjrtBufferHandle,
        rope_sin: rvllm_xla::client::PjrtBufferHandle,
        // Stacked [NUM_LAYERS, ...] tensors for scan
        all_n1g: rvllm_xla::client::PjrtBufferHandle,
        all_qkv: rvllm_xla::client::PjrtBufferHandle,
        all_o: rvllm_xla::client::PjrtBufferHandle,
        all_n2g: rvllm_xla::client::PjrtBufferHandle,
        all_gu: rvllm_xla::client::PjrtBufferHandle,
        all_down: rvllm_xla::client::PjrtBufferHandle,
    }

    fn load_weights(client: &PjrtClientHandle, model_dir: &PathBuf) -> ModelWeights {
        use std::collections::BTreeMap;

        let idx_path = model_dir.join("model.safetensors.index.json");
        let single_path = model_dir.join("model.safetensors");

        // Find all safetensors files
        let shard_paths: Vec<PathBuf> = if idx_path.exists() {
            let idx: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(&idx_path).unwrap()
            ).unwrap();
            let wm = idx["weight_map"].as_object().unwrap();
            let mut shards: Vec<String> = wm.values()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<std::collections::HashSet<_>>()
                .into_iter().collect();
            shards.sort();
            shards.iter().map(|s| model_dir.join(s)).collect()
        } else if single_path.exists() {
            vec![single_path]
        } else {
            panic!("no safetensors found in {:?}", model_dir);
        };

        // Build tensor index
        let mut tensor_data: BTreeMap<String, (Vec<usize>, Vec<u8>, String)> = BTreeMap::new();
        for sp in &shard_paths {
            let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(sp).unwrap()).unwrap() };
            let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&mmap[8..8+header_len]).unwrap();
            let data_start = 8 + header_len;
            for (name, info) in header.as_object().unwrap() {
                if name == "__metadata__" { continue; }
                let shape: Vec<usize> = info["shape"].as_array().unwrap()
                    .iter().map(|v| v.as_u64().unwrap() as usize).collect();
                let dtype = info["dtype"].as_str().unwrap().to_string();
                let offsets = info["data_offsets"].as_array().unwrap();
                let start = offsets[0].as_u64().unwrap() as usize;
                let end = offsets[1].as_u64().unwrap() as usize;
                let bytes = mmap[data_start + start..data_start + end].to_vec();
                tensor_data.insert(name.clone(), (shape, bytes, dtype));
            }
        }

        eprintln!("  {} tensors indexed", tensor_data.len());

        let upload_bf16 = |name: &str, shape: &[i64]| -> rvllm_xla::client::PjrtBufferHandle {
            let (_, bytes, dtype) = tensor_data.get(name)
                .unwrap_or_else(|| panic!("missing tensor: {name}"));
            let bf16_bytes = match dtype.as_str() {
                "BF16" | "bf16" => bytes.clone(),
                "F16" | "f16" => f16_bytes_to_bf16(bytes),
                "F32" | "f32" | "F32" => f32_bytes_to_bf16(bytes),
                other => panic!("unsupported dtype {other} for {name}"),
            };
            client.buffer_from_host(&bf16_bytes, shape, PjrtElementType::BF16, 0)
                .unwrap_or_else(|e| panic!("upload {name}: {e}"))
        };

        let upload_f32 = |name: &str, shape: &[i64]| -> rvllm_xla::client::PjrtBufferHandle {
            let (_, bytes, dtype) = tensor_data.get(name)
                .unwrap_or_else(|| panic!("missing tensor: {name}"));
            let f32_bytes = match dtype.as_str() {
                "F32" | "f32" => bytes.clone(),
                "BF16" | "bf16" => bf16_bytes_to_f32(bytes),
                "F16" | "f16" => f16_bytes_to_f32(bytes),
                other => panic!("unsupported dtype {other} for {name}"),
            };
            client.buffer_from_host(&f32_bytes, shape, PjrtElementType::F32, 0)
                .unwrap_or_else(|e| panic!("upload {name}: {e}"))
        };

        let upload_f16 = |name: &str, shape: &[i64]| -> rvllm_xla::client::PjrtBufferHandle {
            let (_, bytes, dtype) = tensor_data.get(name)
                .unwrap_or_else(|| panic!("missing tensor: {name}"));
            let f16_bytes = match dtype.as_str() {
                "F16" | "f16" => bytes.clone(),
                "BF16" | "bf16" => bf16_bytes_to_f16(bytes),
                "F32" | "f32" => f32_bytes_to_f16(bytes),
                other => panic!("unsupported dtype {other} for {name}"),
            };
            client.buffer_from_host(&f16_bytes, shape, PjrtElementType::F16, 0)
                .unwrap_or_else(|e| panic!("upload {name}: {e}"))
        };

        // Embedding [vocab, hidden] bf16 (fused step uses bf16)
        let embedding = upload_bf16("model.embed_tokens.weight",
            &[VOCAB as i64, HIDDEN as i64]);

        // Final norm [hidden] bf16
        let final_norm = upload_bf16("model.norm.weight", &[HIDDEN as i64]);

        // LM head [vocab, hidden] bf16
        let lm_head_bf16 = if tensor_data.contains_key("lm_head.weight") {
            upload_bf16("lm_head.weight", &[VOCAB as i64, HIDDEN as i64])
        } else {
            upload_bf16("model.embed_tokens.weight", &[VOCAB as i64, HIDDEN as i64])
        };

        // RoPE cos/sin [max_pos, head_dim/2] f32
        let rope_cos = precompute_rope_cos(client, 10000.0);
        let rope_sin = precompute_rope_sin(client, 10000.0);

        // Stack per-layer weights into [NUM_LAYERS, ...] tensors for scan
        let qkv_dim = NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM;
        let mut all_n1g_bytes = Vec::new();
        let mut all_qkv_bytes = Vec::new();
        let mut all_o_bytes = Vec::new();
        let mut all_n2g_bytes = Vec::new();
        let mut all_gu_bytes = Vec::new();
        let mut all_down_bytes = Vec::new();

        for l in 0..NUM_LAYERS {
            let ln = |s: &str| format!("model.layers.{l}.{s}");

            let (_, n1g) = get_bf16(&tensor_data, &ln("input_layernorm.weight"));
            all_n1g_bytes.extend_from_slice(&n1g);

            let q_name = ln("self_attn.q_proj.weight");
            let k_name = ln("self_attn.k_proj.weight");
            let v_name = ln("self_attn.v_proj.weight");
            let qkv = concat_and_transpose_bf16(&tensor_data, &[&q_name, &k_name, &v_name], HIDDEN);
            all_qkv_bytes.extend_from_slice(&qkv);

            let o = transpose_bf16(&tensor_data, &ln("self_attn.o_proj.weight"));
            all_o_bytes.extend_from_slice(&o);

            let (_, n2g) = get_bf16(&tensor_data, &ln("post_attention_layernorm.weight"));
            all_n2g_bytes.extend_from_slice(&n2g);

            let g_name = ln("mlp.gate_proj.weight");
            let u_name = ln("mlp.up_proj.weight");
            let gu = concat_and_transpose_bf16(&tensor_data, &[&g_name, &u_name], HIDDEN);
            all_gu_bytes.extend_from_slice(&gu);

            let d = transpose_bf16(&tensor_data, &ln("mlp.down_proj.weight"));
            all_down_bytes.extend_from_slice(&d);

            if l == 0 || l == NUM_LAYERS - 1 {
                eprintln!("  layer {l} stacked");
            } else if l == 1 {
                eprintln!("  ...");
            }
        }

        let nl = NUM_LAYERS as i64;
        let h = HIDDEN as i64;
        let qd = qkv_dim as i64;
        let inter = INTERMEDIATE as i64;

        let all_n1g = client.buffer_from_host(&all_n1g_bytes, &[nl, h], PjrtElementType::BF16, 0).unwrap();
        let all_qkv = client.buffer_from_host(&all_qkv_bytes, &[nl, h, qd], PjrtElementType::BF16, 0).unwrap();
        let all_o = client.buffer_from_host(&all_o_bytes, &[nl, h, h], PjrtElementType::BF16, 0).unwrap();
        let all_n2g = client.buffer_from_host(&all_n2g_bytes, &[nl, h], PjrtElementType::BF16, 0).unwrap();
        let all_gu = client.buffer_from_host(&all_gu_bytes, &[nl, h, 2 * inter], PjrtElementType::BF16, 0).unwrap();
        let all_down = client.buffer_from_host(&all_down_bytes, &[nl, inter, h], PjrtElementType::BF16, 0).unwrap();

        ModelWeights {
            embedding,
            final_norm,
            lm_head_bf16,
            rope_cos,
            rope_sin,
            all_n1g,
            all_qkv,
            all_o,
            all_n2g,
            all_gu,
            all_down,
        }
    }

    fn get_bf16(
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        name: &str,
    ) -> (Vec<usize>, Vec<u8>) {
        let (shape, bytes, dtype) = tensors.get(name)
            .unwrap_or_else(|| panic!("missing tensor: {name}"));
        let bf16 = match dtype.as_str() {
            "BF16" | "bf16" => bytes.clone(),
            "F16" | "f16" => f16_bytes_to_bf16(bytes),
            "F32" | "f32" => f32_bytes_to_bf16(bytes),
            other => panic!("unsupported dtype {other} for {name}"),
        };
        (shape.clone(), bf16)
    }

    fn transpose_2d_bf16(data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        let mut out = vec![0u8; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                let src = (r * cols + c) * 2;
                let dst = (c * rows + r) * 2;
                out[dst] = data[src];
                out[dst + 1] = data[src + 1];
            }
        }
        out
    }

    fn transpose_bf16(
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        name: &str,
    ) -> Vec<u8> {
        let (shape, bf16) = get_bf16(tensors, name);
        assert_eq!(shape.len(), 2, "transpose requires 2D tensor: {name}");
        transpose_2d_bf16(&bf16, shape[0], shape[1])
    }

    fn concat_and_transpose_bf16(
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        names: &[&str],
        inner_dim: usize,
    ) -> Vec<u8> {
        // Each tensor is [out_dim, inner_dim]. Concat along out_dim, then
        // transpose to [inner_dim, total_out_dim].
        let mut total_out = 0usize;
        let mut parts = Vec::new();
        for name in names {
            let (shape, bf16) = get_bf16(tensors, name);
            assert_eq!(shape.len(), 2);
            assert_eq!(shape[1], inner_dim, "inner dim mismatch for {name}");
            total_out += shape[0];
            parts.push((shape[0], bf16));
        }
        // Concat [total_out, inner_dim]
        let mut cat = Vec::with_capacity(total_out * inner_dim * 2);
        for (_, data) in &parts {
            cat.extend_from_slice(data);
        }
        transpose_2d_bf16(&cat, total_out, inner_dim)
    }

    fn precompute_rope_cos(client: &PjrtClientHandle, theta: f32) -> rvllm_xla::client::PjrtBufferHandle {
        let half = HEAD_DIM / 2;
        let max_pos = 4096;
        let mut data = vec![0f32; max_pos * half];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / HEAD_DIM as f32);
                data[pos * half + i] = (pos as f32 * freq).cos();
            }
        }
        client.buffer_from_host(
            bytemuck::cast_slice(&data),
            &[max_pos as i64, half as i64],
            PjrtElementType::F32, 0,
        ).unwrap()
    }

    fn precompute_rope_sin(client: &PjrtClientHandle, theta: f32) -> rvllm_xla::client::PjrtBufferHandle {
        let half = HEAD_DIM / 2;
        let max_pos = 4096;
        let mut data = vec![0f32; max_pos * half];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / HEAD_DIM as f32);
                data[pos * half + i] = (pos as f32 * freq).sin();
            }
        }
        client.buffer_from_host(
            bytemuck::cast_slice(&data),
            &[max_pos as i64, half as i64],
            PjrtElementType::F32, 0,
        ).unwrap()
    }

    fn f32_bytes_to_bf16(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 4;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let v = f32::from_le_bytes(bytes[4*i..4*i+4].try_into().unwrap());
            out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        out
    }

    fn f32_bytes_to_f16(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 4;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let v = f32::from_le_bytes(bytes[4*i..4*i+4].try_into().unwrap());
            out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        out
    }

    fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 2;
        let mut out = Vec::with_capacity(n * 4);
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[2*i], bytes[2*i+1]]);
            let v = half::bf16::from_bits(bits).to_f32();
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn bf16_bytes_to_f16(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 2;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[2*i], bytes[2*i+1]]);
            let v = half::bf16::from_bits(bits).to_f32();
            out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        out
    }

    fn f16_bytes_to_bf16(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 2;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[2*i], bytes[2*i+1]]);
            let v = half::f16::from_bits(bits).to_f32();
            out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        out
    }

    fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 2;
        let mut out = Vec::with_capacity(n * 4);
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[2*i], bytes[2*i+1]]);
            let v = half::f16::from_bits(bits).to_f32();
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn f32_bytes_to_bf16_bytes(bytes: &[u8]) -> Vec<u8> {
        f32_bytes_to_bf16(bytes)
    }

    fn bf16_bytes_to_f16_bytes(bytes: &[u8]) -> Vec<u8> {
        bf16_bytes_to_f16(bytes)
    }

    fn f16_to_f32_val(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    use std::collections::BTreeMap;
}
