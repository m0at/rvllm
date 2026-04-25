#[cfg(not(feature = "tpu"))]
fn main() {
    eprintln!("m2_prefill_smoke requires --features tpu");
    std::process::exit(1);
}

#[cfg(feature = "tpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;
    use std::path::PathBuf;

    use rvllm_core::{BatchedPrefillPlan, PrefillRequest, ReqId, TokenId};
    use rvllm_fused::{M2PrefillKvDType, M2PrefillScanShape};
    use rvllm_xla::{load_artifact, make_m2_prefill_inputs, PjrtClientHandle};

    let mut args = env::args().skip(1);
    let artifact_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "tpu/out/m2/prefill_scan_artifact".to_string()),
    );
    let prompt_len = parse_arg(args.next(), 20, "prompt_len")?;
    let ctx = parse_arg(args.next(), 2048, "ctx")?;

    let prompt: Vec<TokenId> = (0..prompt_len).map(|i| TokenId((10 + i) as u32)).collect();
    let plan = BatchedPrefillPlan::from_requests(&[PrefillRequest {
        req_id: ReqId(1),
        prompt_tokens: &prompt,
        max_blocks_per_seq: ((ctx + 31) / 32) as u32,
        block_size: 32,
    }])?;
    let shape = M2PrefillScanShape {
        batch: 1,
        prompt_len,
        hidden: 3072,
        ctx,
        num_layers: 62,
        num_kv_heads: 8,
        head_dim: 128,
        kv_dtype: M2PrefillKvDType::Int8,
    };
    let (artifact, mlir, compile_options) = load_artifact(&artifact_dir)?;
    let host_inputs = make_m2_prefill_inputs(&plan, shape)?;

    let mut client = PjrtClientHandle::new()?;
    if let Some(opts) = compile_options {
        client.set_compile_options(opts)?;
    }
    let exe = client.compile_bytes(&mlir)?;
    let buffers = host_inputs
        .iter()
        .map(|input| client.buffer_from_host(&input.bytes, &input.shape, input.dtype, 0))
        .collect::<rvllm_core::Result<Vec<_>>>()?;
    let refs = buffers.iter().collect::<Vec<_>>();
    let outputs = client.execute(&exe, &refs)?;
    eprintln!(
        "m2 prefill smoke executed: artifact_inputs={} host_inputs={} outputs={}",
        artifact.inputs.len(),
        host_inputs.len(),
        outputs.len()
    );
    Ok(())
}

#[cfg(feature = "tpu")]
fn parse_arg(arg: Option<String>, default: usize, name: &'static str) -> Result<usize, String> {
    match arg {
        Some(s) => s
            .parse::<usize>()
            .map_err(|e| format!("{name}: expected usize, got {s:?}: {e}")),
        None => Ok(default),
    }
}
