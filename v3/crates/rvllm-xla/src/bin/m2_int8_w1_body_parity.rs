use std::env;
use std::fs;
use std::path::PathBuf;

use serde::Serialize;

#[cfg(feature = "tpu")]
use rvllm_xla::{
    m2_decode_layer_int8_lowered_body_mlir, tpu_custom_call_backend_config_for_body, M2GraphShape,
    PjrtClientHandle, PjrtElementType, TpuMosaicSerializedBody, TPU_CUSTOM_CALL_TARGET,
};

#[cfg(feature = "tpu")]
const B: usize = 8;
#[cfg(feature = "tpu")]
const K: usize = 3072;
#[cfg(feature = "tpu")]
const N: usize = 128;

#[derive(Serialize)]
struct ParityReport {
    schema: &'static str,
    batch: usize,
    k: usize,
    n: usize,
    pass: bool,
    mismatches: usize,
    first_mismatch: Option<Mismatch>,
}

#[derive(Serialize)]
struct Mismatch {
    row: usize,
    col: usize,
    expected_bits: u16,
    actual_bits: u16,
    expected_f32: f32,
    actual_f32: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = parse_out_path()?;
    let report = run()?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report.pass {
        Ok(())
    } else {
        Err(format!("{} bf16 mismatches", report.mismatches).into())
    }
}

fn parse_out_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut out = PathBuf::from("tpu/out/m2/m2_int8_w1_body_parity.json");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = args.next().ok_or("--out requires a path")?.into(),
            _ => return Err(format!("unknown arg {arg}").into()),
        }
    }
    Ok(out)
}

#[cfg(feature = "tpu")]
fn run() -> Result<ParityReport, Box<dyn std::error::Error>> {
    let shape = M2GraphShape::decode(B, 2048, 1);
    let body_mlir = m2_decode_layer_int8_lowered_body_mlir(&shape, 0)?;
    let body = TpuMosaicSerializedBody::from_lowered_mlir(body_mlir.as_bytes())?;
    let mlir = parity_mlir(&tpu_custom_call_backend_config_for_body(&body));

    let hidden_bits = make_hidden_bf16();
    let weights = make_weights_kx_n();
    let scales = make_scales();
    let expected = reference_bf16(&hidden_bits, &weights, &scales);

    let hidden_bytes = u16_bytes(&hidden_bits);
    let scale_bytes = f32_bytes(&scales);

    let client = PjrtClientHandle::new()?;
    let exe = client.compile(&mlir)?;
    let hidden = client.buffer_from_host(
        &hidden_bytes,
        &[B as i64, K as i64],
        PjrtElementType::BF16,
        0,
    )?;
    let weight = client.buffer_from_host(
        &i8_bytes(&weights),
        &[K as i64, N as i64],
        PjrtElementType::S8,
        0,
    )?;
    let scale = client.buffer_from_host(&scale_bytes, &[N as i64], PjrtElementType::F32, 0)?;
    let mut outputs = client.execute(&exe, &[&hidden, &weight, &scale])?;
    if outputs.len() != 1 {
        return Err(format!("expected one output, got {}", outputs.len()).into());
    }

    let mut actual_bytes = vec![0u8; B * N * 2];
    client.buffer_to_host(&outputs.remove(0), &mut actual_bytes)?;
    let actual = bytes_to_u16(&actual_bytes);
    Ok(compare(&expected, &actual))
}

#[cfg(not(feature = "tpu"))]
fn run() -> Result<ParityReport, Box<dyn std::error::Error>> {
    Err("m2_int8_w1_body_parity requires --features tpu".into())
}

#[cfg(feature = "tpu")]
fn parity_mlir(backend_config: &str) -> String {
    format!(
        r#"module attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, rvllm.kind = "m2_int8_w1_body_parity"}} {{
  func.func public @main(%hidden: tensor<{b}x{k}xbf16>, %w1_block_t: tensor<{k}x{n}xi8>, %w1_row_scales: tensor<{n}xf32>) -> tensor<{b}x{n}xbf16> {{
    %out = "stablehlo.custom_call"(%hidden, %w1_block_t, %w1_row_scales) {{
      call_target_name = "{target}",
      backend_config = "{backend_config}",
      called_computations = [],
      has_side_effect = false,
      api_version = 1 : i32,
      kernel_name = "rvllm.m2.int8_bf16_matmul",
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>]
    }} : (tensor<{b}x{k}xbf16>, tensor<{k}x{n}xi8>, tensor<{n}xf32>) -> tensor<{b}x{n}xbf16>
    return %out : tensor<{b}x{n}xbf16>
  }}
}}
"#,
        b = B,
        k = K,
        n = N,
        target = TPU_CUSTOM_CALL_TARGET,
        backend_config = backend_config,
    )
}

#[cfg(feature = "tpu")]
fn make_hidden_bf16() -> Vec<u16> {
    (0..B * K)
        .map(|idx| {
            let v = ((idx.wrapping_mul(17) % 257) as i32 - 128) as f32 / 96.0;
            f32_to_bf16_bits(v)
        })
        .collect()
}

#[cfg(feature = "tpu")]
fn make_weights_kx_n() -> Vec<i8> {
    (0..K * N)
        .map(|idx| {
            let k = idx / N;
            let n = idx % N;
            (((k * 31 + n * 17 + 13) % 255) as i32 - 127) as i8
        })
        .collect()
}

#[cfg(feature = "tpu")]
fn make_scales() -> Vec<f32> {
    (0..N)
        .map(|n| {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            sign * (0.000_25 + (n % 11) as f32 * 0.000_07)
        })
        .collect()
}

#[cfg(feature = "tpu")]
fn reference_bf16(hidden: &[u16], weights: &[i8], scales: &[f32]) -> Vec<u16> {
    let mut out = vec![0u16; B * N];
    for b in 0..B {
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += bf16_bits_to_f32(hidden[b * K + k]) * weights[k * N + n] as f32;
            }
            out[b * N + n] = f32_to_bf16_bits(acc * scales[n]);
        }
    }
    out
}

#[cfg(feature = "tpu")]
fn compare(expected: &[u16], actual: &[u16]) -> ParityReport {
    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    for (idx, (&want, &got)) in expected.iter().zip(actual.iter()).enumerate() {
        if want != got {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some(Mismatch {
                    row: idx / N,
                    col: idx % N,
                    expected_bits: want,
                    actual_bits: got,
                    expected_f32: bf16_bits_to_f32(want),
                    actual_f32: bf16_bits_to_f32(got),
                });
            }
        }
    }
    ParityReport {
        schema: "rvllm.m2.int8_w1_body_parity.v1",
        batch: B,
        k: K,
        n: N,
        pass: mismatches == 0,
        mismatches,
        first_mismatch,
    }
}

#[cfg(feature = "tpu")]
fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

#[cfg(feature = "tpu")]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(feature = "tpu")]
fn u16_bytes(xs: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 2);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[cfg(feature = "tpu")]
fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

#[cfg(feature = "tpu")]
fn i8_bytes(xs: &[i8]) -> Vec<u8> {
    xs.iter().map(|&x| x as u8).collect()
}

#[cfg(feature = "tpu")]
fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
