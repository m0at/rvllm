use std::env;
use std::fs;
use std::path::PathBuf;

use serde::Serialize;

#[cfg(feature = "tpu")]
use rvllm_fused::{int8_matmul_ref, M2Nvfp4MatmulShape};
#[cfg(feature = "tpu")]
use rvllm_xla::{
    tpu_custom_call_backend_config_for_body, PjrtClientHandle, PjrtElementType,
    TpuMosaicSerializedBody, TPU_CUSTOM_CALL_TARGET,
};

#[cfg(feature = "tpu")]
const B: usize = 8;
#[cfg(feature = "tpu")]
const K: usize = 64;
#[cfg(feature = "tpu")]
const N: usize = 16;

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
    let body_mlir = parity_body_mlir();
    let body = TpuMosaicSerializedBody::from_lowered_mlir(body_mlir.as_bytes())?;
    let mlir = parity_mlir(&tpu_custom_call_backend_config_for_body(&body));

    let hidden_bits = make_hidden_bf16();
    let weights_nx_k = make_weights_nx_k();
    let weights_kx_n = transpose_nx_k_to_kx_n(&weights_nx_k);
    let scales = make_scales();
    let expected = reference_bf16(&hidden_bits, &weights_nx_k, &scales)?;

    let hidden_bytes = u16_bytes(&hidden_bits);
    let scale_bytes = f32_bytes(&scales);

    let client = PjrtClientHandle::new()?;
    let num_devices = client.num_devices();
    let exe = client.compile(&mlir)?;
    let weight_bytes = i8_bytes(&weights_kx_n);
    let hidden_bufs = (0..num_devices)
        .map(|device| {
            client.buffer_from_host(
                &hidden_bytes,
                &[B as i64, K as i64],
                PjrtElementType::BF16,
                device,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weight_bufs = (0..num_devices)
        .map(|device| {
            client.buffer_from_host(
                &weight_bytes,
                &[K as i64, N as i64],
                PjrtElementType::S8,
                device,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scale_bufs = (0..num_devices)
        .map(|device| {
            client.buffer_from_host(&scale_bytes, &[N as i64], PjrtElementType::F32, device)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let per_device_inputs = (0..num_devices)
        .map(|device| {
            vec![
                &hidden_bufs[device],
                &weight_bufs[device],
                &scale_bufs[device],
            ]
        })
        .collect::<Vec<_>>();
    let mut outputs_by_device = client.execute_partitioned(&exe, &per_device_inputs)?;
    if outputs_by_device.is_empty() {
        return Err("partitioned execute returned no device outputs".into());
    }
    let mut outputs = outputs_by_device.remove(0);
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
fn parity_body_mlir() -> String {
    let mut constants = String::new();
    for i in 1..K {
        constants.push_str(&format!("    %c{i} = arith.constant {i} : index\n"));
    }
    let mut acc_init = String::new();
    let mut acc = Vec::with_capacity(B);
    for b in 0..B {
        let name = format!("%acc_{b}_0");
        acc_init.push_str(&format!(
            "    {name} = arith.constant dense<0.000000e+00> : vector<{n}xf32>\n",
            n = N
        ));
        acc.push(name);
    }
    let mut hidden_rows = String::new();
    for b in 0..B {
        let b_idx = idx(b);
        hidden_rows.push_str(&format!(
            r#"    %h_row_2d_{b} = vector.load %hidden[{b_idx}, %c0] : memref<{batch}x{k_total}xbf16>, vector<1x{k_total}xbf16>
    %h_row_{b} = vector.shape_cast %h_row_2d_{b} : vector<1x{k_total}xbf16> to vector<{k_total}xbf16>
    %h_row_f32_{b} = arith.extf %h_row_{b} : vector<{k_total}xbf16> to vector<{k_total}xf32>
"#,
            b = b,
            b_idx = b_idx,
            batch = B,
            k_total = K
        ));
    }
    let mut body = String::new();
    for k in 0..K {
        let k_idx = idx(k);
        body.push_str(&format!(
            r#"    %w_i8_2d_{k} = vector.load %w1_block_t[{k_idx}, %c0] : memref<{k_total}x{n}xi8>, vector<1x{n}xi8>
    %w_i8_{k} = vector.shape_cast %w_i8_2d_{k} : vector<1x{n}xi8> to vector<{n}xi8>
    %w_f32_{k} = arith.sitofp %w_i8_{k} : vector<{n}xi8> to vector<{n}xf32>
    %w_scaled_{k} = arith.mulf %w_f32_{k}, %scale_v : vector<{n}xf32>
"#,
            k = k,
            k_idx = k_idx,
            k_total = K,
            n = N
        ));
        for b in 0..B {
            let next = format!("%acc_{b}_{}", k + 1);
            body.push_str(&format!(
                r#"    %h_f32_{b}_{k} = vector.extract %h_row_f32_{b}[{k}] : f32 from vector<{k_total}xf32>
    %h_vec_{b}_{k} = vector.broadcast %h_f32_{b}_{k} : f32 to vector<{n}xf32>
    %prod_{b}_{k} = arith.mulf %h_vec_{b}_{k}, %w_scaled_{k} : vector<{n}xf32>
    {next} = arith.addf {prev}, %prod_{b}_{k} : vector<{n}xf32>
"#,
                b = b,
                k = k,
                k_total = K,
                n = N,
                next = next,
                prev = acc[b]
            ));
            acc[b] = next;
        }
    }
    let mut stores = String::new();
    for (b, acc_name) in acc.iter().enumerate() {
        let b_idx = idx(b);
        stores.push_str(&format!(
            r#"    %out_{b} = arith.truncf {acc_name} : vector<{n}xf32> to vector<{n}xbf16>
    %out_2d_{b} = vector.shape_cast %out_{b} : vector<{n}xbf16> to vector<1x{n}xbf16>
    vector.store %out_2d_{b}, %hidden_out[{b_idx}, %c0] : memref<{batch}x{n}xbf16>, vector<1x{n}xbf16>
"#,
            b = b,
            acc_name = acc_name,
            n = N,
            b_idx = b_idx,
            batch = B
        ));
    }

    format!(
        r#"module attributes {{"stable_mosaic.version" = "1"}} {{
  func.func @main(
      %hidden: memref<{b}x{k}xbf16>,
      %w1_block_t: memref<{k}x{n}xi8>,
      %w1_row_scales: memref<{n}xf32>,
      %hidden_out: memref<{b}x{n}xbf16>) attributes {{
        dimension_semantics = [],
        scalar_prefetch = 0 : i64,
        scratch_operands = 0 : i64,
        rvllm.int8_probe = "fixed_8x16x64"
      }} {{
    %c0 = arith.constant 0 : index
{constants}
    %scale_v = vector.load %w1_row_scales[%c0] : memref<{n}xf32>, vector<{n}xf32>
{acc_init}
{hidden_rows}
{body}
{stores}
    return
  }}
}}
"#,
        b = B,
        k = K,
        n = N,
        constants = constants,
        acc_init = acc_init,
        hidden_rows = hidden_rows,
        body = body,
        stores = stores
    )
}

#[cfg(feature = "tpu")]
fn idx(i: usize) -> String {
    if i == 0 {
        "%c0".to_string()
    } else {
        format!("%c{i}")
    }
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
fn make_weights_nx_k() -> Vec<i8> {
    (0..N * K)
        .map(|idx| {
            let n = idx / K;
            let k = idx % K;
            (((k * 31 + n * 17 + 13) % 255) as i32 - 127) as i8
        })
        .collect()
}

#[cfg(feature = "tpu")]
fn transpose_nx_k_to_kx_n(weights: &[i8]) -> Vec<i8> {
    let mut out = vec![0i8; K * N];
    for n in 0..N {
        for k in 0..K {
            out[k * N + n] = weights[n * K + k];
        }
    }
    out
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
fn reference_bf16(
    hidden: &[u16],
    weights: &[i8],
    scales: &[f32],
) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let x = hidden
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    let shape = M2Nvfp4MatmulShape { m: B, n: N, k: K };
    let mut out = vec![0.0f32; B * N];
    int8_matmul_ref(&x, weights, scales, shape, &mut out)?;
    Ok(out.into_iter().map(f32_to_bf16_bits).collect())
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
