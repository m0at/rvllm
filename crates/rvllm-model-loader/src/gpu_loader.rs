//! SafeTensors GPU loader -- loads weights directly to CUDA device memory.
//!
//! Memory-maps the safetensors file(s), parses the header to find tensor
//! metadata, then uploads each tensor's raw bytes to GPU.
//!
//! Supports three dtype modes:
//! - `GpuDType::F32`: all weights widened to f32 (original path)
//! - `GpuDType::F16`: f16 kept as-is, bf16 narrowed to f16, f32 narrowed to f16
//!   Halves VRAM and enables hgemm.
//! - FP8: F8_E4M3 weights loaded as raw u8 bytes (1 byte/elem), non-FP8
//!   weights (norms, biases, embeddings) loaded as f32. FP8 weights are
//!   dequantized to f16 on-the-fly during inference, avoiding OOM from
//!   materializing the full f32/f16 model.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;

    use cudarc::driver::{CudaDevice, CudaSlice};
    use memmap2::Mmap;
    use rvllm_core::error::{LLMError, Result};
    use tracing::{debug, info, warn};

    /// Target dtype for GPU weight storage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum GpuDType {
        /// Widen everything to f32 (legacy path).
        F32,
        /// Keep f16 as-is, convert bf16->f16, narrow f32->f16. Halves VRAM.
        F16,
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Load all safetensors weights as f32 (legacy API, unchanged signature).
    pub fn load_weights_to_gpu(
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, CudaSlice<f32>>> {
        if path.is_dir() {
            load_sharded_to_gpu(path, device)
        } else {
            load_single_to_gpu(path, device)
        }
    }

    /// Load all safetensors weights as f16 on GPU.
    ///
    /// F16 weights are uploaded directly (zero widen), BF16 are converted to
    /// f16 on the host, and f32 weights are narrowed to f16. This halves VRAM
    /// usage and enables the hgemm (half-precision GEMM) path.
    pub fn load_weights_to_gpu_f16(
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, CudaSlice<half::f16>>> {
        if path.is_dir() {
            load_sharded_to_gpu_f16(path, device)
        } else {
            load_single_to_gpu_f16(path, device)
        }
    }

    // -----------------------------------------------------------------------
    // F32 path (unchanged)
    // -----------------------------------------------------------------------

    fn load_single_to_gpu(
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, CudaSlice<f32>>> {
        info!("gpu_loader: memory-mapping {}", path.display());

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LLMError::ModelError(format!("mmap failed for {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &mmap;

        let (header, data_start) = parse_safetensors_header(data, path)?;

        let mut weights: HashMap<String, CudaSlice<f32>> = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            let canonical = match normalize_weight_name(name) {
                Some(n) => n,
                None => { continue; }
            };

            let (dtype_str, shape, tensor_bytes) =
                parse_tensor_meta(meta, name, data, data_start)?;
            let numel: usize = shape.iter().product();

            let f32_host = convert_to_f32(tensor_bytes, dtype_str, numel, name)?;

            let gpu_slice = device.htod_sync_copy(&f32_host).map_err(|e| {
                LLMError::GpuError(format!(
                    "htod_sync_copy failed for tensor {} ({} floats): {}",
                    canonical,
                    f32_host.len(),
                    e
                ))
            })?;

            debug!(
                tensor = canonical.as_str(),
                dtype = dtype_str,
                shape = ?shape,
                numel = numel,
                "uploaded tensor to GPU (f32)"
            );

            weights.insert(canonical, gpu_slice);
        }

        info!(
            "gpu_loader: loaded {} tensors from {} to GPU (f32)",
            weights.len(),
            path.display()
        );
        Ok(weights)
    }

    fn load_sharded_to_gpu(
        dir: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, CudaSlice<f32>>> {
        let shard_files = collect_shards(dir)?;

        info!(
            "gpu_loader: loading {} shards from {} to GPU (f32)",
            shard_files.len(),
            dir.display()
        );

        let mut all_weights: HashMap<String, CudaSlice<f32>> = HashMap::new();
        for shard_path in &shard_files {
            let shard = load_single_to_gpu(shard_path, device)?;
            all_weights.extend(shard);
        }

        info!(
            "gpu_loader: loaded {} total tensors from {} shards (f32)",
            all_weights.len(),
            shard_files.len()
        );
        Ok(all_weights)
    }

    // -----------------------------------------------------------------------
    // F16 path (new)
    // -----------------------------------------------------------------------

    fn load_single_to_gpu_f16(
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, CudaSlice<half::f16>>> {
        info!("gpu_loader: memory-mapping {} (f16 mode)", path.display());

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LLMError::ModelError(format!("mmap failed for {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &mmap;

        let (header, data_start) = parse_safetensors_header(data, path)?;

        let mut weights: HashMap<String, CudaSlice<half::f16>> = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            let canonical = match normalize_weight_name(name) {
                Some(n) => n,
                None => { continue; }
            };

            let (dtype_str, shape, tensor_bytes) =
                parse_tensor_meta(meta, name, data, data_start)?;
            let numel: usize = shape.iter().product();

            let f16_host = convert_to_f16(tensor_bytes, dtype_str, numel, name)?;

            let gpu_slice = device.htod_sync_copy(&f16_host).map_err(|e| {
                LLMError::GpuError(format!(
                    "htod_sync_copy failed for tensor {} ({} f16 elems): {}",
                    canonical,
                    f16_host.len(),
                    e
                ))
            })?;

            debug!(
                tensor = canonical.as_str(),
                dtype = dtype_str,
                shape = ?shape,
                numel = numel,
                "uploaded tensor to GPU (f16)"
            );

            weights.insert(canonical, gpu_slice);
        }

        info!(
            "gpu_loader: loaded {} tensors from {} to GPU (f16)",
            weights.len(),
            path.display()
        );
        Ok(weights)
    }

    fn load_sharded_to_gpu_f16(
        dir: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, CudaSlice<half::f16>>> {
        let shard_files = collect_shards(dir)?;

        info!(
            "gpu_loader: loading {} shards from {} to GPU (f16)",
            shard_files.len(),
            dir.display()
        );

        let mut all_weights: HashMap<String, CudaSlice<half::f16>> = HashMap::new();
        for shard_path in &shard_files {
            let shard = load_single_to_gpu_f16(shard_path, device)?;
            all_weights.extend(shard);
        }

        info!(
            "gpu_loader: loaded {} total tensors from {} shards (f16)",
            all_weights.len(),
            shard_files.len()
        );
        Ok(all_weights)
    }

    // -----------------------------------------------------------------------
    // FP8 path: F8_E4M3 as u8, non-FP8 as f32
    // -----------------------------------------------------------------------

    /// Result of loading an FP8 model: projection weights as u8, everything else as f32.
    pub struct Fp8LoadResult {
        /// FP8 E4M3 weights stored as raw u8 bytes on GPU (1 byte per element).
        pub fp8_weights: HashMap<String, CudaSlice<u8>>,
        /// Non-FP8 weights (norms, biases, embeddings) stored as f32 on GPU.
        pub f32_weights: HashMap<String, CudaSlice<f32>>,
        /// Per-tensor scale factors for FP8 weights (if present in the model).
        /// Key is the weight name (e.g. "model.layers.0.self_attn.q_proj.weight"),
        /// value is a single-element f32 scale on GPU.
        pub weight_scales: HashMap<String, CudaSlice<f32>>,
    }

    /// Load safetensors weights for FP8 models.
    ///
    /// F8_E4M3 tensors are uploaded as raw u8 bytes (1 byte/elem).
    /// Tensors named `*.weight_scale` are loaded as f32 scale factors.
    /// All other tensors (norms, biases, embeddings) are converted to f32.
    /// This avoids materializing the full f32/f16 model, preventing OOM.
    pub fn load_weights_to_gpu_fp8(
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<Fp8LoadResult> {
        if path.is_dir() {
            load_sharded_to_gpu_fp8(path, device)
        } else {
            load_single_to_gpu_fp8(path, device)
        }
    }

    fn load_single_to_gpu_fp8(
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<Fp8LoadResult> {
        info!("gpu_loader: memory-mapping {} (fp8 mode)", path.display());

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LLMError::ModelError(format!("mmap failed for {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &mmap;

        let (header, data_start) = parse_safetensors_header(data, path)?;

        let mut fp8_weights: HashMap<String, CudaSlice<u8>> = HashMap::new();
        let mut f32_weights: HashMap<String, CudaSlice<f32>> = HashMap::new();
        let mut weight_scales: HashMap<String, CudaSlice<f32>> = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            // Normalize weight name (strips language_model prefix, skips vision/mtp)
            let canonical = match normalize_weight_name(name) {
                Some(n) => n,
                None => {
                    debug!(tensor = name.as_str(), "skipping (vision/mtp)");
                    continue;
                }
            };

            let (dtype_str, shape, tensor_bytes) =
                parse_tensor_meta(meta, name, data, data_start)?;
            let numel: usize = shape.iter().product();

            if dtype_str == "F8_E4M3" {
                // Upload FP8 weights as raw u8 bytes -- no conversion needed
                let gpu_slice = device.htod_sync_copy(tensor_bytes).map_err(|e| {
                    LLMError::GpuError(format!(
                        "htod_sync_copy failed for fp8 tensor {} ({} bytes): {}",
                        canonical, tensor_bytes.len(), e
                    ))
                })?;
                debug!(
                    tensor = canonical.as_str(),
                    dtype = dtype_str,
                    shape = ?shape,
                    numel = numel,
                    "uploaded tensor to GPU (fp8 u8)"
                );
                fp8_weights.insert(canonical, gpu_slice);
            } else if canonical.ends_with("weight_scale") || canonical.ends_with(".input_scale")
                || canonical.ends_with("weight_scale_inv")
            {
                // Scale factor (per-tensor or block-wise) -- load as f32
                let f32_host = convert_to_f32(tensor_bytes, dtype_str, numel, name)?;
                let gpu_slice = device.htod_sync_copy(&f32_host).map_err(|e| {
                    LLMError::GpuError(format!(
                        "htod_sync_copy failed for scale {} ({} floats): {}",
                        canonical, f32_host.len(), e
                    ))
                })?;
                // Map scale to its parent weight name
                let parent = canonical
                    .strip_suffix(".weight_scale_inv")
                    .or_else(|| canonical.strip_suffix(".weight_scale"))
                    .or_else(|| canonical.strip_suffix(".input_scale"))
                    .map(|p| format!("{p}.weight"))
                    .unwrap_or_else(|| canonical.clone());
                debug!(
                    tensor = canonical.as_str(),
                    parent = parent.as_str(),
                    shape = ?shape,
                    "uploaded scale to GPU (f32, {} elements)", numel
                );
                weight_scales.insert(parent, gpu_slice);
            } else {
                // Non-FP8 tensor (norms, biases, embeddings) -- load as f32
                let f32_host = convert_to_f32(tensor_bytes, dtype_str, numel, name)?;
                let gpu_slice = device.htod_sync_copy(&f32_host).map_err(|e| {
                    LLMError::GpuError(format!(
                        "htod_sync_copy failed for tensor {} ({} floats): {}",
                        canonical, f32_host.len(), e
                    ))
                })?;
                debug!(
                    tensor = canonical.as_str(),
                    dtype = dtype_str,
                    shape = ?shape,
                    "uploaded tensor to GPU (f32)"
                );
                f32_weights.insert(canonical, gpu_slice);
            }
        }

        info!(
            "gpu_loader: loaded {} fp8 + {} f32 + {} scales from {} (fp8 mode)",
            fp8_weights.len(),
            f32_weights.len(),
            weight_scales.len(),
            path.display()
        );

        Ok(Fp8LoadResult { fp8_weights, f32_weights, weight_scales })
    }

    fn load_sharded_to_gpu_fp8(
        dir: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<Fp8LoadResult> {
        let shard_files = collect_shards(dir)?;

        info!(
            "gpu_loader: loading {} shards from {} to GPU (fp8)",
            shard_files.len(),
            dir.display()
        );

        let mut result = Fp8LoadResult {
            fp8_weights: HashMap::new(),
            f32_weights: HashMap::new(),
            weight_scales: HashMap::new(),
        };

        for shard_path in &shard_files {
            let shard = load_single_to_gpu_fp8(shard_path, device)?;
            result.fp8_weights.extend(shard.fp8_weights);
            result.f32_weights.extend(shard.f32_weights);
            result.weight_scales.extend(shard.weight_scales);
        }

        info!(
            "gpu_loader: loaded {} fp8 + {} f32 + {} scales from {} shards",
            result.fp8_weights.len(),
            result.f32_weights.len(),
            result.weight_scales.len(),
            shard_files.len()
        );
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Weight name normalization
    // -----------------------------------------------------------------------

    /// Normalize weight names from various model formats to rvllm's canonical form.
    ///
    /// - Strips `model.language_model.` prefix → `model.` (Qwen3.5 VLM format)
    /// - Skips vision encoder weights (`model.visual.*`) and MTP weights (`mtp.*`)
    ///
    /// Returns `None` if the tensor should be skipped entirely.
    fn normalize_weight_name(name: &str) -> Option<String> {
        // Skip vision encoder and MTP weights
        if name.starts_with("model.visual.") || name.starts_with("mtp.") {
            return None;
        }
        // Strip language_model prefix (Qwen3.5 VLM format)
        if name.starts_with("model.language_model.") {
            Some(name.replacen("model.language_model.", "model.", 1))
        } else {
            Some(name.to_string())
        }
    }

    // -----------------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------------

    /// Parse the safetensors header from raw mmap bytes.
    fn parse_safetensors_header(
        data: &[u8],
        path: &Path,
    ) -> Result<(HashMap<String, serde_json::Value>, usize)> {
        if data.len() < 8 {
            return Err(LLMError::ModelError(
                "safetensors file too small for header".into(),
            ));
        }

        let header_size = u64::from_le_bytes(
            data[..8]
                .try_into()
                .map_err(|_| LLMError::ModelError("invalid header size bytes".into()))?,
        ) as usize;

        if 8 + header_size > data.len() {
            return Err(LLMError::ModelError(
                "header size exceeds file length".into(),
            ));
        }

        let header_bytes = &data[8..8 + header_size];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| LLMError::ModelError(format!("invalid header utf8: {}", e)))?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| LLMError::SerializationError(format!("header json: {}", e)))?;

        Ok((header, 8 + header_size))
    }

    /// Extract dtype, shape, and byte slice for a single tensor from header metadata.
    fn parse_tensor_meta<'a, 'b>(
        meta: &'b serde_json::Value,
        name: &str,
        data: &'a [u8],
        data_start: usize,
    ) -> Result<(&'b str, Vec<usize>, &'a [u8])> {
        let obj = meta.as_object().ok_or_else(|| {
            LLMError::ModelError(format!("tensor {} has non-object meta", name))
        })?;

        let dtype_str = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing dtype", name)))?;

        let shape: Vec<usize> = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing shape", name)))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| LLMError::ModelError("invalid shape element".into()))
            })
            .collect::<Result<Vec<_>>>()?;

        let offsets = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                LLMError::ModelError(format!("tensor {} missing data_offsets", name))
            })?;

        if offsets.len() != 2 {
            return Err(LLMError::ModelError(format!(
                "tensor {} has {} offsets, expected 2",
                name,
                offsets.len()
            )));
        }

        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        let abs_start = data_start + start;
        let abs_end = data_start + end;

        if abs_end > data.len() {
            return Err(LLMError::ModelError(format!(
                "tensor {} data range [{}, {}) exceeds file size {}",
                name,
                abs_start,
                abs_end,
                data.len()
            )));
        }

        Ok((dtype_str, shape, &data[abs_start..abs_end]))
    }

    /// Collect sorted shard file paths from a directory.
    fn collect_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
        let mut shard_files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();
        shard_files.sort();

        if shard_files.is_empty() {
            return Err(LLMError::ModelError(format!(
                "no .safetensors files found in {}",
                dir.display()
            )));
        }
        Ok(shard_files)
    }

    // -----------------------------------------------------------------------
    // GPTQ INT4 path: packed int4 weights + per-group scales + zero points
    // -----------------------------------------------------------------------

    /// Result of loading a GPTQ INT4 model.
    pub struct GptqLoadResult {
        /// Repacked INT4 weights as raw bytes on GPU.
        /// Layout: [N, K/8] as int32 (repacked from GPTQ's [K/8, N]).
        /// Stored as u8 for type compatibility with existing infrastructure.
        pub qweights: HashMap<String, CudaSlice<u8>>,
        /// Per-group scales on GPU: [N, num_groups] as f32.
        pub scales: HashMap<String, CudaSlice<f32>>,
        /// Per-group zero points on GPU: [N, num_groups] as f32 (unpacked from int4).
        pub zeros: HashMap<String, CudaSlice<f32>>,
        /// Non-quantized weights (norms, biases, embeddings) as f32.
        pub f32_weights: HashMap<String, CudaSlice<f32>>,
        /// Group size used in quantization (typically 128).
        pub group_size: usize,
    }

    /// Load safetensors weights for GPTQ INT4 models.
    ///
    /// GPTQ tensors:
    /// - `*.qweight` (I32): packed INT4 weights, shape [K/8, N]
    /// - `*.scales` (F16): per-group scales, shape [K/group_size, N]
    /// - `*.qzeros` (I32): packed INT4 zero points, shape [K/group_size/8, N]
    /// - `*.g_idx` (I32): group indices (optional, ignored for now)
    ///
    /// During loading, weights are repacked from [K/8, N] to [N, K/8] for
    /// coalesced GPU reads in the GEMV kernel.
    pub fn load_weights_to_gpu_gptq(
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<GptqLoadResult> {
        if path.is_dir() {
            load_sharded_to_gpu_gptq(path, device)
        } else {
            load_single_to_gpu_gptq(path, device)
        }
    }

    fn load_single_to_gpu_gptq(
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<GptqLoadResult> {
        info!("gpu_loader: memory-mapping {} (gptq mode)", path.display());

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LLMError::ModelError(format!("mmap failed for {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &mmap;

        let (header, data_start) = parse_safetensors_header(data, path)?;

        let mut qweights: HashMap<String, CudaSlice<u8>> = HashMap::new();
        let mut scales: HashMap<String, CudaSlice<f32>> = HashMap::new();
        let mut zeros: HashMap<String, CudaSlice<f32>> = HashMap::new();
        let mut f32_weights: HashMap<String, CudaSlice<f32>> = HashMap::new();
        let mut group_size: usize = 128; // default

        // First pass: collect raw tensors grouped by layer/projection
        // We need qweight + scales + qzeros together for repacking.
        // Strategy: load all tensors, then repack qweights using scale shapes.
        struct RawGptqTensor {
            bytes: Vec<u8>,
            dtype: String,
            shape: Vec<usize>,
        }
        let mut raw_tensors: HashMap<String, RawGptqTensor> = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            let canonical = match normalize_weight_name(name) {
                Some(n) => n,
                None => {
                    debug!(tensor = name.as_str(), "skipping (vision/mtp)");
                    continue;
                }
            };

            let (dtype_str, shape, tensor_bytes) =
                parse_tensor_meta(meta, name, data, data_start)?;

            raw_tensors.insert(canonical, RawGptqTensor {
                bytes: tensor_bytes.to_vec(),
                dtype: dtype_str.to_string(),
                shape,
            });
        }

        // Process tensors: identify GPTQ components and non-quantized weights
        let tensor_names: Vec<String> = raw_tensors.keys().cloned().collect();

        for name in &tensor_names {
            // Skip GPTQ component tensors (processed with their parent)
            if name.ends_with(".qweight") || name.ends_with(".qzeros")
                || name.ends_with(".scales") || name.ends_with(".g_idx")
            {
                continue;
            }

            // Check if this weight has GPTQ quantization
            let qweight_name = format!("{}.qweight", name.strip_suffix(".weight").unwrap_or(name));
            if raw_tensors.contains_key(&qweight_name) {
                // This is a quantized projection -- process the GPTQ group
                let base = name.strip_suffix(".weight").unwrap_or(name);
                let scales_name = format!("{base}.scales");
                let qzeros_name = format!("{base}.qzeros");

                let qw = raw_tensors.get(&qweight_name).ok_or_else(|| {
                    LLMError::ModelError(format!("missing {qweight_name}"))
                })?;
                let sc = raw_tensors.get(&scales_name).ok_or_else(|| {
                    LLMError::ModelError(format!("missing {scales_name}"))
                })?;

                // qweight shape: [K/8, N] as I32
                let k_packed = qw.shape[0]; // K/8
                let n_out = qw.shape[1];    // N (output features)
                let k = k_packed * 8;       // K (input features)

                // scales shape: [num_groups, N] as F16 or BF16
                let num_groups = sc.shape[0];
                let gs = k / num_groups;
                group_size = gs;

                debug!(
                    base, k, n_out, num_groups, group_size,
                    "repacking GPTQ qweight [{k_packed}, {n_out}] -> [{n_out}, {k_packed}]"
                );

                // Repack qweight from [K/8, N] to [N, K/8] (transpose int32s)
                let qw_i32: &[i32] = unsafe {
                    std::slice::from_raw_parts(
                        qw.bytes.as_ptr() as *const i32,
                        qw.bytes.len() / 4,
                    )
                };
                let mut repacked = vec![0i32; k_packed * n_out];
                for row in 0..k_packed {
                    for col in 0..n_out {
                        repacked[col * k_packed + row] = qw_i32[row * n_out + col];
                    }
                }
                // Upload as raw u8 bytes
                let repacked_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        repacked.as_ptr() as *const u8,
                        repacked.len() * 4,
                    )
                };
                let gpu_qw = device.htod_sync_copy(repacked_bytes).map_err(|e| {
                    LLMError::GpuError(format!("htod qweight {base}: {e}"))
                })?;

                // Canonical name for the weight (matches what gpu_runner expects)
                let weight_name = format!("{base}.weight");
                qweights.insert(weight_name.clone(), gpu_qw);

                // Repack scales from [num_groups, N] to [N, num_groups] as f32
                let sc_f32 = convert_to_f32(&sc.bytes, &sc.dtype, sc.shape.iter().product(), &scales_name)?;
                let mut scales_repacked = vec![0f32; n_out * num_groups];
                for g in 0..num_groups {
                    for col in 0..n_out {
                        scales_repacked[col * num_groups + g] = sc_f32[g * n_out + col];
                    }
                }
                let gpu_sc = device.htod_sync_copy(&scales_repacked).map_err(|e| {
                    LLMError::GpuError(format!("htod scales {base}: {e}"))
                })?;
                scales.insert(weight_name.clone(), gpu_sc);

                // Unpack and repack qzeros from [num_groups/8, N] packed I32 to [N, num_groups] f32
                if let Some(qz) = raw_tensors.get(&qzeros_name) {
                    let qz_i32: &[i32] = unsafe {
                        std::slice::from_raw_parts(
                            qz.bytes.as_ptr() as *const i32,
                            qz.bytes.len() / 4,
                        )
                    };
                    let zp_rows = qz.shape[0]; // num_groups / 8 (or num_groups if not packed)
                    let zp_cols = qz.shape[1]; // N

                    let mut zeros_repacked = vec![0f32; n_out * num_groups];

                    if zp_rows * 8 <= num_groups + 8 {
                        // Packed zero points: 8 INT4 values per int32
                        for zr in 0..zp_rows {
                            for col in 0..zp_cols {
                                let packed = qz_i32[zr * zp_cols + col];
                                for nib in 0..8 {
                                    let group_idx = zr * 8 + nib;
                                    if group_idx < num_groups {
                                        let zp = ((packed >> (nib * 4)) & 0xF) as f32;
                                        zeros_repacked[col * num_groups + group_idx] = zp;
                                    }
                                }
                            }
                        }
                    } else {
                        // Unpacked zero points (less common)
                        for g in 0..num_groups.min(zp_rows) {
                            for col in 0..n_out.min(zp_cols) {
                                zeros_repacked[col * num_groups + g] = qz_i32[g * zp_cols + col] as f32;
                            }
                        }
                    }

                    let gpu_zp = device.htod_sync_copy(&zeros_repacked).map_err(|e| {
                        LLMError::GpuError(format!("htod zeros {base}: {e}"))
                    })?;
                    zeros.insert(weight_name.clone(), gpu_zp);
                } else {
                    // No qzeros -- use zero as zero point (symmetric quantization)
                    let zero_zp = vec![0f32; n_out * num_groups];
                    let gpu_zp = device.htod_sync_copy(&zero_zp).map_err(|e| {
                        LLMError::GpuError(format!("htod zero zeros {base}: {e}"))
                    })?;
                    zeros.insert(weight_name.clone(), gpu_zp);
                }

                continue;
            }

            // Non-quantized tensor (norms, biases, embeddings)
            if let Some(tensor) = raw_tensors.get(name) {
                let numel: usize = tensor.shape.iter().product();
                let f32_host = convert_to_f32(&tensor.bytes, &tensor.dtype, numel, name)?;
                let gpu_slice = device.htod_sync_copy(&f32_host).map_err(|e| {
                    LLMError::GpuError(format!("htod {name}: {e}"))
                })?;
                f32_weights.insert(name.clone(), gpu_slice);
            }
        }

        // Also load GPTQ component tensors that don't have a parent .weight
        // (e.g., standalone qweight entries -- the weight name IS the base)
        for name in &tensor_names {
            if !name.ends_with(".qweight") { continue; }
            let base = name.strip_suffix(".qweight").unwrap_or(name);
            let weight_name = format!("{base}.weight");
            if qweights.contains_key(&weight_name) { continue; }

            // This qweight doesn't have a parent -- process it directly
            let qw = raw_tensors.get(name).unwrap();
            let scales_name = format!("{base}.scales");
            let qzeros_name = format!("{base}.qzeros");

            let k_packed = qw.shape[0];
            let n_out = qw.shape[1];
            let k = k_packed * 8;

            let sc = raw_tensors.get(&scales_name).ok_or_else(|| {
                LLMError::ModelError(format!("missing {scales_name}"))
            })?;
            let num_groups = sc.shape[0];
            group_size = k / num_groups;

            debug!(base, k, n_out, num_groups, group_size, "repacking standalone GPTQ qweight");

            // Repack qweight
            let qw_i32: &[i32] = unsafe {
                std::slice::from_raw_parts(qw.bytes.as_ptr() as *const i32, qw.bytes.len() / 4)
            };
            let mut repacked = vec![0i32; k_packed * n_out];
            for row in 0..k_packed {
                for col in 0..n_out {
                    repacked[col * k_packed + row] = qw_i32[row * n_out + col];
                }
            }
            let repacked_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(repacked.as_ptr() as *const u8, repacked.len() * 4)
            };
            let gpu_qw = device.htod_sync_copy(repacked_bytes).map_err(|e| {
                LLMError::GpuError(format!("htod qweight {base}: {e}"))
            })?;
            qweights.insert(weight_name.clone(), gpu_qw);

            // Repack scales
            let sc_f32 = convert_to_f32(&sc.bytes, &sc.dtype, sc.shape.iter().product(), &scales_name)?;
            let mut scales_repacked = vec![0f32; n_out * num_groups];
            for g in 0..num_groups {
                for col in 0..n_out {
                    scales_repacked[col * num_groups + g] = sc_f32[g * n_out + col];
                }
            }
            let gpu_sc = device.htod_sync_copy(&scales_repacked).map_err(|e| {
                LLMError::GpuError(format!("htod scales {base}: {e}"))
            })?;
            scales.insert(weight_name.clone(), gpu_sc);

            // Unpack/repack qzeros
            if let Some(qz) = raw_tensors.get(&qzeros_name) {
                let qz_i32: &[i32] = unsafe {
                    std::slice::from_raw_parts(qz.bytes.as_ptr() as *const i32, qz.bytes.len() / 4)
                };
                let zp_rows = qz.shape[0];
                let zp_cols = qz.shape[1];
                let mut zeros_repacked = vec![0f32; n_out * num_groups];
                if zp_rows * 8 <= num_groups + 8 {
                    for zr in 0..zp_rows {
                        for col in 0..zp_cols {
                            let packed = qz_i32[zr * zp_cols + col];
                            for nib in 0..8 {
                                let group_idx = zr * 8 + nib;
                                if group_idx < num_groups {
                                    let zp = ((packed >> (nib * 4)) & 0xF) as f32;
                                    zeros_repacked[col * num_groups + group_idx] = zp;
                                }
                            }
                        }
                    }
                } else {
                    for g in 0..num_groups.min(zp_rows) {
                        for col in 0..n_out.min(zp_cols) {
                            zeros_repacked[col * num_groups + g] = qz_i32[g * zp_cols + col] as f32;
                        }
                    }
                }
                let gpu_zp = device.htod_sync_copy(&zeros_repacked).map_err(|e| {
                    LLMError::GpuError(format!("htod zeros {base}: {e}"))
                })?;
                zeros.insert(weight_name.clone(), gpu_zp);
            } else {
                let zero_zp = vec![0f32; n_out * num_groups];
                let gpu_zp = device.htod_sync_copy(&zero_zp).map_err(|e| {
                    LLMError::GpuError(format!("htod zero zeros {base}: {e}"))
                })?;
                zeros.insert(weight_name.clone(), gpu_zp);
            }
        }

        info!(
            "gpu_loader: loaded {} qweights + {} scales + {} zeros + {} f32 from {} (gptq mode, group_size={})",
            qweights.len(), scales.len(), zeros.len(), f32_weights.len(),
            path.display(), group_size
        );

        Ok(GptqLoadResult { qweights, scales, zeros, f32_weights, group_size })
    }

    fn load_sharded_to_gpu_gptq(
        dir: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<GptqLoadResult> {
        let shard_files = collect_shards(dir)?;

        info!(
            "gpu_loader: loading {} shards from {} to GPU (gptq)",
            shard_files.len(), dir.display()
        );

        let mut result = GptqLoadResult {
            qweights: HashMap::new(),
            scales: HashMap::new(),
            zeros: HashMap::new(),
            f32_weights: HashMap::new(),
            group_size: 128,
        };

        for shard_path in &shard_files {
            let shard = load_single_to_gpu_gptq(shard_path, device)?;
            result.qweights.extend(shard.qweights);
            result.scales.extend(shard.scales);
            result.zeros.extend(shard.zeros);
            result.f32_weights.extend(shard.f32_weights);
            result.group_size = shard.group_size;
        }

        info!(
            "gpu_loader: loaded {} qweights + {} f32 from {} shards (gptq, gs={})",
            result.qweights.len(), result.f32_weights.len(),
            shard_files.len(), result.group_size
        );
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Dtype conversion
    // -----------------------------------------------------------------------

    /// Convert raw tensor bytes to `Vec<f32>` based on the safetensors dtype string.
    ///
    /// Supported dtypes: F32 (zero-copy reinterpret), F16, BF16 (widened to f32).
    fn convert_to_f32(
        bytes: &[u8],
        dtype_str: &str,
        numel: usize,
        tensor_name: &str,
    ) -> Result<Vec<f32>> {
        match dtype_str {
            "F32" => {
                if bytes.len() != numel * 4 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} F32 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let mut out = vec![0f32; numel];
                // SAFETY: f32 is Pod, byte count verified.
                let src =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, numel) };
                out.copy_from_slice(src);
                Ok(out)
            }
            "F16" => {
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} F16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let mut out = Vec::with_capacity(numel);
                for i in 0..numel {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    let val = half::f16::from_bits(bits);
                    out.push(val.to_f32());
                }
                Ok(out)
            }
            "BF16" => {
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} BF16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let mut out = Vec::with_capacity(numel);
                for i in 0..numel {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    let val = half::bf16::from_bits(bits);
                    out.push(val.to_f32());
                }
                Ok(out)
            }
            "I32" => {
                if bytes.len() != numel * 4 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} I32 size mismatch: {} bytes for {} elements",
                        tensor_name, bytes.len(), numel
                    )));
                }
                let mut out = vec![0f32; numel];
                let src = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i32, numel) };
                for (i, &v) in src.iter().enumerate() {
                    out[i] = v as f32;
                }
                Ok(out)
            }
            _ => Err(LLMError::ModelError(format!(
                "gpu_loader: unsupported dtype '{}' for tensor '{}', only F32/F16/BF16/I32 supported",
                dtype_str, tensor_name
            ))),
        }
    }

    /// Convert raw tensor bytes to `Vec<half::f16>` for the f16 GPU path.
    ///
    /// - F16: reinterpret bytes directly as half::f16 (no conversion).
    /// - BF16: convert bf16 -> f16 on the host (no intermediate f32 widen).
    /// - F32: narrow f32 -> f16.
    fn convert_to_f16(
        bytes: &[u8],
        dtype_str: &str,
        numel: usize,
        tensor_name: &str,
    ) -> Result<Vec<half::f16>> {
        match dtype_str {
            "F16" => {
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} F16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                // Direct reinterpret -- no conversion needed.
                let mut out = vec![half::f16::ZERO; numel];
                // SAFETY: half::f16 is repr(transparent) over u16, 2 bytes each,
                // byte count verified above. Source is valid mmap data.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr() as *mut u8,
                        bytes.len(),
                    );
                }
                Ok(out)
            }
            "BF16" => {
                // Convert bf16 -> f16 directly without widening to f32.
                // bf16 has 8-bit exponent + 7-bit mantissa
                // f16  has 5-bit exponent + 10-bit mantissa
                // We go bf16 -> f32 -> f16 per element. The bf16->f32 step is
                // a cheap bit shift (no real work), and f32->f16 is the
                // standard narrowing. This avoids allocating a full f32 buffer.
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} BF16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let mut out = Vec::with_capacity(numel);
                for i in 0..numel {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    let bf = half::bf16::from_bits(bits);
                    // bf16->f32 is a trivial bit shift, then f32->f16 narrow
                    out.push(half::f16::from_f32(bf.to_f32()));
                }
                Ok(out)
            }
            "F32" => {
                if bytes.len() != numel * 4 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} F32 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                // Narrow f32 -> f16
                let mut out = Vec::with_capacity(numel);
                // SAFETY: f32 is Pod, byte count verified.
                let src =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, numel) };
                for &v in src {
                    out.push(half::f16::from_f32(v));
                }
                Ok(out)
            }
            _ => Err(LLMError::ModelError(format!(
                "gpu_loader: unsupported dtype '{}' for tensor '{}', only F32/F16/BF16 supported",
                dtype_str, tensor_name
            ))),
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::{load_weights_to_gpu, load_weights_to_gpu_f16, load_weights_to_gpu_fp8, load_weights_to_gpu_gptq, Fp8LoadResult, GptqLoadResult, GpuDType};

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        assert!(true);
    }
}
