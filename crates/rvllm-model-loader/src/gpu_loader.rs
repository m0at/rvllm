//! SafeTensors GPU loader -- loads weights directly to CUDA device memory.
//!
//! Memory-maps the safetensors file(s), parses the header to find tensor
//! metadata, then uploads each tensor's raw bytes to GPU.
//!
//! Supports two dtype modes:
//! - `GpuDType::F32`: all weights widened to f32 (original path)
//! - `GpuDType::F16`: f16 kept as-is, bf16 narrowed to f16, f32 narrowed to f16
//!   Halves VRAM and enables hgemm.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;

    use cudarc::driver::{CudaSlice, CudaStream};
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
        stream: &Arc<CudaStream>,
    ) -> Result<HashMap<String, CudaSlice<f32>>> {
        load_weights_to_gpu_with_shapes(path, stream).map(|(weights, _shapes)| weights)
    }

    /// Load all safetensors weights as f32 and preserve tensor shapes.
    pub fn load_weights_to_gpu_with_shapes(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<f32>>, HashMap<String, Vec<usize>>)> {
        if path.is_dir() {
            load_sharded_to_gpu(path, stream)
        } else {
            load_single_to_gpu(path, stream)
        }
    }

    /// Load all safetensors weights as f16 on GPU.
    ///
    /// F16 weights are uploaded directly (zero widen), BF16 are converted to
    /// f16 on the host, and f32 weights are narrowed to f16. This halves VRAM
    /// usage and enables the hgemm (half-precision GEMM) path.
    pub fn load_weights_to_gpu_f16(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<HashMap<String, CudaSlice<half::f16>>> {
        load_weights_to_gpu_f16_with_shapes(path, stream).map(|(weights, _shapes)| weights)
    }

    /// Load all safetensors weights as f16 on GPU and preserve tensor shapes.
    pub fn load_weights_to_gpu_f16_with_shapes(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<half::f16>>, HashMap<String, Vec<usize>>)> {
        if path.is_dir() {
            load_sharded_to_gpu_f16(path, stream)
        } else {
            load_single_to_gpu_f16(path, stream)
        }
    }

    /// Load all raw `U8` tensors from safetensors files into host memory.
    ///
    /// GPT-OSS stores MXFP4 expert blocks/scales as `U8`, so these tensors
    /// must survive loader setup even though they are not uploaded through the
    /// normal f32/f16 weight maps.
    pub fn load_u8_weights_to_host(path: &Path) -> Result<HashMap<String, Vec<u8>>> {
        if path.is_dir() {
            load_sharded_u8_to_host(path)
        } else {
            load_single_u8_to_host(path)
        }
    }

    // -----------------------------------------------------------------------
    // F32 path (unchanged)
    // -----------------------------------------------------------------------

    fn load_single_to_gpu(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<f32>>, HashMap<String, Vec<usize>>)> {
        info!("gpu_loader: memory-mapping {}", path.display());

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LLMError::ModelError(format!("mmap failed for {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &mmap;

        let (header, data_start) = parse_safetensors_header(data, path)?;

        let mut weights: HashMap<String, CudaSlice<f32>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            let (dtype_str, shape, tensor_bytes) =
                parse_tensor_meta(meta, name, data, data_start)?;
            let numel: usize = shape.iter().product();

            if dtype_str == "U8" {
                debug!(tensor = name.as_str(), shape = ?shape, "skipping U8 tensor on f32 GPU path");
                shapes.insert(name.clone(), shape);
                continue;
            }

            let f32_host = convert_to_f32(tensor_bytes, dtype_str, numel, name)?;

            let gpu_slice = stream.clone_htod(&f32_host).map_err(|e| {
                LLMError::GpuError(format!(
                    "clone_htod failed for tensor {} ({} floats): {}",
                    name,
                    f32_host.len(),
                    e
                ))
            })?;

            debug!(
                tensor = name.as_str(),
                dtype = dtype_str,
                shape = ?shape,
                numel = numel,
                "uploaded tensor to GPU (f32)"
            );

            weights.insert(name.clone(), gpu_slice);
            shapes.insert(name.clone(), shape);
        }

        info!(
            "gpu_loader: loaded {} tensors from {} to GPU (f32)",
            weights.len(),
            path.display()
        );
        Ok((weights, shapes))
    }

    fn load_sharded_to_gpu(
        dir: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<f32>>, HashMap<String, Vec<usize>>)> {
        let shard_files = collect_shards(dir)?;

        info!(
            "gpu_loader: loading {} shards from {} to GPU (f32)",
            shard_files.len(),
            dir.display()
        );

        let mut all_weights: HashMap<String, CudaSlice<f32>> = HashMap::new();
        let mut all_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        for shard_path in &shard_files {
            let (shard_weights, shard_shapes) = load_single_to_gpu(shard_path, stream)?;
            all_weights.extend(shard_weights);
            all_shapes.extend(shard_shapes);
        }

        info!(
            "gpu_loader: loaded {} total tensors from {} shards (f32)",
            all_weights.len(),
            shard_files.len()
        );
        Ok((all_weights, all_shapes))
    }

    // -----------------------------------------------------------------------
    // F16 path (new)
    // -----------------------------------------------------------------------

    fn load_single_to_gpu_f16(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<half::f16>>, HashMap<String, Vec<usize>>)> {
        info!("gpu_loader: memory-mapping {} (f16 mode)", path.display());

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LLMError::ModelError(format!("mmap failed for {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &mmap;

        let (header, data_start) = parse_safetensors_header(data, path)?;

        let mut weights: HashMap<String, CudaSlice<half::f16>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            let (dtype_str, shape, tensor_bytes) =
                parse_tensor_meta(meta, name, data, data_start)?;
            let numel: usize = shape.iter().product();

            if dtype_str == "U8" {
                debug!(tensor = name.as_str(), shape = ?shape, "skipping U8 tensor on f16 GPU path");
                shapes.insert(name.clone(), shape);
                continue;
            }

            let f16_host = convert_to_f16(tensor_bytes, dtype_str, numel, name)?;

            let gpu_slice = stream.clone_htod(&f16_host).map_err(|e| {
                LLMError::GpuError(format!(
                    "clone_htod failed for tensor {} ({} f16 elems): {}",
                    name,
                    f16_host.len(),
                    e
                ))
            })?;

            debug!(
                tensor = name.as_str(),
                dtype = dtype_str,
                shape = ?shape,
                numel = numel,
                "uploaded tensor to GPU (f16)"
            );

            weights.insert(name.clone(), gpu_slice);
            shapes.insert(name.clone(), shape);
        }

        info!(
            "gpu_loader: loaded {} tensors from {} to GPU (f16)",
            weights.len(),
            path.display()
        );
        Ok((weights, shapes))
    }

    fn load_sharded_to_gpu_f16(
        dir: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<half::f16>>, HashMap<String, Vec<usize>>)> {
        let shard_files = collect_shards(dir)?;

        info!(
            "gpu_loader: loading {} shards from {} to GPU (f16)",
            shard_files.len(),
            dir.display()
        );

        let mut all_weights: HashMap<String, CudaSlice<half::f16>> = HashMap::new();
        let mut all_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        for shard_path in &shard_files {
            let (shard_weights, shard_shapes) = load_single_to_gpu_f16(shard_path, stream)?;
            all_weights.extend(shard_weights);
            all_shapes.extend(shard_shapes);
        }

        info!(
            "gpu_loader: loaded {} total tensors from {} shards (f16)",
            all_weights.len(),
            shard_files.len()
        );
        Ok((all_weights, all_shapes))
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

    fn load_single_u8_to_host(path: &Path) -> Result<HashMap<String, Vec<u8>>> {
        info!("gpu_loader: memory-mapping {} (host U8 path)", path.display());

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LLMError::ModelError(format!("mmap failed for {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &mmap;

        let (header, data_start) = parse_safetensors_header(data, path)?;
        let mut weights = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            let (dtype_str, _shape, tensor_bytes) =
                parse_tensor_meta(meta, name, data, data_start)?;
            if dtype_str == "U8" {
                weights.insert(name.clone(), tensor_bytes.to_vec());
            }
        }

        Ok(weights)
    }

    fn load_sharded_u8_to_host(dir: &Path) -> Result<HashMap<String, Vec<u8>>> {
        let shard_files = collect_shards(dir)?;
        let mut all_weights = HashMap::new();
        for shard_path in &shard_files {
            all_weights.extend(load_single_u8_to_host(shard_path)?);
        }
        Ok(all_weights)
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
            _ => Err(LLMError::ModelError(format!(
                "gpu_loader: unsupported dtype '{}' for tensor '{}', only F32/F16/BF16 supported",
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
pub use inner::{
    load_u8_weights_to_host, load_weights_to_gpu, load_weights_to_gpu_f16,
    load_weights_to_gpu_f16_with_shapes, load_weights_to_gpu_with_shapes, GpuDType,
};

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        assert!(true);
    }
}
