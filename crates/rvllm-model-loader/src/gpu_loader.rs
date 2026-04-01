//! SafeTensors GPU loader -- loads weights directly to CUDA device memory as f16.
//!
//! Memory-maps the safetensors file(s), parses the header to find tensor
//! metadata, then uploads each tensor's raw bytes to GPU as f16.
//!
//! All dtypes on disk (F16, BF16, F32) are converted to f16 at load time.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;

    use cudarc::driver::{CudaSlice, CudaStream};
    use memmap2::Mmap;
    use rvllm_core::error::{LLMError, Result};
    use tracing::{debug, info};

    /// Load all safetensors weights as f16 on GPU.
    ///
    /// F16 weights are uploaded directly (zero conversion), BF16 are converted
    /// to f16 on the host, and F32 weights are narrowed to f16.
    pub fn load_weights_to_gpu(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<HashMap<String, CudaSlice<half::f16>>> {
        if path.is_dir() {
            load_sharded_to_gpu(path, stream)
        } else {
            load_single_to_gpu(path, stream)
        }
    }

    fn load_single_to_gpu(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<HashMap<String, CudaSlice<half::f16>>> {
        info!("gpu_loader: memory-mapping {}", path.display());

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

            let (dtype_str, shape, tensor_bytes) =
                parse_tensor_meta(meta, name, data, data_start)?;
            let numel: usize = shape.iter().product();

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
        }

        info!(
            "gpu_loader: loaded {} tensors from {} to GPU (f16)",
            weights.len(),
            path.display()
        );
        Ok(weights)
    }

    fn load_sharded_to_gpu(
        dir: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<HashMap<String, CudaSlice<half::f16>>> {
        let shard_files = collect_shards(dir)?;

        info!(
            "gpu_loader: loading {} shards from {} to GPU (f16)",
            shard_files.len(),
            dir.display()
        );

        let mut all_weights: HashMap<String, CudaSlice<half::f16>> = HashMap::new();
        for shard_path in &shard_files {
            let shard = load_single_to_gpu(shard_path, stream)?;
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
    // Shared helpers
    // -----------------------------------------------------------------------

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

    /// Convert raw tensor bytes to `Vec<half::f16>`.
    ///
    /// - F16: reinterpret bytes directly (zero conversion).
    /// - BF16: convert bf16 -> f16 via f32 intermediate.
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
                let mut out = vec![half::f16::ZERO; numel];
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
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} BF16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let src = unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const u16, numel)
                };
                let mut out = vec![half::f16::ZERO; numel];
                rvllm_zig::bf16_to_f16(src, unsafe {
                    std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, numel)
                });
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
                let src =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, numel) };
                let mut out = vec![half::f16::ZERO; numel];
                rvllm_zig::f32_to_f16(src, unsafe {
                    std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, numel)
                });
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
pub use inner::load_weights_to_gpu;

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        assert!(true);
    }
}
