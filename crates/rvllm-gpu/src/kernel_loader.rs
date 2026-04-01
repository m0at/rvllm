//! CUDA kernel loader: loads PTX files and launches kernels via cudarc.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaStream, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use tracing::{debug, info, trace, warn};

use crate::Result;

/// Known kernel -> function-name mappings for the vllm-rs kernel set.
/// These are the `extern "C" __global__` entry points in each .cu file.
static KERNEL_FUNCTIONS: &[(&str, &[&str])] = &[
    (
        "activation",
        &["silu_kernel", "fused_silu_mul_kernel", "gelu_kernel"],
    ),
    ("argmax", &["argmax_kernel"]),
    ("argmax_f16", &["argmax_f16_kernel"]),
    (
        "activation_f16",
        &[
            "silu_f16_kernel",
            "fused_silu_mul_f16_kernel",
            "gelu_f16_kernel",
        ],
    ),
    (
        "add_bias",
        &["add_bias_kernel", "add_kernel", "add_inplace_kernel"],
    ),
    (
        "attn_gate",
        &["split_gate_kernel", "truncate_q_kernel", "sigmoid_gate_kernel"],
    ),
    (
        "add_bias_f16",
        &["add_bias_f16_kernel", "add_f16_kernel", "add_inplace_f16_kernel"],
    ),
    ("copy_blocks", &["copy_blocks_kernel"]),
    ("copy_blocks_f16", &["copy_blocks_f16_kernel"]),
    ("dequant_f16", &["dequant_f16_to_f32"]),
    (
        "dequant_fp8",
        &[
            "dequant_fp8_to_f16_kernel",
            "dequant_fp8_scaled_to_f16_kernel",
            "dequant_fp8_blockwise_to_f16_kernel",
            "dequant_fp8_to_bf16_kernel",
            "dequant_fp8_scaled_to_bf16_kernel",
            "dequant_fp8_blockwise_to_bf16_kernel",
        ],
    ),
    (
        "fp8_gemv",
        &["fp8_gemv_blockwise_kernel", "fp8_gemv_blockwise_vec_kernel", "fp8_gemv_blockwise_lut_kernel", "fp8_gemv_blockwise_v2_kernel", "fp8_gemv_scaled_kernel", "fp8_gemv_kernel", "fp8_gemv_blockwise_wpr_kernel", "fp8_gemv_blockwise_wpr_lut_kernel", "fp8_gemv_blockwise_wpr_native_kernel"],
    ),
    (
        "fp8_gemv_smem",
        &["fp8_gemv_smem_kernel"],
    ),
    (
        "int4_gemv",
        &["int4_gemv_kernel", "dequant_int4_to_f16_kernel"],
    ),
    ("embedding_gather", &["embedding_gather_kernel"]),
    ("embedding_gather_f16", &["embedding_gather_f16_kernel"]),
    (
        "flash_attention",
        &[
            "flash_attention_2_kernel",
            "flash_attention_2_decode_kernel",
            "flash_attention_2_f16kv_kernel",
            "flash_attention_2_decode_f16kv_kernel",
        ],
    ),
    (
<<<<<<< Updated upstream
        "flash_attention_3",
        &["flash_attention_3_decode_f16io_kernel", "flash_attention_3_decode_gqa_f16io_kernel"],
    ),
    (
        "flash_attention_3_prefill",
        &["flash_attention_3_prefill_f16io_kernel"],
    ),
    (
        "flash_attention_3_v3",
        &["fa3_v3_decode_gqa_kernel", "fa3_v3_decode_kernel", "fa3_v3_combine_f16_kernel"],
    ),
    (
=======
>>>>>>> Stashed changes
        "fp8_kv",
        &[
            "quantize_kv_kernel",
            "dequantize_kv_kernel",
            "quantize_paged_kv_kernel",
            "dequantize_paged_kv_kernel",
        ],
    ),
    ("fused_residual_rmsnorm", &["fused_residual_rmsnorm_kernel"]),
    (
        "fused_residual_rmsnorm_f16",
        &["fused_residual_rmsnorm_f16_kernel"],
    ),
    (
        "paged_attention",
        &["paged_attention_v2_kernel", "paged_attention_v2_f16kv_kernel"],
    ),
    (
        "reshape_and_cache",
        &["reshape_and_cache_kernel", "reshape_and_cache_f16_kernel"],
    ),
    ("reshape_and_cache_f16", &["reshape_and_cache_f16_kernel"]),
    (
        "cast_fp",
<<<<<<< Updated upstream
        &["cast_f32_to_f16_kernel", "cast_f16_to_f32_kernel"],
    ),
    (
        "reshape_and_cache_f16",
        &["reshape_and_cache_f16io_kernel"],
    ),
    (
        "fused_norm_gemv",
        &["fused_norm_gemv_f16_kernel", "fused_norm_gemv_bias_f16_kernel"],
    ),
    (
        "fused_silu_down",
        &["fused_silu_down_f16_kernel", "fused_silu_down_bias_f16_kernel"],
    ),
    (
        "fused_add_norm_qkv_gemv",
        &["fused_cute_add_norm_qkv_gemv", "fused_cute_norm_qkv_gemv", "fused_cute_add_norm_qkv_bias_gemv", "fused_cute_norm_qkv_bias_gemv", "fused_cute_add_norm_qkv_fp8_gemv", "fused_cute_norm_qkv_fp8_gemv", "fused_cute_add_norm_qkv_fp8_bias_gemv", "fused_cute_norm_qkv_fp8_bias_gemv"],
    ),
    (
        "fused_add_norm_gateup_gemv",
        &["fused_cute_add_norm_gateup_gemv", "fused_cute_add_norm_gateup_fp8_gemv"],
    ),
    (
        "fused_oproj_add_norm_gateup_gemv",
        &["fused_cute_oproj_add_norm_gateup_gemv", "fused_cute_oproj_add_norm_gateup_fp8_gemv"],
    ),
    (
        "fused_silu_down_gemv",
        &["fused_cute_silu_down_gemv", "fused_cute_silu_down_fp8_gemv"],
    ),
    (
        "persistent_layer_decode",
        &["persistent_layer_decode_f16"],
    ),
    (
        "megakernel_decode",
        &["megakernel_decode_f16"],
    ),
    (
        "gemv_fp8",
=======
>>>>>>> Stashed changes
        &[
            "cast_f32_to_f16_kernel",
            "cast_f16_to_f32_kernel",
            "cast_f32_to_bf16_kernel",
            "cast_bf16_to_f32_kernel",
            "round_f32_to_bf16_kernel",
        ],
    ),
    ("rms_norm", &["rms_norm_kernel"]),
    (
        "rms_norm_f16",
        &["rms_norm_f16_kernel", "fused_residual_rmsnorm_f16_kernel"],
    ),
    ("rotary_embedding", &["rotary_embedding_kernel"]),
    ("rotary_embedding_f16", &["rotary_embedding_f16_kernel"]),
    ("softmax", &["softmax_kernel"]),
    ("softmax_f16", &["softmax_f16_kernel"]),
    (
        "mamba2_ssm",
        &[
            "mamba2_conv1d_step",
            "mamba2_compute_gates",
            "mamba2_l2_normalize",
            "mamba2_gqa_expand",
            "mamba2_ssm_step",
            "mamba2_norm_gate",
            "mamba2_init_state",
            "mamba2_small_gemv",
        ],
    ),
];

/// Loads and manages CUDA PTX modules, providing kernel launch capabilities.
///
/// Wraps `cudarc::driver::CudaDevice` module management with a higher-level
/// API that understands the vllm-rs kernel naming conventions.
pub struct KernelLoader {
    device: Arc<CudaDevice>,
    loaded_modules: HashMap<String, Vec<&'static str>>,
}

impl KernelLoader {
    /// Create a new KernelLoader and load all .ptx files from `kernel_dir`.
    ///
    /// Falls back to the `RVLLM_KERNEL_DIR` environment variable if `kernel_dir`
    /// does not contain any .ptx files.
    pub fn new(device: Arc<CudaDevice>, kernel_dir: &Path) -> Result<Self> {
        let mut loader = Self {
            device,
            loaded_modules: HashMap::new(),
        };

        let dir = if kernel_dir.exists() && kernel_dir.is_dir() {
            kernel_dir.to_path_buf()
        } else if let Ok(env_dir) = std::env::var("RVLLM_KERNEL_DIR") {
            let p = Path::new(&env_dir).to_path_buf();
            if p.exists() && p.is_dir() {
                p
            } else {
                info!(
                    dir = %kernel_dir.display(),
                    env_dir = %p.display(),
                    "no PTX directory found, kernel loader created empty"
                );
                return Ok(loader);
            }
        } else {
            info!(
                dir = %kernel_dir.display(),
                "no PTX directory found, kernel loader created empty"
            );
            return Ok(loader);
        };

        loader.load_directory(&dir)?;
        Ok(loader)
    }

    /// Create a KernelLoader with no pre-loaded kernels.
    /// Kernels can be loaded later via `load_ptx`.
    pub fn empty(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            loaded_modules: HashMap::new(),
        }
    }

    /// Load a single PTX module from raw bytes (UTF-8 PTX source).
    ///
    /// The `name` is used as the module name for later `get_func` / `launch` calls.
    /// Function names are resolved from the known kernel mapping if available,
    /// otherwise the caller should use `load_ptx_with_functions` to specify them.
    pub fn load_ptx(&mut self, name: &str, ptx_bytes: &[u8]) -> Result<()> {
        let func_names = self.resolve_function_names(name);
        self.load_ptx_with_functions(name, ptx_bytes, &func_names)
    }

    /// Load a PTX module from raw bytes, explicitly specifying function names to register.
    pub fn load_ptx_with_functions(
        &mut self,
        name: &str,
        ptx_bytes: &[u8],
        func_names: &[&'static str],
    ) -> Result<()> {
        let ptx_src = std::str::from_utf8(ptx_bytes).map_err(|e| {
            crate::LLMError::GpuError(format!("PTX bytes for '{name}' are not valid UTF-8: {e}"))
        })?;

        let ptx = Ptx::from_src(ptx_src);

        self.device.load_ptx(ptx, name, func_names).map_err(|e| {
            crate::LLMError::GpuError(format!("failed to load PTX module '{name}': {e}"))
        })?;

        debug!(module = name, functions = ?func_names, "loaded PTX module");
        self.loaded_modules
            .insert(name.to_string(), func_names.to_vec());
        Ok(())
    }

    /// Load a PTX file from disk by path.
    pub fn load_ptx_file(&mut self, name: &str, path: &Path) -> Result<()> {
        let func_names = self.resolve_function_names(name);
        let ptx = Ptx::from_file(path);

        self.device.load_ptx(ptx, name, &func_names).map_err(|e| {
            crate::LLMError::GpuError(format!("failed to load PTX file '{}': {e}", path.display()))
        })?;

        debug!(module = name, path = %path.display(), "loaded PTX file");
        self.loaded_modules
            .insert(name.to_string(), func_names.to_vec());
        Ok(())
    }

    /// Retrieve a loaded CUDA function by module and function name.
    pub fn get_func(&self, module: &str, function: &str) -> Result<CudaFunction> {
        self.device.get_func(module, function).ok_or_else(|| {
            crate::LLMError::GpuError(format!(
                "function '{function}' not found in module '{module}'"
            ))
        })
    }

<<<<<<< Updated upstream
    /// Like get_func but panics if the kernel is missing.
    /// Use this for kernels that MUST be available -- no silent fallbacks.
    pub fn require_func(&self, module: &str, function: &str) -> CudaFunction {
        self.get_func(module, function).unwrap_or_else(|e| {
            panic!("REQUIRED kernel missing: {module}::{function} -- {e}. \
                    All required kernels must be compiled and loadable. \
                    Run `bash kernels/build.sh` to compile kernel PTX files.")
        })
    }

    /// Validate that all required kernels for decode inference are loaded.
    /// Panics with a clear message listing every missing kernel.
    pub fn validate_required_kernels(&self) {
        let required: &[(&str, &str)] = &[
            // Attention
            ("flash_attention_3_v3", "fa3_v3_decode_gqa_kernel"),
            ("flash_attention_3_v3", "fa3_v3_decode_kernel"),
            ("flash_attention_3_v3", "fa3_v3_combine_f16_kernel"),
            // Fused GEMV (T=1 decode)
            ("fused_add_norm_qkv_gemv", "fused_cute_add_norm_qkv_gemv"),
            ("fused_add_norm_qkv_gemv", "fused_cute_norm_qkv_gemv"),
            ("fused_add_norm_qkv_gemv", "fused_cute_add_norm_qkv_bias_gemv"),
            ("fused_add_norm_qkv_gemv", "fused_cute_norm_qkv_bias_gemv"),
            ("fused_add_norm_gateup_gemv", "fused_cute_add_norm_gateup_gemv"),
            ("fused_silu_down_gemv", "fused_cute_silu_down_gemv"),
            ("fused_oproj_add_norm_gateup_gemv", "fused_cute_oproj_add_norm_gateup_gemv"),
            // Fused ops
            ("fused_rope_cache", "fused_rope_cache_f16_kernel"),
            ("fused_residual_rmsnorm_f16", "fused_residual_rmsnorm_f16_kernel"),
            // T>1 MLP
            ("silu_mul_interleaved", "silu_mul_interleaved_f16_kernel"),
            ("deinterleave_qkv", "deinterleave_qkv_f16_kernel"),
            ("add_bias_broadcast", "add_bias_broadcast_f16_kernel"),
            // Core
            ("rms_norm_f16", "rms_norm_f16_kernel"),
            ("activation_f16", "fused_silu_mul_f16_kernel"),
            ("embedding_gather_f16", "embedding_gather_f16_kernel"),
            ("reshape_and_cache_f16", "reshape_and_cache_f16io_kernel"),
            ("rotary_embedding_f16", "rotary_embedding_f16_kernel"),
            ("add_bias_f16", "add_bias_f16_kernel"),
            ("add_bias_f16", "add_f16_kernel"),
        ];

        let mut missing = Vec::new();
        for &(module, function) in required {
            if !self.has_func(module, function) {
                missing.push(format!("  {module}::{function}"));
            }
        }
        if !missing.is_empty() {
            panic!(
                "FATAL: {} required kernel(s) missing:\n{}\n\
                 Run `bash kernels/build.sh` to compile all kernel PTX files.",
                missing.len(),
                missing.join("\n")
            );
        }
        info!("All {} required kernels validated", required.len());
    }

    /// Check if a module has been loaded (PTX or cubin).
=======
    /// Check if a module has been loaded.
>>>>>>> Stashed changes
    pub fn has_module(&self, module: &str) -> bool {
        self.loaded_modules.contains_key(module)
    }

    /// Check if a specific function is available.
    pub fn has_func(&self, module: &str, function: &str) -> bool {
        self.device.has_func(module, function)
    }

    /// Launch a kernel on the device's default stream.
    ///
    /// # Safety
    /// The caller must ensure that the kernel arguments match the kernel signature
    /// exactly: correct types, correct count, correct mutability for output buffers.
    /// See `cudarc::driver::LaunchAsync` for full safety requirements.
    pub unsafe fn launch_raw(
        &self,
        module: &str,
        function: &str,
        cfg: LaunchConfig,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        let func = self.get_func(module, function)?;
        // SAFETY: caller guarantees args match the kernel signature
        func.launch(cfg, args).map_err(|e| {
            crate::LLMError::GpuError(format!("kernel launch {module}::{function} failed: {e}"))
        })
    }

    /// Launch a kernel on a specific stream.
    ///
    /// # Safety
    /// Same requirements as `launch_raw`, plus the caller must ensure no
    /// data races between this stream and other concurrent streams.
    pub unsafe fn launch_on_stream_raw(
        &self,
        module: &str,
        function: &str,
        cfg: LaunchConfig,
        stream: &CudaStream,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        let func = self.get_func(module, function)?;
        // SAFETY: caller guarantees args match the kernel signature and
        // there are no data races on this stream
        func.launch_on_stream(stream, cfg, args).map_err(|e| {
            crate::LLMError::GpuError(format!(
                "kernel launch {module}::{function} on stream failed: {e}"
            ))
        })
    }

    /// Returns a reference to the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

<<<<<<< Updated upstream
    /// Launch a cubin kernel cooperatively (all blocks must be co-resident).
    ///
    /// Uses `cuLaunchCooperativeKernel` so grid-level synchronization via
    /// `cooperative_groups::grid_group::sync()` is available inside the kernel.
    ///
    /// # Safety
    /// Caller must ensure kernel args exactly match the kernel signature.
    pub unsafe fn launch_cooperative_cubin(
        &self,
        module: &str,
        function: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        let cu_func = self.get_cubin_func(module, function)?;
        self.context
            .bind_to_thread()
            .map_err(|e| crate::LLMError::GpuError(format!("CUDA bind failed: {e}")))?;
        crate::cooperative::launch_cooperative(cu_func, grid, block, shared_mem, self.stream.cu_stream(), args)
            .map_err(|e| crate::LLMError::GpuError(format!(
                "cooperative cubin launch {module}::{function} failed: {e}"
            )))
    }

    /// Returns a reference to the underlying CUDA context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Returns a reference to the underlying CUDA stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Returns a reference to the underlying CUDA device (context alias).
    /// Kept for backward compatibility during migration.
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// List all loaded module names (PTX and cubin).
=======
    /// List all loaded module names.
>>>>>>> Stashed changes
    pub fn loaded_modules(&self) -> Vec<&str> {
        self.loaded_modules.keys().map(|s| s.as_str()).collect()
    }

    // --- private helpers ---

    /// Scan a directory for .ptx files and load each one.
    fn load_directory(&mut self, dir: &Path) -> Result<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            crate::LLMError::GpuError(format!("cannot read kernel dir '{}': {e}", dir.display()))
        })?;

        let mut count = 0u32;
        for entry in entries {
            let entry =
                entry.map_err(|e| crate::LLMError::GpuError(format!("readdir error: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("ptx") {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                match self.load_ptx_file(stem, &path) {
                    Ok(()) => count += 1,
                    Err(e) => {
                        warn!(module = stem, error = %e, "skipping PTX file that failed to load");
                    }
                }
            }
        }

        info!(dir = %dir.display(), count, "loaded PTX files from directory");
        Ok(())
    }

    /// Resolve function names for a known kernel module, or fall back to
    /// a convention-based guess (`{name}_kernel`).
    fn resolve_function_names(&self, name: &str) -> Vec<&'static str> {
        for &(module_name, funcs) in KERNEL_FUNCTIONS {
            if module_name == name {
                return funcs.to_vec();
            }
        }
        // Convention fallback: leak a string so we get &'static str.
        // This is intentional -- kernel names live for the process lifetime.
        let fallback: &'static str = Box::leak(format!("{name}_kernel").into_boxed_str());
        trace!(
            module = name,
            fallback,
            "using convention-based function name"
        );
        vec![fallback]
    }
}

/// Helper to build a `LaunchConfig` from grid/block tuples and shared memory size.
pub fn launch_config(
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: shared_mem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn resolve_known_kernel_names() {
        let device = CudaDevice::new(0).unwrap();
        let loader = KernelLoader::empty(device);

        let names = loader.resolve_function_names("activation");
        assert_eq!(
            names,
            &["silu_kernel", "fused_silu_mul_kernel", "gelu_kernel"]
        );

        let names = loader.resolve_function_names("rms_norm");
        assert_eq!(names, &["rms_norm_kernel"]);
    }

    #[test]
    fn resolve_unknown_uses_convention() {
        let device = CudaDevice::new(0).unwrap();
        let loader = KernelLoader::empty(device);

        let names = loader.resolve_function_names("my_custom");
        assert_eq!(names, &["my_custom_kernel"]);
    }

    #[test]
    fn empty_loader_has_no_modules() {
        let device = CudaDevice::new(0).unwrap();
        let loader = KernelLoader::empty(device);
        assert!(loader.loaded_modules().is_empty());
        assert!(!loader.has_module("anything"));
    }

    #[test]
    fn new_with_nonexistent_dir() {
        let device = CudaDevice::new(0).unwrap();
        let loader = KernelLoader::new(device, &PathBuf::from("/nonexistent/path")).unwrap();
        assert!(loader.loaded_modules().is_empty());
    }
}
