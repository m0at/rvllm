//! CUDA kernel loader: loads PTX files and launches kernels via cudarc.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::nvrtc::Ptx;
use tracing::{debug, info, trace};

/// Extract the raw CUfunction handle from a CudaFunction.
/// CudaFunction's first field is `cu_function: sys::CUfunction` (a *mut CUfunc_st).
/// SAFETY: CudaFunction is a non-repr(C) struct but the first field is always at offset 0 for
/// a struct whose first field is a pointer (no padding before it). This is an implementation
/// detail of cudarc 0.19 and is only used in unsafe raw-launch paths.
unsafe fn extract_cu_function(func: &CudaFunction) -> cudarc::driver::sys::CUfunction {
    std::ptr::read((func as *const CudaFunction).cast::<cudarc::driver::sys::CUfunction>())
}

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
    ("add_bias", &["add_bias_kernel", "add_kernel"]),
    (
        "add_bias_f16",
        &["add_bias_f16_kernel", "add_f16_kernel", "add_inplace_f16_kernel"],
    ),
    ("copy_blocks", &["copy_blocks_kernel"]),
    ("embedding_gather", &["embedding_gather_kernel"]),
    ("embedding_gather_f16", &["embedding_gather_f16_kernel"]),
    (
        "flash_attention",
        &[
            "flash_attention_2_kernel",
            "flash_attention_2_decode_kernel",
            "flash_attention_2_f16kv_kernel",
            "flash_attention_2_decode_f16kv_kernel",
            "flash_attention_2_decode_f16io_kernel",
        ],
    ),
    (
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
        "fp8_kv",
        &[
            "quantize_kv_kernel",
            "dequantize_kv_kernel",
            "quantize_paged_kv_kernel",
            "dequantize_paged_kv_kernel",
        ],
    ),
    ("fused_residual_rmsnorm", &["fused_residual_rmsnorm_kernel"]),
    ("fused_residual_rmsnorm_f16", &["fused_residual_rmsnorm_f16_kernel"]),
    (
        "paged_attention",
        &["paged_attention_v2_kernel", "paged_attention_v2_f16kv_kernel"],
    ),
    (
        "split_kv_attention",
        &[
            "split_kv_decode_f16kv_kernel",
            "split_kv_combine_kernel",
            "split_kv_decode_fp8kv_kernel",
            "split_kv_decode_single_f16kv_kernel",
            "split_kv_decode_f16io_kernel",
            "split_kv_combine_f16io_kernel",
        ],
    ),
    (
        "reshape_and_cache",
        &["reshape_and_cache_kernel", "reshape_and_cache_f16_kernel"],
    ),
    (
        "cast_fp",
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
        &[
            "gemv_fp8_kernel",
            "fused_add_norm_fp8_gemv_kernel",
            "fused_norm_fp8_gemv_kernel",
            "fused_silu_down_fp8_gemv_kernel",
        ],
    ),
    (
        "tma_gemv_fp16",
        &["tma_gemv_fp16_kernel", "tma_gemv_fp16_bias_kernel"],
    ),
    (
        "wgmma_gemv",
        &[
            "wgmma_gemv_f16_kernel",
            "wgmma_gemv_f16_unrolled_kernel",
            "wgmma_gemv_f16_splitk_kernel",
            "splitk_f32_to_f16_kernel",
        ],
    ),
    ("cutlass_qkv_bias", &["cutlass_qkv_bias_gemm"]),
    ("cutlass_oproj_residual", &["cutlass_oproj_residual_gemm"]),
    ("cutlass_gateup_silu", &["cutlass_gateup_gemm"]),
    ("cast_f16_to_fp8", &["cast_f16_to_fp8_kernel"]),
    ("jit_add_norm_qkv", &["fused_add_rmsnorm_gemv_3584x4608"]),
    ("jit_add_norm_gateup", &["fused_add_rmsnorm_gemv_3584x37888"]),
    ("jit_silu_down", &["fused_silu_mul_gemv_18944x3584"]),
    ("jit_norm_qkv", &["fused_rmsnorm_gemv_3584x4608"]),
    ("gemv_f16", &["gemv_f16_kernel", "gemv_batched_f16_kernel"]),
    ("rms_norm", &["rms_norm_kernel"]),
    ("rms_norm_f16", &["rms_norm_f16_kernel"]),
    ("rotary_embedding", &["rotary_embedding_kernel"]),
    ("rotary_embedding_f16", &["rotary_embedding_f16_kernel"]),
    ("softmax", &["softmax_kernel"]),
    (
        "fused_lm_head_argmax",
        &[
            "fused_lm_head_argmax_kernel",
            "fused_lm_head_argmax_reduce_kernel",
        ],
    ),
    (
        "fused_lm_head_argmax_f16",
        &["fused_lm_head_argmax_f16_kernel"],
    ),
];

/// Wrapper for a raw CUmodule loaded from cubin bytes.
/// Cubin modules can't go through cudarc's Ptx path because cudarc
/// compiles PTX source, but cubin is already compiled device code.
struct RawCubinModule {
    module: cudarc::driver::sys::CUmodule,
}

impl RawCubinModule {
    unsafe fn new(module: cudarc::driver::sys::CUmodule) -> Self {
        Self { module }
    }

    fn get_function(&self, name: &str) -> std::result::Result<cudarc::driver::sys::CUfunction, String> {
        let c_name = std::ffi::CString::new(name).map_err(|e| format!("invalid function name: {e}"))?;
        let mut func = std::mem::MaybeUninit::<cudarc::driver::sys::CUfunction>::uninit();
        let result = unsafe {
            cudarc::driver::sys::cuModuleGetFunction(
                func.as_mut_ptr(),
                self.module,
                c_name.as_ptr(),
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuModuleGetFunction('{}') failed: {:?}", name, result));
        }
        Ok(unsafe { func.assume_init() })
    }
}

impl Drop for RawCubinModule {
    fn drop(&mut self) {
        unsafe {
            cudarc::driver::sys::cuModuleUnload(self.module);
        }
    }
}

// SAFETY: CUmodule is a device handle that can be used from any thread
// as long as the CUDA context is current. KernelLoader ensures context
// binding before any operation.
unsafe impl Send for RawCubinModule {}
unsafe impl Sync for RawCubinModule {}

/// Loads and manages CUDA PTX modules, providing kernel launch capabilities.
///
/// Wraps `cudarc::driver::CudaContext` module management with a higher-level
/// API that understands the vllm-rs kernel naming conventions.
pub struct KernelLoader {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    modules: HashMap<String, Arc<CudaModule>>,
    cubin_modules: HashMap<String, RawCubinModule>,
    loaded_func_names: HashMap<String, Vec<&'static str>>,
}

impl KernelLoader {
    /// Create a new KernelLoader and load all .ptx files from `kernel_dir`.
    ///
    /// Falls back to the `RVLLM_KERNEL_DIR` environment variable if `kernel_dir`
    /// does not contain any .ptx files.
    pub fn new(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        kernel_dir: &Path,
    ) -> Result<Self> {
        let mut loader = Self {
            context,
            stream,
            modules: HashMap::new(),
            cubin_modules: HashMap::new(),
            loaded_func_names: HashMap::new(),
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
    pub fn empty(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
        Self {
            context,
            stream,
            modules: HashMap::new(),
            cubin_modules: HashMap::new(),
            loaded_func_names: HashMap::new(),
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

        let module = self.context.load_module(ptx).map_err(|e| {
            crate::LLMError::GpuError(format!("failed to load PTX module '{name}': {e}"))
        })?;

        debug!(module = name, functions = ?func_names, "loaded PTX module");
        self.modules.insert(name.to_string(), module);
        self.loaded_func_names
            .insert(name.to_string(), func_names.to_vec());
        Ok(())
    }

    /// Load a cubin module from raw bytes.
    ///
    /// Cubin (compiled binary) modules are required for kernels using
    /// cooperative_groups grid.sync() since PTX compilation downgrades
    /// grid-level sync to block-level bar.sync. The cubin must be compiled
    /// offline with `nvcc -arch=sm_XX -cubin`.
    ///
    /// `cuModuleLoadData` accepts both cubin and PTX, so we use the same
    /// driver call but skip the Ptx wrapper.
    pub fn load_cubin(&mut self, name: &str, cubin_bytes: &[u8]) -> Result<()> {
        let func_names = self.resolve_function_names(name);
        self.load_cubin_with_functions(name, cubin_bytes, &func_names)
    }

    /// Load a cubin module from raw bytes with explicit function names.
    pub fn load_cubin_with_functions(
        &mut self,
        name: &str,
        cubin_bytes: &[u8],
        func_names: &[&'static str],
    ) -> Result<()> {
        use std::ffi::c_void;

        self.context
            .bind_to_thread()
            .map_err(|e| crate::LLMError::GpuError(format!("CUDA bind for cubin load: {e}")))?;

        // cuModuleLoadData accepts cubin bytes directly
        let mut module_handle = std::mem::MaybeUninit::<cudarc::driver::sys::CUmodule>::uninit();
        let result = unsafe {
            cudarc::driver::sys::cuModuleLoadData(
                module_handle.as_mut_ptr(),
                cubin_bytes.as_ptr() as *const c_void,
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(crate::LLMError::GpuError(format!(
                "cuModuleLoadData failed for cubin '{}': {:?}", name, result
            )));
        }

        // Wrap the raw CUmodule in an Arc<CudaModule> via load_module with PTX
        // fallback: create a Ptx from a minimal valid PTX to get a CudaModule,
        // then we actually need the raw handle. Since cudarc doesn't expose
        // CudaModule::from_raw, we use the raw function loading path instead.
        //
        // Store the raw module handle and look up functions via cuModuleGetFunction.
        let raw_module = unsafe { module_handle.assume_init() };
        let wrapped = unsafe { RawCubinModule::new(raw_module) };
        debug!(module = name, functions = ?func_names, "loaded cubin module");
        self.cubin_modules.insert(name.to_string(), wrapped);
        self.loaded_func_names
            .insert(name.to_string(), func_names.to_vec());
        Ok(())
    }

    /// Load a cubin module from a file path.
    pub fn load_cubin_file(&mut self, name: &str, path: &Path) -> Result<()> {
        let bytes = std::fs::read(path).map_err(|e| {
            crate::LLMError::GpuError(format!(
                "failed to read cubin file '{}': {e}", path.display()
            ))
        })?;
        self.load_cubin(name, &bytes)
    }

    /// Load a PTX file from disk by path.
    pub fn load_ptx_file(&mut self, name: &str, path: &Path) -> Result<()> {
        let func_names = self.resolve_function_names(name);
        let ptx = Ptx::from_file(path);

        let module = self.context.load_module(ptx).map_err(|e| {
            crate::LLMError::GpuError(format!("failed to load PTX file '{}': {e}", path.display()))
        })?;

        debug!(module = name, path = %path.display(), "loaded PTX file");
        self.modules.insert(name.to_string(), module);
        self.loaded_func_names
            .insert(name.to_string(), func_names.to_vec());
        Ok(())
    }

    /// Retrieve a loaded CUDA function by module and function name.
    ///
    /// Checks PTX modules first, then cubin modules.
    pub fn get_func(&self, module: &str, function: &str) -> Result<CudaFunction> {
        if let Some(m) = self.modules.get(module) {
            return m.load_function(function).map_err(|e| {
                crate::LLMError::GpuError(format!(
                    "function '{function}' not found in module '{module}': {e}"
                ))
            });
        }
        // Cubin modules use raw CUfunction handles -- not wrapped in CudaFunction.
        // For cubin functions, use get_cubin_func() instead.
        Err(crate::LLMError::GpuError(format!("module '{module}' not loaded")))
    }

    /// Retrieve a raw CUfunction from a cubin module.
    pub fn get_cubin_func(
        &self,
        module: &str,
        function: &str,
    ) -> Result<cudarc::driver::sys::CUfunction> {
        let m = self.cubin_modules.get(module).ok_or_else(|| {
            crate::LLMError::GpuError(format!("cubin module '{module}' not loaded"))
        })?;
        m.get_function(function).map_err(|e| {
            crate::LLMError::GpuError(format!(
                "function '{function}' not found in cubin module '{module}': {e}"
            ))
        })
    }

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
    pub fn has_module(&self, module: &str) -> bool {
        self.modules.contains_key(module) || self.cubin_modules.contains_key(module)
    }

    /// Check if a specific function is available in a loaded module (PTX or cubin).
    pub fn has_func(&self, module: &str, function: &str) -> bool {
        if let Some(m) = self.modules.get(module) {
            return m.load_function(function).ok().is_some();
        }
        if let Some(m) = self.cubin_modules.get(module) {
            return m.get_function(function).is_ok();
        }
        false
    }

    /// Launch a kernel on the loader's stream.
    ///
    /// # Safety
    /// The caller must ensure that the kernel arguments match the kernel signature
    /// exactly: correct types, correct count, correct mutability for output buffers.
    pub unsafe fn launch_raw(
        &self,
        module: &str,
        function: &str,
        cfg: LaunchConfig,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        let func = self.get_func(module, function)?;
        self.context
            .bind_to_thread()
            .map_err(|e| crate::LLMError::GpuError(format!("CUDA bind failed: {e}")))?;
        let cu_func = extract_cu_function(&func);
        cudarc::driver::result::launch_kernel(
            cu_func,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.cu_stream(),
            args,
        )
        .map_err(|e| {
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
        self.context
            .bind_to_thread()
            .map_err(|e| crate::LLMError::GpuError(format!("CUDA bind failed: {e}")))?;
        let cu_func = extract_cu_function(&func);
        cudarc::driver::result::launch_kernel(
            cu_func,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            stream.cu_stream(),
            args,
        )
        .map_err(|e| {
            crate::LLMError::GpuError(format!(
                "kernel launch {module}::{function} on stream failed: {e}"
            ))
        })
    }

    /// Launch a cubin kernel on the loader's stream.
    ///
    /// For persistent kernels that need cooperative launch, use
    /// `get_cubin_func` + `cooperative::launch_cooperative` instead.
    ///
    /// # Safety
    /// Same requirements as `launch_raw`.
    pub unsafe fn launch_cubin_raw(
        &self,
        module: &str,
        function: &str,
        cfg: LaunchConfig,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        let cu_func = self.get_cubin_func(module, function)?;
        self.context
            .bind_to_thread()
            .map_err(|e| crate::LLMError::GpuError(format!("CUDA bind failed: {e}")))?;
        cudarc::driver::result::launch_kernel(
            cu_func,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.cu_stream(),
            args,
        )
        .map_err(|e| {
            crate::LLMError::GpuError(format!(
                "cubin kernel launch {module}::{function} failed: {e}"
            ))
        })
    }

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
    pub fn loaded_modules(&self) -> Vec<&str> {
        self.modules.keys()
            .chain(self.cubin_modules.keys())
            .map(|s| s.as_str())
            .collect()
    }

    // --- private helpers ---

    /// Scan a directory for .ptx and .cubin files and load each one.
    ///
    /// When both `foo.ptx` and `foo.cubin` exist for the same stem, the cubin
    /// takes priority (it preserves grid-level sync for cooperative kernels).
    fn load_directory(&mut self, dir: &Path) -> Result<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            crate::LLMError::GpuError(format!("cannot read kernel dir '{}': {e}", dir.display()))
        })?;

        // Collect all files, preferring .cubin over .ptx when both exist
        let mut ptx_files: HashMap<String, std::path::PathBuf> = HashMap::new();
        let mut cubin_files: HashMap<String, std::path::PathBuf> = HashMap::new();

        for entry in entries {
            let entry =
                entry.map_err(|e| crate::LLMError::GpuError(format!("readdir error: {e}")))?;
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str());
            let stem = path.file_stem().and_then(|s| s.to_str()).map(String::from);
            match (ext, stem) {
                (Some("ptx"), Some(s)) => { ptx_files.insert(s, path); }
                (Some("cubin"), Some(s)) => { cubin_files.insert(s, path); }
                _ => {}
            }
        }

        let mut count = 0u32;

        // Load cubin files first
        for (stem, path) in &cubin_files {
            self.load_cubin_file(stem, path)?;
            count += 1;
        }

        // Load PTX files that don't have a cubin counterpart
        for (stem, path) in &ptx_files {
            if !cubin_files.contains_key(stem) {
                self.load_ptx_file(stem, path)?;
                count += 1;
            } else {
                debug!(stem, "skipping PTX, cubin version loaded");
            }
        }

        info!(
            dir = %dir.display(), count,
            ptx = ptx_files.len(), cubin = cubin_files.len(),
            "loaded kernel files from directory"
        );
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
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let loader = KernelLoader::empty(context, stream);

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
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let loader = KernelLoader::empty(context, stream);

        let names = loader.resolve_function_names("my_custom");
        assert_eq!(names, &["my_custom_kernel"]);
    }

    #[test]
    fn empty_loader_has_no_modules() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let loader = KernelLoader::empty(context, stream);
        assert!(loader.loaded_modules().is_empty());
        assert!(!loader.has_module("anything"));
    }

    #[test]
    fn new_with_nonexistent_dir() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let loader = KernelLoader::new(context, stream, &PathBuf::from("/nonexistent/path")).unwrap();
        assert!(loader.loaded_modules().is_empty());
    }
}
