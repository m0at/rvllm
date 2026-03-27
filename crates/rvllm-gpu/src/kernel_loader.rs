//! CUDA kernel loader: loads PTX files and launches kernels via cudarc.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaStream, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use tracing::{debug, info, trace};

use crate::Result;

/// Known kernel -> function-name mappings for the vllm-rs kernel set.
/// These are the `extern "C" __global__` entry points in each .cu file.
static KERNEL_FUNCTIONS: &[(&str, &[&str])] = &[
    ("activation", &["silu_kernel", "fused_silu_mul_kernel", "gelu_kernel"]),
    ("copy_blocks", &["copy_blocks_kernel"]),
    ("paged_attention", &["paged_attention_v2_kernel"]),
    ("rms_norm", &["rms_norm_kernel"]),
    ("rotary_embedding", &["rotary_embedding_kernel"]),
    ("softmax", &["softmax_kernel"]),
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

        self.device
            .load_ptx(ptx, name, func_names)
            .map_err(|e| {
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

        self.device
            .load_ptx(ptx, name, &func_names)
            .map_err(|e| {
                crate::LLMError::GpuError(format!(
                    "failed to load PTX file '{}': {e}",
                    path.display()
                ))
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

    /// Check if a module has been loaded.
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

    /// List all loaded module names.
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
            let entry = entry.map_err(|e| {
                crate::LLMError::GpuError(format!("readdir error: {e}"))
            })?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("ptx") {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                self.load_ptx_file(stem, &path)?;
                count += 1;
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
        trace!(module = name, fallback, "using convention-based function name");
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
        assert_eq!(names, &["silu_kernel", "fused_silu_mul_kernel", "gelu_kernel"]);

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
