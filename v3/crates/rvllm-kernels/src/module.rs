//! `LoadedModule`: owns a `cuModule` + the kernel function handles
//! dlsym'd from it. Constructed by `KernelLoader::load_ptx`.

use core::marker::PhantomData;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

/// Kernel-function handle usable by `cuLaunchKernel`. Stores the raw
/// `CUfunction` as u64 so the type is the same under both features.
#[derive(Copy, Clone, Debug)]
pub struct KernelFn {
    pub(crate) raw: u64,
    pub(crate) name: &'static str,
}

impl KernelFn {
    pub fn raw(&self) -> u64 {
        self.raw
    }
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// A loaded PTX module. Holds the module handle; Drop calls
/// `cuModuleUnload`. `KernelFn`s obtained from it remain valid for
/// `Self`'s lifetime.
#[derive(Debug)]
pub struct LoadedModule {
    raw: u64,
    path: std::path::PathBuf,
    _not_send_sync: PhantomData<*const ()>,
}

impl LoadedModule {
    /// Load a PTX file at `path` into a CUDA module.
    #[cfg(feature = "cuda")]
    pub fn load_from_file(path: std::path::PathBuf) -> Result<Self> {
        use cudarc::driver::sys::*;
        let cpath = std::ffi::CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
            RvllmError::cuda(
                "CString::new(path)",
                CudaErrorKind::ModuleLoadFailed,
                CudaCtx::setup(),
            )
        })?;
        let mut module: CUmodule = core::ptr::null_mut();
        let r = unsafe { cuModuleLoad(&mut module, cpath.as_ptr()) };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuModuleLoad",
                CudaErrorKind::ModuleLoadFailed,
                CudaCtx {
                    stream: 0,
                    kernel: "cuModuleLoad",
                    launch: None,
                    device: -1,
                },
            ));
        }
        Ok(Self {
            raw: module as u64,
            path,
            _not_send_sync: PhantomData,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load_from_file(path: std::path::PathBuf) -> Result<Self> {
        // No-cuda stub: read the file so we fail loudly if missing.
        let _ = std::fs::read(&path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.clone(),
            source,
        })?;
        Ok(Self {
            raw: 0,
            path,
            _not_send_sync: PhantomData,
        })
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn raw(&self) -> u64 {
        self.raw
    }

    /// Resolve a kernel symbol (e.g. `"argmax_kernel"`) to a handle.
    pub fn get_function(&self, name: &'static str) -> Result<KernelFn> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::*;
            let cname = std::ffi::CString::new(name).map_err(|_| {
                RvllmError::cuda(
                    "CString::new(name)",
                    CudaErrorKind::ModuleLoadFailed,
                    CudaCtx::setup(),
                )
            })?;
            let mut f: CUfunction = core::ptr::null_mut();
            let r = unsafe { cuModuleGetFunction(&mut f, self.raw as CUmodule, cname.as_ptr()) };
            if r != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "cuModuleGetFunction",
                    CudaErrorKind::ModuleLoadFailed,
                    CudaCtx {
                        stream: 0,
                        kernel: name,
                        launch: None,
                        device: -1,
                    },
                ));
            }
            Ok(KernelFn {
                raw: f as u64,
                name,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(KernelFn { raw: 0, name })
        }
    }
}

impl Drop for LoadedModule {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            if self.raw != 0 {
                let _ =
                    cudarc::driver::sys::cuModuleUnload(self.raw as cudarc::driver::sys::CUmodule);
            }
        }
    }
}
