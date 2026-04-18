#![cfg(feature = "tpu")]

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use libloading::{Library, Symbol};
use tracing::info;

use crate::ffi::*;
use crate::{LLMError, Result};

pub struct PjrtClientInner {
    _lib: Library,
    fns: PjrtApiFns,
    client: *mut PjrtClient,
    devices: Vec<*mut PjrtDevice>,
}

unsafe impl Send for PjrtClientInner {}
unsafe impl Sync for PjrtClientInner {}

impl Drop for PjrtClientInner {
    fn drop(&mut self) {
        // PJRT_Client_Destroy would go here, but the client typically
        // lives for the entire process lifetime. Dropping the Library
        // handle unloads libtpu.so.
    }
}

#[derive(Clone)]
pub struct PjrtClientHandle {
    inner: Arc<PjrtClientInner>,
}

impl PjrtClientHandle {
    pub fn new() -> Result<Self> {
        let lib = unsafe {
            Library::new("libtpu.so").map_err(|e| {
                LLMError::GpuError(format!(
                    "failed to dlopen libtpu.so: {e}. \
                     Ensure libtpu is installed (pip install libtpu-nightly or Cloud TPU VM)."
                ))
            })?
        };

        let api_ptr = unsafe {
            let get_api: Symbol<GetPjrtApiFn> =
                lib.get(b"GetPjrtApi").map_err(|e| {
                    LLMError::GpuError(format!(
                        "libtpu.so missing GetPjrtApi symbol: {e}"
                    ))
                })?;
            get_api()
        };

        if api_ptr.is_null() {
            return Err(LLMError::GpuError(
                "GetPjrtApi() returned null".into(),
            ));
        }

        let struct_size = unsafe { (*api_ptr).struct_size };
        if struct_size == 0 {
            return Err(LLMError::GpuError(
                "PJRT_Api struct_size is 0 -- invalid API table".into(),
            ));
        }

        let fns = unsafe { PjrtApiFns::from_api_ptr(api_ptr) };

        info!(
            major = unsafe { (*api_ptr).pjrt_api_version_major },
            minor = unsafe { (*api_ptr).pjrt_api_version_minor },
            "loaded PJRT API from libtpu.so"
        );

        let client = unsafe {
            let mut args = PJRT_Client_Create_Args {
                struct_size: std::mem::size_of::<PJRT_Client_Create_Args>(),
                extension_start: ptr::null_mut(),
                create_options: ptr::null(),
                num_options: 0,
                kv_get_callback: ptr::null(),
                kv_get_user_arg: ptr::null_mut(),
                kv_put_callback: ptr::null(),
                kv_put_user_arg: ptr::null_mut(),
                client: ptr::null_mut(),
            };
            let err = (fns.client_create)(&mut args);
            if !err.is_null() {
                let msg = extract_error_message(&fns, err);
                return Err(LLMError::GpuError(format!(
                    "PJRT_Client_Create failed: {msg}"
                )));
            }
            assert!(!args.client.is_null(), "PJRT_Client_Create returned null client");
            args.client
        };

        let devices = unsafe {
            let mut args = PJRT_Client_Devices_Args {
                struct_size: std::mem::size_of::<PJRT_Client_Devices_Args>(),
                extension_start: ptr::null_mut(),
                client,
                devices: ptr::null(),
                num_devices: 0,
            };
            let err = (fns.client_devices)(&mut args);
            if !err.is_null() {
                let msg = extract_error_message(&fns, err);
                return Err(LLMError::GpuError(format!(
                    "PJRT_Client_Devices failed: {msg}"
                )));
            }
            if args.num_devices == 0 {
                return Err(LLMError::GpuError(
                    "PJRT reports 0 devices -- no TPUs found".into(),
                ));
            }
            std::slice::from_raw_parts(args.devices, args.num_devices).to_vec()
        };

        info!(num_devices = devices.len(), "PJRT client initialized");

        Ok(Self {
            inner: Arc::new(PjrtClientInner {
                _lib: lib,
                fns,
                client,
                devices,
            }),
        })
    }

    pub fn num_devices(&self) -> usize {
        self.inner.devices.len()
    }

    pub fn compile(&self, mlir_text: &str) -> Result<CompiledExecutable> {
        let format = b"mlir";
        let code = mlir_text.as_bytes();

        let program = PJRT_Program {
            struct_size: std::mem::size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: code.as_ptr(),
            code_size: code.len(),
            format: format.as_ptr(),
            format_size: format.len(),
        };

        let exe = unsafe {
            let mut args = PJRT_Client_Compile_Args {
                struct_size: std::mem::size_of::<PJRT_Client_Compile_Args>(),
                extension_start: ptr::null_mut(),
                client: self.inner.client,
                program: &program,
                compile_options: ptr::null(),
                compile_options_size: 0,
                executable: ptr::null_mut(),
            };
            let err = (self.inner.fns.client_compile)(&mut args);
            if !err.is_null() {
                let msg = extract_error_message(&self.inner.fns, err);
                return Err(LLMError::GpuError(format!(
                    "PJRT_Client_Compile failed: {msg}"
                )));
            }
            assert!(!args.executable.is_null(), "PJRT_Client_Compile returned null");
            args.executable
        };

        Ok(CompiledExecutable {
            client: self.clone(),
            raw: exe,
        })
    }

    pub fn buffer_from_host(
        &self,
        data: &[u8],
        shape: &[i64],
        dtype: PjrtElementType,
        device_idx: usize,
    ) -> Result<PjrtBufferHandle> {
        if device_idx >= self.inner.devices.len() {
            return Err(LLMError::GpuError(format!(
                "device index {device_idx} out of range (have {})",
                self.inner.devices.len()
            )));
        }
        let device = self.inner.devices[device_idx];

        let buf = unsafe {
            let mut args = PJRT_Client_BufferFromHostBuffer_Args {
                struct_size: std::mem::size_of::<PJRT_Client_BufferFromHostBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client: self.inner.client,
                data: data.as_ptr() as *const c_void,
                type_: dtype,
                dims: shape.as_ptr(),
                num_dims: shape.len(),
                byte_strides: ptr::null(),
                num_byte_strides: 0,
                host_buffer_semantics:
                    PjrtHostBufferSemantics::ImmutableUntilTransferCompletes,
                device,
                memory: ptr::null_mut(),
                buffer: ptr::null_mut(),
                done_with_host_buffer: ptr::null_mut(),
            };
            let err = (self.inner.fns.client_buffer_from_host)(&mut args);
            if !err.is_null() {
                let msg = extract_error_message(&self.inner.fns, err);
                return Err(LLMError::GpuError(format!(
                    "PJRT_Client_BufferFromHostBuffer failed: {msg}"
                )));
            }
            assert!(!args.buffer.is_null());

            // Wait for transfer to complete before returning
            if !args.done_with_host_buffer.is_null() {
                self.await_event(args.done_with_host_buffer)?;
            }

            args.buffer
        };

        Ok(PjrtBufferHandle {
            client: self.clone(),
            raw: buf,
        })
    }

    pub fn execute(
        &self,
        exe: &CompiledExecutable,
        inputs: &[&PjrtBufferHandle],
    ) -> Result<Vec<PjrtBufferHandle>> {
        let input_ptrs: Vec<*mut PjrtBuffer> =
            inputs.iter().map(|b| b.raw).collect();
        let input_list: *const *mut PjrtBuffer = input_ptrs.as_ptr();

        // Prepare output list -- PJRT allocates output buffers
        let mut output_ptrs: Vec<*mut PjrtBuffer> = vec![ptr::null_mut(); 16];
        let mut output_list: *mut *mut PjrtBuffer = output_ptrs.as_mut_ptr();

        let exec_options = PJRT_ExecuteOptions {
            struct_size: std::mem::size_of::<PJRT_ExecuteOptions>(),
            extension_start: ptr::null_mut(),
            launch_id: 0,
            non_donatable_input_indices: ptr::null(),
            num_non_donatable_input_indices: 0,
        };

        let num_outputs = unsafe {
            let mut output_size: usize = 0;
            let mut args = PJRT_LoadedExecutable_Execute_Args {
                struct_size: std::mem::size_of::<PJRT_LoadedExecutable_Execute_Args>(),
                extension_start: ptr::null_mut(),
                executable: exe.raw,
                options: &exec_options,
                argument_lists: &input_list as *const *const *mut PjrtBuffer,
                num_devices: 1,
                num_args: input_ptrs.len(),
                output_lists: &mut output_list as *const *mut *mut PjrtBuffer,
                output_sizes: &mut output_size,
                device_complete_events: ptr::null_mut(),
                execute_device: ptr::null_mut(),
            };
            let err = (self.inner.fns.loaded_executable_execute)(&mut args);
            if !err.is_null() {
                let msg = extract_error_message(&self.inner.fns, err);
                return Err(LLMError::GpuError(format!(
                    "PJRT_LoadedExecutable_Execute failed: {msg}"
                )));
            }
            output_size
        };

        let results: Vec<PjrtBufferHandle> = output_ptrs[..num_outputs]
            .iter()
            .map(|&raw| PjrtBufferHandle {
                client: self.clone(),
                raw,
            })
            .collect();

        Ok(results)
    }

    pub fn buffer_to_host(
        &self,
        buf: &PjrtBufferHandle,
        dst: &mut [u8],
    ) -> Result<()> {
        unsafe {
            let mut args = PJRT_Buffer_ToHostBuffer_Args {
                struct_size: std::mem::size_of::<PJRT_Buffer_ToHostBuffer_Args>(),
                extension_start: ptr::null_mut(),
                src: buf.raw,
                host_layout: ptr::null(),
                dst: dst.as_mut_ptr() as *mut c_void,
                dst_size: dst.len(),
                event: ptr::null_mut(),
            };
            let err = (self.inner.fns.buffer_to_host)(&mut args);
            if !err.is_null() {
                let msg = extract_error_message(&self.inner.fns, err);
                return Err(LLMError::GpuError(format!(
                    "PJRT_Buffer_ToHostBuffer failed: {msg}"
                )));
            }
            if !args.event.is_null() {
                self.await_event(args.event)?;
            }
        }
        Ok(())
    }

    fn await_event(&self, event: *mut PjrtEvent) -> Result<()> {
        unsafe {
            let mut args = PJRT_Event_Await_Args {
                struct_size: std::mem::size_of::<PJRT_Event_Await_Args>(),
                extension_start: ptr::null_mut(),
                event,
            };
            let err = (self.inner.fns.event_await)(&mut args);
            if !err.is_null() {
                let msg = extract_error_message(&self.inner.fns, err);
                return Err(LLMError::GpuError(format!(
                    "PJRT_Event_Await failed: {msg}"
                )));
            }
            let mut destroy = PJRT_Event_Destroy_Args {
                struct_size: std::mem::size_of::<PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event,
            };
            (self.inner.fns.event_destroy)(&mut destroy);
        }
        Ok(())
    }
}

pub struct CompiledExecutable {
    client: PjrtClientHandle,
    pub(crate) raw: *mut PjrtLoadedExecutable,
}

impl CompiledExecutable {
    pub fn client(&self) -> &PjrtClientHandle {
        &self.client
    }
}

unsafe impl Send for CompiledExecutable {}
unsafe impl Sync for CompiledExecutable {}

pub struct PjrtBufferHandle {
    client: PjrtClientHandle,
    pub(crate) raw: *mut PjrtBuffer,
}

impl PjrtBufferHandle {
    pub fn client(&self) -> &PjrtClientHandle {
        &self.client
    }
}

unsafe impl Send for PjrtBufferHandle {}
unsafe impl Sync for PjrtBufferHandle {}

impl Drop for PjrtBufferHandle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let mut args = PJRT_Buffer_Destroy_Args {
                    struct_size: std::mem::size_of::<PJRT_Buffer_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    buffer: self.raw,
                };
                // Ignore error on destroy -- nothing useful we can do
                let _ = (self.client.inner.fns.buffer_destroy)(&mut args);
            }
        }
    }
}
