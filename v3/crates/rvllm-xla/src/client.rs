#![cfg(feature = "tpu")]

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use libloading::{Library, Symbol};
use rvllm_core::{ConfigError, Result, RvllmError};

use crate::ffi::*;

pub struct PjrtClientInner {
    _lib: Library,
    fns: PjrtApiFns,
    client: *mut PjrtClient,
    devices: Vec<*mut PjrtDevice>,
    compile_options: Option<Vec<u8>>,
}

unsafe impl Send for PjrtClientInner {}
unsafe impl Sync for PjrtClientInner {}

#[derive(Clone)]
pub struct PjrtClientHandle {
    inner: Arc<PjrtClientInner>,
}

impl PjrtClientHandle {
    pub fn new() -> Result<Self> {
        let lib = unsafe {
            Library::new("libtpu.so")
                .map_err(|e| xla_err(format!("failed to dlopen libtpu.so: {e}")))?
        };
        let api_ptr = unsafe {
            let get_api: Symbol<GetPjrtApiFn> = lib
                .get(b"GetPjrtApi")
                .map_err(|e| xla_err(format!("libtpu.so missing GetPjrtApi: {e}")))?;
            get_api()
        };
        if api_ptr.is_null() {
            return Err(xla_err("GetPjrtApi returned null"));
        }
        let fns = unsafe { PjrtApiFns::from_api_ptr(api_ptr) };

        unsafe {
            let mut args = PJRT_Plugin_Initialize_Args {
                struct_size: std::mem::size_of::<PJRT_Plugin_Initialize_Args>(),
                extension_start: ptr::null_mut(),
            };
            let err = (fns.plugin_initialize)(&mut args);
            if !err.is_null() {
                return Err(xla_err(format!(
                    "PJRT_Plugin_Initialize failed: {}",
                    extract_error_message(&fns, err)
                )));
            }
        }

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
                kv_try_get_callback: ptr::null(),
                kv_try_get_user_arg: ptr::null_mut(),
            };
            let err = (fns.client_create)(&mut args);
            if !err.is_null() {
                return Err(xla_err(format!(
                    "PJRT_Client_Create failed: {}",
                    extract_error_message(&fns, err)
                )));
            }
            if args.client.is_null() {
                return Err(xla_err("PJRT_Client_Create returned null client"));
            }
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
                return Err(xla_err(format!(
                    "PJRT_Client_Devices failed: {}",
                    extract_error_message(&fns, err)
                )));
            }
            if args.num_devices == 0 {
                return Err(xla_err("PJRT reports 0 devices"));
            }
            std::slice::from_raw_parts(args.devices, args.num_devices).to_vec()
        };

        Ok(Self {
            inner: Arc::new(PjrtClientInner {
                _lib: lib,
                fns,
                client,
                devices,
                compile_options: None,
            }),
        })
    }

    pub fn num_devices(&self) -> usize {
        self.inner.devices.len()
    }

    pub fn set_compile_options(&mut self, opts: Vec<u8>) -> Result<()> {
        let inner = Arc::get_mut(&mut self.inner)
            .ok_or_else(|| xla_err("cannot set compile options after cloning PJRT client"))?;
        inner.compile_options = Some(opts);
        Ok(())
    }

    pub fn compile(&self, mlir_text: &str) -> Result<CompiledExecutable> {
        self.compile_bytes(mlir_text.as_bytes())
    }

    pub fn compile_bytes(&self, code: &[u8]) -> Result<CompiledExecutable> {
        let format = b"mlir";
        let program = PJRT_Program {
            struct_size: std::mem::size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: code.as_ptr(),
            code_size: code.len(),
            format: format.as_ptr(),
            format_size: format.len(),
        };
        let default_opts: [u8; 6] = [0x22, 0x04, 0x18, 0x01, 0x20, 0x01];
        let opts = self
            .inner
            .compile_options
            .as_deref()
            .unwrap_or(&default_opts);
        let raw = unsafe {
            let mut args = PJRT_Client_Compile_Args {
                struct_size: std::mem::size_of::<PJRT_Client_Compile_Args>(),
                extension_start: ptr::null_mut(),
                client: self.inner.client,
                program: &program,
                compile_options: opts.as_ptr(),
                compile_options_size: opts.len(),
                executable: ptr::null_mut(),
            };
            let err = (self.inner.fns.client_compile)(&mut args);
            if !err.is_null() {
                return Err(xla_err(format!(
                    "PJRT_Client_Compile failed: {}",
                    extract_error_message(&self.inner.fns, err)
                )));
            }
            if args.executable.is_null() {
                return Err(xla_err("PJRT_Client_Compile returned null executable"));
            }
            args.executable
        };
        Ok(CompiledExecutable {
            client: self.clone(),
            raw,
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
            return Err(xla_err(format!(
                "device index {device_idx} out of range, have {}",
                self.inner.devices.len()
            )));
        }
        let raw = unsafe {
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
                host_buffer_semantics: PjrtHostBufferSemantics::ImmutableUntilTransferCompletes,
                device: self.inner.devices[device_idx],
                memory: ptr::null_mut(),
                _layout: ptr::null_mut(),
                done_with_host_buffer: ptr::null_mut(),
                buffer: ptr::null_mut(),
            };
            let err = (self.inner.fns.client_buffer_from_host)(&mut args);
            if !err.is_null() {
                return Err(xla_err(format!(
                    "PJRT_Client_BufferFromHostBuffer failed: {}",
                    extract_error_message(&self.inner.fns, err)
                )));
            }
            if args.buffer.is_null() {
                return Err(xla_err(
                    "PJRT_Client_BufferFromHostBuffer returned null buffer",
                ));
            }
            args.buffer
        };
        Ok(PjrtBufferHandle {
            client: self.clone(),
            raw,
        })
    }

    pub fn execute(
        &self,
        exe: &CompiledExecutable,
        inputs: &[&PjrtBufferHandle],
    ) -> Result<Vec<PjrtBufferHandle>> {
        let input_ptrs: Vec<*mut PjrtBuffer> = inputs.iter().map(|b| b.raw).collect();
        let input_list: *const *mut PjrtBuffer = input_ptrs.as_ptr();
        let mut output_ptrs: Vec<*mut PjrtBuffer> = vec![ptr::null_mut(); 256];
        let mut output_list: *mut *mut PjrtBuffer = output_ptrs.as_mut_ptr();
        let exec_options = PJRT_ExecuteOptions {
            struct_size: std::mem::size_of::<PJRT_ExecuteOptions>(),
            extension_start: ptr::null_mut(),
            send_callbacks: ptr::null(),
            recv_callbacks: ptr::null(),
            num_send_ops: 0,
            num_recv_ops: 0,
            launch_id: 0,
            non_donatable_input_indices: ptr::null(),
            num_non_donatable_input_indices: 0,
            context: ptr::null(),
            call_location: ptr::null(),
            num_tasks: 0,
            task_ids: ptr::null(),
            incarnation_ids: ptr::null(),
        };

        unsafe {
            let mut args = PJRT_LoadedExecutable_Execute_Args {
                struct_size: std::mem::size_of::<PJRT_LoadedExecutable_Execute_Args>(),
                extension_start: ptr::null_mut(),
                executable: exe.raw,
                options: &exec_options,
                argument_lists: &input_list as *const *const *mut PjrtBuffer,
                num_devices: 1,
                num_args: input_ptrs.len(),
                output_lists: &mut output_list as *const *mut *mut PjrtBuffer,
                device_complete_events: ptr::null_mut(),
                execute_device: ptr::null_mut(),
            };
            let err = (self.inner.fns.loaded_executable_execute)(&mut args);
            if !err.is_null() {
                return Err(xla_err(format!(
                    "PJRT_LoadedExecutable_Execute failed: {}",
                    extract_error_message(&self.inner.fns, err)
                )));
            }
        }

        let num_outputs = output_ptrs.iter().take_while(|p| !p.is_null()).count();
        Ok(output_ptrs[..num_outputs]
            .iter()
            .map(|&raw| PjrtBufferHandle {
                client: self.clone(),
                raw,
            })
            .collect())
    }

    pub fn buffer_to_host(&self, buf: &PjrtBufferHandle, dst: &mut [u8]) -> Result<()> {
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
                return Err(xla_err(format!(
                    "PJRT_Buffer_ToHostBuffer failed: {}",
                    extract_error_message(&self.inner.fns, err)
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
                return Err(xla_err(format!(
                    "PJRT_Event_Await failed: {}",
                    extract_error_message(&self.inner.fns, err)
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
                let _ = (self.client.inner.fns.buffer_destroy)(&mut args);
            }
        }
    }
}

fn xla_err(reason: impl Into<String>) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: "xla",
            reason: reason.into(),
        },
        "rvllm_xla",
    )
}
