#![cfg(feature = "tpu")]

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use libloading::{Library, Symbol};
use rvllm_core::{ConfigError, Result, RvllmError};

use crate::executable::{
    PjrtExecutableSignature, PjrtHostBuffer, PjrtProgramFormat, PjrtTensorSpec,
};
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

impl Drop for PjrtClientInner {
    fn drop(&mut self) {
        if !self.client.is_null() {
            unsafe {
                let mut args = PJRT_Client_Destroy_Args {
                    struct_size: std::mem::size_of::<PJRT_Client_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    client: self.client,
                };
                let _ = (self.fns.client_destroy)(&mut args);
            }
            self.client = ptr::null_mut();
        }
    }
}

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
        self.compile_program_bytes(PjrtProgramFormat::Mlir, mlir_text.as_bytes(), None)
    }

    pub fn compile_bytes(&self, code: &[u8]) -> Result<CompiledExecutable> {
        self.compile_program_bytes(PjrtProgramFormat::Mlir, code, None)
    }

    pub fn compile_mlir_module_text(
        &self,
        mlir_text: &str,
        signature: PjrtExecutableSignature,
    ) -> Result<CompiledExecutable> {
        self.compile_module_text(mlir_text, PjrtProgramFormat::Mlir, signature)
    }

    pub fn compile_hlo_module_text(
        &self,
        hlo_text: &str,
        signature: PjrtExecutableSignature,
    ) -> Result<CompiledExecutable> {
        self.compile_module_text(hlo_text, PjrtProgramFormat::Hlo, signature)
    }

    pub fn compile_module_text(
        &self,
        module_text: &str,
        format: PjrtProgramFormat,
        signature: PjrtExecutableSignature,
    ) -> Result<CompiledExecutable> {
        signature.validate()?;
        self.compile_program_bytes(format, module_text.as_bytes(), Some(signature))
    }

    fn compile_program_bytes(
        &self,
        format: PjrtProgramFormat,
        code: &[u8],
        signature: Option<PjrtExecutableSignature>,
    ) -> Result<CompiledExecutable> {
        if code.is_empty() {
            return Err(xla_err("module text must not be empty"));
        }
        let format_bytes = format.as_pjrt_format();
        let program = PJRT_Program {
            struct_size: std::mem::size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: code.as_ptr(),
            code_size: code.len(),
            format: format_bytes.as_ptr(),
            format_size: format_bytes.len(),
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
            format,
            signature,
        })
    }

    pub fn buffer_from_host(
        &self,
        data: &[u8],
        shape: &[i64],
        dtype: PjrtElementType,
        device_idx: usize,
    ) -> Result<PjrtBufferHandle> {
        let spec = PjrtTensorSpec::anonymous(shape.to_vec(), dtype)?;
        self.buffer_from_host_spec(data, &spec, device_idx)
    }

    pub fn buffer_from_host_buffer(
        &self,
        host: &PjrtHostBuffer<'_>,
        device_idx: usize,
    ) -> Result<PjrtBufferHandle> {
        self.buffer_from_host_spec(host.data, &host.spec, device_idx)
    }

    pub fn buffer_from_host_spec(
        &self,
        data: &[u8],
        spec: &PjrtTensorSpec,
        device_idx: usize,
    ) -> Result<PjrtBufferHandle> {
        spec.validate_byte_len(data.len())?;
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
                type_: spec.dtype,
                dims: spec.shape.as_ptr(),
                num_dims: spec.shape.len(),
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
            spec: Some(spec.clone()),
            device_idx: Some(device_idx),
        })
    }

    pub fn execute(
        &self,
        exe: &CompiledExecutable,
        inputs: &[&PjrtBufferHandle],
    ) -> Result<Vec<PjrtBufferHandle>> {
        if let Some(signature) = exe.signature() {
            self.execute_with_signature(exe, inputs, signature)
        } else {
            self.execute_raw(exe, inputs, None)
        }
    }

    pub fn execute_with_buffers(
        &self,
        exe: &CompiledExecutable,
        inputs: &[&PjrtBufferHandle],
    ) -> Result<Vec<PjrtBufferHandle>> {
        let signature = exe
            .signature()
            .ok_or_else(|| xla_err("execute_with_buffers requires an executable signature"))?;
        self.execute_with_signature(exe, inputs, signature)
    }

    pub fn execute_with_signature(
        &self,
        exe: &CompiledExecutable,
        inputs: &[&PjrtBufferHandle],
        signature: &PjrtExecutableSignature,
    ) -> Result<Vec<PjrtBufferHandle>> {
        signature.validate()?;
        self.validate_input_specs(signature, inputs)?;
        self.execute_raw(exe, inputs, Some(&signature.outputs))
    }

    fn execute_raw(
        &self,
        exe: &CompiledExecutable,
        inputs: &[&PjrtBufferHandle],
        output_specs: Option<&[PjrtTensorSpec]>,
    ) -> Result<Vec<PjrtBufferHandle>> {
        self.validate_executable_client(exe)?;
        self.validate_input_clients(inputs)?;
        let input_ptrs: Vec<*mut PjrtBuffer> = inputs.iter().map(|b| b.raw).collect();
        let input_list: *const *mut PjrtBuffer = input_ptrs.as_ptr();
        let output_slots = output_specs.map_or(256, |specs| specs.len());
        let mut output_ptrs: Vec<*mut PjrtBuffer> = vec![ptr::null_mut(); output_slots];
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

        let num_outputs = if let Some(specs) = output_specs {
            if let Some((idx, _)) = output_ptrs.iter().enumerate().find(|(_, p)| p.is_null()) {
                self.destroy_raw_buffers(&output_ptrs);
                return Err(xla_err(format!(
                    "PJRT_LoadedExecutable_Execute returned null output {idx}"
                )));
            }
            specs.len()
        } else {
            output_ptrs.iter().take_while(|p| !p.is_null()).count()
        };
        Ok(output_ptrs[..num_outputs]
            .iter()
            .enumerate()
            .map(|(idx, &raw)| PjrtBufferHandle {
                client: self.clone(),
                raw,
                spec: output_specs.map(|specs| specs[idx].clone()),
                device_idx: inputs.first().and_then(|input| input.device_idx),
            })
            .collect())
    }

    pub fn buffer_to_host(&self, buf: &PjrtBufferHandle, dst: &mut [u8]) -> Result<()> {
        if let Some(spec) = buf.spec() {
            spec.validate_byte_len(dst.len())?;
        }
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

    fn validate_executable_client(&self, exe: &CompiledExecutable) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &exe.client.inner) {
            return Err(xla_err("executable belongs to a different PJRT client"));
        }
        Ok(())
    }

    fn validate_input_clients(&self, inputs: &[&PjrtBufferHandle]) -> Result<()> {
        for (idx, input) in inputs.iter().enumerate() {
            if !Arc::ptr_eq(&self.inner, &input.client.inner) {
                return Err(xla_err(format!(
                    "argument {idx} belongs to a different PJRT client"
                )));
            }
        }
        Ok(())
    }

    fn validate_input_specs(
        &self,
        signature: &PjrtExecutableSignature,
        inputs: &[&PjrtBufferHandle],
    ) -> Result<()> {
        let mut actual = Vec::with_capacity(inputs.len());
        for (idx, input) in inputs.iter().enumerate() {
            let spec = input
                .spec()
                .ok_or_else(|| xla_err(format!("argument {idx} has no tensor spec")))?;
            actual.push(spec.clone());
        }
        signature.validate_arguments(&actual)
    }

    fn destroy_raw_buffers(&self, buffers: &[*mut PjrtBuffer]) {
        for &buffer in buffers {
            if !buffer.is_null() {
                unsafe {
                    let mut args = PJRT_Buffer_Destroy_Args {
                        struct_size: std::mem::size_of::<PJRT_Buffer_Destroy_Args>(),
                        extension_start: ptr::null_mut(),
                        buffer,
                    };
                    let _ = (self.inner.fns.buffer_destroy)(&mut args);
                }
            }
        }
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
    format: PjrtProgramFormat,
    signature: Option<PjrtExecutableSignature>,
}

impl CompiledExecutable {
    pub fn client(&self) -> &PjrtClientHandle {
        &self.client
    }

    pub fn format(&self) -> PjrtProgramFormat {
        self.format
    }

    pub fn signature(&self) -> Option<&PjrtExecutableSignature> {
        self.signature.as_ref()
    }
}

unsafe impl Send for CompiledExecutable {}
unsafe impl Sync for CompiledExecutable {}

impl Drop for CompiledExecutable {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let mut args = PJRT_LoadedExecutable_Destroy_Args {
                    struct_size: std::mem::size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    executable: self.raw,
                };
                let _ = (self.client.inner.fns.loaded_executable_destroy)(&mut args);
            }
            self.raw = ptr::null_mut();
        }
    }
}

pub struct PjrtBufferHandle {
    client: PjrtClientHandle,
    pub(crate) raw: *mut PjrtBuffer,
    spec: Option<PjrtTensorSpec>,
    device_idx: Option<usize>,
}

impl PjrtBufferHandle {
    pub fn client(&self) -> &PjrtClientHandle {
        &self.client
    }

    pub fn spec(&self) -> Option<&PjrtTensorSpec> {
        self.spec.as_ref()
    }

    pub fn device_idx(&self) -> Option<usize> {
        self.device_idx
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
            self.raw = ptr::null_mut();
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
