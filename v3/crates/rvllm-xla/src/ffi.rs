#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::c_void;

use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PjrtElementType {
    INVALID = 0,
    PRED = 1,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    BF16,
    C64,
    C128,
    F8E5M2,
    F8E4M3FN,
}

impl PjrtElementType {
    pub const fn byte_size(self) -> Option<usize> {
        match self {
            Self::PRED | Self::S8 | Self::U8 | Self::F8E5M2 | Self::F8E4M3FN => Some(1),
            Self::S16 | Self::U16 | Self::F16 | Self::BF16 => Some(2),
            Self::S32 | Self::U32 | Self::F32 => Some(4),
            Self::S64 | Self::U64 | Self::F64 | Self::C64 => Some(8),
            Self::C128 => Some(16),
            Self::INVALID => None,
        }
    }
}

pub type PjrtClient = c_void;
pub type PjrtBuffer = c_void;
pub type PjrtLoadedExecutable = c_void;
pub type PjrtEvent = c_void;
pub type PjrtDevice = c_void;
pub type PjrtError = c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PjrtHostBufferSemantics {
    ImmutableOnlyDuringCall = 0,
    ImmutableUntilTransferCompletes = 1,
    ImmutableZeroCopy = 2,
}

#[repr(C)]
pub struct PJRT_Error_Message_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub error: *mut PjrtError,
    pub message: *const u8,
    pub message_size: usize,
}

#[repr(C)]
pub struct PJRT_Error_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub error: *mut PjrtError,
}

#[repr(C)]
pub struct PJRT_Plugin_Initialize_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
}

#[repr(C)]
pub struct PJRT_Client_Create_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub create_options: *const c_void,
    pub num_options: usize,
    pub kv_get_callback: *const c_void,
    pub kv_get_user_arg: *mut c_void,
    pub kv_put_callback: *const c_void,
    pub kv_put_user_arg: *mut c_void,
    pub client: *mut PjrtClient,
    pub kv_try_get_callback: *const c_void,
    pub kv_try_get_user_arg: *mut c_void,
}

#[repr(C)]
pub struct PJRT_Client_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub client: *mut PjrtClient,
}

#[repr(C)]
pub struct PJRT_Client_Devices_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub client: *mut PjrtClient,
    pub devices: *const *mut PjrtDevice,
    pub num_devices: usize,
}

#[repr(C)]
pub struct PJRT_Program {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub code: *const u8,
    pub code_size: usize,
    pub format: *const u8,
    pub format_size: usize,
}

#[repr(C)]
pub struct PJRT_Client_Compile_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub client: *mut PjrtClient,
    pub program: *const PJRT_Program,
    pub compile_options: *const u8,
    pub compile_options_size: usize,
    pub executable: *mut PjrtLoadedExecutable,
}

#[repr(C)]
pub struct PJRT_Client_BufferFromHostBuffer_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub client: *mut PjrtClient,
    pub data: *const c_void,
    pub type_: PjrtElementType,
    pub dims: *const i64,
    pub num_dims: usize,
    pub byte_strides: *const i64,
    pub num_byte_strides: usize,
    pub host_buffer_semantics: PjrtHostBufferSemantics,
    pub device: *mut PjrtDevice,
    pub memory: *mut c_void,
    pub _layout: *mut c_void,
    pub done_with_host_buffer: *mut PjrtEvent,
    pub buffer: *mut PjrtBuffer,
}

#[repr(C)]
pub struct PJRT_LoadedExecutable_Execute_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub executable: *mut PjrtLoadedExecutable,
    pub options: *const PJRT_ExecuteOptions,
    pub argument_lists: *const *const *mut PjrtBuffer,
    pub num_devices: usize,
    pub num_args: usize,
    pub output_lists: *const *mut *mut PjrtBuffer,
    pub device_complete_events: *mut *mut PjrtEvent,
    pub execute_device: *mut PjrtDevice,
}

#[repr(C)]
pub struct PJRT_LoadedExecutable_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub executable: *mut PjrtLoadedExecutable,
}

#[repr(C)]
pub struct PJRT_ExecuteOptions {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub send_callbacks: *const c_void,
    pub recv_callbacks: *const c_void,
    pub num_send_ops: usize,
    pub num_recv_ops: usize,
    pub launch_id: i64,
    pub non_donatable_input_indices: *const i64,
    pub num_non_donatable_input_indices: usize,
    pub context: *const c_void,
    pub call_location: *const u8,
    pub num_tasks: usize,
    pub task_ids: *const i32,
    pub incarnation_ids: *const i64,
}

#[repr(C)]
pub struct PJRT_Buffer_ToHostBuffer_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub src: *mut PjrtBuffer,
    pub host_layout: *const c_void,
    pub dst: *mut c_void,
    pub dst_size: usize,
    pub event: *mut PjrtEvent,
}

#[repr(C)]
pub struct PJRT_Buffer_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub buffer: *mut PjrtBuffer,
}

#[repr(C)]
pub struct PJRT_Event_Await_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub event: *mut PjrtEvent,
}

#[repr(C)]
pub struct PJRT_Event_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub event: *mut PjrtEvent,
}

pub type PjrtPluginInitializeFn =
    unsafe extern "C" fn(*mut PJRT_Plugin_Initialize_Args) -> *mut PjrtError;
pub type PjrtErrorDestroyFn = unsafe extern "C" fn(*mut PJRT_Error_Destroy_Args);
pub type PjrtErrorMessageFn = unsafe extern "C" fn(*mut PJRT_Error_Message_Args);
pub type PjrtClientCreateFn = unsafe extern "C" fn(*mut PJRT_Client_Create_Args) -> *mut PjrtError;
pub type PjrtClientDestroyFn =
    unsafe extern "C" fn(*mut PJRT_Client_Destroy_Args) -> *mut PjrtError;
pub type PjrtClientDevicesFn =
    unsafe extern "C" fn(*mut PJRT_Client_Devices_Args) -> *mut PjrtError;
pub type PjrtClientCompileFn =
    unsafe extern "C" fn(*mut PJRT_Client_Compile_Args) -> *mut PjrtError;
pub type PjrtClientBufferFromHostBufferFn =
    unsafe extern "C" fn(*mut PJRT_Client_BufferFromHostBuffer_Args) -> *mut PjrtError;
pub type PjrtLoadedExecutableExecuteFn =
    unsafe extern "C" fn(*mut PJRT_LoadedExecutable_Execute_Args) -> *mut PjrtError;
pub type PjrtLoadedExecutableDestroyFn =
    unsafe extern "C" fn(*mut PJRT_LoadedExecutable_Destroy_Args) -> *mut PjrtError;
pub type PjrtBufferToHostBufferFn =
    unsafe extern "C" fn(*mut PJRT_Buffer_ToHostBuffer_Args) -> *mut PjrtError;
pub type PjrtBufferDestroyFn =
    unsafe extern "C" fn(*mut PJRT_Buffer_Destroy_Args) -> *mut PjrtError;
pub type PjrtEventAwaitFn = unsafe extern "C" fn(*mut PJRT_Event_Await_Args) -> *mut PjrtError;
pub type PjrtEventDestroyFn = unsafe extern "C" fn(*mut PJRT_Event_Destroy_Args);

#[repr(C)]
pub struct PJRT_Api {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub pjrt_api_version: usize,
    _padding0: usize,
    _padding1: usize,
}

pub type GetPjrtApiFn = unsafe extern "C" fn() -> *const PJRT_Api;

const HEADER_BYTES: usize = 40;
const PTR_SIZE: usize = 8;

const fn fn_offset(index: usize) -> usize {
    HEADER_BYTES + index * PTR_SIZE
}

pub const OFFSET_ERROR_DESTROY: usize = fn_offset(0);
pub const OFFSET_ERROR_MESSAGE: usize = fn_offset(1);
pub const OFFSET_PLUGIN_INITIALIZE: usize = fn_offset(3);
pub const OFFSET_EVENT_DESTROY: usize = fn_offset(5);
pub const OFFSET_EVENT_AWAIT: usize = fn_offset(8);
pub const OFFSET_CLIENT_CREATE: usize = fn_offset(10);
pub const OFFSET_CLIENT_DESTROY: usize = fn_offset(11);
pub const OFFSET_CLIENT_DEVICES: usize = fn_offset(15);
pub const OFFSET_CLIENT_COMPILE: usize = fn_offset(20);
pub const OFFSET_CLIENT_BUFFER_FROM_HOST: usize = fn_offset(22);
pub const OFFSET_LOADED_EXECUTABLE_DESTROY: usize = fn_offset(50);
pub const OFFSET_LOADED_EXECUTABLE_EXECUTE: usize = fn_offset(55);
pub const OFFSET_BUFFER_DESTROY: usize = fn_offset(58);
pub const OFFSET_BUFFER_TO_HOST: usize = fn_offset(70);

pub struct PjrtApiFns {
    pub plugin_initialize: PjrtPluginInitializeFn,
    pub error_destroy: PjrtErrorDestroyFn,
    pub error_message: PjrtErrorMessageFn,
    pub event_destroy: PjrtEventDestroyFn,
    pub event_await: PjrtEventAwaitFn,
    pub client_create: PjrtClientCreateFn,
    pub client_destroy: PjrtClientDestroyFn,
    pub client_devices: PjrtClientDevicesFn,
    pub client_compile: PjrtClientCompileFn,
    pub client_buffer_from_host: PjrtClientBufferFromHostBufferFn,
    pub loaded_executable_destroy: PjrtLoadedExecutableDestroyFn,
    pub loaded_executable_execute: PjrtLoadedExecutableExecuteFn,
    pub buffer_to_host: PjrtBufferToHostBufferFn,
    pub buffer_destroy: PjrtBufferDestroyFn,
}

impl PjrtApiFns {
    pub unsafe fn from_api_ptr(api: *const PJRT_Api) -> Self {
        let base = api as *const u8;
        Self {
            plugin_initialize: read_fn_ptr(base, OFFSET_PLUGIN_INITIALIZE),
            error_destroy: read_fn_ptr(base, OFFSET_ERROR_DESTROY),
            error_message: read_fn_ptr(base, OFFSET_ERROR_MESSAGE),
            event_destroy: read_fn_ptr(base, OFFSET_EVENT_DESTROY),
            event_await: read_fn_ptr(base, OFFSET_EVENT_AWAIT),
            client_create: read_fn_ptr(base, OFFSET_CLIENT_CREATE),
            client_destroy: read_fn_ptr(base, OFFSET_CLIENT_DESTROY),
            client_devices: read_fn_ptr(base, OFFSET_CLIENT_DEVICES),
            client_compile: read_fn_ptr(base, OFFSET_CLIENT_COMPILE),
            client_buffer_from_host: read_fn_ptr(base, OFFSET_CLIENT_BUFFER_FROM_HOST),
            loaded_executable_destroy: read_fn_ptr(base, OFFSET_LOADED_EXECUTABLE_DESTROY),
            loaded_executable_execute: read_fn_ptr(base, OFFSET_LOADED_EXECUTABLE_EXECUTE),
            buffer_to_host: read_fn_ptr(base, OFFSET_BUFFER_TO_HOST),
            buffer_destroy: read_fn_ptr(base, OFFSET_BUFFER_DESTROY),
        }
    }
}

unsafe fn read_fn_ptr<T>(base: *const u8, offset: usize) -> T {
    let ptr = base.add(offset) as *const T;
    std::ptr::read(ptr)
}

pub unsafe fn extract_error_message(fns: &PjrtApiFns, err: *mut PjrtError) -> String {
    let mut args = PJRT_Error_Message_Args {
        struct_size: std::mem::size_of::<PJRT_Error_Message_Args>(),
        extension_start: std::ptr::null_mut(),
        error: err,
        message: std::ptr::null(),
        message_size: 0,
    };
    (fns.error_message)(&mut args);
    let msg = if args.message.is_null() || args.message_size == 0 {
        "unknown PJRT error".to_string()
    } else {
        let slice = std::slice::from_raw_parts(args.message, args.message_size);
        String::from_utf8_lossy(slice).into_owned()
    };
    let mut destroy_args = PJRT_Error_Destroy_Args {
        struct_size: std::mem::size_of::<PJRT_Error_Destroy_Args>(),
        extension_start: std::ptr::null_mut(),
        error: err,
    };
    (fns.error_destroy)(&mut destroy_args);
    msg
}
