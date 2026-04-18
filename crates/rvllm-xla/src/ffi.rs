#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::c_void;

pub type PjrtClient = c_void;
pub type PjrtBuffer = c_void;
pub type PjrtLoadedExecutable = c_void;
pub type PjrtEvent = c_void;
pub type PjrtDevice = c_void;
pub type PjrtError = c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PjrtElementType {
    INVALID = 0,
    PRED = 1,
    S8 = 2,
    S16 = 3,
    S32 = 4,
    S64 = 5,
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,
    F16 = 10,
    F32 = 11,
    F64 = 12,
    BF16 = 16,
    C64 = 15,
    C128 = 18,
    F8E5M2 = 19,
    F8E4M3FN = 20,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PjrtHostBufferSemantics {
    ImmutableOnlyDuringCall = 0,
    ImmutableUntilTransferCompletes = 1,
    ImmutableZeroCopy = 2,
}

// --- PJRT_Error helpers ---

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

// --- PJRT_Client_Create ---

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
}

// --- PJRT_Client_Devices ---

#[repr(C)]
pub struct PJRT_Client_Devices_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub client: *mut PjrtClient,
    pub devices: *const *mut PjrtDevice,
    pub num_devices: usize,
}

// --- PJRT_Program ---

#[repr(C)]
pub struct PJRT_Program {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub code: *const u8,
    pub code_size: usize,
    pub format: *const u8,
    pub format_size: usize,
}

// --- PJRT_Client_Compile ---

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

// --- PJRT_Client_BufferFromHostBuffer ---

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
    pub buffer: *mut PjrtBuffer,
    pub done_with_host_buffer: *mut PjrtEvent,
}

// --- PJRT_LoadedExecutable_Execute ---

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
    pub output_sizes: *mut usize,
    pub device_complete_events: *mut *mut PjrtEvent,
    pub execute_device: *mut PjrtDevice,
}

#[repr(C)]
pub struct PJRT_ExecuteOptions {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub launch_id: i32,
    pub non_donatable_input_indices: *const i64,
    pub num_non_donatable_input_indices: usize,
}

// --- PJRT_Buffer_ToHostBuffer ---

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

// --- PJRT_Buffer_Destroy ---

#[repr(C)]
pub struct PJRT_Buffer_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub buffer: *mut PjrtBuffer,
}

// --- PJRT_Event_Await ---

#[repr(C)]
pub struct PJRT_Event_Await_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub event: *mut PjrtEvent,
}

// --- PJRT_Event_Destroy ---

#[repr(C)]
pub struct PJRT_Event_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub event: *mut PjrtEvent,
}

// --- Function pointer types ---

pub type PjrtErrorDestroyFn =
    unsafe extern "C" fn(*mut PJRT_Error_Destroy_Args);
pub type PjrtErrorMessageFn =
    unsafe extern "C" fn(*mut PJRT_Error_Message_Args);
pub type PjrtClientCreateFn =
    unsafe extern "C" fn(*mut PJRT_Client_Create_Args) -> *mut PjrtError;
pub type PjrtClientDevicesFn =
    unsafe extern "C" fn(*mut PJRT_Client_Devices_Args) -> *mut PjrtError;
pub type PjrtClientCompileFn =
    unsafe extern "C" fn(*mut PJRT_Client_Compile_Args) -> *mut PjrtError;
pub type PjrtClientBufferFromHostBufferFn =
    unsafe extern "C" fn(*mut PJRT_Client_BufferFromHostBuffer_Args) -> *mut PjrtError;
pub type PjrtLoadedExecutableExecuteFn =
    unsafe extern "C" fn(*mut PJRT_LoadedExecutable_Execute_Args) -> *mut PjrtError;
pub type PjrtBufferToHostBufferFn =
    unsafe extern "C" fn(*mut PJRT_Buffer_ToHostBuffer_Args) -> *mut PjrtError;
pub type PjrtBufferDestroyFn =
    unsafe extern "C" fn(*mut PJRT_Buffer_Destroy_Args) -> *mut PjrtError;
pub type PjrtEventAwaitFn =
    unsafe extern "C" fn(*mut PJRT_Event_Await_Args) -> *mut PjrtError;
pub type PjrtEventDestroyFn =
    unsafe extern "C" fn(*mut PJRT_Event_Destroy_Args);

// --- PJRT_Api function table ---
// In the real pjrt_c_api.h this is a massive struct with ~60+ function pointers.
// We define only the fields we need, using padding to match offsets.
// The real struct starts with struct_size, then extension_start, then
// pjrt_api_version.major, pjrt_api_version.minor, then function pointers
// in a specific order. We access them via GetPjrtApi() -> *const PJRT_Api
// and then read the function pointers we need by field offset.
//
// Rather than trying to replicate the exact layout (which is fragile),
// we resolve the function pointers we need from the raw PJRT_Api pointer
// at known byte offsets. This is handled in client.rs.

#[repr(C)]
pub struct PJRT_Api {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub pjrt_api_version_major: i32,
    pub pjrt_api_version_minor: i32,
    // Followed by function pointers in order. We read them by offset.
    // The first function pointer is PJRT_Error_Destroy at offset 24 on 64-bit.
}

// GetPjrtApi is the symbol exported by libtpu.so
pub type GetPjrtApiFn = unsafe extern "C" fn() -> *const PJRT_Api;

// Offsets of function pointers within PJRT_Api (64-bit, from pjrt_c_api.h).
// struct_size(8) + extension_start(8) + major(4) + minor(4) = 24 bytes header
// Then function pointers at 8 bytes each.
// Order from pjrt_c_api.h PJRT_Api struct definition:
//  0: PJRT_Error_Destroy
//  1: PJRT_Error_Message
//  2: PJRT_Error_GetCode
//  3: PJRT_Plugin_Initialize  (added later, but at index 3 in modern API)
//  4: PJRT_Plugin_Attributes
//  5: PJRT_Event_Destroy
//  6: PJRT_Event_IsReady
//  7: PJRT_Event_Error
//  8: PJRT_Event_Await
//  9: PJRT_Event_OnReady
// 10: PJRT_Client_Create
// 11: PJRT_Client_Destroy
// 12: PJRT_Client_PlatformName
// 13: PJRT_Client_ProcessIndex
// 14: PJRT_Client_PlatformVersion
// 15: PJRT_Client_Devices
// 16: PJRT_Client_AddressableDevices
// 17: PJRT_Client_LookupDevice
// 18: PJRT_Client_LookupAddressableDevice
// 19: PJRT_Client_AddressableMemories
// 20: PJRT_Client_Compile
// 21: PJRT_Client_DefaultDeviceAssignment
// 22: PJRT_Client_BufferFromHostBuffer

const HEADER_BYTES: usize = 24; // 8 + 8 + 4 + 4
const PTR_SIZE: usize = 8;

const fn fn_offset(index: usize) -> usize {
    HEADER_BYTES + index * PTR_SIZE
}

pub const OFFSET_ERROR_DESTROY: usize = fn_offset(0);
pub const OFFSET_ERROR_MESSAGE: usize = fn_offset(1);
pub const OFFSET_EVENT_DESTROY: usize = fn_offset(5);
pub const OFFSET_EVENT_AWAIT: usize = fn_offset(8);
pub const OFFSET_CLIENT_CREATE: usize = fn_offset(10);
pub const OFFSET_CLIENT_DEVICES: usize = fn_offset(15);
pub const OFFSET_CLIENT_COMPILE: usize = fn_offset(20);
pub const OFFSET_CLIENT_BUFFER_FROM_HOST: usize = fn_offset(22);

// LoadedExecutable_Execute is further down. In pjrt_c_api.h after the
// LoadedExecutable block. The exact index depends on API version.
// For PJRT API v0.54+:
//  23: PJRT_Device_Id
//  ...through various Device/Memory/Executable functions...
//  ~47: PJRT_LoadedExecutable_Execute
//  We use the known offset. Counting from pjrt_c_api.h:
//  23-28: Device functions (6)
//  29-32: Memory functions (4)
//  33-40: Executable functions (8)
//  41: PJRT_LoadedExecutable_Destroy
//  42: PJRT_LoadedExecutable_GetExecutable
//  43: PJRT_LoadedExecutable_Delete
//  44: PJRT_LoadedExecutable_IsDeleted
//  45: PJRT_LoadedExecutable_Execute
//  46: PJRT_LoadedExecutable_GetCostAnalysis
//  47+: Buffer functions
//  BufferToHostBuffer is around index 53-55 depending on version.

pub const OFFSET_LOADED_EXECUTABLE_EXECUTE: usize = fn_offset(45);

// Buffer functions:
//  After LoadedExecutable block and before TopologicalSort etc.
//  ~48: PJRT_Buffer_Destroy
//  ~49: PJRT_Buffer_ElementType
//  ~50: PJRT_Buffer_Dimensions
//  ~51: PJRT_Buffer_UnpaddedDimensions
//  ~52: PJRT_Buffer_DynamicDimensionIndices
//  ~53: PJRT_Buffer_GetMemoryLayout
//  ~54: PJRT_Buffer_OnDeviceSizeInBytes
//  ~55: PJRT_Buffer_Device
//  ~56: PJRT_Buffer_Memory
//  ~57: PJRT_Buffer_Delete
//  ~58: PJRT_Buffer_IsDeleted
//  ~59: PJRT_Buffer_CopyToDevice
//  ~60: PJRT_Buffer_ToHostBuffer
//  ~61: PJRT_Buffer_IsOnCpu
//  ~62+: more functions

pub const OFFSET_BUFFER_DESTROY: usize = fn_offset(48);
pub const OFFSET_BUFFER_TO_HOST: usize = fn_offset(60);

pub struct PjrtApiFns {
    pub error_destroy: PjrtErrorDestroyFn,
    pub error_message: PjrtErrorMessageFn,
    pub event_destroy: PjrtEventDestroyFn,
    pub event_await: PjrtEventAwaitFn,
    pub client_create: PjrtClientCreateFn,
    pub client_devices: PjrtClientDevicesFn,
    pub client_compile: PjrtClientCompileFn,
    pub client_buffer_from_host: PjrtClientBufferFromHostBufferFn,
    pub loaded_executable_execute: PjrtLoadedExecutableExecuteFn,
    pub buffer_to_host: PjrtBufferToHostBufferFn,
    pub buffer_destroy: PjrtBufferDestroyFn,
}

impl PjrtApiFns {
    pub unsafe fn from_api_ptr(api: *const PJRT_Api) -> Self {
        let base = api as *const u8;
        Self {
            error_destroy: read_fn_ptr(base, OFFSET_ERROR_DESTROY),
            error_message: read_fn_ptr(base, OFFSET_ERROR_MESSAGE),
            event_destroy: read_fn_ptr(base, OFFSET_EVENT_DESTROY),
            event_await: read_fn_ptr(base, OFFSET_EVENT_AWAIT),
            client_create: read_fn_ptr(base, OFFSET_CLIENT_CREATE),
            client_devices: read_fn_ptr(base, OFFSET_CLIENT_DEVICES),
            client_compile: read_fn_ptr(base, OFFSET_CLIENT_COMPILE),
            client_buffer_from_host: read_fn_ptr(base, OFFSET_CLIENT_BUFFER_FROM_HOST),
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
