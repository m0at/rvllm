pub mod buffer;
#[cfg(feature = "tpu")]
pub mod client;
pub mod device;
pub mod ffi;
pub mod module;

pub use rvllm_core::prelude::{LLMError, Result};

pub mod prelude {
    pub use crate::buffer::{XlaBuffer, XlaDtype};
    pub use crate::device::{TpuDevice, XlaDeviceId};
    pub use crate::module::ModuleLoader;
    pub use crate::{LLMError, Result};
}
