//! GPU device descriptor and enumeration.

/// Memory usage snapshot for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

/// Static descriptor for a GPU device.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory: usize,
}

/// Enumerate available GPU devices.
///
/// Under `mock-gpu` this returns a single virtual device.
/// Under `cuda` this queries the CUDA driver for real devices.
pub fn list_devices() -> Vec<GpuDevice> {
    #[cfg(feature = "cuda")]
    {
        cuda_list_devices()
    }
    #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
    {
        vec![GpuDevice {
            id: 0,
            name: "MockGPU-0".into(),
            compute_capability: (8, 0),
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GiB
        }]
    }
    #[cfg(not(any(feature = "mock-gpu", feature = "cuda")))]
    {
        Vec::new()
    }
}

#[cfg(feature = "cuda")]
fn cuda_list_devices() -> Vec<GpuDevice> {
    use cudarc::driver::CudaDevice;

    let count = match CudaDevice::count() {
        Ok(n) => n as usize,
        Err(e) => {
            tracing::warn!("Failed to query CUDA device count: {e}");
            return Vec::new();
        }
    };

    let mut devices = Vec::with_capacity(count);
    for id in 0..count {
        let dev = match CudaDevice::new(id) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(id, "Failed to init CUDA device: {e}");
                continue;
            }
        };

        let name = dev.name().unwrap_or_else(|_| format!("CUDA Device {id}"));

        let (major, minor) = dev
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .and_then(|maj| {
                dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                    .map(|min| (maj as u32, min as u32))
            })
            .unwrap_or((0, 0));

        let total_memory = dev
            .total_mem()
            .unwrap_or(0);

        devices.push(GpuDevice {
            id,
            name,
            compute_capability: (major, minor),
            total_memory,
        });
    }

    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "mock-gpu")]
    fn list_devices_returns_mock() {
        let devs = list_devices();
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].id, 0);
        assert!(devs[0].name.contains("Mock"));
    }

    #[test]
    fn memory_info_eq() {
        let a = MemoryInfo { total: 100, free: 60, used: 40 };
        let b = MemoryInfo { total: 100, free: 60, used: 40 };
        assert_eq!(a, b);
    }
}
