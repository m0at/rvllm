//! CUPTI activity-based GPU kernel profiler.
//!
//! Uses CUPTI's activity API to collect per-kernel timing records with
//! nanosecond precision. The activity API has lower overhead than the
//! callback API because records are written to a ring buffer by the
//! driver and flushed after stream synchronization.

use std::collections::HashMap;

#[cfg(feature = "cuda")]
use std::ffi::CStr;

#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use tracing::{debug, info};

/// A single kernel execution record captured by CUPTI.
#[derive(Debug, Clone)]
pub struct KernelRecord {
    /// Kernel function name (demangled).
    pub name: String,
    /// Duration in nanoseconds.
    pub duration_ns: u64,
    /// Grid dimensions (x, y, z).
    pub grid: (u32, u32, u32),
    /// Block dimensions (x, y, z).
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_mem: u32,
    /// CUDA stream ID.
    pub stream_id: u32,
    /// Start timestamp (ns since epoch).
    pub start_ns: u64,
    /// End timestamp (ns since epoch).
    pub end_ns: u64,
}

/// Aggregated timing for a kernel across multiple invocations.
#[derive(Debug, Clone)]
pub struct KernelAggregate {
    /// Kernel name.
    pub name: String,
    /// Number of invocations.
    pub count: u64,
    /// Total GPU time in nanoseconds.
    pub total_ns: u64,
    /// Minimum single-invocation time.
    pub min_ns: u64,
    /// Maximum single-invocation time.
    pub max_ns: u64,
    /// Representative grid/block config.
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

impl KernelAggregate {
    pub fn avg_ns(&self) -> u64 {
        if self.count == 0 { 0 } else { self.total_ns / self.count }
    }

    pub fn avg_us(&self) -> f64 {
        self.avg_ns() as f64 / 1000.0
    }

    pub fn total_us(&self) -> f64 {
        self.total_ns as f64 / 1000.0
    }

    pub fn total_ms(&self) -> f64 {
        self.total_ns as f64 / 1_000_000.0
    }
}

/// CUPTI activity record types we care about (from cupti.h).
/// These mirror the C enum values.
#[repr(u32)]
#[allow(dead_code)]
enum CuptiActivityKind {
    Kernel = 3,
    // We only need kernel records for profiling.
}

/// Minimal FFI bindings for CUPTI activity API.
/// We dlopen libcupti.so at runtime to avoid a hard build dependency.
#[cfg(feature = "cuda")]
mod ffi {
    use std::ffi::c_void;
    use std::os::raw::c_int;

    pub type CUptiResult = c_int;
    pub const CUPTI_SUCCESS: CUptiResult = 0;

    // Activity record header (first 4 bytes = kind)
    #[repr(C)]
    pub struct CUpti_ActivityKernel {
        pub kind: u32,
        pub _padding: [u8; 4],
        pub start: u64,
        pub end: u64,
        pub device_id: u32,
        pub context_id: u32,
        pub stream_id: u32,
        pub correlation_id: u32,
        pub grid_x: u32,
        pub grid_y: u32,
        pub grid_z: u32,
        pub block_x: u32,
        pub block_y: u32,
        pub block_z: u32,
        pub static_shared_memory: u32,
        pub dynamic_shared_memory: u32,
        pub local_memory_per_thread: u32,
        pub local_memory_total: u32,
        pub registered_per_thread: u32,
        pub name: *const std::ffi::c_char,
    }

    // Function pointer types for dlopen
    pub type FnActivityEnable = unsafe extern "C" fn(kind: u32) -> CUptiResult;
    pub type FnActivityDisable = unsafe extern "C" fn(kind: u32) -> CUptiResult;
    pub type FnActivityFlushAll = unsafe extern "C" fn(flag: u32) -> CUptiResult;
    pub type FnActivityRegisterCallbacks = unsafe extern "C" fn(
        request: Option<unsafe extern "C" fn(*mut *mut u8, *mut usize, usize)>,
        complete: Option<unsafe extern "C" fn(*mut u8, usize, usize)>,
    ) -> CUptiResult;
    pub type FnActivityGetNextRecord = unsafe extern "C" fn(
        buffer: *mut u8,
        valid_size: usize,
        record: *mut *mut c_void,
    ) -> CUptiResult;
}

/// Collects per-kernel GPU timing records via CUPTI activity API.
///
/// Usage:
/// ```ignore
/// let mut profiler = CuptiProfiler::new()?;
/// profiler.start()?;
/// // ... run inference step ...
/// // ... cudaStreamSynchronize ...
/// profiler.stop()?;
/// let records = profiler.records();
/// let aggregates = profiler.aggregate();
/// ```
pub struct CuptiProfiler {
    records: Vec<KernelRecord>,
    #[cfg(feature = "cuda")]
    active: bool,
}

// Global buffer for CUPTI activity records (CUPTI callbacks are C function pointers,
// can't capture Rust state, so we use a global).
#[cfg(feature = "cuda")]
static GLOBAL_RECORDS: Mutex<Vec<KernelRecord>> = Mutex::new(Vec::new());

impl CuptiProfiler {
    /// Create a new profiler. Loads libcupti.so dynamically.
    pub fn new() -> rvllm_core::prelude::Result<Self> {
        Ok(Self {
            records: Vec::new(),
            #[cfg(feature = "cuda")]
            active: false,
        })
    }

    /// Start profiling. Enables CUPTI kernel activity collection.
    #[cfg(feature = "cuda")]
    pub fn start(&mut self) -> rvllm_core::prelude::Result<()> {
        use rvllm_core::prelude::LLMError;

        // Clear any previous records
        if let Ok(mut global) = GLOBAL_RECORDS.lock() {
            global.clear();
        }
        self.records.clear();

        // Load CUPTI and enable kernel activity
        unsafe {
            let lib = libloading::Library::new("libcupti.so.12")
                .or_else(|_| libloading::Library::new("libcupti.so"))
                .map_err(|e| LLMError::GpuError(format!("failed to load libcupti: {e}")))?;

            let enable: libloading::Symbol<ffi::FnActivityEnable> =
                lib.get(b"cuptiActivityEnable")
                    .map_err(|e| LLMError::GpuError(format!("cuptiActivityEnable: {e}")))?;

            let register: libloading::Symbol<ffi::FnActivityRegisterCallbacks> =
                lib.get(b"cuptiActivityRegisterCallbacks")
                    .map_err(|e| LLMError::GpuError(format!("cuptiActivityRegisterCallbacks: {e}")))?;

            // Register buffer callbacks
            let result = register(Some(buffer_request_callback), Some(buffer_complete_callback));
            if result != ffi::CUPTI_SUCCESS {
                return Err(LLMError::GpuError(format!("CUPTI register callbacks failed: {result}")));
            }

            // Enable kernel activity
            let result = enable(CuptiActivityKind::Kernel as u32);
            if result != ffi::CUPTI_SUCCESS {
                return Err(LLMError::GpuError(format!("CUPTI enable kernel activity failed: {result}")));
            }

            // Leak the library handle so it stays loaded
            std::mem::forget(lib);
        }

        self.active = true;
        debug!("CUPTI profiling started");
        Ok(())
    }

    /// Stop profiling. Flushes all activity buffers and collects records.
    #[cfg(feature = "cuda")]
    pub fn stop(&mut self) -> rvllm_core::prelude::Result<()> {
        use rvllm_core::prelude::LLMError;

        if !self.active {
            return Ok(());
        }

        unsafe {
            let lib = libloading::Library::new("libcupti.so.12")
                .or_else(|_| libloading::Library::new("libcupti.so"))
                .map_err(|e| LLMError::GpuError(format!("libcupti reload: {e}")))?;

            let flush: libloading::Symbol<ffi::FnActivityFlushAll> =
                lib.get(b"cuptiActivityFlushAll")
                    .map_err(|e| LLMError::GpuError(format!("cuptiActivityFlushAll: {e}")))?;

            let disable: libloading::Symbol<ffi::FnActivityDisable> =
                lib.get(b"cuptiActivityDisable")
                    .map_err(|e| LLMError::GpuError(format!("cuptiActivityDisable: {e}")))?;

            // Flush all pending records (flag 0 = blocking)
            flush(0);

            // Disable kernel activity
            disable(CuptiActivityKind::Kernel as u32);

            std::mem::forget(lib);
        }

        // Move records from global to local
        if let Ok(mut global) = GLOBAL_RECORDS.lock() {
            self.records = std::mem::take(&mut *global);
        }

        self.active = false;
        info!(num_records = self.records.len(), "CUPTI profiling stopped");
        Ok(())
    }

    /// Non-CUDA stubs
    #[cfg(not(feature = "cuda"))]
    pub fn start(&mut self) -> rvllm_core::prelude::Result<()> {
        Err(rvllm_core::prelude::LLMError::GpuError("CUPTI requires cuda feature".into()))
    }

    #[cfg(not(feature = "cuda"))]
    pub fn stop(&mut self) -> rvllm_core::prelude::Result<()> {
        Ok(())
    }

    /// Get raw kernel records from the last profiling session.
    pub fn records(&self) -> &[KernelRecord] {
        &self.records
    }

    /// Aggregate records by kernel name.
    pub fn aggregate(&self) -> Vec<KernelAggregate> {
        let mut map: HashMap<String, KernelAggregate> = HashMap::new();

        for r in &self.records {
            let entry = map.entry(r.name.clone()).or_insert(KernelAggregate {
                name: r.name.clone(),
                count: 0,
                total_ns: 0,
                min_ns: u64::MAX,
                max_ns: 0,
                grid: r.grid,
                block: r.block,
                shared_mem: r.shared_mem,
            });
            entry.count += 1;
            entry.total_ns += r.duration_ns;
            entry.min_ns = entry.min_ns.min(r.duration_ns);
            entry.max_ns = entry.max_ns.max(r.duration_ns);
        }

        let mut aggs: Vec<KernelAggregate> = map.into_values().collect();
        aggs.sort_by(|a, b| b.total_ns.cmp(&a.total_ns));
        aggs
    }

    /// Print a summary table of the top-N kernels by total GPU time.
    pub fn print_summary(&self, top_n: usize) {
        let aggs = self.aggregate();
        let total_gpu_ns: u64 = aggs.iter().map(|a| a.total_ns).sum();

        println!("{:<60} {:>6} {:>10} {:>10} {:>8}",
            "Kernel", "Count", "Total(ms)", "Avg(us)", "% GPU");
        println!("{}", "-".repeat(100));

        for (i, a) in aggs.iter().enumerate() {
            if i >= top_n { break; }
            let pct = if total_gpu_ns > 0 {
                a.total_ns as f64 / total_gpu_ns as f64 * 100.0
            } else { 0.0 };
            println!("{:<60} {:>6} {:>10.2} {:>10.1} {:>7.1}%",
                &a.name[..a.name.len().min(60)],
                a.count,
                a.total_ms(),
                a.avg_us(),
                pct,
            );
        }

        println!("{}", "-".repeat(100));
        println!("Total GPU time: {:.2} ms, {} unique kernels, {} total launches",
            total_gpu_ns as f64 / 1_000_000.0,
            aggs.len(),
            self.records.len(),
        );
    }
}

// CUPTI activity buffer callbacks (C ABI, called from driver)
#[cfg(feature = "cuda")]
const CUPTI_BUF_SIZE: usize = 8 * 1024 * 1024; // 8 MiB ring buffer

#[cfg(feature = "cuda")]
unsafe extern "C" fn buffer_request_callback(
    buffer: *mut *mut u8,
    size: *mut usize,
    _max_num_records: usize,
) {
    let buf = Box::into_raw(vec![0u8; CUPTI_BUF_SIZE].into_boxed_slice()) as *mut u8;
    *buffer = buf;
    *size = CUPTI_BUF_SIZE;
}

#[cfg(feature = "cuda")]
unsafe extern "C" fn buffer_complete_callback(
    buffer: *mut u8,
    _size: usize,
    valid_size: usize,
) {
    if buffer.is_null() || valid_size == 0 {
        return;
    }

    // Parse kernel activity records from the buffer
    let mut records = Vec::new();

    // Walk the buffer manually: each record starts with a u32 'kind' field
    // Records are variable-size but kernel records are fixed at ~160 bytes.
    // For simplicity, we use cuptiActivityGetNextRecord if available,
    // otherwise parse manually.
    let mut offset = 0;
    while offset + 8 < valid_size {
        let kind = *(buffer.add(offset) as *const u32);
        if kind == CuptiActivityKind::Kernel as u32 {
            let rec = &*(buffer.add(offset) as *const ffi::CUpti_ActivityKernel);
            let name = if rec.name.is_null() {
                "<unknown>".to_string()
            } else {
                CStr::from_ptr(rec.name).to_string_lossy().into_owned()
            };
            records.push(KernelRecord {
                name,
                duration_ns: rec.end.saturating_sub(rec.start),
                grid: (rec.grid_x, rec.grid_y, rec.grid_z),
                block: (rec.block_x, rec.block_y, rec.block_z),
                shared_mem: rec.dynamic_shared_memory,
                stream_id: rec.stream_id,
                start_ns: rec.start,
                end_ns: rec.end,
            });
        }
        // Advance by record size. Kernel records are typically 136-160 bytes.
        // The first 4 bytes are kind, next 4 padding, then fields.
        // We use a fixed stride based on the struct size.
        offset += std::mem::size_of::<ffi::CUpti_ActivityKernel>();
    }

    if !records.is_empty() {
        if let Ok(mut global) = GLOBAL_RECORDS.lock() {
            global.extend(records);
        }
    }

    // Free the buffer
    let _ = Box::from_raw(std::slice::from_raw_parts_mut(buffer, CUPTI_BUF_SIZE));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_creates() {
        let p = CuptiProfiler::new().unwrap();
        assert!(p.records().is_empty());
    }

    #[test]
    fn aggregate_empty() {
        let p = CuptiProfiler::new().unwrap();
        assert!(p.aggregate().is_empty());
    }

    #[test]
    fn aggregate_groups_by_name() {
        let mut p = CuptiProfiler::new().unwrap();
        p.records = vec![
            KernelRecord {
                name: "kern_a".into(), duration_ns: 100, grid: (1,1,1),
                block: (128,1,1), shared_mem: 0, stream_id: 0,
                start_ns: 0, end_ns: 100,
            },
            KernelRecord {
                name: "kern_a".into(), duration_ns: 200, grid: (1,1,1),
                block: (128,1,1), shared_mem: 0, stream_id: 0,
                start_ns: 100, end_ns: 300,
            },
            KernelRecord {
                name: "kern_b".into(), duration_ns: 50, grid: (2,1,1),
                block: (64,1,1), shared_mem: 1024, stream_id: 0,
                start_ns: 300, end_ns: 350,
            },
        ];
        let aggs = p.aggregate();
        assert_eq!(aggs.len(), 2);
        assert_eq!(aggs[0].name, "kern_a"); // highest total
        assert_eq!(aggs[0].count, 2);
        assert_eq!(aggs[0].total_ns, 300);
        assert_eq!(aggs[0].min_ns, 100);
        assert_eq!(aggs[0].max_ns, 200);
        assert_eq!(aggs[1].name, "kern_b");
        assert_eq!(aggs[1].count, 1);
    }
}
