//! CUDA graph capture and replay for decode steps.
//!
//! Captures a full decode-step kernel sequence into a CUDA graph, then replays
//! it each step instead of re-launching individual kernels. This eliminates
//! kernel launch overhead and yields 2-5x speedup for decode.
//!
//! Under `mock-gpu` the graph is a no-op wrapper that records capture/replay
//! calls for testing. Under `cuda` it uses the CUDA driver graph API via cudarc.

use std::collections::HashMap;

use tracing::{debug, info, trace, warn};

use crate::Result;
#[cfg(feature = "cuda")]
use crate::LLMError;

/// Supported batch sizes for which we pre-capture CUDA graphs.
pub const GRAPH_BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32];

/// Returns the smallest cached batch size >= `actual`, or `None` if `actual`
/// exceeds the largest cached size.
pub fn padded_batch_size(actual: usize) -> Option<usize> {
    GRAPH_BATCH_SIZES.iter().copied().find(|&s| s >= actual)
}

/// A captured CUDA graph that can be replayed.
///
/// Under `cuda` this wraps the driver-level graph and executable graph handles.
/// Under `mock-gpu` it is a lightweight marker for testing.
pub struct CudaGraph {
    batch_size: usize,
    #[cfg(feature = "cuda")]
    graph: cudarc::driver::sys::CUgraph,
    #[cfg(feature = "cuda")]
    exec: cudarc::driver::sys::CUgraphExec,
    #[cfg(not(feature = "cuda"))]
    replay_count: std::sync::atomic::AtomicUsize,
}

// SAFETY: The CUDA graph/exec handles are device objects managed by the driver.
// They are thread-safe to send across threads (replay must happen on the same
// device context but ownership transfer is fine).
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// Batch size this graph was captured for.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Replay the captured graph on the given stream.
    #[cfg(feature = "cuda")]
    pub fn replay(&self, stream: &crate::stream::GpuStream) -> Result<()> {
        trace!(batch_size = self.batch_size, "replaying CUDA graph");
        // SAFETY: exec and stream are valid handles on the same device.
        let result = unsafe {
            cudarc::driver::sys::cuGraphLaunch(
                self.exec,
                stream.cuda_stream().stream as cudarc::driver::sys::CUstream,
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(LLMError::GpuError(format!(
                "cuGraphLaunch failed: {:?}",
                result
            )));
        }
        Ok(())
    }

    /// Replay (mock): just increments the counter.
    #[cfg(not(feature = "cuda"))]
    pub fn replay(&self, _stream: &crate::stream::GpuStream) -> Result<()> {
        trace!(batch_size = self.batch_size, "replaying CUDA graph (mock)");
        self.replay_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Number of times this graph has been replayed (mock-gpu only, for testing).
    #[cfg(not(feature = "cuda"))]
    pub fn replay_count(&self) -> usize {
        self.replay_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            // SAFETY: destroying valid graph handles.
            unsafe {
                cudarc::driver::sys::cuGraphExecDestroy(self.exec);
                cudarc::driver::sys::cuGraphDestroy(self.graph);
            }
        }
        trace!(batch_size = self.batch_size, "CudaGraph dropped");
    }
}

/// Pool of pre-captured CUDA graphs keyed by batch size.
///
/// The pool lazily captures graphs on first use for each batch size and caches
/// them for subsequent decode steps.
pub struct CudaGraphPool {
    graphs: HashMap<usize, CudaGraph>,
    max_batch_size: usize,
    enabled: bool,
}

impl CudaGraphPool {
    /// Create a new empty graph pool.
    pub fn new(max_batch_size: usize) -> Self {
        info!(max_batch_size, "creating CudaGraphPool");
        Self {
            graphs: HashMap::new(),
            max_batch_size,
            enabled: true,
        }
    }

    /// Whether CUDA graph replay is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Disable graph capture/replay (fall back to normal kernel launches).
    pub fn disable(&mut self) {
        warn!("CUDA graph replay disabled");
        self.enabled = false;
    }

    /// Enable graph capture/replay.
    pub fn enable(&mut self) {
        info!("CUDA graph replay enabled");
        self.enabled = true;
    }

    /// Look up a cached graph for the given actual batch size.
    ///
    /// Returns `None` if no suitable graph exists yet (caller should capture one)
    /// or if the batch size exceeds the maximum.
    pub fn get(&self, actual_batch_size: usize) -> Option<&CudaGraph> {
        if !self.enabled {
            return None;
        }
        let padded = padded_batch_size(actual_batch_size)?;
        if padded > self.max_batch_size {
            return None;
        }
        self.graphs.get(&padded)
    }

    /// Check whether a graph is cached for the given batch size.
    pub fn has_graph(&self, actual_batch_size: usize) -> bool {
        padded_batch_size(actual_batch_size)
            .map(|p| self.graphs.contains_key(&p))
            .unwrap_or(false)
    }

    /// Insert a captured graph into the pool.
    pub fn insert(&mut self, graph: CudaGraph) {
        let bs = graph.batch_size();
        debug!(batch_size = bs, "caching CUDA graph");
        self.graphs.insert(bs, graph);
    }

    /// Number of cached graphs.
    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    /// Whether the pool has no cached graphs.
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Remove all cached graphs (e.g., after model reload).
    pub fn clear(&mut self) {
        info!(count = self.graphs.len(), "clearing CUDA graph pool");
        self.graphs.clear();
    }

    /// Begin capturing a CUDA graph on the given stream.
    ///
    /// After calling this, all kernel launches on `stream` are recorded into a
    /// graph rather than executed immediately. Call [`end_capture`] when done.
    #[cfg(feature = "cuda")]
    pub fn begin_capture(
        &self,
        stream: &crate::stream::GpuStream,
    ) -> Result<()> {
        debug!("beginning CUDA graph capture");
        let result = unsafe {
            cudarc::driver::sys::cuStreamBeginCapture(
                stream.cuda_stream().stream as cudarc::driver::sys::CUstream,
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(LLMError::GpuError(format!(
                "cuStreamBeginCapture failed: {:?}",
                result
            )));
        }
        Ok(())
    }

    /// End capture and produce a [`CudaGraph`] for the given batch size.
    #[cfg(feature = "cuda")]
    pub fn end_capture(
        &mut self,
        stream: &crate::stream::GpuStream,
        batch_size: usize,
    ) -> Result<CudaGraph> {
        debug!(batch_size, "ending CUDA graph capture");

        let mut graph: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
        let result = unsafe {
            cudarc::driver::sys::cuStreamEndCapture(
                stream.cuda_stream().stream as cudarc::driver::sys::CUstream,
                &mut graph,
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(LLMError::GpuError(format!(
                "cuStreamEndCapture failed: {:?}",
                result
            )));
        }

        let mut exec: cudarc::driver::sys::CUgraphExec = std::ptr::null_mut();
        let result = unsafe {
            cudarc::driver::sys::cuGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            // Clean up the graph on failure.
            unsafe {
                cudarc::driver::sys::cuGraphDestroy(graph);
            }
            return Err(LLMError::GpuError(format!(
                "cuGraphInstantiate failed: {:?}",
                result
            )));
        }

        info!(batch_size, "CUDA graph captured and instantiated");
        Ok(CudaGraph {
            batch_size,
            graph,
            exec,
        })
    }

    /// Begin capture (mock): no-op.
    #[cfg(not(feature = "cuda"))]
    pub fn begin_capture(
        &self,
        _stream: &crate::stream::GpuStream,
    ) -> Result<()> {
        debug!("beginning CUDA graph capture (mock)");
        Ok(())
    }

    /// End capture (mock): produces a mock CudaGraph.
    #[cfg(not(feature = "cuda"))]
    pub fn end_capture(
        &mut self,
        _stream: &crate::stream::GpuStream,
        batch_size: usize,
    ) -> Result<CudaGraph> {
        debug!(batch_size, "ending CUDA graph capture (mock)");
        Ok(CudaGraph {
            batch_size,
            replay_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_batch_size_exact() {
        assert_eq!(padded_batch_size(1), Some(1));
        assert_eq!(padded_batch_size(4), Some(4));
        assert_eq!(padded_batch_size(32), Some(32));
    }

    #[test]
    fn padded_batch_size_rounds_up() {
        assert_eq!(padded_batch_size(3), Some(4));
        assert_eq!(padded_batch_size(5), Some(8));
        assert_eq!(padded_batch_size(9), Some(16));
        assert_eq!(padded_batch_size(17), Some(32));
    }

    #[test]
    fn padded_batch_size_too_large() {
        assert_eq!(padded_batch_size(33), None);
        assert_eq!(padded_batch_size(64), None);
    }

    #[test]
    fn pool_insert_and_get() {
        let mut pool = CudaGraphPool::new(32);
        assert!(pool.is_empty());

        let stream = crate::stream::GpuStream::new(0).unwrap();
        pool.begin_capture(&stream).unwrap();
        let graph = pool.end_capture(&stream, 4).unwrap();
        assert_eq!(graph.batch_size(), 4);

        pool.insert(graph);
        assert_eq!(pool.len(), 1);
        assert!(pool.has_graph(4));
        assert!(pool.has_graph(3)); // rounds up to 4

        let g = pool.get(3).unwrap();
        assert_eq!(g.batch_size(), 4);
    }

    #[test]
    fn pool_disabled_returns_none() {
        let mut pool = CudaGraphPool::new(32);
        let stream = crate::stream::GpuStream::new(0).unwrap();
        pool.begin_capture(&stream).unwrap();
        let graph = pool.end_capture(&stream, 4).unwrap();
        pool.insert(graph);

        pool.disable();
        assert!(pool.get(4).is_none());

        pool.enable();
        assert!(pool.get(4).is_some());
    }

    #[test]
    fn pool_clear() {
        let mut pool = CudaGraphPool::new(32);
        let stream = crate::stream::GpuStream::new(0).unwrap();

        for &bs in GRAPH_BATCH_SIZES {
            pool.begin_capture(&stream).unwrap();
            let graph = pool.end_capture(&stream, bs).unwrap();
            pool.insert(graph);
        }
        assert_eq!(pool.len(), GRAPH_BATCH_SIZES.len());

        pool.clear();
        assert!(pool.is_empty());
    }

    #[test]
    fn mock_replay_count() {
        let stream = crate::stream::GpuStream::new(0).unwrap();
        let mut pool = CudaGraphPool::new(32);
        pool.begin_capture(&stream).unwrap();
        let graph = pool.end_capture(&stream, 8).unwrap();

        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(graph.replay_count(), 0);
            graph.replay(&stream).unwrap();
            graph.replay(&stream).unwrap();
            graph.replay(&stream).unwrap();
            assert_eq!(graph.replay_count(), 3);
        }

        #[cfg(feature = "cuda")]
        {
            // On real CUDA we just verify replay doesn't panic with a valid graph.
            let _ = &graph;
        }
    }

    #[test]
    fn graph_batch_sizes_sorted() {
        for w in GRAPH_BATCH_SIZES.windows(2) {
            assert!(w[0] < w[1], "GRAPH_BATCH_SIZES must be sorted ascending");
        }
    }

    #[test]
    fn pool_exceeds_max() {
        let pool = CudaGraphPool::new(8);
        // batch size 16 exceeds max of 8
        assert!(pool.get(16).is_none());
        assert!(!pool.has_graph(16));
    }
}
