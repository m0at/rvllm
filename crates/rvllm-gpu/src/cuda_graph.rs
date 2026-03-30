//! CUDA graph capture and replay for decode steps.
//!
//! Captures a full decode-step kernel sequence into a CUDA graph, then replays
//! it each step instead of re-launching individual kernels. This eliminates
//! kernel launch overhead and yields 2-5x speedup for decode.
//!
//! Under `cuda-graphs` feature, uses the CUDA driver graph API via cudarc.
//! Under plain `cuda` or `mock-gpu`, the graph pool is a no-op: capture/replay
//! calls succeed but do nothing, so the engine falls back to normal launches.

use std::collections::HashMap;

use tracing::{debug, info, trace, warn};

#[cfg(feature = "cuda-graphs")]
use crate::LLMError;
use crate::Result;

/// Supported batch sizes for which we pre-capture CUDA graphs.
/// Extended range covers high-concurrency decode (inspired by b12x/sglang).
pub const GRAPH_BATCH_SIZES: &[usize] = &[
    1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
    104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192,
    200, 208, 216, 224, 232, 240, 248, 256,
];

/// Returns the smallest cached batch size >= `actual`, or `None` if `actual`
/// exceeds the largest cached size.
pub fn padded_batch_size(actual: usize) -> Option<usize> {
    GRAPH_BATCH_SIZES.iter().copied().find(|&s| s >= actual)
}

/// A captured CUDA graph that can be replayed.
pub struct CudaGraph {
    batch_size: usize,
    #[cfg(feature = "cuda-graphs")]
    inner: cudarc::driver::CudaGraph,
    #[cfg(not(feature = "cuda-graphs"))]
    replay_count: std::sync::atomic::AtomicUsize,
}

// SAFETY: The underlying CUDA graph/exec handles are device objects managed by
// the driver. They are thread-safe to send across threads; cudarc's CudaGraph
// holds raw pointers so doesn't auto-impl Send/Sync, but we guarantee
// single-writer access via &self in replay and pool ownership.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// Batch size this graph was captured for.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Replay the captured graph on `stream`.
    ///
    /// The graph is launched via cuGraphLaunch on the provided stream. All
    /// metadata must be written to the persistent GPU buffers on the SAME
    /// stream BEFORE calling this so CUDA stream ordering guarantees the
    /// kernels see the updated data.
    #[cfg(feature = "cuda-graphs")]
    pub fn replay(&self, stream: &crate::stream::GpuStream) -> Result<()> {
        trace!(batch_size = self.batch_size, "replaying CUDA graph");
        let cu_stream_ref = stream.cuda_stream();
        let cu_stream = cu_stream_ref.cu_stream();
        cu_stream_ref.context().bind_to_thread()
            .map_err(|e| LLMError::GpuError(format!("bind_to_thread: {e}")))?;
        unsafe {
            cudarc::driver::result::graph::launch(self.inner.cu_graph_exec(), cu_stream)
        }.map_err(|e| LLMError::GpuError(format!("cuGraphLaunch failed: {e}")))?;
        Ok(())
    }

    /// Replay (no-op): just increments the counter.
    #[cfg(not(feature = "cuda-graphs"))]
    pub fn replay(&self, _stream: &crate::stream::GpuStream) -> Result<()> {
        trace!(batch_size = self.batch_size, "replaying CUDA graph (no-op)");
        self.replay_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Number of times this graph has been replayed (no-op builds only, for testing).
    #[cfg(not(feature = "cuda-graphs"))]
    pub fn replay_count(&self) -> usize {
        self.replay_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        // Under cuda-graphs, cudarc::driver::CudaGraph handles cleanup in its own Drop.
        trace!(batch_size = self.batch_size, "CudaGraph dropped");
    }
}

/// Pool of pre-captured CUDA graphs keyed by batch size.
pub struct CudaGraphPool {
    graphs: HashMap<usize, CudaGraph>,
    max_batch_size: usize,
    enabled: bool,
}

impl CudaGraphPool {
    pub fn new(max_batch_size: usize) -> Self {
        info!(max_batch_size, "creating CudaGraphPool");
        Self {
            graphs: HashMap::new(),
            max_batch_size,
            enabled: true,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn disable(&mut self) {
        warn!("CUDA graph replay disabled");
        self.enabled = false;
    }

    pub fn enable(&mut self) {
        info!("CUDA graph replay enabled");
        self.enabled = true;
    }

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

    /// Get graph for exact batch size (no padding).
    pub fn get_exact(&self, batch_size: usize) -> Option<&CudaGraph> {
        if !self.enabled { return None; }
        self.graphs.get(&batch_size)
    }

    pub fn has_graph(&self, actual_batch_size: usize) -> bool {
        padded_batch_size(actual_batch_size)
            .map(|p| self.graphs.contains_key(&p))
            .unwrap_or(false)
    }

    pub fn insert(&mut self, graph: CudaGraph) {
        let bs = graph.batch_size();
        debug!(batch_size = bs, "caching CUDA graph");
        self.graphs.insert(bs, graph);
    }

    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    pub fn clear(&mut self) {
        info!(count = self.graphs.len(), "clearing CUDA graph pool");
        self.graphs.clear();
    }

    /// Begin capturing a CUDA graph on the given stream.
    #[cfg(feature = "cuda-graphs")]
    pub fn begin_capture(&self, stream: &crate::stream::GpuStream) -> Result<()> {
        debug!("beginning CUDA graph capture");
        stream
            .cuda_stream()
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
            .map_err(|e| LLMError::GpuError(format!("cuStreamBeginCapture failed: {e}")))?;
        Ok(())
    }

    /// End capture and produce a [`CudaGraph`] for the given batch size.
    #[cfg(feature = "cuda-graphs")]
    pub fn end_capture(
        &mut self,
        stream: &crate::stream::GpuStream,
        batch_size: usize,
    ) -> Result<CudaGraph> {
        use cudarc::driver::sys::CUgraphInstantiate_flags;
        debug!(batch_size, "ending CUDA graph capture");

        let inner = stream
            .cuda_stream()
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| LLMError::GpuError(format!("cuStreamEndCapture failed: {e}")))?
            .ok_or_else(|| {
                LLMError::GpuError("cuStreamEndCapture returned null graph".to_string())
            })?;

        info!(batch_size, "CUDA graph captured and instantiated");
        Ok(CudaGraph {
            batch_size,
            inner,
        })
    }

    /// Begin capture (no-op when cuda-graphs feature is off).
    #[cfg(not(feature = "cuda-graphs"))]
    pub fn begin_capture(&self, _stream: &crate::stream::GpuStream) -> Result<()> {
        debug!("beginning CUDA graph capture (no-op)");
        Ok(())
    }

    /// End capture (no-op): produces a stub CudaGraph.
    #[cfg(not(feature = "cuda-graphs"))]
    pub fn end_capture(
        &mut self,
        _stream: &crate::stream::GpuStream,
        batch_size: usize,
    ) -> Result<CudaGraph> {
        debug!(batch_size, "ending CUDA graph capture (no-op)");
        Ok(CudaGraph {
            batch_size,
            replay_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Begin capturing a CUDA graph on a raw CudaStream (for use from GpuModelRunner).
    #[cfg(feature = "cuda-graphs")]
    pub fn begin_capture_on(&self, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Result<()> {
        debug!("beginning CUDA graph capture (raw stream)");
        stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
            .map_err(|e| LLMError::GpuError(format!("cuStreamBeginCapture failed: {e}")))?;
        Ok(())
    }

    /// End capture on a raw CudaStream and produce a [`CudaGraph`].
    #[cfg(feature = "cuda-graphs")]
    pub fn end_capture_on(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        batch_size: usize,
    ) -> Result<CudaGraph> {
        use cudarc::driver::sys::CUgraphInstantiate_flags;
        debug!(batch_size, "ending CUDA graph capture (raw stream)");

        let inner = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| LLMError::GpuError(format!("cuStreamEndCapture failed: {e}")))?
            .ok_or_else(|| {
                LLMError::GpuError("cuStreamEndCapture returned null graph".to_string())
            })?;

        info!(batch_size, "CUDA graph captured and instantiated (raw stream)");
        Ok(CudaGraph {
            batch_size,
            inner,
        })
    }

    /// Begin capture (no-op when cuda-graphs feature is off, cuda still available).
    #[cfg(all(feature = "cuda", not(feature = "cuda-graphs")))]
    pub fn begin_capture_on(&self, _stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Result<()> {
        debug!("beginning CUDA graph capture (raw stream, no-op)");
        Ok(())
    }

    /// End capture (no-op): produces a stub CudaGraph.
    #[cfg(all(feature = "cuda", not(feature = "cuda-graphs")))]
    pub fn end_capture_on(
        &mut self,
        _stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        batch_size: usize,
    ) -> Result<CudaGraph> {
        debug!(batch_size, "ending CUDA graph capture (raw stream, no-op)");
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
        assert_eq!(padded_batch_size(17), Some(24));
        assert_eq!(padded_batch_size(33), Some(40));
        assert_eq!(padded_batch_size(65), Some(72));
        assert_eq!(padded_batch_size(129), Some(136));
    }

    #[test]
    fn padded_batch_size_too_large() {
        assert_eq!(padded_batch_size(257), None);
        assert_eq!(padded_batch_size(512), None);
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

        #[cfg(not(feature = "cuda-graphs"))]
        {
            assert_eq!(graph.replay_count(), 0);
            graph.replay(&stream).unwrap();
            graph.replay(&stream).unwrap();
            graph.replay(&stream).unwrap();
            assert_eq!(graph.replay_count(), 3);
        }

        #[cfg(feature = "cuda-graphs")]
        {
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
        assert!(pool.get(16).is_none());
        assert!(!pool.has_graph(16));
    }
}
