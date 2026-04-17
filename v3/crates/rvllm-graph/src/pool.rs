//! Per-bucket captured-graph pool.
//!
//! Graphs are captured at engine init (not lazily during warmup).
//! Each bucket stores (cuGraphExec handle, layout hash, fingerprint).
//! On replay, the runtime asserts the current bucket's `MetadataLayout`
//! hash still matches the captured one — drift = typed err.

use std::collections::BTreeMap;

use rvllm_core::{CudaErrorKind, GraphError, MetaLayoutHash, Result, RvllmError};
use rvllm_metadata::MetadataLayout;

/// The sha256 of a captured graph's structural signature — kernel
/// names and their arg-scalar values, in the order they were recorded.
/// Stored so a debug-build replay can re-walk the graph and verify.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct GraphFingerprint(pub [u8; 32]);

/// One captured graph. `exec` is the CUgraphExec handle stored as u64
/// so the type is the same under both cuda/no-cuda features.
#[derive(Debug)]
pub struct CapturedGraph {
    pub bucket: u32,
    pub max_blocks: u32,
    pub layout_hash: MetaLayoutHash,
    pub fingerprint: GraphFingerprint,
    exec: u64,
}

impl CapturedGraph {
    /// Capture a CUDA graph by recording `body`'s kernel launches on
    /// `stream`. The closure MUST launch only stream-capture-safe ops
    /// (no cudaMemcpy-sync, no host allocs that trigger implicit sync).
    ///
    /// # Safety
    /// Caller ensures the stream is a valid non-default CUDA stream and
    /// no other work is in flight on it.
    #[cfg(feature = "cuda")]
    pub unsafe fn capture(
        bucket: u32,
        max_blocks: u32,
        layout_hash: MetaLayoutHash,
        fingerprint: GraphFingerprint,
        stream: u64,
        body: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        use cudarc::driver::sys::*;
        let r = cuStreamBeginCapture_v2(
            stream as CUstream,
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        );
        if r != CUresult::CUDA_SUCCESS {
            return Err(graph_err(GraphError::CaptureFailed, bucket));
        }
        let body_res = body();
        let mut raw: CUgraph = core::ptr::null_mut();
        let r = cuStreamEndCapture(stream as CUstream, &mut raw);
        if r != CUresult::CUDA_SUCCESS {
            body_res?;
            return Err(graph_err(GraphError::CaptureFailed, bucket));
        }
        body_res?;
        let mut exec: CUgraphExec = core::ptr::null_mut();
        let r = cuGraphInstantiateWithFlags(&mut exec, raw, 0);
        let _ = cuGraphDestroy(raw);
        if r != CUresult::CUDA_SUCCESS {
            return Err(graph_err(GraphError::InstantiateFailed, bucket));
        }
        Ok(Self {
            bucket,
            max_blocks,
            layout_hash,
            fingerprint,
            exec: exec as u64,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub unsafe fn capture(
        bucket: u32,
        max_blocks: u32,
        layout_hash: MetaLayoutHash,
        fingerprint: GraphFingerprint,
        _stream: u64,
        body: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        body()?;
        Ok(Self {
            bucket,
            max_blocks,
            layout_hash,
            fingerprint,
            exec: 0,
        })
    }

    /// Launch the captured graph on `stream`.
    ///
    /// # Safety
    /// All device pointers the captured kernels reference must still be
    /// valid and the layout hash must still match (the pool's
    /// check_before_replay enforces the latter).
    pub unsafe fn replay(&self, stream: u64) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::*;
            let r = cuGraphLaunch(self.exec as CUgraphExec, stream as CUstream);
            if r != CUresult::CUDA_SUCCESS {
                return Err(graph_err(
                    GraphError::ReplayFailed {
                        cuda: CudaErrorKind::LaunchFailed,
                        kernel_at_fault: None,
                    },
                    self.bucket,
                ));
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
        }
        Ok(())
    }

    pub fn exec(&self) -> u64 {
        self.exec
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            if self.exec != 0 {
                let _ = cudarc::driver::sys::cuGraphExecDestroy(
                    self.exec as cudarc::driver::sys::CUgraphExec,
                );
            }
        }
    }
}

fn graph_err(kind: GraphError, bucket: u32) -> RvllmError {
    RvllmError::graph(kind, bucket)
}

/// Pool of graphs keyed by `(bucket, max_blocks)`.
#[derive(Default, Debug)]
pub struct GraphPool {
    graphs: BTreeMap<(u32, u32), CapturedGraph>,
}

impl GraphPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, g: CapturedGraph) {
        self.graphs.insert((g.bucket, g.max_blocks), g);
    }

    pub fn get(&self, bucket: u32, max_blocks: u32) -> Option<&CapturedGraph> {
        self.graphs.get(&(bucket, max_blocks))
    }

    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Before replay, verify the current layout for `(bucket, max_blocks)`
    /// still matches the hash baked into the captured graph. Drift =
    /// typed error (graph must be re-captured or config is wrong).
    pub fn check_before_replay(
        &self,
        bucket: u32,
        max_blocks: u32,
        current: &MetadataLayout,
    ) -> Result<&CapturedGraph> {
        let g = self.get(bucket, max_blocks).ok_or_else(|| {
            RvllmError::graph(GraphError::BucketMissing { padded_batch: bucket }, bucket)
        })?;
        let h = current.hash();
        if h != g.layout_hash {
            return Err(RvllmError::graph(
                GraphError::CaptureMetadataMismatch {
                    captured: g.layout_hash,
                    replay: h,
                },
                bucket,
            ));
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_graph(bucket: u32, max_blocks: u32) -> CapturedGraph {
        let layout = MetadataLayout::compute(bucket, max_blocks);
        CapturedGraph {
            bucket,
            max_blocks,
            layout_hash: layout.hash(),
            fingerprint: GraphFingerprint([0u8; 32]),
            exec: 0,
        }
    }

    #[test]
    fn matching_layout_replays() {
        let mut pool = GraphPool::new();
        pool.insert(fake_graph(128, 129));
        let layout = MetadataLayout::compute(128, 129);
        assert!(pool.check_before_replay(128, 129, &layout).is_ok());
    }

    #[test]
    fn drift_returns_typed_err() {
        let mut pool = GraphPool::new();
        pool.insert(fake_graph(128, 129));
        let wrong = MetadataLayout::compute(128, 257);
        let err = pool.check_before_replay(128, 129, &wrong).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("CaptureMetadataMismatch"));
    }

    #[test]
    fn missing_bucket_is_typed_err() {
        let pool = GraphPool::new();
        let layout = MetadataLayout::compute(1, 8);
        let err = pool.check_before_replay(1, 8, &layout).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("BucketMissing"));
        assert!(s.contains("padded_batch: 1"));
    }
}
