//! Double-buffered pinned DtoH for argmax token IDs.
//!
//! `PinnedTokens` owns two pinned buffers + two CUevents. Each
//! `launch_dtoh` writes into the current write buffer and returns a
//! `DtoHTicket<'p>` that borrows `&mut PinnedTokens`. The ticket is
//! the *only* way to reach the read buffer; calling `wait` consumes
//! the ticket. This makes "read before wait" and "launch before
//! collect" compile errors.

use core::marker::PhantomData;

use rvllm_core::{Result, TokenId};
use rvllm_mem::PinnedPool;

/// Pool of 2 pinned argmax buffers + event coordination.
pub struct PinnedTokens {
    pub(crate) pool: PinnedPool<i32>,
    // Under feature "cuda", this also carries two Event handles.
    #[cfg(feature = "cuda")]
    _events: PhantomData<()>,
}

impl PinnedTokens {
    pub fn new(max_tokens: usize) -> Result<Self> {
        Ok(Self {
            pool: PinnedPool::new(max_tokens)?,
            #[cfg(feature = "cuda")]
            _events: PhantomData,
        })
    }

    /// Issue the DtoH for `num_tokens` tokens into the write buffer,
    /// record the event, flip, and return a ticket the caller must
    /// collect later.
    pub fn launch_dtoh(&mut self, num_tokens: u32) -> DtoHTicket<'_> {
        let buf_idx = self.pool.write_idx();
        // Real impl: cuMemcpyDtoHAsync(write_buf, argmax_device, 4*N);
        //            cuEventRecord(events[buf_idx]);
        self.pool.flip();
        DtoHTicket {
            pool: self,
            buf_idx,
            num_tokens,
            _marker: PhantomData,
        }
    }
}

/// Consume-once handle proving a DtoH was launched. The only way to
/// reach the tokens is `wait()`, which synchronizes the event and
/// returns the slice.
#[must_use = "DtoHTicket must be wait()-ed or dropped; silent drop leaks one pipeline slot"]
pub struct DtoHTicket<'p> {
    pool: &'p mut PinnedTokens,
    buf_idx: usize,
    num_tokens: u32,
    _marker: PhantomData<*const ()>, // !Send !Sync
}

impl<'p> DtoHTicket<'p> {
    /// Block on the event and return the token-id slice. Consumes self
    /// so a second `wait` cannot compile.
    pub fn wait(self) -> &'p [TokenId] {
        // Real impl: cuEventSynchronize(events[self.buf_idx]);
        // The buf at `buf_idx` was the WRITE buffer before flip; after
        // flip, `pool.read_idx() == buf_idx`.
        let buf = self.pool.pool.read_buf();
        let slice: &[i32] = &buf.as_slice()[..self.num_tokens as usize];
        // Safety: TokenId is #[repr(transparent)] over u32; i32 and u32
        // have the same size/align. We reinterpret knowing the kernel
        // writes non-negative values only (argmax indices).
        unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const TokenId, slice.len()) }
    }

    pub fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_then_wait_flow_compiles() {
        let mut pool = PinnedTokens::new(16).unwrap();
        // Host-stub: the "written" buffer is all zeros; wait reads it.
        let ticket = pool.launch_dtoh(4);
        let tokens = ticket.wait();
        assert_eq!(tokens.len(), 4);
        assert!(tokens.iter().all(|t| t.0 == 0));
    }

    // Negative compile tests (double-launch, wait-skipped) belong in
    // rvllm-invariants's trybuild harness (Phase A.5 to add).
}
