//! FFI bindings for persistent_layer_v2 and megakernel_v2 cubin kernels.
//!
//! These are cooperative kernels compiled to .cubin via nvcc. They are loaded
//! at runtime via cuModuleLoadData and launched via cuLaunchCooperativeKernel.
//!
//! persistent_layer_v2: executes one full transformer layer in a single launch.
//! megakernel_v2: executes all layers + LM head in a single launch via instruction interpreter.

use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::sys::{self, CUfunction, CUmodule, CUstream};
use cudarc::driver::CudaContext;
use tracing::{debug, info};

// ============================================================================
// Raw cubin module (same pattern as kernel_loader.rs RawCubinModule)
// ============================================================================

struct RawModule {
    module: CUmodule,
}

impl RawModule {
    unsafe fn load(bytes: &[u8]) -> Result<Self, String> {
        let mut handle = std::mem::MaybeUninit::<CUmodule>::uninit();
        let result = sys::cuModuleLoadData(handle.as_mut_ptr(), bytes.as_ptr() as *const c_void);
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuModuleLoadData failed: {:?}", result));
        }
        Ok(Self {
            module: handle.assume_init(),
        })
    }

    fn get_function(&self, name: &str) -> Result<CUfunction, String> {
        let c_name =
            std::ffi::CString::new(name).map_err(|e| format!("invalid function name: {e}"))?;
        let mut func = std::mem::MaybeUninit::<CUfunction>::uninit();
        let result =
            unsafe { sys::cuModuleGetFunction(func.as_mut_ptr(), self.module, c_name.as_ptr()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuModuleGetFunction('{}') failed: {:?}",
                name, result
            ));
        }
        Ok(unsafe { func.assume_init() })
    }
}

impl Drop for RawModule {
    fn drop(&mut self) {
        unsafe {
            sys::cuModuleUnload(self.module);
        }
    }
}

unsafe impl Send for RawModule {}
unsafe impl Sync for RawModule {}

// v2 megakernel uses direct per-layer weight arrays (no instruction tape).

// ============================================================================
// PersistentV2Kernels -- loaded kernel handles
// ============================================================================

/// Holds loaded cubin modules and resolved function handles for the
/// persistent v2 layer kernel and megakernel.
pub struct PersistentV2Kernels {
    _layer_module: Option<RawModule>,
    _mega_module: Option<RawModule>,
    _v3_module: Option<RawModule>,
    layer_func: Option<CUfunction>,
    mega_func: Option<CUfunction>,
    v3_func: Option<CUfunction>,
    context: Arc<CudaContext>,
}

unsafe impl Send for PersistentV2Kernels {}
unsafe impl Sync for PersistentV2Kernels {}

impl PersistentV2Kernels {
    /// Load persistent_layer_v2 and megakernel_v2 cubins from candidate paths.
    ///
    /// Searches in order:
    ///   1. `kernels/sm_90/persistent_layer_v2.cubin`
    ///   2. `kernels/sm_90/megakernel_v2.cubin`
    ///
    /// Returns Ok even if only one kernel is found. Returns Err only if neither
    /// cubin can be loaded.
    pub fn load(context: Arc<CudaContext>) -> Result<Self, String> {
        context
            .bind_to_thread()
            .map_err(|e| format!("CUDA bind for persistent_v2 load: {e}"))?;

        let layer_candidates = ["kernels/sm_90/persistent_layer_v2.cubin"];
        let mega_candidates = ["kernels/sm_90/megakernel_v2.cubin"];
        let v3_candidates = ["kernels/sm_90/persistent_layer_v3.cubin"];

        let (layer_module, layer_func) =
            Self::try_load_cubin(&layer_candidates, "persistent_layer_v2_f16");
        let (mega_module, mega_func) = Self::try_load_cubin(&mega_candidates, "megakernel_v2_f16");
        let (v3_module, v3_func) = Self::try_load_cubin(&v3_candidates, "persistent_layer_v3_f16");

        if layer_func.is_none() && mega_func.is_none() && v3_func.is_none() {
            return Err("persistent_v2: no cubin files found in any candidate path".into());
        }

        if layer_func.is_some() {
            info!("persistent_layer_v2 cubin loaded");
        }
        if mega_func.is_some() {
            info!("megakernel_v2 cubin loaded");
        }
        if v3_func.is_some() {
            info!("persistent_layer_v3 cubin loaded");
        }

        Ok(Self {
            _layer_module: layer_module,
            _mega_module: mega_module,
            _v3_module: v3_module,
            layer_func,
            mega_func,
            v3_func,
            context,
        })
    }

    fn try_load_cubin(
        candidates: &[&str],
        func_name: &str,
    ) -> (Option<RawModule>, Option<CUfunction>) {
        for path_str in candidates {
            let path = Path::new(path_str);
            if !path.exists() {
                debug!(path = %path.display(), "persistent_v2 cubin not found, skipping");
                continue;
            }
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    debug!(path = %path.display(), error = %e, "failed to read cubin");
                    continue;
                }
            };
            match unsafe { RawModule::load(&bytes) } {
                Ok(module) => match module.get_function(func_name) {
                    Ok(func) => {
                        info!(path = %path.display(), func = func_name, "loaded cubin");
                        return (Some(module), Some(func));
                    }
                    Err(e) => {
                        debug!(path = %path.display(), error = %e, "cubin loaded but function not found");
                    }
                },
                Err(e) => {
                    debug!(path = %path.display(), error = %e, "cuModuleLoadData failed");
                }
            }
        }
        (None, None)
    }

    /// Whether the single-layer persistent kernel is available.
    pub fn has_layer_kernel(&self) -> bool {
        self.layer_func.is_some()
    }

    /// Whether the megakernel (all-layers) is available.
    pub fn has_mega_kernel(&self) -> bool {
        self.mega_func.is_some()
    }

    /// Whether the v3 layer kernel (non-cooperative, 1024 blocks) is available.
    pub fn has_v3_kernel(&self) -> bool {
        self.v3_func.is_some()
    }

    /// Raw CUfunction for the layer kernel (for occupancy queries).
    pub fn layer_cu_function(&self) -> Option<CUfunction> {
        self.layer_func
    }

    /// Raw CUfunction for the megakernel (for occupancy queries).
    pub fn mega_cu_function(&self) -> Option<CUfunction> {
        self.mega_func
    }

    // ========================================================================
    // Scratch buffer size helpers
    // ========================================================================

    /// Sync flags buffer size for a single persistent_layer_v2 launch.
    /// PLV2_NUM_SYNCS = 6.
    pub fn layer_sync_flags_size() -> usize {
        6 * std::mem::size_of::<i32>()
    }

    /// Sync flags buffer size for the megakernel.
    /// Layout: num_layers * 7 counters + 2 global counters.
    pub fn megakernel_sync_flags_size(num_layers: usize) -> usize {
        (num_layers * 7 + 2) * std::mem::size_of::<i32>()
    }

    /// Compute split-KV scratch buffer sizes for the attention combine phase.
    ///
    /// Returns `(max_buf_size, sum_buf_size, acc_buf_size)` in bytes.
    ///   - `max_buf`: `[max_splits * num_kv_heads * heads_per_group]` f32
    ///   - `sum_buf`: `[max_splits * num_kv_heads * heads_per_group]` f32
    ///   - `acc_buf`: `[max_splits * num_kv_heads * heads_per_group * head_dim]` f16
    pub fn split_kv_scratch_size(
        num_kv_heads: usize,
        max_splits: usize,
        heads_per_group: usize,
        head_dim: usize,
    ) -> (usize, usize, usize) {
        let total_heads = num_kv_heads * heads_per_group;
        let max_buf = max_splits * total_heads * std::mem::size_of::<f32>();
        let sum_buf = max_splits * total_heads * std::mem::size_of::<f32>();
        let acc_buf = max_splits * total_heads * head_dim * std::mem::size_of::<u16>(); // f16 = 2 bytes
        (max_buf, sum_buf, acc_buf)
    }

    // ========================================================================
    // Launch: persistent_layer_v2_f16
    // ========================================================================

    /// Launch a single persistent transformer layer.
    ///
    /// `sync_flags_ptr` must point to a device buffer of at least
    /// `layer_sync_flags_size()` bytes, **zeroed** before this call
    /// (caller must issue cudaMemsetAsync).
    ///
    /// # Safety
    /// All device pointers must be valid and correctly sized.
    /// The stream must be on the same CUDA context as the loaded module.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_layer(
        &self,
        stream: CUstream,
        grid_blocks: u32,
        shared_mem: u32,
        // Outputs
        mlp_out: u64,
        residual_out: u64,
        // Inputs
        prev_residual: u64,
        prev_mlp: u64, // 0 for layer 0
        // Attention I/O
        key_cache: u64,
        value_cache: u64,
        block_tables: u64,
        context_lens: u64,
        positions: u64,
        slot_mapping: u64,
        rope_cos: u64,
        rope_sin: u64,
        // Weights
        norm_w: u64,
        qkv_weight: u64,
        qkv_bias: u64, // 0 if no bias
        o_weight: u64,
        post_norm_w: u64,
        gateup_weight: u64,
        down_weight: u64,
        // Scratch buffers
        qkv_scratch: u64,
        attn_scratch: u64,
        oproj_scratch: u64,
        gateup_scratch: u64,
        // Split-KV scratch
        split_max_buf: u64,
        split_sum_buf: u64,
        split_acc_buf: u64,
        max_splits: i32,
        // Config
        eps: f32,
        attn_scale: f32,
        hidden_size: i32,
        q_dim: i32,
        kv_dim: i32,
        qkv_dim: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        intermediate_size: i32,
        gate_up_dim: i32,
        block_size: i32,
        max_context_len: i32,
        max_blocks_per_seq: i32,
        sync_flags: u64,
    ) -> Result<(), String> {
        let func = self
            .layer_func
            .ok_or("persistent_layer_v2_f16 not loaded")?;

        self.context
            .bind_to_thread()
            .map_err(|e| format!("CUDA bind: {e}"))?;

        // Set dynamic shared memory limit if above 48 KB default
        if shared_mem > 49152 {
            let result = unsafe {
                sys::cuFuncSetAttribute(
                    func,
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem as i32,
                )
            };
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "cuFuncSetAttribute(smem={shared_mem}): {:?}",
                    result
                ));
            }
        }

        // Build args array -- each element is a pointer to the arg value.
        // Order matches the kernel signature exactly.
        #[allow(clippy::cast_ptr_alignment)]
        let mut args: [*mut c_void; 42] = [
            &mlp_out as *const u64 as *mut c_void,
            &residual_out as *const u64 as *mut c_void,
            &prev_residual as *const u64 as *mut c_void,
            &prev_mlp as *const u64 as *mut c_void,
            &key_cache as *const u64 as *mut c_void,
            &value_cache as *const u64 as *mut c_void,
            &block_tables as *const u64 as *mut c_void,
            &context_lens as *const u64 as *mut c_void,
            &positions as *const u64 as *mut c_void,
            &slot_mapping as *const u64 as *mut c_void,
            &rope_cos as *const u64 as *mut c_void,
            &rope_sin as *const u64 as *mut c_void,
            // Weights
            &norm_w as *const u64 as *mut c_void,
            &qkv_weight as *const u64 as *mut c_void,
            &qkv_bias as *const u64 as *mut c_void,
            &o_weight as *const u64 as *mut c_void,
            &post_norm_w as *const u64 as *mut c_void,
            &gateup_weight as *const u64 as *mut c_void,
            &down_weight as *const u64 as *mut c_void,
            // Scratch
            &qkv_scratch as *const u64 as *mut c_void,
            &attn_scratch as *const u64 as *mut c_void,
            &oproj_scratch as *const u64 as *mut c_void,
            &gateup_scratch as *const u64 as *mut c_void,
            // Split-KV scratch
            &split_max_buf as *const u64 as *mut c_void,
            &split_sum_buf as *const u64 as *mut c_void,
            &split_acc_buf as *const u64 as *mut c_void,
            &max_splits as *const i32 as *mut c_void,
            // Config scalars
            &eps as *const f32 as *mut c_void,
            &attn_scale as *const f32 as *mut c_void,
            &hidden_size as *const i32 as *mut c_void,
            &q_dim as *const i32 as *mut c_void,
            &kv_dim as *const i32 as *mut c_void,
            &qkv_dim as *const i32 as *mut c_void,
            &num_heads as *const i32 as *mut c_void,
            &num_kv_heads as *const i32 as *mut c_void,
            &head_dim as *const i32 as *mut c_void,
            &intermediate_size as *const i32 as *mut c_void,
            &gate_up_dim as *const i32 as *mut c_void,
            &block_size as *const i32 as *mut c_void,
            &max_context_len as *const i32 as *mut c_void,
            &max_blocks_per_seq as *const i32 as *mut c_void,
            &sync_flags as *const u64 as *mut c_void,
        ];

        // Cooperative launch -- all blocks must be co-resident
        unsafe {
            crate::cooperative::launch_cooperative(
                func,
                (grid_blocks, 1, 1),
                (256, 1, 1), // PLV2_THREADS
                shared_mem,
                stream,
                &mut args,
            )
            .map_err(|e| format!("persistent_layer_v2 cooperative launch: {e}"))?;
        }

        Ok(())
    }

    // ========================================================================
    // Launch: megakernel_v2_f16
    // ========================================================================

    /// Launch the megakernel that executes all layers + embedding + LM head + argmax.
    ///
    /// `sync_flags` must point to a zeroed device buffer of at least
    /// `megakernel_sync_flags_size(num_layers)` bytes.
    ///
    /// Weight arrays are per-layer pointer arrays (GPU buffers of u64 pointers,
    /// one entry per layer).
    ///
    /// # Safety
    /// All device pointers must be valid and correctly sized.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_megakernel(
        &self,
        stream: CUstream,
        grid_blocks: u32,
        shared_mem: u32,
        // Per-layer weight pointer arrays (each is a GPU array of num_layers pointers)
        norm_weights: u64,
        qkv_weights: u64,
        qkv_biases: u64,
        o_weights: u64,
        post_norm_weights: u64,
        gateup_weights: u64,
        down_weights: u64,
        // Per-layer KV cache pointer arrays
        key_caches: u64,
        value_caches: u64,
        // Global weights
        embed_tokens: u64,
        final_norm_weight: u64,
        lm_head_weight: u64,
        // Input metadata
        input_token: u64,
        positions: u64,
        slot_mapping: u64,
        block_tables: u64,
        context_lens: u64,
        rope_cos: u64,
        rope_sin: u64,
        // Scratch buffers
        hidden_buf: u64,
        mlp_buf: u64,
        qkv_scratch: u64,
        attn_scratch: u64,
        oproj_scratch: u64,
        gateup_scratch: u64,
        logits_scratch: u64,
        // Split-KV scratch
        split_max_buf: u64,
        split_sum_buf: u64,
        split_acc_buf: u64,
        max_splits: i32,
        // Sync
        sync_flags: u64,
        // Config
        num_layers: i32,
        hidden_size: i32,
        q_dim: i32,
        kv_dim: i32,
        qkv_dim: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        intermediate_size: i32,
        gate_up_dim: i32,
        block_size: i32,
        max_context_len: i32,
        max_blocks_per_seq: i32,
        vocab_size: i32,
        eps: f32,
        attn_scale: f32,
        // Output
        output_token: u64,
    ) -> Result<(), String> {
        let func = self.mega_func.ok_or("megakernel_v2_f16 not loaded")?;

        self.context
            .bind_to_thread()
            .map_err(|e| format!("CUDA bind: {e}"))?;

        if shared_mem > 49152 {
            let result = unsafe {
                sys::cuFuncSetAttribute(
                    func,
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem as i32,
                )
            };
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "cuFuncSetAttribute(smem={shared_mem}): {:?}",
                    result
                ));
            }
        }

        #[allow(clippy::cast_ptr_alignment)]
        let mut args: [*mut c_void; 48] = [
            // Per-layer weight arrays
            &norm_weights as *const u64 as *mut c_void,
            &qkv_weights as *const u64 as *mut c_void,
            &qkv_biases as *const u64 as *mut c_void,
            &o_weights as *const u64 as *mut c_void,
            &post_norm_weights as *const u64 as *mut c_void,
            &gateup_weights as *const u64 as *mut c_void,
            &down_weights as *const u64 as *mut c_void,
            // KV caches
            &key_caches as *const u64 as *mut c_void,
            &value_caches as *const u64 as *mut c_void,
            // Global weights
            &embed_tokens as *const u64 as *mut c_void,
            &final_norm_weight as *const u64 as *mut c_void,
            &lm_head_weight as *const u64 as *mut c_void,
            // Input metadata
            &input_token as *const u64 as *mut c_void,
            &positions as *const u64 as *mut c_void,
            &slot_mapping as *const u64 as *mut c_void,
            &block_tables as *const u64 as *mut c_void,
            &context_lens as *const u64 as *mut c_void,
            &rope_cos as *const u64 as *mut c_void,
            &rope_sin as *const u64 as *mut c_void,
            // Scratch
            &hidden_buf as *const u64 as *mut c_void,
            &mlp_buf as *const u64 as *mut c_void,
            &qkv_scratch as *const u64 as *mut c_void,
            &attn_scratch as *const u64 as *mut c_void,
            &oproj_scratch as *const u64 as *mut c_void,
            &gateup_scratch as *const u64 as *mut c_void,
            &logits_scratch as *const u64 as *mut c_void,
            // Split-KV
            &split_max_buf as *const u64 as *mut c_void,
            &split_sum_buf as *const u64 as *mut c_void,
            &split_acc_buf as *const u64 as *mut c_void,
            &max_splits as *const i32 as *mut c_void,
            // Sync
            &sync_flags as *const u64 as *mut c_void,
            // Config
            &num_layers as *const i32 as *mut c_void,
            &hidden_size as *const i32 as *mut c_void,
            &q_dim as *const i32 as *mut c_void,
            &kv_dim as *const i32 as *mut c_void,
            &qkv_dim as *const i32 as *mut c_void,
            &num_heads as *const i32 as *mut c_void,
            &num_kv_heads as *const i32 as *mut c_void,
            &head_dim as *const i32 as *mut c_void,
            &intermediate_size as *const i32 as *mut c_void,
            &gate_up_dim as *const i32 as *mut c_void,
            &block_size as *const i32 as *mut c_void,
            &max_context_len as *const i32 as *mut c_void,
            &max_blocks_per_seq as *const i32 as *mut c_void,
            &vocab_size as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
            &attn_scale as *const f32 as *mut c_void,
            // Output
            &output_token as *const u64 as *mut c_void,
        ];

        unsafe {
            crate::cooperative::launch_cooperative(
                func,
                (grid_blocks, 1, 1),
                (256, 1, 1), // MK2_THREADS
                shared_mem,
                stream,
                &mut args,
            )
            .map_err(|e| format!("megakernel_v2 cooperative launch: {e}"))?;
        }

        Ok(())
    }

    // ========================================================================
    // Occupancy helpers
    // ========================================================================

    /// Compute max cooperative grid size for the layer kernel.
    pub fn layer_max_grid(&self, shared_mem: u32) -> Result<u32, String> {
        let func = self.layer_func.ok_or("layer kernel not loaded")?;
        unsafe {
            crate::cooperative::max_cooperative_grid(func, 256, shared_mem, &self.context)
                .map_err(|e| format!("layer occupancy query: {e}"))
        }
    }

    /// Compute max cooperative grid size for the megakernel.
    pub fn mega_max_grid(&self, shared_mem: u32) -> Result<u32, String> {
        let func = self.mega_func.ok_or("mega kernel not loaded")?;
        unsafe {
            crate::cooperative::max_cooperative_grid(func, 256, shared_mem, &self.context)
                .map_err(|e| format!("mega occupancy query: {e}"))
        }
    }

    /// Compute shared memory requirement for persistent_layer_v2 / megakernel_v2.
    ///
    /// Layout in GEMV phases (1/4/5/6):
    ///   normed_bytes = align128(hidden_size * 4 + WARPS * 4)
    ///   + weight double buffer = 2 * WARPS * RPW * TILE_K * 2 = 32768 bytes
    ///
    /// Layout in attention phase (3):
    ///   2 * BC * head_dim * 2 (KV tiles) + HPG * STRIDE * 4 (scores) + WARPS * 4
    ///
    /// Returns max of the two.
    pub fn layer_shared_mem(hidden_size: usize, head_dim: usize, heads_per_group: usize) -> u32 {
        // GEMV phases: normed f32 array + scratch + weight double buffer
        let warps = 8usize;
        let normed_bytes = (hidden_size * 4 + warps * 4 + 127) & !127; // align to 128
        let weight_double_buf = 2 * warps * 8 * 128 * 2; // 2 * WARPS * RPW * TILE_K * sizeof(half)
        let gemv_smem = normed_bytes + weight_double_buf;

        // Attention phase: KV tiles + scores + warp scratch
        let bc = 64usize;
        let hpg = heads_per_group.min(8);
        let stride = bc + 1; // bank conflict avoidance
        let attn_smem = 2 * bc * head_dim * 2 + hpg * stride * 4 + warps * 4;

        gemv_smem.max(attn_smem) as u32
    }

    // ========================================================================
    // V3: non-cooperative layer kernel (1024 blocks, ~17KB shared memory)
    // ========================================================================

    /// Compute shared memory for persistent_layer_v3.
    ///
    /// V3 has no weight double buffer. Shared memory is max of:
    ///   - GEMV phases: hidden_size * 4 (normed f32) + WARPS * 4 (scratch)
    ///   - Attention: 2 * BC * head_dim * 2 (KV tiles) + PLV3_FA_HPG * STRIDE * 4 + WARPS * 4
    pub fn v3_shared_mem(hidden_size: usize, head_dim: usize, heads_per_group: usize) -> u32 {
        let warps = 8usize;
        // GEMV phases: normed f32 + warp scratch (no weight buffer)
        let gemv_smem = hidden_size * 4 + warps * 4;

        // Attention phase: KV tiles + scores + warp scratch
        let bc = 32usize; // PLV3_FA_BC
        let _ = heads_per_group;
        let hpg = 8usize; // PLV3_FA_HPG scratch is fixed-size in the kernel
        let stride = bc + 1; // bank conflict avoidance
        let attn_smem = 2 * bc * head_dim * 2 + hpg * stride * 4 + warps * 4;

        gemv_smem.max(attn_smem) as u32
    }

    /// Compute the v3 grid size: min(1024, blocks_per_sm * num_sms).
    pub fn v3_max_grid(&self, shared_mem: u32) -> Result<u32, String> {
        let func = self.v3_func.ok_or("v3 kernel not loaded")?;
        let bpsm = unsafe {
            crate::cooperative::max_active_blocks_per_sm(func, 256, shared_mem)
                .map_err(|e| format!("v3 occupancy query: {e}"))?
        };
        let sms =
            crate::cooperative::sm_count(&self.context).map_err(|e| format!("v3 sm_count: {e}"))?;
        // v3 targets 1024 blocks (non-cooperative, no co-residency requirement)
        Ok((bpsm * sms).min(1024))
    }

    /// Launch persistent_layer_v3_f16 (non-cooperative, regular cuLaunchKernel).
    ///
    /// Same signature as v2 layer kernel. The difference is:
    ///   - Regular launch (not cooperative) -- blocks don't need to be co-resident
    ///   - Grid up to 1024 blocks (vs 256 for v2)
    ///   - ~17KB shared memory (vs ~47KB for v2)
    #[allow(clippy::too_many_arguments)]
    pub fn launch_v3_layer(
        &self,
        stream: CUstream,
        grid_blocks: u32,
        shared_mem: u32,
        // Outputs
        mlp_out: u64,
        residual_out: u64,
        // Inputs
        prev_residual: u64,
        prev_mlp: u64,
        // Attention I/O
        key_cache: u64,
        value_cache: u64,
        block_tables: u64,
        context_lens: u64,
        positions: u64,
        slot_mapping: u64,
        rope_cos: u64,
        rope_sin: u64,
        // Weights
        norm_w: u64,
        qkv_weight: u64,
        qkv_bias: u64,
        o_weight: u64,
        post_norm_w: u64,
        gateup_weight: u64,
        down_weight: u64,
        // Scratch buffers
        qkv_scratch: u64,
        attn_scratch: u64,
        oproj_scratch: u64,
        gateup_scratch: u64,
        // Split-KV scratch
        split_max_buf: u64,
        split_sum_buf: u64,
        split_acc_buf: u64,
        max_splits: i32,
        // Config
        eps: f32,
        attn_scale: f32,
        hidden_size: i32,
        q_dim: i32,
        kv_dim: i32,
        qkv_dim: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        intermediate_size: i32,
        gate_up_dim: i32,
        block_size: i32,
        max_context_len: i32,
        max_blocks_per_seq: i32,
        sync_flags: u64,
    ) -> Result<(), String> {
        let func = self.v3_func.ok_or("persistent_layer_v3_f16 not loaded")?;

        self.context
            .bind_to_thread()
            .map_err(|e| format!("CUDA bind: {e}"))?;

        if shared_mem > 49152 {
            let result = unsafe {
                sys::cuFuncSetAttribute(
                    func,
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem as i32,
                )
            };
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "cuFuncSetAttribute(smem={shared_mem}): {:?}",
                    result
                ));
            }
        }

        #[allow(clippy::cast_ptr_alignment)]
        let mut args: [*mut c_void; 42] = [
            &mlp_out as *const u64 as *mut c_void,
            &residual_out as *const u64 as *mut c_void,
            &prev_residual as *const u64 as *mut c_void,
            &prev_mlp as *const u64 as *mut c_void,
            &key_cache as *const u64 as *mut c_void,
            &value_cache as *const u64 as *mut c_void,
            &block_tables as *const u64 as *mut c_void,
            &context_lens as *const u64 as *mut c_void,
            &positions as *const u64 as *mut c_void,
            &slot_mapping as *const u64 as *mut c_void,
            &rope_cos as *const u64 as *mut c_void,
            &rope_sin as *const u64 as *mut c_void,
            &norm_w as *const u64 as *mut c_void,
            &qkv_weight as *const u64 as *mut c_void,
            &qkv_bias as *const u64 as *mut c_void,
            &o_weight as *const u64 as *mut c_void,
            &post_norm_w as *const u64 as *mut c_void,
            &gateup_weight as *const u64 as *mut c_void,
            &down_weight as *const u64 as *mut c_void,
            &qkv_scratch as *const u64 as *mut c_void,
            &attn_scratch as *const u64 as *mut c_void,
            &oproj_scratch as *const u64 as *mut c_void,
            &gateup_scratch as *const u64 as *mut c_void,
            &split_max_buf as *const u64 as *mut c_void,
            &split_sum_buf as *const u64 as *mut c_void,
            &split_acc_buf as *const u64 as *mut c_void,
            &max_splits as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
            &attn_scale as *const f32 as *mut c_void,
            &hidden_size as *const i32 as *mut c_void,
            &q_dim as *const i32 as *mut c_void,
            &kv_dim as *const i32 as *mut c_void,
            &qkv_dim as *const i32 as *mut c_void,
            &num_heads as *const i32 as *mut c_void,
            &num_kv_heads as *const i32 as *mut c_void,
            &head_dim as *const i32 as *mut c_void,
            &intermediate_size as *const i32 as *mut c_void,
            &gate_up_dim as *const i32 as *mut c_void,
            &block_size as *const i32 as *mut c_void,
            &max_context_len as *const i32 as *mut c_void,
            &max_blocks_per_seq as *const i32 as *mut c_void,
            &sync_flags as *const u64 as *mut c_void,
        ];

        // Non-cooperative launch: blocks don't need to be co-resident.
        // This allows up to 1024 blocks (vs 256 for cooperative v2).
        unsafe {
            sys::cuLaunchKernel(
                func,
                grid_blocks,
                1,
                1,
                256,
                1,
                1, // PLV3_THREADS
                shared_mem,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .result()
            .map_err(|e| format!("persistent_layer_v3 launch: {e}"))?;
        }

        Ok(())
    }

    /// Compute total scratch buffer sizes for megakernel_v2 in elements (not bytes).
    ///
    /// Returns (hidden_buf, mlp_buf, qkv_scratch, attn_scratch, oproj_scratch,
    ///          gateup_scratch, logits_scratch) all in f16 element counts.
    pub fn megakernel_scratch_elems(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
    ) -> (usize, usize, usize, usize, usize, usize, usize) {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let gate_up_dim = intermediate_size * 2;
        (
            hidden_size, // hidden_buf
            hidden_size, // mlp_buf
            qkv_dim,     // qkv_scratch
            q_dim,       // attn_scratch
            hidden_size, // oproj_scratch
            gate_up_dim, // gateup_scratch
            vocab_size,  // logits_scratch
        )
    }
}
