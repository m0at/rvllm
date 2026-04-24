//! Kernel signature descriptors.
//!
//! Each kernel shipped in the tarball has an entry here that names its
//! PTX function symbol and its expected argument kinds. The loader
//! checks the symbol is present; downstream crates (fused / attention /
//! cutlass) consume these descriptors to build typed launchers.

use rvllm_core::DType;

/// Argument category at the FFI boundary.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArgKind {
    DevicePtr,
    Scalar(DType),
    /// Fixed-size workspace pointer; size queried via a companion fn.
    WorkspacePtr,
    /// A stream handle (u64).
    Stream,
}

/// One kernel's FFI signature.
#[derive(Clone, Debug)]
pub struct KernelSig {
    pub name: &'static str,
    pub module: &'static str,
    pub args: &'static [ArgKind],
}

/// Built-in catalog of every fused kernel the v3 runtime uses.
/// Matches `v3/specs/12-fused.md` §catalog.
pub const FUSED_KERNELS: &[KernelSig] = &[
    KernelSig {
        name: "embedding_gather",
        module: "embedding",
        args: &[
            ArgKind::DevicePtr,          // out f16 [T, H]
            ArgKind::DevicePtr,          // weights f16 [V, H]
            ArgKind::DevicePtr,          // token_ids i32 [T]
            ArgKind::Scalar(DType::I32), // hidden_size
            ArgKind::Scalar(DType::I32), // vocab_size
        ],
    },
    KernelSig {
        name: "fused_add_rmsnorm_fp8_quant",
        module: "fused_norm_quant",
        args: &[
            ArgKind::DevicePtr,          // out fp8 [T, H]
            ArgKind::DevicePtr,          // scale f32 [T]
            ArgKind::DevicePtr,          // residual_out f16 [T, H]
            ArgKind::DevicePtr,          // in hidden f16
            ArgKind::DevicePtr,          // residual_in f16
            ArgKind::DevicePtr,          // gamma f16
            ArgKind::Scalar(DType::F32), // eps
            ArgKind::Scalar(DType::I32), // hidden_size
        ],
    },
    KernelSig {
        name: "fused_rmsnorm_fp8_quant",
        module: "fused_norm_quant",
        args: &[
            ArgKind::DevicePtr, // out fp8
            ArgKind::DevicePtr, // scale f32
            ArgKind::DevicePtr, // in f16
            ArgKind::DevicePtr, // gamma f16
            ArgKind::Scalar(DType::F32),
            ArgKind::Scalar(DType::I32),
        ],
    },
    KernelSig {
        name: "quantize_fp8_per_token",
        module: "fused_norm_quant",
        args: &[
            ArgKind::DevicePtr,          // out fp8
            ArgKind::DevicePtr,          // scale f32
            ArgKind::DevicePtr,          // in f16
            ArgKind::Scalar(DType::I32), // dim
        ],
    },
    KernelSig {
        name: "fused_rope_kv_write",
        module: "fused_rope_kv",
        args: &[
            ArgKind::DevicePtr,          // qkv in/out (Q rotated in-place)
            ArgKind::DevicePtr,          // k_cache
            ArgKind::DevicePtr,          // v_cache
            ArgKind::DevicePtr,          // positions
            ArgKind::DevicePtr,          // cos
            ArgKind::DevicePtr,          // sin
            ArgKind::DevicePtr,          // slot_mapping
            ArgKind::Scalar(DType::I32), // q_dim
            ArgKind::Scalar(DType::I32), // kv_dim
            ArgKind::Scalar(DType::I32), // num_tokens
        ],
    },
    KernelSig {
        name: "fused_silu_mul_fp8_quant",
        module: "fused_silu_quant",
        args: &[
            ArgKind::DevicePtr,          // out fp8
            ArgKind::DevicePtr,          // scale f32
            ArgKind::DevicePtr,          // gate_up f16
            ArgKind::Scalar(DType::I32), // num_tokens
            ArgKind::Scalar(DType::I32), // intermediate
        ],
    },
    KernelSig {
        name: "argmax",
        module: "argmax",
        args: &[
            ArgKind::DevicePtr,          // logits f32 [T, V]
            ArgKind::DevicePtr,          // out i32 [T]
            ArgKind::Scalar(DType::I32), // vocab_size
        ],
    },
    KernelSig {
        name: "residual_add_f16",
        module: "residual",
        args: &[
            ArgKind::DevicePtr,          // x in/out
            ArgKind::DevicePtr,          // y
            ArgKind::Scalar(DType::I32), // n
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_expected_kernels() {
        let names: Vec<&str> = FUSED_KERNELS.iter().map(|k| k.name).collect();
        assert!(names.contains(&"fused_add_rmsnorm_fp8_quant"));
        assert!(names.contains(&"fused_silu_mul_fp8_quant"));
        assert!(names.contains(&"argmax"));
        assert_eq!(FUSED_KERNELS.len(), 8);
    }

    #[test]
    fn every_kernel_has_at_least_one_device_ptr() {
        for k in FUSED_KERNELS {
            assert!(
                k.args.iter().any(|a| matches!(a, ArgKind::DevicePtr)),
                "kernel {} has no DevicePtr arg",
                k.name
            );
        }
    }
}
