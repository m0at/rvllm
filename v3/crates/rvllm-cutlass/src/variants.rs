//! CUTLASS FP8 GEMM variant catalog.
//!
//! Each variant is one (tile, cluster, mainloop_schedule) template
//! instantiation. The epilogue schedule is paired via the
//! `MatchedPair` trait bound — any new variant with a schedule
//! mismatch fails to compile.
//!
//! The v2 audit (documented in `v3/IMPL_PLAN.md` section 4) flagged
//! v0/v2/v3/v4/v6/v8/v11/v12/v13/v14 of `cutlass_fp8_gemm_*` as
//! mismatched. Those cannot exist here by construction.

use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::schedule::{MatchedPair, Schedule, ScheduleTag};

/// A type-safe CUTLASS variant: paired mainloop/epilogue schedules
/// plus its runtime identity (tile + cluster + id).
///
/// `(M, E): MatchedPair` is what prevents WS/Coop mis-pairing from
/// compiling.
#[derive(Debug)]
pub struct Variant<M: Schedule, E: Schedule>
where
    (M, E): MatchedPair,
{
    pub id: VariantId,
    pub tile: TileShape,
    pub cluster: ClusterShape,
    _phantom: PhantomData<(M, E)>,
}

impl<M: Schedule, E: Schedule> Variant<M, E>
where
    (M, E): MatchedPair,
{
    pub const fn new(id: VariantId, tile: TileShape, cluster: ClusterShape) -> Self {
        Self {
            id,
            tile,
            cluster,
            _phantom: PhantomData,
        }
    }

    pub fn schedule_names(&self) -> (&'static str, &'static str) {
        (M::NAME, E::NAME)
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct VariantId(pub u32);

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct TileShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
impl TileShape {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct ClusterShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
impl ClusterShape {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
    pub const fn one() -> Self {
        Self::new(1, 1, 1)
    }
}

/// Serializable variant descriptor for `policy.json`.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct VariantDescriptor {
    pub id: VariantId,
    pub tile: TileShape,
    pub cluster: ClusterShape,
    pub mainloop: ScheduleTag,
    pub epilogue: ScheduleTag,
}

impl VariantDescriptor {
    /// Validates runtime-form variant. Callers loading policy.json must
    /// run this on every entry; unmatched pairs are rejected (typed).
    pub fn validate(&self) -> bool {
        self.mainloop.matches(self.epilogue)
    }
}

// ---------------------------------------------------------------------------
// Canonical v3 variant catalog. Only matched pairs. Each entry names
// the tile/cluster/schedule that CUTLASS instantiates on the GPU side;
// the Rust type parameters ensure the pair compiles.
// ---------------------------------------------------------------------------

/// Non-residual FP8 GEMM: baseline Coop variants for decode/prefill GEMMs.
pub const FP8_GEMM_COOP_128_128_128: Variant<crate::schedule::Coop, crate::schedule::Coop> =
    Variant::new(
        VariantId(0),
        TileShape::new(128, 128, 128),
        ClusterShape::one(),
    );

pub const FP8_GEMM_COOP_128_256_128: Variant<crate::schedule::Coop, crate::schedule::Coop> =
    Variant::new(
        VariantId(1),
        TileShape::new(128, 256, 128),
        ClusterShape::one(),
    );

pub const FP8_GEMM_WS_64_128_128: Variant<crate::schedule::WS, crate::schedule::WS> = Variant::new(
    VariantId(2),
    TileShape::new(64, 128, 128),
    ClusterShape::one(),
);

/// FP8FastAccum variants — used when autotune says they win at the shape.
pub const FP8_GEMM_FP8COOP_128_128_128: Variant<
    crate::schedule::Fp8Coop,
    crate::schedule::Fp8Coop,
> = Variant::new(
    VariantId(3),
    TileShape::new(128, 128, 128),
    ClusterShape::one(),
);

pub const FP8_GEMM_FP8WS_64_128_128: Variant<crate::schedule::Fp8WS, crate::schedule::Fp8WS> =
    Variant::new(
        VariantId(4),
        TileShape::new(64, 128, 128),
        ClusterShape::one(),
    );

/// Residual-fused FP8 GEMM (o_proj). Coop/Coop matches the
/// TmaWarpSpecializedCooperative epilogue in the .cu file. This is the
/// variant v2 pinned after the April 16 audit.
pub const FP8_GEMM_RESIDUAL_COOP: Variant<crate::schedule::Coop, crate::schedule::Coop> =
    Variant::new(
        VariantId(100),
        TileShape::new(128, 128, 128),
        ClusterShape::one(),
    );

/// Human-readable descriptor for every variant we ship, in policy form.
/// The engine writes this to `manifest.json::variants` at build time so
/// the deploy tarball carries the catalog next to the code.
pub fn canonical_variants() -> Vec<VariantDescriptor> {
    vec![
        VariantDescriptor {
            id: VariantId(0),
            tile: TileShape::new(128, 128, 128),
            cluster: ClusterShape::one(),
            mainloop: ScheduleTag::Coop,
            epilogue: ScheduleTag::Coop,
        },
        VariantDescriptor {
            id: VariantId(1),
            tile: TileShape::new(128, 256, 128),
            cluster: ClusterShape::one(),
            mainloop: ScheduleTag::Coop,
            epilogue: ScheduleTag::Coop,
        },
        VariantDescriptor {
            id: VariantId(2),
            tile: TileShape::new(64, 128, 128),
            cluster: ClusterShape::one(),
            mainloop: ScheduleTag::WS,
            epilogue: ScheduleTag::WS,
        },
        VariantDescriptor {
            id: VariantId(3),
            tile: TileShape::new(128, 128, 128),
            cluster: ClusterShape::one(),
            mainloop: ScheduleTag::Fp8Coop,
            epilogue: ScheduleTag::Fp8Coop,
        },
        VariantDescriptor {
            id: VariantId(4),
            tile: TileShape::new(64, 128, 128),
            cluster: ClusterShape::one(),
            mainloop: ScheduleTag::Fp8WS,
            epilogue: ScheduleTag::Fp8WS,
        },
        VariantDescriptor {
            id: VariantId(100),
            tile: TileShape::new(128, 128, 128),
            cluster: ClusterShape::one(),
            mainloop: ScheduleTag::Coop,
            epilogue: ScheduleTag::Coop,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_canonical_variant_is_matched_pair() {
        for v in canonical_variants() {
            assert!(
                v.validate(),
                "variant {} has mismatched schedules {:?}/{:?}",
                v.id.0,
                v.mainloop,
                v.epilogue,
            );
        }
    }

    #[test]
    fn schedule_names_from_type_params() {
        let (m, e) = FP8_GEMM_RESIDUAL_COOP.schedule_names();
        assert_eq!(m, "KernelTmaWarpSpecializedCooperative");
        assert_eq!(e, "KernelTmaWarpSpecializedCooperative");
    }

    #[test]
    fn rejects_ws_coop_mismatch_at_runtime() {
        let bad = VariantDescriptor {
            id: VariantId(999),
            tile: TileShape::new(64, 128, 128),
            cluster: ClusterShape::one(),
            mainloop: ScheduleTag::WS,
            epilogue: ScheduleTag::Coop,
        };
        assert!(!bad.validate());
    }
}
