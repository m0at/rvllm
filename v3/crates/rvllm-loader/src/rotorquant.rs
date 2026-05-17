//! RotorQuant metadata plus tiny CPU reference helpers.

use std::str::FromStr;

use rvllm_core::{ConfigError, Result, RvllmError};

pub const ROTORQUANT_MIN_BITS: u8 = 2;
pub const ROTORQUANT_MAX_BITS: u8 = 4;
pub const ROTORQUANT_V1_CHUNK_DIM: usize = 128;
pub const ROTORQUANT_KV_KIND_COUNT: usize = 2;
pub const ROTORQUANT_V1_RESIDUAL_BITS: Option<u8> = None;
pub const ROTORQUANT_KV_KERNELS_AVAILABLE_V1: bool = false;
pub const REFERENCE_MAX_VALUES: usize = 4096;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RotorQuantMode {
    RotorCl3,
    Planar2,
    Iso4,
}

impl RotorQuantMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RotorCl3 => "rotor_cl3",
            Self::Planar2 => "planar2",
            Self::Iso4 => "iso4",
        }
    }

    pub const fn block_dim(self) -> usize {
        match self {
            Self::RotorCl3 => 3,
            Self::Planar2 => 2,
            Self::Iso4 => 4,
        }
    }
}

impl FromStr for RotorQuantMode {
    type Err = RvllmError;

    fn from_str(raw: &str) -> Result<Self> {
        match raw {
            "1" | "rotor" | "rotor_cl3" | "cl3" => Ok(Self::RotorCl3),
            "planar2" | "planar" => Ok(Self::Planar2),
            "iso4" | "iso" => Ok(Self::Iso4),
            other => Err(invalid(
                "rotorquant.mode",
                format!("unsupported RotorQuant mode {other:?}"),
            )),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RotorQuantKvPath {
    RotorQuantDecode,
    FallbackFp8OrF16,
}

pub fn select_kv_cache_path(is_prefill: bool, kernels_available: bool) -> RotorQuantKvPath {
    if is_prefill || !kernels_available {
        RotorQuantKvPath::FallbackFp8OrF16
    } else {
        RotorQuantKvPath::RotorQuantDecode
    }
}

pub fn select_v1_kv_cache_path(is_prefill: bool) -> RotorQuantKvPath {
    select_kv_cache_path(is_prefill, ROTORQUANT_KV_KERNELS_AVAILABLE_V1)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CodebookMetadata {
    pub entries: usize,
    pub value_bytes: usize,
}

impl CodebookMetadata {
    pub fn new(entries: usize, value_bytes: usize) -> Result<Self> {
        if entries == 0 {
            return Err(invalid("rotorquant.codebook.entries", "expected > 0"));
        }
        if value_bytes == 0 {
            return Err(invalid("rotorquant.codebook.value_bytes", "expected > 0"));
        }
        Ok(Self {
            entries,
            value_bytes,
        })
    }

    pub fn for_bits(bits: u8) -> Result<Self> {
        validate_bits(bits)?;
        Self::new(1usize << bits, 4)
    }

    pub fn bytes(self) -> Result<usize> {
        self.entries
            .checked_mul(self.value_bytes)
            .ok_or_else(|| invalid("rotorquant.codebook", "byte size overflow"))
    }

    pub fn validate_for_bits(self, bits: u8) -> Result<()> {
        validate_bits(bits)?;
        if self.value_bytes == 0 {
            return Err(invalid("rotorquant.codebook.value_bytes", "expected > 0"));
        }
        let expected = 1usize << bits;
        if self.entries != expected {
            return Err(invalid(
                "rotorquant.codebook.entries",
                format!("expected {expected} entries for {bits}-bit quantization"),
            ));
        }
        self.bytes()?;
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RotorQuantMetadata {
    pub mode: RotorQuantMode,
    pub bits: u8,
    pub chunk_dim: usize,
    pub residual_bits: Option<u8>,
    pub codebook: CodebookMetadata,
}

impl RotorQuantMetadata {
    pub fn new(
        mode: RotorQuantMode,
        bits: u8,
        chunk_dim: usize,
        residual_bits: Option<u8>,
        codebook: CodebookMetadata,
    ) -> Result<Self> {
        validate_bits(bits)?;
        validate_chunk_dim(chunk_dim)?;
        validate_residual_bits(residual_bits)?;
        codebook.validate_for_bits(bits)?;
        Ok(Self {
            mode,
            bits,
            chunk_dim,
            residual_bits,
            codebook,
        })
    }

    pub fn with_default_codebook(
        mode: RotorQuantMode,
        bits: u8,
        chunk_dim: usize,
        residual_bits: Option<u8>,
    ) -> Result<Self> {
        Self::new(
            mode,
            bits,
            chunk_dim,
            residual_bits,
            CodebookMetadata::for_bits(bits)?,
        )
    }

    pub fn v1_kv(mode: RotorQuantMode, bits: u8) -> Result<Self> {
        Self::with_default_codebook(
            mode,
            bits,
            ROTORQUANT_V1_CHUNK_DIM,
            ROTORQUANT_V1_RESIDUAL_BITS,
        )
    }

    pub fn packed_values_bytes(self, value_count: usize) -> Result<usize> {
        packed_bytes_for_values(value_count, self.bits)
    }

    pub fn packed_chunk_bytes(self) -> Result<usize> {
        self.packed_values_bytes(self.chunk_dim)
    }

    pub fn residual_bytes(self, value_count: usize) -> Result<usize> {
        match self.residual_bits {
            Some(bits) => packed_bytes_for_bit_width(value_count, bits, "rotorquant.residual_bits"),
            None => Ok(0),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RotorQuantKvLayoutSummary {
    pub metadata: RotorQuantMetadata,
    pub layers: usize,
    pub blocks: usize,
    pub block_size: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub chunks_per_head: usize,
    pub packed_chunk_bytes: usize,
    pub residual_chunk_bytes: usize,
    pub packed_values_bytes: usize,
    pub residual_values_bytes: usize,
    pub rot_param_bytes: usize,
    pub codebook_bytes: usize,
}

impl RotorQuantKvLayoutSummary {
    pub fn new(
        metadata: RotorQuantMetadata,
        layers: usize,
        blocks: usize,
        block_size: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        validate_nonzero("rotorquant.layers", layers)?;
        validate_nonzero("rotorquant.blocks", blocks)?;
        validate_nonzero("rotorquant.block_size", block_size)?;
        validate_nonzero("rotorquant.kv_heads", kv_heads)?;
        validate_nonzero("rotorquant.head_dim", head_dim)?;
        if head_dim % metadata.chunk_dim != 0 {
            return Err(invalid(
                "rotorquant.head_dim",
                "head_dim must be divisible by chunk_dim",
            ));
        }
        let chunks_per_head = head_dim / metadata.chunk_dim;
        let packed_chunk_bytes = metadata.packed_chunk_bytes()?;
        let residual_chunk_bytes = metadata.residual_bytes(metadata.chunk_dim)?;
        let cache_chunks = checked_product(
            &[
                layers,
                ROTORQUANT_KV_KIND_COUNT,
                blocks,
                block_size,
                kv_heads,
                chunks_per_head,
            ],
            "rotorquant.kv_layout",
        )?;
        let packed_values_bytes = checked_mul(
            cache_chunks,
            packed_chunk_bytes,
            "rotorquant.kv_layout.values",
        )?;
        let residual_values_bytes = checked_mul(
            cache_chunks,
            residual_chunk_bytes,
            "rotorquant.kv_layout.residuals",
        )?;
        let rot_param_slots = checked_product(
            &[
                layers,
                ROTORQUANT_KV_KIND_COUNT,
                kv_heads,
                chunks_per_head,
                rotation_param_f32s_per_chunk(metadata.mode),
                std::mem::size_of::<f32>(),
            ],
            "rotorquant.kv_layout.rot_params",
        )?;
        Ok(Self {
            metadata,
            layers,
            blocks,
            block_size,
            kv_heads,
            head_dim,
            chunks_per_head,
            packed_chunk_bytes,
            residual_chunk_bytes,
            packed_values_bytes,
            residual_values_bytes,
            rot_param_bytes: rot_param_slots,
            codebook_bytes: metadata.codebook.bytes()?,
        })
    }

    pub fn values_shape(self) -> [usize; 7] {
        [
            self.layers,
            ROTORQUANT_KV_KIND_COUNT,
            self.blocks,
            self.block_size,
            self.kv_heads,
            self.chunks_per_head,
            self.packed_chunk_bytes,
        ]
    }

    pub fn residual_shape(self) -> Option<[usize; 7]> {
        (self.residual_chunk_bytes != 0).then_some([
            self.layers,
            ROTORQUANT_KV_KIND_COUNT,
            self.blocks,
            self.block_size,
            self.kv_heads,
            self.chunks_per_head,
            self.residual_chunk_bytes,
        ])
    }

    pub fn cache_payload_bytes(self) -> Result<usize> {
        checked_add(
            self.packed_values_bytes,
            self.residual_values_bytes,
            "rotorquant.kv_layout.payload",
        )
    }
}

pub const fn rotation_param_f32s_per_chunk(mode: RotorQuantMode) -> usize {
    match mode {
        RotorQuantMode::RotorCl3 => 4,
        RotorQuantMode::Planar2 => 2,
        RotorQuantMode::Iso4 => 8,
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Planar2Rotation {
    pub c: f32,
    pub s: f32,
}

impl Planar2Rotation {
    pub fn new(c: f32, s: f32) -> Result<Self> {
        if !c.is_finite() || !s.is_finite() {
            return Err(invalid("rotorquant.planar2", "rotation must be finite"));
        }
        let norm = c.mul_add(c, s * s);
        if (norm - 1.0).abs() > 1e-3 {
            return Err(invalid(
                "rotorquant.planar2",
                "rotation must be unit length",
            ));
        }
        Ok(Self { c, s })
    }

    pub fn from_angle(angle: f32) -> Result<Self> {
        if !angle.is_finite() {
            return Err(invalid("rotorquant.planar2", "angle must be finite"));
        }
        Self::new(angle.cos(), angle.sin())
    }

    pub const fn inverse(self) -> Self {
        Self {
            c: self.c,
            s: -self.s,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Iso4Rotation {
    pub left: [f32; 4],
    pub right: [f32; 4],
}

impl Iso4Rotation {
    pub fn new(left: [f32; 4], right: [f32; 4]) -> Result<Self> {
        validate_unit_quat("rotorquant.iso4.left", left)?;
        validate_unit_quat("rotorquant.iso4.right", right)?;
        Ok(Self { left, right })
    }

    pub const fn identity() -> Self {
        Self {
            left: [1.0, 0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn from_angles(left: f32, right: f32) -> Result<Self> {
        if !left.is_finite() || !right.is_finite() {
            return Err(invalid("rotorquant.iso4", "angles must be finite"));
        }
        Self::new(
            [left.cos(), left.sin(), 0.0, 0.0],
            [right.cos(), right.sin(), 0.0, 0.0],
        )
    }

    pub fn inverse(self) -> Self {
        Self {
            left: quat_conj(self.left),
            right: quat_conj(self.right),
        }
    }
}

pub fn validate_bits(bits: u8) -> Result<()> {
    if !(ROTORQUANT_MIN_BITS..=ROTORQUANT_MAX_BITS).contains(&bits) {
        return Err(invalid("rotorquant.bits", "expected 2, 3, or 4"));
    }
    Ok(())
}

pub fn packed_bytes_for_values(value_count: usize, bits: u8) -> Result<usize> {
    validate_bits(bits)?;
    packed_bytes_for_bit_width(value_count, bits, "rotorquant.bits")
}

fn packed_bytes_for_bit_width(value_count: usize, bits: u8, field: &'static str) -> Result<usize> {
    if bits == 0 || bits > 8 {
        return Err(invalid(field, "expected bit width in 1..=8"));
    }
    let bit_count = value_count
        .checked_mul(bits as usize)
        .ok_or_else(|| invalid("rotorquant.value_count", "packed bit size overflow"))?;
    bit_count
        .checked_add(7)
        .map(|bits| bits / 8)
        .ok_or_else(|| invalid("rotorquant.value_count", "packed byte size overflow"))
}

pub fn pack_indices_ref(indices: &[u8], bits: u8) -> Result<Vec<u8>> {
    validate_ref_count(indices.len())?;
    validate_bits(bits)?;
    let max_index = 1u8 << bits;
    if indices.iter().any(|&idx| idx >= max_index) {
        return Err(invalid(
            "rotorquant.indices",
            format!("index must be < {max_index}"),
        ));
    }
    let mut packed = vec![0u8; packed_bytes_for_values(indices.len(), bits)?];
    let mut bit = 0usize;
    for &idx in indices {
        let byte = bit / 8;
        let shift = bit % 8;
        packed[byte] |= idx << shift;
        if shift + bits as usize > 8 {
            packed[byte + 1] |= idx >> (8 - shift);
        }
        bit += bits as usize;
    }
    Ok(packed)
}

pub fn unpack_indices_ref(packed: &[u8], value_count: usize, bits: u8) -> Result<Vec<u8>> {
    validate_ref_count(value_count)?;
    validate_bits(bits)?;
    let needed = packed_bytes_for_values(value_count, bits)?;
    if packed.len() < needed {
        return Err(invalid(
            "rotorquant.packed",
            format!("expected at least {needed} bytes"),
        ));
    }
    let mask = (1u16 << bits) - 1;
    let mut indices = Vec::with_capacity(value_count);
    let mut bit = 0usize;
    for _ in 0..value_count {
        let byte = bit / 8;
        let shift = bit % 8;
        let mut word = (packed[byte] as u16) >> shift;
        if shift + bits as usize > 8 {
            word |= (packed[byte + 1] as u16) << (8 - shift);
        }
        indices.push((word & mask) as u8);
        bit += bits as usize;
    }
    Ok(indices)
}

pub fn dequantize_codebook_ref(
    packed: &[u8],
    value_count: usize,
    bits: u8,
    codebook: &[f32],
) -> Result<Vec<f32>> {
    validate_bits(bits)?;
    let expected = 1usize << bits;
    if codebook.len() != expected {
        return Err(invalid(
            "rotorquant.codebook",
            format!("expected {expected} entries"),
        ));
    }
    if codebook.iter().any(|v| !v.is_finite()) {
        return Err(invalid("rotorquant.codebook", "entries must be finite"));
    }
    let indices = unpack_indices_ref(packed, value_count, bits)?;
    Ok(indices
        .into_iter()
        .map(|idx| codebook[idx as usize])
        .collect())
}

pub fn rotate_planar2_blocks_ref(
    values: &[f32],
    rotations: &[Planar2Rotation],
) -> Result<Vec<f32>> {
    validate_ref_blocks(values, 2, rotations.len(), "rotorquant.planar2")?;
    for &rotation in rotations {
        Planar2Rotation::new(rotation.c, rotation.s)?;
    }
    let mut out = values.to_vec();
    for (xy, r) in out.chunks_exact_mut(2).zip(rotations) {
        let x = xy[0];
        let y = xy[1];
        xy[0] = r.c.mul_add(x, -r.s * y);
        xy[1] = r.s.mul_add(x, r.c * y);
    }
    Ok(out)
}

pub fn inverse_rotate_planar2_blocks_ref(
    values: &[f32],
    rotations: &[Planar2Rotation],
) -> Result<Vec<f32>> {
    let inverse: Vec<_> = rotations
        .iter()
        .copied()
        .map(Planar2Rotation::inverse)
        .collect();
    rotate_planar2_blocks_ref(values, &inverse)
}

pub fn rotate_iso4_blocks_ref(values: &[f32], rotations: &[Iso4Rotation]) -> Result<Vec<f32>> {
    validate_ref_blocks(values, 4, rotations.len(), "rotorquant.iso4")?;
    for &rotation in rotations {
        Iso4Rotation::new(rotation.left, rotation.right)?;
    }
    let mut out = Vec::with_capacity(values.len());
    for (xyzw, r) in values.chunks_exact(4).zip(rotations) {
        let q = [xyzw[0], xyzw[1], xyzw[2], xyzw[3]];
        out.extend_from_slice(&quat_mul(quat_mul(r.left, q), r.right));
    }
    Ok(out)
}

pub fn inverse_rotate_iso4_blocks_ref(
    values: &[f32],
    rotations: &[Iso4Rotation],
) -> Result<Vec<f32>> {
    let inverse: Vec<_> = rotations
        .iter()
        .copied()
        .map(Iso4Rotation::inverse)
        .collect();
    rotate_iso4_blocks_ref(values, &inverse)
}

fn validate_chunk_dim(chunk_dim: usize) -> Result<()> {
    if chunk_dim == 0 {
        return Err(invalid("rotorquant.chunk_dim", "expected > 0"));
    }
    Ok(())
}

fn validate_nonzero(field: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(invalid(field, "expected > 0"));
    }
    Ok(())
}

fn validate_residual_bits(residual_bits: Option<u8>) -> Result<()> {
    match residual_bits {
        Some(0) => Err(invalid("rotorquant.residual_bits", "use None to disable")),
        Some(bits) if bits > 8 => Err(invalid("rotorquant.residual_bits", "expected <= 8")),
        _ => Ok(()),
    }
}

fn validate_ref_count(value_count: usize) -> Result<()> {
    if value_count > REFERENCE_MAX_VALUES {
        return Err(invalid(
            "rotorquant.reference",
            format!("expected <= {REFERENCE_MAX_VALUES} values"),
        ));
    }
    Ok(())
}

fn validate_ref_blocks(
    values: &[f32],
    block_dim: usize,
    rotations: usize,
    field: &'static str,
) -> Result<()> {
    validate_ref_count(values.len())?;
    if values.len() % block_dim != 0 {
        return Err(invalid(
            field,
            format!("value count must be multiple of {block_dim}"),
        ));
    }
    let blocks = values.len() / block_dim;
    if rotations != blocks {
        return Err(invalid(
            field,
            format!("expected {blocks} rotations, got {rotations}"),
        ));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(field, "values must be finite"));
    }
    Ok(())
}

fn checked_product(values: &[usize], field: &'static str) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| checked_mul(acc, value, field))
}

fn checked_mul(a: usize, b: usize, field: &'static str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid(field, "byte size overflow"))
}

fn checked_add(a: usize, b: usize, field: &'static str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid(field, "byte size overflow"))
}

fn validate_unit_quat(field: &'static str, q: [f32; 4]) -> Result<()> {
    if q.iter().any(|v| !v.is_finite()) {
        return Err(invalid(field, "quaternion must be finite"));
    }
    let norm = q.iter().map(|v| v * v).sum::<f32>();
    if (norm - 1.0).abs() > 1e-3 {
        return Err(invalid(field, "quaternion must be unit length"));
    }
    Ok(())
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn invalid(field: &'static str, reason: impl Into<String>) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.into(),
        },
        field,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_modes() -> Result<()> {
        assert_eq!(
            "rotor_cl3".parse::<RotorQuantMode>()?,
            RotorQuantMode::RotorCl3
        );
        assert_eq!("rotor".parse::<RotorQuantMode>()?, RotorQuantMode::RotorCl3);
        assert_eq!(
            "planar2".parse::<RotorQuantMode>()?,
            RotorQuantMode::Planar2
        );
        assert_eq!("iso4".parse::<RotorQuantMode>()?, RotorQuantMode::Iso4);
        assert!("off".parse::<RotorQuantMode>().is_err());
        Ok(())
    }

    #[test]
    fn sizes_packed_values() -> Result<()> {
        assert_eq!(packed_bytes_for_values(0, 2)?, 0);
        assert_eq!(packed_bytes_for_values(4, 2)?, 1);
        assert_eq!(packed_bytes_for_values(5, 2)?, 2);
        assert_eq!(packed_bytes_for_values(3, 3)?, 2);
        assert_eq!(packed_bytes_for_values(128, 4)?, 64);

        let meta =
            RotorQuantMetadata::with_default_codebook(RotorQuantMode::Iso4, 4, 128, Some(1))?;
        assert_eq!(meta.packed_chunk_bytes()?, 64);
        assert_eq!(meta.residual_bytes(128)?, 16);
        assert_eq!(meta.codebook.bytes()?, 64);
        Ok(())
    }

    #[test]
    fn kv_layout_summary_shapes_and_fallback_rule() -> Result<()> {
        let meta = RotorQuantMetadata::v1_kv(RotorQuantMode::Iso4, 4)?;
        assert_eq!(meta.residual_bits, ROTORQUANT_V1_RESIDUAL_BITS);

        let layout = RotorQuantKvLayoutSummary::new(meta, 2, 3, 4, 5, 256)?;
        assert_eq!(layout.chunks_per_head, 2);
        assert_eq!(layout.values_shape(), [2, 2, 3, 4, 5, 2, 64]);
        assert_eq!(layout.residual_shape(), None);
        assert_eq!(layout.packed_values_bytes, 30_720);
        assert_eq!(layout.residual_values_bytes, 0);
        assert_eq!(layout.rot_param_bytes, 1_280);
        assert_eq!(layout.codebook_bytes, 64);
        assert_eq!(layout.cache_payload_bytes()?, 30_720);

        assert_eq!(
            select_kv_cache_path(false, true),
            RotorQuantKvPath::RotorQuantDecode
        );
        assert_eq!(
            select_kv_cache_path(true, true),
            RotorQuantKvPath::FallbackFp8OrF16
        );
        assert_eq!(
            select_v1_kv_cache_path(false),
            RotorQuantKvPath::FallbackFp8OrF16
        );
        Ok(())
    }

    #[test]
    fn residual_bits_are_explicit_non_v1_metadata() -> Result<()> {
        let meta =
            RotorQuantMetadata::with_default_codebook(RotorQuantMode::Planar2, 4, 128, Some(1))?;
        let layout = RotorQuantKvLayoutSummary::new(meta, 1, 2, 8, 3, 128)?;
        assert_eq!(layout.values_shape(), [1, 2, 2, 8, 3, 1, 64]);
        assert_eq!(layout.residual_shape(), Some([1, 2, 2, 8, 3, 1, 16]));
        assert_eq!(layout.residual_values_bytes, 1_536);
        Ok(())
    }

    #[test]
    fn rejects_bad_metadata() {
        assert!(
            RotorQuantMetadata::with_default_codebook(RotorQuantMode::Planar2, 1, 128, None,)
                .is_err()
        );
        assert!(
            RotorQuantMetadata::with_default_codebook(RotorQuantMode::Planar2, 4, 0, None,)
                .is_err()
        );
        assert!(RotorQuantMetadata::with_default_codebook(
            RotorQuantMode::Planar2,
            4,
            128,
            Some(0),
        )
        .is_err());
    }

    #[test]
    fn dequantizes_packed_codebook_indices() -> Result<()> {
        let packed = [0xe4u8];
        let codebook = [-1.0, -0.25, 0.25, 1.0];
        assert_eq!(
            dequantize_codebook_ref(&packed, 4, 2, &codebook)?,
            vec![-1.0, -0.25, 0.25, 1.0]
        );
        assert_eq!(pack_indices_ref(&[0, 1, 2, 3], 2)?, packed);
        Ok(())
    }

    #[test]
    fn packed_dequant_planar2_parity_harness() -> Result<()> {
        let indices = [0, 1, 2, 3, 3, 2, 1, 0];
        let codebook = [-1.0, -0.25, 0.25, 1.0];
        let packed = pack_indices_ref(&indices, 2)?;
        assert_eq!(
            unpack_indices_ref(&packed, indices.len(), 2)?,
            indices.to_vec()
        );

        let dequant = dequantize_codebook_ref(&packed, indices.len(), 2, &codebook)?;
        let rotations = [
            Planar2Rotation::from_angle(0.125)?,
            Planar2Rotation::from_angle(-0.375)?,
            Planar2Rotation::from_angle(0.5)?,
            Planar2Rotation::from_angle(-0.25)?,
        ];
        let rotated = rotate_planar2_blocks_ref(&dequant, &rotations)?;
        let back = inverse_rotate_planar2_blocks_ref(&rotated, &rotations)?;
        assert_close(&dequant, &back, 1e-5);
        Ok(())
    }

    #[test]
    fn packed_dequant_iso4_parity_harness() -> Result<()> {
        let indices = [0, 1, 2, 3, 4, 5, 6, 7];
        let codebook = [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0];
        let packed = pack_indices_ref(&indices, 3)?;
        assert_eq!(
            unpack_indices_ref(&packed, indices.len(), 3)?,
            indices.to_vec()
        );

        let dequant = dequantize_codebook_ref(&packed, indices.len(), 3, &codebook)?;
        let rotations = [
            Iso4Rotation::from_angles(0.25, -0.125)?,
            Iso4Rotation::from_angles(-0.5, 0.375)?,
        ];
        let rotated = rotate_iso4_blocks_ref(&dequant, &rotations)?;
        let back = inverse_rotate_iso4_blocks_ref(&rotated, &rotations)?;
        assert_close(&dequant, &back, 1e-5);
        Ok(())
    }

    #[test]
    fn planar2_rotation_round_trips() -> Result<()> {
        let values = [1.0, 2.0, -3.0, 4.0];
        let rotations = [
            Planar2Rotation::from_angle(0.25)?,
            Planar2Rotation::from_angle(-0.75)?,
        ];
        let rotated = rotate_planar2_blocks_ref(&values, &rotations)?;
        let back = inverse_rotate_planar2_blocks_ref(&rotated, &rotations)?;
        assert_close(&values, &back, 1e-5);
        Ok(())
    }

    #[test]
    fn iso4_rotation_round_trips() -> Result<()> {
        let values = [1.0, 2.0, -3.0, 4.0, -0.5, 1.25, 0.75, -2.0];
        let rotations = [
            Iso4Rotation::from_angles(0.2, -0.4)?,
            Iso4Rotation::from_angles(-0.7, 0.1)?,
        ];
        let rotated = rotate_iso4_blocks_ref(&values, &rotations)?;
        let back = inverse_rotate_iso4_blocks_ref(&rotated, &rotations)?;
        assert_close(&values, &back, 1e-5);
        Ok(())
    }

    fn assert_close(expected: &[f32], got: &[f32], tol: f32) {
        assert_eq!(expected.len(), got.len());
        for (a, b) in expected.iter().zip(got) {
            assert!((a - b).abs() <= tol, "expected {expected:?}, got {got:?}");
        }
    }
}
