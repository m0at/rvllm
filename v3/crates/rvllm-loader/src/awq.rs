//! AWQ metadata detection without touching tensor payloads.

use std::collections::BTreeMap;
use std::fmt;

pub type AwqRefResult<T> = std::result::Result<T, AwqReferenceError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AwqReferenceError {
    MissingBits,
    MissingGroupSize,
    UnsupportedBits {
        bits: u8,
    },
    InvalidGroupSize {
        group_size: usize,
    },
    MissingQZeros,
    UnexpectedQZeros,
    TensorLen {
        tensor: &'static str,
        expected: usize,
        got: usize,
    },
    NonFinite {
        tensor: &'static str,
        index: usize,
    },
    InvalidClipRatio {
        index: usize,
    },
}

impl fmt::Display for AwqReferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AwqReferenceError::MissingBits => write!(f, "AWQ bits metadata is missing"),
            AwqReferenceError::MissingGroupSize => {
                write!(f, "AWQ group_size metadata is missing")
            }
            AwqReferenceError::UnsupportedBits { bits } => {
                write!(
                    f,
                    "unsupported AWQ bits={bits}; only 4-bit AWQ is supported"
                )
            }
            AwqReferenceError::InvalidGroupSize { group_size } => {
                write!(f, "invalid AWQ group_size={group_size}")
            }
            AwqReferenceError::MissingQZeros => {
                write!(f, "AWQ zero_point=true requires qzeros")
            }
            AwqReferenceError::UnexpectedQZeros => {
                write!(f, "AWQ zero_point=false must not include qzeros")
            }
            AwqReferenceError::TensorLen {
                tensor,
                expected,
                got,
            } => write!(f, "{tensor} length {got} != expected {expected}"),
            AwqReferenceError::NonFinite { tensor, index } => {
                write!(f, "{tensor}[{index}] is not finite")
            }
            AwqReferenceError::InvalidClipRatio { index } => {
                write!(f, "AWQ clip_ratios[{index}] must be finite and > 0")
            }
        }
    }
}

impl std::error::Error for AwqReferenceError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AwqDequantConfig {
    pub bits: u8,
    pub group_size: usize,
    pub zero_point: bool,
}

impl AwqDequantConfig {
    pub fn new(bits: u8, group_size: usize, zero_point: bool) -> AwqRefResult<Self> {
        validate_bits(bits)?;
        if group_size == 0 {
            return Err(AwqReferenceError::InvalidGroupSize { group_size });
        }
        Ok(Self {
            bits,
            group_size,
            zero_point,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AwqFormat {
    #[default]
    None,
    QWeightQZerosScales,
    PackedWeightScale,
    Mixed,
    ConfigOnly,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LayerAwqNames {
    pub base: String,
    pub qweight: Option<String>,
    pub qzeros: Option<String>,
    pub scales: Option<String>,
    pub g_idx: Option<String>,
    pub packed_weight: Option<String>,
    pub packed_scale: Option<String>,
    pub packed_zero_point: Option<String>,
}

impl LayerAwqNames {
    fn new(base: &str) -> Self {
        Self {
            base: base.to_string(),
            ..Self::default()
        }
    }

    pub fn has_qweight_convention(&self) -> bool {
        self.qweight.is_some() || self.qzeros.is_some() || self.g_idx.is_some()
    }

    pub fn has_packed_weight_scale_convention(&self) -> bool {
        self.packed_weight.is_some()
            || self.packed_scale.is_some()
            || self.packed_zero_point.is_some()
    }

    pub fn has_zero_point(&self) -> bool {
        self.qzeros.is_some() || self.packed_zero_point.is_some()
    }

    pub fn is_ready(&self, zero_point: Option<bool>) -> bool {
        self.is_qweight_ready(zero_point) || self.is_packed_weight_scale_ready(zero_point)
    }

    pub fn is_qweight_ready(&self, zero_point: Option<bool>) -> bool {
        self.qweight.is_some()
            && self.scales.is_some()
            && (self.qzeros.is_some() || zero_point == Some(false))
    }

    pub fn is_packed_weight_scale_ready(&self, zero_point: Option<bool>) -> bool {
        self.packed_weight.is_some()
            && self.packed_scale.is_some()
            && (zero_point != Some(true) || self.packed_zero_point.is_some())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AwqTensorSet {
    pub format: AwqFormat,
    pub layers: Vec<LayerAwqNames>,
    pub group_size: Option<usize>,
    pub bits: Option<u8>,
    pub zero_point: Option<bool>,
    pub zero_point_present: bool,
    pub g_idx_present: bool,
    pub awq_config_present: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AwqW4A8CandidateStatus {
    Ready,
    MetadataOnly,
    MissingTensors,
    UnsupportedFormat,
    InvalidConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AwqW4A8Candidate {
    pub status: AwqW4A8CandidateStatus,
    pub format: AwqFormat,
    pub layer_count: usize,
    pub ready_layer_count: usize,
    pub config: Option<AwqDequantConfig>,
    pub reason: Option<String>,
}

impl AwqW4A8Candidate {
    pub fn is_ready(&self) -> bool {
        matches!(self.status, AwqW4A8CandidateStatus::Ready)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AwqProtectedLane {
    pub group: usize,
    pub row: usize,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AwqQuantizedRef {
    pub qweight: Vec<u32>,
    pub qzeros: Option<Vec<u32>>,
    pub scales: Vec<f32>,
    pub clip_ratios: Vec<f32>,
    pub protected_lanes: Vec<AwqProtectedLane>,
}

impl AwqTensorSet {
    pub fn inspect<I, S>(tensor_names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::inspect_with_config(tensor_names, None)
    }

    pub fn inspect_with_config<I, S>(tensor_names: I, config: Option<&serde_json::Value>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut set = inspect_tensor_names(tensor_names);
        if let Some(config) = config {
            let meta = AwqConfigMeta::from_value(config);
            set.apply_config(meta);
        }
        set.refresh();
        set
    }

    pub fn inspect_with_config_str<I, S>(tensor_names: I, config: Option<&str>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut set = inspect_tensor_names(tensor_names);
        if let Some(config) = config {
            let meta = AwqConfigMeta::from_str(config);
            set.apply_config(meta);
        }
        set.refresh();
        set
    }

    pub fn ready_layer_count(&self) -> usize {
        let zero_point = Some(self.inferred_zero_point());
        self.layers
            .iter()
            .filter(|layer| layer.is_ready(zero_point))
            .count()
    }

    pub fn is_ready(&self) -> bool {
        !self.layers.is_empty()
            && self.ready_layer_count() == self.layers.len()
            && !matches!(self.format, AwqFormat::None | AwqFormat::ConfigOnly)
            && self.dequant_config().is_ok()
    }

    pub fn dequant_config(&self) -> AwqRefResult<AwqDequantConfig> {
        let bits = self.bits.ok_or(AwqReferenceError::MissingBits)?;
        let group_size = self.group_size.ok_or(AwqReferenceError::MissingGroupSize)?;
        let zero_point = self.zero_point.unwrap_or(self.zero_point_present);
        if self.zero_point == Some(true) && !self.zero_point_present {
            return Err(AwqReferenceError::MissingQZeros);
        }
        if self.zero_point == Some(false) && self.zero_point_present {
            return Err(AwqReferenceError::UnexpectedQZeros);
        }
        AwqDequantConfig::new(bits, group_size, zero_point)
    }

    pub fn w4a8_candidate(&self) -> AwqW4A8Candidate {
        let layer_count = self.layers.len();
        let ready_layer_count = self.ready_layer_count();
        let config = self.dequant_config();

        match self.format {
            AwqFormat::None => AwqW4A8Candidate {
                status: AwqW4A8CandidateStatus::MissingTensors,
                format: self.format,
                layer_count,
                ready_layer_count,
                config: config.ok(),
                reason: Some("no AWQ tensor naming convention detected".into()),
            },
            AwqFormat::ConfigOnly => AwqW4A8Candidate {
                status: AwqW4A8CandidateStatus::MetadataOnly,
                format: self.format,
                layer_count,
                ready_layer_count,
                config: config.ok(),
                reason: Some(
                    "AWQ config present but no quantized weight tensors were found".into(),
                ),
            },
            AwqFormat::Mixed => AwqW4A8Candidate {
                status: AwqW4A8CandidateStatus::UnsupportedFormat,
                format: self.format,
                layer_count,
                ready_layer_count,
                config: config.ok(),
                reason: Some(
                    "mixed AWQ tensor layouts need an explicit per-layer dispatch plan".into(),
                ),
            },
            AwqFormat::QWeightQZerosScales | AwqFormat::PackedWeightScale => match config {
                Ok(config) if layer_count > 0 && ready_layer_count == layer_count => {
                    AwqW4A8Candidate {
                        status: AwqW4A8CandidateStatus::Ready,
                        format: self.format,
                        layer_count,
                        ready_layer_count,
                        config: Some(config),
                        reason: None,
                    }
                }
                Ok(config) => AwqW4A8Candidate {
                    status: AwqW4A8CandidateStatus::MissingTensors,
                    format: self.format,
                    layer_count,
                    ready_layer_count,
                    config: Some(config),
                    reason: Some(format!(
                        "only {ready_layer_count}/{layer_count} AWQ layers have complete tensors"
                    )),
                },
                Err(err) => AwqW4A8Candidate {
                    status: AwqW4A8CandidateStatus::InvalidConfig,
                    format: self.format,
                    layer_count,
                    ready_layer_count,
                    config: None,
                    reason: Some(err.to_string()),
                },
            },
        }
    }

    fn inferred_zero_point(&self) -> bool {
        self.zero_point.unwrap_or(self.zero_point_present)
    }

    fn apply_config(&mut self, meta: AwqConfigMeta) {
        self.group_size = meta.group_size.or(self.group_size);
        self.bits = meta.bits.or(self.bits);
        self.zero_point = meta.zero_point.or(self.zero_point);
        self.awq_config_present |= meta.awq;
    }

    fn refresh(&mut self) {
        self.zero_point_present = self.layers.iter().any(LayerAwqNames::has_zero_point);
        self.g_idx_present = self.layers.iter().any(|layer| layer.g_idx.is_some());

        let has_qweight = self
            .layers
            .iter()
            .any(LayerAwqNames::has_qweight_convention);
        let has_packed = self
            .layers
            .iter()
            .any(LayerAwqNames::has_packed_weight_scale_convention);

        self.format = match (has_qweight, has_packed, self.awq_config_present) {
            (true, true, _) => AwqFormat::Mixed,
            (true, false, _) => AwqFormat::QWeightQZerosScales,
            (false, true, _) => AwqFormat::PackedWeightScale,
            (false, false, true) => AwqFormat::ConfigOnly,
            (false, false, false) => AwqFormat::None,
        };
    }
}

pub fn unpack_awq_qweight_ref(
    qweight: &[u32],
    rows: usize,
    cols: usize,
    bits: u8,
) -> AwqRefResult<Vec<u8>> {
    validate_bits(bits)?;
    let pack = values_per_word(bits);
    let expected = ceil_div(rows, pack) * cols;
    validate_len("qweight", qweight.len(), expected)?;

    let mut out = vec![0; rows * cols];
    for row in 0..rows {
        let packed_row = row / pack;
        let lane = row % pack;
        for col in 0..cols {
            let word = qweight[packed_row * cols + col];
            out[row * cols + col] = extract_awq_lane(word, lane, bits) as u8;
        }
    }
    Ok(out)
}

pub fn unpack_awq_qzeros_ref(
    qzeros: &[u32],
    groups: usize,
    cols: usize,
    bits: u8,
) -> AwqRefResult<Vec<u16>> {
    validate_bits(bits)?;
    let pack = values_per_word(bits);
    let packed_cols = ceil_div(cols, pack);
    let expected = groups * packed_cols;
    validate_len("qzeros", qzeros.len(), expected)?;

    let mut out = vec![0; groups * cols];
    for group in 0..groups {
        for col in 0..cols {
            let word = qzeros[group * packed_cols + col / pack];
            out[group * cols + col] = extract_awq_lane(word, col % pack, bits) as u16 + 1;
        }
    }
    Ok(out)
}

pub fn dequantize_awq_qweight_ref(
    qweight: &[u32],
    qzeros: Option<&[u32]>,
    scales: &[f32],
    rows: usize,
    cols: usize,
    config: AwqDequantConfig,
) -> AwqRefResult<Vec<f32>> {
    let config = AwqDequantConfig::new(config.bits, config.group_size, config.zero_point)?;
    let groups = ceil_div(rows, config.group_size);
    validate_len("scales", scales.len(), groups * cols)?;
    if config.zero_point && qzeros.is_none() {
        return Err(AwqReferenceError::MissingQZeros);
    }
    if !config.zero_point && qzeros.is_some() {
        return Err(AwqReferenceError::UnexpectedQZeros);
    }

    let weights = unpack_awq_qweight_ref(qweight, rows, cols, config.bits)?;
    let zeros = if config.zero_point {
        Some(unpack_awq_qzeros_ref(
            qzeros.unwrap_or(&[]),
            groups,
            cols,
            config.bits,
        )?)
    } else {
        None
    };

    let mut out = vec![0.0; rows * cols];
    for row in 0..rows {
        let group = row / config.group_size;
        for col in 0..cols {
            let idx = row * cols + col;
            let scale = scales[group * cols + col];
            let value = weights[idx] as f32;
            let value = if let Some(zeros) = zeros.as_ref() {
                value - zeros[group * cols + col] as f32
            } else {
                value
            };
            out[idx] = value * scale;
        }
    }
    Ok(out)
}

pub fn quantize_awq_groups_ref(
    weights: &[f32],
    activation_importance: &[f32],
    rows: usize,
    cols: usize,
    config: AwqDequantConfig,
    clip_ratios: &[f32],
    protected_per_group: usize,
) -> AwqRefResult<AwqQuantizedRef> {
    let config = AwqDequantConfig::new(config.bits, config.group_size, config.zero_point)?;
    validate_len("weights", weights.len(), rows * cols)?;
    validate_len("activation_importance", activation_importance.len(), rows)?;
    validate_finite("weights", weights)?;
    validate_finite("activation_importance", activation_importance)?;
    validate_clip_ratios(clip_ratios)?;

    let groups = ceil_div(rows, config.group_size);
    let mut qvalues = vec![0u8; rows * cols];
    let mut scales = vec![0.0f32; groups * cols];
    let mut zeros = if config.zero_point {
        Some(vec![1u16; groups * cols])
    } else {
        None
    };
    let mut chosen_clips = vec![1.0f32; groups * cols];

    for group in 0..groups {
        let start = group * config.group_size;
        let end = rows.min(start + config.group_size);
        for col in 0..cols {
            let best = search_awq_group_col(
                weights,
                activation_importance,
                start,
                end,
                cols,
                col,
                config,
                clip_ratios,
            );
            scales[group * cols + col] = best.scale;
            chosen_clips[group * cols + col] = best.clip_ratio;
            if let Some(zeros) = zeros.as_mut() {
                zeros[group * cols + col] = best.zero;
            }
            for row in start..end {
                qvalues[row * cols + col] =
                    quantize_awq_value(weights[row * cols + col], best, config.bits) as u8;
            }
        }
    }

    let qweight = pack_awq_qweight_ref(&qvalues, rows, cols, config.bits);
    let qzeros = zeros.map(|zeros| pack_awq_qzeros_ref(&zeros, groups, cols, config.bits));
    let protected_lanes = select_awq_protected_lanes_ref(
        activation_importance,
        config.group_size,
        protected_per_group,
    )?;

    Ok(AwqQuantizedRef {
        qweight,
        qzeros,
        scales,
        clip_ratios: chosen_clips,
        protected_lanes,
    })
}

pub fn select_awq_protected_lanes_ref(
    activation_importance: &[f32],
    group_size: usize,
    protected_per_group: usize,
) -> AwqRefResult<Vec<AwqProtectedLane>> {
    if group_size == 0 {
        return Err(AwqReferenceError::InvalidGroupSize { group_size });
    }
    validate_finite("activation_importance", activation_importance)?;

    let groups = ceil_div(activation_importance.len(), group_size);
    let mut out = Vec::new();
    for group in 0..groups {
        let start = group * group_size;
        let end = activation_importance.len().min(start + group_size);
        let mut lanes = (start..end)
            .map(|row| AwqProtectedLane {
                group,
                row,
                score: activation_importance[row].abs(),
            })
            .collect::<Vec<_>>();
        lanes.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.row.cmp(&b.row)));
        out.extend(lanes.into_iter().take(protected_per_group.min(end - start)));
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
struct AwqGroupColQuant {
    scale: f32,
    zero: u16,
    clip_ratio: f32,
}

fn search_awq_group_col(
    weights: &[f32],
    activation_importance: &[f32],
    start: usize,
    end: usize,
    cols: usize,
    col: usize,
    config: AwqDequantConfig,
    clip_ratios: &[f32],
) -> AwqGroupColQuant {
    let max_abs = (start..end)
        .map(|row| weights[row * cols + col].abs())
        .fold(0.0f32, f32::max);
    if max_abs == 0.0 {
        return AwqGroupColQuant {
            scale: 1.0,
            zero: 1,
            clip_ratio: clip_ratios.first().copied().unwrap_or(1.0),
        };
    }

    let mut best = AwqGroupColQuant {
        scale: 1.0,
        zero: 1,
        clip_ratio: clip_ratios[0],
    };
    let mut best_loss = f32::INFINITY;
    for &clip_ratio in clip_ratios {
        let candidate =
            make_awq_group_col_quant(weights, start, end, cols, col, config, max_abs, clip_ratio);
        let loss = awq_group_col_loss(
            weights,
            activation_importance,
            start,
            end,
            cols,
            col,
            candidate,
            config.bits,
        );
        if loss < best_loss {
            best_loss = loss;
            best = candidate;
        }
    }
    best
}

fn make_awq_group_col_quant(
    weights: &[f32],
    start: usize,
    end: usize,
    cols: usize,
    col: usize,
    config: AwqDequantConfig,
    max_abs: f32,
    clip_ratio: f32,
) -> AwqGroupColQuant {
    let qmax = ((1u32 << config.bits) - 1) as f32;
    let clip = max_abs * clip_ratio;
    if config.zero_point {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for row in start..end {
            let v = weights[row * cols + col].clamp(-clip, clip);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if lo == hi {
            return AwqGroupColQuant {
                scale: 1.0,
                zero: 1,
                clip_ratio,
            };
        }
        let scale = ((hi - lo) / qmax).max(f32::MIN_POSITIVE);
        let zero = (-lo / scale).round().clamp(1.0, qmax + 1.0) as u16;
        AwqGroupColQuant {
            scale,
            zero,
            clip_ratio,
        }
    } else {
        let scale = (clip / qmax).max(f32::MIN_POSITIVE);
        AwqGroupColQuant {
            scale,
            zero: 0,
            clip_ratio,
        }
    }
}

fn awq_group_col_loss(
    weights: &[f32],
    activation_importance: &[f32],
    start: usize,
    end: usize,
    cols: usize,
    col: usize,
    quant: AwqGroupColQuant,
    bits: u8,
) -> f32 {
    let mut loss = 0.0;
    for row in start..end {
        let w = weights[row * cols + col];
        let q = quantize_awq_value(w, quant, bits) as f32;
        let dequant = (q - quant.zero as f32) * quant.scale;
        let err = dequant - w;
        loss += activation_importance[row].abs() * err * err;
    }
    loss
}

fn quantize_awq_value(value: f32, quant: AwqGroupColQuant, bits: u8) -> u32 {
    let qmax = ((1u32 << bits) - 1) as f32;
    (value / quant.scale + quant.zero as f32)
        .round()
        .clamp(0.0, qmax) as u32
}

fn pack_awq_qweight_ref(values: &[u8], rows: usize, cols: usize, bits: u8) -> Vec<u32> {
    let pack = values_per_word(bits);
    let mut out = vec![0; ceil_div(rows, pack) * cols];
    for row in 0..rows {
        for col in 0..cols {
            let idx = (row / pack) * cols + col;
            out[idx] |= (values[row * cols + col] as u32) << ((row % pack) * bits as usize);
        }
    }
    out
}

fn pack_awq_qzeros_ref(zeros: &[u16], groups: usize, cols: usize, bits: u8) -> Vec<u32> {
    let pack = values_per_word(bits);
    let packed_cols = ceil_div(cols, pack);
    let mut out = vec![0; groups * packed_cols];
    for group in 0..groups {
        for col in 0..cols {
            let zero = zeros[group * cols + col].clamp(1, 1u16 << bits);
            let idx = group * packed_cols + col / pack;
            out[idx] |= ((zero - 1) as u32) << ((col % pack) * bits as usize);
        }
    }
    out
}

fn validate_bits(bits: u8) -> AwqRefResult<()> {
    if bits == 4 {
        Ok(())
    } else {
        Err(AwqReferenceError::UnsupportedBits { bits })
    }
}

fn values_per_word(bits: u8) -> usize {
    32 / bits as usize
}

fn extract_awq_lane(word: u32, lane: usize, bits: u8) -> u32 {
    let mask = (1u32 << bits) - 1;
    (word >> (lane * bits as usize)) & mask
}

fn validate_len(tensor: &'static str, got: usize, expected: usize) -> AwqRefResult<()> {
    if got == expected {
        Ok(())
    } else {
        Err(AwqReferenceError::TensorLen {
            tensor,
            expected,
            got,
        })
    }
}

fn validate_finite(tensor: &'static str, values: &[f32]) -> AwqRefResult<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(AwqReferenceError::NonFinite { tensor, index });
        }
    }
    Ok(())
}

fn validate_clip_ratios(clip_ratios: &[f32]) -> AwqRefResult<()> {
    if clip_ratios.is_empty() {
        return Err(AwqReferenceError::InvalidClipRatio { index: 0 });
    }
    for (index, ratio) in clip_ratios.iter().enumerate() {
        if !ratio.is_finite() || *ratio <= 0.0 {
            return Err(AwqReferenceError::InvalidClipRatio { index });
        }
    }
    Ok(())
}

fn ceil_div(n: usize, d: usize) -> usize {
    if n == 0 {
        0
    } else {
        (n - 1) / d + 1
    }
}

fn inspect_tensor_names<I, S>(tensor_names: I) -> AwqTensorSet
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut layers = BTreeMap::<String, LayerAwqNames>::new();
    let mut plain_weights = BTreeMap::<String, String>::new();

    for name in tensor_names {
        let name = name.as_ref();
        if let Some(base) = strip_suffix(name, ".qweight") {
            set_slot(&mut layer(&mut layers, base).qweight, name);
        } else if let Some(base) = strip_suffix(name, ".qzeros") {
            set_slot(&mut layer(&mut layers, base).qzeros, name);
        } else if let Some(base) = strip_suffix(name, ".scales") {
            set_slot(&mut layer(&mut layers, base).scales, name);
        } else if let Some(base) = strip_suffix(name, ".g_idx") {
            set_slot(&mut layer(&mut layers, base).g_idx, name);
        } else if let Some(base) = strip_any_suffix(
            name,
            &[
                ".weight_scale",
                ".weight_scales",
                ".weight.scale",
                ".weight.scales",
                ".scale",
            ],
        ) {
            set_slot(&mut layer(&mut layers, base).packed_scale, name);
        } else if let Some(base) = strip_any_suffix(
            name,
            &[
                ".weight_zero_point",
                ".weight_zeros",
                ".weight_zero",
                ".zero_point",
                ".zeros",
            ],
        ) {
            set_slot(&mut layer(&mut layers, base).packed_zero_point, name);
        } else if let Some(base) = strip_any_suffix(
            name,
            &[
                ".weight_packed",
                ".packed_weight",
                ".qweight_packed",
                ".weight.pack",
                ".weight_q",
            ],
        ) {
            set_slot(&mut layer(&mut layers, base).packed_weight, name);
        } else if let Some(base) = strip_suffix(name, ".weight") {
            plain_weights
                .entry(base.to_string())
                .or_insert_with(|| name.to_string());
        }
    }

    for (base, weight) in plain_weights {
        let Some(layer) = layers.get_mut(&base) else {
            continue;
        };
        let qweight_style = layer.has_qweight_convention();
        if !qweight_style {
            if layer.packed_scale.is_none() {
                layer.packed_scale = layer.scales.clone();
            }
            if layer.packed_zero_point.is_none() {
                layer.packed_zero_point = layer.qzeros.clone();
            }
        }
        if layer.packed_scale.is_some() || layer.packed_zero_point.is_some() {
            set_slot(&mut layer.packed_weight, &weight);
        }
    }

    let layers = layers
        .into_values()
        .filter(|layer| {
            layer.has_qweight_convention() || layer.has_packed_weight_scale_convention()
        })
        .collect();
    let mut set = AwqTensorSet {
        layers,
        ..AwqTensorSet::default()
    };
    set.refresh();
    set
}

fn layer<'a>(layers: &'a mut BTreeMap<String, LayerAwqNames>, base: &str) -> &'a mut LayerAwqNames {
    layers
        .entry(base.to_string())
        .or_insert_with(|| LayerAwqNames::new(base))
}

fn set_slot(slot: &mut Option<String>, name: &str) {
    if slot.is_none() {
        *slot = Some(name.to_string());
    }
}

fn strip_suffix<'a>(name: &'a str, suffix: &str) -> Option<&'a str> {
    let base = name.strip_suffix(suffix)?;
    if base.is_empty() {
        None
    } else {
        Some(base)
    }
}

fn strip_any_suffix<'a>(name: &'a str, suffixes: &[&str]) -> Option<&'a str> {
    suffixes
        .iter()
        .find_map(|suffix| strip_suffix(name, suffix))
}

#[derive(Clone, Debug, Default)]
struct AwqConfigMeta {
    awq: bool,
    group_size: Option<usize>,
    bits: Option<u8>,
    zero_point: Option<bool>,
}

impl AwqConfigMeta {
    fn from_str(s: &str) -> Self {
        let mut meta = Self::default();
        meta.apply_text(s);
        meta
    }

    fn from_value(value: &serde_json::Value) -> Self {
        let mut meta = Self::default();
        meta.apply_value(value);
        meta
    }

    fn apply_value(&mut self, value: &serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, value) in map {
                    self.apply_pair(key, value);
                    self.apply_value(value);
                }
            }
            serde_json::Value::Array(values) => {
                for value in values {
                    self.apply_value(value);
                }
            }
            serde_json::Value::String(s) => self.apply_text(s),
            _ => {}
        }
    }

    fn apply_pair(&mut self, key: &str, value: &serde_json::Value) {
        let key = normalize_key(key);
        if key.contains("awq") {
            self.awq = true;
        }

        match key.as_str() {
            "quantmethod" | "quantizationmethod" => {
                if value_string_contains(value, "awq") {
                    self.awq = true;
                }
            }
            "format" | "version" => {
                if value_string_contains(value, "awq") {
                    self.awq = true;
                }
            }
            "bits" | "wbit" | "weightbits" | "numbits" => {
                self.bits = value_u64(value)
                    .and_then(|n| u8::try_from(n).ok())
                    .or(self.bits);
            }
            "groupsize" | "qgroupsize" => {
                self.group_size = value_u64(value)
                    .and_then(|n| usize::try_from(n).ok())
                    .or(self.group_size);
            }
            "zeropoint" | "zp" => {
                self.zero_point = value_bool(value).or(self.zero_point);
            }
            "symmetric" => {
                if let Some(symmetric) = value_bool(value) {
                    self.zero_point = Some(!symmetric);
                }
            }
            _ => {}
        }
    }

    fn apply_text(&mut self, s: &str) {
        let text = s.trim();
        if text.is_empty() {
            return;
        }
        if text.to_ascii_lowercase().contains("awq") {
            self.awq = true;
        }
        if text.starts_with('{') || text.starts_with('[') {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
                self.apply_value(&value);
            }
        }
    }
}

fn normalize_key(key: &str) -> String {
    key.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

fn value_string_contains(value: &serde_json::Value, needle: &str) -> bool {
    value
        .as_str()
        .map(|s| s.to_ascii_lowercase().contains(needle))
        .unwrap_or(false)
}

fn value_u64(value: &serde_json::Value) -> Option<u64> {
    if let Some(n) = value.as_u64() {
        return Some(n);
    }
    if let Some(n) = value.as_i64() {
        return u64::try_from(n).ok();
    }
    value.as_str()?.trim().parse().ok()
}

fn value_bool(value: &serde_json::Value) -> Option<bool> {
    if let Some(b) = value.as_bool() {
        return Some(b);
    }
    match value.as_str()?.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_qweight(rows: usize, cols: usize, values: &[u8]) -> Vec<u32> {
        assert_eq!(values.len(), rows * cols);
        let pack = values_per_word(4);
        let mut out = vec![0; ceil_div(rows, pack) * cols];
        for row in 0..rows {
            for col in 0..cols {
                let idx = (row / pack) * cols + col;
                out[idx] |= (values[row * cols + col] as u32) << ((row % pack) * 4);
            }
        }
        out
    }

    fn pack_qzeros(groups: usize, cols: usize, zeros: &[u16]) -> Vec<u32> {
        assert_eq!(zeros.len(), groups * cols);
        let pack = values_per_word(4);
        let mut out = vec![0; groups * ceil_div(cols, pack)];
        for group in 0..groups {
            for col in 0..cols {
                let zero = zeros[group * cols + col];
                assert!((1..=16).contains(&zero));
                let idx = group * ceil_div(cols, pack) + col / pack;
                out[idx] |= ((zero - 1) as u32) << ((col % pack) * 4);
            }
        }
        out
    }

    #[test]
    fn detects_qweight_awq_names_and_config() {
        let names = [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.g_idx",
            "model.layers.0.mlp.up_proj.qweight",
            "model.layers.0.mlp.up_proj.qzeros",
            "model.layers.0.mlp.up_proj.scales",
        ];
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": true,
                "version": "GEMM"
            }
        });

        let set = AwqTensorSet::inspect_with_config(names, Some(&config));

        assert_eq!(set.format, AwqFormat::QWeightQZerosScales);
        assert_eq!(set.bits, Some(4));
        assert_eq!(set.group_size, Some(128));
        assert_eq!(set.zero_point, Some(true));
        assert!(set.zero_point_present);
        assert!(set.g_idx_present);
        assert!(set.is_ready());
        assert_eq!(set.layers.len(), 2);
        assert_eq!(set.ready_layer_count(), 2);
        assert_eq!(
            set.w4a8_candidate(),
            AwqW4A8Candidate {
                status: AwqW4A8CandidateStatus::Ready,
                format: AwqFormat::QWeightQZerosScales,
                layer_count: 2,
                ready_layer_count: 2,
                config: Some(AwqDequantConfig {
                    bits: 4,
                    group_size: 128,
                    zero_point: true,
                }),
                reason: None,
            }
        );
    }

    #[test]
    fn detects_packed_weight_scale_names() {
        let names = [
            "model.layers.1.mlp.down_proj.weight",
            "model.layers.1.mlp.down_proj.weight_scale",
            "model.layers.1.mlp.down_proj.weight_zero_point",
        ];
        let config = r#"{
            "quantization_config": {
                "quant_method": "awq",
                "w_bit": "4",
                "q_group_size": "64",
                "zero_point": "true"
            }
        }"#;

        let set = AwqTensorSet::inspect_with_config_str(names, Some(config));

        assert_eq!(set.format, AwqFormat::PackedWeightScale);
        assert_eq!(set.bits, Some(4));
        assert_eq!(set.group_size, Some(64));
        assert_eq!(set.zero_point, Some(true));
        assert!(set.zero_point_present);
        assert!(set.is_ready());
        assert_eq!(set.layers[0].packed_weight.as_deref(), Some(names[0]));
    }

    #[test]
    fn realistic_awq_checkpoint_fixture_builds_w4a8_candidate() {
        let names = [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.g_idx",
            "model.layers.0.self_attn.k_proj.qweight",
            "model.layers.0.self_attn.k_proj.qzeros",
            "model.layers.0.self_attn.k_proj.scales",
            "model.layers.0.self_attn.v_proj.qweight",
            "model.layers.0.self_attn.v_proj.qzeros",
            "model.layers.0.self_attn.v_proj.scales",
            "model.layers.0.self_attn.o_proj.qweight",
            "model.layers.0.self_attn.o_proj.qzeros",
            "model.layers.0.self_attn.o_proj.scales",
            "model.layers.0.mlp.gate_proj.qweight",
            "model.layers.0.mlp.gate_proj.qzeros",
            "model.layers.0.mlp.gate_proj.scales",
            "model.layers.0.mlp.up_proj.qweight",
            "model.layers.0.mlp.up_proj.qzeros",
            "model.layers.0.mlp.up_proj.scales",
            "model.layers.0.mlp.down_proj.qweight",
            "model.layers.0.mlp.down_proj.qzeros",
            "model.layers.0.mlp.down_proj.scales",
            "model.embed_tokens.weight",
        ];
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": true,
                "version": "GEMM"
            }
        });

        let set = AwqTensorSet::inspect_with_config(names, Some(&config));
        let candidate = set.w4a8_candidate();

        assert!(candidate.is_ready());
        assert_eq!(candidate.layer_count, 7);
        assert_eq!(candidate.ready_layer_count, 7);
        assert_eq!(candidate.config.unwrap().group_size, 128);
    }

    #[test]
    fn w4a8_candidate_rejects_config_only_and_mixed_layouts() {
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": true
            }
        });
        let config_only = AwqTensorSet::inspect_with_config(Vec::<&str>::new(), Some(&config));
        assert_eq!(
            config_only.w4a8_candidate().status,
            AwqW4A8CandidateStatus::MetadataOnly
        );

        let mixed = AwqTensorSet::inspect_with_config(
            [
                "model.layers.0.self_attn.q_proj.qweight",
                "model.layers.0.self_attn.q_proj.qzeros",
                "model.layers.0.self_attn.q_proj.scales",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.self_attn.o_proj.weight_scale",
                "model.layers.0.self_attn.o_proj.weight_zero_point",
            ],
            Some(&config),
        );
        assert_eq!(mixed.format, AwqFormat::Mixed);
        assert_eq!(
            mixed.w4a8_candidate().status,
            AwqW4A8CandidateStatus::UnsupportedFormat
        );
    }

    #[test]
    fn supports_weight_plus_scales_when_zero_point_is_disabled() {
        let names = [
            "model.layers.2.self_attn.o_proj.weight",
            "model.layers.2.self_attn.o_proj.scales",
        ];
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": false
            }
        });

        let set = AwqTensorSet::inspect_with_config(names, Some(&config));

        assert_eq!(set.format, AwqFormat::PackedWeightScale);
        assert!(!set.zero_point_present);
        assert!(set.is_ready());
    }

    #[test]
    fn qweight_missing_qzeros_is_not_ready_when_zero_point_expected() {
        let names = [
            "model.layers.0.self_attn.k_proj.qweight",
            "model.layers.0.self_attn.k_proj.scales",
        ];
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "zero_point": true
            }
        });

        let set = AwqTensorSet::inspect_with_config(names, Some(&config));

        assert_eq!(set.format, AwqFormat::QWeightQZerosScales);
        assert_eq!(set.ready_layer_count(), 0);
        assert!(!set.is_ready());
    }

    #[test]
    fn config_only_is_awq_but_not_ready() {
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128
            }
        });

        let set = AwqTensorSet::inspect_with_config(Vec::<&str>::new(), Some(&config));

        assert_eq!(set.format, AwqFormat::ConfigOnly);
        assert!(set.awq_config_present);
        assert_eq!(set.bits, Some(4));
        assert_eq!(set.group_size, Some(128));
        assert!(!set.is_ready());
    }

    #[test]
    fn ordinary_weight_tensors_are_ignored() {
        let names = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "lm_head.weight",
        ];

        let set = AwqTensorSet::inspect(names);

        assert_eq!(set.format, AwqFormat::None);
        assert!(set.layers.is_empty());
        assert!(!set.is_ready());
    }

    #[test]
    fn compressed_tensors_symmetric_false_implies_zero_points() {
        let names = [
            "model.layers.3.mlp.gate_proj.weight",
            "model.layers.3.mlp.gate_proj.weight_scale",
            "model.layers.3.mlp.gate_proj.weight_zero_point",
        ];
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "group_size": 128,
                            "symmetric": false
                        }
                    }
                }
            },
            "awq": true
        });

        let set = AwqTensorSet::inspect_with_config(names, Some(&config));

        assert_eq!(set.format, AwqFormat::PackedWeightScale);
        assert_eq!(set.bits, Some(4));
        assert_eq!(set.group_size, Some(128));
        assert_eq!(set.zero_point, Some(true));
        assert!(set.is_ready());
    }

    #[test]
    fn missing_zero_point_is_inferred_from_qzeros() {
        let names = [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.scales",
        ];
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128
            }
        });

        let set = AwqTensorSet::inspect_with_config(names, Some(&config));

        assert_eq!(
            set.dequant_config(),
            Ok(AwqDequantConfig {
                bits: 4,
                group_size: 128,
                zero_point: true,
            })
        );
        assert!(set.is_ready());
    }

    #[test]
    fn missing_zero_point_without_qzeros_is_symmetric() {
        let names = [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.scales",
        ];
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128
            }
        });

        let set = AwqTensorSet::inspect_with_config(names, Some(&config));

        assert_eq!(
            set.dequant_config(),
            Ok(AwqDequantConfig {
                bits: 4,
                group_size: 128,
                zero_point: false,
            })
        );
        assert!(set.is_ready());
    }

    #[test]
    fn missing_bits_or_group_size_rejects_readiness() {
        let names = [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.scales",
        ];

        let set = AwqTensorSet::inspect(names);
        assert_eq!(set.dequant_config(), Err(AwqReferenceError::MissingBits));
        assert!(!set.is_ready());

        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4
            }
        });
        let set = AwqTensorSet::inspect_with_config(names, Some(&config));
        assert_eq!(
            set.dequant_config(),
            Err(AwqReferenceError::MissingGroupSize)
        );
        assert!(!set.is_ready());
    }

    #[test]
    fn rejects_unsupported_bits_and_zero_point_mismatch() {
        assert_eq!(
            AwqDequantConfig::new(3, 128, true),
            Err(AwqReferenceError::UnsupportedBits { bits: 3 })
        );

        let names = [
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.scales",
        ];
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": false
            }
        });

        let set = AwqTensorSet::inspect_with_config(names, Some(&config));

        assert_eq!(
            set.dequant_config(),
            Err(AwqReferenceError::UnexpectedQZeros)
        );
        assert!(!set.is_ready());
    }

    #[test]
    fn unpacks_awq_qweight_and_qzeros_lsb_first() {
        let qweight = pack_qweight(3, 2, &[3, 7, 5, 4, 4, 9]);
        let qzeros = pack_qzeros(2, 2, &[1, 2, 4, 1]);

        assert_eq!(
            unpack_awq_qweight_ref(&qweight, 3, 2, 4),
            Ok(vec![3, 7, 5, 4, 4, 9])
        );
        assert_eq!(
            unpack_awq_qzeros_ref(&qzeros, 2, 2, 4),
            Ok(vec![1, 2, 4, 1])
        );
    }

    #[test]
    fn dequantizes_awq_qweight_qzeros_scales() {
        let qweight = pack_qweight(3, 2, &[3, 7, 5, 4, 4, 9]);
        let qzeros = pack_qzeros(2, 2, &[1, 2, 4, 1]);
        let scales = [0.5, 2.0, 1.5, 0.25];
        let config = AwqDequantConfig::new(4, 2, true).unwrap();

        let dequant =
            dequantize_awq_qweight_ref(&qweight, Some(&qzeros), &scales, 3, 2, config).unwrap();

        assert_eq!(dequant, vec![1.0, 10.0, 2.0, 4.0, 0.0, 2.0]);
    }

    #[test]
    fn dequantizes_symmetric_awq_without_qzeros() {
        let qweight = pack_qweight(2, 2, &[2, 3, 4, 5]);
        let scales = [0.5, 2.0];
        let config = AwqDequantConfig::new(4, 2, false).unwrap();

        let dequant = dequantize_awq_qweight_ref(&qweight, None, &scales, 2, 2, config).unwrap();

        assert_eq!(dequant, vec![1.0, 6.0, 2.0, 10.0]);
    }

    #[test]
    fn activation_aware_clip_search_downweights_outliers() {
        let weights = [1.0, 1.1, 1.2, 8.0];
        let activation_importance = [10.0, 10.0, 10.0, 0.0001];
        let config = AwqDequantConfig::new(4, 4, false).unwrap();

        let quant = quantize_awq_groups_ref(
            &weights,
            &activation_importance,
            4,
            1,
            config,
            &[1.0, 0.25],
            0,
        )
        .unwrap();
        let dequant =
            dequantize_awq_qweight_ref(&quant.qweight, None, &quant.scales, 4, 1, config).unwrap();

        assert_eq!(quant.clip_ratios, vec![0.25]);
        assert!((dequant[2] - 1.2).abs() < 1e-6);
        assert!(dequant[3] < 2.1);
    }

    #[test]
    fn zero_point_group_quantization_handles_signed_weights() {
        let weights = [-1.0, 0.0, 1.0, 2.0];
        let activation_importance = [1.0; 4];
        let config = AwqDequantConfig::new(4, 4, true).unwrap();

        let quant =
            quantize_awq_groups_ref(&weights, &activation_importance, 4, 1, config, &[1.0], 0)
                .unwrap();
        let dequant = dequantize_awq_qweight_ref(
            &quant.qweight,
            quant.qzeros.as_deref(),
            &quant.scales,
            4,
            1,
            config,
        )
        .unwrap();

        assert_eq!(quant.scales, vec![0.2]);
        assert!(quant.qzeros.is_some());
        assert_close(&dequant, &weights, 1e-6);
    }

    #[test]
    fn protected_lanes_select_top_activation_rows_per_group() {
        let activation_importance = [0.1, -7.0, 2.0, 9.0, 4.0];

        let lanes = select_awq_protected_lanes_ref(&activation_importance, 2, 1).unwrap();

        assert_eq!(
            lanes,
            vec![
                AwqProtectedLane {
                    group: 0,
                    row: 1,
                    score: 7.0,
                },
                AwqProtectedLane {
                    group: 1,
                    row: 3,
                    score: 9.0,
                },
                AwqProtectedLane {
                    group: 2,
                    row: 4,
                    score: 4.0,
                },
            ]
        );
    }

    #[test]
    fn quantize_ref_returns_protected_lane_metadata() {
        let weights = [1.0, 2.0, 3.0, 4.0];
        let activation_importance = [0.5, 3.0, 2.0, 1.0];
        let config = AwqDequantConfig::new(4, 2, false).unwrap();

        let quant =
            quantize_awq_groups_ref(&weights, &activation_importance, 4, 1, config, &[1.0], 1)
                .unwrap();

        assert_eq!(
            quant.protected_lanes,
            vec![
                AwqProtectedLane {
                    group: 0,
                    row: 1,
                    score: 3.0,
                },
                AwqProtectedLane {
                    group: 1,
                    row: 2,
                    score: 2.0,
                },
            ]
        );
        assert!(quant.qzeros.is_none());
    }

    #[test]
    fn quantize_ref_rejects_bad_calibration_inputs() {
        let config = AwqDequantConfig::new(4, 2, false).unwrap();
        assert_eq!(
            quantize_awq_groups_ref(&[1.0, f32::NAN], &[1.0, 1.0], 2, 1, config, &[1.0], 0)
                .unwrap_err(),
            AwqReferenceError::NonFinite {
                tensor: "weights",
                index: 1,
            }
        );
        assert_eq!(
            quantize_awq_groups_ref(&[1.0, 2.0], &[1.0, 1.0], 2, 1, config, &[], 0).unwrap_err(),
            AwqReferenceError::InvalidClipRatio { index: 0 }
        );
    }

    fn assert_close(got: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(got.len(), expected.len());
        for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
            assert!(
                (got - expected).abs() <= tol,
                "index {index}: got {got}, expected {expected}"
            );
        }
    }
}
