use bitvec::prelude::*;
use std::fmt::Display;
use tracing::{debug, trace};

pub use crate::data_loader::FeatureDataType;
use crate::data_loader::{DataValue, Dataset};
use crate::error::EntroGdError;

const MAX_DECIMAL_SCALE: u8 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatScalingMode {
    Disabled,
    ScaledSignedInt,
    ScaledOffsetSignedInt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreprocessOptions {
    pub float_scaling: FloatScalingMode,
    pub max_decimal_scale: u8,
    pub integer_zero_normalization: bool,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        PreprocessOptions {
            float_scaling: FloatScalingMode::ScaledOffsetSignedInt,
            max_decimal_scale: MAX_DECIMAL_SCALE,
            integer_zero_normalization: true,
        }
    }
}

/// Optional transform metadata used for a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureTransform {
    None,
    ScaledSignedInt { decimal_scale: u8 },
    OffsetSignedInt { min_value: i64 },
    OffsetUnsignedInt { min_value: u64 },
    ScaledOffsetSignedInt { decimal_scale: u8, min_value: i64 },
}

/// Per-feature schema entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSpec {
    pub data_type: FeatureDataType,
    pub bits: usize,
    pub transform: FeatureTransform,
}

impl FeatureSpec {
    pub fn new(data_type: FeatureDataType, bits: usize) -> Self {
        FeatureSpec {
            data_type,
            bits,
            transform: FeatureTransform::None,
        }
    }
}

/// High-level metadata describing a bit-packed dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitDataCompressionInfo {
    pub features: Vec<FeatureSpec>,
    pub original_size_bits: usize,
    pub n_data_samples: usize,
    pub m_condensed_samples: Option<usize>,
    pub m_condensed_sample_weights: Option<Vec<usize>>,
    feature_offsets: Vec<usize>,
    chunk_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageReconstructionInfo {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    /// 0 = sRGB + linear alpha, 1 = all linear
    pub colorspace: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitDataReconstructionInfo {
    Tabular,
    Image(ImageReconstructionInfo),
}

/// High-level metadata describing a bit-packed dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitDataInfo {
    pub compression: BitDataCompressionInfo,
    pub reconstruction: BitDataReconstructionInfo,
}

impl BitDataInfo {
    pub fn new(
        features: Vec<FeatureSpec>,
        original_size_bits: usize,
    ) -> Result<Self, EntroGdError> {
        Self::new_with_reconstruction_info(
            features,
            original_size_bits,
            BitDataReconstructionInfo::Tabular,
        )
    }

    pub fn new_with_reconstruction_info(
        features: Vec<FeatureSpec>,
        original_size_bits: usize,
        reconstruction: BitDataReconstructionInfo,
    ) -> Result<Self, EntroGdError> {
        debug!(
            "Creating BitDataInfo with {} features and original_size_bits={}.",
            features.len(),
            original_size_bits
        );
        if features.is_empty() {
            return Err(EntroGdError::InvalidFeatureSpec {
                message: "features is empty".to_string(),
            });
        }

        let mut offsets = Vec::with_capacity(features.len());
        let mut running = 0usize;
        for (idx, spec) in features.iter().enumerate() {
            trace!(
                "Feature spec {} => type={:?}, bits={}, transform={:?}, offset={}",
                idx, spec.data_type, spec.bits, spec.transform, running
            );
            if spec.bits == 0 {
                return Err(EntroGdError::InvalidFeatureSpec {
                    message: "feature bits must be > 0".to_string(),
                });
            }
            match spec.data_type {
                FeatureDataType::F32 => match spec.transform {
                    FeatureTransform::None if spec.bits != 32 => {
                        return Err(EntroGdError::InvalidFeatureSpec {
                            message: "f32 features without transform must use 32 bits".to_string(),
                        });
                    }
                    FeatureTransform::ScaledSignedInt { .. }
                    | FeatureTransform::ScaledOffsetSignedInt { .. }
                        if spec.bits > 64 =>
                    {
                        return Err(EntroGdError::InvalidFeatureSpec {
                            message: "scaled f32 features must use <= 64 bits".to_string(),
                        });
                    }
                    FeatureTransform::OffsetSignedInt { .. }
                    | FeatureTransform::OffsetUnsignedInt { .. } => {
                        return Err(EntroGdError::InvalidFeatureSpec {
                            message: "f32 features cannot use integer offset transforms"
                                .to_string(),
                        });
                    }
                    _ => {}
                },
                FeatureDataType::F64 => match spec.transform {
                    FeatureTransform::None if spec.bits != 64 => {
                        return Err(EntroGdError::InvalidFeatureSpec {
                            message: "f64 features without transform must use 64 bits".to_string(),
                        });
                    }
                    FeatureTransform::ScaledSignedInt { .. }
                    | FeatureTransform::ScaledOffsetSignedInt { .. }
                        if spec.bits > 64 =>
                    {
                        return Err(EntroGdError::InvalidFeatureSpec {
                            message: "scaled f64 features must use <= 64 bits".to_string(),
                        });
                    }
                    FeatureTransform::OffsetSignedInt { .. }
                    | FeatureTransform::OffsetUnsignedInt { .. } => {
                        return Err(EntroGdError::InvalidFeatureSpec {
                            message: "f64 features cannot use integer offset transforms"
                                .to_string(),
                        });
                    }
                    _ => {}
                },
                FeatureDataType::SignedInt => match spec.transform {
                    FeatureTransform::None | FeatureTransform::OffsetSignedInt { .. } => {}
                    _ => {
                        return Err(EntroGdError::InvalidFeatureSpec {
                            message:
                                "signed integer features can only use signed offset transforms"
                                    .to_string(),
                        });
                    }
                },
                FeatureDataType::UnsignedInt => match spec.transform {
                    FeatureTransform::None | FeatureTransform::OffsetUnsignedInt { .. } => {}
                    _ => {
                        return Err(EntroGdError::InvalidFeatureSpec {
                            message:
                                "unsigned integer features can only use unsigned offset transforms"
                                    .to_string(),
                        });
                    }
                },
            }
            offsets.push(running);
            running += spec.bits;
        }

        let compression = BitDataCompressionInfo {
            features,
            original_size_bits,
            n_data_samples: original_size_bits / running,
            m_condensed_samples: None,
            m_condensed_sample_weights: None,
            feature_offsets: offsets,
            chunk_size: running,
        };

        let info = BitDataInfo {
            compression,
            reconstruction,
        };

        debug!(
            "BitDataInfo ready: chunk_size={} bits, n_data_samples={}",
            info.chunk_size(),
            info.n_data_samples()
        );

        Ok(info)
    }

    pub fn features(&self) -> &[FeatureSpec] {
        &self.compression.features
    }

    pub fn feature_spec(&self, feature_idx: usize) -> &FeatureSpec {
        &self.compression.features[feature_idx]
    }

    pub fn original_size_bits(&self) -> usize {
        self.compression.original_size_bits
    }

    pub fn n_data_samples(&self) -> usize {
        self.compression.n_data_samples
    }

    pub fn m_condensed_samples(&self) -> Option<usize> {
        self.compression.m_condensed_samples
    }

    pub fn m_condensed_sample_weights(&self) -> Option<&[usize]> {
        self.compression.m_condensed_sample_weights.as_deref()
    }

    pub fn set_condensed_sample_weights(&mut self, weights: Option<Vec<usize>>) {
        self.compression.m_condensed_samples = weights.as_ref().map(Vec::len);
        self.compression.m_condensed_sample_weights = weights;
    }

    pub fn num_features(&self) -> usize {
        self.compression.features.len()
    }

    pub fn chunk_size(&self) -> usize {
        self.compression.chunk_size
    }

    pub fn feature_bits(&self, feature_idx: usize) -> usize {
        self.compression.features[feature_idx].bits
    }

    pub fn feature_offset(&self, feature_idx: usize) -> usize {
        self.compression.feature_offsets[feature_idx]
    }

    pub fn feature_index_for_bit(&self, bit_pos: usize) -> Option<usize> {
        if bit_pos >= self.compression.chunk_size {
            return None;
        }
        match self.compression.feature_offsets.binary_search(&bit_pos) {
            Ok(idx) => Some(idx),
            Err(0) => None,
            Err(idx) => Some(idx - 1),
        }
    }

    pub fn with_original_size_bits(&self, original_size_bits: usize) -> Self {
        let chunk_size = self.compression.chunk_size;
        BitDataInfo {
            compression: BitDataCompressionInfo {
                features: self.compression.features.clone(),
                original_size_bits,
                n_data_samples: original_size_bits / chunk_size,
                m_condensed_samples: self.compression.m_condensed_samples,
                m_condensed_sample_weights: self.compression.m_condensed_sample_weights.clone(),
                feature_offsets: self.compression.feature_offsets.clone(),
                chunk_size,
            },
            reconstruction: self.reconstruction.clone(),
        }
    }
}

/// Represents the bit-level preprocessed data where each row is a contiguous chunk of bits.
#[derive(Debug, Clone)]
pub struct BitData {
    /// The underlying bit storage - all chunks stored contiguously
    pub data: BitVec<usize, Msb0>,
    /// Total bits per chunk
    pub chunk_size: usize,
    /// Number of rows/chunks (number of records)
    pub num_rows: usize,
}

impl BitData {
    /// Extend the BitData with additional bits from a BitSlice (used for adding condensed samples)
    pub fn extend_from_bitslice(
        &mut self,
        bits: &BitSlice<usize, Msb0>,
    ) -> Result<(), EntroGdError> {
        if bits.len() != self.chunk_size {
            return Err(EntroGdError::BitSliceLengthMismatch {
                expected: self.chunk_size,
                actual: bits.len(),
            });
        }
        self.data.extend(bits);
        self.num_rows += 1; // Treat the new bits as an additional row/chunk
        Ok(())
    }

    /// Get a slice of bits for a specific row/chunk
    pub fn get_chunk(&self, row: usize) -> &BitSlice<usize, Msb0> {
        let start = row * self.chunk_size;
        let end = start + self.chunk_size;
        &self.data[start..end]
    }

    /// Get a specific bit by row and bit position within the chunk
    pub fn get_bit(&self, row: usize, bit_in_chunk: usize) -> bool {
        let idx = row * self.chunk_size + bit_in_chunk;
        self.data[idx]
    }

    /// Get raw access to the underlying bit vector
    pub fn raw(&self) -> &BitVec<usize, Msb0> {
        &self.data
    }

    /// Get the total number of bits stored
    pub fn total_bits(&self) -> usize {
        self.data.len()
    }
}

/// Combines BitData with its metadata for algorithms that need both.
#[derive(Debug, Clone)]
pub struct BitDataSet {
    pub data: BitData,
    pub info: BitDataInfo,
}

impl BitDataSet {
    /// Create BitDataSet from a Dataset using per-feature schema.
    /// This is the core method used by tabular preprocessing.
    pub fn from_dataset_with_schema(
        dataset: &Dataset,
        features: Vec<FeatureSpec>,
    ) -> Result<Self, EntroGdError> {
        let num_features = dataset.num_columns();
        debug!(
            rows = dataset.num_rows(),
            columns = num_features,
            schema_features = features.len(),
            "building BitDataSet with explicit schema"
        );
        if features.len() != num_features {
            return Err(EntroGdError::InvalidFeatureSpec {
                message: format!(
                    "feature spec count {} does not match dataset columns {}",
                    features.len(),
                    num_features
                ),
            });
        }

        for (idx, spec) in features.iter().enumerate() {
            let col_type = dataset.column_type(idx);
            debug!(
                feature_idx = idx,
                expected_column_type = ?col_type,
                spec_type = ?spec.data_type,
                bits = spec.bits,
                transform = ?spec.transform,
                "loaded feature spec"
            );
            if col_type != spec.data_type {
                return Err(EntroGdError::InvalidFeatureSpec {
                    message: format!(
                        "feature {} type {:?} does not match dataset column type {:?}",
                        idx, spec.data_type, col_type
                    ),
                });
            }
        }
        let info = BitDataInfo::new(features, 0)?;
        let chunk_size = info.chunk_size();
        let num_rows = dataset.num_rows();
        let total_bits = chunk_size * num_rows;

        debug!(
            chunk_size_bits = chunk_size,
            rows = num_rows,
            total_bits,
            "packing rows into bitstream"
        );

        let mut data = BitVec::<usize, Msb0>::with_capacity(total_bits);
        for row_idx in 0..num_rows {
            for col_idx in 0..num_features {
                let spec = info.feature_spec(col_idx);
                let value = dataset.value_at(row_idx, col_idx);
                let bits = value_to_bits(value, spec);
                if row_idx < 2 {
                    trace!(
                        row_idx,
                        feature_idx = col_idx,
                        value = ?value,
                        bits = spec.bits,
                        transform = ?spec.transform,
                        "packed feature value into bitstream"
                    );
                }
                push_bits(&mut data, bits, spec.bits);
            }
        }

        let data = BitData {
            data,
            chunk_size,
            num_rows,
        };
        let info = info.with_original_size_bits(chunk_size * num_rows);

        debug!(
            rows = num_rows,
            features = info.num_features(),
            chunk_size_bits = chunk_size,
            total_bits = chunk_size * num_rows,
            "BitDataSet preprocessing complete"
        );

        Ok(BitDataSet { data, info })
    }

    /// Get a slice of bits for a specific feature within a row
    pub fn get_feature(&self, row: usize, feature: usize) -> &BitSlice<usize, Msb0> {
        let chunk_start = row * self.data.chunk_size;
        let feat_start = chunk_start + self.info.feature_offset(feature);
        let feat_end = feat_start + self.info.feature_bits(feature);
        &self.data.data[feat_start..feat_end]
    }

    /// Get a specific bit by row, feature, and bit index within the feature
    pub fn get_bit_by_feature(&self, row: usize, feature: usize, bit: usize) -> bool {
        let idx = row * self.data.chunk_size + self.info.feature_offset(feature) + bit;
        self.data.data[idx]
    }

    pub fn num_rows(&self) -> usize {
        self.data.num_rows
    }

    pub fn num_features(&self) -> usize {
        self.info.num_features()
    }

    pub fn chunk_size(&self) -> usize {
        self.data.chunk_size
    }

    pub fn feature_bits(&self, feature_idx: usize) -> usize {
        self.info.feature_bits(feature_idx)
    }

    pub fn feature_offset(&self, feature_idx: usize) -> usize {
        self.info.feature_offset(feature_idx)
    }

    pub fn feature_index_for_bit(&self, bit_pos: usize) -> Option<usize> {
        self.info.feature_index_for_bit(bit_pos)
    }

    pub fn get_bit(&self, row: usize, bit_in_chunk: usize) -> bool {
        self.data.get_bit(row, bit_in_chunk)
    }

    pub fn get_chunk(&self, row: usize) -> &BitSlice<usize, Msb0> {
        self.data.get_chunk(row)
    }
}

impl Display for BitDataSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BitData:")?;
        writeln!(f, "  Number of rows: {}", self.data.num_rows)?;
        writeln!(f, "  Number of features: {}", self.info.num_features())?;
        writeln!(f, "  Chunk size (bits per row): {}", self.data.chunk_size)?;
        writeln!(f, "  Total bits: {}", self.data.total_bits())?;
        write!(f, "Bit  : ")?;
        for bit_pos in 0..self.data.chunk_size {
            write!(f, "{} ", bit_pos)?;
        }
        for row in 0..self.data.num_rows {
            writeln!(f)?;
            write!(f, "Row {}: ", row)?;
            for bit_pos in 0..self.data.chunk_size {
                let bit = self.data.get_bit(row, bit_pos);
                write!(f, "{} ", if bit { '1' } else { '0' })?;
            }
        }
        Ok(())
    }
}

fn value_to_bits(value: DataValue, spec: &FeatureSpec) -> u64 {
    match (value, spec.data_type) {
        (DataValue::Unsigned(v), FeatureDataType::UnsignedInt) => match spec.transform {
            FeatureTransform::None => v,
            FeatureTransform::OffsetUnsignedInt { min_value } => v.saturating_sub(min_value),
            _ => unreachable!("invalid transform for unsigned feature"),
        },
        (DataValue::Signed(v), FeatureDataType::SignedInt) => match spec.transform {
            FeatureTransform::OffsetSignedInt { min_value } => {
                let shifted = (v as i128) - (min_value as i128);
                shifted as u64
            }
            FeatureTransform::None => {
                if spec.bits == 64 {
                    v as u64
                } else {
                    let mask = (1u128 << spec.bits) - 1;
                    (v as i128 as u128 & mask) as u64
                }
            }
            _ => unreachable!("invalid transform for signed feature"),
        },
        (DataValue::F32(v), FeatureDataType::F32) => match spec.transform {
            FeatureTransform::None => v.to_bits() as u64,
            FeatureTransform::ScaledSignedInt { decimal_scale } => {
                let scaled = scale_float_to_i64(v as f64, decimal_scale);
                value_to_bits(
                    DataValue::Signed(scaled),
                    &FeatureSpec::new(FeatureDataType::SignedInt, spec.bits),
                )
            }
            FeatureTransform::ScaledOffsetSignedInt {
                decimal_scale,
                min_value,
            } => {
                let scaled = scale_float_to_i64(v as f64, decimal_scale);
                ((scaled as i128) - (min_value as i128)) as u64
            }
            _ => unreachable!("invalid transform for f32 feature"),
        },
        (DataValue::F64(v), FeatureDataType::F64) => match spec.transform {
            FeatureTransform::None => v.to_bits(),
            FeatureTransform::ScaledSignedInt { decimal_scale } => {
                let scaled = scale_float_to_i64(v, decimal_scale);
                value_to_bits(
                    DataValue::Signed(scaled),
                    &FeatureSpec::new(FeatureDataType::SignedInt, spec.bits),
                )
            }
            FeatureTransform::ScaledOffsetSignedInt {
                decimal_scale,
                min_value,
            } => {
                let scaled = scale_float_to_i64(v, decimal_scale);
                ((scaled as i128) - (min_value as i128)) as u64
            }
            _ => unreachable!("invalid transform for f64 feature"),
        },
        _ => unreachable!("feature type mismatch when packing bits"),
    }
}

pub fn decode_value_from_bits(bits: &BitSlice<usize, Msb0>, spec: &FeatureSpec) -> DataValue {
    let value = bits_to_u64(bits);
    match spec.data_type {
        FeatureDataType::UnsignedInt => match spec.transform {
            FeatureTransform::None => DataValue::Unsigned(value),
            FeatureTransform::OffsetUnsignedInt { min_value } => {
                DataValue::Unsigned(value.wrapping_add(min_value))
            }
            _ => unreachable!("invalid transform for unsigned feature"),
        },
        FeatureDataType::SignedInt => match spec.transform {
            FeatureTransform::None => DataValue::Signed(decode_signed(value, spec.bits)),
            FeatureTransform::OffsetSignedInt { min_value } => {
                DataValue::Signed((value as i128 + min_value as i128) as i64)
            }
            _ => unreachable!("invalid transform for signed feature"),
        },
        FeatureDataType::F32 => match spec.transform {
            FeatureTransform::None => DataValue::F32(f32::from_bits(value as u32)),
            FeatureTransform::ScaledSignedInt { decimal_scale } => {
                let scaled = decode_signed(value, spec.bits);
                let factor = 10f64.powi(decimal_scale as i32);
                DataValue::F32((scaled as f64 / factor) as f32)
            }
            FeatureTransform::ScaledOffsetSignedInt {
                decimal_scale,
                min_value,
            } => {
                let scaled = value as i128 + min_value as i128;
                let factor = 10f64.powi(decimal_scale as i32);
                DataValue::F32((scaled as f64 / factor) as f32)
            }
            _ => unreachable!("invalid transform for f32 feature"),
        },
        FeatureDataType::F64 => match spec.transform {
            FeatureTransform::None => DataValue::F64(f64::from_bits(value)),
            FeatureTransform::ScaledSignedInt { decimal_scale } => {
                let scaled = decode_signed(value, spec.bits);
                let factor = 10f64.powi(decimal_scale as i32);
                DataValue::F64(scaled as f64 / factor)
            }
            FeatureTransform::ScaledOffsetSignedInt {
                decimal_scale,
                min_value,
            } => {
                let scaled = value as i128 + min_value as i128;
                let factor = 10f64.powi(decimal_scale as i32);
                DataValue::F64(scaled as f64 / factor)
            }
            _ => unreachable!("invalid transform for f64 feature"),
        },
    }
}

fn bits_to_u64(bits: &BitSlice<usize, Msb0>) -> u64 {
    let mut value = 0u64;
    for bit in bits {
        value = (value << 1) | (*bit as u64);
    }
    value
}

fn decode_signed(value: u64, bits: usize) -> i64 {
    if bits == 64 {
        value as i64
    } else {
        let shift = 64 - bits;
        ((value << shift) as i64) >> shift
    }
}

fn scale_float_to_i64(value: f64, decimal_scale: u8) -> i64 {
    let factor = 10f64.powi(decimal_scale as i32);
    (value * factor).round() as i64
}

fn push_bits(stream: &mut BitVec<usize, Msb0>, mut value: u64, bits: usize) {
    if bits == 64 {
        for shift in (0..64).rev() {
            stream.push(((value >> shift) & 1) == 1);
        }
        return;
    }
    if bits < 64 {
        let mask = (1u64 << bits) - 1;
        value &= mask;
    }
    for shift in (0..bits).rev() {
        stream.push(((value >> shift) & 1) == 1);
    }
}


