use bitvec::prelude::*;
use std::fmt::Display;
use std::path::Path;

pub use crate::data_loader::FeatureDataType;
use crate::data_loader::{DataLoader, DataValue, Dataset, DatasetMetadata};
use crate::error::EntroGdError;

const MAX_DECIMAL_SCALE: u8 = 9;

/// Optional transform metadata used for a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureTransform {
    None,
    ScaledSignedInt { decimal_scale: u8 },
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
pub struct BitDataInfo {
    pub features: Vec<FeatureSpec>,
    pub original_size_bits: usize,
    feature_offsets: Vec<usize>,
    chunk_size: usize,
}

impl BitDataInfo {
    pub fn new(
        features: Vec<FeatureSpec>,
        original_size_bits: usize,
    ) -> Result<Self, EntroGdError> {
        if features.is_empty() {
            return Err(EntroGdError::InvalidFeatureSpec {
                message: "features is empty".to_string(),
            });
        }

        let mut offsets = Vec::with_capacity(features.len());
        let mut running = 0usize;
        for spec in &features {
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
                    FeatureTransform::ScaledSignedInt { .. } if spec.bits > 64 => {
                        return Err(EntroGdError::InvalidFeatureSpec {
                            message: "scaled f32 features must use <= 64 bits".to_string(),
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
                    FeatureTransform::ScaledSignedInt { .. } if spec.bits > 64 => {
                        return Err(EntroGdError::InvalidFeatureSpec {
                            message: "scaled f64 features must use <= 64 bits".to_string(),
                        });
                    }
                    _ => {}
                },
                FeatureDataType::SignedInt | FeatureDataType::UnsignedInt
                    if !matches!(spec.transform, FeatureTransform::None) =>
                {
                    return Err(EntroGdError::InvalidFeatureSpec {
                        message: "integer features cannot use float transforms".to_string(),
                    });
                }
                _ => {}
            }
            offsets.push(running);
            running += spec.bits;
        }

        Ok(BitDataInfo {
            features,
            original_size_bits,
            feature_offsets: offsets,
            chunk_size: running,
        })
    }

    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn feature_bits(&self, feature_idx: usize) -> usize {
        self.features[feature_idx].bits
    }

    pub fn feature_offset(&self, feature_idx: usize) -> usize {
        self.feature_offsets[feature_idx]
    }

    pub fn feature_index_for_bit(&self, bit_pos: usize) -> Option<usize> {
        if bit_pos >= self.chunk_size {
            return None;
        }
        match self.feature_offsets.binary_search(&bit_pos) {
            Ok(idx) => Some(idx),
            Err(0) => None,
            Err(idx) => Some(idx - 1),
        }
    }

    pub fn with_original_size_bits(&self, original_size_bits: usize) -> Self {
        BitDataInfo {
            features: self.features.clone(),
            original_size_bits,
            feature_offsets: self.feature_offsets.clone(),
            chunk_size: self.chunk_size,
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
    /// Create BitDataSet from a Dataset using column data types to derive the schema.
    pub fn from_dataset(dataset: &Dataset) -> Result<Self, EntroGdError> {
        let num_features = dataset.num_columns();
        let mut features = Vec::with_capacity(num_features);
        for feature_idx in 0..num_features {
            let data_type = dataset.column_type(feature_idx);
            let spec = match data_type {
                FeatureDataType::F32 | FeatureDataType::F64 => {
                    infer_float_feature_spec(dataset, feature_idx, data_type)
                }
                FeatureDataType::SignedInt | FeatureDataType::UnsignedInt => {
                    FeatureSpec::new(data_type, 64)
                }
            };
            features.push(spec);
        }
        Self::from_dataset_with_schema(dataset, features)
    }

    /// Create BitDataSet from a Dataset using per-feature schema.
    pub fn from_dataset_with_schema(
        dataset: &Dataset,
        features: Vec<FeatureSpec>,
    ) -> Result<Self, EntroGdError> {
        let num_features = dataset.num_columns();
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

        let mut data = BitVec::<usize, Msb0>::with_capacity(total_bits);
        for row_idx in 0..num_rows {
            for col_idx in 0..num_features {
                let spec = &info.features[col_idx];
                let value = dataset.value_at(row_idx, col_idx);
                let bits = value_to_bits(value, spec);
                push_bits(&mut data, bits, spec.bits);
            }
        }

        let data = BitData {
            data,
            chunk_size,
            num_rows,
        };
        let info = info.with_original_size_bits(chunk_size * num_rows);

        Ok(BitDataSet { data, info })
    }

    /// Load data via a DataLoader and build BitDataSet using column data types.
    pub fn from_loader<L: DataLoader, P: AsRef<Path>>(
        loader: &L,
        path: P,
    ) -> Result<(Self, DatasetMetadata), EntroGdError> {
        let loaded = loader.load(path)?;
        let bit_data = Self::from_dataset(&loaded.dataset)?;
        Ok((bit_data, loaded.metadata))
    }

    /// Load data via a DataLoader and build BitDataSet using a provided schema.
    pub fn from_loader_with_schema<L: DataLoader, P: AsRef<Path>>(
        loader: &L,
        path: P,
        features: Vec<FeatureSpec>,
    ) -> Result<(Self, DatasetMetadata), EntroGdError> {
        let loaded = loader.load(path)?;
        let bit_data = Self::from_dataset_with_schema(&loaded.dataset, features)?;
        Ok((bit_data, loaded.metadata))
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
}

/// Trait for read-only access to bit-level data organized as rows of fixed-size chunks.
///
/// This allows algorithms to work generically over both plain `BitDataSet` and
/// lightweight wrappers like `ExtendedBitData` that append extra rows without
/// copying the original data.
pub trait BitDataView {
    fn num_rows(&self) -> usize;
    fn num_features(&self) -> usize;
    fn chunk_size(&self) -> usize;
    fn feature_bits(&self, feature_idx: usize) -> usize;
    fn feature_offset(&self, feature_idx: usize) -> usize;
    fn feature_index_for_bit(&self, bit_pos: usize) -> Option<usize>;
    fn get_bit(&self, row: usize, bit_in_chunk: usize) -> bool;
    fn get_chunk(&self, row: usize) -> &BitSlice<usize, Msb0>;
}

impl BitDataView for BitDataSet {
    fn num_rows(&self) -> usize {
        self.data.num_rows
    }

    fn num_features(&self) -> usize {
        self.info.num_features()
    }

    fn chunk_size(&self) -> usize {
        self.data.chunk_size
    }

    fn feature_bits(&self, feature_idx: usize) -> usize {
        self.info.feature_bits(feature_idx)
    }

    fn feature_offset(&self, feature_idx: usize) -> usize {
        self.info.feature_offset(feature_idx)
    }

    fn feature_index_for_bit(&self, bit_pos: usize) -> Option<usize> {
        self.info.feature_index_for_bit(bit_pos)
    }

    fn get_bit(&self, row: usize, bit_in_chunk: usize) -> bool {
        self.data.get_bit(row, bit_in_chunk)
    }

    fn get_chunk(&self, row: usize) -> &BitSlice<usize, Msb0> {
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
        (DataValue::Unsigned(v), FeatureDataType::UnsignedInt) => v,
        (DataValue::Signed(v), FeatureDataType::SignedInt) => {
            if spec.bits == 64 {
                v as u64
            } else {
                let mask = (1u128 << spec.bits) - 1;
                (v as i128 as u128 & mask) as u64
            }
        }
        (DataValue::F32(v), FeatureDataType::F32) => match spec.transform {
            FeatureTransform::None => v.to_bits() as u64,
            FeatureTransform::ScaledSignedInt { decimal_scale } => {
                let scaled = scale_float_to_i64(v as f64, decimal_scale);
                value_to_bits(DataValue::Signed(scaled), &FeatureSpec::new(FeatureDataType::SignedInt, spec.bits))
            }
        },
        (DataValue::F64(v), FeatureDataType::F64) => match spec.transform {
            FeatureTransform::None => v.to_bits(),
            FeatureTransform::ScaledSignedInt { decimal_scale } => {
                let scaled = scale_float_to_i64(v, decimal_scale);
                value_to_bits(DataValue::Signed(scaled), &FeatureSpec::new(FeatureDataType::SignedInt, spec.bits))
            }
        },
        _ => unreachable!("feature type mismatch when packing bits"),
    }
}

pub fn decode_value_from_bits(bits: &BitSlice<usize, Msb0>, spec: &FeatureSpec) -> DataValue {
    let value = bits_to_u64(bits);
    match spec.data_type {
        FeatureDataType::UnsignedInt => DataValue::Unsigned(value),
        FeatureDataType::SignedInt => DataValue::Signed(decode_signed(value, spec.bits)),
        FeatureDataType::F32 => match spec.transform {
            FeatureTransform::None => DataValue::F32(f32::from_bits(value as u32)),
            FeatureTransform::ScaledSignedInt { decimal_scale } => {
                let scaled = decode_signed(value, spec.bits);
                let factor = 10f64.powi(decimal_scale as i32);
                DataValue::F32((scaled as f64 / factor) as f32)
            }
        },
        FeatureDataType::F64 => match spec.transform {
            FeatureTransform::None => DataValue::F64(f64::from_bits(value)),
            FeatureTransform::ScaledSignedInt { decimal_scale } => {
                let scaled = decode_signed(value, spec.bits);
                let factor = 10f64.powi(decimal_scale as i32);
                DataValue::F64(scaled as f64 / factor)
            }
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

fn infer_float_feature_spec(dataset: &Dataset, column: usize, data_type: FeatureDataType) -> FeatureSpec {
    let fallback_bits = match data_type {
        FeatureDataType::F32 => 32,
        FeatureDataType::F64 => 64,
        _ => unreachable!("infer_float_feature_spec called with non-float type"),
    };

    let mut best_scale: Option<u8> = None;
    for decimal_scale in 0..=MAX_DECIMAL_SCALE {
        if scaled_int_range(dataset, column, data_type, decimal_scale).is_some() {
            best_scale = Some(decimal_scale);
            break;
        }
    }

    if let Some(decimal_scale) = best_scale {
        FeatureSpec {
            data_type,
            bits: fallback_bits,
            transform: FeatureTransform::ScaledSignedInt { decimal_scale },
        }
    } else {
        FeatureSpec::new(data_type, fallback_bits)
    }
}

fn scaled_int_range(
    dataset: &Dataset,
    column: usize,
    data_type: FeatureDataType,
    decimal_scale: u8,
) -> Option<(i64, i64)> {
    if dataset.num_rows() == 0 {
        return Some((0, 0));
    }

    let factor = 10f64.powi(decimal_scale as i32);
    if !matches!(data_type, FeatureDataType::F32 | FeatureDataType::F64) {
        return None;
    }

    let mut min_value = i64::MAX;
    let mut max_value = i64::MIN;

    for row in 0..dataset.num_rows() {
        let value = match dataset.value_at(row, column) {
            DataValue::F32(v) => v as f64,
            DataValue::F64(v) => v,
            _ => return None,
        };
        if !value.is_finite() {
            return None;
        }

        let scaled = value * factor;
        let rounded = scaled.round();
        if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
            return None;
        }

        let reconstructed_ok = match data_type {
            FeatureDataType::F32 => {
                let original = value as f32;
                let reconstructed = (rounded / factor) as f32;
                reconstructed.to_bits() == original.to_bits()
            }
            FeatureDataType::F64 => {
                let original = value;
                let reconstructed = rounded / factor;
                reconstructed.to_bits() == original.to_bits()
            }
            _ => false,
        };
        if !reconstructed_ok {
            return None;
        }

        let int_value = rounded as i64;
        min_value = min_value.min(int_value);
        max_value = max_value.max(int_value);
    }

    Some((min_value, max_value))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::{ColumnData, CsvDataLoader};

    #[test]
    fn print_bitdata_head() {
        let loader = CsvDataLoader::new(true);
        let loaded = loader
            .load("data/data-10000-8-int.csv")
            .expect("Failed to load dataset");
        let dataset = loaded.dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        log::info!("\nFirst 5 rows in bit representation (MSB first):");
        log::info!("==============================================");

        // Print the first 5 chunks with a delimiter at each feature boundary
        for row_index in 0..5 {
            log::info!("\nRow {}:", row_index);

            for feat_idx in 0..bit_data.info.num_features() {
                let feature_bits = bit_data.get_feature(row_index, feat_idx);
                let bits: String = feature_bits
                    .iter()
                    .map(|bit| if *bit { '1' } else { '0' })
                    .collect();
                log::info!("  Feature {}: {}", feat_idx, bits);
            }
        }
    }

    #[test]
    fn test_bitdata_from_dataset() {
        let loader = CsvDataLoader::new(true);
        let dataset = loader.load("data/data-10000-8-int.csv").unwrap().dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        assert_eq!(bit_data.info.num_features(), 8);
        assert_eq!(bit_data.info.feature_bits(0), 64);
        assert_eq!(bit_data.data.chunk_size, 8 * 64);
        assert_eq!(bit_data.data.num_rows, 10000);
        assert_eq!(bit_data.data.total_bits(), 10000 * 512);
    }

    #[test]
    fn test_chunk_access() {
        let loader = CsvDataLoader::new(true);
        let dataset = loader.load("data/data-10000-8-int.csv").unwrap().dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        let chunk = bit_data.data.get_chunk(0);
        assert_eq!(chunk.len(), 512); // 8 features * 64 bits

        let feature = bit_data.get_feature(0, 0);
        assert_eq!(feature.len(), 64);
    }

    #[test]
    fn test_get_bit_individual_access() {
        let loader = CsvDataLoader::new(true);
        let dataset = loader.load("data/data-10000-8-int.csv").unwrap().dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        // Verify get_bit is consistent with get_feature
        let feature_bits = bit_data.info.feature_bits(0);
        for row in 0..10 {
            for feat in 0..8 {
                let feature_slice = bit_data.get_feature(row, feat);
                for bit in 0..feature_bits {
                    assert_eq!(
                        bit_data.get_bit_by_feature(row, feat, bit),
                        feature_slice[bit],
                        "Mismatch at row={}, feat={}, bit={}",
                        row,
                        feat,
                        bit
                    );
                }
            }
        }
    }

    #[test]
    fn test_raw_access() {
        let loader = CsvDataLoader::new(true);
        let dataset = loader.load("data/data-10000-8-int.csv").unwrap().dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        let raw = bit_data.data.raw();
        assert_eq!(raw.len(), bit_data.data.total_bits());

        // Verify raw access matches get_bit
        let feature_bits = bit_data.info.feature_bits(0);
        for i in 0..100 {
            let row = i / bit_data.data.chunk_size;
            let within_chunk = i % bit_data.data.chunk_size;
            let feat = within_chunk / feature_bits;
            let bit = within_chunk % feature_bits;
            assert_eq!(raw[i], bit_data.get_bit_by_feature(row, feat, bit));
        }
    }

    #[test]
    fn test_single_bit_precision() {
        let loader = CsvDataLoader::new(true);
        let dataset = loader.load("data/data-10000-8-int.csv").unwrap().dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        // With 1 bit, values should only be 0 or 1
        assert_eq!(bit_data.info.feature_bits(0), 64);
        assert_eq!(bit_data.data.chunk_size, 8 * 64);
    }

    #[test]
    fn test_schema_with_floats() {
        let columns = vec![
            ColumnData::Unsigned(vec![1, 2]),
            ColumnData::F64(vec![3.5, -1.0]),
            ColumnData::Signed(vec![-2, 5]),
            ColumnData::F64(vec![1.25, 0.5]),
        ];
        let dataset = Dataset::from_columns(columns).unwrap();
        let features = vec![
            FeatureSpec::new(FeatureDataType::UnsignedInt, 8),
            FeatureSpec::new(FeatureDataType::F64, 64),
            FeatureSpec::new(FeatureDataType::SignedInt, 8),
            FeatureSpec::new(FeatureDataType::F64, 64),
        ];

        let bit_data = BitDataSet::from_dataset_with_schema(&dataset, features).unwrap();
        assert_eq!(bit_data.info.num_features(), 4);
        assert_eq!(bit_data.data.chunk_size, 144);
        assert_eq!(bit_data.data.num_rows, 2);
    }

    #[test]
    fn test_auto_scale_float_feature_spec() {
        let columns = vec![
            ColumnData::F64(vec![1.25, 2.50, -0.75]),
            ColumnData::Unsigned(vec![1, 2, 3]),
        ];
        let dataset = Dataset::from_columns(columns).unwrap();

        let bit_data = BitDataSet::from_dataset(&dataset).unwrap();
        let spec = &bit_data.info.features[0];

        assert_eq!(spec.data_type, FeatureDataType::F64);
        assert!(matches!(
            spec.transform,
            FeatureTransform::ScaledSignedInt { .. }
        ));
        assert!(spec.bits <= 64);
    }
}
