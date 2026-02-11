use bitvec::prelude::*;
use std::fmt::Display;

use crate::data_loader::Dataset;
use crate::error::EntroGdError;

/// Supported feature data types for parsing and bit packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureDataType {
    SignedInt,
    UnsignedInt,
    F32,
    F64,
}

/// Per-feature schema entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSpec {
    pub data_type: FeatureDataType,
    pub bits: usize,
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
    pub fn new(features: Vec<FeatureSpec>, original_size_bits: usize) -> Result<Self, EntroGdError> {
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
                FeatureDataType::F32 if spec.bits != 32 => {
                    return Err(EntroGdError::InvalidFeatureSpec {
                        message: "f32 features must use 32 bits".to_string(),
                    });
                }
                FeatureDataType::F64 if spec.bits != 64 => {
                    return Err(EntroGdError::InvalidFeatureSpec {
                        message: "f64 features must use 64 bits".to_string(),
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
    pub fn extend_from_bitslice(&mut self, bits: &BitSlice<usize, Msb0>) -> Result<(), EntroGdError> {
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
    /// Create BitDataSet from a Dataset using a uniform bits-per-feature schema.
    pub fn from_dataset(dataset: &Dataset, bits_per_feature: usize) -> Result<Self, EntroGdError> {
        let num_features = dataset.num_columns();
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UnsignedInt,
                bits: bits_per_feature,
            };
            num_features
        ];
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
        let info = BitDataInfo::new(features, 0)?;
        let chunk_size = info.chunk_size();
        let num_rows = dataset.num_rows();
        let total_bits = chunk_size * num_rows;

        let mut data = BitVec::<usize, Msb0>::with_capacity(total_bits);
        for (row_idx, row) in dataset.rows().iter().enumerate() {
            for (col_idx, value_str) in row.iter().enumerate() {
                let spec = &info.features[col_idx];
                let bits = match parse_value_to_bits(value_str, spec) {
                    Ok(bits) => bits,
                    Err(ParseValueError::Int(source)) => {
                        return Err(EntroGdError::ParseValue {
                            value: value_str.to_string(),
                            row: row_idx,
                            column: col_idx,
                            source,
                        });
                    }
                    Err(ParseValueError::Float(source)) => {
                        return Err(EntroGdError::ParseFloatValue {
                            value: value_str.to_string(),
                            row: row_idx,
                            column: col_idx,
                            source,
                        });
                    }
                };
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

enum ParseValueError {
    Int(std::num::ParseIntError),
    Float(std::num::ParseFloatError),
}

fn parse_value_to_bits(value: &str, spec: &FeatureSpec) -> Result<u64, ParseValueError> {
    match spec.data_type {
        FeatureDataType::UnsignedInt => value.trim().parse::<u64>().map_err(ParseValueError::Int),
        FeatureDataType::SignedInt => {
            let v = value.trim().parse::<i64>().map_err(ParseValueError::Int)?;
            let bits = if spec.bits == 64 {
                v as u64
            } else {
                let mask = (1u128 << spec.bits) - 1;
                (v as i128 as u128 & mask) as u64
            };
            Ok(bits)
        }
        FeatureDataType::F32 => {
            let v: f32 = value.trim().parse().map_err(ParseValueError::Float)?;
            Ok(v.to_bits() as u64)
        }
        FeatureDataType::F64 => {
            let v: f64 = value.trim().parse().map_err(ParseValueError::Float)?;
            Ok(v.to_bits())
        }
    }
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
    use std::mem::size_of;

    #[test]
    fn print_bitdata_head() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).expect("Failed to load dataset");
        let bits_per_feature = size_of::<u8>() * 8;

        let bit_data = BitDataSet::from_dataset(&dataset, bits_per_feature)
            .expect("Failed to create BitData from dataset");

        println!("\nFirst 5 rows in bit representation (MSB first):");
        println!("==============================================");

        // Print the first 5 chunks with a delimiter at each feature boundary
        for row_index in 0..5 {
            println!("\nRow {}:", row_index);

            for feat_idx in 0..bit_data.info.num_features() {
                print!("  Feature {}: ", feat_idx);
                let feature_bits = bit_data.get_feature(row_index, feat_idx);
                for i in 0..feature_bits.len() {
                    print!("{}", if feature_bits[i] { '1' } else { '0' });
                }
                println!();
            }
        }
    }

    #[test]
    fn test_bitdata_from_dataset() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bits_per_feature = 8;

        let bit_data = BitDataSet::from_dataset(&dataset, bits_per_feature)
            .expect("Failed to create BitData from dataset");

        assert_eq!(bit_data.info.num_features(), 8);
        assert_eq!(bit_data.info.feature_bits(0), 8);
        assert_eq!(bit_data.data.chunk_size, 8 * 8); // 64 bits per row
        assert_eq!(bit_data.data.num_rows, 10000);
        assert_eq!(bit_data.data.total_bits(), 10000 * 64);
    }

    #[test]
    fn test_chunk_access() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bit_data = BitDataSet::from_dataset(&dataset, 8)
            .expect("Failed to create BitData from dataset");

        let chunk = bit_data.data.get_chunk(0);
        assert_eq!(chunk.len(), 64); // 8 features * 8 bits

        let feature = bit_data.get_feature(0, 0);
        assert_eq!(feature.len(), 8);
    }

    #[test]
    fn test_get_bit_individual_access() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bit_data = BitDataSet::from_dataset(&dataset, 8)
            .expect("Failed to create BitData from dataset");

        // Verify get_bit is consistent with get_feature
        for row in 0..10 {
            for feat in 0..8 {
                let feature_slice = bit_data.get_feature(row, feat);
                for bit in 0..8 {
                    assert_eq!(
                        bit_data.get_bit_by_feature(row, feat, bit),
                        feature_slice[bit],
                        "Mismatch at row={}, feat={}, bit={}",
                        row, feat, bit
                    );
                }
            }
        }
    }

    #[test]
    fn test_raw_access() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bit_data = BitDataSet::from_dataset(&dataset, 8)
            .expect("Failed to create BitData from dataset");

        let raw = bit_data.data.raw();
        assert_eq!(raw.len(), bit_data.data.total_bits());

        // Verify raw access matches get_bit
        for i in 0..100 {
            let row = i / 64;
            let within_chunk = i % 64;
            let feat = within_chunk / 8;
            let bit = within_chunk % 8;
            assert_eq!(raw[i], bit_data.get_bit_by_feature(row, feat, bit));
        }
    }

    #[test]
    fn test_single_bit_precision() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bit_data = BitDataSet::from_dataset(&dataset, 1)
            .expect("Failed to create BitData from dataset");

        // With 1 bit, values should only be 0 or 1
        assert_eq!(bit_data.info.feature_bits(0), 1);
        assert_eq!(bit_data.data.chunk_size, 8);
    }

    #[test]
    fn test_schema_with_floats() {
        let rows = vec![
            vec!["1".to_string(), "3.5".to_string(), "-2".to_string(), "1.25".to_string()],
            vec!["2".to_string(), "-1.0".to_string(), "5".to_string(), "0.5".to_string()],
        ];
        let dataset = Dataset::from_rows(rows);
        let features = vec![
            FeatureSpec { data_type: FeatureDataType::UnsignedInt, bits: 8 },
            FeatureSpec { data_type: FeatureDataType::F32, bits: 32 },
            FeatureSpec { data_type: FeatureDataType::SignedInt, bits: 8 },
            FeatureSpec { data_type: FeatureDataType::F64, bits: 64 },
        ];

        let bit_data = BitDataSet::from_dataset_with_schema(&dataset, features).unwrap();
        assert_eq!(bit_data.info.num_features(), 4);
        assert_eq!(bit_data.data.chunk_size, 112);
        assert_eq!(bit_data.data.num_rows, 2);
    }
}
