use bitvec::prelude::*;
use std::fmt::Display;

use crate::data_loader::Dataset;

/// Represents the bit-level preprocessed data where each feature is stored as bits
/// and all features for a record are concatenated into a contiguous "chunk" of bits.
#[derive(Debug, Clone)]
pub struct BitData {
    /// The underlying bit storage - all chunks stored contiguously
    pub data: BitVec<usize, Msb0>,
    /// Number of bits per feature
    pub bits_per_feature: usize,
    /// Number of features per chunk (per record)
    pub num_features: usize,
    /// Total bits per chunk (bits_per_feature * num_features)
    pub chunk_size: usize,
    /// Number of rows/chunks (number of records)
    pub num_rows: usize,
}

impl BitData {
    /// Create BitData from a Dataset by converting each value to its bit representation
    /// 
    /// # Arguments
    /// * `dataset` - The source dataset with heterogeneous column types
    /// * `bits_per_feature` - Number of bits to use for each feature value
    /// 
    /// # Note
    /// This function loads the raw bit representation without preprocessing.
    /// For integer columns, it takes the lower bits_per_feature bits of each value.
    /// For float columns, it converts to integer first (truncation).
    pub fn from_dataset(dataset: &Dataset, bits_per_feature: usize) -> Self {
        let num_features = dataset.num_columns();
        let num_rows = dataset.num_rows();
        let chunk_size = bits_per_feature * num_features;
        let total_bits = chunk_size * num_rows;

        let mut data = BitVec::<usize, Msb0>::with_capacity(total_bits);

        dataset.rows().iter().for_each(|row| {
            let row_parsed: Vec<u8> = row.iter().map(|value_str| {
                // Parse the value as u64
                value_str.parse().unwrap_or(0)
            }).collect();
            let row_bits: BitVec<u8, Msb0> = BitVec::from_vec(row_parsed);
            data.extend(row_bits);
        });

        BitData {
            data,
            bits_per_feature,
            num_features,
            chunk_size,
            num_rows,
        }
    }
    
    /// Get a slice of bits for a specific row/chunk
    pub fn get_chunk(&self, row: usize) -> &BitSlice<usize, Msb0> {
        let start = row * self.chunk_size;
        let end = start + self.chunk_size;
        &self.data[start..end]
    }
    
    /// Get a slice of bits for a specific feature within a row
    pub fn get_feature(&self, row: usize, feature: usize) -> &BitSlice<usize, Msb0> {
        let chunk_start = row * self.chunk_size;
        let feat_start = chunk_start + feature * self.bits_per_feature;
        let feat_end = feat_start + self.bits_per_feature;
        &self.data[feat_start..feat_end]
    }
    
    /// Get a specific bit by row, feature, and bit index within the feature
    pub fn get_bit_by_feature(&self, row: usize, feature: usize, bit: usize) -> bool {
        let idx = row * self.chunk_size + feature * self.bits_per_feature + bit;
        self.data[idx]
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

impl Display for BitData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BitData:")?;
        writeln!(f, "  Number of rows: {}", self.num_rows)?;
        writeln!(f, "  Number of features: {}", self.num_features)?;
        writeln!(f, "  Bits per feature: {}", self.bits_per_feature)?;
        writeln!(f, "  Chunk size (bits per row): {}", self.chunk_size)?;
        writeln!(f, "  Total bits: {}", self.total_bits())?;
        write!(f, "Bit  : ")?;
        for bit_pos in 0..self.chunk_size {
            write!(f, "{} ", bit_pos)?;
        }
        for row in 0..self.num_rows {
            writeln!(f)?;
            write!(f, "Row {}: ", row)?;
            for bit_pos in 0..self.chunk_size {
                let bit = self.get_bit(row, bit_pos);
                write!(f, "{} ", if bit { '1' } else { '0' })?;
                if bit_pos % self.bits_per_feature == self.bits_per_feature - 1 && bit_pos + 1 != self.chunk_size {
                    write!(f, "| ")?;
                }
            }
        }
        Ok(())
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
        
        let bit_data = BitData::from_dataset(&dataset, bits_per_feature);
        
        println!("\nFirst 5 rows in bit representation (MSB first):");
        println!("==============================================");
        
        // Print the first 5 chunks with a delimiter at each feature boundary
        for row_index in 0..5 {
            println!("\nRow {}:", row_index);
            
            // Also print the original values for comparison
            
            // Print bits
            let chunk = bit_data.get_chunk(row_index);
            for feat_idx in 0..bit_data.num_features {
                print!("  Feature {}: ", feat_idx);
                let feat_start = feat_idx * bits_per_feature;
                let feat_end = feat_start + bits_per_feature;
                
                for i in feat_start..feat_end {
                    print!("{}", if chunk[i] { '1' } else { '0' });
                    if (i - feat_start + 1) % 8 == 0 && i + 1 != feat_end {
                        print!(" ");
                    }
                }
                
                // Also show the reconstructed value
                let feature_bits = bit_data.get_feature(row_index, feat_idx);
                let value: u64 = feature_bits.load_le();
                print!(" = {} (as int: {})", value, value);
                println!();
            }
        }
    }
    
    #[test]
    fn test_bitdata_from_dataset() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bits_per_feature = 8;
        
        let bit_data = BitData::from_dataset(&dataset, bits_per_feature);
        
        assert_eq!(bit_data.num_features, 8);
        assert_eq!(bit_data.bits_per_feature, 8);
        assert_eq!(bit_data.chunk_size, 8 * 8); // 64 bits per row
        assert_eq!(bit_data.num_rows, 10000);
        assert_eq!(bit_data.total_bits(), 10000 * 64);
    }
    
    #[test]
    fn test_chunk_access() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bit_data = BitData::from_dataset(&dataset, 8);
        
        let chunk = bit_data.get_chunk(0);
        assert_eq!(chunk.len(), 64); // 8 features * 8 bits
        
        let feature = bit_data.get_feature(0, 0);
        assert_eq!(feature.len(), 8);
    }
    
    #[test]
    fn test_different_bit_widths() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        
        // Test with 4 bits per feature
        let bit_data_4 = BitData::from_dataset(&dataset, 4);
        assert_eq!(bit_data_4.bits_per_feature, 4);
        assert_eq!(bit_data_4.chunk_size, 8 * 4); // 32 bits per row
        assert_eq!(bit_data_4.total_bits(), 10000 * 32);
        
        // Test with 16 bits per feature
        let bit_data_16 = BitData::from_dataset(&dataset, 16);
        assert_eq!(bit_data_16.bits_per_feature, 16);
        assert_eq!(bit_data_16.chunk_size, 8 * 16); // 128 bits per row
        assert_eq!(bit_data_16.total_bits(), 10000 * 128);
    }
    
    #[test]
    fn test_get_bit_individual_access() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bit_data = BitData::from_dataset(&dataset, 8);
        
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
    fn test_quantization_bounds() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bit_data = BitData::from_dataset(&dataset, 8);
        
        // With 8 bits, values should be in [0, 255]
        // Check that feature values are within valid quantized range
        for row in 0..100 {
            for feat in 0..8 {
                let feature_slice = bit_data.get_feature(row, feat);
                let value: u64 = feature_slice.load_le();
                assert!(value <= 255, "Quantized value {} exceeds 8-bit max at row={}, feat={}", value, row, feat);
            }
        }
    }
    
    #[test]
    fn test_chunk_feature_consistency() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bit_data = BitData::from_dataset(&dataset, 8);
        
        // Verify that get_chunk and get_feature return consistent data
        for row in 0..10 {
            let chunk = bit_data.get_chunk(row);
            for feat in 0..8 {
                let feature = bit_data.get_feature(row, feat);
                let chunk_slice = &chunk[feat * 8..(feat + 1) * 8];
                assert_eq!(feature, chunk_slice, "Mismatch at row={}, feat={}", row, feat);
            }
        }
    }
    
    #[test]
    fn test_raw_access() {
        let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
        let bit_data = BitData::from_dataset(&dataset, 8);
        
        let raw = bit_data.raw();
        assert_eq!(raw.len(), bit_data.total_bits());
        
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
        let bit_data = BitData::from_dataset(&dataset, 1);
        
        // With 1 bit, values should only be 0 or 1
        assert_eq!(bit_data.bits_per_feature, 1);
        assert_eq!(bit_data.chunk_size, 8);
        
        for row in 0..100 {
            for feat in 0..8 {
                let val: u64 = bit_data.get_feature(row, feat).load_le();
                assert!(val <= 1, "1-bit value should be 0 or 1, got {}", val);
            }
        }
    }
}
