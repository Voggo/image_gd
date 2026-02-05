use crate::compression::base_bit_groups::BaseBitGroups;
use crate::compression::entropy;
use crate::preprocessor::BitData;
use bitvec::prelude::*;

/// Represents the compressed output
#[derive(Debug, Clone)]
pub struct CompressedData {
    /// Base table mapping base patterns to their frequencies or encodings
    pub base_table: Vec<(BitVec<usize, Msb0>, usize)>,
    /// The encoded data stream
    pub encoded_data: DeviationData,
    /// Metadata for decompression (column count, base bits used, etc.)
    pub metadata: CompressionMetadata,
    /// Bit positions used as base bits during compression
    pub base_bit_positions: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    pub num_features: usize,
    pub original_size: usize,
    /// Total bits per row/chunk
    pub chunk_size: usize,
}

#[derive(Debug, Clone)]
pub struct DeviationSample {
        pub deviation: BitVec<usize, Msb0>,
        pub id: BitVec<usize, Msb0>,
    }

#[derive(Debug, Clone)]
pub struct DeviationData {
    encoded_bit_stream: BitVec<usize, Msb0>,
    num_samples: usize,
    num_deviation_bits: usize,
    num_id_bits: usize,
}

impl DeviationData {
    pub fn new(
        encoded_bit_stream: BitVec<usize, Msb0>,
        num_samples: usize,
        num_deviation_bits: usize,
        num_id_bits: usize,
    ) -> Self {
        DeviationData {
            encoded_bit_stream,
            num_samples,
            num_deviation_bits,
            num_id_bits,
        }
    }

    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        if sample_idx >= self.num_samples {
            return None;
        }
        let start_bit = sample_idx * (self.num_deviation_bits + self.num_id_bits);
        let end_bit = start_bit + self.num_deviation_bits + self.num_id_bits;
        Some(DeviationSample {
            deviation: self.encoded_bit_stream[start_bit..start_bit + self.num_deviation_bits].to_bitvec(),
            id: self.encoded_bit_stream[start_bit + self.num_deviation_bits..end_bit].to_bitvec(),
        })
    }

    pub fn get_encoded_size(&self) -> usize {
        self.encoded_bit_stream.len()
    }

    pub fn get_num_samples(&self) -> usize {
        self.num_samples
    }

    pub fn get_num_deviation_bits(&self) -> usize {
        self.num_deviation_bits
    }

    pub fn get_num_id_bits(&self) -> usize {
        self.num_id_bits
    }
}


/// Main compression function - transforms BitData into CompressedData
pub fn compress(bit_data: &BitData) -> CompressedData {
    let entropy = entropy::calculate_entropy(bit_data);
    let constant_base_bits = get_constant_base_bits(&entropy);
    let mut base_bit_groups = BaseBitGroups::new(bit_data, constant_base_bits);

    // Phase 1: Optimize the base bit groups for best compression ratio
    optimize_base_bit_groups(&bit_data, &mut base_bit_groups, entropy);

    // Phase 2: Extract the base table from the optimized groups
    let base_table = base_bit_groups.get_bases(bit_data);

    // Phase 3: Encode the data using the optimized base groups
    let encoded_data = encode_data(bit_data, &base_bit_groups);

    let base_bit_positions = base_bit_groups.get_base_bit_positions().to_owned();
    
    CompressedData {
        base_table,
        encoded_data,
        metadata: CompressionMetadata {
            num_features: bit_data.num_features,
            original_size: calculate_original_size(bit_data),
            chunk_size: bit_data.get_chunk_size(),
        },
        base_bit_positions,
    }
}

/// Return the constant base bits [usize], if any, from where entropy is zero
fn get_constant_base_bits(entropy: &[(usize, f64)]) -> Option<Vec<usize>> {
    let constant_bits: Vec<usize> = entropy
        .iter()
        .filter_map(|(idx, ent)| if *ent == 0.0 { Some(*idx) } else { None })
        .collect();

    if constant_bits.is_empty() {
        None
    } else {
        Some(constant_bits)
    }
}

/// Optimize base bit groups by iteratively adding bit positions based on entropy.
/// This modifies base_bit_groups in place to achieve the best compression ratio.
fn optimize_base_bit_groups(
    bit_data: &BitData,
    base_bit_groups: &mut BaseBitGroups,
    mut entropy: Vec<(usize, f64)>,
) {
    let tau = 10usize; // threshold for base selection
    let mut tau_count = 0usize; // count of non-improving additions
    let mut best_base_bit_groups = (*base_bit_groups).clone();
    let mut best_compressed_size = calculate_compressed_size(bit_data, &best_base_bit_groups);
    entropy.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut trial_base_bit_groups = (*base_bit_groups).clone();
    for (bit_position, _) in entropy.iter() {
        trial_base_bit_groups.add_bit_position(*bit_position, &bit_data);
        let trial_compressed_size = calculate_compressed_size(&bit_data, &trial_base_bit_groups);
        // debug print the trial compressed size vs best compressed size
        println!(
            "Trial bit position: {}, Trial compressed size: {}, Best compressed size: {}",
            bit_position, trial_compressed_size, best_compressed_size
        );
        if trial_compressed_size < best_compressed_size {
            best_compressed_size = trial_compressed_size;
            best_base_bit_groups = trial_base_bit_groups.clone();
            tau_count = 0;
        } else {
            tau_count += 1;
        }
        if tau_count >= tau {
            break;
        }
    }
    *base_bit_groups = best_base_bit_groups;
}

/// Encode the actual data using the base bit groups and base table
fn encode_data(bit_data: &BitData, base_bit_groups: &BaseBitGroups) -> DeviationData {
    let mut encoded_bit_stream = BitVec::<usize, Msb0>::new();
    let base_bit_positions = base_bit_groups.get_base_bit_positions().to_owned();
    let l_id = (base_bit_groups.get_num_bases() as f64).log2().ceil() as usize;
    let chunk_size = bit_data.get_chunk_size();
    let num_bits_per_base = base_bit_groups.get_num_bits_per_base();
    
    // Calculate deviation bits (all bits that are NOT base bits)
    let num_deviation_bits = if num_bits_per_base > chunk_size {
        0
    } else {
        chunk_size - num_bits_per_base
    };
    
    for (id, group) in base_bit_groups.get_groups().iter().enumerate() {
        let id_bits = id.view_bits::<Msb0>();
        let start = id_bits.len() - l_id;
        let id_bits_truncated = &id_bits[start..];
        for &row in group.iter() {
            let chunk = bit_data.get_chunk(row);
            // Encode deviation bits: all bits EXCEPT those at base_bit_positions
            for bit_pos in 0..chunk_size {
                if !base_bit_positions.contains(&bit_pos) {
                    encoded_bit_stream.push(chunk[bit_pos]);
                }
            }
            // Append the ID bits
            encoded_bit_stream.extend_from_bitslice(id_bits_truncated);
        }
    }
    
    DeviationData::new(
        encoded_bit_stream,
        bit_data.get_num_rows(),
        num_deviation_bits,
        l_id,
    )
}

/// Calculate the original uncompressed size in bits
fn calculate_original_size(bit_data: &BitData) -> usize {
    bit_data.get_num_rows() * bit_data.get_chunk_size()
}

/// Calculate the compressed size in bits
/// n_b * l_b + (n + m) * (l_d + l_id)+ m * l_w + S_params
/// for now, dont include the m, l_w and the params
fn calculate_compressed_size(bit_data: &BitData, base_bit_groups: &BaseBitGroups) -> usize {
    let n_b = base_bit_groups.get_num_bases(); // number of bases 
    let l_b = base_bit_groups.get_num_bits_per_base(); // bits per base
    let n = bit_data.get_num_rows(); // number of rows/samples
    let m = 0usize; // number of bases in condensed sample
    let chunk_size = bit_data.get_chunk_size();
    
    // Ensure l_b doesn't exceed chunk_size to prevent underflow
    let l_b = std::cmp::min(l_b, chunk_size);
    let l_d = chunk_size - l_b; // bits per deviation
    let l_id = (n_b as f64).log2().ceil() as usize; // bits per id
    let s_params = 0usize; // size of additional parameters in bits (not implemented yet)

    n_b * l_b + (n + m) * (l_d + l_id) + m * 0 + s_params
}

/// Decompress the compressed data back to BitData
pub fn decompress(compressed: &CompressedData) -> Result<BitData, String> {
    // Extract metadata
    let num_features = compressed.metadata.num_features;
    let chunk_size = compressed.metadata.chunk_size;
    let num_rows = compressed.encoded_data.get_num_samples();
    let bits_per_feature = chunk_size / num_features;
    let base_bit_positions = &compressed.base_bit_positions;
    
    // Reconstruct the bit data
    let mut reconstructed_bits = BitVec::<usize, Msb0>::with_capacity(chunk_size * num_rows);
    
    for sample_idx in 0..num_rows {
        // Get the sample using the new get_sample() method
        let sample = match compressed.encoded_data.get_sample(sample_idx) {
            Some(s) => s,
            None => {
                return Err(format!(
                    "Failed to retrieve sample at index {}",
                    sample_idx
                ));
            }
        };
        
        // Create a mutable chunk initialized to zeros
        let mut chunk = bitvec![usize, Msb0; 0; chunk_size];
        
        // Convert id bits to index for base_table lookup
        let mut base_id = 0usize;
        for i in 0..sample.id.len() {
            base_id = (base_id << 1) | (sample.id[i] as usize);
        }
        
        // Get the base pattern from the base table
        if base_id >= compressed.base_table.len() {
            return Err(format!(
                "Invalid base ID: {} (base table has {} entries)",
                base_id,
                compressed.base_table.len()
            ));
        }
        
        let base_pattern = &compressed.base_table[base_id].0;
        
        // First, fill all positions from the base pattern
        for bit_pos in 0..chunk_size.min(base_pattern.len()) {
            chunk.set(bit_pos, base_pattern[bit_pos]);
        }
        
        // Then, overwrite the non-base positions with the deviation bits
        let mut deviation_bit_idx = 0;
        for bit_pos in 0..chunk_size {
            if !base_bit_positions.contains(&bit_pos) {
                if deviation_bit_idx < sample.deviation.len() {
                    chunk.set(bit_pos, sample.deviation[deviation_bit_idx]);
                    deviation_bit_idx += 1;
                }
            }
        }
        
        reconstructed_bits.extend_from_bitslice(&chunk);
    }
    
    Ok(BitData {
        data: reconstructed_bits,
        bits_per_feature,
        num_features,
        chunk_size,
        num_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::BitData;
    use pretty_assertions::assert_eq;

    // Tests for get_constant_base_bits
    #[test]
    fn test_get_constant_base_bits_all_constant() {
        let entropy = vec![(0, 0.0), (1, 0.0), (2, 0.0)];
        let result = get_constant_base_bits(&entropy);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn test_get_constant_base_bits_some_constant() {
        let entropy = vec![(0, 0.5), (1, 0.0), (2, 1.2), (3, 0.0)];
        let result = get_constant_base_bits(&entropy);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![1, 3]);
    }

    #[test]
    fn test_get_constant_base_bits_none_constant() {
        let entropy = vec![(0, 0.5), (1, 0.3), (2, 1.2)];
        let result = get_constant_base_bits(&entropy);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_constant_base_bits_empty() {
        let entropy: Vec<(usize, f64)> = vec![];
        let result = get_constant_base_bits(&entropy);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_constant_base_bits_single_constant() {
        let entropy = vec![(5, 0.0)];
        let result = get_constant_base_bits(&entropy);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![5]);
    }

    #[test]
    fn test_get_constant_base_bits_single_non_constant() {
        let entropy = vec![(0, 1.5)];
        let result = get_constant_base_bits(&entropy);
        assert!(result.is_none());
    }

    // Helper function to create test BitData
    fn create_test_bit_data(
        num_rows: usize,
        bits_per_feature: usize,
        num_features: usize,
    ) -> BitData {
        let chunk_size = bits_per_feature * num_features;
        let total_bits = chunk_size * num_rows;

        let data = bitvec![usize, Msb0; 0; total_bits];

        BitData {
            data,
            bits_per_feature,
            num_features,
            chunk_size,
            num_rows,
        }
    }

    // Tests for calculate_original_size
    #[test]
    fn test_calculate_original_size_basic() {
        let bit_data = create_test_bit_data(100, 1, 8);
        let original_size = calculate_original_size(&bit_data);
        assert_eq!(original_size, 100 * 8);
    }

    #[test]
    fn test_calculate_original_size_single_row() {
        let bit_data = create_test_bit_data(1, 2, 8);
        let original_size = calculate_original_size(&bit_data);
        assert_eq!(original_size, 1 * 16);
    }

    #[test]
    fn test_calculate_original_size_large_data() {
        let bit_data = create_test_bit_data(10000, 2, 32);
        let original_size = calculate_original_size(&bit_data);
        assert_eq!(original_size, 10000 * 64);
    }

    #[test]
    fn test_calculate_original_size_zero_rows() {
        let bit_data = create_test_bit_data(0, 1, 8);
        let original_size = calculate_original_size(&bit_data);
        assert_eq!(original_size, 0);
    }

    #[test]
    fn test_calculate_original_size_single_bit() {
        let bit_data = create_test_bit_data(100, 1, 1);
        let original_size = calculate_original_size(&bit_data);
        assert_eq!(original_size, 100);
    }

    // Tests for calculate_compressed_size
    #[test]
    fn test_calculate_compressed_size_basic() {
        let bit_data = create_test_bit_data(100, 1, 32);
        let base_bit_groups = BaseBitGroups::new(&bit_data, None);
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0 (no bases), l_b = 0, n = 100, m = 0, l_d = 32, l_id = 0
        // Expected: 0 * 0 + (100 + 0) * (32 + 0) + 0 * 0 + 0 = 3200
        assert_eq!(compressed_size, 3200);
    }

    #[test]
    fn test_calculate_compressed_size_single_row() {
        let bit_data = create_test_bit_data(1, 2, 8);
        let base_bit_groups = BaseBitGroups::new(&bit_data, None);
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, l_b = 0, n = 1, l_d = 16, l_id = 0
        // Expected: 0 + (1 + 0) * (16 + 0) + 0 = 16
        assert_eq!(compressed_size, 16);
    }

    #[test]
    fn test_calculate_compressed_size_large_data() {
        let bit_data = create_test_bit_data(10000, 2, 32);
        let base_bit_groups = BaseBitGroups::new(&bit_data, None);
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, l_b = 0, n = 10000, l_d = 64, l_id = 0
        // Expected: 0 + (10000 + 0) * (64 + 0) + 0 = 640000
        assert_eq!(compressed_size, 640000);
    }

    #[test]
    fn test_calculate_compressed_size_with_constant_bits() {
        let bit_data = create_test_bit_data(100, 1, 32);
        let constant_bits = Some(vec![0, 1]); // 2 constant bits
        let base_bit_groups = BaseBitGroups::new(&bit_data, constant_bits);
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // With constant bits, we expect some bases to be created
        // The exact value depends on BaseBitGroups implementation, but should be > 0
        assert!(compressed_size > 0);
    }

    #[test]
    fn test_calculate_compressed_size_power_of_two_rows() {
        // Test with a power-of-2 number of rows to verify log2 calculation
        let bit_data = create_test_bit_data(16, 1, 32);
        let base_bit_groups = BaseBitGroups::new(&bit_data, None);
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, n = 16, l_d = 32, l_id = 0
        // Expected: 0 + (16 + 0) * (32 + 0) + 0 = 512
        assert_eq!(compressed_size, 512);
    }


    // Tests for compress function
    #[test]
    fn test_compress_basic() {
        // Use larger data to avoid optimization issues in edge cases
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = compress(&bit_data);

        // Check that we get a CompressedData struct with valid fields
        assert_eq!(compressed.metadata.num_features, 16);
        assert_eq!(compressed.metadata.original_size, 100 * 32);
        assert!(compressed.base_table.len() > 0);
        assert!(compressed.encoded_data.get_encoded_size() > 0);
    }

    #[test]
    fn test_compress_single_row() {
        let bit_data = create_test_bit_data(1, 2, 16);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features, 16);
        assert_eq!(compressed.metadata.original_size, 32);
        assert_eq!(compressed.encoded_data.get_num_samples(), 1);
    }

    #[test]
    fn test_compress_large_data() {
        let bit_data = create_test_bit_data(1000, 2, 32);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features, 32);
        assert_eq!(compressed.metadata.original_size, 1000 * 64);
        assert_eq!(compressed.encoded_data.get_num_samples(), 1000);
    }

    #[test]
    fn test_compress_preserves_sample_count() {
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.encoded_data.get_num_samples(), 100);
    }

    #[test]
    fn test_compress_metadata_accuracy() {
        let bit_data = create_test_bit_data(50, 2, 8);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features, 8);
        assert_eq!(compressed.metadata.original_size, 50 * 16);
    }

    #[test]
    fn test_compress_encoded_data_structure() {
        let bit_data = create_test_bit_data(20, 2, 16);
        let compressed = compress(&bit_data);

        // Verify that encoded data contains all samples
        assert_eq!(compressed.encoded_data.get_num_samples(), 20);
        
        // Verify each sample can be retrieved
        for i in 0..20 {
            let sample = compressed.encoded_data.get_sample(i);
            assert!(sample.is_some());
        }
    }

    #[test]
    fn test_compress_invalid_sample_index() {
        let bit_data = create_test_bit_data(10, 2, 16);
        let compressed = compress(&bit_data);

        // Verify that invalid indices return None
        assert!(compressed.encoded_data.get_sample(10).is_none());
        assert!(compressed.encoded_data.get_sample(100).is_none());
    }

    #[test]
    fn test_compress_base_table_non_empty() {
        let bit_data = create_test_bit_data(50, 2, 32);
        let compressed = compress(&bit_data);

        // Base table should contain entries for each base group
        assert!(compressed.base_table.len() > 0);
        
        // Each base table entry should have a count > 0
        for (_base, count) in &compressed.base_table {
            assert!(*count > 0);
        }
    }

    #[test]
    fn test_compress_varying_row_counts() {
        // Test with different numbers of rows
        let row_counts = vec![1, 5, 10, 50, 100, 500];

        for num_rows in row_counts {
            let bit_data = create_test_bit_data(num_rows, 2, 16);
            let compressed = compress(&bit_data);

            assert_eq!(compressed.metadata.num_features, 16);
            assert_eq!(
                compressed.metadata.original_size,
                num_rows * 32
            );
            assert_eq!(compressed.encoded_data.get_num_samples(), num_rows);
        }
    }

    #[test]
    fn test_compress_encoded_data_retrieval() {
        let bit_data = create_test_bit_data(20, 2, 16);
        let compressed = compress(&bit_data);

        // Verify we can retrieve samples and they have expected structure
        for i in 0..20 {
            let sample = compressed.encoded_data.get_sample(i);
            assert!(sample.is_some());
            let sample_bits = sample.unwrap();
            // Sample should contain deviation bits + id bits
            assert!(sample_bits.deviation.len() + sample_bits.id.len() > 0);
        }
    }

    #[test]
    fn test_compress_consistency() {
        // Compress the same data twice and verify results are consistent
        let bit_data = create_test_bit_data(50, 2, 16);
        
        let compressed1 = compress(&bit_data);
        let compressed2 = compress(&bit_data);

        assert_eq!(
            compressed1.metadata.original_size,
            compressed2.metadata.original_size
        );
        assert_eq!(
            compressed1.encoded_data.get_num_samples(),
            compressed2.encoded_data.get_num_samples()
        );
        assert_eq!(
            compressed1.base_table.len(),
            compressed2.base_table.len()
        );
    }

    #[test]
    fn test_compress_power_of_two_rows() {
        // Test with power-of-2 number of rows to verify log2 calculations
        let powers_of_two = vec![1, 2, 4, 8, 16, 32, 64];

        for num_rows in powers_of_two {
            let bit_data = create_test_bit_data(num_rows, 2, 16);
            let compressed = compress(&bit_data);

            assert_eq!(compressed.encoded_data.get_num_samples(), num_rows);
            assert_eq!(compressed.metadata.original_size, num_rows * 32);
        }
    }

    #[test]
    fn test_compress_small_chunks() {
        // Test with 8-bit chunks (1 byte per feature, 1 feature)
        let bit_data = create_test_bit_data(100, 1, 8);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features, 8);
        assert_eq!(compressed.metadata.original_size, 100 * 8);
        assert_eq!(compressed.encoded_data.get_num_samples(), 100);
    }

    #[test]
    fn test_compress_medium_chunks() {
        // Test with 16-bit chunks
        let bit_data = create_test_bit_data(100, 1, 16);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features, 16);
        assert_eq!(compressed.metadata.original_size, 100 * 16);
        assert_eq!(compressed.encoded_data.get_num_samples(), 100);
    }

    #[test]
    fn test_compress_large_chunks() {
        // Test with 32-bit chunks
        let bit_data = create_test_bit_data(100, 1, 32);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features, 32);
        assert_eq!(compressed.metadata.original_size, 100 * 32);
        assert_eq!(compressed.encoded_data.get_num_samples(), 100);
    }

    #[test]
    fn test_deviation_data_out_of_bounds() {
        let bit_data = create_test_bit_data(50, 2, 16);
        let compressed = compress(&bit_data);

        // Test boundary conditions for sample retrieval
        let last_valid = compressed.encoded_data.get_sample(49);
        assert!(last_valid.is_some());

        let first_invalid = compressed.encoded_data.get_sample(50);
        assert!(first_invalid.is_none());
    }

    #[test]
    fn test_compressed_data_structure() {
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = compress(&bit_data);

        // Verify CompressedData has all required fields
        assert!(compressed.base_table.len() > 0);
        assert!(compressed.encoded_data.get_num_samples() > 0);
        assert_eq!(compressed.metadata.num_features, 16);
        assert!(compressed.metadata.original_size > 0);
    }

    fn create_simulated_bit_data(
        num_rows: usize,
        bits_per_feature: usize,
        num_features: usize,
    ) -> BitData {
        let chunk_size = bits_per_feature * num_features;
        let total_bits = chunk_size * num_rows;
        let mut data = BitVec::<usize, Msb0>::with_capacity(total_bits);

        // Create simulated data with patterns
        for row in 0..num_rows {
            for bit_pos in 0..chunk_size {
                // Create a pattern: alternating bits with some structure
                // For even rows: bit pattern based on position
                // For odd rows: inverted pattern
                let bit = if row % 2 == 0 {
                    (bit_pos % 3 == 0) || (bit_pos % 5 == 0)
                } else {
                    !((bit_pos % 3 == 0) || (bit_pos % 5 == 0))
                };
                data.push(bit);
            }
        }

        BitData {
            data,
            bits_per_feature,
            num_features,
            chunk_size,
            num_rows,
        }
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        // Create test bit data with zeros
        let bit_data = create_test_bit_data(50, 2, 16);
        
        // Compress the data
        let compressed = compress(&bit_data);
        
        // Decompress the data
        let decompressed = decompress(&compressed).expect("Decompression failed");
        
        // Verify the decompressed data has the same structure as the original
        assert_eq!(decompressed.num_features, bit_data.num_features);
        assert_eq!(decompressed.num_rows, bit_data.num_rows);
        assert_eq!(decompressed.chunk_size, bit_data.chunk_size);
        assert_eq!(decompressed.bits_per_feature, bit_data.bits_per_feature);
        assert_eq!(decompressed.data.len(), bit_data.data.len());
    }

    #[test]
    fn test_compress_decompress_with_simulated_data_debug() {
        // Create a very simple test case with minimal data
        let bit_data = create_simulated_bit_data(5, 1, 8); // 5 rows, 8-bit chunks
        
        println!("\n=== INPUT DATA ===");
        println!("Num rows: {}", bit_data.num_rows);
        println!("Chunk size: {}", bit_data.chunk_size);
        println!("Total bits: {}", bit_data.data.len());
        for row in 0..bit_data.num_rows {
            let start = row * bit_data.chunk_size;
            let end = start + bit_data.chunk_size;
            let chunk: Vec<u8> = bit_data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            println!("Row {}: {:?}", row, chunk);
        }
        
        // Compress the data
        println!("\n=== COMPRESSING ===");
        let compressed = compress(&bit_data);
        
        println!("Base table entries: {}", compressed.base_table.len());
        println!("Encoded bit stream size: {}", compressed.encoded_data.get_encoded_size());
        println!("Num samples: {}", compressed.encoded_data.get_num_samples());
        println!("Num deviation bits: {}", compressed.encoded_data.get_num_deviation_bits());
        println!("Num ID bits: {}", compressed.encoded_data.get_num_id_bits());
        
        // Decompress the data
        println!("\n=== DECOMPRESSING ===");
        let decompressed = decompress(&compressed).expect("Decompression failed");
        
        println!("Decompressed num rows: {}", decompressed.num_rows);
        println!("Decompressed chunk size: {}", decompressed.chunk_size);
        
        println!("\n=== COMPARING ===");
        // Check row by row
        for row in 0..bit_data.num_rows {
            let start = row * bit_data.chunk_size;
            let end = start + bit_data.chunk_size;
            
            let original_chunk: Vec<u8> = bit_data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            let decompressed_chunk: Vec<u8> = decompressed.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            
            if original_chunk == decompressed_chunk {
                println!("Row {}: OK", row);
            } else {
                println!("Row {} MISMATCH:", row);
                println!("  Original:     {:?}", original_chunk);
                println!("  Decompressed: {:?}", decompressed_chunk);
            }
        }
        
        // Verify the decompressed data matches the original exactly
        assert_eq!(decompressed.data, bit_data.data, "Decompressed data does not match original");
    }
}
