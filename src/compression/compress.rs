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
}

#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    pub num_features: usize,
    pub original_size: usize,
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

    pub fn get_sample(&self, sample_idx: usize) -> Option<BitVec<usize, Msb0>> {
        if sample_idx >= self.num_samples {
            return None;
        }
        let start_bit = sample_idx * (self.num_deviation_bits + self.num_id_bits);
        let end_bit = start_bit + self.num_deviation_bits + self.num_id_bits;
        Some(self.encoded_bit_stream[start_bit..end_bit].to_bitvec())
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

    CompressedData {
        base_table,
        encoded_data,
        metadata: CompressionMetadata {
            num_features: bit_data.num_features,
            original_size: calculate_original_size(bit_data),
        },
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
    let base_bit_mask = base_bit_groups.get_base_bit_mask();
    for group in base_bit_groups.get_groups().iter() {
        for &row in group.iter() {
            let chunk = bit_data.get_chunk(row);
            
        }
    }
    
    todo!("Implement the encoding logic here");
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
    let l_d = bit_data.get_chunk_size() - l_b; // bits per deviation
    let l_id = (n_b as f64).log2().ceil() as usize; // bits per id
    let s_params = 0usize; // size of additional parameters in bits (not implemented yet)

    n_b * l_b + (n + m) * (l_d + l_id) + m * 0 + s_params
}

/// Decompress the compressed data back to BitData
pub fn decompress(compressed: &CompressedData) -> Result<BitData, String> {
    // TODO: Reverse the compression process
    Err("Decompression not yet implemented".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::BitData;

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

        let mut data = bitvec![usize, Msb0; 0; total_bits];

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

    #[test]
    fn test_build_base_table_empty() {
        let bit_data = create_test_bit_data(0, 1, 8);
        let mut base_bit_groups = BaseBitGroups::new(&bit_data, None);
        let entropy = vec![];
        let base_table = build_base_table(&bit_data, &mut base_bit_groups, entropy);
        assert!(base_table.is_empty());
    }

    #[test]
    fn test_build_base_table_no_start_bases() {
        let bit_data = create_test_bit_data(100, 1, 8);
        let mut base_bit_groups = BaseBitGroups::new(&bit_data, None);
        let entropy = vec![
            (0, 1.0),
            (1, 1.0),
            (2, 1.0),
            (3, 1.0),
            (4, 1.0),
            (5, 1.0),
            (6, 1.0),
            (7, 1.0),
        ];
        let base_table = build_base_table(&bit_data, &mut base_bit_groups, entropy);

        println!("Base Table: {:?}", base_table);
        assert!(vec![(bitvec![usize, Msb0; 0; 8], 100)] == base_table);
    }

    #[test]
    fn test_build_base_table() {
        // Create a simple BitData for testing
        let num_rows = 22;
        let chunk_size = 8;
        let num_features = 8;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, false, true, true, false, // Row 0
            true, true, true, true, true, true, false, false, // Row 1
            true, true, true, false, true, false, true, true, // Row 2
            true, false, true, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false, // Row 5
            true, false, true, false, false, true, true, false, // Row 6
            true, true, true, true, true, true, false, false, // Row 7
            true, true, true, false, true, false, true, true, // Row 8
            true, false, true, false, true, true, false, true, // Row 9
            true, true, false, true, true, true, false, false, // Row 10
            true, false, true, false, false, true, true, false, // Row 11
            true, true, true, true, true, true, false, false, // Row 12
            true, true, true, false, true, false, true, true, // Row 13
            true, false, true, false, true, true, false, true, // Row 14
            true, true, false, true, true, true, false, false, // Row 15
            true, false, true, false, false, true, true, false, // Row 16
            true, true, true, true, true, true, false, false, // Row 17
            true, true, true, false, true, false, true, true, // Row 18
            true, false, true, false, true, true, false, true, // Row 19
            false, true, false, false, true, true, false, true, // Row 20
            true, true, false, true, true, true, false, false, // Row 21
        ];
        let bit_data = BitData {
            data: data.into_iter().collect(),
            bits_per_feature: bits_per_feature,
            num_features: num_features,
            num_rows,
            chunk_size,
        };
        let mut base_bit_groups = BaseBitGroups::new(&bit_data, None);
        let entropy = entropy::calculate_entropy(&bit_data);
        let base_table = build_base_table(&bit_data, &mut base_bit_groups, entropy);
        println!("bit mask: {:?}", base_bit_groups.get_base_bit_mask());
        println!("Base Table: {:?}", base_table);
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        // TODO: Test that compress/decompress is lossless
    }
}
