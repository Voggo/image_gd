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
    pub encoded_data: BitVec<usize, Msb0>,
    /// Metadata for decompression (column count, base bits used, etc.)
    pub metadata: CompressionMetadata,
}

#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    pub num_features: usize,
    pub original_size: usize,
}

/// Main compression function - transforms BitData into CompressedData
pub fn compress(bit_data: &BitData) -> CompressedData {
    let entropy = entropy::calculate_entropy(bit_data);
    let constant_base_bits = get_constant_base_bits(&entropy);
    let base_bit_groups = BaseBitGroups::new(bit_data, constant_base_bits);

    let base_table = build_base_table(&bit_data, &base_bit_groups, entropy);
    let encoded_data = encode_data(bit_data, &base_bit_groups, &base_table);

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

/// Build the base table from base bit groups
fn build_base_table(bit_data: &BitData, base_bit_groups: &BaseBitGroups, mut entropy: Vec<(usize, f64)>) -> Vec<(BitVec<usize, Msb0>, usize)> {
    let tau = 10usize; // threshold for base selection
    let mut tau_count = 0usize; // count of non-improving additions
    let mut best_base_bit_groups = (*base_bit_groups).clone();
    let mut best_compressed_size = calculate_compressed_size(bit_data, &best_base_bit_groups);
    entropy.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (bit_position, _) in entropy.iter() {
        let mut trial_base_bit_groups = (*base_bit_groups).clone();
        trial_base_bit_groups.add_bit_position(*bit_position, &bit_data);
        let trial_compressed_size = calculate_compressed_size(&bit_data, &trial_base_bit_groups);
        if trial_compressed_size < best_compressed_size {
            best_compressed_size = trial_compressed_size;
            best_base_bit_groups = trial_base_bit_groups;
            tau_count = 0;
        } else {
            tau_count += 1;
        }
        if tau_count >= tau {
            break;
        }
    }
    best_base_bit_groups.get_bases(bit_data)
}

/// Encode the actual data using the base bit groups and base table
fn encode_data(
    bit_data: &BitData,
    base_bit_groups: &BaseBitGroups,
    base_table: &[(BitVec<usize, Msb0>, usize)],
) -> BitVec<usize, Msb0> {
    // TODO: Encode each row by:
    // 1. Extracting the base bits pattern
    // 2. Looking up the pattern in the base table
    // 3. Encoding the deviation from the base pattern
    BitVec::new()
}

/// Calculate the original uncompressed size in bits
fn calculate_original_size(bit_data: &BitData) -> usize {
    bit_data.get_num_rows() * bit_data.get_chunk_size()
}

/// Calculate the compressed size in bits
/// n_b * l_b + (n + m) * (l_d + l_id)+ m * l_w + S_params
/// for now, dont include the m, l_w and the params
fn calculate_compressed_size(bit_data: &BitData, base_bit_groups: &BaseBitGroups) -> usize {
    let n_b = base_bit_groups.get_num_bases();
    let l_b = base_bit_groups.get_num_bases(); // bits per base
    let n = bit_data.get_num_rows();
    let m = 0usize; // number of bases in condensed sample
    let l_d = bit_data.get_chunk_size() - l_b; // bits per deviation
    let l_id = (n as f64).log2().ceil() as usize * n_b; // bits per id
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
    fn create_test_bit_data(num_rows: usize, bits_per_feature: usize, num_features: usize) -> BitData {
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
        let base_bit_groups = BaseBitGroups::new(&bit_data, None);
        let entropy = vec![];
        let base_table = build_base_table(&bit_data, &base_bit_groups, entropy);
        assert!(base_table.is_empty());
    }

    #[test]
    fn test_build_base_table_no_bases() {
        let bit_data = create_test_bit_data(100, 1, 8);
        let base_bit_groups = BaseBitGroups::new(&bit_data, None);
        let entropy = vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0)];
        let base_table = build_base_table(&bit_data, &base_bit_groups, entropy);

        println!("Base Table: {:?}", base_table);

    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        // TODO: Test that compress/decompress is lossless
    }
}
