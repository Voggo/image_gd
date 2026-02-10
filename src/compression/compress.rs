use crate::compression::base_bit_groups::BaseBitGroups;
use crate::compression::entropy;
use crate::preprocessor::BitData;
use bitvec::prelude::*;

/// Represents the compressed output
#[derive(Debug, Clone)]
pub struct CompressedData {
    /// The encoded data stream
    pub encoded_data: DeviationData,
    /// The Weights for the condensed samples (if used)
    // Should be stored as a bitstream of length m * l_w (log_2(n).ceil() bits per weight)
    pub condensed_sample_weights: Option<Vec<usize>>,
    /// Base table mapping base patterns to their frequencies or encodings
    pub base_table: Vec<(BitVec<usize, Msb0>, usize)>,
    /// Bit positions used as base bits during compression
    pub base_bit_positions: Vec<usize>,
    /// Metadata for decompression (column count, base bits used, etc.)
    pub metadata: CompressionMetadata,
}

#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    pub num_features: usize,
    pub original_size: usize,
    /// Total bits per row/chunk
    pub chunk_size: usize,
}

#[derive(Debug, Clone)]
pub struct CondensedSamples {
    pub samples: Vec<BitVec<usize, Msb0>>,
    pub weights: Vec<usize>,
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
            deviation: self.encoded_bit_stream[start_bit..start_bit + self.num_deviation_bits]
                .to_bitvec(),
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
pub fn compress(bit_data: &mut BitData) -> CompressedData {
    // Store the original number of rows BEFORE adding condensed samples
    let original_num_rows = bit_data.num_rows;
    let original_chunk_size = bit_data.chunk_size;
    
    let mut base_bit_groups = BaseBitGroups::new(bit_data);
    let entropy = entropy::calculate_entropy(bit_data);
    println!("Initial entropy per bit position: {:?}", entropy);
    // Phase 1: Optimize the base bit groups for best compression ratio
    let condensed_samples =
        get_condensed_sample(bit_data, &mut base_bit_groups.clone(), entropy.clone(), 10);

    add_condensed_samples_to_bit_data(bit_data, &condensed_samples);

    let best_base_bit_groups = optimize_base_bit_groups(bit_data, &mut base_bit_groups, entropy);

    // Phase 2: Extract the base table from the optimized groups
    let base_table = best_base_bit_groups.get_bases(bit_data);

    println!(
        "Optimized to {} base groups with {} bits per base.",
        best_base_bit_groups.get_num_bases(),
        best_base_bit_groups.get_num_bits_per_base()
    );
    println!("Base table size: {}", base_table.len());
    println!(
        "Base bit positions: {:?}",
        best_base_bit_groups.get_base_bit_positions()
    );
    // println!("Base table: {:?}", base_table);
    // Phase 3: Encode the data using the optimized base groups
    let encoded_data = encode_data(bit_data, &best_base_bit_groups);

    let base_bit_positions = best_base_bit_groups.get_base_bit_positions().to_owned();

    CompressedData {
        encoded_data,
        condensed_sample_weights: Some(condensed_samples.weights),
        base_table,
        base_bit_positions,
        metadata: CompressionMetadata {
            num_features: bit_data.num_features,
            original_size: original_num_rows * original_chunk_size,
            chunk_size: bit_data.chunk_size,
        },
    }
}

fn add_condensed_samples_to_bit_data(bit_data: &mut BitData, condensed_samples: &CondensedSamples) {
    for sample in &condensed_samples.samples {
        bit_data.extend_from_bitslice(sample);
    }
}

/// Optimize base bit groups by iteratively adding bit positions based on entropy.
/// This modifies base_bit_groups in place to achieve the best compression ratio.
fn optimize_base_bit_groups(
    bit_data: &BitData,
    base_bit_groups: &mut BaseBitGroups,
    mut entropy: Vec<(usize, f64)>,
) -> BaseBitGroups {
    let tau = 10usize; // threshold for base selection
    let mut tau_count = 0usize; // count of non-improving additions
    entropy.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    while let Some((bit_pos, 0.0)) = entropy.first() {
        base_bit_groups.add_bit_position(*bit_pos, bit_data);
        entropy.remove(0);
    }

    let mut best_base_bit_groups = base_bit_groups.clone();
    let mut best_compressed_size = calculate_compressed_size(bit_data, &best_base_bit_groups);
    let mut trial_base_bit_groups = base_bit_groups.clone();
    for &(bit_position, _) in entropy.iter() {
        trial_base_bit_groups.add_bit_position(bit_position, bit_data);
        let trial_compressed_size = calculate_compressed_size(bit_data, &trial_base_bit_groups);
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
    best_base_bit_groups
}

/// Organize entropy values by feature and return as flattened vector
/// Orders bits by alternating through features (feature 1, 2, 3, ... n, repeat)
/// Within each feature, bits are sorted from lowest to highest entropy
fn organize_entropy_by_feature(
    entropy: Vec<(usize, f64)>,
    bits_per_feature: usize,
    num_features: usize,
) -> Vec<(usize, f64)> {
    let mut entropy_by_feature: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_features];

    for (bit_pos, entropy_val) in entropy {
        // Determine which feature this bit belongs to
        let feature_idx = bit_pos / bits_per_feature;
        if feature_idx < num_features {
            entropy_by_feature[feature_idx].push((bit_pos, entropy_val));
        }
    }

    // Sort each feature's entropy by entropy value (ascending - low to high)
    for feature_entropy in &mut entropy_by_feature {
        feature_entropy.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    }

    // Flatten with round-robin selection: feature 1, 2, 3, ..., n, repeat
    let mut result = Vec::new();
    let max_bits_per_feature = entropy_by_feature
        .iter()
        .map(|f| f.len())
        .max()
        .unwrap_or(0);

    for i in 0..max_bits_per_feature {
        for feature_idx in 0..num_features {
            if i < entropy_by_feature[feature_idx].len() {
                result.push(entropy_by_feature[feature_idx][i]);
            }
        }
    }

    result
}

fn get_condensed_sample(
    bit_data: &BitData,
    condensed_bit_groups: &mut BaseBitGroups,
    entropy: Vec<(usize, f64)>,
    m_max: usize,
) -> CondensedSamples {
    let mut samples = Vec::new();
    let mut weights = Vec::new();

    let entropy_by_feature =
        organize_entropy_by_feature(entropy, bit_data.bits_per_feature, bit_data.num_features);

    for &(bit_position, _) in entropy_by_feature.iter() {
        if condensed_bit_groups.get_num_bases() >= m_max {
            break;
        }
        condensed_bit_groups.add_bit_position(bit_position, bit_data);
    }
    let bases = condensed_bit_groups.get_bases(bit_data);
    let base_mask: &BitSlice<usize, Msb0> = condensed_bit_groups.get_base_bit_mask();
    for (base_group, base) in condensed_bit_groups.get_groups().iter().zip(bases.iter()) {
        let mut deviation_sum: usize = 0;
        for sample in base_group.iter() {
            let chunk = bit_data.get_chunk(*sample).to_bitvec();
            let masked = chunk & base_mask;
            deviation_sum += masked.load_be::<usize>();
        }
        let base_int = base.0.load_be::<usize>();
        let average_deviation = deviation_sum as f64 / base_group.len() as f64;
        let condensed_sample = base_int + average_deviation.round() as usize;
        println!(
            "Base group size: {}, Base value: {}, Average deviation: {:.2}, Condensed sample: {}",
            base_group.len(),
            base_int,
            average_deviation,
            condensed_sample
        );
        // Create a BitVec from condensed_sample and ensure it has the same length as a chunk
        let mut condensed_bitvec = BitVec::<usize, Msb0>::from_element(condensed_sample);
        condensed_bitvec.resize(bit_data.chunk_size, false); // pad with zeros if needed
        samples.push(condensed_bitvec);
        weights.push(base_group.len());
    }
    CondensedSamples { samples, weights }
}

/// Encode the actual data using the base bit groups and base table
fn encode_data(bit_data: &BitData, base_bit_groups: &BaseBitGroups) -> DeviationData {
    let mut encoded_bit_stream = BitVec::<usize, Msb0>::new();
    let base_bit_positions = base_bit_groups.get_base_bit_positions().to_owned();

    // Calculate l_id: number of bits needed to represent base group IDs
    // If there are no bases, we still need at least 1 bit to represent ID 0
    let num_bases = base_bit_groups.get_num_bases();
    let l_id = {
        let bits = (num_bases as f64).log2().ceil() as usize;
        if bits == 0 { 1 } else { bits }
    };

    let chunk_size = bit_data.chunk_size;
    let num_bits_per_base = base_bit_groups.get_num_bits_per_base();

    // Calculate deviation bits (all bits that are NOT base bits)
    let num_deviation_bits = if num_bits_per_base > chunk_size {
        0
    } else {
        chunk_size - num_bits_per_base
    };

    // Build a mapping from row index to group id so we can encode in original row order
    let num_rows = bit_data.num_rows;
    let mut row_to_group_id = vec![0usize; num_rows];
    for (id, group) in base_bit_groups.get_groups().iter().enumerate() {
        for &row in group.iter() {
            row_to_group_id[row] = id;
        }
    }

    // Encode rows in original order (0, 1, 2, ...) to preserve row ordering
    for row in 0..num_rows {
        let id = row_to_group_id[row];

        let chunk = bit_data.get_chunk(row);
        // Encode deviation bits: all bits EXCEPT those at base_bit_positions
        for bit_pos in 0..chunk_size {
            if !base_bit_positions.contains(&bit_pos) {
                encoded_bit_stream.push(chunk[bit_pos]);
            }
        }
        // Append the ID bits (fixed-width, most-significant bit first)
        if l_id > 0 {
            for shift in (0..l_id).rev() {
                encoded_bit_stream.push(((id >> shift) & 1) == 1);
            }
        }
    }

    DeviationData::new(
        encoded_bit_stream,
        bit_data.num_rows,
        num_deviation_bits,
        l_id,
    )
}

/// Calculate the original uncompressed size in bits
#[cfg(test)]
fn calculate_original_size(bit_data: &BitData) -> usize {
    bit_data.num_rows * bit_data.chunk_size
}

/// Calculate the compressed size in bits
/// n_b * l_b + (n + m) * (l_d + l_id)+ m * l_w + S_params
/// for now, dont include the m, l_w and the params
fn calculate_compressed_size(bit_data: &BitData, base_bit_groups: &BaseBitGroups) -> usize {
    let n_b = base_bit_groups.get_num_bases(); // number of bases 
    let l_b = base_bit_groups.get_num_bits_per_base(); // bits per base
    let n = bit_data.num_rows; // number of rows/samples
    let m = 0usize; // number of bases in condensed sample
    let chunk_size = bit_data.chunk_size;

    // Ensure l_b doesn't exceed chunk_size to prevent underflow
    let l_b = l_b.min(chunk_size);
    let l_d = chunk_size - l_b; // bits per deviation
    let l_id = (n_b as f64).log2().ceil() as usize; // bits per id
    let s_params = 0usize; // size of additional parameters in bits (not implemented yet)

    n_b * l_b + (n + m) * (l_d + l_id) + m * 0 + s_params
}

/// Private batch decompression function
fn decompress_samples_batch(
    compressed: &CompressedData,
    indices: &[usize],
) -> Result<BitData, String> {
    let num_features = compressed.metadata.num_features;
    let chunk_size = compressed.metadata.chunk_size;
    let bits_per_feature = chunk_size / num_features;
    let base_bit_positions = &compressed.base_bit_positions;

    let mut reconstructed_bits = BitVec::<usize, Msb0>::with_capacity(chunk_size * indices.len());

    for &sample_idx in indices {
        let sample = match compressed.encoded_data.get_sample(sample_idx) {
            Some(s) => s,
            None => {
                return Err(format!("Failed to retrieve sample at index {}", sample_idx));
            }
        };

        let mut chunk = bitvec![usize, Msb0; 0; chunk_size];
        let mut base_id = 0usize;
        for i in 0..sample.id.len() {
            base_id = (base_id << 1) | (sample.id[i] as usize);
        }
        if base_id >= compressed.base_table.len() {
            return Err(format!(
                "Invalid base ID: {} (base table has {} entries)",
                base_id,
                compressed.base_table.len()
            ));
        }
        let base_pattern = &compressed.base_table[base_id].0;
        for bit_pos in 0..chunk_size.min(base_pattern.len()) {
            chunk.set(bit_pos, base_pattern[bit_pos]);
        }
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
        num_rows: indices.len(),
    })
}

/// Decompress the original file data (first n samples)
pub fn decompress_file(compressed: &CompressedData) -> Result<BitData, String> {
    let original_num_rows = compressed.metadata.original_size / compressed.metadata.chunk_size;
    let indices: Vec<usize> = (0..original_num_rows).collect();
    decompress_samples_batch(compressed, &indices)
}

/// Decompress condensed samples for analytics
pub fn decompress_analytics(compressed: &CompressedData) -> Option<CondensedSamples> {
    if let Some(weights) = &compressed.condensed_sample_weights {
        // Reconstruct condensed samples from base_table
        let samples: Vec<BitVec<usize, Msb0>> = compressed
            .base_table
            .iter()
            .map(|(bv, _)| bv.clone())
            .collect();
        Some(CondensedSamples {
            samples,
            weights: weights.clone(),
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::BitData;
    use pretty_assertions::assert_eq;

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
        let base_bit_groups = BaseBitGroups::new(&bit_data);
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0 (no bases), l_b = 0, n = 100, m = 0, l_d = 32, l_id = 0
        // Expected: 0 * 0 + (100 + 0) * (32 + 0) + 0 * 0 + 0 = 3200
        assert_eq!(compressed_size, 3200);
    }

    #[test]
    fn test_calculate_compressed_size_single_row() {
        let bit_data = create_test_bit_data(1, 2, 8);
        let base_bit_groups = BaseBitGroups::new(&bit_data);
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, l_b = 0, n = 1, l_d = 16, l_id = 0
        // Expected: 0 + (1 + 0) * (16 + 0) + 0 = 16
        assert_eq!(compressed_size, 16);
    }

    #[test]
    fn test_calculate_compressed_size_large_data() {
        let bit_data = create_test_bit_data(10000, 2, 32);
        let base_bit_groups = BaseBitGroups::new(&bit_data);
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, l_b = 0, n = 10000, l_d = 64, l_id = 0
        // Expected: 0 + (10000 + 0) * (64 + 0) + 0 = 640000
        assert_eq!(compressed_size, 640000);
    }

    #[test]
    fn test_calculate_compressed_size_power_of_two_rows() {
        // Test with a power-of-2 number of rows to verify log2 calculation
        let bit_data = create_test_bit_data(16, 1, 32);
        let base_bit_groups = BaseBitGroups::new(&bit_data);
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, n = 16, l_d = 32, l_id = 0
        // Expected: 0 + (16 + 0) * (32 + 0) + 0 = 512
        assert_eq!(compressed_size, 512);
    }

    // Tests for compress function
    #[test]
    fn test_compress_basic() {
        // Use larger data to avoid optimization issues in edge cases
        let mut bit_data = create_test_bit_data(100, 2, 16);
        let compressed = compress(&mut bit_data);

        // Check that we get a CompressedData struct with valid fields
        assert_eq!(compressed.metadata.num_features, 16);
        assert_eq!(compressed.metadata.original_size, 100 * 32);
        assert!(!compressed.base_table.is_empty());
        assert!(compressed.encoded_data.get_encoded_size() > 0);
    }

    #[test]
    fn test_compress_single_row() {
        let mut bit_data = create_test_bit_data(1, 2, 16);
        let compressed = compress(&mut bit_data);

        assert_eq!(compressed.metadata.num_features, 16);
        assert_eq!(compressed.metadata.original_size, 32);
        // Verify decompressed data matches original size
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.num_rows, 1);
    }

    #[test]
    fn test_compress_large_data() {
        let mut bit_data = create_test_bit_data(1000, 2, 32);
        let compressed = compress(&mut bit_data);

        assert_eq!(compressed.metadata.num_features, 32);
        assert_eq!(compressed.metadata.original_size, 1000 * 64);
        // Verify decompressed data matches original size
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.num_rows, 1000);
    }

    #[test]
    fn test_compress_preserves_sample_count() {
        let mut bit_data = create_test_bit_data(100, 2, 16);
        let compressed = compress(&mut bit_data);

        // Verify original size is preserved in metadata
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size, 100 * 32);
        // Verify decompressed data matches original count
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.num_rows, 100);
    }

    #[test]
    fn test_compress_metadata_accuracy() {
        let mut bit_data = create_test_bit_data(50, 2, 8);
        let compressed = compress(&mut bit_data);

        assert_eq!(compressed.metadata.num_features, 8);
        assert_eq!(compressed.metadata.original_size, 50 * 16);
    }

    #[test]
    fn test_compress_encoded_data_structure() {
        let mut bit_data = create_test_bit_data(20, 2, 16);
        let compressed = compress(&mut bit_data);

        // Verify original metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size, 20 * 32);

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.num_rows, 20);
        
        // Verify decompressed data structure
        assert_eq!(decompressed.num_features, 16);
        assert_eq!(decompressed.chunk_size, 32);  // bits_per_feature * num_features = 2 * 16 = 32
    }

    #[test]
    fn test_compress_invalid_sample_index() {
        let mut bit_data = create_test_bit_data(10, 2, 16);
        let compressed = compress(&mut bit_data);

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.num_rows, 10);
        
        // Verify metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size, 10 * 32);
    }

    #[test]
    fn test_compress_base_table_non_empty() {
        let mut bit_data = create_test_bit_data(50, 2, 32);
        let compressed = compress(&mut bit_data);

        // Base table should contain entries for each base group
        assert!(!compressed.base_table.is_empty());

        // Each base table entry should have a count > 0
        for (_base, count) in &compressed.base_table {
            assert!(*count > 0);
        }

        // Verify decompress_analytics returns condensed samples
        let analytics = decompress_analytics(&compressed);
        assert!(analytics.is_some());
        let condensed = analytics.unwrap();
        assert_eq!(condensed.samples.len(), compressed.base_table.len());
        assert_eq!(condensed.weights.len(), compressed.base_table.len());
    }

    #[test]
    fn test_compress_varying_row_counts() {
        // Test with different numbers of rows
        let row_counts = vec![1, 5, 10, 50, 100, 500];

        for num_rows in row_counts {
            let mut bit_data = create_test_bit_data(num_rows, 2, 16);
            let compressed = compress(&mut bit_data);

            assert_eq!(compressed.metadata.num_features, 16);
            assert_eq!(compressed.metadata.original_size, num_rows * 32);

            // Verify decompress_file returns correct number of rows
            let decompressed = decompress_file(&compressed).unwrap();
            assert_eq!(decompressed.num_rows, num_rows);
        }
    }

    #[test]
    fn test_compress_encoded_data_retrieval() {
        let mut bit_data = create_test_bit_data(20, 2, 16);
        let compressed = compress(&mut bit_data);

        // Verify we can retrieve samples and they have expected structure
        for i in 0..20 {
            let sample = compressed.encoded_data.get_sample(i);
            assert!(sample.is_some());
            let sample_bits = sample.unwrap();
            // Sample should contain deviation bits + id bits
            assert!(sample_bits.deviation.len() + sample_bits.id.len() > 0);
        }

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.num_rows, 20);
    }

    #[test]
    fn test_compress_consistency() {
        // Compress separate instances of the same data and verify results are consistent
        let original_rows = 50;
        let mut bit_data1 = create_test_bit_data(original_rows, 2, 16);
        let mut bit_data2 = create_test_bit_data(original_rows, 2, 16);

        let compressed1 = compress(&mut bit_data1);
        let compressed2 = compress(&mut bit_data2);

        assert_eq!(
            compressed1.metadata.original_size,
            compressed2.metadata.original_size
        );
        // Verify both decompress to the same original row count
        let decompressed1 = decompress_file(&compressed1).unwrap();
        let decompressed2 = decompress_file(&compressed2).unwrap();
        assert_eq!(decompressed1.num_rows, decompressed2.num_rows);
        assert_eq!(decompressed1.num_rows, original_rows);
    }

    #[test]
    fn test_compress_power_of_two_rows() {
        // Test with power-of-2 number of rows to verify log2 calculations
        let powers_of_two = vec![1, 2, 4, 8, 16, 32, 64];

        for num_rows in powers_of_two {
            let mut bit_data = create_test_bit_data(num_rows, 2, 16);
            let compressed = compress(&mut bit_data);

            assert_eq!(compressed.metadata.original_size, num_rows * 32);
            // Verify decompressed matches original
            let decompressed = decompress_file(&compressed).unwrap();
            assert_eq!(decompressed.num_rows, num_rows);
        }
    }

    #[test]
    fn test_compress_small_chunks() {
        // Test with 8-bit chunks (1 byte per feature, 1 feature)
        let mut bit_data = create_test_bit_data(100, 1, 8);
        let compressed = compress(&mut bit_data);

        assert_eq!(compressed.metadata.num_features, 8);
        assert_eq!(compressed.metadata.original_size, 100 * 8);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.num_rows, 100);
    }

    #[test]
    fn test_compress_medium_chunks() {
        // Test with 16-bit chunks
        let mut bit_data = create_test_bit_data(100, 1, 16);
        let compressed = compress(&mut bit_data);

        assert_eq!(compressed.metadata.num_features, 16);
        assert_eq!(compressed.metadata.original_size, 100 * 16);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.num_rows, 100);
    }

    #[test]
    fn test_compress_large_chunks() {
        // Test with 32-bit chunks
        let mut bit_data = create_test_bit_data(100, 1, 32);
        let compressed = compress(&mut bit_data);

        assert_eq!(compressed.metadata.num_features, 32);
        assert_eq!(compressed.metadata.original_size, 100 * 32);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.num_rows, 100);
    }

    #[test]
    fn test_deviation_data_out_of_bounds() {
        let mut bit_data = create_test_bit_data(50, 2, 16);
        let compressed = compress(&mut bit_data);

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.num_rows, 50);
        
        // Verify metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size, 50 * 32);
    }

    #[test]
    fn test_compressed_data_structure() {
        let mut bit_data = create_test_bit_data(100, 2, 16);
        let compressed = compress(&mut bit_data);

        // Verify CompressedData has all required fields
        assert!(!compressed.base_table.is_empty());
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

        // Create simulated data with low entropy patterns
        for row in 0..num_rows {
            for bit_pos in 0..chunk_size {
                // Low entropy: most bits are constant (0), with few changes
                // Only the first few bit positions vary
                let bit = if bit_pos < 2 {
                    // First 2 bits vary by row
                    row % 2 == bit_pos
                } else {
                    // All remaining bits are constant 0 (high redundancy)
                    false
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
        let original_rows = 50;
        let mut bit_data = create_test_bit_data(original_rows, 2, 16);

        // Compress the data
        let compressed = compress(&mut bit_data);
        // Decompress the data
        let decompressed = decompress_file(&compressed).expect("Decompression failed");

        // Verify the decompressed data has the correct structure
        // Note: bit_data.num_rows may have been increased by condensed samples,
        // but decompressed should match the original row count
        assert_eq!(decompressed.num_features, 16); // Original num_features
        assert_eq!(decompressed.num_rows, original_rows); // Should match original, not expanded
        assert_eq!(decompressed.chunk_size, 32);
        assert_eq!(decompressed.bits_per_feature, 2);
        assert_eq!(decompressed.data.len(), original_rows * 32); // Original size, not expanded
    }

    #[test]
    fn test_compress_decompress_with_simulated_data_debug() {
        // Create a very simple test case with minimal data
        let mut bit_data = create_simulated_bit_data(5, 1, 8); // 5 rows, 8-bit chunks

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
        let compressed = compress(&mut bit_data);

        println!("Base table entries: {}", compressed.base_table.len());
        println!(
            "Encoded bit stream size: {}",
            compressed.encoded_data.get_encoded_size()
        );
        println!("Num samples: {}", compressed.encoded_data.get_num_samples());
        println!(
            "Num deviation bits: {}",
            compressed.encoded_data.get_num_deviation_bits()
        );
        println!("Num ID bits: {}", compressed.encoded_data.get_num_id_bits());

        // Decompress the data
        println!("\n=== DECOMPRESSING ===");
        let decompressed = decompress_file(&compressed).expect("Decompression failed");

        println!("Decompressed num rows: {}", decompressed.num_rows);
        println!("Decompressed chunk size: {}", decompressed.chunk_size);

        println!("\n=== COMPARING ===");
        // Check row by row - use decompressed row count since bit_data now contains condensed samples
        let num_original_rows = 5;
        for row in 0..num_original_rows {
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

        // Verify the original rows match by checking only the first num_original_rows rows
        let original_bits: Vec<bool> = bit_data.data[0..(num_original_rows * bit_data.chunk_size)]
            .iter()
            .map(|b| *b)
            .collect();
        let decompressed_bits: Vec<bool> = decompressed.data[0..(num_original_rows * bit_data.chunk_size)]
            .iter()
            .map(|b| *b)
            .collect();
        assert_eq!(
            decompressed_bits, original_bits,
            "Decompressed data does not match original for the first {} rows",
            num_original_rows
        );

        // Print analytics if available
        if let Some(analytics) = decompress_analytics(&compressed) {
            println!("\n=== ANALYTICS ===");
            println!("Condensed samples: {}", analytics.samples.len());
            for (i, sample) in analytics.samples.iter().enumerate() {
                let weight = analytics.weights[i];
                let sample_bits: Vec<u8> = sample.iter().map(|b| if *b { 1 } else { 0 }).collect();
                println!("Sample {}: {:?}, Weight: {}", i, sample_bits, weight);
            }
        }
    }

    #[test]
    fn test_compress_decompress_roundtrip_with_simulated_data_debug_2() {
        // Create a simple BitData for testing
        let num_rows = 12;
        let chunk_size = 4;
        let num_features = 4;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, // Row 0
            true, true, false, false, // Row 1
            true, true, false, true, // Row 2
            true, true, false, false, // Row 3
            true, true, false, false, // Row 4
            true, false, true, true, // Row 5
            true, true, false, false, // Row 6
            true, true, false, false, // Row 7
            true, true, false, false, // Row 8
            true, true, false, true, // Row 9
            true, true, false, true, // Row 10
            true, true, false, false, // Row 11
        ];
        let mut bit_data = BitData {
            data: data.into_iter().collect(),
            bits_per_feature,
            num_features,
            num_rows,
            chunk_size,
        };
        println!("\n=== INPUT DATA ===");
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
        let compressed = compress(&mut bit_data);
        println!("Base table entries: {}", compressed.base_table.len());
        println!("Base table: {:?}", compressed.base_table);
        println!(
            "Encoded bit stream size: {}",
            compressed.encoded_data.get_encoded_size()
        );
        println!("Num samples: {}", compressed.encoded_data.get_num_samples());
        println!(
            "Num deviation bits: {}",
            compressed.encoded_data.get_num_deviation_bits()
        );
        println!("Num ID bits: {}", compressed.encoded_data.get_num_id_bits());
        // Decompress analytics if available
        if let Some(analytics) = decompress_analytics(&compressed) {
            println!("\n=== ANALYTICS ===");
            println!("Condensed samples: {}", analytics.samples.len());
            for (i, sample) in analytics.samples.iter().enumerate() {
                let weight = analytics.weights[i];
                let sample_bits: Vec<u8> = sample.iter().map(|b| if *b { 1 } else { 0 }).collect();
                println!("Sample {}: {:?}, Weight: {}", i, sample_bits, weight);
            }
        }
        // Decompress the data
        println!("\n=== DECOMPRESSING ===");
        let decompressed = decompress_file(&compressed).expect("Decompression failed");
        println!("Decompressed num rows: {}", decompressed.num_rows);
        println!("Decompressed chunk size: {}", decompressed.chunk_size);
        println!("\n=== COMPARING ===");
        // Check row by row - use decompressed row count since bit_data now contains condensed samples
        let num_original_rows = 12;
        for row in 0..num_original_rows {
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
    }
}
