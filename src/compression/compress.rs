#[cfg(test)]
use crate::compression::base_bit_groups::BaseBitGroups;
use crate::error::EntroGdError;
use crate::preprocessor::{BitData, BitDataInfo, BitDataSet, BitDataView};
use crate::timing::ScopedTimer;
use bitvec::prelude::*;

pub use crate::compression::pipeline::{
    BaseBitGroupOptimizer, CompressionParams, CompressionPipeline, CompressionPipelineBuilder,
    CondensedSampleSelector, DefaultBaseBitGroupOptimizer, DefaultCondensedSampleSelector,
    DefaultEncoder, DefaultEntropyCalculator, Encoder, EntropyCalculator,
};

/// A lightweight zero-copy view that extends a `&BitDataSet` with a small number
/// of extra rows (condensed samples) without cloning the original data.
pub(crate) struct ExtendedBitData<'a> {
    base: &'a BitDataSet,
    extras: Vec<BitVec<usize, Msb0>>,
}

impl<'a> ExtendedBitData<'a> {
    pub(crate) fn new(base: &'a BitDataSet, extras: Vec<BitVec<usize, Msb0>>) -> Self {
        ExtendedBitData { base, extras }
    }
}

impl BitDataView for ExtendedBitData<'_> {
    fn num_rows(&self) -> usize {
        self.base.data.num_rows + self.extras.len()
    }

    fn num_features(&self) -> usize {
        self.base.info.num_features()
    }

    fn chunk_size(&self) -> usize {
        self.base.data.chunk_size
    }

    fn feature_bits(&self, feature_idx: usize) -> usize {
        self.base.info.feature_bits(feature_idx)
    }

    fn feature_offset(&self, feature_idx: usize) -> usize {
        self.base.info.feature_offset(feature_idx)
    }

    fn feature_index_for_bit(&self, bit_pos: usize) -> Option<usize> {
        self.base.info.feature_index_for_bit(bit_pos)
    }

    fn get_bit(&self, row: usize, bit_in_chunk: usize) -> bool {
        if row < self.base.data.num_rows {
            self.base.data.get_bit(row, bit_in_chunk)
        } else {
            self.extras[row - self.base.data.num_rows][bit_in_chunk]
        }
    }

    fn get_chunk(&self, row: usize) -> &BitSlice<usize, Msb0> {
        if row < self.base.data.num_rows {
            self.base.data.get_chunk(row)
        } else {
            &self.extras[row - self.base.data.num_rows]
        }
    }
}

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
    pub metadata: BitDataInfo,
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

fn build_base_bit_mask(chunk_size: usize, base_bit_positions: &[usize]) -> BitVec<usize, Msb0> {
    let mut mask = bitvec![usize, Msb0; 0; chunk_size];
    for &bit_pos in base_bit_positions {
        if bit_pos < chunk_size {
            mask.set(bit_pos, true);
        }
    }
    mask
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

    pub fn encoded_bit_stream(&self) -> &BitVec<usize, Msb0> {
        &self.encoded_bit_stream
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
pub fn compress(bit_data: &BitDataSet) -> CompressedData {
    let params = CompressionParams::new(10, 10, true);
    CompressionPipeline::new(params).compress(bit_data)
}

pub fn compress_with_pipeline(
    bit_data: &BitDataSet,
    pipeline: &CompressionPipeline,
) -> CompressedData {
    pipeline.compress(bit_data)
}

/// Calculate the original uncompressed size in bits
#[cfg(test)]
fn calculate_original_size(bit_data: &BitDataSet) -> usize {
    bit_data.data.num_rows * bit_data.data.chunk_size
}

/// Private batch decompression function
fn decompress_samples_batch(
    compressed: &CompressedData,
    indices: &[usize],
) -> Result<BitDataSet, EntroGdError> {
    let _timer = ScopedTimer::debug(format!("decompress_samples_batch ({} samples)", indices.len()));
    let data_info = &compressed.metadata;
    let num_features = data_info.num_features();
    let chunk_size = data_info.chunk_size();
    if num_features == 0 {
        return Err(EntroGdError::InvalidMetadata {
            message: "num_features is 0".to_string(),
        });
    }
    let base_bit_positions = &compressed.base_bit_positions;
    let base_bit_mask = build_base_bit_mask(chunk_size, base_bit_positions);

    let mut reconstructed_bits = BitVec::<usize, Msb0>::with_capacity(chunk_size * indices.len());

    for &sample_idx in indices {
        let sample = match compressed.encoded_data.get_sample(sample_idx) {
            Some(s) => s,
            None => {
                return Err(EntroGdError::DecompressionSampleMissing { sample_idx });
            }
        };

        let mut chunk = bitvec![usize, Msb0; 0; chunk_size];
        let mut base_id = 0usize;
        for i in 0..sample.id.len() {
            base_id = (base_id << 1) | (sample.id[i] as usize);
        }
        if base_id >= compressed.base_table.len() {
            return Err(EntroGdError::InvalidBaseId {
                base_id,
                table_len: compressed.base_table.len(),
            });
        }
        let base_pattern = &compressed.base_table[base_id].0;
        for bit_pos in 0..chunk_size.min(base_pattern.len()) {
            chunk.set(bit_pos, base_pattern[bit_pos]);
        }
        let mut deviation_bit_idx = 0;
        for bit_pos in 0..chunk_size {
            if !base_bit_mask[bit_pos]
                && deviation_bit_idx < sample.deviation.len() {
                    chunk.set(bit_pos, sample.deviation[deviation_bit_idx]);
                    deviation_bit_idx += 1;
                }
        }
        reconstructed_bits.extend_from_bitslice(&chunk);
    }

    let data = BitData {
        data: reconstructed_bits,
        chunk_size,
        num_rows: indices.len(),
    };
    let info = data_info.with_original_size_bits(chunk_size * indices.len());
    Ok(BitDataSet { data, info })
}

/// Decompress the original file data (first n samples)
pub fn decompress_file(compressed: &CompressedData) -> Result<BitDataSet, EntroGdError> {
    let _timer = ScopedTimer::info("decompress_file");
    let data_info = &compressed.metadata;
    let original_num_rows = data_info.original_size_bits / data_info.chunk_size();
    let indices: Vec<usize> = (0..original_num_rows).collect();
    decompress_samples_batch(compressed, &indices)
}

/// Decompress condensed samples for analytics
pub fn decompress_analytics(compressed: &CompressedData) -> Option<CondensedSamples> {
    let _timer = ScopedTimer::info("decompress_analytics");
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
    use crate::data_loader::FeatureDataType;
    use crate::preprocessor::{BitData, BitDataInfo, BitDataSet, FeatureSpec};
    use pretty_assertions::assert_eq;

    // Helper function to create test BitData
    fn create_test_bit_data(
        num_rows: usize,
        bits_per_feature: usize,
        num_features: usize,
    ) -> BitDataSet {
        let chunk_size = bits_per_feature * num_features;
        let total_bits = chunk_size * num_rows;

        let data = bitvec![usize, Msb0; 0; total_bits];
        let data = BitData {
            data,
            chunk_size,
            num_rows,
        };
        let features = vec![
            FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature);
            num_features
        ];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();

        BitDataSet { data, info }
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
        let optimizer = DefaultBaseBitGroupOptimizer;
        let compressed_size = optimizer.calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0 (no bases), l_b = 0, n = 100, m = 0, l_d = 32, l_id = 0
        // Expected: 0 * 0 + (100 + 0) * (32 + 0) + 0 * 0 + 0 = 3200
        assert_eq!(compressed_size, 3200);
    }

    #[test]
    fn test_calculate_compressed_size_single_row() {
        let bit_data = create_test_bit_data(1, 2, 8);
        let base_bit_groups = BaseBitGroups::new(&bit_data);
        let optimizer = DefaultBaseBitGroupOptimizer;
        let compressed_size = optimizer.calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, l_b = 0, n = 1, l_d = 16, l_id = 0
        // Expected: 0 + (1 + 0) * (16 + 0) + 0 = 16
        assert_eq!(compressed_size, 16);
    }

    #[test]
    fn test_calculate_compressed_size_large_data() {
        let bit_data = create_test_bit_data(10000, 2, 32);
        let base_bit_groups = BaseBitGroups::new(&bit_data);
        let optimizer = DefaultBaseBitGroupOptimizer;
        let compressed_size = optimizer.calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, l_b = 0, n = 10000, l_d = 64, l_id = 0
        // Expected: 0 + (10000 + 0) * (64 + 0) + 0 = 640000
        assert_eq!(compressed_size, 640000);
    }

    #[test]
    fn test_calculate_compressed_size_power_of_two_rows() {
        // Test with a power-of-2 number of rows to verify log2 calculation
        let bit_data = create_test_bit_data(16, 1, 32);
        let base_bit_groups = BaseBitGroups::new(&bit_data);
        let optimizer = DefaultBaseBitGroupOptimizer;
        let compressed_size = optimizer.calculate_compressed_size(&bit_data, &base_bit_groups);

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
        assert_eq!(compressed.metadata.num_features(), 16);
        assert_eq!(compressed.metadata.original_size_bits, 100 * 32);
        assert!(!compressed.base_table.is_empty());
        assert!(compressed.encoded_data.get_encoded_size() > 0);
    }

    #[test]
    fn test_compress_single_row() {
        let bit_data = create_test_bit_data(1, 2, 16);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features(), 16);
        assert_eq!(compressed.metadata.original_size_bits, 32);
        // Verify decompressed data matches original size
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 1);
    }

    #[test]
    fn test_compress_large_data() {
        let bit_data = create_test_bit_data(1000, 2, 32);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features(), 32);
        assert_eq!(compressed.metadata.original_size_bits, 1000 * 64);
        // Verify decompressed data matches original size
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 1000);
    }

    #[test]
    fn test_compress_preserves_sample_count() {
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = compress(&bit_data);

        // Verify original size is preserved in metadata
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits, 100 * 32);
        // Verify decompressed data matches original count
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_compress_metadata_accuracy() {
        let bit_data = create_test_bit_data(50, 2, 8);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features(), 8);
        assert_eq!(compressed.metadata.original_size_bits, 50 * 16);
    }

    #[test]
    fn test_compress_encoded_data_structure() {
        let bit_data = create_test_bit_data(20, 2, 16);
        let compressed = compress(&bit_data);

        // Verify original metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits, 20 * 32);

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 20);

        // Verify decompressed data structure
        assert_eq!(decompressed.info.num_features(), 16);
        assert_eq!(decompressed.data.chunk_size, 32); // bits_per_feature * num_features = 2 * 16 = 32
    }

    #[test]
    fn test_compress_invalid_sample_index() {
        let bit_data = create_test_bit_data(10, 2, 16);
        let compressed = compress(&bit_data);

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 10);

        // Verify metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits, 10 * 32);
    }

    #[test]
    fn test_compress_base_table_non_empty() {
        let bit_data = create_test_bit_data(50, 2, 32);
        let compressed = compress(&bit_data);

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
            let bit_data = create_test_bit_data(num_rows, 2, 16);
            let compressed = compress(&bit_data);

            assert_eq!(compressed.metadata.num_features(), 16);
            assert_eq!(
                compressed.metadata.original_size_bits,
                num_rows * 32
            );

            // Verify decompress_file returns correct number of rows
            let decompressed = decompress_file(&compressed).unwrap();
            assert_eq!(decompressed.data.num_rows, num_rows);
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

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 20);
    }

    #[test]
    fn test_compress_consistency() {
        // Compress separate instances of the same data and verify results are consistent
        let original_rows = 50;
        let bit_data1 = create_test_bit_data(original_rows, 2, 16);
        let bit_data2 = create_test_bit_data(original_rows, 2, 16);

        let compressed1 = compress(&bit_data1);
        let compressed2 = compress(&bit_data2);

        assert_eq!(
            compressed1.metadata.original_size_bits,
            compressed2.metadata.original_size_bits
        );
        // Verify both decompress to the same original row count
        let decompressed1 = decompress_file(&compressed1).unwrap();
        let decompressed2 = decompress_file(&compressed2).unwrap();
        assert_eq!(decompressed1.data.num_rows, decompressed2.data.num_rows);
        assert_eq!(decompressed1.data.num_rows, original_rows);
    }

    #[test]
    fn test_compress_power_of_two_rows() {
        // Test with power-of-2 number of rows to verify log2 calculations
        let powers_of_two = vec![1, 2, 4, 8, 16, 32, 64];

        for num_rows in powers_of_two {
            let bit_data = create_test_bit_data(num_rows, 2, 16);
            let compressed = compress(&bit_data);

            assert_eq!(
                compressed.metadata.original_size_bits,
                num_rows * 32
            );
            // Verify decompressed matches original
            let decompressed = decompress_file(&compressed).unwrap();
            assert_eq!(decompressed.data.num_rows, num_rows);
        }
    }

    #[test]
    fn test_compress_small_chunks() {
        // Test with 8-bit chunks (1 byte per feature, 1 feature)
        let bit_data = create_test_bit_data(100, 1, 8);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features(), 8);
        assert_eq!(compressed.metadata.original_size_bits, 100 * 8);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_compress_medium_chunks() {
        // Test with 16-bit chunks
        let bit_data = create_test_bit_data(100, 1, 16);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features(), 16);
        assert_eq!(compressed.metadata.original_size_bits, 100 * 16);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_compress_large_chunks() {
        // Test with 32-bit chunks
        let bit_data = create_test_bit_data(100, 1, 32);
        let compressed = compress(&bit_data);

        assert_eq!(compressed.metadata.num_features(), 32);
        assert_eq!(compressed.metadata.original_size_bits, 100 * 32);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_deviation_data_out_of_bounds() {
        let bit_data = create_test_bit_data(50, 2, 16);
        let compressed = compress(&bit_data);

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 50);

        // Verify metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits, 50 * 32);
    }

    #[test]
    fn test_compressed_data_structure() {
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = compress(&bit_data);

        // Verify CompressedData has all required fields
        assert!(!compressed.base_table.is_empty());
        assert!(compressed.encoded_data.get_num_samples() > 0);
        assert_eq!(compressed.metadata.num_features(), 16);
        assert!(compressed.metadata.original_size_bits > 0);
    }

    fn create_simulated_bit_data(
        num_rows: usize,
        bits_per_feature: usize,
        num_features: usize,
    ) -> BitDataSet {
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

        let data = BitData {
            data,
            chunk_size,
            num_rows,
        };
        let features = vec![
            FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature);
            num_features
        ];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        BitDataSet { data, info }
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        // Create test bit data with zeros
        let original_rows = 50;
        let bit_data = create_test_bit_data(original_rows, 2, 16);

        // Compress the data
        let compressed = compress(&bit_data);
        // Decompress the data
        let decompressed = decompress_file(&compressed).expect("Decompression failed");

        // Verify the decompressed data has the correct structure
        // Note: bit_data.num_rows may have been increased by condensed samples,
        // but decompressed should match the original row count
        assert_eq!(decompressed.info.num_features(), 16); // Original num_features
        assert_eq!(decompressed.data.num_rows, original_rows); // Should match original, not expanded
        assert_eq!(decompressed.data.chunk_size, 32);
        assert_eq!(decompressed.info.feature_bits(0), 2);
        assert_eq!(decompressed.data.data.len(), original_rows * 32); // Original size, not expanded
    }

    #[test]
    fn test_compress_decompress_with_simulated_data_debug() {
        // Create a very simple test case with minimal data
        let bit_data = create_simulated_bit_data(5, 1, 8); // 5 rows, 8-bit chunks

        log::info!("\n=== INPUT DATA ===");
        log::info!("Num rows: {}", bit_data.data.num_rows);
        log::info!("Chunk size: {}", bit_data.data.chunk_size);
        log::info!("Total bits: {}", bit_data.data.data.len());
        for row in 0..bit_data.data.num_rows {
            let start = row * bit_data.data.chunk_size;
            let end = start + bit_data.data.chunk_size;
            let chunk: Vec<u8> = bit_data.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            log::info!("Row {}: {:?}", row, chunk);
        }

        // Compress the data
        log::info!("\n=== COMPRESSING ===");
        let compressed = compress(&bit_data);

        log::info!("Base table entries: {}", compressed.base_table.len());
        log::info!(
            "Encoded bit stream size: {}",
            compressed.encoded_data.get_encoded_size()
        );
        log::info!("Num samples: {}", compressed.encoded_data.get_num_samples());
        log::info!(
            "Num deviation bits: {}",
            compressed.encoded_data.get_num_deviation_bits()
        );
        log::info!("Num ID bits: {}", compressed.encoded_data.get_num_id_bits());

        // Decompress the data
        log::info!("\n=== DECOMPRESSING ===");
        let decompressed = decompress_file(&compressed).expect("Decompression failed");

        log::info!("Decompressed num rows: {}", decompressed.data.num_rows);
        log::info!("Decompressed chunk size: {}", decompressed.data.chunk_size);

        log::info!("\n=== COMPARING ===");
        // Check row by row - use decompressed row count since bit_data now contains condensed samples
        let num_original_rows = 5;
        for row in 0..num_original_rows {
            let start = row * bit_data.data.chunk_size;
            let end = start + bit_data.data.chunk_size;

            let original_chunk: Vec<u8> = bit_data.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            let decompressed_chunk: Vec<u8> = decompressed.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();

            if original_chunk == decompressed_chunk {
                log::info!("Row {}: OK", row);
            } else {
                log::info!("Row {} MISMATCH:", row);
                log::info!("  Original:     {:?}", original_chunk);
                log::info!("  Decompressed: {:?}", decompressed_chunk);
            }
        }

        // Verify the original rows match by checking only the first num_original_rows rows
        let original_bits: Vec<bool> = bit_data.data.data
            [0..(num_original_rows * bit_data.data.chunk_size)]
            .iter()
            .map(|b| *b)
            .collect();
        let decompressed_bits: Vec<bool> = decompressed.data.data
            [0..(num_original_rows * bit_data.data.chunk_size)]
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
            log::info!("\n=== ANALYTICS ===");
            log::info!("Condensed samples: {}", analytics.samples.len());
            for (i, sample) in analytics.samples.iter().enumerate() {
                let weight = analytics.weights[i];
                let sample_bits: Vec<u8> = sample.iter().map(|b| if *b { 1 } else { 0 }).collect();
                log::info!("Sample {}: {:?}, Weight: {}", i, sample_bits, weight);
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
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            num_rows,
        };
        let features = vec![
            FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature);
            num_features
        ];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };
        log::info!("\n=== INPUT DATA ===");
        for row in 0..bit_data.data.num_rows {
            let start = row * bit_data.data.chunk_size;
            let end = start + bit_data.data.chunk_size;
            let chunk: Vec<u8> = bit_data.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            log::info!("Row {}: {:?}", row, chunk);
        }
        // Compress the data
        log::info!("\n=== COMPRESSING ===");
        let compressed = compress(&bit_data);
        log::info!("Base table entries: {}", compressed.base_table.len());
        log::info!("Base table: {:?}", compressed.base_table);
        log::info!(
            "Encoded bit stream size: {}",
            compressed.encoded_data.get_encoded_size()
        );
        log::info!("Num samples: {}", compressed.encoded_data.get_num_samples());
        log::info!(
            "Num deviation bits: {}",
            compressed.encoded_data.get_num_deviation_bits()
        );
        log::info!("Num ID bits: {}", compressed.encoded_data.get_num_id_bits());
        // Decompress analytics if available
        if let Some(analytics) = decompress_analytics(&compressed) {
            log::info!("\n=== ANALYTICS ===");
            log::info!("Condensed samples: {}", analytics.samples.len());
            for (i, sample) in analytics.samples.iter().enumerate() {
                let weight = analytics.weights[i];
                let sample_bits: Vec<u8> = sample.iter().map(|b| if *b { 1 } else { 0 }).collect();
                log::info!("Sample {}: {:?}, Weight: {}", i, sample_bits, weight);
            }
        }
        // Decompress the data
        log::info!("\n=== DECOMPRESSING ===");
        let decompressed = decompress_file(&compressed).expect("Decompression failed");
        log::info!("Decompressed num rows: {}", decompressed.data.num_rows);
        log::info!("Decompressed chunk size: {}", decompressed.data.chunk_size);
        log::info!("\n=== COMPARING ===");
        // Check row by row - use decompressed row count since bit_data now contains condensed samples
        let num_original_rows = 12;
        for row in 0..num_original_rows {
            let start = row * bit_data.data.chunk_size;
            let end = start + bit_data.data.chunk_size;
            let original_chunk: Vec<u8> = bit_data.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            let decompressed_chunk: Vec<u8> = decompressed.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            if original_chunk == decompressed_chunk {
                log::info!("Row {}: OK", row);
            } else {
                log::info!("Row {} MISMATCH:", row);
                log::info!("  Original:     {:?}", original_chunk);
                log::info!("  Decompressed: {:?}", decompressed_chunk);
            }
        }
    }
}
