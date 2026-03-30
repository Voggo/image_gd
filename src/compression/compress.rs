mod huffman;
mod rle;

pub(crate) use self::huffman::build_huffman_code_map;
pub(crate) use self::rle::{RLE_LONG_MAX, RLE_SHORT_MAX, RLE_TERMINATOR_PAYLOAD};
use crate::compression::tabular_preprocessor::{BitDataInfo, BitDataReconstructionInfo};
use crate::error::EntroGdError;
use bitvec::prelude::*;
use fxhash::FxHashMap;

pub(crate) fn huffman_row_layout(
    metadata: &BitDataInfo,
) -> Result<(usize, usize, usize), EntroGdError> {
    let chunk_size = metadata.chunk_size();
    if chunk_size == 0 {
        return Err(EntroGdError::InvalidMetadata {
            message: "chunk_size is 0".to_string(),
        });
    }

    let original_num_samples = metadata.original_size_bits() / chunk_size;
    match metadata.reconstruction {
        BitDataReconstructionInfo::Image(info) => {
            let row_count = info.height as usize;
            let pixel_grouping = info.pixel_grouping as usize;
            if pixel_grouping == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "image pixel_grouping must be > 0".to_string(),
                });
            }
            let row_width = (info.width as usize).div_ceil(pixel_grouping);
            if row_count == 0 && original_num_samples == 0 {
                return Ok((0, 0, 0));
            }
            if row_count == 0 || row_width == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "invalid image row layout width={} height={} pixel_grouping={}",
                        info.width, info.height, info.pixel_grouping
                    ),
                });
            }
            let expected_samples =
                row_count
                    .checked_mul(row_width)
                    .ok_or_else(|| EntroGdError::InvalidMetadata {
                        message: "image row layout overflows sample count".to_string(),
                    })?;
            if expected_samples != original_num_samples {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "image row layout mismatch: height * ceil(width / pixel_grouping) = {}, original samples = {}",
                        expected_samples, original_num_samples
                    ),
                });
            }
            Ok((original_num_samples, row_count, row_width))
        }
        BitDataReconstructionInfo::Tabular => Ok((original_num_samples, original_num_samples, 1)),
    }
}

/// Represents the compressed output
#[derive(Debug, Clone)]
pub struct CompressedData {
    /// The encoded data stream
    pub encoded_data: EncodedData,
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

impl CompressedData {
    pub fn new(encoded_data: EncodedData, metadata: BitDataInfo) -> Self {
        CompressedData {
            encoded_data,
            condensed_sample_weights: None,
            base_table: Vec::new(),
            base_bit_positions: Vec::new(),
            metadata,
        }
    }
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

pub(crate) struct DeviationSampleRef<'a> {
    pub deviation: &'a BitSlice<usize, Msb0>,
    pub id: &'a BitSlice<usize, Msb0>,
}

#[derive(Debug, Clone)]
pub struct DeviationData {
    encoded_bit_stream: BitVec<usize, Msb0>,
    num_samples: usize,
    num_deviation_bits: usize,
    num_id_bits: usize,
}

#[derive(Debug, Clone)]
pub struct RleDeviationData {
    symbol_bit_stream: BitVec<usize, Msb0>,
    rm_values: Vec<(u8, u8)>,
    rm_control_stream: BitVec<usize, Msb0>,
    num_samples: usize,
    num_deviation_bits: usize,
    num_id_bits: usize,
}

#[derive(Debug, Clone)]
pub struct HuffmanDeviationData {
    pixel_bit_stream: BitVec<usize, Msb0>,
    raw_deviation_bit_stream: Option<BitVec<usize, Msb0>>,
    canonical_symbols: Vec<u64>,
    canonical_code_lengths: Vec<u8>,
    row_offsets: Vec<u32>,
    num_samples: usize,
    original_num_samples: usize,
    num_deviation_bits: usize,
    huffman_symbol_num_deviation_bits: usize,
    num_id_bits: usize,
    row_width: usize,
    max_code_length: u8,
    decode_by_length: Vec<FxHashMap<u32, u64>>,
    original_stream_end_offset_bits: usize,
}

#[derive(Debug, Clone)]
pub enum EncodedData {
    Normal(DeviationData),
    Rle(RleDeviationData),
    Huffman(HuffmanDeviationData),
}

pub(crate) fn build_base_bit_mask(
    chunk_size: usize,
    base_bit_positions: &[usize],
) -> BitVec<usize, Msb0> {
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

    pub(crate) fn for_each_sample(
        &self,
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        let sample_width = self.num_deviation_bits + self.num_id_bits;

        for sample_idx in 0..self.num_samples {
            let start_bit = sample_idx * sample_width;
            let end_bit = start_bit + sample_width;
            f(DeviationSampleRef {
                deviation: &self.encoded_bit_stream[start_bit..start_bit + self.num_deviation_bits],
                id: &self.encoded_bit_stream[start_bit + self.num_deviation_bits..end_bit],
            })?;
        }

        Ok(())
    }
}

impl EncodedData {
    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        match self {
            EncodedData::Normal(data) => data.get_sample(sample_idx),
            EncodedData::Rle(data) => data.get_sample(sample_idx),
            EncodedData::Huffman(data) => data.get_sample(sample_idx),
        }
    }

    pub fn get_encoded_size(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_encoded_size(),
            EncodedData::Rle(data) => data.get_encoded_size(),
            EncodedData::Huffman(data) => data.get_encoded_size(),
        }
    }

    pub fn encoded_bit_stream(&self) -> &BitVec<usize, Msb0> {
        match self {
            EncodedData::Normal(data) => data.encoded_bit_stream(),
            EncodedData::Rle(data) => data.symbol_bit_stream(),
            EncodedData::Huffman(data) => data.pixel_bit_stream(),
        }
    }

    pub fn get_num_samples(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_num_samples(),
            EncodedData::Rle(data) => data.get_num_samples(),
            EncodedData::Huffman(data) => data.get_num_samples(),
        }
    }

    pub fn get_num_deviation_bits(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_num_deviation_bits(),
            EncodedData::Rle(data) => data.get_num_deviation_bits(),
            EncodedData::Huffman(data) => data.get_num_deviation_bits(),
        }
    }

    pub fn get_num_id_bits(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_num_id_bits(),
            EncodedData::Rle(data) => data.get_num_id_bits(),
            EncodedData::Huffman(data) => data.get_num_id_bits(),
        }
    }

    pub(crate) fn for_each_sample(
        &self,
        f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        match self {
            EncodedData::Normal(data) => data.for_each_sample(f),
            EncodedData::Rle(data) => data.for_each_sample(f),
            EncodedData::Huffman(data) => data.for_each_sample(f),
        }
    }

    pub fn to_raw_deviation_data(&self) -> DeviationData {
        match self {
            EncodedData::Normal(data) => data.clone(),
            EncodedData::Rle(data) => data
                .to_deviation_data()
                .expect("RLE encoded data should be valid when converting to raw deviation data"),
            EncodedData::Huffman(data) => data.to_deviation_data().expect(
                "Huffman encoded data should be valid when converting to raw deviation data",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::base_bits::BaseBitGroups;
    use crate::compression::base_selection::SelectBases;
    use crate::compression::base_selection::calculate_compressed_size;
    use crate::compression::condensed_samples::GenCondensedSamples;
    use crate::compression::decompression::decompress_samples_batch;
    use crate::compression::decompression::{
        DecompressRowsData, decompress_analytics, decompress_file,
    };
    use crate::compression::encoding::{EncodeData, EncodeDataRLE};
    use crate::compression::entropy::EntropyOptimized;
    use crate::compression::tabular_preprocessor::{BitData, BitDataInfo, BitDataSet, FeatureSpec};
    use crate::data_loader::FeatureDataType;
    use crate::filter_pipeline::{Filter, FilterExt};
    use pretty_assertions::assert_eq;
    use std::sync::Arc;

    /// Calculate the original uncompressed size in bits
    fn calculate_original_size(bit_data: &BitDataSet) -> usize {
        bit_data.data.num_rows * bit_data.data.chunk_size
    }

    fn get_compression_pipeline() -> impl Filter<Input = BitDataSet, Output = CompressedData> {
        EntropyOptimized {}
            .then(GenCondensedSamples { m_max: 50 })
            .then(SelectBases { patience: 10 })
            .then(EncodeData {})
    }

    fn get_rle_compression_pipeline() -> impl Filter<Input = BitDataSet, Output = CompressedData> {
        EntropyOptimized {}
            .then(GenCondensedSamples { m_max: 100 })
            .then(SelectBases { patience: 5 })
            .then(EncodeDataRLE {})
    }

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
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
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
        assert_eq!(original_size, 16);
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
        let base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0 (no bases), len_b = 0, len_d = 32, len_id = 0
        // size_deviations = 100 * 32 = 3200
        // size_params = 16*32 + 16 + 32 = 560
        // total = 3760
        assert_eq!(compressed_size, 3760);
    }

    #[test]
    fn test_calculate_compressed_size_single_row() {
        let bit_data = create_test_bit_data(1, 2, 8);
        let base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, len_b = 0, len_d = 16, len_id = 0
        // size_deviations = 1 * 16 = 16
        // size_params = 16*8 + 16 + 16 = 160
        // total = 176
        assert_eq!(compressed_size, 176);
    }

    #[test]
    fn test_calculate_compressed_size_large_data() {
        let bit_data = create_test_bit_data(10000, 2, 32);
        let base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, len_b = 0, len_d = 64, len_id = 0
        // size_deviations = 10000 * 64 = 640000
        // size_params = 16*32 + 16 + 64 = 592
        // total = 640592
        assert_eq!(compressed_size, 640592);
    }

    #[test]
    fn test_calculate_compressed_size_power_of_two_rows() {
        // Test with a power-of-2 number of rows to verify log2 calculation
        let bit_data = create_test_bit_data(16, 1, 32);
        let base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, len_b = 0, len_d = 32, len_id = 0
        // size_deviations = 16 * 32 = 512
        // size_params = 16*32 + 16 + 32 = 560
        // total = 1072
        assert_eq!(compressed_size, 1072);
    }

    // Tests for compress function
    #[test]
    fn test_compress_basic() {
        // Use larger data to avoid optimization issues in edge cases
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Check that we get a CompressedData struct with valid fields
        assert_eq!(compressed.metadata.num_features(), 16);
        assert_eq!(compressed.metadata.original_size_bits(), 100 * 32);
        assert!(!compressed.base_table.is_empty());
        assert!(compressed.encoded_data.get_encoded_size() > 0);
    }

    #[test]
    fn test_compress_single_row() {
        let bit_data = create_test_bit_data(1, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 16);
        assert_eq!(compressed.metadata.original_size_bits(), 32);
        // Verify decompressed data matches original size
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 1);
    }

    #[test]
    fn test_compress_large_data() {
        let bit_data = create_test_bit_data(1000, 2, 32);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 32);
        assert_eq!(compressed.metadata.original_size_bits(), 1000 * 64);
        // Verify decompressed data matches original size
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 1000);
    }

    #[test]
    fn test_compress_preserves_sample_count() {
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify original size is preserved in metadata
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits(), 100 * 32);
        // Verify decompressed data matches original count
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_compress_metadata_accuracy() {
        let bit_data = create_test_bit_data(50, 2, 8);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 8);
        assert_eq!(compressed.metadata.original_size_bits(), 50 * 16);
    }

    #[test]
    fn test_compress_encoded_data_structure() {
        let bit_data = create_test_bit_data(20, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify original metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits(), 20 * 32);

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
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 10);

        // Verify metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits(), 10 * 32);
    }

    #[test]
    fn test_compress_base_table_non_empty() {
        let bit_data = create_test_bit_data(50, 2, 32);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

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
            let compressed = get_compression_pipeline()
                .process(bit_data.clone())
                .unwrap();

            assert_eq!(compressed.metadata.num_features(), 16);
            assert_eq!(compressed.metadata.original_size_bits(), num_rows * 32);

            // Verify decompress_file returns correct number of rows
            let decompressed = decompress_file(&compressed).unwrap();
            assert_eq!(decompressed.data.num_rows, num_rows);
        }
    }

    #[test]
    fn test_compress_encoded_data_retrieval() {
        let bit_data = create_test_bit_data(20, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

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

        let compressed1 = get_compression_pipeline()
            .process(bit_data1.clone())
            .unwrap();
        let compressed2 = get_compression_pipeline()
            .process(bit_data2.clone())
            .unwrap();

        assert_eq!(
            compressed1.metadata.original_size_bits(),
            compressed2.metadata.original_size_bits()
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
            let compressed = get_compression_pipeline()
                .process(bit_data.clone())
                .unwrap();

            assert_eq!(compressed.metadata.original_size_bits(), num_rows * 32);
            // Verify decompressed matches original
            let decompressed = decompress_file(&compressed).unwrap();
            assert_eq!(decompressed.data.num_rows, num_rows);
        }
    }

    #[test]
    fn test_compress_small_chunks() {
        // Test with 8-bit chunks (1 byte per feature, 1 feature)
        let bit_data = create_test_bit_data(100, 1, 8);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 8);
        assert_eq!(compressed.metadata.original_size_bits(), 100 * 8);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_compress_medium_chunks() {
        // Test with 16-bit chunks
        let bit_data = create_test_bit_data(100, 1, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 16);
        assert_eq!(compressed.metadata.original_size_bits(), 100 * 16);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_compress_large_chunks() {
        // Test with 32-bit chunks
        let bit_data = create_test_bit_data(100, 1, 32);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 32);
        assert_eq!(compressed.metadata.original_size_bits(), 100 * 32);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_deviation_data_out_of_bounds() {
        let bit_data = create_test_bit_data(50, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 50);

        // Verify metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits(), 50 * 32);
    }

    #[test]
    fn test_compressed_data_structure() {
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify CompressedData has all required fields
        assert!(!compressed.base_table.is_empty());
        assert!(compressed.encoded_data.get_num_samples() > 0);
        assert_eq!(compressed.metadata.num_features(), 16);
        assert!(compressed.metadata.original_size_bits() > 0);
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
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        BitDataSet { data, info }
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        // Create test bit data with zeros
        let original_rows = 50;
        let bit_data = create_test_bit_data(original_rows, 2, 16);

        // Compress the data
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
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

        tracing::info!("\n=== INPUT DATA ===");
        tracing::info!("Num rows: {}", bit_data.data.num_rows);
        tracing::info!("Chunk size: {}", bit_data.data.chunk_size);
        tracing::info!("Total bits: {}", bit_data.data.data.len());
        for row in 0..bit_data.data.num_rows {
            let start = row * bit_data.data.chunk_size;
            let end = start + bit_data.data.chunk_size;
            let chunk: Vec<u8> = bit_data.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            tracing::info!("Row {}: {:?}", row, chunk);
        }

        // Compress the data
        tracing::info!("\n=== COMPRESSING ===");
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        tracing::info!("Base table entries: {}", compressed.base_table.len());
        tracing::info!(
            "Encoded bit stream size: {}",
            compressed.encoded_data.get_encoded_size()
        );
        tracing::info!("Num samples: {}", compressed.encoded_data.get_num_samples());
        tracing::info!(
            "Num deviation bits: {}",
            compressed.encoded_data.get_num_deviation_bits()
        );
        tracing::info!("Num ID bits: {}", compressed.encoded_data.get_num_id_bits());

        // Decompress the data
        tracing::info!("\n=== DECOMPRESSING ===");
        let decompressed = decompress_file(&compressed).expect("Decompression failed");

        tracing::info!("Decompressed num rows: {}", decompressed.data.num_rows);
        tracing::info!("Decompressed chunk size: {}", decompressed.data.chunk_size);

        tracing::info!("\n=== COMPARING ===");
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
                tracing::info!("Row {}: OK", row);
            } else {
                tracing::info!("Row {} MISMATCH:", row);
                tracing::info!("  Original:     {:?}", original_chunk);
                tracing::info!("  Decompressed: {:?}", decompressed_chunk);
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
            tracing::info!("\n=== ANALYTICS ===");
            tracing::info!("Condensed samples: {}", analytics.samples.len());
            for (i, sample) in analytics.samples.iter().enumerate() {
                let weight = analytics.weights[i];
                let sample_bits: Vec<u8> = sample.iter().map(|b| if *b { 1 } else { 0 }).collect();
                tracing::info!("Sample {}: {:?}, Weight: {}", i, sample_bits, weight);
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
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };
        tracing::info!("\n=== INPUT DATA ===");
        for row in 0..bit_data.data.num_rows {
            let start = row * bit_data.data.chunk_size;
            let end = start + bit_data.data.chunk_size;
            let chunk: Vec<u8> = bit_data.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            tracing::info!("Row {}: {:?}", row, chunk);
        }
        // Compress the data
        tracing::info!("\n=== COMPRESSING ===");
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        tracing::info!("Base table entries: {}", compressed.base_table.len());
        tracing::info!("Base table: {:?}", compressed.base_table);
        tracing::info!(
            "Encoded bit stream size: {}",
            compressed.encoded_data.get_encoded_size()
        );
        tracing::info!("Num samples: {}", compressed.encoded_data.get_num_samples());
        tracing::info!(
            "Num deviation bits: {}",
            compressed.encoded_data.get_num_deviation_bits()
        );
        tracing::info!("Num ID bits: {}", compressed.encoded_data.get_num_id_bits());
        // Decompress analytics if available
        if let Some(analytics) = decompress_analytics(&compressed) {
            tracing::info!("\n=== ANALYTICS ===");
            tracing::info!("Condensed samples: {}", analytics.samples.len());
            for (i, sample) in analytics.samples.iter().enumerate() {
                let weight = analytics.weights[i];
                let sample_bits: Vec<u8> = sample.iter().map(|b| if *b { 1 } else { 0 }).collect();
                tracing::info!("Sample {}: {:?}, Weight: {}", i, sample_bits, weight);
            }
        }
        // Decompress the data
        tracing::info!("\n=== DECOMPRESSING ===");
        let decompressed = decompress_file(&compressed).expect("Decompression failed");
        tracing::info!("Decompressed num rows: {}", decompressed.data.num_rows);
        tracing::info!("Decompressed chunk size: {}", decompressed.data.chunk_size);
        tracing::info!("\n=== COMPARING ===");
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
                tracing::info!("Row {}: OK", row);
            } else {
                tracing::info!("Row {} MISMATCH:", row);
                tracing::info!("  Original:     {:?}", original_chunk);
                tracing::info!("  Decompressed: {:?}", decompressed_chunk);
            }
        }
    }

    #[test]
    fn test_decompress_rows_random_subset() {
        let bit_data = create_test_bit_data(20, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        let indices = vec![0, 3, 7, 19];

        let subset = decompress_samples_batch(&compressed, &indices).unwrap();

        assert_eq!(subset.data.num_rows, indices.len());
        assert_eq!(subset.data.chunk_size, 32);
        assert_eq!(subset.info.num_features(), 16);
    }

    #[test]
    fn test_decompress_rows_filter() {
        let bit_data = create_test_bit_data(10, 2, 8);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        let filter = DecompressRowsData;
        let subset = filter
            .process((Arc::clone(&Arc::new(compressed)), vec![1, 5, 9]))
            .unwrap();

        assert_eq!(subset.data.num_rows, 3);
        assert_eq!(subset.data.chunk_size, 16);
        assert_eq!(subset.info.num_features(), 8);
    }

    #[test]
    fn test_encode_data_rle_roundtrip() {
        let original_rows = 64;
        let bit_data = create_simulated_bit_data(original_rows, 1, 8);
        let compressed = get_rle_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        let decompressed = decompress_file(&compressed).unwrap();

        assert_eq!(decompressed.data.num_rows, original_rows);
        assert_eq!(decompressed.data.chunk_size, bit_data.chunk_size());
        assert_eq!(
            decompressed.data.data,
            bit_data.data.data[0..(original_rows * bit_data.chunk_size())]
        );
    }

    #[test]
    fn test_encode_data_rle_sample_retrieval() {
        let bit_data = create_simulated_bit_data(32, 1, 8);
        let compressed = get_rle_compression_pipeline().process(bit_data).unwrap();

        for i in 0..compressed.encoded_data.get_num_samples() {
            let sample = compressed.encoded_data.get_sample(i);
            assert!(sample.is_some());
            let sample = sample.unwrap();
            assert_eq!(
                sample.deviation.len() + sample.id.len(),
                compressed.encoded_data.get_num_deviation_bits()
                    + compressed.encoded_data.get_num_id_bits()
            );
        }
    }

    #[test]
    fn test_encode_data_rle_control_stream_terminator_and_bounds() {
        let bit_data = create_simulated_bit_data(600, 1, 8);
        let compressed = get_rle_compression_pipeline().process(bit_data).unwrap();

        let rle = match &compressed.encoded_data {
            EncodedData::Rle(data) => data,
            _ => panic!("expected RLE encoded data"),
        };

        for &(r, m) in rle.rm_values() {
            assert!(r <= 134, "r must fit packed RLE control encoding (<= 134)");
            assert!(m <= 134, "m must fit packed RLE control encoding (<= 134)");
        }

        let ctrl = rle.rm_control_stream();
        assert!(ctrl.len() >= 8);
        let end = &ctrl[ctrl.len() - 8..ctrl.len()];
        let as_u8 = end.iter().fold(0u8, |acc, b| (acc << 1) | (*b as u8));
        assert_eq!(as_u8, 0xFF);
    }
}
