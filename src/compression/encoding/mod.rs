mod context;
mod fused_dictionary;
mod huffman;
mod huffman_codec;
mod layout;
mod rle;
mod types;

use self::context::{encode_rows_as_symbol_stream, prepare_encoding_context};
use self::fused_dictionary::encode_data_fused_dictionary;
use self::huffman::{
    encode_data_huffman as encode_data_huffman_core,
    encode_data_huffman_base_id_only as encode_data_huffman_base_id_only_core,
};

pub(crate) use self::huffman_codec::build_huffman_code_map;
pub(crate) use self::layout::{build_base_bit_mask, huffman_row_layout};
pub(crate) use self::rle::{RLE_LONG_MAX, RLE_SHORT_MAX, RLE_TERMINATOR_PAYLOAD};
pub use self::types::{
    CompressedData, CondensedSamples, DeviationData, DeviationSample, EncodedData,
    HuffmanDeviationData, RleDeviationData,
};

use crate::compression::base_bits::BaseBit;
use crate::compression::preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use crate::utils::bits_needed_nonzero;

const RLE_MAX_CONTROL_VALUE: usize = 134;
const RLE_MAX_RUN_LEN: usize = RLE_MAX_CONTROL_VALUE + 1;

struct BaseLayoutInfo {
    variable_positions: Vec<usize>,
    constant_zero_positions: Vec<usize>,
    constant_one_positions: Vec<usize>,
    variable_indices: Vec<usize>,
}

fn derive_base_layout_from_selected_bases(
    selected_positions: &[usize],
    selected_bases: &[(crate::BitStream, usize)],
) -> BaseLayoutInfo {
    if selected_bases.is_empty() {
        return BaseLayoutInfo {
            variable_positions: Vec::new(),
            constant_zero_positions: Vec::new(),
            constant_one_positions: Vec::new(),
            variable_indices: Vec::new(),
        };
    }

    let selected_len = selected_bases[0].0.len();
    let mut variable_positions = Vec::new();
    let mut constant_zero_positions = Vec::new();
    let mut constant_one_positions = Vec::new();
    let mut variable_indices = Vec::new();

    for selected_idx in 0..selected_len {
        let first_value = unsafe { *selected_bases[0].0.get_unchecked(selected_idx) };
        let is_variable = selected_bases.iter().skip(1).any(|(base_bits, _)| {
            base_bits
                .get(selected_idx)
                .map(|bit| *bit != first_value)
                .unwrap_or(false)
        });

        let Some(&bit_position) = selected_positions.get(selected_idx) else {
            continue;
        };

        if is_variable {
            variable_indices.push(selected_idx);
            variable_positions.push(bit_position);
        } else if first_value {
            constant_one_positions.push(bit_position);
        } else {
            constant_zero_positions.push(bit_position);
        }
    }

    BaseLayoutInfo {
        variable_positions,
        constant_zero_positions,
        constant_one_positions,
        variable_indices,
    }
}

fn project_selected_bases_to_variable(
    selected_bases: &[(crate::BitStream, usize)],
    variable_indices: &[usize],
) -> Vec<(crate::BitStream, usize)> {
    selected_bases
        .iter()
        .map(|(selected_bits, count)| {
            let mut variable_bits = crate::BitStream::with_capacity(variable_indices.len());
            for &selected_idx in variable_indices {
                variable_bits.push(
                    selected_bits
                        .get(selected_idx)
                        .map(|bit| *bit)
                        .unwrap_or(false),
                );
            }
            (variable_bits, *count)
        })
        .collect()
}

pub struct EncodeData {}

impl Filter for EncodeData {
    type Input = (BitDataSet, Box<dyn BaseBit>);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format");
        let (bit_data, base_bit_groups) = input;
        let mut compressed = CompressedData::new(
            EncodedData::Normal(encode_data(&bit_data, base_bit_groups.as_ref())),
            bit_data.info.clone(),
        );
        let selected_base_table = base_bit_groups.get_bases(&bit_data);
        let selected_positions = base_bit_groups.get_base_bit_positions().to_vec();
        let layout =
            derive_base_layout_from_selected_bases(&selected_positions, &selected_base_table);
        compressed.base_table = base_bit_groups.get_variable_bases(&bit_data);
        compressed.base_bit_positions = selected_positions;
        compressed.variable_base_bit_positions = layout.variable_positions;
        compressed.constant_zero_bit_positions = layout.constant_zero_positions;
        compressed.constant_one_bit_positions = layout.constant_one_positions;
        compressed.condensed_sample_weights = bit_data
            .info
            .m_condensed_sample_weights()
            .map(|weights| weights.to_vec());
        Ok(compressed)
    }
}

pub struct EncodeDataOptimized {}

impl Filter for EncodeDataOptimized {
    type Input = (BitDataSet, Box<dyn BaseBit>);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format (optimized)");
        let (bit_data, base_bit_groups) = input;
        let mut compressed = CompressedData::new(
            EncodedData::Normal(encode_data_optimized(&bit_data, base_bit_groups.as_ref())),
            bit_data.info.clone(),
        );
        let selected_base_table = base_bit_groups.get_bases(&bit_data);
        let selected_positions = base_bit_groups.get_base_bit_positions().to_vec();
        let layout =
            derive_base_layout_from_selected_bases(&selected_positions, &selected_base_table);
        compressed.base_table = base_bit_groups.get_variable_bases(&bit_data);
        compressed.base_bit_positions = selected_positions;
        compressed.variable_base_bit_positions = layout.variable_positions;
        compressed.constant_zero_bit_positions = layout.constant_zero_positions;
        compressed.constant_one_bit_positions = layout.constant_one_positions;
        compressed.condensed_sample_weights = bit_data
            .info
            .m_condensed_sample_weights()
            .map(|weights| weights.to_vec());
        Ok(compressed)
    }
}

pub struct EncodeDataRLE {}

impl Filter for EncodeDataRLE {
    type Input = (BitDataSet, Box<dyn BaseBit>);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format (optimized + RLE)");
        let (bit_data, base_bit_groups) = input;
        let mut compressed = CompressedData::new(
            EncodedData::Rle(encode_data_rle(&bit_data, base_bit_groups.as_ref())),
            bit_data.info.clone(),
        );
        let selected_base_table = base_bit_groups.get_bases(&bit_data);
        let selected_positions = base_bit_groups.get_base_bit_positions().to_vec();
        let layout =
            derive_base_layout_from_selected_bases(&selected_positions, &selected_base_table);
        compressed.base_table = base_bit_groups.get_variable_bases(&bit_data);
        compressed.base_bit_positions = selected_positions;
        compressed.variable_base_bit_positions = layout.variable_positions;
        compressed.constant_zero_bit_positions = layout.constant_zero_positions;
        compressed.constant_one_bit_positions = layout.constant_one_positions;
        compressed.condensed_sample_weights = bit_data
            .info
            .m_condensed_sample_weights()
            .map(|weights| weights.to_vec());
        Ok(compressed)
    }
}

pub struct EncodeDataFusedDictionary {}

impl Filter for EncodeDataFusedDictionary {
    type Input = (BitDataSet, Box<dyn BaseBit>);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(
            "Encoding data into compressed format (fused dictionary + id/deviation)",
        );
        let (bit_data, base_bit_groups) = input;
        let fused = encode_data_fused_dictionary(&bit_data, base_bit_groups.as_ref());

        let mut compressed = CompressedData::new(
            EncodedData::Normal(fused.deviation_data),
            bit_data.info.clone(),
        );
        let selected_positions = base_bit_groups.get_base_bit_positions().to_vec();
        let layout = derive_base_layout_from_selected_bases(&selected_positions, &fused.base_table);
        compressed.base_table =
            project_selected_bases_to_variable(&fused.base_table, &layout.variable_indices);
        compressed.base_bit_positions = selected_positions;
        compressed.variable_base_bit_positions = layout.variable_positions;
        compressed.constant_zero_bit_positions = layout.constant_zero_positions;
        compressed.constant_one_bit_positions = layout.constant_one_positions;
        compressed.condensed_sample_weights = bit_data
            .info
            .m_condensed_sample_weights()
            .map(|weights| weights.to_vec());
        Ok(compressed)
    }
}

pub struct EncodeDataHuffman {}

impl Filter for EncodeDataHuffman {
    type Input = (BitDataSet, Box<dyn BaseBit>);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format (Huffman)");
        let (bit_data, base_bit_groups) = input;
        let context = prepare_encoding_context(&bit_data, base_bit_groups.as_ref());
        let mut compressed = CompressedData::new(
            EncodedData::Huffman(encode_data_huffman_core(&bit_data, &context)?),
            bit_data.info.clone(),
        );
        let selected_base_table = base_bit_groups.get_bases(&bit_data);
        let selected_positions = base_bit_groups.get_base_bit_positions().to_vec();
        let layout =
            derive_base_layout_from_selected_bases(&selected_positions, &selected_base_table);
        compressed.base_table = base_bit_groups.get_variable_bases(&bit_data);
        compressed.base_bit_positions = selected_positions;
        compressed.variable_base_bit_positions = layout.variable_positions;
        compressed.constant_zero_bit_positions = layout.constant_zero_positions;
        compressed.constant_one_bit_positions = layout.constant_one_positions;
        compressed.condensed_sample_weights = bit_data
            .info
            .m_condensed_sample_weights()
            .map(|weights| weights.to_vec());

        Ok(compressed)
    }
}

pub struct EncodeDataHuffmanBaseIdOnly {}

impl Filter for EncodeDataHuffmanBaseIdOnly {
    type Input = (BitDataSet, Box<dyn BaseBit>);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer =
            ScopedTimer::info("Encoding data into compressed format (Huffman base-id only)");
        let (bit_data, base_bit_groups) = input;
        let context = prepare_encoding_context(&bit_data, base_bit_groups.as_ref());
        let mut compressed = CompressedData::new(
            EncodedData::Huffman(encode_data_huffman_base_id_only_core(&bit_data, &context)?),
            bit_data.info.clone(),
        );
        let selected_base_table = base_bit_groups.get_bases(&bit_data);
        let selected_positions = base_bit_groups.get_base_bit_positions().to_vec();
        let layout =
            derive_base_layout_from_selected_bases(&selected_positions, &selected_base_table);
        compressed.base_table = base_bit_groups.get_variable_bases(&bit_data);
        compressed.base_bit_positions = selected_positions;
        compressed.variable_base_bit_positions = layout.variable_positions;
        compressed.constant_zero_bit_positions = layout.constant_zero_positions;
        compressed.constant_one_bit_positions = layout.constant_one_positions;
        compressed.condensed_sample_weights = bit_data
            .info
            .m_condensed_sample_weights()
            .map(|weights| weights.to_vec());

        Ok(compressed)
    }
}

fn encode_data<B: BaseBit + ?Sized>(bit_data: &BitDataSet, base_bit_groups: &B) -> DeviationData {
    let mut encoded_bit_stream = crate::BitStream::new();
    let base_bit_mask = base_bit_groups.get_base_bit_mask();

    let num_bases = base_bit_groups.get_num_bases();
    let l_id = bits_needed_nonzero(num_bases);

    let chunk_size = bit_data.chunk_size();
    let num_bits_per_base = base_bit_groups.get_num_bits_per_base();
    let num_deviation_bits = chunk_size.saturating_sub(num_bits_per_base);

    let num_rows = bit_data.num_rows();
    let mut row_to_group_id = vec![0usize; num_rows];
    for (id, group) in base_bit_groups.get_groups().iter().enumerate() {
        for &row in group.iter() {
            row_to_group_id[row] = id;
        }
    }

    for (row, id_ref) in row_to_group_id.iter().enumerate().take(num_rows) {
        let id = *id_ref;
        let chunk = unsafe { bit_data.get_chunk_unchecked(row) };

        for bit_pos in 0..chunk_size {
            if !unsafe { *base_bit_mask.get_unchecked(bit_pos) } {
                encoded_bit_stream.push(unsafe { *chunk.get_unchecked(bit_pos) });
            }
        }

        if l_id > 0 {
            for shift in (0..l_id).rev() {
                encoded_bit_stream.push(((id >> shift) & 1) == 1);
            }
        }
    }

    DeviationData::new(
        encoded_bit_stream,
        bit_data.num_rows(),
        num_deviation_bits,
        l_id,
    )
}

fn encode_data_optimized<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> DeviationData {
    let context = prepare_encoding_context(bit_data, base_bit_groups);
    let encoded_bit_stream = encode_rows_as_symbol_stream(bit_data, &context);

    DeviationData::new(
        encoded_bit_stream,
        bit_data.num_rows(),
        context.num_deviation_bits,
        context.l_id,
    )
}

fn encode_data_rle<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> RleDeviationData {
    let context = prepare_encoding_context(bit_data, base_bit_groups);
    let symbol_width = context.num_deviation_bits + context.l_id;
    let num_rows = bit_data.num_rows();
    let raw_symbol_stream = encode_rows_as_symbol_stream(bit_data, &context);
    let mut symbol_stream = crate::BitStream::new();
    let mut rm_values: Vec<(u8, u8)> = Vec::new();

    if num_rows == 0 || symbol_width == 0 {
        return RleDeviationData::new(
            symbol_stream,
            rm_values,
            num_rows,
            context.num_deviation_bits,
            context.l_id,
        );
    }

    let mut i = 0usize;
    while i < num_rows {
        let current_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i);

        let mut run_len = 1usize;
        while i + run_len < num_rows && run_len < RLE_MAX_RUN_LEN {
            let next_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i + run_len);
            if next_symbol == current_symbol {
                run_len += 1;
            } else {
                break;
            }
        }

        let r_encoded: u8;
        if run_len >= 2 {
            r_encoded = (run_len - 1) as u8;
            symbol_stream.extend_from_bitslice(current_symbol);
            i += run_len;
        } else {
            r_encoded = 0;
        }

        let literal_start = i;
        let mut literal_count = 0usize;
        while i < num_rows && literal_count < RLE_MAX_CONTROL_VALUE {
            let this_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i);

            let mut lookahead_run = 1usize;
            while i + lookahead_run < num_rows && lookahead_run < RLE_MAX_RUN_LEN {
                let lookahead_symbol =
                    symbol_slice(&raw_symbol_stream, symbol_width, i + lookahead_run);
                if lookahead_symbol == this_symbol {
                    lookahead_run += 1;
                } else {
                    break;
                }
            }

            if lookahead_run >= 2 {
                break;
            }

            symbol_stream.extend_from_bitslice(this_symbol);
            literal_count += 1;
            i += 1;
        }

        if r_encoded == 0 && literal_count == 0 {
            symbol_stream.extend_from_bitslice(current_symbol);
            literal_count = 1;
            i = literal_start + 1;
        }

        rm_values.push((r_encoded, literal_count as u8));
    }

    RleDeviationData::new(
        symbol_stream,
        rm_values,
        num_rows,
        context.num_deviation_bits,
        context.l_id,
    )
}

fn symbol_slice(
    symbol_stream: &crate::BitStream,
    symbol_width: usize,
    row: usize,
) -> &crate::BitView {
    let start = row * symbol_width;
    let end = start + symbol_width;
    debug_assert!(end <= symbol_stream.len());
    unsafe { symbol_stream.get_unchecked(start..end) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::base_bits::BaseBitGroups;
    use crate::compression::preprocessor::{
        BitData, BitDataInfo, FeatureDataType, FeatureSpec, FeatureTransform,
    };

    /// Helper function to create a simple BitDataSet for testing
    /// Creates a dataset with `num_rows` rows and `chunk_size` bits per row
    fn create_test_bit_data_set(num_rows: usize, chunk_size: usize) -> BitDataSet {
        let mut data = crate::BitStream::with_capacity(num_rows * chunk_size);
        for _ in 0..num_rows * chunk_size {
            data.push(false);
        }

        let bit_data = BitData {
            data,
            chunk_size,
            stride: chunk_size,
            num_rows,
        };

        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UnsignedInt,
            bits: chunk_size,
            transform: FeatureTransform::None,
        }];

        let info = BitDataInfo::new(features, num_rows * chunk_size)
            .expect("Failed to create BitDataInfo");

        BitDataSet {
            data: bit_data,
            info,
        }
    }

    /// Helper function to create a BitDataSet with specific bit patterns
    fn create_test_bit_data_set_with_pattern(rows_and_bits: Vec<Vec<bool>>) -> BitDataSet {
        assert!(!rows_and_bits.is_empty(), "Must have at least one row");
        let chunk_size = rows_and_bits[0].len();
        let num_rows = rows_and_bits.len();

        let mut data = crate::BitStream::with_capacity(num_rows * chunk_size);
        for row in &rows_and_bits {
            assert_eq!(row.len(), chunk_size, "All rows must have the same size");
            for &bit in row {
                data.push(bit);
            }
        }

        let bit_data = BitData {
            data,
            chunk_size,
            stride: chunk_size,
            num_rows,
        };

        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UnsignedInt,
            bits: chunk_size,
            transform: FeatureTransform::None,
        }];

        let info = BitDataInfo::new(features, num_rows * chunk_size)
            .expect("Failed to create BitDataInfo");

        BitDataSet {
            data: bit_data,
            info,
        }
    }

    /// Helper function to create a simple BaseBitGroups instance for testing
    /// This creates a groups structure where all rows are initially in one group
    fn create_test_base_bit_groups(num_rows: usize, chunk_size: usize) -> BaseBitGroups {
        BaseBitGroups::new(num_rows, chunk_size)
    }

    /// Helper function to add bit positions to a BaseBitGroups instance
    fn add_base_bits(groups: &mut BaseBitGroups, bit_data: &BitDataSet, bit_positions: &[usize]) {
        for &pos in bit_positions {
            groups.add_bit_position(bit_data, pos);
        }
    }

    #[test]
    fn test_encode_data_basic() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        // Add some base bits at positions 0 and 1
        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let result = encode_data(&bit_data, &base_groups);

        // Should have encoded 6 deviation bits per row (8 - 2 = 6)
        assert_eq!(result.get_num_deviation_bits(), 6);
        assert_eq!(result.get_num_samples(), 4);
    }

    #[test]
    fn test_encode_data_optimized_basic() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let result = encode_data_optimized(&bit_data, &base_groups);

        assert_eq!(result.get_num_deviation_bits(), 6);
        assert_eq!(result.get_num_samples(), 4);
    }

    #[test]
    fn test_encode_data_rle_basic() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let result = encode_data_rle(&bit_data, &base_groups);

        assert_eq!(result.get_num_deviation_bits(), 6);
        assert_eq!(result.get_num_samples(), 4);
        // RLE should produce rm_values entries
        assert!(!result.rm_values().is_empty());
    }

    #[test]
    fn test_encode_data_huffman_basic() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let context = prepare_encoding_context(&bit_data, &base_groups);
        let result = encode_data_huffman_core(&bit_data, &context).unwrap();

        assert_eq!(result.get_num_deviation_bits(), 6);
        assert_eq!(result.get_num_samples(), 4);
        assert_eq!(result.row_offsets().len(), 4);
    }

    #[test]
    fn test_encode_data_huffman_matches_raw_symbols() {
        let bit_data = create_test_bit_data_set_with_pattern(vec![
            vec![false, false, false, true, false, true, false, true],
            vec![false, false, true, true, false, false, false, true],
            vec![false, false, false, false, true, true, false, false],
            vec![false, false, true, false, true, false, true, false],
        ]);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let raw = encode_data_optimized(&bit_data, &base_groups);
        let context = prepare_encoding_context(&bit_data, &base_groups);
        let huffman = encode_data_huffman_core(&bit_data, &context).unwrap();

        assert_eq!(
            huffman.to_deviation_data().unwrap().encoded_bit_stream(),
            raw.encoded_bit_stream()
        );
    }

    #[test]
    fn test_encode_data_huffman_base_id_only_basic() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let context = prepare_encoding_context(&bit_data, &base_groups);
        let result = encode_data_huffman_base_id_only_core(&bit_data, &context).unwrap();

        assert_eq!(result.get_num_deviation_bits(), 6);
        assert_eq!(result.get_num_id_bits(), 1);
        assert_eq!(result.get_num_samples(), 4);
        assert_eq!(result.row_offsets().len(), 4);

        let raw = encode_data_optimized(&bit_data, &base_groups);
        assert_eq!(
            result.to_deviation_data().unwrap().encoded_bit_stream(),
            raw.encoded_bit_stream()
        );
    }

    #[test]
    fn test_encode_data_empty_dataset() {
        let bit_data = create_test_bit_data_set(0, 8);
        let base_groups = create_test_base_bit_groups(0, 8);

        let result = encode_data(&bit_data, &base_groups);

        assert_eq!(result.get_num_samples(), 0);
        assert_eq!(result.encoded_bit_stream().len(), 0);
    }

    #[test]
    fn test_encode_data_no_base_bits() {
        let bit_data = create_test_bit_data_set(2, 8);
        let base_groups = create_test_base_bit_groups(2, 8);

        let result = encode_data(&bit_data, &base_groups);

        // With no base bits, all 8 bits should be encoded as deviation
        assert_eq!(result.get_num_deviation_bits(), 8);
        assert_eq!(result.get_num_id_bits(), 1); // log2(1) = 0, but clamped to 1
    }

    #[test]
    fn test_encode_data_all_base_bits() {
        let bit_data = create_test_bit_data_set(2, 8);
        let mut base_groups = create_test_base_bit_groups(2, 8);

        // Add all bit positions as base bits
        add_base_bits(&mut base_groups, &bit_data, &[0, 1, 2, 3, 4, 5, 6, 7]);

        let result = encode_data(&bit_data, &base_groups);

        // With all bits as base bits, no deviation bits
        assert_eq!(result.get_num_deviation_bits(), 0);
        // With 2 rows, all zero pattern creates 1 group, so l_id = 1
        assert_eq!(result.get_num_id_bits(), 1);
    }

    #[test]
    fn test_prepare_encoding_context_creates_valid_id_bits() {
        let bit_data = create_test_bit_data_set(8, 8);
        let mut base_groups = create_test_base_bit_groups(8, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0]);

        let context = prepare_encoding_context(&bit_data, &base_groups);

        // With 8 identical zero rows, adding 1 base bit creates at most 2 groups (zeros and ones)
        // Since all rows are zeros, we get 1 group, so l_id = 1
        assert_eq!(context.l_id, 1);
        assert_eq!(context.id_bits_per_base.len(), 1);
    }

    #[test]
    fn test_encode_filter_process() {
        let bit_data = create_test_bit_data_set(2, 8);
        let mut base_groups = create_test_base_bit_groups(2, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let filter = EncodeData {};
        let result = filter.process((bit_data, Box::new(base_groups)));

        assert!(result.is_ok());
        let compressed = result.unwrap();
        assert_eq!(compressed.metadata.chunk_size(), 8);
    }

    #[test]
    fn test_encode_data_optimized_filter_process() {
        let bit_data = create_test_bit_data_set(2, 8);
        let mut base_groups = create_test_base_bit_groups(2, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let filter = EncodeDataOptimized {};
        let result = filter.process((bit_data, Box::new(base_groups)));

        assert!(result.is_ok());
        let compressed = result.unwrap();
        assert_eq!(compressed.metadata.chunk_size(), 8);
    }

    #[test]
    fn test_encode_data_rle_filter_process() {
        let bit_data = create_test_bit_data_set(2, 8);
        let mut base_groups = create_test_base_bit_groups(2, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let filter = EncodeDataRLE {};
        let result = filter.process((bit_data, Box::new(base_groups)));

        assert!(result.is_ok());
        let compressed = result.unwrap();
        assert_eq!(compressed.metadata.chunk_size(), 8);
    }

    #[test]
    fn test_encode_data_fused_dictionary_matches_standard_on_full_groups() {
        let patterns = vec![
            vec![false, false, false, true, false, true, false, true],
            vec![false, false, true, true, false, false, false, true],
            vec![true, true, false, false, true, true, false, false],
            vec![true, true, true, false, true, false, true, false],
        ];
        let bit_data = create_test_bit_data_set_with_pattern(patterns);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let fused = encode_data_fused_dictionary(&bit_data, &base_groups);
        let optimized = encode_data_optimized(&bit_data, &base_groups);

        assert_eq!(
            fused.deviation_data.encoded_bit_stream(),
            optimized.encoded_bit_stream()
        );
        assert_eq!(fused.base_table, base_groups.get_bases(&bit_data));
    }

    #[test]
    fn test_encode_data_fused_dictionary_supports_partial_training_groups() {
        let full_data = create_test_bit_data_set_with_pattern(vec![
            vec![false, false, false, false],
            vec![false, false, true, false],
            vec![true, true, false, true],
            vec![true, true, true, true],
        ]);
        let training_subset = create_test_bit_data_set_with_pattern(vec![
            vec![false, false, false, false],
            vec![false, false, true, false],
        ]);

        let mut trained_groups = create_test_base_bit_groups(2, 4);
        add_base_bits(&mut trained_groups, &training_subset, &[0, 1]);

        let fused = encode_data_fused_dictionary(&full_data, &trained_groups);
        let mut counts: Vec<usize> = fused
            .base_table
            .iter()
            .map(|(_base, count)| *count)
            .collect();
        counts.sort_unstable();

        assert_eq!(fused.base_table.len(), 2);
        assert_eq!(counts, vec![2, 2]);
        assert_eq!(fused.deviation_data.get_num_samples(), 4);
        assert_eq!(fused.deviation_data.get_num_deviation_bits(), 2);
        assert_eq!(fused.deviation_data.get_num_id_bits(), 1);
    }

    #[test]
    fn test_encode_data_fused_dictionary_filter_process() {
        let bit_data = create_test_bit_data_set(2, 8);
        let mut base_groups = create_test_base_bit_groups(2, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let filter = EncodeDataFusedDictionary {};
        let result = filter.process((bit_data, Box::new(base_groups)));

        assert!(result.is_ok());
        let compressed = result.unwrap();
        assert_eq!(compressed.metadata.chunk_size(), 8);
    }

    #[test]
    fn test_encode_data_with_specific_patterns() {
        // Create a dataset with specific bit patterns: all 0s followed by all 1s
        let patterns = vec![
            vec![false, false, false, false, false, false, false, false],
            vec![false, false, false, false, false, false, false, false],
            vec![true, true, true, true, true, true, true, true],
            vec![true, true, true, true, true, true, true, true],
        ];

        let bit_data = create_test_bit_data_set_with_pattern(patterns);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 4]);

        let result = encode_data(&bit_data, &base_groups);

        assert_eq!(result.get_num_samples(), 4);
        assert_eq!(result.get_num_deviation_bits(), 6);
    }

    #[test]
    fn test_encoding_context_row_to_group_mapping() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0]);

        let context = prepare_encoding_context(&bit_data, &base_groups);

        // Initially all rows should be in same group, but after adding a base bit
        // the rows might be split into groups
        assert_eq!(context.row_to_group_id.len(), 4);
    }

    #[test]
    fn test_symbol_stream_encoding_consistency() {
        let bit_data = create_test_bit_data_set(3, 8);
        let mut base_groups = create_test_base_bit_groups(3, 8);

        add_base_bits(&mut base_groups, &bit_data, &[1, 3]);

        let result_normal = encode_data(&bit_data, &base_groups);
        let result_optimized = encode_data_optimized(&bit_data, &base_groups);

        // Both encoding methods should produce the same size output
        assert_eq!(
            result_normal.encoded_bit_stream().len(),
            result_optimized.encoded_bit_stream().len()
        );
    }

    #[test]
    fn test_single_row_encoding() {
        let bit_data = create_test_bit_data_set(1, 16);
        let mut base_groups = create_test_base_bit_groups(1, 16);

        add_base_bits(&mut base_groups, &bit_data, &[0, 5, 10, 15]);

        let result = encode_data(&bit_data, &base_groups);

        assert_eq!(result.get_num_samples(), 1);
        assert_eq!(result.get_num_deviation_bits(), 12);
        assert_eq!(result.get_num_id_bits(), 1); // log2(1) = 0, but clamped to 1
    }

    #[test]
    fn test_large_dataset_encoding() {
        let bit_data = create_test_bit_data_set(256, 32);
        let mut base_groups = create_test_base_bit_groups(256, 32);

        add_base_bits(&mut base_groups, &bit_data, &[0, 8, 16, 24]);

        let result = encode_data(&bit_data, &base_groups);

        assert_eq!(result.get_num_samples(), 256);
        assert_eq!(result.get_num_deviation_bits(), 28);
        // Since all rows have same bit pattern (all zeros), adding bits won't split them
        // So we get 1 group -> l_id = 1
        assert_eq!(result.get_num_id_bits(), 1);
    }

    #[test]
    fn test_rle_compression_with_repeated_rows() {
        // Create a dataset where rows alternate between two patterns
        let patterns = vec![
            vec![true, false, true, false, true, false, true, false],
            vec![true, false, true, false, true, false, true, false],
            vec![false, true, false, true, false, true, false, true],
            vec![true, false, true, false, true, false, true, false],
            vec![true, false, true, false, true, false, true, false],
        ];

        let bit_data = create_test_bit_data_set_with_pattern(patterns);
        let mut base_groups = create_test_base_bit_groups(5, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 2]);

        let result = encode_data_rle(&bit_data, &base_groups);

        assert_eq!(result.get_num_samples(), 5);
        // RLE should have some run-length encoding
        assert!(!result.rm_values().is_empty());
    }
}
