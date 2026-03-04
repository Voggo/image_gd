use crate::compression::base_bits::BaseBit;
use crate::compression::compress::{CompressedData, DeviationData, EncodedData, RleDeviationData};
use crate::compression::tabular_preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use bitvec::prelude::*;

const RLE_MAX_CONTROL_VALUE: usize = 134;
const RLE_MAX_RUN_LEN: usize = RLE_MAX_CONTROL_VALUE + 1;

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
        compressed.base_table = base_bit_groups.get_bases(&bit_data);
        compressed.base_bit_positions = base_bit_groups.get_base_bit_positions().to_vec();
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
        compressed.base_table = base_bit_groups.get_bases(&bit_data);
        compressed.base_bit_positions = base_bit_groups.get_base_bit_positions().to_vec();
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
        compressed.base_table = base_bit_groups.get_bases(&bit_data);
        compressed.base_bit_positions = base_bit_groups.get_base_bit_positions().to_vec();
        compressed.condensed_sample_weights = bit_data
            .info
            .m_condensed_sample_weights()
            .map(|weights| weights.to_vec());
        Ok(compressed)
    }
}

struct EncodingContext {
    l_id: usize,
    num_deviation_bits: usize,
    row_to_group_id: Vec<usize>,
    deviation_ranges: Vec<(usize, usize)>,
    id_bits_per_base: Vec<BitVec<usize, Msb0>>,
}

fn prepare_encoding_context<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> EncodingContext {
    let base_bit_mask = base_bit_groups.get_base_bit_mask();
    let num_bases = base_bit_groups.get_num_bases();
    let l_id = {
        let bits = (num_bases as f64).log2().ceil() as usize;
        if bits == 0 { 1 } else { bits }
    };

    let chunk_size = bit_data.chunk_size();
    let num_bits_per_base = base_bit_groups.get_num_bits_per_base();
    let num_deviation_bits = chunk_size.saturating_sub(num_bits_per_base);

    let mut deviation_positions = Vec::with_capacity(num_deviation_bits);
    for bit_pos in 0..chunk_size {
        if !base_bit_mask[bit_pos] {
            deviation_positions.push(bit_pos);
        }
    }

    let mut deviation_ranges: Vec<(usize, usize)> = Vec::new();
    if let Some(&first_pos) = deviation_positions.first() {
        let mut range_start = first_pos;
        let mut prev = first_pos;
        for &pos in deviation_positions.iter().skip(1) {
            if pos == prev + 1 {
                prev = pos;
            } else {
                deviation_ranges.push((range_start, prev + 1));
                range_start = pos;
                prev = pos;
            }
        }
        deviation_ranges.push((range_start, prev + 1));
    }

    let mut id_bits_per_base: Vec<BitVec<usize, Msb0>> = Vec::new();
    if l_id > 0 {
        id_bits_per_base = Vec::with_capacity(num_bases);
        for id in 0..num_bases {
            let mut id_bits = BitVec::<usize, Msb0>::with_capacity(l_id);
            for shift in (0..l_id).rev() {
                id_bits.push(((id >> shift) & 1) == 1);
            }
            id_bits_per_base.push(id_bits);
        }
    }

    let num_rows = bit_data.num_rows();
    let mut row_to_group_id = vec![0usize; num_rows];
    for (id, group) in base_bit_groups.get_groups().iter().enumerate() {
        for &row in group.iter() {
            row_to_group_id[row] = id;
        }
    }

    EncodingContext {
        l_id,
        num_deviation_bits,
        row_to_group_id,
        deviation_ranges,
        id_bits_per_base,
    }
}

fn encode_rows_as_symbol_stream(
    bit_data: &BitDataSet,
    context: &EncodingContext,
) -> BitVec<usize, Msb0> {
    let symbol_width = context.num_deviation_bits + context.l_id;
    let num_rows = bit_data.num_rows();
    let mut symbol_stream = BitVec::<usize, Msb0>::with_capacity(num_rows * symbol_width);

    for (row, id_ref) in context.row_to_group_id.iter().enumerate().take(num_rows) {
        let id = *id_ref;
        let chunk = bit_data.get_chunk(row);

        for &(start, end) in &context.deviation_ranges {
            symbol_stream.extend_from_bitslice(&chunk[start..end]);
        }

        if context.l_id > 0 {
            symbol_stream.extend_from_bitslice(context.id_bits_per_base[id].as_bitslice());
        }
    }

    symbol_stream
}

fn append_row_symbol_to(
    bit_data: &BitDataSet,
    context: &EncodingContext,
    row: usize,
    out: &mut BitVec<usize, Msb0>,
) {
    let id = context.row_to_group_id[row];
    let chunk = bit_data.get_chunk(row);

    for &(start, end) in &context.deviation_ranges {
        out.extend_from_bitslice(&chunk[start..end]);
    }

    if context.l_id > 0 {
        out.extend_from_bitslice(context.id_bits_per_base[id].as_bitslice());
    }
}

fn make_row_symbol(
    bit_data: &BitDataSet,
    context: &EncodingContext,
    row: usize,
) -> BitVec<usize, Msb0> {
    let symbol_width = context.num_deviation_bits + context.l_id;
    let mut symbol = BitVec::<usize, Msb0>::with_capacity(symbol_width);
    append_row_symbol_to(bit_data, context, row, &mut symbol);
    symbol
}

fn encode_data<B: BaseBit + ?Sized>(bit_data: &BitDataSet, base_bit_groups: &B) -> DeviationData {
    let mut encoded_bit_stream = BitVec::<usize, Msb0>::new();
    let base_bit_mask = base_bit_groups.get_base_bit_mask();

    let num_bases = base_bit_groups.get_num_bases();
    let l_id = {
        let bits = (num_bases as f64).log2().ceil() as usize;
        if bits == 0 { 1 } else { bits }
    };

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
        let chunk = bit_data.get_chunk(row);

        for bit_pos in 0..chunk_size {
            if !base_bit_mask[bit_pos] {
                encoded_bit_stream.push(chunk[bit_pos]);
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
    let mut symbol_stream = BitVec::<usize, Msb0>::new();
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
        let current_symbol = make_row_symbol(bit_data, &context, i);

        let mut run_len = 1usize;
        while i + run_len < num_rows && run_len < RLE_MAX_RUN_LEN {
            let next_symbol = make_row_symbol(bit_data, &context, i + run_len);
            if next_symbol == current_symbol {
                run_len += 1;
            } else {
                break;
            }
        }

        let r_encoded: u8;
        if run_len >= 2 {
            r_encoded = (run_len - 1) as u8;
            symbol_stream.extend_from_bitslice(current_symbol.as_bitslice());
            i += run_len;
        } else {
            r_encoded = 0;
        }

        let literal_start = i;
        let mut literal_count = 0usize;
        while i < num_rows && literal_count < RLE_MAX_CONTROL_VALUE {
            let this_symbol = make_row_symbol(bit_data, &context, i);

            let mut lookahead_run = 1usize;
            while i + lookahead_run < num_rows && lookahead_run < RLE_MAX_RUN_LEN {
                let lookahead_symbol = make_row_symbol(bit_data, &context, i + lookahead_run);
                if lookahead_symbol == this_symbol {
                    lookahead_run += 1;
                } else {
                    break;
                }
            }

            if lookahead_run >= 2 {
                break;
            }

            symbol_stream.extend_from_bitslice(this_symbol.as_bitslice());
            literal_count += 1;
            i += 1;
        }

        if r_encoded == 0 && literal_count == 0 {
            symbol_stream.extend_from_bitslice(current_symbol.as_bitslice());
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
        let mut data = BitVec::<usize, Msb0>::with_capacity(num_rows * chunk_size);
        for _ in 0..num_rows * chunk_size {
            data.push(false);
        }

        let bit_data = BitData {
            data,
            chunk_size,
            num_rows,
        };

        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UnsignedInt,
            bits: chunk_size,
            transform: FeatureTransform::None,
        }];

        let info = BitDataInfo::new(features, 0).expect("Failed to create BitDataInfo");

        BitDataSet { data: bit_data, info }
    }

    /// Helper function to create a BitDataSet with specific bit patterns
    fn create_test_bit_data_set_with_pattern(
        rows_and_bits: Vec<Vec<bool>>,
    ) -> BitDataSet {
        assert!(!rows_and_bits.is_empty(), "Must have at least one row");
        let chunk_size = rows_and_bits[0].len();
        let num_rows = rows_and_bits.len();

        let mut data = BitVec::<usize, Msb0>::with_capacity(num_rows * chunk_size);
        for row in &rows_and_bits {
            assert_eq!(
                row.len(),
                chunk_size,
                "All rows must have the same size"
            );
            for &bit in row {
                data.push(bit);
            }
        }

        let bit_data = BitData {
            data,
            chunk_size,
            num_rows,
        };

        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UnsignedInt,
            bits: chunk_size,
            transform: FeatureTransform::None,
        }];

        let info = BitDataInfo::new(features, 0).expect("Failed to create BitDataInfo");

        BitDataSet { data: bit_data, info }
    }

    /// Helper function to create a simple BaseBitGroups instance for testing
    /// This creates a groups structure where all rows are initially in one group
    fn create_test_base_bit_groups(num_rows: usize, chunk_size: usize) -> BaseBitGroups {
        BaseBitGroups::new(num_rows, chunk_size)
    }

    /// Helper function to add bit positions to a BaseBitGroups instance
    fn add_base_bits(
        groups: &mut BaseBitGroups,
        bit_data: &BitDataSet,
        bit_positions: &[usize],
    ) {
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
