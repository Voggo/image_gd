mod context;
mod fused_dictionary;
mod huffman;
mod huffman_codec;
mod layout;
mod rle;
mod types;

use self::context::{build_encoding_context, encode_rows_as_symbol_stream};
use self::fused_dictionary::encode_data_fused_dictionary;
use self::huffman::{
    encode_data_huffman as encode_data_huffman_core,
    encode_data_huffman_base_id_only as encode_data_huffman_base_id_only_core,
};

pub(crate) use self::huffman_codec::build_huffman_code_map;
pub(crate) use self::layout::{build_base_bit_mask, huffman_row_layout};
pub(crate) use self::rle::{RLE_LONG_MAX, RLE_SHORT_MAX, RLE_TERMINATOR_PAYLOAD};
pub use self::types::{
    BaseTable, CompressedData, CondensedSamples, DeviationData, DeviationSample, EncodedData,
    HuffmanDeviationData, RleDeviationData,
};

use crate::compression::base_bits::BaseBit;
use crate::compression::base_table::{
    PreEncodeContext, build_base_layout, project_selected_bases_to_variable,
};
use crate::compression::preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
const RLE_MAX_CONTROL_VALUE: usize = 134;
const RLE_MAX_RUN_LEN: usize = RLE_MAX_CONTROL_VALUE + 1;

pub struct EncodeData {}

impl Filter for EncodeData {
    type Input = PreEncodeContext;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format");
        let encoded = EncodedData::Normal(encode_data(&input));
        Ok(build_compressed_data(input, encoded))
    }
}

pub struct EncodeDataOptimized {}

impl Filter for EncodeDataOptimized {
    type Input = PreEncodeContext;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format (optimized)");
        let encoded = EncodedData::Normal(encode_data(&input));
        Ok(build_compressed_data(input, encoded))
    }
}

pub struct EncodeDataRLE {}

impl Filter for EncodeDataRLE {
    type Input = PreEncodeContext;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format (optimized + RLE)");
        let encoded = EncodedData::Rle(encode_data_rle(&input));
        Ok(build_compressed_data(input, encoded))
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
        let layout = build_base_layout(&selected_positions, &fused.base_table);
        compressed.base_table = BaseTable::Raw(project_selected_bases_to_variable(
            &selected_positions,
            &layout.variable_base_bit_positions,
            &fused.base_table,
        ));
        compressed.base_bit_positions = selected_positions;
        compressed.variable_base_bit_positions = layout.variable_base_bit_positions;
        compressed.constant_zero_bit_positions = layout.constant_zero_bit_positions;
        compressed.constant_one_bit_positions = layout.constant_one_bit_positions;
        compressed.condensed_sample_weights = bit_data
            .info
            .m_condensed_sample_weights()
            .map(|weights| weights.to_vec());
        Ok(compressed)
    }
}

pub struct EncodeDataHuffman {}

impl Filter for EncodeDataHuffman {
    type Input = PreEncodeContext;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format (Huffman)");
        let context = build_encoding_context(&input);
        let encoded = EncodedData::Huffman(encode_data_huffman_core(&input.bit_data, &context)?);
        Ok(build_compressed_data(input, encoded))
    }
}

pub struct EncodeDataHuffmanBaseIdOnly {}

impl Filter for EncodeDataHuffmanBaseIdOnly {
    type Input = PreEncodeContext;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer =
            ScopedTimer::info("Encoding data into compressed format (Huffman base-id only)");
        let context = build_encoding_context(&input);
        let encoded = EncodedData::Huffman(encode_data_huffman_base_id_only_core(
            &input.bit_data,
            &context,
        )?);
        Ok(build_compressed_data(input, encoded))
    }
}

fn build_compressed_data(input: PreEncodeContext, encoded_data: EncodedData) -> CompressedData {
    let mut compressed = CompressedData::new(encoded_data, input.bit_data.info.clone());
    compressed.base_table = BaseTable::Raw(input.variable_base_table);
    compressed.base_bit_positions = input.layout.selected_base_bit_positions;
    compressed.variable_base_bit_positions = input.layout.variable_base_bit_positions;
    compressed.constant_zero_bit_positions = input.layout.constant_zero_bit_positions;
    compressed.constant_one_bit_positions = input.layout.constant_one_bit_positions;
    compressed.condensed_sample_weights = input
        .bit_data
        .info
        .m_condensed_sample_weights()
        .map(|weights| weights.to_vec());
    compressed
}

fn encode_data(input: &PreEncodeContext) -> DeviationData {
    let context = build_encoding_context(input);
    let encoded_bit_stream = encode_rows_as_symbol_stream(&input.bit_data, &context);

    DeviationData::new(
        encoded_bit_stream,
        input.bit_data.num_rows(),
        context.num_deviation_bits,
        context.l_id,
    )
}

fn encode_data_rle(input: &PreEncodeContext) -> RleDeviationData {
    let context = build_encoding_context(input);
    let symbol_width = context.num_deviation_bits + context.l_id;
    let num_rows = input.bit_data.num_rows();
    let raw_symbol_stream = encode_rows_as_symbol_stream(&input.bit_data, &context);
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
