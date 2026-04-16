mod context;
mod fused_dictionary;
mod huffman;
mod huffman_codec;
mod layout;
mod rle;
mod types;

use bitvec::prelude::*;

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
    BaseTable, CompressedData, CondensedSamples, DeltaBaseTableData, DeviationData,
    DeviationSample, EncodedData, HuffmanDeviationData, RleDeviationData,
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
        compressed.layout = layout;
        compressed.layout.selected_base_bit_positions = selected_positions;
        compressed.entropy_sorted_column_order = None;
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

pub struct DeltaEncodeBaseTable {}

#[derive(Debug, Clone, Copy, Default)]
struct DeltaBitStats {
    total_written_bits: usize,
    prefix_bits: usize,
    payload_bits: usize,
    minimally_necessary_bits: usize,
}

impl Filter for DeltaEncodeBaseTable {
    type Input = CompressedData;
    type Output = CompressedData;

    fn process(&self, mut input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Delta encoding base table (post-encoding)");

        let Some(order) = input.entropy_sorted_column_order.clone() else {
            tracing::warn!(
                "DeltaEncodeBaseTable skipped: missing entropy_sorted_column_order; keeping raw base table"
            );
            if !matches!(input.base_table, BaseTable::Raw(_)) {
                input.base_table = BaseTable::Raw(input.base_table.as_raw().to_vec());
            }
            return Ok(input);
        };

        let raw_rows = input.base_table.as_raw().to_vec();
        if raw_rows.is_empty() {
            input.base_table = BaseTable::Delta(DeltaBaseTableData {
                raw_rows,
                first_sort_key: crate::BitStream::new(),
                delta_bit_stream: crate::BitStream::new(),
                delta_count: 0,
                sort_column_order: order,
            });
            return Ok(input);
        }

        let row_width = raw_rows[0].0.len();
        if order.iter().any(|&idx| idx >= row_width) {
            tracing::warn!(
                row_width,
                "DeltaEncodeBaseTable skipped: entropy order contains out-of-range column index; keeping raw base table"
            );
            input.base_table = BaseTable::Raw(raw_rows);
            return Ok(input);
        }

        let mut key_rows = Vec::with_capacity(raw_rows.len());
        for (row_bits, _) in &raw_rows {
            key_rows.push(build_sort_key_bits(row_bits.as_bitslice(), &order));
        }

        let first_sort_key = key_rows[0].clone();
        let mut delta_bit_stream = crate::BitStream::new();
        let mut delta_stats = DeltaBitStats::default();
        for pair in key_rows.windows(2) {
            let prev = pair[0].as_bitslice();
            let curr = pair[1].as_bitslice();

            if compare_unsigned(prev, curr) == std::cmp::Ordering::Less {
                tracing::warn!(
                    "DeltaEncodeBaseTable skipped: base rows are not monotonic for descending key deltas; keeping raw base table"
                );
                input.base_table = BaseTable::Raw(raw_rows);
                return Ok(input);
            }

            let delta = subtract_unsigned(prev, curr);
            if is_zero_bits(delta.as_bitslice()) {
                tracing::warn!(
                    "DeltaEncodeBaseTable skipped: encountered zero delta (expected unique sorted rows); keeping raw base table"
                );
                input.base_table = BaseTable::Raw(raw_rows);
                return Ok(input);
            }

            let d = subtract_one(delta.as_bitslice());
            let stats = encode_adjusted_delta_bits_with_stats(
                d.as_bitslice(),
                row_width,
                &mut delta_bit_stream,
            );
            delta_stats.total_written_bits += stats.total_written_bits;
            delta_stats.prefix_bits += stats.prefix_bits;
            delta_stats.payload_bits += stats.payload_bits;
            delta_stats.minimally_necessary_bits += stats.minimally_necessary_bits;
        }

        let raw_base_table_bits = raw_rows.len() * row_width;
        let delta_base_table_bits = first_sort_key.len() + delta_bit_stream.len();
        let delta_count = key_rows.len().saturating_sub(1);

        let quantization_overhead_bits = delta_stats
            .payload_bits
            .saturating_sub(delta_stats.minimally_necessary_bits);
        let total_overhead_bits = delta_stats
            .total_written_bits
            .saturating_sub(delta_stats.minimally_necessary_bits);

        let encoded_id_deviation_bits = input.encoded_data.get_encoded_size();
        let combined_bits = encoded_id_deviation_bits + delta_base_table_bits;
        let base_table_share_pct = if combined_bits == 0 {
            0.0
        } else {
            100.0 * delta_base_table_bits as f64 / combined_bits as f64
        };
        let id_deviation_share_pct = if combined_bits == 0 {
            0.0
        } else {
            100.0 * encoded_id_deviation_bits as f64 / combined_bits as f64
        };

        let raw_to_delta_ratio = if raw_base_table_bits == 0 {
            0.0
        } else {
            delta_base_table_bits as f64 / raw_base_table_bits as f64
        };

        tracing::debug!(
            raw_base_table_bits,
            delta_base_table_bits,
            delta_count,
            row_width,
            raw_to_delta_ratio,
            "Delta base-table size summary"
        );
        tracing::debug!(
            minimally_necessary_bits = delta_stats.minimally_necessary_bits,
            unary_prefix_overhead_bits = delta_stats.prefix_bits,
            quantization_overhead_bits,
            total_overhead_bits,
            encoded_delta_bits = delta_stats.total_written_bits,
            "Delta coding overhead summary"
        );
        tracing::debug!(
            encoded_base_table_bits = delta_base_table_bits,
            encoded_id_deviation_bits,
            base_table_share_pct,
            id_deviation_share_pct,
            combined_bits,
            "Compressed payload composition (base-table vs id/deviation)"
        );

        input.base_table = BaseTable::Delta(DeltaBaseTableData {
            raw_rows,
            first_sort_key,
            delta_bit_stream,
            delta_count: key_rows.len().saturating_sub(1),
            sort_column_order: order,
        });
        Ok(input)
    }
}

fn build_compressed_data(input: PreEncodeContext, encoded_data: EncodedData) -> CompressedData {
    let mut compressed = CompressedData::new(encoded_data, input.bit_data.info.clone());
    compressed.base_table = BaseTable::Raw(input.variable_base_table);
    compressed.layout = input.layout;
    compressed.entropy_sorted_column_order = input.entropy_sorted_column_order;
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

fn build_sort_key_bits(row: &crate::BitView, order: &[usize]) -> crate::BitStream {
    let mut out = crate::BitStream::with_capacity(order.len());
    for &col_idx in order.iter().rev() {
        out.push(row.get(col_idx).map(|b| *b).unwrap_or(false));
    }
    out
}

fn is_zero_bits(bits: &crate::BitView) -> bool {
    bits.not_any()
}

fn compare_unsigned(lhs: &crate::BitView, rhs: &crate::BitView) -> std::cmp::Ordering {
    let max_len = lhs.len().max(rhs.len());
    for idx in (0..max_len).rev() {
        let l = lhs.get(idx).map(|b| *b).unwrap_or(false);
        let r = rhs.get(idx).map(|b| *b).unwrap_or(false);
        match l.cmp(&r) {
            std::cmp::Ordering::Equal => continue,
            non_equal => return non_equal,
        }
    }
    std::cmp::Ordering::Equal
}

fn subtract_unsigned(minuend: &crate::BitView, subtrahend: &crate::BitView) -> crate::BitStream {
    let max_len = minuend.len().max(subtrahend.len());
    let mut out = crate::BitStream::with_capacity(max_len);

    let mut borrow: i8 = 0;
    for idx in 0..max_len {
        let a = if minuend.get(idx).map(|b| *b).unwrap_or(false) {
            1
        } else {
            0
        };
        let b = if subtrahend.get(idx).map(|b| *b).unwrap_or(false) {
            1
        } else {
            0
        };

        let mut diff = a - b - borrow;
        if diff < 0 {
            diff += 2;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(diff == 1);
    }

    while out.last().map(|bit| *bit) == Some(false) {
        out.pop();
    }

    out
}

fn subtract_one(bits: &crate::BitView) -> crate::BitStream {
    let mut out = bits.to_bitvec();
    for idx in 0..out.len() {
        if out[idx] {
            out.set(idx, false);
            break;
        }
        out.set(idx, true);
    }
    while out.last().map(|bit| *bit) == Some(false) {
        out.pop();
    }
    out
}

fn bits_to_u64(bits: &crate::BitView) -> u64 {
    let mut value = 0u64;
    for idx in (0..bits.len()).rev() {
        value = (value << 1) | (bits.get(idx).map(|b| *b).unwrap_or(false) as u64);
    }
    value
}

// fn write_u64_lsb_bits(value: u64, width: usize, out: &mut crate::BitStream) {
//     for shift in 0..width {
//         out.push(((value >> shift) & 1) == 1);
//     }
// }

fn write_u64_bits<O: BitOrder>(value: u64, width: usize, out: &mut BitVec<usize, O>) {
    let bits = value.view_bits::<O>();
    out.extend_from_bitslice(&bits[..width]);
}

fn encode_adjusted_delta_bits_with_stats(
    d_bits: &crate::BitView,
    lb: usize,
    out: &mut crate::BitStream,
) -> DeltaBitStats {
    let mut stats = DeltaBitStats {
        minimally_necessary_bits: d_bits.len(),
        ..DeltaBitStats::default()
    };

    let d_len = d_bits.len();

    if d_len <= 38 {
        let d = bits_to_u64(d_bits);

        const STARTS: [u64; 8] = [0, 4, 20, 84, 1108, 9300, 74772, 598068];
        const WIDTHS: [usize; 8] = [2, 4, 7, 10, 13, 16, 19, 38];

        for tier in 0..8 {
            let start = STARTS[tier];
            let width = WIDTHS[tier];
            let max = start + ((1u128 << width) as u64) - 1;
            if d <= max {
                for _ in 0..tier {
                    out.push(true);
                }
                out.push(false);
                write_u64_bits(d - start, width, out);
                stats.prefix_bits = tier + 1;
                stats.payload_bits = width;
                stats.total_written_bits = stats.prefix_bits + stats.payload_bits;
                return stats;
            }
        }
    }

    for _ in 0..8 {
        out.push(true);
    }
    for idx in 0..lb {
        out.push(d_bits.get(idx).map(|b| *b).unwrap_or(false));
    }
    stats.prefix_bits = 8;
    stats.payload_bits = lb;
    stats.total_written_bits = stats.prefix_bits + stats.payload_bits;
    stats
}
