use bitvec::prelude::*;

use super::encoding_core::{BaseTable, CompressedData, DeltaBaseTableData};

use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;

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
                first_sort_key: BitVec::new(),
                delta_bit_stream: BitVec::new(),
                delta_count: 0,
                sort_column_order: order,
                codec_id: 1, // BASE_TABLE_TAG_DELTA_UNARY
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
        let mut delta_bit_stream = BitVec::new();
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
            delta_count,
            sort_column_order: order,
            codec_id: 1, // BASE_TABLE_TAG_DELTA_UNARY
        });
        Ok(input)
    }
}

pub struct DeltaEncodeBaseTableFixed {}

impl Filter for DeltaEncodeBaseTableFixed {
    type Input = CompressedData;
    type Output = CompressedData;

    fn process(&self, mut input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Delta encoding base table with fixed prefix (post-encoding)");

        let Some(order) = input.entropy_sorted_column_order.clone() else {
            tracing::warn!(
                "DeltaEncodeBaseTableFixed skipped: missing entropy_sorted_column_order; keeping raw base table"
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
                first_sort_key: BitVec::new(),
                delta_bit_stream: BitVec::new(),
                delta_count: 0,
                sort_column_order: order,
                codec_id: 2, // BASE_TABLE_TAG_DELTA_FIXED
            });
            return Ok(input);
        }

        let row_width = raw_rows[0].0.len();
        if order.iter().any(|&idx| idx >= row_width) {
            tracing::warn!(
                row_width,
                "DeltaEncodeBaseTableFixed skipped: entropy order contains out-of-range column index; keeping raw base table"
            );
            input.base_table = BaseTable::Raw(raw_rows);
            return Ok(input);
        }

        let mut key_rows = Vec::with_capacity(raw_rows.len());
        for (row_bits, _) in &raw_rows {
            key_rows.push(build_sort_key_bits(row_bits.as_bitslice(), &order));
        }

        let first_sort_key = key_rows[0].clone();
        let mut delta_bit_stream = BitVec::new();
        let mut delta_stats = DeltaBitStats::default();
        for pair in key_rows.windows(2) {
            let prev = pair[0].as_bitslice();
            let curr = pair[1].as_bitslice();

            if compare_unsigned(prev, curr) == std::cmp::Ordering::Less {
                tracing::warn!(
                    "DeltaEncodeBaseTableFixed skipped: base rows are not monotonic for descending key deltas; keeping raw base table"
                );
                input.base_table = BaseTable::Raw(raw_rows);
                return Ok(input);
            }

            let delta = subtract_unsigned(prev, curr);
            if is_zero_bits(delta.as_bitslice()) {
                tracing::warn!(
                    "DeltaEncodeBaseTableFixed skipped: encountered zero delta (expected unique sorted rows); keeping raw base table"
                );
                input.base_table = BaseTable::Raw(raw_rows);
                return Ok(input);
            }

            let d = subtract_one(delta.as_bitslice());
            let stats = encode_adjusted_delta_bits_with_stats_fixed(
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
            "Delta base-table size summary (fixed prefix)"
        );
        tracing::debug!(
            minimally_necessary_bits = delta_stats.minimally_necessary_bits,
            fixed_prefix_overhead_bits = delta_stats.prefix_bits,
            quantization_overhead_bits,
            total_overhead_bits,
            encoded_delta_bits = delta_stats.total_written_bits,
            "Delta coding overhead summary (fixed prefix)"
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
            delta_count,
            sort_column_order: order,
            codec_id: 2, // BASE_TABLE_TAG_DELTA_FIXED
        });
        Ok(input)
    }
}

fn build_sort_key_bits(row: &crate::BitView, order: &[usize]) -> BitVec<usize, Lsb0>{
    let mut out = BitVec::with_capacity(order.len());
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

fn subtract_unsigned(minuend: &crate::BitView, subtrahend: &crate::BitView) -> BitVec<usize, Lsb0>{
    let max_len = minuend.len().max(subtrahend.len());
    let mut out = BitVec::with_capacity(max_len);

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

fn subtract_one(bits: &crate::BitView) -> BitVec<usize, Lsb0> {
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

fn write_u64_bits<O: BitOrder>(value: u64, width: usize, out: &mut BitVec<usize, O>) {
    let bits = value.view_bits::<O>();
    out.extend_from_bitslice(&bits[..width]);
}

// Runs at compile time
// Used to quickly change the unary prefix lengths and payload bit widths for delta encoding
pub const fn get_delta_codec() -> [(usize, u64); 5] {
    const CODE_NUM: usize = 5;
    const BIT_WIDTHS: [usize; 5] = [2, 5, 16, 32, 48];
    let mut codec = [(0usize, 0u64); CODE_NUM];
    let mut starts = [0u64; CODE_NUM];
    let mut cumulative = 0u64;
    let mut i = 0;
    while i < CODE_NUM {
        starts[i] = cumulative;
        cumulative += 1 << BIT_WIDTHS[i];
        codec[i] = (BIT_WIDTHS[i], starts[i]);
        i += 1;
    }

    codec
}

// Fixed 4-bit prefix code for delta encoding (16 tiers)
// Runs at compile time with zero runtime overhead
pub const fn get_delta_codec_fixed() -> [(usize, u64); 16] {
    const CODE_NUM: usize = 16;
    const BIT_WIDTHS: [usize; 16] = [1, 3, 6, 8, 14, 16, 18, 20, 22, 24, 26, 28, 31, 35, 42, 48];
    let mut codec = [(0usize, 0u64); CODE_NUM];
    let mut starts = [0u64; CODE_NUM];
    let mut cumulative = 0u64;
    let mut i = 0;
    while i < CODE_NUM - 1 {  // Don't overflow on the last tier
        starts[i] = cumulative;
        cumulative += 1 << BIT_WIDTHS[i];
        codec[i] = (BIT_WIDTHS[i], starts[i]);
        i += 1;
    }
    // Last tier: handled as overflow, doesn't need a computed start
    codec[CODE_NUM - 1] = (BIT_WIDTHS[CODE_NUM - 1], cumulative);

    codec
}

// Generic encoding function: takes codec array, prefix bit width, and overflow indicator
fn encode_adjusted_delta_bits_generic(
    d_bits: &crate::BitView,
    lb: usize,
    out: &mut BitVec<usize, Lsb0>,
    codec: &[(usize, u64)],
    prefix_bits: usize,
    use_unary_prefix: bool,
) -> DeltaBitStats {
    let mut stats = DeltaBitStats {
        minimally_necessary_bits: d_bits.len(),
        ..DeltaBitStats::default()
    };

    let d_len = d_bits.len();
    let overflow_tier = codec.len();

    if d_len <= 38 {
        let d = bits_to_u64(d_bits);

        for tier in 0..codec.len() {
            let (width, start) = codec[tier];
            let max = start + ((1u128 << width) as u64) - 1;
            if d <= max {
                // Write prefix (either unary or fixed)
                if use_unary_prefix {
                    // Unary: tier 0s followed by a 0
                    for _ in 0..tier {
                        out.push(true);
                    }
                    out.push(false);
                    stats.prefix_bits = tier + 1;
                } else {
                    // Fixed: tier as fixed-width value
                    for i in 0..prefix_bits {
                        out.push(((tier >> i) & 1) == 1);
                    }
                    stats.prefix_bits = prefix_bits;
                }
                write_u64_bits(d - start, width, out);
                stats.payload_bits = width;
                stats.total_written_bits = stats.prefix_bits + stats.payload_bits;
                return stats;
            }
        }
    }

    // Overflow case
    if use_unary_prefix {
        // Unary overflow: all 1s in prefix
        for _ in 0..overflow_tier {
            out.push(true);
        }
        stats.prefix_bits = overflow_tier;
    } else {
        // Fixed overflow: fixed pattern for overflow tier ID
        let overflow_id = overflow_tier - 1;
        for i in 0..prefix_bits {
            out.push(((overflow_id >> i) & 1) == 1);
        }
        stats.prefix_bits = prefix_bits;
    }
    for idx in 0..lb {
        out.push(d_bits.get(idx).map(|b| *b).unwrap_or(false));
    }
    stats.payload_bits = lb;
    stats.total_written_bits = stats.prefix_bits + stats.payload_bits;
    stats
}

fn encode_adjusted_delta_bits_with_stats(
    d_bits: &crate::BitView,
    lb: usize,
    out: &mut BitVec<usize, Lsb0>,
) -> DeltaBitStats {
    const CODEC: [(usize, u64); 5] = get_delta_codec();
    encode_adjusted_delta_bits_generic(
        d_bits,
        lb,
        out,
        &CODEC,
        5,     // prefix_bits (not used for unary, but kept for signature)
        true,  // use_unary_prefix
    )
}

fn encode_adjusted_delta_bits_with_stats_fixed(
    d_bits: &crate::BitView,
    lb: usize,
    out: &mut BitVec<usize, Lsb0>,
) -> DeltaBitStats {
    const CODEC: [(usize, u64); 16] = get_delta_codec_fixed();
    const PREFIX_BITS: usize = 4; // log2(16 tiers) = 4 bits
    encode_adjusted_delta_bits_generic(
        d_bits,
        lb,
        out,
        &CODEC,
        PREFIX_BITS,
        false, // use_unary_prefix
    )
}
