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
            delta_count,
            sort_column_order: order,
        });
        Ok(input)
    }
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
