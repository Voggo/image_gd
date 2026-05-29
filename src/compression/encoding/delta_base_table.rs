use bitvec::prelude::*;

use super::encoding_core::{BaseTable, CompressedData, DeltaBaseTableData};

use crate::compression::file_format::tags::{
    BASE_TABLE_TAG_DELTA_FIXED, BASE_TABLE_TAG_DELTA_UNARY,
};
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
        let num_bases = raw_rows.len();
        if raw_rows.is_empty() {
            input.base_table = BaseTable::Delta(DeltaBaseTableData {
                num_bases: 0,
                first_sort_key: BitVec::new(),
                delta_bit_stream: BitVec::new(),
                delta_count: 0,
                sort_column_order: order,
                codec_id: BASE_TABLE_TAG_DELTA_UNARY,
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

        let raw_base_table_bits = num_bases * row_width;
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
            num_bases,
            first_sort_key,
            delta_bit_stream,
            delta_count,
            sort_column_order: order,
            codec_id: BASE_TABLE_TAG_DELTA_UNARY,
        });
        Ok(input)
    }
}

pub struct DeltaEncodeBaseTableFixed {}

impl Filter for DeltaEncodeBaseTableFixed {
    type Input = CompressedData;
    type Output = CompressedData;

    fn process(&self, mut input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer =
            ScopedTimer::info("Delta encoding base table with fixed prefix (post-encoding)");

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
        let num_bases = raw_rows.len();
        if raw_rows.is_empty() {
            input.base_table = BaseTable::Delta(DeltaBaseTableData {
                num_bases: 0,
                first_sort_key: BitVec::new(),
                delta_bit_stream: BitVec::new(),
                delta_count: 0,
                sort_column_order: order,
                codec_id: BASE_TABLE_TAG_DELTA_FIXED,
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

        let raw_base_table_bits = num_bases * row_width;
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
            num_bases,
            first_sort_key,
            delta_bit_stream,
            delta_count,
            sort_column_order: order,
            codec_id: BASE_TABLE_TAG_DELTA_FIXED,
        });
        Ok(input)
    }
}

fn build_sort_key_bits(row: &BitSlice<usize, Lsb0>, order: &[usize]) -> BitVec<usize, Lsb0> {
    let mut out = BitVec::with_capacity(order.len());
    for &col_idx in order.iter().rev() {
        out.push(row.get(col_idx).map(|b| *b).unwrap_or(false));
    }
    out
}

fn is_zero_bits(bits: &BitSlice<usize, Lsb0>) -> bool {
    bits.not_any()
}

fn compare_unsigned(
    lhs: &BitSlice<usize, Lsb0>,
    rhs: &BitSlice<usize, Lsb0>,
) -> std::cmp::Ordering {
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

fn subtract_unsigned(
    minuend: &BitSlice<usize, Lsb0>,
    subtrahend: &BitSlice<usize, Lsb0>,
) -> BitVec<usize, Lsb0> {
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

fn subtract_one(bits: &BitSlice<usize, Lsb0>) -> BitVec<usize, Lsb0> {
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


// Update these from prefix_scheme.py output if you want
pub const fn get_delta_codec() -> [usize; 30] {
    [
        9, 19, 28, 37, 45, 54, 63, 73, 82, 89, 97, 106, 115, 123, 131, 140, 147, 156, 163, 170,
        175, 180, 185, 189, 193, 197, 202, 207, 210, 212,
    ]
}


// Update these from prefix_scheme.py if you want
pub const fn get_delta_codec_fixed() -> [usize; 64] {
    [
        1, 3, 5, 6, 7, 9, 11, 12, 14, 15, 17, 19, 20, 22, 24, 26, 28, 29, 31, 33, 35, 37, 39, 41,
        43, 45, 47, 49, 51, 54, 57, 60, 63, 65, 68, 71, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100,
        104, 107, 111, 115, 119, 123, 128, 133, 140, 145, 150, 156, 162, 170, 177, 187, 197, 212,
    ]
}

// Subtract 2^n from a BitVec (no u128; bitvec borrow propagation).
// Precondition: bits >= 2^n.
fn subtract_pow2(mut bits: BitVec<usize, Lsb0>, n: usize) -> BitVec<usize, Lsb0> {
    let mut k = n;
    while k < bits.len() && !bits[k] {
        k += 1;
    }
    bits.set(k, false);
    for j in n..k {
        bits.set(j, true);
    }
    while bits.last().map(|b| *b) == Some(false) {
        bits.pop();
    }
    bits
}

// Add 2^n to a BitVec (no u128; bitvec carry propagation).
fn add_pow2(mut bits: BitVec<usize, Lsb0>, n: usize) -> BitVec<usize, Lsb0> {
    if n >= bits.len() {
        bits.resize(n, false);
        bits.push(true);
        return bits;
    }
    let mut idx = n;
    loop {
        if idx >= bits.len() {
            bits.push(true);
            break;
        }
        let bit = bits[idx];
        bits.set(idx, !bit);
        if !bit {
            break;
        }
        idx += 1;
    }
    bits
}

// Incremental encoding: check remainder.len() <= width at each tier, subtracting
// 2^width when it doesn't fit. No u128 needed; works for any lb.
fn encode_adjusted_delta_bits_generic(
    d_bits: &BitSlice<usize, Lsb0>,
    lb: usize,
    out: &mut BitVec<usize, Lsb0>,
    tier_widths: &[usize],
    prefix_bits: usize,
    use_unary_prefix: bool,
) -> DeltaBitStats {
    let d_len = d_bits.len();
    let n_active = tier_widths.len();
    let mut remainder = d_bits.to_bitvec();

    for (tier, &width) in tier_widths.iter().enumerate() {
        if remainder.len() <= width {
            let p_bits = if use_unary_prefix {
                for _ in 0..tier {
                    out.push(true);
                }
                out.push(false);
                tier + 1
            } else {
                for i in 0..prefix_bits {
                    out.push(((tier >> i) & 1) == 1);
                }
                prefix_bits
            };
            for idx in 0..width {
                out.push(remainder.get(idx).map(|b| *b).unwrap_or(false));
            }
            return DeltaBitStats {
                minimally_necessary_bits: d_len,
                prefix_bits: p_bits,
                payload_bits: width,
                total_written_bits: p_bits + width,
            };
        }
        remainder = subtract_pow2(remainder, width);
    }

    // Overflow: write the original d in lb raw bits (not the modified remainder).
    let p_bits = if use_unary_prefix {
        for _ in 0..n_active {
            out.push(true);
        }
        n_active
    } else {
        for i in 0..prefix_bits {
            out.push(((n_active >> i) & 1) == 1);
        }
        prefix_bits
    };
    for idx in 0..lb {
        out.push(d_bits.get(idx).map(|b| *b).unwrap_or(false));
    }
    DeltaBitStats {
        minimally_necessary_bits: d_len,
        prefix_bits: p_bits,
        payload_bits: lb,
        total_written_bits: p_bits + lb,
    }
}

fn encode_adjusted_delta_bits_with_stats(
    d_bits: &BitSlice<usize, Lsb0>,
    lb: usize,
    out: &mut BitVec<usize, Lsb0>,
) -> DeltaBitStats {
    const TIER_WIDTHS: [usize; 30] = get_delta_codec();
    encode_adjusted_delta_bits_generic(d_bits, lb, out, &TIER_WIDTHS, 5, true)
}

fn encode_adjusted_delta_bits_with_stats_fixed(
    d_bits: &BitSlice<usize, Lsb0>,
    lb: usize,
    out: &mut BitVec<usize, Lsb0>,
) -> DeltaBitStats {
    const TIER_WIDTHS: [usize; 64] = get_delta_codec_fixed();
    // 6-bit prefix supports tiers 0..62 as normal; tier 63 (= 0b111111 = all-ones) is overflow.
    encode_adjusted_delta_bits_generic(d_bits, lb, out, &TIER_WIDTHS[..63], 6, false)
}

// ── Delta decode ─────────────────────────────────────────────────────────────

impl DeltaBaseTableData {
    pub fn decode_rows(&self) -> Result<Vec<(BitVec<usize, Lsb0>, usize)>, EntroGdError> {
        decode_delta_base_rows(
            self.num_bases,
            self.first_sort_key.len(),
            &self.sort_column_order,
            &self.first_sort_key,
            self.delta_count,
            &self.delta_bit_stream,
            self.codec_id,
        )
    }
}

fn decode_delta_base_rows(
    num_bases: usize,
    lb: usize,
    order: &[usize],
    first_sort_key: &BitSlice<usize, Lsb0>,
    delta_count: usize,
    delta_bit_stream: &BitSlice<usize, Lsb0>,
    codec_tag: u8,
) -> Result<Vec<(BitVec<usize, Lsb0>, usize)>, EntroGdError> {
    let _timer = ScopedTimer::debug("Decoding delta-encoded base table rows");
    if num_bases == 0 {
        return Ok(Vec::new());
    }

    if delta_count != num_bases.saturating_sub(1) {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "delta_count mismatch: expected {}, got {}",
                num_bases.saturating_sub(1),
                delta_count
            ),
        });
    }

    if order.iter().any(|&idx| idx >= lb) {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta sort column order contains out-of-range index".to_string(),
        });
    }

    let mut rows = Vec::with_capacity(num_bases);
    let mut prev_key = first_sort_key.to_bitvec();
    rows.push((sort_key_to_row(&prev_key, order, lb), 0usize));

    let mut bit_pos = 0usize;
    for _ in 0..delta_count {
        let d = decode_adjusted_delta(delta_bit_stream, &mut bit_pos, lb, codec_tag)?;
        let delta = add_one(d.as_bitslice());
        let next_key = subtract_unsigned(prev_key.as_bitslice(), delta.as_bitslice());
        rows.push((sort_key_to_row(&next_key, order, lb), 0usize));
        prev_key = next_key;
    }

    if bit_pos != delta_bit_stream.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta bitstream has trailing/unused bits".to_string(),
        });
    }

    Ok(rows)
}

fn sort_key_to_row(
    sort_key: &BitVec<usize, Lsb0>,
    order: &[usize],
    lb: usize,
) -> BitVec<usize, Lsb0> {
    let mut row = BitVec::repeat(false, lb);
    for (rank, &col_idx) in order.iter().enumerate() {
        let key_idx = lb.saturating_sub(1 + rank);
        let bit = sort_key.get(key_idx).map(|b| *b).unwrap_or(false);
        if col_idx < lb {
            row.set(col_idx, bit);
        }
    }
    row
}

fn decode_adjusted_delta(
    bits: &BitSlice<usize, Lsb0>,
    bit_pos: &mut usize,
    lb: usize,
    codec_tag: u8,
) -> Result<BitVec<usize, Lsb0>, EntroGdError> {
    if codec_tag == BASE_TABLE_TAG_DELTA_UNARY {
        decode_adjusted_delta_unary(bits, bit_pos, lb)
    } else if codec_tag == BASE_TABLE_TAG_DELTA_FIXED {
        decode_adjusted_delta_fixed(bits, bit_pos, lb)
    } else {
        Err(EntroGdError::InvalidMetadata {
            message: format!("unsupported delta codec tag {}", codec_tag),
        })
    }
}

fn decode_adjusted_delta_unary(
    bits: &BitSlice<usize, Lsb0>,
    bit_pos: &mut usize,
    lb: usize,
) -> Result<BitVec<usize, Lsb0>, EntroGdError> {
    const TIER_WIDTHS: [usize; 30] = get_delta_codec();
    let n_active = TIER_WIDTHS.len();

    let mut tier = 0usize;
    while *bit_pos < bits.len() && bits[*bit_pos] {
        tier += 1;
        *bit_pos += 1;
        if tier == n_active {
            break;
        }
    }

    if tier < n_active {
        if *bit_pos >= bits.len() || bits[*bit_pos] {
            return Err(EntroGdError::InvalidMetadata {
                message: "invalid delta prefix terminator".to_string(),
            });
        }
        *bit_pos += 1;
    }

    let payload_width = if tier < n_active {
        TIER_WIDTHS[tier]
    } else {
        lb
    };

    if *bit_pos + payload_width > bits.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta payload exceeds bitstream".to_string(),
        });
    }

    let payload = unsafe { bits.get_unchecked(*bit_pos..*bit_pos + payload_width) };
    *bit_pos += payload_width;

    if tier == n_active {
        return Ok(payload.to_bitvec()); // overflow: payload is d directly
    }

    // d = payload + start_t  where  start_t = sum_{i < tier} 2^TIER_WIDTHS[i]
    let mut d = payload.to_bitvec();
    for i in 0..tier {
        d = add_pow2(d, TIER_WIDTHS[i]);
    }
    Ok(d)
}

fn decode_adjusted_delta_fixed(
    bits: &BitSlice<usize, Lsb0>,
    bit_pos: &mut usize,
    lb: usize,
) -> Result<BitVec<usize, Lsb0>, EntroGdError> {
    const TIER_WIDTHS: [usize; 64] = get_delta_codec_fixed();
    const PREFIX_BITS: usize = 6;
    // 63 normal tiers (0..62); tier 63 = 0b111111 = all-ones in 6 bits = overflow sentinel.
    const N_ACTIVE: usize = TIER_WIDTHS.len() - 1;

    if *bit_pos + PREFIX_BITS > bits.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "insufficient bits for delta tier ID".to_string(),
        });
    }

    let mut tier = 0usize;
    for i in 0..PREFIX_BITS {
        if bits[*bit_pos + i] {
            tier |= 1usize << i;
        }
    }
    *bit_pos += PREFIX_BITS;

    if tier > N_ACTIVE {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta tier ID out of range".to_string(),
        });
    }

    let payload_width = if tier < N_ACTIVE {
        TIER_WIDTHS[tier]
    } else {
        lb
    };

    if *bit_pos + payload_width > bits.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta payload exceeds bitstream".to_string(),
        });
    }

    let payload = unsafe { bits.get_unchecked(*bit_pos..*bit_pos + payload_width) };
    *bit_pos += payload_width;

    if tier == N_ACTIVE {
        return Ok(payload.to_bitvec()); // overflow: payload is d directly
    }

    // d = payload + start_t  where  start_t = sum_{i < tier} 2^TIER_WIDTHS[i]
    let mut d = payload.to_bitvec();
    for i in 0..tier {
        d = add_pow2(d, TIER_WIDTHS[i]);
    }
    Ok(d)
}

fn add_one(bits: &BitSlice<usize, Lsb0>) -> BitVec<usize, Lsb0> {
    let mut out = bits.to_bitvec();
    let mut carry = true;
    let mut idx = 0usize;
    while carry {
        if idx >= out.len() {
            out.push(true);
            break;
        }
        let bit = out[idx];
        out.set(idx, !bit);
        carry = bit;
        idx += 1;
    }
    out
}
