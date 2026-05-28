use bitvec::prelude::*;

use super::encoding_core::{BaseTable, CompressedData, DeltaBaseTableData};

use crate::compression::file_format::tags::{BASE_TABLE_TAG_DELTA_FIXED, BASE_TABLE_TAG_DELTA_UNARY};
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

fn bits_to_u128(bits: &BitSlice<usize, Lsb0>) -> u128 {
    if bits.is_empty() {
        0
    } else {
        bits.load_le::<u128>()
    }
}

fn write_u128_bits<O: BitOrder>(value: u128, width: usize, out: &mut BitVec<usize, O>) {
    let bytes = value.to_le_bytes();
    let bits = bytes.view_bits::<O>();
    out.extend_from_bitslice(&bits[..width]);
}

// Runs at compile time
// Used to quickly change the unary prefix lengths and payload bit widths for delta encoding
pub const fn get_delta_codec() -> [(usize, u128); 5] {
    const CODE_NUM: usize = 5;
    const BIT_WIDTHS: [usize; 5] = [2, 5, 16, 32, 48];
    let mut codec = [(0usize, 0u128); CODE_NUM];
    let mut starts = [0u128; CODE_NUM];
    let mut cumulative = 0u128;
    let mut i = 0;
    while i < CODE_NUM {
        starts[i] = cumulative;
        cumulative += 1u128 << BIT_WIDTHS[i];
        codec[i] = (BIT_WIDTHS[i], starts[i]);
        i += 1;
    }

    codec
}

// Fixed 5-bit prefix code for delta encoding (32 tiers)
// Runs at compile time with zero runtime overhead
pub const fn get_delta_codec_fixed() -> [(usize, u128); 32] {
    const CODE_NUM: usize = 32;
    const BIT_WIDTHS: [usize; 32] = [
        2, 6, 9, 12, 15, 18, 21, 25, 29, 33, 37, 41, 45, 49, 53, 58, 63, 68, 74, 80, 86, 93, 101,
        111, 120, 130, 142, 153, 171, 183, 202, 231,
    ];
    let mut codec = [(0usize, 0u128); CODE_NUM];
    let mut starts = [0u128; CODE_NUM];
    let mut cumulative = 0u128;
    let mut i = 0;
    while i < CODE_NUM - 1 {
        // Don't overflow on the last tier
        starts[i] = cumulative;
        if BIT_WIDTHS[i] >= 128 {
            cumulative = u128::MAX;
        } else {
            cumulative = cumulative.saturating_add(1u128 << BIT_WIDTHS[i]);
        }
        codec[i] = (BIT_WIDTHS[i], starts[i]);
        i += 1;
    }
    // Last tier: handled as overflow, doesn't need a computed start
    codec[CODE_NUM - 1] = (BIT_WIDTHS[CODE_NUM - 1], cumulative);

    codec
}

// Generic encoding function: takes codec array, prefix bit width, and overflow indicator
fn encode_adjusted_delta_bits_generic(
    d_bits: &BitSlice<usize, Lsb0>,
    lb: usize,
    out: &mut BitVec<usize, Lsb0>,
    codec: &[(usize, u128)],
    prefix_bits: usize,
    use_unary_prefix: bool,
) -> DeltaBitStats {
    let mut stats = DeltaBitStats {
        minimally_necessary_bits: d_bits.len(),
        ..DeltaBitStats::default()
    };

    let d_len = d_bits.len();
    let overflow_tier = codec.len();

    if d_len <= 127 {
        let d = bits_to_u128(d_bits);

        for tier in 0..codec.len() {
            let (width, start) = codec[tier];

            // Skip tiers that overflow u128 bounds
            if width >= 128 || start == u128::MAX {
                break;
            }

            let max = start.saturating_add((1u128 << width) - 1);
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
                write_u128_bits(d - start, width, out);
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
    d_bits: &BitSlice<usize, Lsb0>,
    lb: usize,
    out: &mut BitVec<usize, Lsb0>,
) -> DeltaBitStats {
    const CODEC: [(usize, u128); 5] = get_delta_codec();
    encode_adjusted_delta_bits_generic(
        d_bits, lb, out, &CODEC,
        5,    // prefix_bits (not used for unary, but kept for signature)
        true, // use_unary_prefix
    )
}

fn encode_adjusted_delta_bits_with_stats_fixed(
    d_bits: &BitSlice<usize, Lsb0>,
    lb: usize,
    out: &mut BitVec<usize, Lsb0>,
) -> DeltaBitStats {
    const CODEC: [(usize, u128); 32] = get_delta_codec_fixed();
    const PREFIX_BITS: usize = 5; // log2(32 tiers) = 5 bits
    encode_adjusted_delta_bits_generic(
        d_bits,
        lb,
        out,
        &CODEC,
        PREFIX_BITS,
        false, // use_unary_prefix
    )
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
    const CODEC: [(usize, u128); 5] = get_delta_codec();
    let overflow_tier = CODEC.len();

    let mut tier = 0usize;
    while *bit_pos < bits.len() && bits[*bit_pos] {
        tier += 1;
        *bit_pos += 1;
        if tier == overflow_tier {
            break;
        }
    }

    if tier < overflow_tier {
        if *bit_pos >= bits.len() || bits[*bit_pos] {
            return Err(EntroGdError::InvalidMetadata {
                message: "invalid delta prefix terminator".to_string(),
            });
        }
        *bit_pos += 1;
    }

    let (payload_width, start): (usize, u128) = if tier < CODEC.len() {
        CODEC[tier]
    } else if tier == overflow_tier {
        (lb, 0)
    } else {
        return Err(EntroGdError::InvalidMetadata {
            message: "invalid delta tier".to_string(),
        });
    };

    if *bit_pos + payload_width > bits.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta payload exceeds bitstream".to_string(),
        });
    }

    let payload = unsafe { bits.get_unchecked(*bit_pos..*bit_pos + payload_width) };
    *bit_pos += payload_width;

    if tier == overflow_tier {
        return Ok(payload.to_bitvec());
    }

    let mut payload_value = 0u128;
    for idx in 0..payload_width {
        if payload[idx] {
            payload_value |= 1u128 << idx;
        }
    }
    let value = start + payload_value;

    let mut out = BitVec::new();
    let mut v = value;
    while v > 0 {
        out.push((v & 1) == 1);
        v >>= 1;
    }
    Ok(out)
}

fn decode_adjusted_delta_fixed(
    bits: &BitSlice<usize, Lsb0>,
    bit_pos: &mut usize,
    lb: usize,
) -> Result<BitVec<usize, Lsb0>, EntroGdError> {
    const CODEC: [(usize, u128); 32] = get_delta_codec_fixed();
    const PREFIX_BITS: usize = 5;
    const OVERFLOW_TIER: usize = CODEC.len() - 1;

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

    if tier >= CODEC.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta tier ID out of range".to_string(),
        });
    }

    let (payload_width, start): (usize, u128) = if tier < CODEC.len() - 1 {
        CODEC[tier]
    } else {
        (lb, 0)
    };

    if *bit_pos + payload_width > bits.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta payload exceeds bitstream".to_string(),
        });
    }

    let payload = unsafe { bits.get_unchecked(*bit_pos..*bit_pos + payload_width) };
    *bit_pos += payload_width;

    if tier == OVERFLOW_TIER {
        return Ok(payload.to_bitvec());
    }

    let mut payload_value = 0u128;
    for idx in 0..payload_width {
        if payload[idx] {
            payload_value |= 1u128 << idx;
        }
    }
    let value = start + payload_value;

    let mut out = BitVec::new();
    let mut v = value;
    while v > 0 {
        out.push((v & 1) == 1);
        v >>= 1;
    }
    Ok(out)
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
