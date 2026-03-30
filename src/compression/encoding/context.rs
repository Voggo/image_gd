use bitvec::prelude::*;

use crate::compression::base_bits::BaseBit;
use crate::compression::preprocessor::BitDataSet;
use crate::utils::bits_needed_nonzero;

pub(super) struct EncodingContext {
    pub(super) l_id: usize,
    pub(super) num_deviation_bits: usize,
    pub(super) row_to_group_id: Vec<usize>,
    pub(super) deviation_ranges: Vec<(usize, usize)>,
    pub(super) id_bits_per_base: Vec<BitVec<usize, Msb0>>,
}

pub(super) fn build_deviation_ranges(
    base_bit_mask: &BitSlice<usize, Msb0>,
    chunk_size: usize,
    num_deviation_bits: usize,
) -> Vec<(usize, usize)> {
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

    deviation_ranges
}

pub(super) fn prepare_encoding_context<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> EncodingContext {
    let base_bit_mask = base_bit_groups.get_base_bit_mask();
    let num_bases = base_bit_groups.get_num_bases();
    let l_id = bits_needed_nonzero(num_bases);

    let chunk_size = bit_data.chunk_size();
    let num_bits_per_base = base_bit_groups.get_num_bits_per_base();
    let num_deviation_bits = chunk_size.saturating_sub(num_bits_per_base);

    let deviation_ranges = build_deviation_ranges(base_bit_mask, chunk_size, num_deviation_bits);

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

pub(super) fn encode_rows_as_symbol_stream(
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
