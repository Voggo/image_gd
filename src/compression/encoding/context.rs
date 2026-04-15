use crate::compression::base_table::PreEncodeContext;
use crate::compression::preprocessor::BitDataSet;
use crate::utils::bits_needed_nonzero;

pub(super) struct EncodingContext {
    pub(super) l_id: usize,
    pub(super) num_deviation_bits: usize,
    pub(super) row_to_group_id: Vec<usize>,
    pub(super) deviation_ranges: Vec<(usize, usize)>,
    pub(super) id_bits_per_base: Vec<crate::BitStream>,
    pub(super) _entropy_sorted_column_order: Option<Vec<usize>>,
}

pub(super) fn build_deviation_ranges(
    base_bit_mask: &crate::BitView,
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

pub(super) fn build_encoding_context(input: &PreEncodeContext) -> EncodingContext {
    let chunk_size = input.bit_data.chunk_size();
    let num_bits_per_base = input.layout.selected_base_bit_positions.len();
    let num_deviation_bits = chunk_size.saturating_sub(num_bits_per_base);
    let num_bases = input.variable_base_table.len();
    let l_id = bits_needed_nonzero(num_bases);

    let mut base_bit_mask = crate::BitStream::repeat(false, chunk_size);
    for &bit_pos in &input.layout.selected_base_bit_positions {
        if bit_pos < chunk_size {
            base_bit_mask.set(bit_pos, true);
        }
    }
    let deviation_ranges =
        build_deviation_ranges(base_bit_mask.as_bitslice(), chunk_size, num_deviation_bits);

    let mut id_bits_per_base: Vec<crate::BitStream> = Vec::new();
    if l_id > 0 {
        id_bits_per_base = Vec::with_capacity(num_bases);
        for id in 0..num_bases {
            let mut id_bits = crate::BitStream::with_capacity(l_id);
            for shift in (0..l_id).rev() {
                id_bits.push(((id >> shift) & 1) == 1);
            }
            id_bits_per_base.push(id_bits);
        }
    }

    EncodingContext {
        l_id,
        num_deviation_bits,
        row_to_group_id: input.row_to_base_id.clone(),
        deviation_ranges,
        id_bits_per_base,
        _entropy_sorted_column_order: input.entropy_sorted_column_order.clone(),
    }
}

pub(super) fn encode_rows_as_symbol_stream(
    bit_data: &BitDataSet,
    context: &EncodingContext,
) -> crate::BitStream {
    let symbol_width = context.num_deviation_bits + context.l_id;
    let num_rows = bit_data.num_rows();
    let mut symbol_stream = crate::BitStream::with_capacity(num_rows * symbol_width);

    for (row, id_ref) in context.row_to_group_id.iter().enumerate().take(num_rows) {
        let id = *id_ref;
        let chunk = unsafe { bit_data.get_chunk_unchecked(row) };

        for &(start, end) in &context.deviation_ranges {
            symbol_stream.extend_from_bitslice(unsafe { chunk.get_unchecked(start..end) });
        }

        if context.l_id > 0 {
            symbol_stream.extend_from_bitslice(context.id_bits_per_base[id].as_bitslice());
        }
    }

    symbol_stream
}
