use crate::compression::base_bits::BaseBit;
use crate::compression::compress::{CompressedData, DeviationData, EncodedData, RleDeviationData};
use crate::compression::tabular_preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use bitvec::prelude::*;

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
        while i + run_len < num_rows && run_len < 255 {
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
        while i < num_rows && literal_count < 254 {
            let this_symbol = make_row_symbol(bit_data, &context, i);

            let mut lookahead_run = 1usize;
            while i + lookahead_run < num_rows && lookahead_run < 255 {
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
