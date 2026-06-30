use bitvec::prelude::*;
use rayon::prelude::*;

use super::encoding_core::{
    BaseTable, CompressedData, DeviationData, EncodedData, build_deviation_ranges,
};

use crate::compression::base_bits::{BaseBit, EncodingContext};
use crate::compression::base_selection::BaseSelectionContext;
use crate::compression::base_table::{
    build_base_layout_from_constant_polarity, project_selected_bases_to_variable,
};
use crate::compression::preprocessor::BitDataSet;
use crate::utils::bits_needed_nonzero;
use crate::{EntroGdError, Filter, ScopedTimer};

pub struct EncodeDataFusedDictionary {}

impl Filter for EncodeDataFusedDictionary {
    type Input = BaseSelectionContext;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(
            "Encoding data into compressed format (fused dictionary + id/deviation)",
        );
        let BaseSelectionContext {
            bit_data,
            base_bits,
            constant_bit_polarity,
        } = input;
        let fused = encode_data_fused_dictionary(&bit_data, base_bits.as_ref());

        let mut compressed = CompressedData::new(
            EncodedData::Normal(fused.deviation_data),
            bit_data.info.clone(),
        );
        let selected_positions = base_bits.get_base_bit_positions().to_vec();
        let layout = build_base_layout_from_constant_polarity(
            bit_data.chunk_size(),
            &selected_positions,
            &constant_bit_polarity.constant_zero_bit_positions,
            &constant_bit_polarity.constant_one_bit_positions,
        );
        let variable_positions = layout.variable_base_bit_positions();
        compressed.base_table = BaseTable::Raw(project_selected_bases_to_variable(
            &selected_positions,
            &variable_positions,
            &fused.base_table,
        ));
        compressed.layout = layout;
        compressed.entropy_sorted_column_order = None;
        compressed.condensed_sample_weights = bit_data
            .info
            .m_condensed_sample_weights()
            .map(|weights| weights.to_vec());
        Ok(compressed)
    }
}

pub(super) struct FusedEncodingResult {
    pub(super) deviation_data: DeviationData,
    pub(super) base_table: Vec<(BitVec<usize, Lsb0>, usize)>,
}

pub(super) fn encode_data_fused_dictionary<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> FusedEncodingResult {
    let base_bit_mask = base_bit_groups.get_base_bit_mask();
    let num_bits_per_base = base_bit_groups.get_num_bits_per_base();
    let chunk_size = bit_data.chunk_size();
    let num_rows = bit_data.num_rows();
    let num_deviation_bits = chunk_size.saturating_sub(num_bits_per_base);
    let deviation_ranges = build_deviation_ranges(base_bit_mask, chunk_size, num_deviation_bits);

    let EncodingContext {
        row_to_id: row_to_group_id,
        base_table,
    } = base_bit_groups.get_encoding_context(bit_data, false);

    let num_bases = base_table.len();
    let l_id = bits_needed_nonzero(num_bases);
    let mut id_bits_per_base: Vec<BitVec<usize, Lsb0>> = Vec::new();
    if l_id > 0 {
        id_bits_per_base = Vec::with_capacity(num_bases);
        for id in 0..num_bases {
            let mut id_bits = BitVec::repeat(false, l_id);
            id_bits.as_mut_bitslice().store_le(id);
            id_bits_per_base.push(id_bits);
        }
    }

    let symbol_width = num_deviation_bits + l_id;
    let chunk_size = (num_rows / (rayon::current_num_threads() * 4)).clamp(256, 4096);

    let chunk_results: Vec<BitVec<usize, Lsb0>> = (0..num_rows)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|row_chunk| {
            let mut chunk_stream = BitVec::with_capacity(row_chunk.len() * symbol_width);

            for row in row_chunk {
                let id = row_to_group_id[row];
                let chunk = unsafe { bit_data.get_chunk_unchecked(row) };

                for &(start, end) in &deviation_ranges {
                    chunk_stream.extend_from_bitslice(unsafe { chunk.get_unchecked(start..end) });
                }

                if l_id > 0 {
                    chunk_stream.extend_from_bitslice(id_bits_per_base[id].as_bitslice());
                }
            }
            chunk_stream
        })
        .collect();

    let mut encoded_bit_stream = BitVec::with_capacity(num_rows * symbol_width);
    for chunk_stream in chunk_results {
        encoded_bit_stream.extend_from_bitslice(chunk_stream.as_bitslice());
    }

    FusedEncodingResult {
        deviation_data: DeviationData::new(encoded_bit_stream, num_rows, num_deviation_bits, l_id),
        base_table,
    }
}
