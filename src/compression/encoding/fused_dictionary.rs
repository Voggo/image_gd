use bitvec::prelude::*;
use fxhash::FxHashMap;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};

use super::encoding_core::{
    BaseTable, CompressedData, DeviationData, EncodedData, build_deviation_ranges,
};

use crate::compression::base_bits::BaseBit;
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

enum SignatureKey {
    PackedU128(u128),
    Bytes(Vec<u8>),
}

impl PartialEq for SignatureKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SignatureKey::PackedU128(lhs), SignatureKey::PackedU128(rhs)) => lhs == rhs,
            (SignatureKey::Bytes(lhs), SignatureKey::Bytes(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}

impl Eq for SignatureKey {}

impl Hash for SignatureKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            SignatureKey::PackedU128(value) => {
                0u8.hash(state);
                value.hash(state);
            }
            SignatureKey::Bytes(bytes) => {
                1u8.hash(state);
                bytes.hash(state);
            }
        }
    }
}

fn build_signature_key(
    chunk: &BitSlice<usize, Lsb0>,
    base_bit_positions: &[usize],
) -> SignatureKey {
    if base_bit_positions.len() <= 128 {
        let mut packed = 0u128;
        for &bit_pos in base_bit_positions {
            packed = (packed << 1) | (unsafe { *chunk.get_unchecked(bit_pos) } as u128);
        }
        SignatureKey::PackedU128(packed)
    } else {
        let num_bytes = base_bit_positions.len().div_ceil(8);
        let mut bytes = vec![0u8; num_bytes];
        for (idx, &bit_pos) in base_bit_positions.iter().enumerate() {
            if unsafe { *chunk.get_unchecked(bit_pos) } {
                let byte_idx = idx / 8;
                let bit_in_byte = 7 - (idx % 8);
                bytes[byte_idx] |= 1u8 << bit_in_byte;
            }
        }
        SignatureKey::Bytes(bytes)
    }
}

pub(super) fn encode_data_fused_dictionary<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> FusedEncodingResult {
    let base_bit_mask = base_bit_groups.get_base_bit_mask();
    let selected_bit_positions = base_bit_groups.get_base_bit_positions();

    let chunk_size = bit_data.chunk_size();
    let num_bits_per_base = base_bit_groups.get_num_bits_per_base();
    let num_deviation_bits = chunk_size.saturating_sub(num_bits_per_base);
    let deviation_ranges = build_deviation_ranges(base_bit_mask, chunk_size, num_deviation_bits);

    let num_rows = bit_data.num_rows();
    let mut signature_to_id: FxHashMap<SignatureKey, usize> =
        FxHashMap::with_capacity_and_hasher(num_rows.min(1024), Default::default());
    let mut row_to_group_id = Vec::with_capacity(num_rows);
    let mut representative_rows: Vec<usize> = Vec::new();
    let mut base_counts: Vec<usize> = Vec::new();
    for row in 0..num_rows {
        let chunk = unsafe { bit_data.get_chunk_unchecked(row) };
        let signature = build_signature_key(chunk, selected_bit_positions);
        let id = if let Some(existing_id) = signature_to_id.get(&signature).copied() {
            base_counts[existing_id] += 1;
            existing_id
        } else {
            let new_id = representative_rows.len();
            signature_to_id.insert(signature, new_id);
            representative_rows.push(row);
            base_counts.push(1);
            new_id
        };
        row_to_group_id.push(id);
    }

    let num_bases = representative_rows.len();
    let l_id = bits_needed_nonzero(num_bases);
    let mut id_bits_per_base: Vec<BitVec<usize, Lsb0>> = Vec::new();
    if l_id > 0 {
        id_bits_per_base = Vec::with_capacity(num_bases);
        for id in 0..num_bases {
            let mut id_bits = BitVec::repeat(false, l_id);
            id_bits.as_mut_bitslice().store_be(id);
            id_bits_per_base.push(id_bits);
        }
    }

    let symbol_width = num_deviation_bits + l_id;
    // Process rows in parallel chunks to build symbol segments
    let chunk_size = (num_rows / (rayon::current_num_threads() * 4))
        .max(256)
        .min(4096);
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

    // Merge all chunks in order into the final stream
    let mut encoded_bit_stream = BitVec::with_capacity(num_rows * symbol_width);
    for chunk_stream in chunk_results {
        encoded_bit_stream.extend_from_bitslice(chunk_stream.as_bitslice());
    }
    let mut base_table = Vec::with_capacity(num_bases);
    for (id, &representative_row) in representative_rows.iter().enumerate() {
        let chunk = unsafe { bit_data.get_chunk_unchecked(representative_row) };
        let mut packed_base = BitVec::with_capacity(selected_bit_positions.len());
        for &bit_pos in selected_bit_positions {
            packed_base.push(unsafe { *chunk.get_unchecked(bit_pos) });
        }
        base_table.push((packed_base, base_counts[id]));
    }

    FusedEncodingResult {
        deviation_data: DeviationData::new(encoded_bit_stream, num_rows, num_deviation_bits, l_id),
        base_table,
    }
}
