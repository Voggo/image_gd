use bitvec::prelude::*;
use fxhash::FxHashMap;
use std::hash::{Hash, Hasher};

use super::context::build_deviation_ranges;

use crate::compression::base_bits::BaseBit;
use crate::compression::compress::DeviationData;
use crate::compression::preprocessor::BitDataSet;
use crate::utils::bits_needed_nonzero;

pub(super) struct FusedEncodingResult {
    pub(super) deviation_data: DeviationData,
    pub(super) base_table: Vec<(BitVec<usize, Msb0>, usize)>,
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
    chunk: &BitSlice<usize, Msb0>,
    base_bit_positions: &[usize],
) -> SignatureKey {
    if base_bit_positions.len() <= 128 {
        let mut packed = 0u128;
        for &bit_pos in base_bit_positions {
            packed = (packed << 1) | (chunk[bit_pos] as u128);
        }
        SignatureKey::PackedU128(packed)
    } else {
        let num_bytes = base_bit_positions.len().div_ceil(8);
        let mut bytes = vec![0u8; num_bytes];
        for (idx, &bit_pos) in base_bit_positions.iter().enumerate() {
            if chunk[bit_pos] {
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
    let base_bit_positions = base_bit_groups.get_base_bit_positions();

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
        let chunk = bit_data.get_chunk(row);
        let signature = build_signature_key(chunk, base_bit_positions);
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

    let symbol_width = num_deviation_bits + l_id;
    let mut encoded_bit_stream = BitVec::<usize, Msb0>::with_capacity(num_rows * symbol_width);
    for (row, id_ref) in row_to_group_id.iter().enumerate().take(num_rows) {
        let id = *id_ref;
        let chunk = bit_data.get_chunk(row);

        for &(start, end) in &deviation_ranges {
            encoded_bit_stream.extend_from_bitslice(&chunk[start..end]);
        }

        if l_id > 0 {
            encoded_bit_stream.extend_from_bitslice(id_bits_per_base[id].as_bitslice());
        }
    }

    let mut base_table = Vec::with_capacity(num_bases);
    for (id, &representative_row) in representative_rows.iter().enumerate() {
        let base: BitVec<usize, Msb0> = bit_data.get_chunk(representative_row).to_bitvec();
        base_table.push((base & base_bit_mask, base_counts[id]));
    }

    FusedEncodingResult {
        deviation_data: DeviationData::new(encoded_bit_stream, num_rows, num_deviation_bits, l_id),
        base_table,
    }
}
