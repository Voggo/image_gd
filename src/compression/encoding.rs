use crate::compression::base_bits::BaseBit;
use crate::compression::compress::{
    CompressedData, DeviationData, EncodedData, HuffmanDeviationData, RleDeviationData,
    build_huffman_code_map, huffman_row_layout,
};
use crate::compression::tabular_preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use crate::utils::bits_needed_nonzero;
use bitvec::prelude::*;
use fxhash::FxHashMap;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};

const RLE_MAX_CONTROL_VALUE: usize = 134;
const RLE_MAX_RUN_LEN: usize = RLE_MAX_CONTROL_VALUE + 1;

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

pub struct EncodeDataFusedDictionary {}

impl Filter for EncodeDataFusedDictionary {
    type Input = (BitDataSet, Box<dyn BaseBit>);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(
            "Encoding data into compressed format (fused dictionary + id/deviation)",
        );
        let (bit_data, base_bit_groups) = input;
        let fused = encode_data_fused_dictionary(&bit_data, base_bit_groups.as_ref());

        let mut compressed = CompressedData::new(
            EncodedData::Normal(fused.deviation_data),
            bit_data.info.clone(),
        );
        compressed.base_table = fused.base_table;
        compressed.base_bit_positions = base_bit_groups.get_base_bit_positions().to_vec();
        compressed.condensed_sample_weights = bit_data
            .info
            .m_condensed_sample_weights()
            .map(|weights| weights.to_vec());
        Ok(compressed)
    }
}

pub struct EncodeDataHuffman {}

impl Filter for EncodeDataHuffman {
    type Input = (BitDataSet, Box<dyn BaseBit>);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format (Huffman)");
        let (bit_data, base_bit_groups) = input;
        let mut compressed = CompressedData::new(
            EncodedData::Huffman(encode_data_huffman(&bit_data, base_bit_groups.as_ref())?),
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

pub struct EncodeDataHuffmanBaseIdOnly {}

impl Filter for EncodeDataHuffmanBaseIdOnly {
    type Input = (BitDataSet, Box<dyn BaseBit>);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer =
            ScopedTimer::info("Encoding data into compressed format (Huffman base-id only)");
        let (bit_data, base_bit_groups) = input;
        let mut compressed = CompressedData::new(
            EncodedData::Huffman(encode_data_huffman_base_id_only(
                &bit_data,
                base_bit_groups.as_ref(),
            )?),
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

struct FusedEncodingResult {
    deviation_data: DeviationData,
    base_table: Vec<(BitVec<usize, Msb0>, usize)>,
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

fn build_deviation_ranges(
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

fn encode_data_fused_dictionary<B: BaseBit + ?Sized>(
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

fn prepare_encoding_context<B: BaseBit + ?Sized>(
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

fn encode_data<B: BaseBit + ?Sized>(bit_data: &BitDataSet, base_bit_groups: &B) -> DeviationData {
    let mut encoded_bit_stream = BitVec::<usize, Msb0>::new();
    let base_bit_mask = base_bit_groups.get_base_bit_mask();

    let num_bases = base_bit_groups.get_num_bases();
    let l_id = bits_needed_nonzero(num_bases);

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
    let raw_symbol_stream = encode_rows_as_symbol_stream(bit_data, &context);
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
        let current_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i);

        let mut run_len = 1usize;
        while i + run_len < num_rows && run_len < RLE_MAX_RUN_LEN {
            let next_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i + run_len);
            if next_symbol == current_symbol {
                run_len += 1;
            } else {
                break;
            }
        }

        let r_encoded: u8;
        if run_len >= 2 {
            r_encoded = (run_len - 1) as u8;
            symbol_stream.extend_from_bitslice(current_symbol);
            i += run_len;
        } else {
            r_encoded = 0;
        }

        let literal_start = i;
        let mut literal_count = 0usize;
        while i < num_rows && literal_count < RLE_MAX_CONTROL_VALUE {
            let this_symbol = symbol_slice(&raw_symbol_stream, symbol_width, i);

            let mut lookahead_run = 1usize;
            while i + lookahead_run < num_rows && lookahead_run < RLE_MAX_RUN_LEN {
                let lookahead_symbol =
                    symbol_slice(&raw_symbol_stream, symbol_width, i + lookahead_run);
                if lookahead_symbol == this_symbol {
                    lookahead_run += 1;
                } else {
                    break;
                }
            }

            if lookahead_run >= 2 {
                break;
            }

            symbol_stream.extend_from_bitslice(this_symbol);
            literal_count += 1;
            i += 1;
        }

        if r_encoded == 0 && literal_count == 0 {
            symbol_stream.extend_from_bitslice(current_symbol);
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

fn encode_data_huffman<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> Result<HuffmanDeviationData, EntroGdError> {
    let context = prepare_encoding_context(bit_data, base_bit_groups);
    let frequencies = build_symbol_frequencies(bit_data, &context)?;
    let (canonical_symbols, canonical_code_lengths) = build_canonical_huffman_table(&frequencies)?;
    let codes_by_symbol = build_huffman_code_map(&canonical_symbols, &canonical_code_lengths)?;

    let (original_num_samples, row_count, row_width) = huffman_row_layout(&bit_data.info)?;

    let mut pixel_bit_stream = BitVec::<usize, Msb0>::new();
    let mut row_offsets = Vec::with_capacity(row_count);

    for row_idx in 0..row_count {
        row_offsets.push(u32::try_from(pixel_bit_stream.len()).map_err(|_| {
            EntroGdError::InvalidMetadata {
                message: "Huffman row offset does not fit into u32".to_string(),
            }
        })?);

        let row_start = row_idx * row_width;
        for sample_idx in row_start..row_start + row_width {
            let symbol = symbol_for_row(bit_data, &context, sample_idx)?;
            let (code, code_len) = codes_by_symbol.get(&symbol).copied().ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: format!("missing Huffman code for symbol {}", symbol),
                }
            })?;
            append_code_bits(&mut pixel_bit_stream, code, code_len);
        }
    }

    for sample_idx in original_num_samples..bit_data.num_rows() {
        let symbol = symbol_for_row(bit_data, &context, sample_idx)?;
        let (code, code_len) =
            codes_by_symbol
                .get(&symbol)
                .copied()
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: format!("missing Huffman code for symbol {}", symbol),
                })?;
        append_code_bits(&mut pixel_bit_stream, code, code_len);
    }

    HuffmanDeviationData::new(
        pixel_bit_stream,
        canonical_symbols,
        canonical_code_lengths,
        row_offsets,
        bit_data.num_rows(),
        original_num_samples,
        context.num_deviation_bits,
        context.l_id,
        row_width,
    )
}

fn encode_data_huffman_base_id_only<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> Result<HuffmanDeviationData, EntroGdError> {
    // This variant intentionally Huffman-encodes only base IDs.
    // Deviation bits are not encoded in the Huffman payload and are represented as zero-length
    // (`num_deviation_bits = 0`) in the resulting stream metadata.
    let context = prepare_encoding_context(bit_data, base_bit_groups);
    let frequencies = build_base_id_frequencies(&context);
    let (canonical_symbols, canonical_code_lengths) = build_canonical_huffman_table(&frequencies)?;
    let codes_by_symbol = build_huffman_code_map(&canonical_symbols, &canonical_code_lengths)?;

    let (original_num_samples, row_count, row_width) = huffman_row_layout(&bit_data.info)?;

    let mut pixel_bit_stream = BitVec::<usize, Msb0>::new();
    let mut raw_deviation_bit_stream =
        BitVec::<usize, Msb0>::with_capacity(bit_data.num_rows() * context.num_deviation_bits);
    let mut row_offsets = Vec::with_capacity(row_count);

    for row_idx in 0..row_count {
        row_offsets.push(u32::try_from(pixel_bit_stream.len()).map_err(|_| {
            EntroGdError::InvalidMetadata {
                message: "Huffman row offset does not fit into u32".to_string(),
            }
        })?);

        let row_start = row_idx * row_width;
        for sample_idx in row_start..row_start + row_width {
            let chunk = bit_data.get_chunk(sample_idx);
            for &(start, end) in &context.deviation_ranges {
                raw_deviation_bit_stream.extend_from_bitslice(&chunk[start..end]);
            }

            let symbol = context.row_to_group_id[sample_idx] as u64;
            let (code, code_len) = codes_by_symbol.get(&symbol).copied().ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: format!("missing Huffman code for base-id symbol {}", symbol),
                }
            })?;
            append_code_bits(&mut pixel_bit_stream, code, code_len);
        }
    }

    for sample_idx in original_num_samples..bit_data.num_rows() {
        let chunk = bit_data.get_chunk(sample_idx);
        for &(start, end) in &context.deviation_ranges {
            raw_deviation_bit_stream.extend_from_bitslice(&chunk[start..end]);
        }

        let symbol = context.row_to_group_id[sample_idx] as u64;
        let (code, code_len) =
            codes_by_symbol
                .get(&symbol)
                .copied()
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: format!("missing Huffman code for base-id symbol {}", symbol),
                })?;
        append_code_bits(&mut pixel_bit_stream, code, code_len);
    }

    HuffmanDeviationData::new_base_id_only(
        pixel_bit_stream,
        raw_deviation_bit_stream,
        canonical_symbols,
        canonical_code_lengths,
        row_offsets,
        bit_data.num_rows(),
        original_num_samples,
        context.num_deviation_bits,
        context.l_id,
        row_width,
    )
}

fn build_symbol_frequencies(
    bit_data: &BitDataSet,
    context: &EncodingContext,
) -> Result<FxHashMap<u64, usize>, EntroGdError> {
    let mut frequencies = FxHashMap::default();

    for row in 0..bit_data.num_rows() {
        let symbol = symbol_for_row(bit_data, context, row)?;
        *frequencies.entry(symbol).or_insert(0) += 1;
    }
    // Print sorted frequencies for debugging
    let mut freq_vec: Vec<(u64, usize)> = frequencies
        .iter()
        .map(|(&symbol, &freq)| (symbol, freq))
        .collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
    tracing::debug!(frequencies = ?freq_vec, "Built symbol frequencies for Huffman encoding");

    Ok(frequencies)
}

fn build_base_id_frequencies(context: &EncodingContext) -> FxHashMap<u64, usize> {
    let mut frequencies = FxHashMap::default();
    for &base_id in &context.row_to_group_id {
        *frequencies.entry(base_id as u64).or_insert(0) += 1;
    }

    let mut freq_vec: Vec<(u64, usize)> = frequencies
        .iter()
        .map(|(&symbol, &freq)| (symbol, freq))
        .collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
    tracing::debug!(frequencies = ?freq_vec, "Built base-id symbol frequencies for Huffman encoding");

    frequencies
}

fn symbol_for_row(
    bit_data: &BitDataSet,
    context: &EncodingContext,
    row: usize,
) -> Result<u64, EntroGdError> {
    let symbol_width = context.num_deviation_bits + context.l_id;
    if symbol_width > 64 {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "Huffman symbol width {} exceeds supported 64-bit range",
                symbol_width
            ),
        });
    }

    let chunk = bit_data.get_chunk(row);
    let mut deviation = 0u64;
    for &(start, end) in &context.deviation_ranges {
        for bit in &chunk[start..end] {
            deviation = (deviation << 1) | (*bit as u64);
        }
    }

    let id = context.row_to_group_id[row] as u64;
    Ok((id << context.num_deviation_bits) | deviation)
}

fn build_canonical_huffman_table(
    frequencies: &FxHashMap<u64, usize>,
) -> Result<(Vec<u64>, Vec<u8>), EntroGdError> {
    if frequencies.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if frequencies.len() > u32::MAX as usize {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "Huffman symbol table has {} entries, exceeding u32::MAX",
                frequencies.len()
            ),
        });
    }

    let mut symbols: Vec<SymbolFrequency> = frequencies
        .iter()
        .map(|(&symbol, &frequency)| SymbolFrequency { symbol, frequency })
        .collect();
    // using unstable as it does not matter how we break ties, and it is faster than stable
    symbols.sort_unstable_by(|a, b| a.symbol.cmp(&b.symbol));

    if symbols.len() == 1 {
        return Ok((vec![symbols[0].symbol], vec![1]));
    }

    let raw_lengths = build_huffman_code_lengths(&symbols)?;

    let mut canonical_entries: Vec<(u64, u8)> = symbols
        .iter()
        .zip(raw_lengths.iter())
        .map(|(symbol, &len)| (symbol.symbol, len as u8))
        .collect();
    canonical_entries.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    let (canonical_symbols, canonical_lengths): (Vec<u64>, Vec<u8>) =
        canonical_entries.into_iter().unzip();
    Ok((canonical_symbols, canonical_lengths))
}

fn build_huffman_code_lengths(symbols: &[SymbolFrequency]) -> Result<Vec<usize>, EntroGdError> {
    let mut nodes: Vec<HuffmanNode> = symbols
        .iter()
        .map(|symbol| HuffmanNode {
            weight: symbol.frequency,
            min_symbol: symbol.symbol,
            parent: None,
        })
        .collect();
    // The heap is a MasHeap by default therefore we wrap all elements is a Reverse that flips all comparisons
    let mut heap = BinaryHeap::<Reverse<(usize, u64, usize)>>::new();

    for (idx, node) in nodes.iter().enumerate() {
        heap.push(Reverse((node.weight, node.min_symbol, idx)));
    }

    while heap.len() > 1 {
        let Reverse((left_weight, left_min_symbol, left_idx)) = heap.pop().unwrap();
        let Reverse((right_weight, right_min_symbol, right_idx)) = heap.pop().unwrap();
        let parent_idx = nodes.len();
        nodes[left_idx].parent = Some(parent_idx);
        nodes[right_idx].parent = Some(parent_idx);
        nodes.push(HuffmanNode {
            weight: left_weight.checked_add(right_weight).ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: "Huffman frequency sum overflow".to_string(),
                }
            })?,
            min_symbol: left_min_symbol.min(right_min_symbol),
            parent: None,
        });
        heap.push(Reverse((
            left_weight
                .checked_add(right_weight)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "Huffman frequency sum overflow".to_string(),
                })?,
            left_min_symbol.min(right_min_symbol),
            parent_idx,
        )));
    }

    let mut lengths = Vec::with_capacity(symbols.len());
    for leaf_idx in 0..symbols.len() {
        let mut depth = 0usize;
        let mut current = leaf_idx;
        while let Some(parent) = nodes[current].parent {
            depth += 1;
            current = parent;
        }
        lengths.push(depth.max(1));
    }

    Ok(lengths)
}

fn append_code_bits(out: &mut BitVec<usize, Msb0>, code: u32, code_len: u8) {
    for shift in (0..code_len as usize).rev() {
        out.push(((code >> shift) & 1) == 1);
    }
}

#[derive(Debug, Clone, Copy)]
struct SymbolFrequency {
    symbol: u64,
    frequency: usize,
}

#[derive(Debug, Clone, Copy)]
struct HuffmanNode {
    weight: usize,
    min_symbol: u64,
    parent: Option<usize>,
}

fn symbol_slice(
    symbol_stream: &BitVec<usize, Msb0>,
    symbol_width: usize,
    row: usize,
) -> &BitSlice<usize, Msb0> {
    let start = row * symbol_width;
    &symbol_stream[start..start + symbol_width]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::base_bits::BaseBitGroups;
    use crate::compression::preprocessor::{
        BitData, BitDataInfo, FeatureDataType, FeatureSpec, FeatureTransform,
    };

    /// Helper function to create a simple BitDataSet for testing
    /// Creates a dataset with `num_rows` rows and `chunk_size` bits per row
    fn create_test_bit_data_set(num_rows: usize, chunk_size: usize) -> BitDataSet {
        let mut data = BitVec::<usize, Msb0>::with_capacity(num_rows * chunk_size);
        for _ in 0..num_rows * chunk_size {
            data.push(false);
        }

        let bit_data = BitData {
            data,
            chunk_size,
            num_rows,
        };

        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UnsignedInt,
            bits: chunk_size,
            transform: FeatureTransform::None,
        }];

        let info = BitDataInfo::new(features, num_rows * chunk_size)
            .expect("Failed to create BitDataInfo");

        BitDataSet {
            data: bit_data,
            info,
        }
    }

    /// Helper function to create a BitDataSet with specific bit patterns
    fn create_test_bit_data_set_with_pattern(rows_and_bits: Vec<Vec<bool>>) -> BitDataSet {
        assert!(!rows_and_bits.is_empty(), "Must have at least one row");
        let chunk_size = rows_and_bits[0].len();
        let num_rows = rows_and_bits.len();

        let mut data = BitVec::<usize, Msb0>::with_capacity(num_rows * chunk_size);
        for row in &rows_and_bits {
            assert_eq!(row.len(), chunk_size, "All rows must have the same size");
            for &bit in row {
                data.push(bit);
            }
        }

        let bit_data = BitData {
            data,
            chunk_size,
            num_rows,
        };

        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UnsignedInt,
            bits: chunk_size,
            transform: FeatureTransform::None,
        }];

        let info = BitDataInfo::new(features, num_rows * chunk_size)
            .expect("Failed to create BitDataInfo");

        BitDataSet {
            data: bit_data,
            info,
        }
    }

    /// Helper function to create a simple BaseBitGroups instance for testing
    /// This creates a groups structure where all rows are initially in one group
    fn create_test_base_bit_groups(num_rows: usize, chunk_size: usize) -> BaseBitGroups {
        BaseBitGroups::new(num_rows, chunk_size)
    }

    /// Helper function to add bit positions to a BaseBitGroups instance
    fn add_base_bits(groups: &mut BaseBitGroups, bit_data: &BitDataSet, bit_positions: &[usize]) {
        for &pos in bit_positions {
            groups.add_bit_position(bit_data, pos);
        }
    }

    #[test]
    fn test_encode_data_basic() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        // Add some base bits at positions 0 and 1
        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let result = encode_data(&bit_data, &base_groups);

        // Should have encoded 6 deviation bits per row (8 - 2 = 6)
        assert_eq!(result.get_num_deviation_bits(), 6);
        assert_eq!(result.get_num_samples(), 4);
    }

    #[test]
    fn test_encode_data_optimized_basic() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let result = encode_data_optimized(&bit_data, &base_groups);

        assert_eq!(result.get_num_deviation_bits(), 6);
        assert_eq!(result.get_num_samples(), 4);
    }

    #[test]
    fn test_encode_data_rle_basic() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let result = encode_data_rle(&bit_data, &base_groups);

        assert_eq!(result.get_num_deviation_bits(), 6);
        assert_eq!(result.get_num_samples(), 4);
        // RLE should produce rm_values entries
        assert!(!result.rm_values().is_empty());
    }

    #[test]
    fn test_encode_data_huffman_basic() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let result = encode_data_huffman(&bit_data, &base_groups).unwrap();

        assert_eq!(result.get_num_deviation_bits(), 6);
        assert_eq!(result.get_num_samples(), 4);
        assert_eq!(result.row_offsets().len(), 4);
    }

    #[test]
    fn test_encode_data_huffman_matches_raw_symbols() {
        let bit_data = create_test_bit_data_set_with_pattern(vec![
            vec![false, false, false, true, false, true, false, true],
            vec![false, false, true, true, false, false, false, true],
            vec![false, false, false, false, true, true, false, false],
            vec![false, false, true, false, true, false, true, false],
        ]);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let raw = encode_data_optimized(&bit_data, &base_groups);
        let huffman = encode_data_huffman(&bit_data, &base_groups).unwrap();

        assert_eq!(
            huffman.to_deviation_data().unwrap().encoded_bit_stream(),
            raw.encoded_bit_stream()
        );
    }

    #[test]
    fn test_encode_data_huffman_base_id_only_basic() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let result = encode_data_huffman_base_id_only(&bit_data, &base_groups).unwrap();

        assert_eq!(result.get_num_deviation_bits(), 6);
        assert_eq!(result.get_num_id_bits(), 1);
        assert_eq!(result.get_num_samples(), 4);
        assert_eq!(result.row_offsets().len(), 4);

        let raw = encode_data_optimized(&bit_data, &base_groups);
        assert_eq!(
            result.to_deviation_data().unwrap().encoded_bit_stream(),
            raw.encoded_bit_stream()
        );
    }

    #[test]
    fn test_encode_data_empty_dataset() {
        let bit_data = create_test_bit_data_set(0, 8);
        let base_groups = create_test_base_bit_groups(0, 8);

        let result = encode_data(&bit_data, &base_groups);

        assert_eq!(result.get_num_samples(), 0);
        assert_eq!(result.encoded_bit_stream().len(), 0);
    }

    #[test]
    fn test_encode_data_no_base_bits() {
        let bit_data = create_test_bit_data_set(2, 8);
        let base_groups = create_test_base_bit_groups(2, 8);

        let result = encode_data(&bit_data, &base_groups);

        // With no base bits, all 8 bits should be encoded as deviation
        assert_eq!(result.get_num_deviation_bits(), 8);
        assert_eq!(result.get_num_id_bits(), 1); // log2(1) = 0, but clamped to 1
    }

    #[test]
    fn test_encode_data_all_base_bits() {
        let bit_data = create_test_bit_data_set(2, 8);
        let mut base_groups = create_test_base_bit_groups(2, 8);

        // Add all bit positions as base bits
        add_base_bits(&mut base_groups, &bit_data, &[0, 1, 2, 3, 4, 5, 6, 7]);

        let result = encode_data(&bit_data, &base_groups);

        // With all bits as base bits, no deviation bits
        assert_eq!(result.get_num_deviation_bits(), 0);
        // With 2 rows, all zero pattern creates 1 group, so l_id = 1
        assert_eq!(result.get_num_id_bits(), 1);
    }

    #[test]
    fn test_prepare_encoding_context_creates_valid_id_bits() {
        let bit_data = create_test_bit_data_set(8, 8);
        let mut base_groups = create_test_base_bit_groups(8, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0]);

        let context = prepare_encoding_context(&bit_data, &base_groups);

        // With 8 identical zero rows, adding 1 base bit creates at most 2 groups (zeros and ones)
        // Since all rows are zeros, we get 1 group, so l_id = 1
        assert_eq!(context.l_id, 1);
        assert_eq!(context.id_bits_per_base.len(), 1);
    }

    #[test]
    fn test_encode_filter_process() {
        let bit_data = create_test_bit_data_set(2, 8);
        let mut base_groups = create_test_base_bit_groups(2, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let filter = EncodeData {};
        let result = filter.process((bit_data, Box::new(base_groups)));

        assert!(result.is_ok());
        let compressed = result.unwrap();
        assert_eq!(compressed.metadata.chunk_size(), 8);
    }

    #[test]
    fn test_encode_data_optimized_filter_process() {
        let bit_data = create_test_bit_data_set(2, 8);
        let mut base_groups = create_test_base_bit_groups(2, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let filter = EncodeDataOptimized {};
        let result = filter.process((bit_data, Box::new(base_groups)));

        assert!(result.is_ok());
        let compressed = result.unwrap();
        assert_eq!(compressed.metadata.chunk_size(), 8);
    }

    #[test]
    fn test_encode_data_rle_filter_process() {
        let bit_data = create_test_bit_data_set(2, 8);
        let mut base_groups = create_test_base_bit_groups(2, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let filter = EncodeDataRLE {};
        let result = filter.process((bit_data, Box::new(base_groups)));

        assert!(result.is_ok());
        let compressed = result.unwrap();
        assert_eq!(compressed.metadata.chunk_size(), 8);
    }

    #[test]
    fn test_encode_data_fused_dictionary_matches_standard_on_full_groups() {
        let patterns = vec![
            vec![false, false, false, true, false, true, false, true],
            vec![false, false, true, true, false, false, false, true],
            vec![true, true, false, false, true, true, false, false],
            vec![true, true, true, false, true, false, true, false],
        ];
        let bit_data = create_test_bit_data_set_with_pattern(patterns);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let fused = encode_data_fused_dictionary(&bit_data, &base_groups);
        let optimized = encode_data_optimized(&bit_data, &base_groups);

        assert_eq!(
            fused.deviation_data.encoded_bit_stream(),
            optimized.encoded_bit_stream()
        );
        assert_eq!(fused.base_table, base_groups.get_bases(&bit_data));
    }

    #[test]
    fn test_encode_data_fused_dictionary_supports_partial_training_groups() {
        let full_data = create_test_bit_data_set_with_pattern(vec![
            vec![false, false, false, false],
            vec![false, false, true, false],
            vec![true, true, false, true],
            vec![true, true, true, true],
        ]);
        let training_subset = create_test_bit_data_set_with_pattern(vec![
            vec![false, false, false, false],
            vec![false, false, true, false],
        ]);

        let mut trained_groups = create_test_base_bit_groups(2, 4);
        add_base_bits(&mut trained_groups, &training_subset, &[0, 1]);

        let fused = encode_data_fused_dictionary(&full_data, &trained_groups);
        let mut counts: Vec<usize> = fused
            .base_table
            .iter()
            .map(|(_base, count)| *count)
            .collect();
        counts.sort_unstable();

        assert_eq!(fused.base_table.len(), 2);
        assert_eq!(counts, vec![2, 2]);
        assert_eq!(fused.deviation_data.get_num_samples(), 4);
        assert_eq!(fused.deviation_data.get_num_deviation_bits(), 2);
        assert_eq!(fused.deviation_data.get_num_id_bits(), 1);
    }

    #[test]
    fn test_encode_data_fused_dictionary_filter_process() {
        let bit_data = create_test_bit_data_set(2, 8);
        let mut base_groups = create_test_base_bit_groups(2, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 1]);

        let filter = EncodeDataFusedDictionary {};
        let result = filter.process((bit_data, Box::new(base_groups)));

        assert!(result.is_ok());
        let compressed = result.unwrap();
        assert_eq!(compressed.metadata.chunk_size(), 8);
    }

    #[test]
    fn test_encode_data_with_specific_patterns() {
        // Create a dataset with specific bit patterns: all 0s followed by all 1s
        let patterns = vec![
            vec![false, false, false, false, false, false, false, false],
            vec![false, false, false, false, false, false, false, false],
            vec![true, true, true, true, true, true, true, true],
            vec![true, true, true, true, true, true, true, true],
        ];

        let bit_data = create_test_bit_data_set_with_pattern(patterns);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 4]);

        let result = encode_data(&bit_data, &base_groups);

        assert_eq!(result.get_num_samples(), 4);
        assert_eq!(result.get_num_deviation_bits(), 6);
    }

    #[test]
    fn test_encoding_context_row_to_group_mapping() {
        let bit_data = create_test_bit_data_set(4, 8);
        let mut base_groups = create_test_base_bit_groups(4, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0]);

        let context = prepare_encoding_context(&bit_data, &base_groups);

        // Initially all rows should be in same group, but after adding a base bit
        // the rows might be split into groups
        assert_eq!(context.row_to_group_id.len(), 4);
    }

    #[test]
    fn test_symbol_stream_encoding_consistency() {
        let bit_data = create_test_bit_data_set(3, 8);
        let mut base_groups = create_test_base_bit_groups(3, 8);

        add_base_bits(&mut base_groups, &bit_data, &[1, 3]);

        let result_normal = encode_data(&bit_data, &base_groups);
        let result_optimized = encode_data_optimized(&bit_data, &base_groups);

        // Both encoding methods should produce the same size output
        assert_eq!(
            result_normal.encoded_bit_stream().len(),
            result_optimized.encoded_bit_stream().len()
        );
    }

    #[test]
    fn test_single_row_encoding() {
        let bit_data = create_test_bit_data_set(1, 16);
        let mut base_groups = create_test_base_bit_groups(1, 16);

        add_base_bits(&mut base_groups, &bit_data, &[0, 5, 10, 15]);

        let result = encode_data(&bit_data, &base_groups);

        assert_eq!(result.get_num_samples(), 1);
        assert_eq!(result.get_num_deviation_bits(), 12);
        assert_eq!(result.get_num_id_bits(), 1); // log2(1) = 0, but clamped to 1
    }

    #[test]
    fn test_large_dataset_encoding() {
        let bit_data = create_test_bit_data_set(256, 32);
        let mut base_groups = create_test_base_bit_groups(256, 32);

        add_base_bits(&mut base_groups, &bit_data, &[0, 8, 16, 24]);

        let result = encode_data(&bit_data, &base_groups);

        assert_eq!(result.get_num_samples(), 256);
        assert_eq!(result.get_num_deviation_bits(), 28);
        // Since all rows have same bit pattern (all zeros), adding bits won't split them
        // So we get 1 group -> l_id = 1
        assert_eq!(result.get_num_id_bits(), 1);
    }

    #[test]
    fn test_rle_compression_with_repeated_rows() {
        // Create a dataset where rows alternate between two patterns
        let patterns = vec![
            vec![true, false, true, false, true, false, true, false],
            vec![true, false, true, false, true, false, true, false],
            vec![false, true, false, true, false, true, false, true],
            vec![true, false, true, false, true, false, true, false],
            vec![true, false, true, false, true, false, true, false],
        ];

        let bit_data = create_test_bit_data_set_with_pattern(patterns);
        let mut base_groups = create_test_base_bit_groups(5, 8);

        add_base_bits(&mut base_groups, &bit_data, &[0, 2]);

        let result = encode_data_rle(&bit_data, &base_groups);

        assert_eq!(result.get_num_samples(), 5);
        // RLE should have some run-length encoding
        assert!(!result.rm_values().is_empty());
    }
}
