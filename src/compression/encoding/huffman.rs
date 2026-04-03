use fxhash::FxHashMap;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::context::EncodingContext;

use crate::compression::compress::{
    HuffmanDeviationData, build_huffman_code_map, huffman_row_layout,
};
use crate::compression::preprocessor::BitDataSet;
use crate::error::EntroGdError;

pub(super) fn encode_data_huffman(
    bit_data: &BitDataSet,
    context: &EncodingContext,
) -> Result<HuffmanDeviationData, EntroGdError> {
    let frequencies = build_symbol_frequencies(bit_data, context)?;
    let (canonical_symbols, canonical_code_lengths) = build_canonical_huffman_table(&frequencies)?;
    let codes_by_symbol = build_huffman_code_map(&canonical_symbols, &canonical_code_lengths)?;

    let (original_num_samples, row_count, row_width) = huffman_row_layout(&bit_data.info)?;

    let mut pixel_bit_stream = crate::BitStream::new();
    let mut row_offsets = Vec::with_capacity(row_count);

    for row_idx in 0..row_count {
        row_offsets.push(u32::try_from(pixel_bit_stream.len()).map_err(|_| {
            EntroGdError::InvalidMetadata {
                message: "Huffman row offset does not fit into u32".to_string(),
            }
        })?);

        let row_start = row_idx * row_width;
        for sample_idx in row_start..row_start + row_width {
            let symbol = symbol_for_row(bit_data, context, sample_idx)?;
            let (code, code_len) = codes_by_symbol.get(&symbol).copied().ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: format!("missing Huffman code for symbol {}", symbol),
                }
            })?;
            append_code_bits(&mut pixel_bit_stream, code, code_len);
        }
    }

    for sample_idx in original_num_samples..bit_data.num_rows() {
        let symbol = symbol_for_row(bit_data, context, sample_idx)?;
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

pub(super) fn encode_data_huffman_base_id_only(
    bit_data: &BitDataSet,
    context: &EncodingContext,
) -> Result<HuffmanDeviationData, EntroGdError> {
    let frequencies = build_base_id_frequencies(context);
    let (canonical_symbols, canonical_code_lengths) = build_canonical_huffman_table(&frequencies)?;
    let codes_by_symbol = build_huffman_code_map(&canonical_symbols, &canonical_code_lengths)?;

    let (original_num_samples, row_count, row_width) = huffman_row_layout(&bit_data.info)?;

    let mut pixel_bit_stream = crate::BitStream::new();
    let mut raw_deviation_bit_stream =
        crate::BitStream::with_capacity(bit_data.num_rows() * context.num_deviation_bits);
    let mut row_offsets = Vec::with_capacity(row_count);

    for row_idx in 0..row_count {
        row_offsets.push(u32::try_from(pixel_bit_stream.len()).map_err(|_| {
            EntroGdError::InvalidMetadata {
                message: "Huffman row offset does not fit into u32".to_string(),
            }
        })?);

        let row_start = row_idx * row_width;
        for sample_idx in row_start..row_start + row_width {
            let chunk = unsafe { bit_data.get_chunk_unchecked(sample_idx) };
            for &(start, end) in &context.deviation_ranges {
                raw_deviation_bit_stream
                    .extend_from_bitslice(unsafe { chunk.get_unchecked(start..end) });
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
        let chunk = unsafe { bit_data.get_chunk_unchecked(sample_idx) };
        for &(start, end) in &context.deviation_ranges {
            raw_deviation_bit_stream
                .extend_from_bitslice(unsafe { chunk.get_unchecked(start..end) });
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

    let chunk = unsafe { bit_data.get_chunk_unchecked(row) };
    let mut deviation = 0u64;
    for &(start, end) in &context.deviation_ranges {
        for bit in unsafe { chunk.get_unchecked(start..end) } {
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

fn append_code_bits(out: &mut crate::BitStream, code: u32, code_len: u8) {
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
