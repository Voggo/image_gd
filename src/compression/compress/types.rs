use bitvec::prelude::*;
use fxhash::FxHashMap;

use crate::compression::preprocessor::BitDataInfo;

/// Represents the compressed output.
#[derive(Debug, Clone)]
pub struct CompressedData {
    /// The encoded data stream.
    pub encoded_data: EncodedData,
    /// The weights for the condensed samples (if used).
    // Should be stored as a bitstream of length m * l_w (log_2(n).ceil() bits per weight).
    pub condensed_sample_weights: Option<Vec<usize>>,
    /// Base table mapping base patterns to their frequencies or encodings.
    pub base_table: Vec<(BitVec<usize, Msb0>, usize)>,
    /// Bit positions used as base bits during compression.
    pub base_bit_positions: Vec<usize>,
    /// Metadata for decompression (column count, base bits used, etc.).
    pub metadata: BitDataInfo,
}

#[derive(Debug, Clone)]
pub struct CondensedSamples {
    pub samples: Vec<BitVec<usize, Msb0>>,
    pub weights: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct DeviationSample {
    pub deviation: BitVec<usize, Msb0>,
    pub id: BitVec<usize, Msb0>,
}

pub(crate) struct DeviationSampleRef<'a> {
    pub deviation: &'a BitSlice<usize, Msb0>,
    pub id: &'a BitSlice<usize, Msb0>,
}

#[derive(Debug, Clone)]
pub struct DeviationData {
    pub(super) encoded_bit_stream: BitVec<usize, Msb0>,
    pub(super) num_samples: usize,
    pub(super) num_deviation_bits: usize,
    pub(super) num_id_bits: usize,
}

#[derive(Debug, Clone)]
pub struct RleDeviationData {
    pub(super) symbol_bit_stream: BitVec<usize, Msb0>,
    pub(super) rm_values: Vec<(u8, u8)>,
    pub(super) rm_control_stream: BitVec<usize, Msb0>,
    pub(super) num_samples: usize,
    pub(super) num_deviation_bits: usize,
    pub(super) num_id_bits: usize,
}

#[derive(Debug, Clone)]
pub struct HuffmanDeviationData {
    pub(super) pixel_bit_stream: BitVec<usize, Msb0>,
    pub(super) raw_deviation_bit_stream: Option<BitVec<usize, Msb0>>,
    pub(super) canonical_symbols: Vec<u64>,
    pub(super) canonical_code_lengths: Vec<u8>,
    pub(super) row_offsets: Vec<u32>,
    pub(super) num_samples: usize,
    pub(super) original_num_samples: usize,
    pub(super) num_deviation_bits: usize,
    pub(super) huffman_symbol_num_deviation_bits: usize,
    pub(super) num_id_bits: usize,
    pub(super) row_width: usize,
    pub(super) max_code_length: u8,
    pub(super) decode_by_length: Vec<FxHashMap<u32, u64>>,
    pub(super) original_stream_end_offset_bits: usize,
}

#[derive(Debug, Clone)]
pub enum EncodedData {
    Normal(DeviationData),
    Rle(RleDeviationData),
    Huffman(HuffmanDeviationData),
}
