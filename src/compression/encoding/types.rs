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
    pub base_table: BaseTable,
    /// Selected bit positions used as base bits during compression.
    pub base_bit_positions: Vec<usize>,
    /// Selected base-bit positions that vary across bases.
    pub variable_base_bit_positions: Vec<usize>,
    /// Selected base-bit positions that are constant and equal to zero.
    pub constant_zero_bit_positions: Vec<usize>,
    /// Selected base-bit positions that are constant and equal to one.
    pub constant_one_bit_positions: Vec<usize>,
    /// Metadata for decompression (column count, base bits used, etc.).
    pub metadata: BitDataInfo,
}

#[derive(Debug, Clone)]
pub struct CondensedSamples {
    pub samples: Vec<crate::BitStream>,
    pub weights: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct DeviationSample {
    pub deviation: crate::BitStream,
    pub id: crate::BitStream,
}

pub(crate) struct DeviationSampleRef<'a> {
    pub deviation: &'a crate::BitView,
    pub id: &'a crate::BitView,
}

#[derive(Debug, Clone)]
pub struct DeviationData {
    pub(super) encoded_bit_stream: crate::BitStream,
    pub(super) num_samples: usize,
    pub(super) num_deviation_bits: usize,
    pub(super) num_id_bits: usize,
}

#[derive(Debug, Clone)]
pub struct RleDeviationData {
    pub(super) symbol_bit_stream: crate::BitStream,
    pub(super) rm_values: Vec<(u8, u8)>,
    pub(super) rm_control_stream: crate::BitStream,
    pub(super) num_samples: usize,
    pub(super) num_deviation_bits: usize,
    pub(super) num_id_bits: usize,
}

#[derive(Debug, Clone)]
pub struct HuffmanDeviationData {
    pub(super) pixel_bit_stream: crate::BitStream,
    pub(super) raw_deviation_bit_stream: Option<crate::BitStream>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseTable {
    Raw(Vec<(crate::BitStream, usize)>),
    Delta(Vec<(crate::BitStream, usize)>),
}

impl BaseTable {
    pub fn as_raw(&self) -> &[(crate::BitStream, usize)] {
        match self {
            BaseTable::Raw(table) | BaseTable::Delta(table) => table.as_slice(),
        }
    }

    pub fn as_raw_mut(&mut self) -> &mut Vec<(crate::BitStream, usize)> {
        match self {
            BaseTable::Raw(table) | BaseTable::Delta(table) => table,
        }
    }

    pub fn len(&self) -> usize {
        self.as_raw().len()
    }

    pub fn is_empty(&self) -> bool {
        self.as_raw().is_empty()
    }
}
