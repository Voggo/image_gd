use bitvec::prelude::*;
use fxhash::FxHashMap;

use super::{DeviationData, DeviationSample, DeviationSampleRef, HuffmanDeviationData};
use crate::error::EntroGdError;

const HUFFMAN_CODE_LENGTH_BITS: usize = 5;
const HUFFMAN_MAX_CODE_LENGTH: u8 = 24;

#[derive(Debug, Clone, Copy)]
struct HuffmanCode {
    symbol: u64,
    code: u32,
    len: u8,
}

impl HuffmanDeviationData {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pixel_bit_stream: BitVec<usize, Msb0>,
        canonical_symbols: Vec<u64>,
        canonical_code_lengths: Vec<u8>,
        row_offsets: Vec<u32>,
        num_samples: usize,
        original_num_samples: usize,
        num_deviation_bits: usize,
        num_id_bits: usize,
        row_width: usize,
    ) -> Result<Self, EntroGdError> {
        Self::new_internal(
            pixel_bit_stream,
            None,
            canonical_symbols,
            canonical_code_lengths,
            row_offsets,
            num_samples,
            original_num_samples,
            num_deviation_bits,
            num_deviation_bits,
            num_id_bits,
            row_width,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_base_id_only(
        pixel_bit_stream: BitVec<usize, Msb0>,
        raw_deviation_bit_stream: BitVec<usize, Msb0>,
        canonical_symbols: Vec<u64>,
        canonical_code_lengths: Vec<u8>,
        row_offsets: Vec<u32>,
        num_samples: usize,
        original_num_samples: usize,
        num_deviation_bits: usize,
        num_id_bits: usize,
        row_width: usize,
    ) -> Result<Self, EntroGdError> {
        Self::new_internal(
            pixel_bit_stream,
            Some(raw_deviation_bit_stream),
            canonical_symbols,
            canonical_code_lengths,
            row_offsets,
            num_samples,
            original_num_samples,
            num_deviation_bits,
            0,
            num_id_bits,
            row_width,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_internal(
        pixel_bit_stream: BitVec<usize, Msb0>,
        raw_deviation_bit_stream: Option<BitVec<usize, Msb0>>,
        canonical_symbols: Vec<u64>,
        canonical_code_lengths: Vec<u8>,
        row_offsets: Vec<u32>,
        num_samples: usize,
        original_num_samples: usize,
        num_deviation_bits: usize,
        huffman_symbol_num_deviation_bits: usize,
        num_id_bits: usize,
        row_width: usize,
    ) -> Result<Self, EntroGdError> {
        if canonical_symbols.len() != canonical_code_lengths.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "Huffman symbol and length tables differ in size".to_string(),
            });
        }
        if original_num_samples > num_samples {
            return Err(EntroGdError::InvalidMetadata {
                message: "original sample count exceeds total sample count".to_string(),
            });
        }
        if original_num_samples > 0 && row_width == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "row width must be > 0 when original samples are present".to_string(),
            });
        }

        let expected_row_count = if original_num_samples == 0 {
            0
        } else {
            if !original_num_samples.is_multiple_of(row_width) {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "original sample count {} is not divisible by row width {}",
                        original_num_samples, row_width
                    ),
                });
            }
            original_num_samples / row_width
        };

        if row_offsets.len() != expected_row_count {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "row offset count mismatch: expected {}, got {}",
                    expected_row_count,
                    row_offsets.len()
                ),
            });
        }

        let symbol_width = huffman_symbol_num_deviation_bits + num_id_bits;
        if symbol_width > 64 {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "Huffman symbol width {} exceeds supported 64-bit range",
                    symbol_width
                ),
            });
        }

        if huffman_symbol_num_deviation_bits > num_deviation_bits {
            return Err(EntroGdError::InvalidMetadata {
                message: "Huffman symbol deviation width cannot exceed declared deviation width"
                    .to_string(),
            });
        }

        let expected_raw_deviation_len =
            num_samples.checked_mul(num_deviation_bits).ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: "raw deviation stream length overflow".to_string(),
                }
            })?;

        if let Some(raw) = raw_deviation_bit_stream.as_ref() {
            if raw.len() != expected_raw_deviation_len {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "raw deviation stream length mismatch: expected {}, got {}",
                        expected_raw_deviation_len,
                        raw.len()
                    ),
                });
            }
            if huffman_symbol_num_deviation_bits != 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message:
                        "raw deviation stream can only be used with base-id-only Huffman symbols"
                            .to_string(),
                });
            }
        }

        let max_symbol = if symbol_width >= 64 {
            u64::MAX
        } else {
            (1u64 << symbol_width) - 1
        };
        for &symbol in &canonical_symbols {
            if symbol > max_symbol {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "Huffman symbol {} exceeds declared symbol width {}",
                        symbol, symbol_width
                    ),
                });
            }
        }

        let codes = rebuild_huffman_codes(&canonical_symbols, &canonical_code_lengths)?;
        let max_code_length = codes.iter().map(|code| code.len).max().unwrap_or(0);
        let mut decode_by_length = vec![FxHashMap::default(); max_code_length as usize + 1];

        for code in &codes {
            if decode_by_length[code.len as usize]
                .insert(code.code, code.symbol)
                .is_some()
            {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "duplicate Huffman code {} with length {}",
                        code.code, code.len
                    ),
                });
            }
        }

        if num_samples > 0 && codes.is_empty() {
            return Err(EntroGdError::InvalidMetadata {
                message: "non-empty Huffman stream is missing a symbol table".to_string(),
            });
        }

        let mut previous_offset = 0usize;
        for &offset in &row_offsets {
            let current_offset = offset as usize;
            if current_offset < previous_offset {
                return Err(EntroGdError::InvalidMetadata {
                    message: "row offsets must be non-decreasing".to_string(),
                });
            }
            if current_offset > pixel_bit_stream.len() {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "row offset {} exceeds Huffman pixel stream length {}",
                        current_offset,
                        pixel_bit_stream.len()
                    ),
                });
            }
            previous_offset = current_offset;
        }

        let mut data = HuffmanDeviationData {
            pixel_bit_stream,
            raw_deviation_bit_stream,
            canonical_symbols,
            canonical_code_lengths,
            row_offsets,
            num_samples,
            original_num_samples,
            num_deviation_bits,
            huffman_symbol_num_deviation_bits,
            num_id_bits,
            row_width,
            max_code_length,
            decode_by_length,
            original_stream_end_offset_bits: 0,
        };

        data.original_stream_end_offset_bits = if data.original_num_samples == 0 {
            0
        } else {
            let last_row_offset = data.row_offsets[data.row_offsets.len() - 1] as usize;
            let last_row_len = data.row_width;
            data.advance_by_symbols(last_row_offset, last_row_len)?
        };

        if data.original_stream_end_offset_bits > data.pixel_bit_stream.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "decoded original stream length exceeds Huffman bitstream size"
                    .to_string(),
            });
        }

        Ok(data)
    }

    pub fn pixel_bit_stream(&self) -> &BitVec<usize, Msb0> {
        &self.pixel_bit_stream
    }

    pub fn raw_deviation_bit_stream(&self) -> Option<&BitVec<usize, Msb0>> {
        self.raw_deviation_bit_stream.as_ref()
    }

    pub fn huffman_symbol_num_deviation_bits(&self) -> usize {
        self.huffman_symbol_num_deviation_bits
    }

    pub fn canonical_symbols(&self) -> &[u64] {
        &self.canonical_symbols
    }

    pub fn canonical_code_lengths(&self) -> &[u8] {
        &self.canonical_code_lengths
    }

    pub fn row_offsets(&self) -> &[u32] {
        &self.row_offsets
    }

    pub fn row_width(&self) -> usize {
        self.row_width
    }

    pub fn original_num_samples(&self) -> usize {
        self.original_num_samples
    }

    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        if sample_idx >= self.num_samples {
            return None;
        }

        let (start_bit, symbols_to_decode) = if sample_idx < self.original_num_samples {
            let row_idx = sample_idx / self.row_width;
            let col_idx = sample_idx % self.row_width;
            (
                *self.row_offsets.get(row_idx)? as usize,
                col_idx.saturating_add(1),
            )
        } else {
            (
                self.original_stream_end_offset_bits,
                sample_idx
                    .checked_sub(self.original_num_samples)?
                    .saturating_add(1),
            )
        };

        let mut bit_pos = start_bit;
        let mut symbol = 0u64;
        for _ in 0..symbols_to_decode {
            let (decoded_symbol, decoded_len) = self.decode_one(bit_pos).ok()?;
            symbol = decoded_symbol;
            bit_pos += decoded_len;
        }

        let deviation = if let Some(raw_deviation) = &self.raw_deviation_bit_stream {
            let start = sample_idx.checked_mul(self.num_deviation_bits)?;
            let end = start.checked_add(self.num_deviation_bits)?;
            raw_deviation.get(start..end)?.to_bitvec()
        } else {
            let deviation_mask = bit_mask(self.num_deviation_bits);
            let deviation_value = symbol & deviation_mask;
            bitvec_from_u64(deviation_value, self.num_deviation_bits)
        };
        let id_value = symbol >> self.huffman_symbol_num_deviation_bits;

        Some(DeviationSample {
            deviation,
            id: bitvec_from_u64(id_value, self.num_id_bits),
        })
    }

    pub fn get_num_samples(&self) -> usize {
        self.num_samples
    }

    pub fn get_num_deviation_bits(&self) -> usize {
        self.num_deviation_bits
    }

    pub fn get_num_id_bits(&self) -> usize {
        self.num_id_bits
    }

    pub fn get_encoded_size(&self) -> usize {
        let symbol_width = self.huffman_symbol_num_deviation_bits + self.num_id_bits;
        16 + 8
            + 8
            + 32
            + self.row_offsets.len() * 32
            + self.canonical_symbols.len() * (symbol_width + HUFFMAN_CODE_LENGTH_BITS)
            + self
                .raw_deviation_bit_stream
                .as_ref()
                .map(|raw| raw.len())
                .unwrap_or(0)
            + self.pixel_bit_stream.len()
    }

    pub(crate) fn for_each_sample(
        &self,
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        let raw = self.to_deviation_data()?;
        raw.for_each_sample(|sample| {
            f(DeviationSampleRef {
                deviation: sample.deviation,
                id: sample.id,
            })
        })
    }

    pub fn to_deviation_data(&self) -> Result<DeviationData, EntroGdError> {
        let symbol_width = self.num_deviation_bits + self.num_id_bits;
        let expected_bits = self.num_samples.checked_mul(symbol_width).ok_or_else(|| {
            EntroGdError::InvalidMetadata {
                message: "decoded Huffman deviation stream length overflow".to_string(),
            }
        })?;
        let mut raw = BitVec::<usize, Msb0>::with_capacity(expected_bits);
        let mut bit_pos = 0usize;

        for sample_idx in 0..self.num_samples {
            let (symbol, consumed_bits) = self.decode_one(bit_pos)?;

            if let Some(raw_deviation) = &self.raw_deviation_bit_stream {
                let deviation_start = sample_idx * self.num_deviation_bits;
                let deviation_end = deviation_start + self.num_deviation_bits;
                raw.extend_from_bitslice(&raw_deviation[deviation_start..deviation_end]);
            } else {
                let deviation_value = symbol & bit_mask(self.num_deviation_bits);
                append_symbol_bits(&mut raw, deviation_value, self.num_deviation_bits);
            }

            let id_value = symbol >> self.huffman_symbol_num_deviation_bits;
            append_symbol_bits(&mut raw, id_value, self.num_id_bits);
            bit_pos += consumed_bits;
        }

        Ok(DeviationData::new(
            raw,
            self.num_samples,
            self.num_deviation_bits,
            self.num_id_bits,
        ))
    }

    fn advance_by_symbols(
        &self,
        start_bit: usize,
        symbol_count: usize,
    ) -> Result<usize, EntroGdError> {
        let mut bit_pos = start_bit;
        for _ in 0..symbol_count {
            let (_, consumed_bits) = self.decode_one(bit_pos)?;
            bit_pos += consumed_bits;
        }
        Ok(bit_pos)
    }

    fn decode_one(&self, bit_pos: usize) -> Result<(u64, usize), EntroGdError> {
        if self.max_code_length == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "attempted to decode from an empty Huffman table".to_string(),
            });
        }

        let mut code = 0u32;
        for len in 1..=self.max_code_length as usize {
            let next_bit_pos = bit_pos + len - 1;
            if next_bit_pos >= self.pixel_bit_stream.len() {
                return Err(EntroGdError::InvalidMetadata {
                    message: "unexpected end of Huffman pixel stream".to_string(),
                });
            }

            code = (code << 1) | u32::from(self.pixel_bit_stream[next_bit_pos]);
            if let Some(symbol) = self.decode_by_length[len].get(&code) {
                return Ok((*symbol, len));
            }
        }

        Err(EntroGdError::InvalidMetadata {
            message: "failed to decode Huffman symbol from bitstream".to_string(),
        })
    }
}

fn rebuild_huffman_codes(
    canonical_symbols: &[u64],
    canonical_code_lengths: &[u8],
) -> Result<Vec<HuffmanCode>, EntroGdError> {
    if canonical_symbols.is_empty() {
        return Ok(Vec::new());
    }

    let mut codes = Vec::with_capacity(canonical_symbols.len());
    let mut next_code = 0u32;
    let mut previous_len = 0u8;
    let mut previous_symbol = None;

    for (&symbol, &len) in canonical_symbols.iter().zip(canonical_code_lengths.iter()) {
        if !(1..=HUFFMAN_MAX_CODE_LENGTH).contains(&len) {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("invalid Huffman code length {}", len),
            });
        }

        if let Some((prev_len, prev_symbol_value)) = previous_symbol
            && (len < prev_len || (len == prev_len && symbol <= prev_symbol_value))
        {
            return Err(EntroGdError::InvalidMetadata {
                message: "Huffman canonical table is not sorted by (length, symbol)".to_string(),
            });
        }

        next_code <<= (len - previous_len) as u32;
        if (next_code as u64) >= (1u64 << len) {
            return Err(EntroGdError::InvalidMetadata {
                message: "Huffman canonical table violates prefix-code bounds".to_string(),
            });
        }

        codes.push(HuffmanCode {
            symbol,
            code: next_code,
            len,
        });

        previous_len = len;
        previous_symbol = Some((len, symbol));
        next_code = next_code.saturating_add(1);
    }

    Ok(codes)
}

pub(crate) fn build_huffman_code_map(
    canonical_symbols: &[u64],
    canonical_code_lengths: &[u8],
) -> Result<FxHashMap<u64, (u32, u8)>, EntroGdError> {
    let mut codes_by_symbol = FxHashMap::default();
    for code in rebuild_huffman_codes(canonical_symbols, canonical_code_lengths)? {
        if codes_by_symbol
            .insert(code.symbol, (code.code, code.len))
            .is_some()
        {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("duplicate Huffman symbol {}", code.symbol),
            });
        }
    }
    Ok(codes_by_symbol)
}

fn bit_mask(width: usize) -> u64 {
    if width == 0 {
        0
    } else if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn bitvec_from_u64(value: u64, width: usize) -> BitVec<usize, Msb0> {
    let mut bits = BitVec::<usize, Msb0>::with_capacity(width);
    append_symbol_bits(&mut bits, value, width);
    bits
}

fn append_symbol_bits(out: &mut BitVec<usize, Msb0>, value: u64, width: usize) {
    for shift in (0..width).rev() {
        out.push(((value >> shift) & 1) == 1);
    }
}
