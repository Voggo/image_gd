use fxhash::{FxHashMap, FxHashSet};
use std::fs;
use std::path::{Path, PathBuf};

use super::bit_io::{BitReader, BitWriter};
use super::path_utils::{ensure_csv_extension, ensure_egd_extension};
use super::tags::{
    BASE_TABLE_TAG_DELTA, BASE_TABLE_TAG_RAW, ENCODING_TAG_HUFFMAN_BASE_ID_ONLY,
    ENCODING_TAG_HUFFMAN_CANONICAL, ENCODING_TAG_NORMAL, ENCODING_TAG_RLE_RM_PACKED,
    HUFFMAN_CODE_LENGTH_BITS, decode_data_type, encode_data_type,
};

use crate::compression::decompression::{decompress_file, write_bitdata_as_csv};
use crate::compression::encoding::{
    BaseTable, CompressedData, DeltaBaseTableData, DeviationData, EncodedData,
    HuffmanDeviationData, RLE_LONG_MAX, RLE_SHORT_MAX, RLE_TERMINATOR_PAYLOAD, RleDeviationData,
};
use crate::compression::preprocessor::{BitDataInfo, BitDataSet, FeatureSpec, FeatureTransform};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::utils::bits_needed_nonzero;

pub const MAGIC_BYTES: [u8; 3] = *b"EGD";
pub const FORMAT_VERSION: u8 = 1;

fn decode_delta_base_rows(
    num_bases: usize,
    lb: usize,
    order: &[usize],
    first_sort_key: &crate::BitView,
    delta_count: usize,
    delta_bit_stream: &crate::BitView,
) -> Result<Vec<(crate::BitStream, usize)>, EntroGdError> {
    if num_bases == 0 {
        return Ok(Vec::new());
    }

    if delta_count != num_bases.saturating_sub(1) {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "delta_count mismatch: expected {}, got {}",
                num_bases.saturating_sub(1),
                delta_count
            ),
        });
    }

    if order.iter().any(|&idx| idx >= lb) {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta sort column order contains out-of-range index".to_string(),
        });
    }

    let mut rows = Vec::with_capacity(num_bases);
    let mut prev_key = first_sort_key.to_bitvec();
    rows.push((sort_key_to_row(&prev_key, order, lb), 0usize));

    let mut bit_pos = 0usize;
    for _ in 0..delta_count {
        let d = decode_adjusted_delta(delta_bit_stream, &mut bit_pos, lb)?;
        let delta = add_one(d.as_bitslice());
        let next_key = subtract_unsigned(prev_key.as_bitslice(), delta.as_bitslice());
        rows.push((sort_key_to_row(&next_key, order, lb), 0usize));
        prev_key = next_key;
    }

    if bit_pos != delta_bit_stream.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta bitstream has trailing/unused bits".to_string(),
        });
    }

    Ok(rows)
}

fn sort_key_to_row(sort_key: &crate::BitStream, order: &[usize], lb: usize) -> crate::BitStream {
    let mut row = crate::BitStream::repeat(false, lb);
    for (rank, &col_idx) in order.iter().enumerate() {
        let key_idx = lb.saturating_sub(1 + rank);
        let bit = sort_key.get(key_idx).map(|b| *b).unwrap_or(false);
        if col_idx < lb {
            row.set(col_idx, bit);
        }
    }
    row
}

fn decode_adjusted_delta(
    bits: &crate::BitView,
    bit_pos: &mut usize,
    lb: usize,
) -> Result<crate::BitStream, EntroGdError> {
    let mut tier = 0usize;
    while *bit_pos < bits.len() && bits[*bit_pos] {
        tier += 1;
        *bit_pos += 1;
        if tier == 8 {
            break;
        }
    }

    if tier < 8 {
        if *bit_pos >= bits.len() || bits[*bit_pos] {
            return Err(EntroGdError::InvalidMetadata {
                message: "invalid delta prefix terminator".to_string(),
            });
        }
        *bit_pos += 1;
    }

    let (payload_width, start): (usize, u64) = match tier {
        0 => (2, 0),
        1 => (4, 4),
        2 => (7, 20),
        3 => (10, 84),
        4 => (13, 1108),
        5 => (16, 9300),
        6 => (19, 74772),
        7 => (38, 598068),
        8 => (lb, 0),
        _ => {
            return Err(EntroGdError::InvalidMetadata {
                message: "invalid delta tier".to_string(),
            });
        }
    };

    if *bit_pos + payload_width > bits.len() {
        return Err(EntroGdError::InvalidMetadata {
            message: "delta payload exceeds bitstream".to_string(),
        });
    }

    let payload = unsafe { bits.get_unchecked(*bit_pos..*bit_pos + payload_width) };
    *bit_pos += payload_width;

    if tier == 8 {
        return Ok(payload.to_bitvec());
    }

    let mut payload_value = 0u64;
    for idx in 0..payload_width {
        if payload[idx] {
            payload_value |= 1u64 << idx;
        }
    }
    let value = start + payload_value;

    let mut out = crate::BitStream::new();
    let mut v = value;
    while v > 0 {
        out.push((v & 1) == 1);
        v >>= 1;
    }
    Ok(out)
}

fn add_one(bits: &crate::BitView) -> crate::BitStream {
    let mut out = bits.to_bitvec();
    let mut carry = true;
    let mut idx = 0usize;
    while carry {
        if idx >= out.len() {
            out.push(true);
            break;
        }
        let bit = out[idx];
        out.set(idx, !bit);
        carry = bit;
        idx += 1;
    }
    out
}

fn subtract_unsigned(minuend: &crate::BitView, subtrahend: &crate::BitView) -> crate::BitStream {
    let max_len = minuend.len().max(subtrahend.len());
    let mut out = crate::BitStream::with_capacity(max_len);

    let mut borrow: i8 = 0;
    for idx in 0..max_len {
        let a = if minuend.get(idx).map(|b| *b).unwrap_or(false) {
            1
        } else {
            0
        };
        let b = if subtrahend.get(idx).map(|b| *b).unwrap_or(false) {
            1
        } else {
            0
        };
        let mut diff = a - b - borrow;
        if diff < 0 {
            diff += 2;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(diff == 1);
    }
    while out.last().map(|b| *b) == Some(false) {
        out.pop();
    }
    out
}

/// In-memory EGD file contents that can be saved to disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EgdFile {
    bytes: Vec<u8>,
}

impl EgdFile {
    /// Build an EGD file from compression output.
    ///
    /// Bitstream layout:
    /// 1) header: magic ("EGD") + version
    /// 2) global: n (u64), m (u64), num_features (u64)
    /// 3) feature metadata (per feature):
    ///    tag=1 (u8), data_type (u8), transform_tag (u8), [transform params], bits_per_feature (u16), 2-bit bit-state per feature bit
    ///    bit-state encoding: 0=deviation, 1=variable base, 2=constant-zero base, 3=constant-one base
    /// 4) zero-padding to next byte
    /// 5) condensed weights bitstream (m * ceil(log2(n)) bits)
    /// 6) zero-padding to next byte
    /// 7) base table: num_bases (u64) then packed base bits for each base
    /// 8) byte-align, then encoded data section:
    ///    - encoding tag (u8)
    ///    - payload (depends on tag)
    ///      tag=0: raw encoded data stream bits
    ///      tag=1: rm-control packed stream (alternating r,m 4/8-bit packets, terminated by 0xFF), then symbol stream bits
    ///      tag=2: canonical Huffman table + row offsets + pixel bitstream
    pub fn from_compressed_data(compressed: &CompressedData) -> Result<Self, EntroGdError> {
        let data_info = &compressed.metadata;
        let num_features = data_info.num_features();
        if num_features == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "num_features is 0".to_string(),
            });
        }

        let chunk_size = data_info.chunk_size();
        let original_num_rows = data_info.original_size_bits() / chunk_size;
        let condensed_weights = compressed
            .condensed_sample_weights
            .as_deref()
            .unwrap_or(&[]);

        let selected_positions = compressed.layout.selected_base_bit_positions.clone();
        let variable_positions = compressed.layout.variable_base_bit_positions.clone();
        let constant_zero_positions = compressed.layout.constant_zero_bit_positions.clone();
        let constant_one_positions = compressed.layout.constant_one_bit_positions.clone();

        let variable_set: FxHashSet<usize> = variable_positions.iter().copied().collect();
        let constant_zero_set: FxHashSet<usize> = constant_zero_positions.iter().copied().collect();
        let constant_one_set: FxHashSet<usize> = constant_one_positions.iter().copied().collect();

        let selected_from_split: FxHashSet<usize> = variable_set
            .iter()
            .copied()
            .chain(constant_zero_set.iter().copied())
            .chain(constant_one_set.iter().copied())
            .collect();
        let selected_set: FxHashSet<usize> = selected_positions.iter().copied().collect();
        if selected_from_split != selected_set {
            return Err(EntroGdError::InvalidMetadata {
                message: "selected base positions do not match variable/constant split".to_string(),
            });
        }

        let mut variable_position_to_index =
            FxHashMap::with_capacity_and_hasher(variable_positions.len(), Default::default());
        for (idx, &bit_pos) in variable_positions.iter().enumerate() {
            if variable_position_to_index.insert(bit_pos, idx).is_some() {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("duplicate variable base bit position {}", bit_pos),
                });
            }
        }

        let mut variable_positions_in_metadata_order = Vec::with_capacity(variable_positions.len());

        let n_u64 =
            u64::try_from(original_num_rows).map_err(|_| EntroGdError::InvalidMetadata {
                message: "n does not fit into u64".to_string(),
            })?;
        let m_u64 =
            u64::try_from(condensed_weights.len()).map_err(|_| EntroGdError::InvalidMetadata {
                message: "m does not fit into u64".to_string(),
            })?;
        let num_features_u64 =
            u64::try_from(num_features).map_err(|_| EntroGdError::InvalidMetadata {
                message: "num_features does not fit into u64".to_string(),
            })?;

        let mut writer = BitWriter::new();

        // Header
        writer.write_u8(MAGIC_BYTES[0]);
        writer.write_u8(MAGIC_BYTES[1]);
        writer.write_u8(MAGIC_BYTES[2]);
        writer.write_u8(FORMAT_VERSION);

        // Global
        writer.write_u64(n_u64);
        writer.write_u64(m_u64);
        writer.write_u64(num_features_u64);

        // Feature metadata
        for feature_idx in 0..num_features {
            let feature_spec = data_info.feature_spec(feature_idx);
            let feature_bits = data_info.feature_bits(feature_idx);
            let bits_per_feature =
                u16::try_from(feature_bits).map_err(|_| EntroGdError::InvalidMetadata {
                    message: format!(
                        "feature {} bits {} does not fit into u16",
                        feature_idx, feature_bits
                    ),
                })?;

            writer.write_u8(1); // BitInfo tag
            writer.write_u8(encode_data_type(feature_spec.data_type));
            match feature_spec.transform {
                FeatureTransform::None => writer.write_u8(0),
                FeatureTransform::ScaledSignedInt { decimal_scale } => {
                    writer.write_u8(1);
                    writer.write_u8(decimal_scale);
                }
                FeatureTransform::OffsetSignedInt { min_value } => {
                    writer.write_u8(2);
                    writer.write_u64(min_value as u64);
                }
                FeatureTransform::OffsetUnsignedInt { min_value } => {
                    writer.write_u8(3);
                    writer.write_u64(min_value);
                }
                FeatureTransform::ScaledOffsetSignedInt {
                    decimal_scale,
                    min_value,
                } => {
                    writer.write_u8(4);
                    writer.write_u8(decimal_scale);
                    writer.write_u64(min_value as u64);
                }
            }
            writer.write_u16(bits_per_feature);

            let offset = data_info.feature_offset(feature_idx);
            for local_bit in 0..feature_bits {
                let global_bit = offset + local_bit;
                let bit_state = if variable_set.contains(&global_bit) {
                    variable_positions_in_metadata_order.push(global_bit);
                    1usize
                } else if constant_zero_set.contains(&global_bit) {
                    2usize
                } else if constant_one_set.contains(&global_bit) {
                    3usize
                } else {
                    0usize
                };
                writer.write_usize_bits(bit_state, 2);
            }
        }

        // Align after feature metadata
        writer.align_to_byte();

        let mut metadata_variable_position_to_index = FxHashMap::with_capacity_and_hasher(
            variable_positions_in_metadata_order.len(),
            Default::default(),
        );
        for (metadata_idx, &global_bit) in variable_positions_in_metadata_order.iter().enumerate() {
            metadata_variable_position_to_index.insert(global_bit, metadata_idx);
        }

        // Condensed sample weights
        let weight_bits = bits_needed_nonzero(original_num_rows);
        for &weight in condensed_weights {
            writer.write_usize_bits(weight, weight_bits);
        }

        // Align after weights
        writer.align_to_byte();

        // Base table
        match &compressed.base_table {
            BaseTable::Raw(base_table) => {
                writer.write_u8(BASE_TABLE_TAG_RAW);
                let num_bases =
                    u64::try_from(base_table.len()).map_err(|_| EntroGdError::InvalidMetadata {
                        message: "num_bases does not fit into u64".to_string(),
                    })?;
                writer.write_u64(num_bases);
                for (base_bits, _) in base_table {
                    for &global_bit in &variable_positions_in_metadata_order {
                        let variable_idx = *variable_position_to_index
                            .get(&global_bit)
                            .ok_or_else(|| EntroGdError::InvalidMetadata {
                                message: format!(
                                    "variable base position {} missing from index map",
                                    global_bit
                                ),
                            })?;
                        writer.write_bit(base_bits.get(variable_idx).map(|b| *b).unwrap_or(false));
                    }
                }
            }
            BaseTable::Delta(delta) => {
                writer.write_u8(BASE_TABLE_TAG_DELTA);
                let num_bases = u64::try_from(delta.raw_rows.len()).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "num_bases does not fit into u64".to_string(),
                    }
                })?;
                writer.write_u64(num_bases);

                let mapped_order: Vec<usize> = delta
                    .sort_column_order
                    .iter()
                    .map(|&current_idx| {
                        let &global_bit = variable_positions.get(current_idx).ok_or_else(|| {
                            EntroGdError::InvalidMetadata {
                                message: format!(
                                    "delta sort column index {} out of range {}",
                                    current_idx,
                                    variable_positions.len()
                                ),
                            }
                        })?;
                        metadata_variable_position_to_index
                            .get(&global_bit)
                            .copied()
                            .ok_or_else(|| EntroGdError::InvalidMetadata {
                                message: format!(
                                    "variable base position {} missing from metadata index map",
                                    global_bit
                                ),
                            })
                    })
                    .collect::<Result<Vec<_>, EntroGdError>>()?;

                let order_len = u64::try_from(mapped_order.len()).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "sort_column_order length does not fit into u64".to_string(),
                    }
                })?;
                writer.write_u64(order_len);
                let index_bits = bits_needed_nonzero(variable_positions.len().max(1));
                for &idx in &mapped_order {
                    writer.write_usize_bits(idx, index_bits);
                }

                let lb = variable_positions.len();
                if !delta.raw_rows.is_empty() {
                    for idx in 0..lb {
                        writer
                            .write_bit(delta.first_sort_key.get(idx).map(|b| *b).unwrap_or(false));
                    }
                }

                writer.write_u64(u64::try_from(delta.delta_count).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "delta_count does not fit into u64".to_string(),
                    }
                })?);
                writer.write_u64(u64::try_from(delta.delta_bit_stream.len()).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "delta bitstream length does not fit into u64".to_string(),
                    }
                })?);
                writer.write_bitslice(delta.delta_bit_stream.as_bitslice());
            }
        }

        // Encoded data section: byte-aligned + tagged payload.
        writer.align_to_byte();
        match &compressed.encoded_data {
            EncodedData::Normal(raw) => {
                writer.write_u8(ENCODING_TAG_NORMAL);
                writer.write_bitslice(raw.encoded_bit_stream());
            }
            EncodedData::Rle(rle) => {
                writer.write_u8(ENCODING_TAG_RLE_RM_PACKED);
                writer.write_bitslice(rle.rm_control_stream());
                writer.write_bitslice(rle.symbol_bit_stream());
            }
            EncodedData::Huffman(huffman) => {
                let is_base_id_only = huffman.raw_deviation_bit_stream().is_some();
                writer.write_u8(if is_base_id_only {
                    ENCODING_TAG_HUFFMAN_BASE_ID_ONLY
                } else {
                    ENCODING_TAG_HUFFMAN_CANONICAL
                });

                writer.write_u32(u32::try_from(huffman.canonical_symbols().len()).map_err(
                    |_| EntroGdError::InvalidMetadata {
                        message: "Huffman symbol count does not fit into u32".to_string(),
                    },
                )?);
                writer.write_u8(u8::try_from(huffman.get_num_id_bits()).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "Huffman l_id does not fit into u8".to_string(),
                    }
                })?);
                writer.write_u8(u8::try_from(huffman.get_num_deviation_bits()).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "Huffman l_d does not fit into u8".to_string(),
                    }
                })?);
                writer.write_u32(u32::try_from(huffman.row_offsets().len()).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "Huffman row count does not fit into u32".to_string(),
                    }
                })?);

                let symbol_width = if is_base_id_only {
                    huffman.get_num_id_bits()
                } else {
                    huffman.get_num_id_bits() + huffman.get_num_deviation_bits()
                };
                for (&symbol, &code_len) in huffman
                    .canonical_symbols()
                    .iter()
                    .zip(huffman.canonical_code_lengths().iter())
                {
                    writer.write_u64_bits(symbol, symbol_width);
                    writer.write_usize_bits((code_len - 1) as usize, HUFFMAN_CODE_LENGTH_BITS);
                }
                for &offset in huffman.row_offsets() {
                    writer.write_u32(offset);
                }

                if let Some(raw_deviation) = huffman.raw_deviation_bit_stream() {
                    writer.write_bitslice(raw_deviation);
                }
                writer.write_bitslice(huffman.pixel_bit_stream());
            }
        }

        Ok(EgdFile {
            bytes: writer.into_bytes(),
        })
    }

    /// Build an EGD file wrapper from raw bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        EgdFile { bytes }
    }

    /// Load an `.egd` file from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, EntroGdError> {
        let bytes = fs::read(path)?;
        Ok(EgdFile { bytes })
    }

    /// Parse this EGD file into an in-memory `CompressedData` object.
    pub fn to_compressed_data(&self) -> Result<CompressedData, EntroGdError> {
        let mut reader = BitReader::new(&self.bytes);

        // Header
        let m0 = reader.read_u8()?;
        let m1 = reader.read_u8()?;
        let m2 = reader.read_u8()?;
        if [m0, m1, m2] != MAGIC_BYTES {
            return Err(EntroGdError::InvalidMetadata {
                message: "invalid magic bytes (expected EGD)".to_string(),
            });
        }
        let version = reader.read_u8()?;
        if version != FORMAT_VERSION {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported format version {} (expected {})",
                    version, FORMAT_VERSION
                ),
            });
        }

        // Global
        let n = usize::try_from(reader.read_u64()?).map_err(|_| EntroGdError::InvalidMetadata {
            message: "n does not fit into usize".to_string(),
        })?;
        let m = usize::try_from(reader.read_u64()?).map_err(|_| EntroGdError::InvalidMetadata {
            message: "m does not fit into usize".to_string(),
        })?;
        let num_features =
            usize::try_from(reader.read_u64()?).map_err(|_| EntroGdError::InvalidMetadata {
                message: "num_features does not fit into usize".to_string(),
            })?;
        if num_features == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "num_features is 0".to_string(),
            });
        }

        // Feature metadata
        let mut features = Vec::with_capacity(num_features);
        let mut base_bit_positions = Vec::new();
        let mut variable_base_bit_positions = Vec::new();
        let mut constant_zero_bit_positions = Vec::new();
        let mut constant_one_bit_positions = Vec::new();
        let mut running_offset = 0usize;
        for feature_idx in 0..num_features {
            let tag = reader.read_u8()?;
            if tag != 1 {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "unsupported feature metadata tag {} at index {}",
                        tag, feature_idx
                    ),
                });
            }
            let data_type = decode_data_type(reader.read_u8()?)?;
            let transform = match reader.read_u8()? {
                0 => FeatureTransform::None,
                1 => {
                    let decimal_scale = reader.read_u8()?;
                    FeatureTransform::ScaledSignedInt { decimal_scale }
                }
                2 => {
                    let min_value = reader.read_u64()? as i64;
                    FeatureTransform::OffsetSignedInt { min_value }
                }
                3 => {
                    let min_value = reader.read_u64()?;
                    FeatureTransform::OffsetUnsignedInt { min_value }
                }
                4 => {
                    let decimal_scale = reader.read_u8()?;
                    let min_value = reader.read_u64()? as i64;
                    FeatureTransform::ScaledOffsetSignedInt {
                        decimal_scale,
                        min_value,
                    }
                }
                transform_tag => {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "unsupported transform tag {} at feature {}",
                            transform_tag, feature_idx
                        ),
                    });
                }
            };
            let bits = usize::from(reader.read_u16()?);
            if bits == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("feature {} has 0 bits", feature_idx),
                });
            }

            features.push(FeatureSpec {
                data_type,
                bits,
                transform,
            });

            for local_bit in 0..bits {
                let global_bit = running_offset + local_bit;
                match reader.read_usize_bits(2)? {
                    0 => {}
                    1 => {
                        base_bit_positions.push(global_bit);
                        variable_base_bit_positions.push(global_bit);
                    }
                    2 => {
                        base_bit_positions.push(global_bit);
                        constant_zero_bit_positions.push(global_bit);
                    }
                    3 => {
                        base_bit_positions.push(global_bit);
                        constant_one_bit_positions.push(global_bit);
                    }
                    other => {
                        return Err(EntroGdError::InvalidMetadata {
                            message: format!(
                                "invalid base bit-state {} at feature {} local bit {}",
                                other, feature_idx, local_bit
                            ),
                        });
                    }
                }
            }
            running_offset =
                running_offset
                    .checked_add(bits)
                    .ok_or_else(|| EntroGdError::InvalidMetadata {
                        message: "chunk size overflow while parsing features".to_string(),
                    })?;
        }

        // Align after feature metadata
        reader.align_to_byte();

        // Condensed sample weights
        let weight_bits = bits_needed_nonzero(n);
        let mut weights = Vec::with_capacity(m);
        for _ in 0..m {
            weights.push(reader.read_usize_bits(weight_bits)?);
        }

        // Align after weights
        reader.align_to_byte();

        // Base table
        let base_table_tag = reader.read_u8()?;
        let num_bases =
            usize::try_from(reader.read_u64()?).map_err(|_| EntroGdError::InvalidMetadata {
                message: "num_bases does not fit into usize".to_string(),
            })?;

        let chunk_size = features.iter().map(|f| f.bits).sum::<usize>();
        if base_bit_positions.len() > chunk_size {
            return Err(EntroGdError::InvalidMetadata {
                message: "base bit positions exceed chunk size".to_string(),
            });
        }

        let mut entropy_sorted_column_order: Option<Vec<usize>> = None;
        let mut base_table = match base_table_tag {
            BASE_TABLE_TAG_RAW => {
                let mut rows = Vec::with_capacity(num_bases);
                for _ in 0..num_bases {
                    let mut base_bits =
                        crate::BitStream::with_capacity(variable_base_bit_positions.len());
                    for _ in 0..variable_base_bit_positions.len() {
                        base_bits.push(reader.read_bit()?);
                    }
                    rows.push((base_bits, 0usize));
                }
                BaseTable::Raw(rows)
            }
            BASE_TABLE_TAG_DELTA => {
                let order_len = usize::try_from(reader.read_u64()?).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "sort_column_order length does not fit into usize".to_string(),
                    }
                })?;
                let lb = variable_base_bit_positions.len();
                let index_bits = bits_needed_nonzero(lb.max(1));
                let mut order = Vec::with_capacity(order_len);
                for _ in 0..order_len {
                    let idx = reader.read_usize_bits(index_bits)?;
                    if idx >= lb {
                        return Err(EntroGdError::InvalidMetadata {
                            message: format!("delta sort column index {} out of range {}", idx, lb),
                        });
                    }
                    order.push(idx);
                }
                let mut first_sort_key = crate::BitStream::with_capacity(lb);
                if num_bases > 0 {
                    for _ in 0..lb {
                        first_sort_key.push(reader.read_bit()?);
                    }
                }

                let delta_count = usize::try_from(reader.read_u64()?).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "delta_count does not fit into usize".to_string(),
                    }
                })?;
                let delta_bit_len = usize::try_from(reader.read_u64()?).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "delta bitstream length does not fit into usize".to_string(),
                    }
                })?;
                let delta_bit_stream = reader.read_bits(delta_bit_len)?;

                let rows = decode_delta_base_rows(
                    num_bases,
                    lb,
                    &order,
                    first_sort_key.as_bitslice(),
                    delta_count,
                    delta_bit_stream.as_bitslice(),
                )?;
                entropy_sorted_column_order = Some(order.clone());
                BaseTable::Delta(DeltaBaseTableData {
                    raw_rows: rows,
                    first_sort_key,
                    delta_bit_stream,
                    delta_count,
                    sort_column_order: order,
                })
            }
            other => {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("unsupported base-table tag {}", other),
                });
            }
        };

        let num_samples = n
            .checked_add(m)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "n + m overflows usize".to_string(),
            })?;
        let num_id_bits = bits_needed_nonzero(num_bases);
        let num_deviation_bits = chunk_size.saturating_sub(base_bit_positions.len());

        let bits_per_sample = num_deviation_bits + num_id_bits;

        // Encoded data section (byte-aligned + tagged payload)
        reader.align_to_byte();
        let encoding_tag = reader.read_u8()?;
        let encoded_data = match encoding_tag {
            ENCODING_TAG_NORMAL => {
                let expected_encoded_len =
                    num_samples.checked_mul(bits_per_sample).ok_or_else(|| {
                        EntroGdError::InvalidMetadata {
                            message: "encoded stream expected length overflow".to_string(),
                        }
                    })?;
                let encoded_stream = reader.read_bits(expected_encoded_len)?;
                EncodedData::Normal(DeviationData::new(
                    encoded_stream,
                    num_samples,
                    num_deviation_bits,
                    num_id_bits,
                ))
            }
            ENCODING_TAG_RLE_RM_PACKED => {
                let mut rm_values: Vec<(u8, u8)> = Vec::new();
                let mut flat_values: Vec<u8> = Vec::new();
                loop {
                    let is_long_packet = reader.read_bit()?;
                    if !is_long_packet {
                        let val = reader.read_usize_bits(3)? as u8;
                        flat_values.push(val);
                        continue;
                    }

                    let payload = reader.read_usize_bits(7)? as u8;
                    if payload == RLE_TERMINATOR_PAYLOAD {
                        break;
                    }

                    let val = payload.saturating_add(8);
                    if !(RLE_SHORT_MAX + 1..=RLE_LONG_MAX).contains(&val) {
                        return Err(EntroGdError::InvalidMetadata {
                            message: "invalid rm control packet value".to_string(),
                        });
                    }
                    flat_values.push(val);
                }

                if !flat_values.len().is_multiple_of(2) {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "invalid rm control stream: odd number of values".to_string(),
                    });
                }

                for pair in flat_values.chunks_exact(2) {
                    rm_values.push((pair[0], pair[1]));
                }

                let symbol_count = rm_values
                    .iter()
                    .try_fold(0usize, |acc, (r, m_val)| {
                        let run_symbols = if *r > 0 { 1usize } else { 0usize };
                        let literals = *m_val as usize;
                        acc.checked_add(run_symbols + literals)
                    })
                    .ok_or_else(|| EntroGdError::InvalidMetadata {
                        message: "rm symbol count overflow".to_string(),
                    })?;

                let expected_symbol_bits =
                    symbol_count.checked_mul(bits_per_sample).ok_or_else(|| {
                        EntroGdError::InvalidMetadata {
                            message: "rle symbol stream expected length overflow".to_string(),
                        }
                    })?;

                let symbol_stream = reader.read_bits(expected_symbol_bits)?;
                let rle = RleDeviationData::new(
                    symbol_stream,
                    rm_values,
                    num_samples,
                    num_deviation_bits,
                    num_id_bits,
                );
                EncodedData::Normal(rle.to_deviation_data()?)
            }
            ENCODING_TAG_HUFFMAN_CANONICAL => {
                let symbol_count = usize::try_from(reader.read_u32()?).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "Huffman symbol count does not fit into usize".to_string(),
                    }
                })?;
                let huffman_num_id_bits = usize::from(reader.read_u8()?);
                let huffman_num_deviation_bits = usize::from(reader.read_u8()?);
                let row_count = usize::try_from(reader.read_u32()?).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "Huffman row count does not fit into usize".to_string(),
                    }
                })?;

                if huffman_num_id_bits != num_id_bits {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "Huffman l_id mismatch: header={}, expected={}",
                            huffman_num_id_bits, num_id_bits
                        ),
                    });
                }
                if huffman_num_deviation_bits != num_deviation_bits {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "Huffman l_d mismatch: header={}, expected={}",
                            huffman_num_deviation_bits, num_deviation_bits
                        ),
                    });
                }

                let symbol_width = huffman_num_id_bits + huffman_num_deviation_bits;
                let mut canonical_symbols = Vec::with_capacity(symbol_count);
                let mut canonical_code_lengths = Vec::with_capacity(symbol_count);
                for _ in 0..symbol_count {
                    canonical_symbols.push(reader.read_u64_bits(symbol_width)?);
                    canonical_code_lengths
                        .push(reader.read_usize_bits(HUFFMAN_CODE_LENGTH_BITS)? as u8 + 1);
                }

                let mut row_offsets = Vec::with_capacity(row_count);
                for _ in 0..row_count {
                    row_offsets.push(reader.read_u32()?);
                }

                let row_width = if row_count == 0 {
                    0
                } else {
                    if !n.is_multiple_of(row_count) {
                        return Err(EntroGdError::InvalidMetadata {
                            message: format!(
                                "original sample count {} is not divisible by Huffman row count {}",
                                n, row_count
                            ),
                        });
                    }
                    n / row_count
                };

                EncodedData::Huffman(HuffmanDeviationData::new(
                    reader.read_bits(reader.remaining_bits())?,
                    canonical_symbols,
                    canonical_code_lengths,
                    row_offsets,
                    num_samples,
                    n,
                    huffman_num_deviation_bits,
                    huffman_num_id_bits,
                    row_width,
                )?)
            }
            ENCODING_TAG_HUFFMAN_BASE_ID_ONLY => {
                let symbol_count = usize::try_from(reader.read_u32()?).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "Huffman symbol count does not fit into usize".to_string(),
                    }
                })?;
                let huffman_num_id_bits = usize::from(reader.read_u8()?);
                let huffman_num_deviation_bits = usize::from(reader.read_u8()?);
                let row_count = usize::try_from(reader.read_u32()?).map_err(|_| {
                    EntroGdError::InvalidMetadata {
                        message: "Huffman row count does not fit into usize".to_string(),
                    }
                })?;

                if huffman_num_id_bits != num_id_bits {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "Huffman l_id mismatch: header={}, expected={}",
                            huffman_num_id_bits, num_id_bits
                        ),
                    });
                }
                if huffman_num_deviation_bits != num_deviation_bits {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "Huffman l_d mismatch: header={}, expected={}",
                            huffman_num_deviation_bits, num_deviation_bits
                        ),
                    });
                }

                let symbol_width = huffman_num_id_bits;
                let mut canonical_symbols = Vec::with_capacity(symbol_count);
                let mut canonical_code_lengths = Vec::with_capacity(symbol_count);
                for _ in 0..symbol_count {
                    canonical_symbols.push(reader.read_u64_bits(symbol_width)?);
                    canonical_code_lengths
                        .push(reader.read_usize_bits(HUFFMAN_CODE_LENGTH_BITS)? as u8 + 1);
                }

                let mut row_offsets = Vec::with_capacity(row_count);
                for _ in 0..row_count {
                    row_offsets.push(reader.read_u32()?);
                }

                let row_width = if row_count == 0 {
                    0
                } else {
                    if !n.is_multiple_of(row_count) {
                        return Err(EntroGdError::InvalidMetadata {
                            message: format!(
                                "original sample count {} is not divisible by Huffman row count {}",
                                n, row_count
                            ),
                        });
                    }
                    n / row_count
                };

                let raw_deviation_len = num_samples
                    .checked_mul(huffman_num_deviation_bits)
                    .ok_or_else(|| EntroGdError::InvalidMetadata {
                        message: "Huffman raw deviation stream expected length overflow"
                            .to_string(),
                    })?;
                let raw_deviation_stream = reader.read_bits(raw_deviation_len)?;

                EncodedData::Huffman(HuffmanDeviationData::new_base_id_only(
                    reader.read_bits(reader.remaining_bits())?,
                    raw_deviation_stream,
                    canonical_symbols,
                    canonical_code_lengths,
                    row_offsets,
                    num_samples,
                    n,
                    huffman_num_deviation_bits,
                    huffman_num_id_bits,
                    row_width,
                )?)
            }
            other => {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("unsupported encoded-data tag {}", other),
                });
            }
        };

        // Any remaining bits must be zero-padding in the final byte.
        if encoding_tag != ENCODING_TAG_HUFFMAN_CANONICAL
            && encoding_tag != ENCODING_TAG_HUFFMAN_BASE_ID_ONLY
        {
            while reader.remaining_bits() > 0 {
                if reader.read_bit()? {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "non-zero trailing bits after encoded stream".to_string(),
                    });
                }
            }
        }

        // Reconstruct base frequencies from encoded IDs.
        if num_bases > 0 {
            let mut counts = vec![0usize; num_bases];
            for sample_idx in 0..num_samples {
                let sample = encoded_data.get_sample(sample_idx).ok_or_else(|| {
                    EntroGdError::InvalidMetadata {
                        message: format!(
                            "failed to decode sample {} from encoded stream",
                            sample_idx
                        ),
                    }
                })?;

                let mut base_id = 0usize;
                for bit in sample.id.iter() {
                    base_id = (base_id << 1) | (*bit as usize);
                }

                if base_id >= num_bases {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "encoded base id {} out of range for {} bases",
                            base_id, num_bases
                        ),
                    });
                }
                counts[base_id] += 1;
            }
            for (idx, count) in counts.into_iter().enumerate() {
                base_table.as_raw_mut()[idx].1 = count;
            }
        }

        let original_size_bits =
            n.checked_mul(chunk_size)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "original size overflow".to_string(),
                })?;
        let mut metadata = BitDataInfo::new(features, original_size_bits)?;
        metadata.set_condensed_sample_weights(if m == 0 { None } else { Some(weights.clone()) });

        Ok(CompressedData {
            encoded_data,
            condensed_sample_weights: if m == 0 { None } else { Some(weights) },
            base_table,
            layout: crate::compression::base_table::BaseLayoutInfo {
                selected_base_bit_positions: base_bit_positions,
                variable_base_bit_positions,
                constant_zero_bit_positions,
                constant_one_bit_positions,
            },
            entropy_sorted_column_order,
            metadata,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Save to disk. If extension is not `.egd`, it is replaced with `.egd`.
    pub fn save<P: AsRef<Path>>(&self, output_path: P) -> Result<PathBuf, EntroGdError> {
        let target = ensure_egd_extension(output_path.as_ref());
        fs::write(&target, &self.bytes)?;
        Ok(target)
    }
}

pub struct SaveEgdFile {
    pub output_path: PathBuf,
}

impl Filter for SaveEgdFile {
    type Input = CompressedData;
    type Output = PathBuf;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let egd_file = EgdFile::from_compressed_data(&input)?;
        egd_file.save(&self.output_path)
    }
}

/// Convenience API requested by caller:
/// input: `CompressedData`, effect: save an `.egd` file, output: `Result`.
pub fn save_compressed_as_egd<P: AsRef<Path>>(
    compressed: &CompressedData,
    output_path: P,
) -> Result<PathBuf, EntroGdError> {
    EgdFile::from_compressed_data(compressed)?.save(output_path)
}

pub struct LoadEgdFile {}

impl Filter for LoadEgdFile {
    type Input = PathBuf;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        EgdFile::load(input)?.to_compressed_data()
    }
}

/// Load an `.egd` file from disk and parse it into `CompressedData`.
pub fn load_compressed_from_egd<P: AsRef<Path>>(
    input_path: P,
) -> Result<CompressedData, EntroGdError> {
    EgdFile::load(input_path)?.to_compressed_data()
}

/// Load an `.egd` file and fully decompress its payload into bit data.
pub fn load_and_decompress_egd<P: AsRef<Path>>(input_path: P) -> Result<BitDataSet, EntroGdError> {
    let compressed = load_compressed_from_egd(input_path)?;
    decompress_file(&compressed)
}

/// Load an `.egd` file, decompress it, and write a CSV file.
pub fn decompress_egd_to_csv<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
    headers: Option<&[String]>,
) -> Result<PathBuf, EntroGdError> {
    let bit_data = load_and_decompress_egd(input_path)?;
    let target = ensure_csv_extension(output_path.as_ref());
    write_bitdata_as_csv(&bit_data, &target, headers)?;
    Ok(target)
}
