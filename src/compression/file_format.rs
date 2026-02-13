use bitvec::prelude::*;

use crate::compression::compress::CompressedData;
use crate::preprocessor::BitDataInfo;
use crate::error::EntroGdError;

pub const MAGIC_BYTES: [u8; 3] = *b"EGD";
pub const FORMAT_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileHeader {
    pub magic: [u8; 3],
    pub version: u8,
}

impl FileHeader {
    pub fn new() -> Self {
        FileHeader {
            magic: MAGIC_BYTES,
            version: FORMAT_VERSION,
        }
    }

    pub fn size_bits(&self) -> usize {
        32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalInfo {
    pub n: u64,
    pub m: u64,
    pub num_features: u64,
}

impl GlobalInfo {
    pub fn size_bits(&self) -> usize {
        24 * 8
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureMetadataEntry {
    /// Stores bits per feature and a base-bit mask for the feature.
    /// Layout: tag (8 bits) + bits_per_feature (16 bits) + base_bit_mask bits
    BitInfo {
        bits_per_feature: u16,
        base_bit_mask: BitVec<usize, Msb0>,
    },
    /// Custom tag + raw bytes
    Custom {
        tag: u8,
        data: Vec<u8>,
    },
}

impl FeatureMetadataEntry {
    pub fn tag(&self) -> u8 {
        match self {
            FeatureMetadataEntry::BitInfo { .. } => 1,
            FeatureMetadataEntry::Custom { tag, .. } => *tag,
        }
    }

    pub fn size_bits(&self) -> usize {
        match self {
            FeatureMetadataEntry::BitInfo {
                base_bit_mask, ..
            } => 8 + 16 + base_bit_mask.len(),
            FeatureMetadataEntry::Custom { data, .. } => 8 + (data.len() * 8),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseTable {
    pub num_bases: u64,
    pub num_base_bits: usize,
    pub bases: Vec<BitVec<usize, Msb0>>,
}

impl BaseTable {
    pub fn size_bits(&self) -> usize {
        let base_bits = self.num_base_bits * self.bases.len();
        // Prefix with num_bases (u64)
        (8 * 8) + base_bits
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedFile {
    pub header: FileHeader,
    pub global: GlobalInfo,
    /// One entry per feature
    pub feature_metadata: Vec<FeatureMetadataEntry>,
    /// Weights packed into a bitstream (length m * ceil(log2 n))
    pub weights: BitVec<usize, Msb0>,
    pub base_table: BaseTable,
    /// Remainder of file (encoded data stream)
    pub data_stream: BitVec<usize, Msb0>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileLayoutSizes {
    pub magic_version_bits: usize,
    pub global_info_bits: usize,
    pub feature_metadata_bits: usize,
    pub padding_after_features_bits: usize,
    pub weights_bits: usize,
    pub padding_after_weights_bits: usize,
    pub base_table_bits: usize,
    pub data_stream_bits: usize,
    pub total_bits: usize,
}

impl CompressedFile {
    pub fn from_compressed_data(compressed: &CompressedData) -> Result<Self, EntroGdError> {
        let data_info = &compressed.metadata.data_info;
        let num_features = data_info.num_features();
        if num_features == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "num_features is 0".to_string(),
            });
        }
        let original_num_rows = data_info.original_size_bits / data_info.chunk_size();

        let condensed_weights = compressed
            .condensed_sample_weights
            .as_deref()
            .unwrap_or(&[]);
        let weights = build_weights_bitstream(condensed_weights, original_num_rows);
        let m = condensed_weights.len() as u64;

        let mut base_positions = compressed.base_bit_positions.clone();
        base_positions.sort_unstable();

        let feature_metadata = build_feature_metadata(data_info, &base_positions);
        let base_table = build_base_table(&compressed.base_table, &base_positions);

        Ok(CompressedFile {
            header: FileHeader::new(),
            global: GlobalInfo {
                n: original_num_rows as u64,
                m,
                num_features: num_features as u64,
            },
            feature_metadata,
            weights,
            base_table,
            data_stream: compressed.encoded_data.encoded_bit_stream().clone(),
        })
    }

    pub fn layout_sizes_bits(&self) -> FileLayoutSizes {
        let magic_version_bits = self.header.size_bits();
        let global_info_bits = self.global.size_bits();
        let feature_metadata_bits = self
            .feature_metadata
            .iter()
            .map(|entry| entry.size_bits())
            .sum();
        let padding_after_features_bits = pad_to_byte(magic_version_bits
            + global_info_bits
            + feature_metadata_bits);
        let weights_bits = self.weights.len();
        let padding_after_weights_bits = pad_to_byte(
            magic_version_bits
                + global_info_bits
                + feature_metadata_bits
                + padding_after_features_bits
                + weights_bits,
        );
        let base_table_bits = self.base_table.size_bits();
        let data_stream_bits = self.data_stream.len();
        let total_bits = magic_version_bits
            + global_info_bits
            + feature_metadata_bits
            + padding_after_features_bits
            + weights_bits
            + padding_after_weights_bits
            + base_table_bits
            + data_stream_bits;

        FileLayoutSizes {
            magic_version_bits,
            global_info_bits,
            feature_metadata_bits,
            padding_after_features_bits,
            weights_bits,
            padding_after_weights_bits,
            base_table_bits,
            data_stream_bits,
            total_bits,
        }
    }
}

fn build_feature_metadata(
    data_info: &BitDataInfo,
    base_bit_positions: &[usize],
) -> Vec<FeatureMetadataEntry> {
    let num_features = data_info.num_features();
    let mut feature_masks = (0..num_features)
        .map(|idx| bitvec![usize, Msb0; 0; data_info.feature_bits(idx)])
        .collect::<Vec<_>>();

    for &bit_pos in base_bit_positions {
        if let Some(feature_idx) = data_info.feature_index_for_bit(bit_pos) {
            let bit_in_feature = bit_pos - data_info.feature_offset(feature_idx);
            if feature_idx < num_features {
                feature_masks[feature_idx].set(bit_in_feature, true);
            }
        }
    }

    feature_masks
        .into_iter()
        .enumerate()
        .map(|(idx, mask)| FeatureMetadataEntry::BitInfo {
            bits_per_feature: data_info.feature_bits(idx) as u16,
            base_bit_mask: mask,
        })
        .collect()
}

fn build_base_table(
    base_table: &[(BitVec<usize, Msb0>, usize)],
    base_bit_positions: &[usize],
) -> BaseTable {
    let num_base_bits = base_bit_positions.len();
    let bases = base_table
        .iter()
        .map(|(base_bits, _count)| pack_base_bits(base_bits, base_bit_positions))
        .collect::<Vec<_>>();

    BaseTable {
        num_bases: bases.len() as u64,
        num_base_bits,
        bases,
    }
}

fn pack_base_bits(
    base_bits: &BitVec<usize, Msb0>,
    base_bit_positions: &[usize],
) -> BitVec<usize, Msb0> {
    let mut packed = BitVec::<usize, Msb0>::with_capacity(base_bit_positions.len());
    for &bit_pos in base_bit_positions {
        if bit_pos < base_bits.len() {
            packed.push(base_bits[bit_pos]);
        } else {
            packed.push(false);
        }
    }
    packed
}

fn build_weights_bitstream(weights: &[usize], n: usize) -> BitVec<usize, Msb0> {
    let l_w = bits_needed(n);
    let mut stream = BitVec::<usize, Msb0>::with_capacity(weights.len() * l_w);
    for &weight in weights {
        for shift in (0..l_w).rev() {
            stream.push(((weight >> shift) & 1) == 1);
        }
    }
    stream
}

fn bits_needed(value: usize) -> usize {
    if value <= 1 {
        1
    } else {
        (value as f64).log2().ceil() as usize
    }
}

fn pad_to_byte(bit_count: usize) -> usize {
    let remainder = bit_count % 8;
    if remainder == 0 {
        0
    } else {
        8 - remainder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::compress::compress;
    use crate::data_loader::FeatureDataType;
    use crate::preprocessor::{BitData, BitDataInfo, BitDataSet, FeatureSpec};

    #[test]
    fn test_file_layout_sizes() {
        let data = BitData {
            data: bitvec![usize, Msb0; 0; 64],
            num_rows: 8,
            chunk_size: 8,
        };
        let features = vec![
            FeatureSpec {
                data_type: FeatureDataType::UnsignedInt,
                bits: 1,
            };
            8
        ];
        let info = BitDataInfo::new(features, 64).unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = compress(&bit_data);
        let file = CompressedFile::from_compressed_data(&compressed).unwrap();
        let sizes = file.layout_sizes_bits();

        assert_eq!(sizes.magic_version_bits, 32);
        assert_eq!(sizes.global_info_bits, 192);
        assert!(sizes.total_bits > 0);
    }
}
