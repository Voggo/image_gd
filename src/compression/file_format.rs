use bitvec::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

use crate::compression::compress::CompressedData;
use crate::error::EntroGdError;

pub const MAGIC_BYTES: [u8; 3] = *b"EGD";
pub const FORMAT_VERSION: u8 = 1;

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
    /// 3) feature metadata (per feature): tag=1 (u8), bits_per_feature (u16), base_bit_mask bits
    /// 4) zero-padding to next byte
    /// 5) condensed weights bitstream (m * ceil(log2(n)) bits)
    /// 6) zero-padding to next byte
    /// 7) base table: num_bases (u64) then packed base bits for each base
    /// 8) encoded data stream bits
    pub fn from_compressed_data(compressed: &CompressedData) -> Result<Self, EntroGdError> {
        let data_info = &compressed.metadata;
        let num_features = data_info.num_features();
        if num_features == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: "num_features is 0".to_string(),
            });
        }

        let chunk_size = data_info.chunk_size();
        let original_num_rows = data_info.original_size_bits / chunk_size;
        let condensed_weights = compressed.condensed_sample_weights.as_deref().unwrap_or(&[]);

        let mut base_positions = compressed.base_bit_positions.clone();
        base_positions.sort_unstable();

        let n_u64 = u64::try_from(original_num_rows).map_err(|_| EntroGdError::InvalidMetadata {
            message: "n does not fit into u64".to_string(),
        })?;
        let m_u64 = u64::try_from(condensed_weights.len()).map_err(|_| EntroGdError::InvalidMetadata {
            message: "m does not fit into u64".to_string(),
        })?;
        let num_features_u64 = u64::try_from(num_features).map_err(|_| EntroGdError::InvalidMetadata {
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
            let feature_bits = data_info.feature_bits(feature_idx);
            let bits_per_feature = u16::try_from(feature_bits).map_err(|_| EntroGdError::InvalidMetadata {
                message: format!("feature {} bits {} does not fit into u16", feature_idx, feature_bits),
            })?;

            writer.write_u8(1); // BitInfo tag
            writer.write_u16(bits_per_feature);

            let offset = data_info.feature_offset(feature_idx);
            for local_bit in 0..feature_bits {
                let global_bit = offset + local_bit;
                writer.write_bit(base_positions.binary_search(&global_bit).is_ok());
            }
        }

        // Align after feature metadata
        writer.align_to_byte();

        // Condensed sample weights
        let weight_bits = bits_needed(original_num_rows);
        for &weight in condensed_weights {
            writer.write_usize_bits(weight, weight_bits);
        }

        // Align after weights
        writer.align_to_byte();

        // Base table
        let num_bases = u64::try_from(compressed.base_table.len()).map_err(|_| EntroGdError::InvalidMetadata {
            message: "num_bases does not fit into u64".to_string(),
        })?;
        writer.write_u64(num_bases);
        for (base_bits, _) in &compressed.base_table {
            for &bit_pos in &base_positions {
                writer.write_bit(base_bits.get(bit_pos).map(|b| *b).unwrap_or(false));
            }
        }

        // Data stream
        writer.write_bitslice(compressed.encoded_data.encoded_bit_stream());

        Ok(EgdFile {
            bytes: writer.into_bytes(),
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

/// Convenience API requested by caller:
/// input: `CompressedData`, effect: save an `.egd` file, output: `Result`.
pub fn save_compressed_as_egd<P: AsRef<Path>>(
    compressed: &CompressedData,
    output_path: P,
) -> Result<PathBuf, EntroGdError> {
    EgdFile::from_compressed_data(compressed)?.save(output_path)
}

fn ensure_egd_extension(path: &Path) -> PathBuf {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("egd") => path.to_path_buf(),
        _ => {
            let mut out = path.to_path_buf();
            out.set_extension("egd");
            out
        }
    }
}

fn bits_needed(value: usize) -> usize {
    if value <= 1 {
        1
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

#[derive(Debug, Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bit_len: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self::default()
    }

    fn write_bit(&mut self, bit: bool) {
        let byte_index = self.bit_len / 8;
        let bit_in_byte = self.bit_len % 8;
        if byte_index == self.bytes.len() {
            self.bytes.push(0);
        }
        if bit {
            self.bytes[byte_index] |= 1 << (7 - bit_in_byte);
        }
        self.bit_len += 1;
    }

    fn write_bitslice(&mut self, bits: &BitSlice<usize, Msb0>) {
        for bit in bits {
            self.write_bit(*bit);
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.write_usize_bits(value as usize, 8);
    }

    fn write_u16(&mut self, value: u16) {
        self.write_usize_bits(value as usize, 16);
    }

    fn write_u64(&mut self, value: u64) {
        for shift in (0..64).rev() {
            self.write_bit(((value >> shift) & 1) == 1);
        }
    }

    fn write_usize_bits(&mut self, value: usize, width: usize) {
        for shift in (0..width).rev() {
            self.write_bit(((value >> shift) & 1) == 1);
        }
    }

    fn align_to_byte(&mut self) {
        while !self.bit_len.is_multiple_of(8) {
            self.write_bit(false);
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::compress::compress;
    use crate::data_loader::FeatureDataType;
    use crate::preprocessor::{BitData, BitDataInfo, BitDataSet, FeatureSpec};

    #[test]
    fn test_build_and_save_egd() {
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
        let egd = EgdFile::from_compressed_data(&compressed).unwrap();

        assert!(egd.as_bytes().len() >= 4);
        assert_eq!(&egd.as_bytes()[0..3], &MAGIC_BYTES);
        assert_eq!(egd.as_bytes()[3], FORMAT_VERSION);

        let output = std::env::temp_dir().join("entro_gd_test_output");
        let saved = egd.save(&output).unwrap();
        assert_eq!(saved.extension().and_then(|s| s.to_str()), Some("egd"));
        assert!(saved.exists());
        let _ = std::fs::remove_file(saved);
    }
}
