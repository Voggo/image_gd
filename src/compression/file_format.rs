use bitvec::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

use crate::compression::compress::{CompressedData, DeviationData};
use crate::compression::preprocessor::{
    BitDataInfo, BitDataReconstructionInfo, FeatureSpec, FeatureTransform,
    ImageReconstructionInfo,
};
use crate::data_loader::FeatureDataType;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;

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
    /// 3) feature metadata (per feature):
    ///    tag=1 (u8), data_type (u8), transform_tag (u8), [transform params], bits_per_feature (u16), base_bit_mask bits
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
        let original_num_rows = data_info.original_size_bits() / chunk_size;
        let condensed_weights = compressed
            .condensed_sample_weights
            .as_deref()
            .unwrap_or(&[]);

        let mut base_positions = compressed.base_bit_positions.clone();
        base_positions.sort_unstable();

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
        let num_bases = u64::try_from(compressed.base_table.len()).map_err(|_| {
            EntroGdError::InvalidMetadata {
                message: "num_bases does not fit into u64".to_string(),
            }
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
                if reader.read_bit()? {
                    base_bit_positions.push(running_offset + local_bit);
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
        let weight_bits = bits_needed(n);
        let mut weights = Vec::with_capacity(m);
        for _ in 0..m {
            weights.push(reader.read_usize_bits(weight_bits)?);
        }

        // Align after weights
        reader.align_to_byte();

        // Base table
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

        let mut base_table = Vec::with_capacity(num_bases);
        for _ in 0..num_bases {
            let mut base_bits = bitvec![usize, Msb0; 0; chunk_size];
            for &bit_pos in &base_bit_positions {
                let bit = reader.read_bit()?;
                base_bits.set(bit_pos, bit);
            }
            base_table.push((base_bits, 0usize));
        }

        let num_samples = n
            .checked_add(m)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "n + m overflows usize".to_string(),
            })?;
        let num_id_bits = bits_needed(num_bases);
        let num_deviation_bits = chunk_size.saturating_sub(base_bit_positions.len());

        let bits_per_sample = num_deviation_bits + num_id_bits;
        let expected_encoded_len = num_samples.checked_mul(bits_per_sample).ok_or_else(|| {
            EntroGdError::InvalidMetadata {
                message: "encoded stream expected length overflow".to_string(),
            }
        })?;

        // Encoded data stream (exact logical bit-length).
        let encoded_stream = reader.read_bits(expected_encoded_len)?;

        // Any remaining bits must be zero-padding in the final byte.
        while reader.remaining_bits() > 0 {
            if reader.read_bit()? {
                return Err(EntroGdError::InvalidMetadata {
                    message: "non-zero trailing bits after encoded stream".to_string(),
                });
            }
        }

        if encoded_stream.len() != expected_encoded_len {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "encoded stream length mismatch: expected {}, got {}",
                    expected_encoded_len,
                    encoded_stream.len()
                ),
            });
        }

        // Reconstruct base frequencies from encoded IDs.
        if num_bases > 0 {
            let mut counts = vec![0usize; num_bases];
            for sample_idx in 0..num_samples {
                let id_start = sample_idx * bits_per_sample + num_deviation_bits;
                let id_end = id_start + num_id_bits;
                let mut base_id = 0usize;
                for bit in &encoded_stream[id_start..id_end] {
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
                base_table[idx].1 = count;
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
            encoded_data: DeviationData::new(
                encoded_stream,
                num_samples,
                num_deviation_bits,
                num_id_bits,
            ),
            condensed_sample_weights: if m == 0 { None } else { Some(weights) },
            base_table,
            base_bit_positions,
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

pub const IMAGE_MAGIC_BYTES: [u8; 3] = *b"IGD";
pub const IMAGE_FORMAT_VERSION: u8 = 1;

/// In-memory IGD file contents (image + compressed payload) that can be saved to disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgdFile {
    bytes: Vec<u8>,
}

impl IgdFile {
    pub fn from_compressed_data(compressed: &CompressedData) -> Result<Self, EntroGdError> {
        let image_info = match compressed.metadata.reconstruction {
            BitDataReconstructionInfo::Image(info) => info,
            _ => {
                return Err(EntroGdError::InvalidMetadata {
                    message: "cannot build IGD from non-image reconstruction metadata".to_string(),
                });
            }
        };

        let egd_payload = EgdFile::from_compressed_data(compressed)?;
        let payload = egd_payload.as_bytes();
        let payload_len = u64::try_from(payload.len()).map_err(|_| EntroGdError::InvalidMetadata {
            message: "IGD payload length does not fit into u64".to_string(),
        })?;

        let mut bytes = Vec::with_capacity(22 + payload.len());
        bytes.extend_from_slice(&IMAGE_MAGIC_BYTES);
        bytes.push(IMAGE_FORMAT_VERSION);
        bytes.extend_from_slice(&image_info.width.to_be_bytes());
        bytes.extend_from_slice(&image_info.height.to_be_bytes());
        bytes.push(image_info.channels);
        bytes.push(image_info.colorspace);
        bytes.extend_from_slice(&payload_len.to_be_bytes());
        bytes.extend_from_slice(payload);

        Ok(IgdFile { bytes })
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        IgdFile { bytes }
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, EntroGdError> {
        let bytes = fs::read(path)?;
        Ok(IgdFile { bytes })
    }

    pub fn to_compressed_data(&self) -> Result<CompressedData, EntroGdError> {
        const HEADER_LEN: usize = 22;
        if self.bytes.len() < HEADER_LEN {
            return Err(EntroGdError::InvalidMetadata {
                message: "IGD file too short".to_string(),
            });
        }

        if self.bytes[0..3] != IMAGE_MAGIC_BYTES {
            return Err(EntroGdError::InvalidMetadata {
                message: "invalid magic bytes (expected IGD)".to_string(),
            });
        }
        if self.bytes[3] != IMAGE_FORMAT_VERSION {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported IGD format version {} (expected {})",
                    self.bytes[3], IMAGE_FORMAT_VERSION
                ),
            });
        }

        let width = u32::from_be_bytes([self.bytes[4], self.bytes[5], self.bytes[6], self.bytes[7]]);
        let height =
            u32::from_be_bytes([self.bytes[8], self.bytes[9], self.bytes[10], self.bytes[11]]);
        let channels = self.bytes[12];
        let colorspace = self.bytes[13];
        if !matches!(channels, 3 | 4) {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("unsupported channel count {} in IGD metadata", channels),
            });
        }

        let payload_len = u64::from_be_bytes([
            self.bytes[14],
            self.bytes[15],
            self.bytes[16],
            self.bytes[17],
            self.bytes[18],
            self.bytes[19],
            self.bytes[20],
            self.bytes[21],
        ]) as usize;

        let payload_start = HEADER_LEN;
        let payload_end = payload_start
            .checked_add(payload_len)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "IGD payload length overflow".to_string(),
            })?;
        if payload_end != self.bytes.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: "IGD payload length does not match file size".to_string(),
            });
        }

        let payload = self.bytes[payload_start..payload_end].to_vec();
        let mut compressed = EgdFile::from_bytes(payload).to_compressed_data()?;
        if compressed.metadata.num_features() != channels as usize {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "IGD channels {} do not match compressed feature count {}",
                    channels,
                    compressed.metadata.num_features()
                ),
            });
        }
        compressed.metadata.reconstruction = BitDataReconstructionInfo::Image(ImageReconstructionInfo {
            width,
            height,
            channels,
            colorspace,
        });

        Ok(compressed)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn save<P: AsRef<Path>>(&self, output_path: P) -> Result<PathBuf, EntroGdError> {
        let target = ensure_igd_extension(output_path.as_ref());
        fs::write(&target, &self.bytes)?;
        Ok(target)
    }
}

pub struct SaveIgdFile {
    pub output_path: PathBuf,
}

impl Filter for SaveIgdFile {
    type Input = CompressedData;
    type Output = PathBuf;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let igd_file = IgdFile::from_compressed_data(&input)?;
        igd_file.save(&self.output_path)
    }
}

pub struct LoadIgdFile {}

impl Filter for LoadIgdFile {
    type Input = PathBuf;
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        IgdFile::load(input)?.to_compressed_data()
    }
}

pub fn save_compressed_as_igd<P: AsRef<Path>>(
    compressed: &CompressedData,
    output_path: P,
) -> Result<PathBuf, EntroGdError> {
    IgdFile::from_compressed_data(compressed)?.save(output_path)
}

pub fn load_compressed_from_igd<P: AsRef<Path>>(
    input_path: P,
) -> Result<CompressedData, EntroGdError> {
    IgdFile::load(input_path)?.to_compressed_data()
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

fn ensure_igd_extension(path: &Path) -> PathBuf {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("igd") => path.to_path_buf(),
        _ => {
            let mut out = path.to_path_buf();
            out.set_extension("igd");
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

fn encode_data_type(data_type: FeatureDataType) -> u8 {
    match data_type {
        FeatureDataType::SignedInt => 0,
        FeatureDataType::UnsignedInt => 1,
        FeatureDataType::F32 => 2,
        FeatureDataType::F64 => 3,
    }
}

fn decode_data_type(tag: u8) -> Result<FeatureDataType, EntroGdError> {
    match tag {
        0 => Ok(FeatureDataType::SignedInt),
        1 => Ok(FeatureDataType::UnsignedInt),
        2 => Ok(FeatureDataType::F32),
        3 => Ok(FeatureDataType::F64),
        _ => Err(EntroGdError::InvalidMetadata {
            message: format!("unsupported feature data type tag {}", tag),
        }),
    }
}

#[derive(Debug, Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bit_len: usize,
}

#[derive(Debug)]
struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    fn read_bit(&mut self) -> Result<bool, EntroGdError> {
        if self.bit_pos >= self.bytes.len() * 8 {
            return Err(EntroGdError::InvalidMetadata {
                message: "unexpected end of EGD bitstream".to_string(),
            });
        }
        let byte_index = self.bit_pos / 8;
        let bit_in_byte = self.bit_pos % 8;
        let bit = ((self.bytes[byte_index] >> (7 - bit_in_byte)) & 1) == 1;
        self.bit_pos += 1;
        Ok(bit)
    }

    fn read_u8(&mut self) -> Result<u8, EntroGdError> {
        let mut value = 0u8;
        for _ in 0..8 {
            value = (value << 1) | (self.read_bit()? as u8);
        }
        Ok(value)
    }

    fn read_u16(&mut self) -> Result<u16, EntroGdError> {
        let mut value = 0u16;
        for _ in 0..16 {
            value = (value << 1) | (self.read_bit()? as u16);
        }
        Ok(value)
    }

    fn read_u64(&mut self) -> Result<u64, EntroGdError> {
        let mut value = 0u64;
        for _ in 0..64 {
            value = (value << 1) | (self.read_bit()? as u64);
        }
        Ok(value)
    }

    fn read_usize_bits(&mut self, width: usize) -> Result<usize, EntroGdError> {
        if width > usize::BITS as usize {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("cannot read {} bits into usize", width),
            });
        }
        let mut value = 0usize;
        for _ in 0..width {
            value = (value << 1) | (self.read_bit()? as usize);
        }
        Ok(value)
    }

    fn align_to_byte(&mut self) {
        while !self.bit_pos.is_multiple_of(8) {
            self.bit_pos += 1;
        }
    }

    fn remaining_bits(&self) -> usize {
        self.bytes.len() * 8 - self.bit_pos
    }

    fn read_bits(&mut self, len: usize) -> Result<BitVec<usize, Msb0>, EntroGdError> {
        if len > self.remaining_bits() {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "requested {} bits, but only {} remain",
                    len,
                    self.remaining_bits()
                ),
            });
        }
        let mut out = BitVec::<usize, Msb0>::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_bit()?);
        }
        Ok(out)
    }
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
    use crate::compression::compress::build_compression_pipeline;
    use crate::compression::preprocessor::{
        BitData, BitDataInfo, BitDataReconstructionInfo, BitDataSet, FeatureSpec,
        ImageReconstructionInfo,
    };
    use crate::data_loader::FeatureDataType;
    use crate::filter_pipeline::Filter;

    fn get_compression_pipeline() -> impl Filter<Input = BitDataSet, Output = CompressedData> {
        build_compression_pipeline(100, 5)
    }

    #[test]
    fn test_build_and_save_egd() {
        let data = BitData {
            data: bitvec![usize, Msb0; 0; 64],
            num_rows: 1,
            chunk_size: 64,
        };
        let features = vec![FeatureSpec::new(FeatureDataType::UnsignedInt, 8); 8];
        let info = BitDataInfo::new(features, 64).unwrap();
        let bit_data = BitDataSet { data, info };
        let pipeline = get_compression_pipeline();
        let compressed = pipeline.process(bit_data).unwrap();
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

    #[test]
    fn test_roundtrip_egd_to_compressed_data() {
        let data = BitData {
            data: bitvec![usize, Msb0; 0; 320],
            num_rows: 5,
            chunk_size: 64,
        };
        let features = vec![
            FeatureSpec::new(FeatureDataType::UnsignedInt, 32),
            FeatureSpec::new(FeatureDataType::UnsignedInt, 32),
        ];
        let info = BitDataInfo::new(features, 320).unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_compression_pipeline().process(bit_data).unwrap();

        let egd = EgdFile::from_compressed_data(&compressed).unwrap();
        let loaded = egd.to_compressed_data().unwrap();

        assert_eq!(loaded.metadata, compressed.metadata);
        assert_eq!(loaded.base_bit_positions, compressed.base_bit_positions);
        assert_eq!(
            loaded.condensed_sample_weights,
            compressed.condensed_sample_weights
        );
        assert_eq!(
            loaded.encoded_data.encoded_bit_stream(),
            compressed.encoded_data.encoded_bit_stream()
        );
        assert_eq!(
            loaded.encoded_data.get_num_samples(),
            compressed.encoded_data.get_num_samples()
        );
        assert_eq!(
            loaded.encoded_data.get_num_deviation_bits(),
            compressed.encoded_data.get_num_deviation_bits()
        );
        assert_eq!(
            loaded.encoded_data.get_num_id_bits(),
            compressed.encoded_data.get_num_id_bits()
        );
        assert_eq!(loaded.base_table.len(), compressed.base_table.len());
        for (lhs, rhs) in loaded.base_table.iter().zip(compressed.base_table.iter()) {
            assert_eq!(lhs.0, rhs.0);
            assert_eq!(lhs.1, rhs.1);
        }
    }

    #[test]
    fn test_roundtrip_igd_to_compressed_data_with_image_metadata() {
        let data = BitData {
            data: bitvec![usize, Msb0; 0; 96],
            num_rows: 4,
            chunk_size: 24,
        };
        let features = vec![FeatureSpec::new(FeatureDataType::UnsignedInt, 8); 3];
        let info = BitDataInfo::new_with_reconstruction_info(
            features,
            96,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 2,
                channels: 3,
                colorspace: 0,
            }),
        )
        .unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_compression_pipeline().process(bit_data).unwrap();

        let igd = IgdFile::from_compressed_data(&compressed).unwrap();
        let loaded = igd.to_compressed_data().unwrap();

        assert_eq!(loaded.metadata.num_features(), 3);
        assert_eq!(loaded.metadata.original_size_bits(), 96);
        assert!(matches!(
            loaded.metadata.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 2,
                channels: 3,
                colorspace: 0
            })
        ));
    }
}
