use super::types::DeviationSampleRef;
use super::{CompressedData, DeviationData, DeviationSample, EncodedData};
use crate::compression::preprocessor::{BitDataInfo, BitDataReconstructionInfo};
use crate::error::EntroGdError;
use bitvec::prelude::*;

pub(crate) fn huffman_row_layout(
    metadata: &BitDataInfo,
) -> Result<(usize, usize, usize), EntroGdError> {
    let chunk_size = metadata.chunk_size();
    if chunk_size == 0 {
        return Err(EntroGdError::InvalidMetadata {
            message: "chunk_size is 0".to_string(),
        });
    }

    let original_num_samples = metadata.original_size_bits() / chunk_size;
    match metadata.reconstruction {
        BitDataReconstructionInfo::Image(info) => {
            let row_count = info.height as usize;
            let pixel_grouping = info.pixel_grouping as usize;
            if pixel_grouping == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "image pixel_grouping must be > 0".to_string(),
                });
            }
            let row_width = (info.width as usize).div_ceil(pixel_grouping);
            if row_count == 0 && original_num_samples == 0 {
                return Ok((0, 0, 0));
            }
            if row_count == 0 || row_width == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "invalid image row layout width={} height={} pixel_grouping={}",
                        info.width, info.height, info.pixel_grouping
                    ),
                });
            }
            let expected_samples =
                row_count
                    .checked_mul(row_width)
                    .ok_or_else(|| EntroGdError::InvalidMetadata {
                        message: "image row layout overflows sample count".to_string(),
                    })?;
            if expected_samples != original_num_samples {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "image row layout mismatch: height * ceil(width / pixel_grouping) = {}, original samples = {}",
                        expected_samples, original_num_samples
                    ),
                });
            }
            Ok((original_num_samples, row_count, row_width))
        }
        BitDataReconstructionInfo::Tabular => Ok((original_num_samples, original_num_samples, 1)),
    }
}

impl CompressedData {
    pub fn new(encoded_data: EncodedData, metadata: BitDataInfo) -> Self {
        CompressedData {
            encoded_data,
            condensed_sample_weights: None,
            base_table: Vec::new(),
            base_bit_positions: Vec::new(),
            variable_base_bit_positions: Vec::new(),
            constant_zero_bit_positions: Vec::new(),
            constant_one_bit_positions: Vec::new(),
            metadata,
        }
    }
}

pub(crate) fn build_base_bit_mask(
    chunk_size: usize,
    base_bit_positions: &[usize],
) -> crate::BitStream {
    let mut mask = bitvec![usize, crate::BitOrder; 0; chunk_size];
    for &bit_pos in base_bit_positions {
        if bit_pos < chunk_size {
            mask.set(bit_pos, true);
        }
    }
    mask
}

impl DeviationData {
    pub fn new(
        encoded_bit_stream: crate::BitStream,
        num_samples: usize,
        num_deviation_bits: usize,
        num_id_bits: usize,
    ) -> Self {
        DeviationData {
            encoded_bit_stream,
            num_samples,
            num_deviation_bits,
            num_id_bits,
        }
    }

    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        if sample_idx >= self.num_samples {
            return None;
        }
        let start_bit = sample_idx * (self.num_deviation_bits + self.num_id_bits);
        let end_bit = start_bit + self.num_deviation_bits + self.num_id_bits;
        debug_assert!(end_bit <= self.encoded_bit_stream.len());
        Some(DeviationSample {
            deviation: unsafe {
                self.encoded_bit_stream
                    .get_unchecked(start_bit..start_bit + self.num_deviation_bits)
            }
            .to_bitvec(),
            id: unsafe {
                self.encoded_bit_stream
                    .get_unchecked(start_bit + self.num_deviation_bits..end_bit)
            }
            .to_bitvec(),
        })
    }

    pub fn get_encoded_size(&self) -> usize {
        self.encoded_bit_stream.len()
    }

    pub fn encoded_bit_stream(&self) -> &crate::BitStream {
        &self.encoded_bit_stream
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

    pub(crate) fn for_each_sample_n(
        &self,
        limit: usize,
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        let sample_width = self.num_deviation_bits + self.num_id_bits;
        let capped_limit = limit.min(self.num_samples);

        for sample_idx in 0..capped_limit {
            let start_bit = sample_idx * sample_width;
            let end_bit = start_bit + sample_width;
            debug_assert!(end_bit <= self.encoded_bit_stream.len());
            f(DeviationSampleRef {
                deviation: unsafe {
                    self.encoded_bit_stream
                        .get_unchecked(start_bit..start_bit + self.num_deviation_bits)
                },
                id: unsafe {
                    self.encoded_bit_stream
                        .get_unchecked(start_bit + self.num_deviation_bits..end_bit)
                },
            })?;
        }

        Ok(())
    }
}

impl EncodedData {
    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        match self {
            EncodedData::Normal(data) => data.get_sample(sample_idx),
            EncodedData::Rle(data) => data.get_sample(sample_idx),
            EncodedData::Huffman(data) => data.get_sample(sample_idx),
        }
    }

    pub fn get_encoded_size(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_encoded_size(),
            EncodedData::Rle(data) => data.get_encoded_size(),
            EncodedData::Huffman(data) => data.get_encoded_size(),
        }
    }

    pub fn encoded_bit_stream(&self) -> &crate::BitStream {
        match self {
            EncodedData::Normal(data) => data.encoded_bit_stream(),
            EncodedData::Rle(data) => data.symbol_bit_stream(),
            EncodedData::Huffman(data) => data.pixel_bit_stream(),
        }
    }

    pub fn get_num_samples(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_num_samples(),
            EncodedData::Rle(data) => data.get_num_samples(),
            EncodedData::Huffman(data) => data.get_num_samples(),
        }
    }

    pub fn get_num_deviation_bits(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_num_deviation_bits(),
            EncodedData::Rle(data) => data.get_num_deviation_bits(),
            EncodedData::Huffman(data) => data.get_num_deviation_bits(),
        }
    }

    pub fn get_num_id_bits(&self) -> usize {
        match self {
            EncodedData::Normal(data) => data.get_num_id_bits(),
            EncodedData::Rle(data) => data.get_num_id_bits(),
            EncodedData::Huffman(data) => data.get_num_id_bits(),
        }
    }

    pub(crate) fn for_each_sample_n(
        &self,
        limit: usize,
        f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        match self {
            EncodedData::Normal(data) => data.for_each_sample_n(limit, f),
            EncodedData::Rle(data) => data.for_each_sample_n(limit, f),
            EncodedData::Huffman(data) => data.for_each_sample_n(limit, f),
        }
    }

    pub fn to_raw_deviation_data(&self) -> DeviationData {
        match self {
            EncodedData::Normal(data) => data.clone(),
            EncodedData::Rle(data) => data
                .to_deviation_data()
                .expect("RLE encoded data should be valid when converting to raw deviation data"),
            EncodedData::Huffman(data) => data.to_deviation_data().expect(
                "Huffman encoded data should be valid when converting to raw deviation data",
            ),
        }
    }
}
