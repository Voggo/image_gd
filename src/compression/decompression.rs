use crate::compression::compress::{CompressedData, CondensedSamples, build_base_bit_mask};
use crate::compression::preprocessor::{
    BitData, BitDataReconstructionInfo, BitDataSet, ImageColorModel, ImageGroupingTransform,
    append_row_padding, decode_value_from_bits,
};
use crate::data_loader::DataValue;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use crate::utils::{min_position_bits, signed_half_wrapped};
use bitvec::prelude::*;
use image::{RgbImage, RgbaImage};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

pub struct DecompressRowsData;

impl Filter for DecompressRowsData {
    type Input = (Arc<CompressedData>, Vec<usize>);
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!("Decompressing {} rows", input.1.len()));
        let (compressed, indices) = input;
        decompress_samples_batch(compressed.as_ref(), &indices)
    }
}

pub(crate) fn decompress_samples_batch(
    compressed: &CompressedData,
    indices: &[usize],
) -> Result<BitDataSet, EntroGdError> {
    let data_info = &compressed.metadata;
    let num_features = data_info.num_features();
    let chunk_size = data_info.chunk_size();
    let stride = data_info.row_stride();
    let row_padding_bits = stride.saturating_sub(chunk_size);
    let original_num_rows = data_info.original_size_bits() / chunk_size;
    if num_features == 0 {
        return Err(EntroGdError::InvalidMetadata {
            message: "num_features is 0".to_string(),
        });
    }
    if let Some(&sample_idx) = indices.iter().find(|&&idx| idx >= original_num_rows) {
        return Err(EntroGdError::DecompressionSampleMissing { sample_idx });
    }
    let base_bit_positions = &compressed.base_bit_positions;
    let base_bit_mask = build_base_bit_mask(chunk_size, base_bit_positions);

    let mut reconstructed_bits = crate::BitStream::with_capacity(stride * indices.len());

    if row_padding_bits == 0 {
        for &sample_idx in indices {
            let sample = match compressed.encoded_data.get_sample(sample_idx) {
                Some(s) => s,
                None => {
                    return Err(EntroGdError::DecompressionSampleMissing { sample_idx });
                }
            };

            append_reconstructed_chunk(
                &mut reconstructed_bits,
                &base_bit_mask,
                &compressed.base_table,
                chunk_size,
                sample.deviation.as_bitslice(),
                sample.id.as_bitslice(),
            )?;
        }
    } else {
        for &sample_idx in indices {
            let sample = match compressed.encoded_data.get_sample(sample_idx) {
                Some(s) => s,
                None => {
                    return Err(EntroGdError::DecompressionSampleMissing { sample_idx });
                }
            };

            append_reconstructed_chunk(
                &mut reconstructed_bits,
                &base_bit_mask,
                &compressed.base_table,
                chunk_size,
                sample.deviation.as_bitslice(),
                sample.id.as_bitslice(),
            )?;

            append_row_padding(&mut reconstructed_bits, row_padding_bits);
        }
    }

    let data = BitData {
        data: reconstructed_bits,
        chunk_size,
        stride,
        num_rows: indices.len(),
    };
    let info = data_info.with_original_size_bits_and_row_stride(chunk_size * indices.len(), stride);
    Ok(BitDataSet { data, info })
}

pub struct DecompressFileData {}

impl Filter for DecompressFileData {
    type Input = CompressedData;
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Decompressing entire file data");
        decompress_file(&input)
    }
}

pub fn decompress_file(compressed: &CompressedData) -> Result<BitDataSet, EntroGdError> {
    let data_info = &compressed.metadata;
    let chunk_size = data_info.chunk_size();
    let stride = data_info.row_stride();
    let row_padding_bits = stride.saturating_sub(chunk_size);
    let original_num_rows = data_info.original_size_bits() / chunk_size;
    let base_bit_mask = build_base_bit_mask(chunk_size, &compressed.base_bit_positions);
    let mut reconstructed_bits = crate::BitStream::with_capacity(stride * original_num_rows);
    let mut decoded_rows = 0usize;

    if row_padding_bits == 0 {
        compressed.encoded_data.for_each_sample(|sample| {
            if decoded_rows >= original_num_rows {
                return Ok(());
            }

            append_reconstructed_chunk(
                &mut reconstructed_bits,
                &base_bit_mask,
                &compressed.base_table,
                chunk_size,
                sample.deviation,
                sample.id,
            )?;

            decoded_rows += 1;
            Ok(())
        })?;
    } else {
        compressed.encoded_data.for_each_sample(|sample| {
            if decoded_rows >= original_num_rows {
                return Ok(());
            }

            append_reconstructed_chunk(
                &mut reconstructed_bits,
                &base_bit_mask,
                &compressed.base_table,
                chunk_size,
                sample.deviation,
                sample.id,
            )?;

            append_row_padding(&mut reconstructed_bits, row_padding_bits);

            decoded_rows += 1;
            Ok(())
        })?;
    }

    if decoded_rows != original_num_rows {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "decoded row count mismatch: expected {}, got {}",
                original_num_rows, decoded_rows
            ),
        });
    }

    let data = BitData {
        data: reconstructed_bits,
        chunk_size,
        stride,
        num_rows: original_num_rows,
    };
    let info =
        data_info.with_original_size_bits_and_row_stride(chunk_size * original_num_rows, stride);
    Ok(BitDataSet { data, info })
}

fn append_reconstructed_chunk(
    out: &mut crate::BitStream,
    base_bit_mask: &crate::BitStream,
    base_table: &[(crate::BitStream, usize)],
    chunk_size: usize,
    deviation_bits: &crate::BitView,
    id_bits: &crate::BitView,
) -> Result<(), EntroGdError> {
    let mut chunk = bitvec![usize, crate::BitOrder; 0; chunk_size];
    let base_id = decode_base_id(id_bits);
    if base_id >= base_table.len() {
        return Err(EntroGdError::InvalidBaseId {
            base_id,
            table_len: base_table.len(),
        });
    }

    let base_pattern = &base_table[base_id].0;
    let base_len = chunk_size.min(base_pattern.len());
    for bit_pos in 0..base_len {
        chunk.set(bit_pos, unsafe { *base_pattern.get_unchecked(bit_pos) });
    }

    let mut deviation_bit_idx = 0;
    for bit_pos in 0..chunk_size {
        if !unsafe { *base_bit_mask.get_unchecked(bit_pos) }
            && deviation_bit_idx < deviation_bits.len()
        {
            chunk.set(bit_pos, unsafe {
                *deviation_bits.get_unchecked(deviation_bit_idx)
            });
            deviation_bit_idx += 1;
        }
    }

    out.extend_from_bitslice(&chunk);
    Ok(())
}

fn decode_base_id(id_bits: &crate::BitView) -> usize {
    id_bits
        .iter()
        .fold(0usize, |base_id, bit| (base_id << 1) | usize::from(*bit))
}

pub struct DecompressAnalytics {}

impl Filter for DecompressAnalytics {
    type Input = CompressedData;
    type Output = Option<CondensedSamples>;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Decompressing condensed samples for analytics");
        Ok(decompress_analytics(&input))
    }
}

pub fn decompress_analytics(compressed: &CompressedData) -> Option<CondensedSamples> {
    if let Some(weights) = &compressed.condensed_sample_weights {
        let samples: Vec<crate::BitStream> = compressed
            .base_table
            .iter()
            .map(|(bv, _)| bv.clone())
            .collect();
        Some(CondensedSamples {
            samples,
            weights: weights.clone(),
        })
    } else {
        None
    }
}

/// Write decompressed bit data back to a CSV file.
pub fn write_bitdata_as_csv<P: AsRef<Path>>(
    bit_data: &BitDataSet,
    output_path: P,
    headers: Option<&[String]>,
) -> Result<(), EntroGdError> {
    let mut file = File::create(output_path)?;

    if let Some(headers) = headers {
        writeln!(file, "{}", headers.join(","))?;
    }

    for row in 0..bit_data.data.num_rows {
        let mut values: Vec<String> = Vec::with_capacity(bit_data.info.num_features());
        for feature in 0..bit_data.info.num_features() {
            let feature_bits = unsafe { bit_data.get_feature_unchecked(row, feature) };
            let spec = bit_data.info.feature_spec(feature);
            let formatted = match decode_value_from_bits(feature_bits, spec) {
                DataValue::Unsigned(v) => v.to_string(),
                DataValue::Signed(v) => v.to_string(),
                DataValue::F32(v) => v.to_string(),
                DataValue::F64(v) => v.to_string(),
            };
            values.push(formatted);
        }
        writeln!(file, "{}", values.join(","))?;
    }

    Ok(())
}

/// Write decompressed image bit data back to an image file.
pub fn write_bitdata_as_image<P: AsRef<Path>>(
    bit_data: &BitDataSet,
    output_path: P,
) -> Result<(), EntroGdError> {
    let image_info = match &bit_data.info.reconstruction {
        BitDataReconstructionInfo::Image(info) => *info,
        BitDataReconstructionInfo::Tabular => {
            return Err(EntroGdError::InvalidMetadata {
                message: "cannot write image from tabular reconstruction metadata".to_string(),
            });
        }
    };

    let channels = image_info.channels as usize;
    let pixel_grouping = image_info.pixel_grouping as usize;
    if bit_data.num_features() != channels {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "feature count {} does not match image channel count {}",
                bit_data.num_features(),
                channels
            ),
        });
    }

    let grouped_width = (image_info.width as usize).div_ceil(pixel_grouping);
    let expected_rows = grouped_width * image_info.height as usize;
    if bit_data.num_rows() != expected_rows {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "grouped image row count {} does not match expected {}",
                bit_data.num_rows(),
                expected_rows
            ),
        });
    }

    let mut raw =
        Vec::with_capacity(image_info.width as usize * image_info.height as usize * channels);
    for row in 0..bit_data.num_rows() {
        let group_x = row % grouped_width;
        let mut decoded_channels = Vec::with_capacity(channels);
        for feature in 0..channels {
            let feature_bits = unsafe { bit_data.get_feature_unchecked(row, feature) };
            decoded_channels.push(decode_grouped_feature(
                feature_bits,
                pixel_grouping,
                image_info.grouping_transform,
            )?);
        }

        for offset in 0..pixel_grouping {
            let pixel_x = group_x * pixel_grouping + offset;
            if pixel_x < image_info.width as usize {
                for channel_values in &decoded_channels {
                    raw.push(channel_values[offset]);
                }
            }
        }
    }

    let output_raw = match image_info.color_model {
        ImageColorModel::Rgb => raw,
        ImageColorModel::YCoCg => convert_ycocg_to_rgb_channels(&raw, channels),
        ImageColorModel::YCoCgR => convert_ycocg_r_to_rgb_channels(&raw, channels),
    };

    save_raw_image(
        output_path.as_ref(),
        image_info.width,
        image_info.height,
        image_info.channels,
        output_raw,
    )
}

/// Write decompressed data to a format-appropriate output file.
///
/// - `Tabular` reconstruction metadata writes CSV.
/// - `Image` reconstruction metadata writes an image file.
pub fn write_bitdata_to_output<P: AsRef<Path>>(
    bit_data: &BitDataSet,
    output_path: P,
    headers: Option<&[String]>,
) -> Result<(), EntroGdError> {
    match bit_data.info.reconstruction {
        BitDataReconstructionInfo::Tabular => write_bitdata_as_csv(bit_data, output_path, headers),
        BitDataReconstructionInfo::Image(_) => write_bitdata_as_image(bit_data, output_path),
    }
}

fn byte_from_bits(bits: &crate::BitView) -> Result<u8, EntroGdError> {
    if bits.len() != 8 {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("expected 8 bits for image byte, got {}", bits.len()),
        });
    }

    Ok(bits
        .iter()
        .fold(0u8, |acc, bit| (acc << 1) | u8::from(*bit)))
}

fn bits_to_u16(bits: &crate::BitView) -> Result<u16, EntroGdError> {
    if bits.len() > 16 {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("expected at most 16 bits, got {}", bits.len()),
        });
    }

    Ok(bits
        .iter()
        .fold(0u16, |acc, bit| (acc << 1) | u16::from(*bit)))
}

fn decode_grouped_feature(
    bits: &crate::BitView,
    pixel_grouping: usize,
    grouping_transform: ImageGroupingTransform,
) -> Result<Vec<u8>, EntroGdError> {
    match grouping_transform {
        ImageGroupingTransform::Raw => bits.chunks(8).map(byte_from_bits).collect(),
        ImageGroupingTransform::ForFirstPixel => {
            if pixel_grouping == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "pixel_grouping must be > 0".to_string(),
                });
            }

            let expected_bits = 8 + pixel_grouping.saturating_sub(1) * 9;
            if bits.len() != expected_bits {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "invalid FOR(first pixel) feature width: expected {}, got {}",
                        expected_bits,
                        bits.len()
                    ),
                });
            }

            let anchor = byte_from_bits(&bits[0..8])? as i16;
            let mut values = Vec::with_capacity(pixel_grouping);
            values.push(anchor as u8);

            for offset in 0..pixel_grouping.saturating_sub(1) {
                let start = 8 + offset * 9;
                let end = start + 9;
                let biased = bits_to_u16(&bits[start..end])? as i16;
                let value = anchor + biased - 255;
                if !(0..=255).contains(&value) {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!("decoded grouped image byte out of range: {}", value),
                    });
                }
                values.push(value as u8);
            }

            Ok(values)
        }
        ImageGroupingTransform::ForMin => {
            if pixel_grouping == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "pixel_grouping must be > 0".to_string(),
                });
            }

            let position_bits = min_position_bits(pixel_grouping);
            let expected_bits = 8 + position_bits + pixel_grouping.saturating_sub(1) * 9;
            if bits.len() != expected_bits {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "invalid FOR(min) feature width: expected {}, got {}",
                        expected_bits,
                        bits.len()
                    ),
                });
            }

            let anchor = byte_from_bits(&bits[0..8])? as u16;
            let min_position = if position_bits == 0 {
                0
            } else {
                bits_to_u16(&bits[8..8 + position_bits])? as usize
            };
            if min_position >= pixel_grouping {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("FOR(min) min position {} out of bounds", min_position),
                });
            }

            let mut values = Vec::with_capacity(pixel_grouping);
            let mut residual_cursor = 8 + position_bits;
            for idx in 0..pixel_grouping {
                if idx == min_position {
                    values.push(anchor as u8);
                    continue;
                }

                let biased = bits_to_u16(&bits[residual_cursor..residual_cursor + 9])? as i16;
                let value = anchor as i16 + biased - 255;
                if !(0..=255).contains(&value) {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!("decoded grouped image byte out of range: {}", value),
                    });
                }
                values.push(value as u8);
                residual_cursor += 9;
            }

            Ok(values)
        }
    }
}

fn convert_ycocg_to_rgb_channels(raw: &[u8], channels: usize) -> Vec<u8> {
    let mut out = raw.to_vec();
    for pixel in out.chunks_exact_mut(channels) {
        let y = i16::from(pixel[0]);
        let co = i16::from(pixel[1]) - 128;
        let cg = i16::from(pixel[2]) - 128;

        let r = y + co - cg;
        let g = y + cg;
        let b = y - co - cg;

        pixel[0] = clamp_to_u8(r);
        pixel[1] = clamp_to_u8(g);
        pixel[2] = clamp_to_u8(b);
    }
    out
}

fn convert_ycocg_r_to_rgb_channels(raw: &[u8], channels: usize) -> Vec<u8> {
    let mut out = raw.to_vec();
    for pixel in out.chunks_exact_mut(channels) {
        let y = pixel[0];
        let co = pixel[1].wrapping_sub(128);
        let cg = pixel[2].wrapping_sub(128);

        let t = y.wrapping_sub(signed_half_wrapped(cg));
        let g = cg.wrapping_add(t);
        let b = t.wrapping_sub(signed_half_wrapped(co));
        let r = b.wrapping_add(co);

        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }
    out
}

fn clamp_to_u8(value: i16) -> u8 {
    value.clamp(0, 255) as u8
}

fn save_raw_image(
    output_path: &Path,
    width: u32,
    height: u32,
    channels: u8,
    raw: Vec<u8>,
) -> Result<(), EntroGdError> {
    match channels {
        3 => {
            let image = RgbImage::from_raw(width, height, raw).ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: "raw RGB buffer size does not match width*height*3".to_string(),
                }
            })?;
            image.save(output_path)?;
        }
        4 => {
            let image = RgbaImage::from_raw(width, height, raw).ok_or_else(|| {
                EntroGdError::InvalidMetadata {
                    message: "raw RGBA buffer size does not match width*height*4".to_string(),
                }
            })?;
            image.save(output_path)?;
        }
        _ => {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("unsupported channel count {}", channels),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::compress::{DeviationData, DeviationSample, EncodedData};
    use crate::compression::preprocessor::{
        BitDataInfo, FeatureDataType, FeatureSpec, FeatureTransform,
    };

    // ============================================================================
    // HELPER FUNCTIONS FOR CONSTRUCTING TEST DATA
    // ============================================================================

    /// Creates a simple BitVec with a specific pattern for testing.
    /// Pattern: alternating bits if alternate=true, all zeros if alternate=false
    fn create_bit_pattern(size: usize, alternate: bool) -> crate::BitStream {
        let mut bits = crate::BitStream::with_capacity(size);
        for i in 0..size {
            bits.push(if alternate { i % 2 == 0 } else { false });
        }
        bits
    }

    /// Creates a BitDataInfo for testing with a single feature
    fn create_test_bit_data_info(chunk_size: usize, num_rows: usize) -> BitDataInfo {
        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UnsignedInt,
            bits: chunk_size,
            transform: FeatureTransform::None,
        }];
        BitDataInfo::new(features, chunk_size * num_rows).unwrap()
    }

    /// Creates a deviation sample with specified ID and deviation bits
    fn create_deviation_sample(id_bits: Vec<bool>, deviation_bits: Vec<bool>) -> DeviationSample {
        let mut id = crate::BitStream::with_capacity(id_bits.len());
        for bit in id_bits {
            id.push(bit);
        }
        let mut deviation = crate::BitStream::with_capacity(deviation_bits.len());
        for bit in deviation_bits {
            deviation.push(bit);
        }
        DeviationSample { deviation, id }
    }

    /// Creates normal (non-RLE) encoded data with specified samples
    fn create_normal_encoded_data(samples: Vec<DeviationSample>) -> EncodedData {
        let num_samples = samples.len();
        let num_id_bits = if num_samples > 0 {
            samples[0].id.len()
        } else {
            0
        };
        let num_deviation_bits = if num_samples > 0 {
            samples[0].deviation.len()
        } else {
            0
        };

        let mut encoded_bit_stream = crate::BitStream::new();
        for sample in samples {
            encoded_bit_stream.extend(&sample.deviation);
            encoded_bit_stream.extend(&sample.id);
        }

        EncodedData::Normal(DeviationData::new(
            encoded_bit_stream,
            num_samples,
            num_deviation_bits,
            num_id_bits,
        ))
    }

    /// Creates a base table for testing
    fn create_base_table(num_bases: usize, chunk_size: usize) -> Vec<(crate::BitStream, usize)> {
        (0..num_bases)
            .map(|i| {
                let pattern = create_bit_pattern(chunk_size, i % 2 == 0);
                (pattern, i) // frequency is just the index for simplicity
            })
            .collect()
    }

    /// Creates a base bit positions list
    fn create_base_bit_positions(chunk_size: usize, num_base_bits: usize) -> Vec<usize> {
        (0..chunk_size.min(num_base_bits)).collect()
    }

    /// Creates a minimal CompressedData structure for testing
    fn create_minimal_compressed_data(
        chunk_size: usize,
        num_rows: usize,
        num_bases: usize,
        num_base_bits: usize,
    ) -> CompressedData {
        let metadata = create_test_bit_data_info(chunk_size, num_rows);
        let base_table = create_base_table(num_bases, chunk_size);
        let base_bit_positions = create_base_bit_positions(chunk_size, num_base_bits);

        // Create sample encoded data - one sample per row
        let samples: Vec<DeviationSample> = (0..num_rows)
            .map(|i| {
                let num_deviation_bits = chunk_size.saturating_sub(num_base_bits);
                create_deviation_sample(vec![i % 2 == 0], vec![false; num_deviation_bits])
            })
            .collect();
        let encoded_data = create_normal_encoded_data(samples);

        CompressedData {
            encoded_data,
            condensed_sample_weights: None,
            base_table,
            base_bit_positions,
            metadata,
        }
    }

    /// Creates a CompressedData with condensed sample weights and analytics data
    fn create_compressed_data_with_analytics(
        chunk_size: usize,
        num_rows: usize,
        num_bases: usize,
        num_base_bits: usize,
        _num_condensed: usize,
    ) -> CompressedData {
        let mut compressed =
            create_minimal_compressed_data(chunk_size, num_rows, num_bases, num_base_bits);

        // Add weights for condensed samples
        compressed.condensed_sample_weights = Some((0..num_bases).map(|i| i * 10).collect());

        compressed
    }

    // ============================================================================
    // UNIT TESTS
    // ============================================================================

    #[test]
    fn test_decompress_samples_batch_single_sample() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let indices = vec![0];
        let result = decompress_samples_batch(&compressed, &indices).unwrap();

        assert_eq!(result.data.num_rows, 1);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size);
    }

    #[test]
    fn test_decompress_samples_batch_multiple_samples() {
        let chunk_size = 16;
        let num_rows = 10;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 4, 8);

        let indices = vec![0, 2, 5, 9];
        let result = decompress_samples_batch(&compressed, &indices).unwrap();

        assert_eq!(result.data.num_rows, 4);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size * 4);
    }

    #[test]
    fn test_decompress_samples_batch_all_samples() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let indices: Vec<usize> = (0..num_rows).collect();
        let result = decompress_samples_batch(&compressed, &indices).unwrap();

        assert_eq!(result.data.num_rows, num_rows);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size * num_rows);
    }

    #[test]
    fn test_decompress_samples_batch_out_of_bounds_index() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let indices = vec![0, 10]; // Index 10 is out of bounds
        let result = decompress_samples_batch(&compressed, &indices);

        assert!(result.is_err());
        match result {
            Err(EntroGdError::DecompressionSampleMissing { sample_idx }) => {
                assert_eq!(sample_idx, 10);
            }
            _ => panic!("Expected DecompressionSampleMissing error"),
        }
    }

    #[test]
    fn test_decompress_samples_batch_empty_indices() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let indices = vec![];
        let result = decompress_samples_batch(&compressed, &indices).unwrap();

        assert_eq!(result.data.num_rows, 0);
        assert_eq!(result.data.total_bits(), 0);
    }

    #[test]
    fn test_decompress_samples_batch_zero_features_error() {
        // This test is skipped because it's difficult to create invalid BitDataInfo
        // through the public API - the validation prevents construction of invalid metadata
    }

    #[test]
    fn test_decompress_file_all_rows() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let result = decompress_file(&compressed).unwrap();

        assert_eq!(result.data.num_rows, num_rows);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size * num_rows);
    }

    #[test]
    fn test_decompress_file_single_row() {
        let chunk_size = 16;
        let num_rows = 1;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 4, 8);

        let result = decompress_file(&compressed).unwrap();

        assert_eq!(result.data.num_rows, 1);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size);
    }

    #[test]
    fn test_decompress_file_large_dataset() {
        let chunk_size = 32;
        let num_rows = 100;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 8, 16);

        let result = decompress_file(&compressed).unwrap();

        assert_eq!(result.data.num_rows, num_rows);
        assert_eq!(result.data.chunk_size, chunk_size);
        assert_eq!(result.data.total_bits(), chunk_size * num_rows);
    }

    #[test]
    fn test_decompress_analytics_with_weights() {
        let chunk_size = 8;
        let num_rows = 5;
        let num_bases = 4;
        let compressed =
            create_compressed_data_with_analytics(chunk_size, num_rows, num_bases, 4, 4);

        let result = decompress_analytics(&compressed);

        assert!(result.is_some());
        let analytics = result.unwrap();
        assert_eq!(analytics.samples.len(), num_bases);
        assert_eq!(analytics.weights.len(), num_bases);
        // Check that weights are as expected (0, 10, 20, 30)
        for (i, &weight) in analytics.weights.iter().enumerate() {
            assert_eq!(weight, i * 10);
        }
    }

    #[test]
    fn test_decompress_analytics_without_weights() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let result = decompress_analytics(&compressed);

        assert!(result.is_none());
    }

    #[test]
    fn test_decompress_analytics_samples_match_base_table() {
        let chunk_size = 8;
        let num_rows = 5;
        let num_bases = 4;
        let compressed =
            create_compressed_data_with_analytics(chunk_size, num_rows, num_bases, 4, 4);

        let result = decompress_analytics(&compressed).unwrap();

        // Verify that each sample matches the base table
        for (i, sample) in result.samples.iter().enumerate() {
            assert_eq!(sample.len(), compressed.base_table[i].0.len());
            for bit_idx in 0..sample.len() {
                assert_eq!(sample[bit_idx], compressed.base_table[i].0[bit_idx]);
            }
        }
    }

    #[test]
    fn test_decompressed_data_consistency() {
        let chunk_size = 16;
        let num_rows = 3;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 4, 8);

        // Decompress all rows
        let all_rows = decompress_file(&compressed).unwrap();

        // Decompress specific rows in same order
        let specific_indices = vec![0, 1, 2];
        let specific_rows = decompress_samples_batch(&compressed, &specific_indices).unwrap();

        // Results should match
        assert_eq!(all_rows.data.num_rows, specific_rows.data.num_rows);
        assert_eq!(all_rows.data.total_bits(), specific_rows.data.total_bits());
        assert_eq!(all_rows.data.data, specific_rows.data.data);
    }

    #[test]
    fn test_decompressed_data_partial_subset() {
        let chunk_size = 16;
        let num_rows = 10;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 4, 8);

        // Decompress specific subset
        let indices = vec![1, 3, 7];
        let result = decompress_samples_batch(&compressed, &indices).unwrap();

        assert_eq!(result.data.num_rows, 3);
        assert_eq!(result.data.chunk_size, chunk_size);
    }

    #[test]
    fn test_decompressed_metadata_preservation() {
        let chunk_size = 24;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 8);

        let indices = vec![0, 2];
        let result = decompress_samples_batch(&compressed, &indices).unwrap();

        // Metadata should be preserved and adjusted
        assert_eq!(result.info.chunk_size(), chunk_size);
        assert_eq!(result.info.n_data_samples(), 2); // We decompressed 2 samples
        assert_eq!(result.info.original_size_bits(), chunk_size * 2);
    }

    #[test]
    fn test_decompressed_filter_interface() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let filter = DecompressFileData {};
        let result = filter.process(compressed).unwrap();

        assert_eq!(result.data.num_rows, num_rows);
    }

    #[test]
    fn test_decompressed_rows_filter_interface() {
        let chunk_size = 8;
        let num_rows = 5;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 4);

        let filter = DecompressRowsData;
        let compressed_arc = Arc::new(compressed);
        let indices = vec![0, 2];
        let result = filter.process((compressed_arc, indices)).unwrap();

        assert_eq!(result.data.num_rows, 2);
    }

    #[test]
    fn test_decompress_with_single_base_bit() {
        let chunk_size = 8;
        let num_rows = 3;
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, 1);

        let indices = vec![0];
        let result = decompress_samples_batch(&compressed, &indices).unwrap();

        assert_eq!(result.data.num_rows, 1);
        assert_eq!(result.data.total_bits(), chunk_size);
    }

    #[test]
    fn test_decompress_with_all_base_bits() {
        let chunk_size = 8;
        let num_rows = 3;
        // All bits are base bits, no deviation bits
        let compressed = create_minimal_compressed_data(chunk_size, num_rows, 2, chunk_size);

        let indices = vec![0, 1];
        let result = decompress_samples_batch(&compressed, &indices).unwrap();

        assert_eq!(result.data.num_rows, 2);
        assert_eq!(result.data.total_bits(), chunk_size * 2);
    }
}
