use bitvec::prelude::*;
use image::DynamicImage;
use std::path::PathBuf;

pub use crate::compression::preprocessor::{ImageColorModel, ImageGroupingTransform};

use crate::compression::preprocessor::{
    BitData, BitDataInfo, BitDataReconstructionInfo, BitDataSet, FeatureSpec,
    ImageReconstructionInfo,
};
use crate::data_loader::FeatureDataType;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::utils::{min_position_bits, signed_half_wrapped};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageColorSpace {
    /// 0 = sRGB + linear alpha
    SrgbWithLinearAlpha = 0,
    /// 1 = all linear
    Linear = 1,
}

impl ImageColorSpace {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BuildImageBitDataSet {
    pub colorspace: ImageColorSpace,
    pub color_model: ImageColorModel,
    pub pixel_grouping: u32,
    pub grouping_transform: ImageGroupingTransform,
}

#[derive(Debug, Clone)]
struct ImageBuildInput {
    width: u32,
    height: u32,
    channels: u8,
    colorspace: u8,
    raw: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct ImageBuildOptions {
    color_model: ImageColorModel,
    pixel_grouping: u32,
    grouping_transform: ImageGroupingTransform,
}

impl Default for BuildImageBitDataSet {
    fn default() -> Self {
        Self {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: 1,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
        }
    }
}

impl Filter for BuildImageBitDataSet {
    type Input = PathBuf;
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let decoded = image::open(&input)?;
        match decoded {
            DynamicImage::ImageRgb8(img) => build_image_bitdataset(
                ImageBuildInput {
                    width: img.width(),
                    height: img.height(),
                    channels: 3,
                    colorspace: self.colorspace.as_u8(),
                    raw: img.into_raw(),
                },
                ImageBuildOptions {
                    color_model: self.color_model,
                    pixel_grouping: self.pixel_grouping,
                    grouping_transform: self.grouping_transform,
                },
            ),
            DynamicImage::ImageRgba8(img) => build_image_bitdataset(
                ImageBuildInput {
                    width: img.width(),
                    height: img.height(),
                    channels: 4,
                    colorspace: self.colorspace.as_u8(),
                    raw: img.into_raw(),
                },
                ImageBuildOptions {
                    color_model: self.color_model,
                    pixel_grouping: self.pixel_grouping,
                    grouping_transform: self.grouping_transform,
                },
            ),
            other => Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported image color type for dataset ingestion: {:?} (expected RGB8 or RGBA8)",
                    other.color()
                ),
            }),
        }
    }
}

fn build_image_bitdataset(
    input: ImageBuildInput,
    options: ImageBuildOptions,
) -> Result<BitDataSet, EntroGdError> {
    let ImageBuildInput {
        width,
        height,
        channels,
        colorspace,
        raw,
    } = input;
    let ImageBuildOptions {
        color_model,
        pixel_grouping,
        grouping_transform,
    } = options;

    if !matches!(channels, 3 | 4) {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("unsupported channel count {} (expected 3 or 4)", channels),
        });
    }
    if !matches!(colorspace, 0 | 1) {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("unsupported colorspace {} (expected 0 or 1)", colorspace),
        });
    }
    if pixel_grouping == 0 {
        return Err(EntroGdError::InvalidMetadata {
            message: "pixel_grouping must be > 0".to_string(),
        });
    }

    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| EntroGdError::InvalidMetadata {
            message: "image dimensions overflow row count".to_string(),
        })?;
    let channels_usize = channels as usize;
    let expected_len =
        pixels
            .checked_mul(channels_usize)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "image dimensions overflow raw data length".to_string(),
            })?;
    if raw.len() != expected_len {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "invalid raw image length: expected {}, got {}",
                expected_len,
                raw.len()
            ),
        });
    }

    let pixels_per_group = pixel_grouping as usize;
    let feature_bits = feature_bits_for_grouping(pixels_per_group, grouping_transform)?;
    let grouped_width = (width as usize).div_ceil(pixels_per_group);
    let rows = grouped_width.checked_mul(height as usize).ok_or_else(|| {
        EntroGdError::InvalidMetadata {
            message: "image dimensions overflow grouped row count".to_string(),
        }
    })?;

    let features =
        vec![FeatureSpec::new(FeatureDataType::UnsignedInt, feature_bits); channels_usize];
    let reconstruction = BitDataReconstructionInfo::Image(ImageReconstructionInfo {
        width,
        height,
        channels,
        color_model,
        pixel_grouping,
        grouping_transform,
        colorspace,
    });

    let chunk_size =
        channels_usize
            .checked_mul(feature_bits)
            .ok_or_else(|| EntroGdError::InvalidMetadata {
                message: "image chunk size overflow".to_string(),
            })?;
    let total_bits = rows
        .checked_mul(chunk_size)
        .ok_or_else(|| EntroGdError::InvalidMetadata {
            message: "image bitdata size overflow".to_string(),
        })?;
    let info = BitDataInfo::new_with_reconstruction_info(features, total_bits, reconstruction)?;

    let encoded_raw = match color_model {
        ImageColorModel::Rgb => raw,
        ImageColorModel::YCoCg => convert_rgb_to_ycocg_channels(&raw, channels_usize),
        ImageColorModel::YCoCgR => convert_rgb_to_ycocg_r_channels(&raw, channels_usize),
    };

    let mut bitstream = BitVec::<usize, Msb0>::with_capacity(total_bits);

    let width_usize = width as usize;
    let height_usize = height as usize;
    for y in 0..height_usize {
        for group_x in 0..grouped_width {
            for channel in 0..channels_usize {
                let grouped_values = grouped_channel_values(
                    &encoded_raw,
                    width_usize,
                    channels_usize,
                    y,
                    group_x,
                    channel,
                    pixels_per_group,
                );
                encode_grouped_channel(&mut bitstream, &grouped_values, grouping_transform);
            }
        }
    }

    let data = BitData {
        data: bitstream,
        chunk_size,
        num_rows: rows,
    };

    Ok(BitDataSet { data, info })
}

fn push_bits_u8(out: &mut BitVec<usize, Msb0>, value: u8) {
    for shift in (0..8).rev() {
        out.push(((value >> shift) & 1) == 1);
    }
}

fn push_bits_u16(out: &mut BitVec<usize, Msb0>, value: u16, bit_count: usize) {
    for shift in (0..bit_count).rev() {
        out.push(((value >> shift) & 1) == 1);
    }
}

fn feature_bits_for_grouping(
    pixels_per_group: usize,
    grouping_transform: ImageGroupingTransform,
) -> Result<usize, EntroGdError> {
    match grouping_transform {
        ImageGroupingTransform::Raw => {
            pixels_per_group
                .checked_mul(8)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "pixel grouping overflows feature bit width".to_string(),
                })
        }
        ImageGroupingTransform::ForFirstPixel => {
            if pixels_per_group == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "pixel_grouping must be > 0".to_string(),
                });
            }

            let residual_bits = pixels_per_group
                .saturating_sub(1)
                .checked_mul(9)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "pixel grouping overflows residual bit width".to_string(),
                })?;
            8usize
                .checked_add(residual_bits)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "pixel grouping overflows feature bit width".to_string(),
                })
        }
        ImageGroupingTransform::ForMin => {
            if pixels_per_group == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "pixel_grouping must be > 0".to_string(),
                });
            }

            let position_bits = min_position_bits(pixels_per_group);
            let residual_bits = pixels_per_group
                .saturating_sub(1)
                .checked_mul(9)
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "pixel grouping overflows residual bit width".to_string(),
                })?;
            8usize
                .checked_add(position_bits)
                .and_then(|bits| bits.checked_add(residual_bits))
                .ok_or_else(|| EntroGdError::InvalidMetadata {
                    message: "pixel grouping overflows feature bit width".to_string(),
                })
        }
    }
}

fn grouped_channel_values(
    raw: &[u8],
    width: usize,
    channels: usize,
    y: usize,
    group_x: usize,
    channel: usize,
    pixels_per_group: usize,
) -> Vec<u8> {
    let mut values = Vec::with_capacity(pixels_per_group);
    for offset in 0..pixels_per_group {
        let pixel_x = group_x * pixels_per_group + offset;
        if pixel_x < width {
            let pixel_index = y * width + pixel_x;
            let raw_index = pixel_index * channels + channel;
            values.push(raw[raw_index]);
        } else {
            values.push(0);
        }
    }
    values
}

fn encode_grouped_channel(
    out: &mut BitVec<usize, Msb0>,
    values: &[u8],
    grouping_transform: ImageGroupingTransform,
) {
    match grouping_transform {
        ImageGroupingTransform::Raw => {
            for &value in values {
                push_bits_u8(out, value);
            }
        }
        ImageGroupingTransform::ForFirstPixel => {
            let anchor = values.first().copied().unwrap_or(0);
            encode_for_anchor(out, values, anchor);
        }
        ImageGroupingTransform::ForMin => {
            let (min_position, anchor) = values
                .iter()
                .copied()
                .enumerate()
                .min_by_key(|&(idx, value)| (value, idx))
                .unwrap_or((0, 0));
            push_bits_u8(out, anchor);
            let position_bits = min_position_bits(values.len());
            push_bits_u16(out, min_position as u16, position_bits);
            for (idx, &value) in values.iter().enumerate() {
                if idx == min_position {
                    continue;
                }
                let delta = value as i16 - anchor as i16;
                let biased = (delta + 255) as u16;
                push_bits_u16(out, biased, 9);
            }
        }
    }
}

fn encode_for_anchor(out: &mut BitVec<usize, Msb0>, values: &[u8], anchor: u8) {
    push_bits_u8(out, anchor);
    for &value in values.iter().skip(1) {
        let delta = value as i16 - anchor as i16;
        let biased = (delta + 255) as u16;
        push_bits_u16(out, biased, 9);
    }
}

fn convert_rgb_to_ycocg_channels(raw: &[u8], channels: usize) -> Vec<u8> {
    tracing::warn!("By using the YCoCg color model, When converting back to RGB is lossy!");
    let mut out = raw.to_vec();
    for pixel in out.chunks_exact_mut(channels) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        let y = (r >> 2).wrapping_add(g >> 1).wrapping_add(b >> 2);
        let co = ((i16::from(r >> 1) - i16::from(b >> 1)) + 128) as u8;
        let cg = ((-i16::from(r >> 2) + i16::from(g >> 1) - i16::from(b >> 2)) + 128) as u8;

        pixel[0] = y;
        pixel[1] = co;
        pixel[2] = cg;
    }
    out
}

fn convert_rgb_to_ycocg_r_channels(raw: &[u8], channels: usize) -> Vec<u8> {
    let mut out = raw.to_vec();
    for pixel in out.chunks_exact_mut(channels) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        let co = r.wrapping_sub(b);
        let t = b.wrapping_add(signed_half_wrapped(co));
        let cg = g.wrapping_sub(t);
        let y = t.wrapping_add(signed_half_wrapped(cg));

        pixel[0] = y;
        pixel[1] = co.wrapping_add(128);
        pixel[2] = cg.wrapping_add(128);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage, Rgba, RgbaImage};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp_path(name: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}.png", name, ts))
    }

    #[test]
    fn loads_rgb_image_as_rows_times_channels() {
        let path = unique_tmp_path("entro_gd_rgb");
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.put_pixel(1, 0, Rgb([40, 50, 60]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: 1,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(bit_data.num_rows(), 2);
        assert_eq!(bit_data.num_features(), 3);
        assert_eq!(bit_data.chunk_size(), 24);
        assert_eq!(bit_data.info.original_size_bits(), 48);
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 1,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: 1,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn loads_rgba_image_with_four_features() {
        let path = unique_tmp_path("entro_gd_rgba");
        let mut img = RgbaImage::new(1, 2);
        img.put_pixel(0, 0, Rgba([1, 2, 3, 4]));
        img.put_pixel(0, 1, Rgba([5, 6, 7, 8]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::Linear,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: 1,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(bit_data.num_rows(), 2);
        assert_eq!(bit_data.num_features(), 4);
        assert_eq!(bit_data.chunk_size(), 32);
        assert_eq!(bit_data.info.original_size_bits(), 64);
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 1,
                height: 2,
                channels: 4,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: 1,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 1
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    fn bits_to_u16(bits: &BitSlice<usize, Msb0>) -> u16 {
        bits.iter()
            .fold(0u16, |acc, bit| (acc << 1) | u16::from(*bit))
    }

    fn decode_grouped_feature(
        bits: &BitSlice<usize, Msb0>,
        pixel_grouping: usize,
        grouping_transform: ImageGroupingTransform,
    ) -> Vec<u8> {
        match grouping_transform {
            ImageGroupingTransform::Raw => bits
                .chunks(8)
                .map(|chunk| {
                    chunk
                        .iter()
                        .fold(0u8, |acc, bit| (acc << 1) | u8::from(*bit))
                })
                .collect(),
            ImageGroupingTransform::ForFirstPixel => {
                let anchor = bits_to_u16(&bits[0..8]) as i16;
                let mut values = Vec::with_capacity(pixel_grouping);
                values.push(anchor as u8);
                for offset in 0..pixel_grouping.saturating_sub(1) {
                    let start = 8 + offset * 9;
                    let end = start + 9;
                    let biased = bits_to_u16(&bits[start..end]) as i16;
                    values.push((anchor + biased - 255) as u8);
                }
                values
            }
            ImageGroupingTransform::ForMin => {
                let anchor = bits_to_u16(&bits[0..8]) as i16;
                let position_bits = min_position_bits(pixel_grouping);
                let min_position = if position_bits == 0 {
                    0
                } else {
                    bits_to_u16(&bits[8..8 + position_bits]) as usize
                };
                let mut values = Vec::with_capacity(pixel_grouping);
                let mut residual_cursor = 8 + position_bits;
                for idx in 0..pixel_grouping {
                    if idx == min_position {
                        values.push(anchor as u8);
                    } else {
                        let biased =
                            bits_to_u16(&bits[residual_cursor..residual_cursor + 9]) as i16;
                        values.push((anchor + biased - 255) as u8);
                        residual_cursor += 9;
                    }
                }
                values
            }
        }
    }

    #[test]
    fn loads_grouped_rgb_pixels_into_wider_features() {
        let path = unique_tmp_path("entro_gd_grouped_rgb");
        let mut img = RgbImage::new(4, 1);
        img.put_pixel(0, 0, Rgb([1, 11, 21]));
        img.put_pixel(1, 0, Rgb([2, 12, 22]));
        img.put_pixel(2, 0, Rgb([3, 13, 23]));
        img.put_pixel(3, 0, Rgb([4, 14, 24]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: 4,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(bit_data.num_rows(), 1);
        assert_eq!(bit_data.num_features(), 3);
        assert_eq!(bit_data.feature_bits(0), 35);
        assert_eq!(bit_data.chunk_size(), 105);
        assert_eq!(bit_data.info.original_size_bits(), 105);
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 0),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 1),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![11, 12, 13, 14]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 2),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![21, 22, 23, 24]
        );
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 4,
                height: 1,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: 4,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn zero_pads_partial_group_at_row_end() {
        let path = unique_tmp_path("entro_gd_grouped_rgb_partial");
        let mut img = RgbImage::new(3, 1);
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.put_pixel(1, 0, Rgb([40, 50, 60]));
        img.put_pixel(2, 0, Rgb([70, 80, 90]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: 4,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(bit_data.num_rows(), 1);
        assert_eq!(bit_data.feature_bits(0), 35);
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 0),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![10, 40, 70, 0]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 1),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![20, 50, 80, 0]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 2),
                4,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![30, 60, 90, 0]
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn loads_grouped_rgb_pixels_with_for_min() {
        let path = unique_tmp_path("entro_gd_grouped_rgb_for_min");
        let mut img = RgbImage::new(4, 1);
        img.put_pixel(0, 0, Rgb([9, 14, 21]));
        img.put_pixel(1, 0, Rgb([4, 18, 17]));
        img.put_pixel(2, 0, Rgb([7, 16, 19]));
        img.put_pixel(3, 0, Rgb([5, 15, 20]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::Rgb,
            pixel_grouping: 4,
            grouping_transform: ImageGroupingTransform::ForMin,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(bit_data.num_rows(), 1);
        assert_eq!(bit_data.feature_bits(0), 37);
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 0),
                4,
                ImageGroupingTransform::ForMin
            ),
            vec![9, 4, 7, 5]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 1),
                4,
                ImageGroupingTransform::ForMin
            ),
            vec![14, 18, 16, 15]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 2),
                4,
                ImageGroupingTransform::ForMin
            ),
            vec![21, 17, 19, 20]
        );
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 4,
                height: 1,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: 4,
                grouping_transform: ImageGroupingTransform::ForMin,
                colorspace: 0
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn encodes_rgb_as_ycocg_channels() {
        let path = unique_tmp_path("entro_gd_ycocg_rgb");
        let mut img = RgbImage::new(1, 1);
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::YCoCg,
            pixel_grouping: 1,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 0),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![19]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 1),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![118]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 2),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![129]
        );
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                color_model: ImageColorModel::YCoCg,
                ..
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn leaves_alpha_unchanged_for_rgba_ycocg() {
        let path = unique_tmp_path("entro_gd_ycocg_rgba");
        let mut img = RgbaImage::new(1, 1);
        img.put_pixel(0, 0, Rgba([100, 110, 120, 130]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::Linear,
            color_model: ImageColorModel::YCoCg,
            pixel_grouping: 1,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 3),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![130]
        );

        let _ = std::fs::remove_file(path);
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

    #[test]
    fn encodes_rgb_as_ycocg_r_channels() {
        let path = unique_tmp_path("entro_gd_ycocg_r_rgb");
        let mut img = RgbImage::new(1, 1);
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::YCoCgR,
            pixel_grouping: 1,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 0),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![20]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 1),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![108]
        );
        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 2),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![128]
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ycocg_r_roundtrip_is_reversible_for_sampled_rgb_space() {
        for r in (0u8..=255).step_by(17) {
            for g in (0u8..=255).step_by(17) {
                for b in (0u8..=255).step_by(17) {
                    let raw = vec![r, g, b];
                    let transformed = convert_rgb_to_ycocg_r_channels(&raw, 3);
                    let recovered = convert_ycocg_r_to_rgb_channels(&transformed, 3);
                    assert_eq!(recovered, raw);
                }
            }
        }
    }

    #[test]
    fn leaves_alpha_unchanged_for_rgba_ycocg_r() {
        let path = unique_tmp_path("entro_gd_ycocg_r_rgba");
        let mut img = RgbaImage::new(1, 1);
        img.put_pixel(0, 0, Rgba([100, 110, 120, 130]));
        img.save(&path).unwrap();

        let filter = BuildImageBitDataSet {
            colorspace: ImageColorSpace::Linear,
            color_model: ImageColorModel::YCoCgR,
            pixel_grouping: 1,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(
            decode_grouped_feature(
                bit_data.get_feature(0, 3),
                1,
                ImageGroupingTransform::ForFirstPixel
            ),
            vec![130]
        );

        let _ = std::fs::remove_file(path);
    }
}
