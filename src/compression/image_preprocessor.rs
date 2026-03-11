use bitvec::prelude::*;
use image::DynamicImage;
use std::path::PathBuf;

use crate::compression::tabular_preprocessor::{
    BitData, BitDataInfo, BitDataReconstructionInfo, BitDataSet, FeatureSpec,
    ImageReconstructionInfo,
};
use crate::data_loader::FeatureDataType;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;

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
    pub pixel_grouping: u32,
}

impl Default for BuildImageBitDataSet {
    fn default() -> Self {
        Self {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            pixel_grouping: 1,
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
                img.width(),
                img.height(),
                3,
                self.pixel_grouping,
                self.colorspace.as_u8(),
                img.into_raw(),
            ),
            DynamicImage::ImageRgba8(img) => build_image_bitdataset(
                img.width(),
                img.height(),
                4,
                self.pixel_grouping,
                self.colorspace.as_u8(),
                img.into_raw(),
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
    width: u32,
    height: u32,
    channels: u8,
    pixel_grouping: u32,
    colorspace: u8,
    raw: Vec<u8>,
) -> Result<BitDataSet, EntroGdError> {
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
        pixels.checked_mul(channels_usize)
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
    let feature_bits = pixels_per_group.checked_mul(8).ok_or_else(|| {
        EntroGdError::InvalidMetadata {
            message: "pixel grouping overflows feature bit width".to_string(),
        }
    })?;
    let grouped_width = (width as usize).div_ceil(pixels_per_group);
    let rows = grouped_width
        .checked_mul(height as usize)
        .ok_or_else(|| EntroGdError::InvalidMetadata {
            message: "image dimensions overflow grouped row count".to_string(),
        })?;

    let features = vec![FeatureSpec::new(FeatureDataType::UnsignedInt, feature_bits); channels_usize];
    let reconstruction = BitDataReconstructionInfo::Image(ImageReconstructionInfo {
        width,
        height,
        channels,
        pixel_grouping,
        colorspace,
    });

    let chunk_size = channels_usize
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

    let mut bitstream = BitVec::<usize, Msb0>::with_capacity(total_bits);

    let width_usize = width as usize;
    let height_usize = height as usize;
    for y in 0..height_usize {
        for group_x in 0..grouped_width {
            for channel in 0..channels_usize {
                for offset in 0..pixels_per_group {
                    let pixel_x = group_x * pixels_per_group + offset;
                    if pixel_x < width_usize {
                        let pixel_index = y * width_usize + pixel_x;
                        let raw_index = pixel_index * channels_usize + channel;
                        push_bits_u8(&mut bitstream, raw[raw_index]);
                    } else {
                        push_bits_u8(&mut bitstream, 0);
                    }
                }
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
            pixel_grouping: 1,
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
                pixel_grouping: 1,
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
            pixel_grouping: 1,
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
                pixel_grouping: 1,
                colorspace: 1
            })
        ));

        let _ = std::fs::remove_file(path);
    }

    fn feature_bits_to_bytes(bits: &BitSlice<usize, Msb0>) -> Vec<u8> {
        bits.chunks(8)
            .map(|chunk| {
                chunk.iter().fold(0u8, |acc, bit| (acc << 1) | u8::from(*bit))
            })
            .collect()
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
            pixel_grouping: 4,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(bit_data.num_rows(), 1);
        assert_eq!(bit_data.num_features(), 3);
        assert_eq!(bit_data.feature_bits(0), 32);
        assert_eq!(bit_data.chunk_size(), 96);
        assert_eq!(bit_data.info.original_size_bits(), 96);
        assert_eq!(feature_bits_to_bytes(bit_data.get_feature(0, 0)), vec![1, 2, 3, 4]);
        assert_eq!(feature_bits_to_bytes(bit_data.get_feature(0, 1)), vec![11, 12, 13, 14]);
        assert_eq!(feature_bits_to_bytes(bit_data.get_feature(0, 2)), vec![21, 22, 23, 24]);
        assert!(matches!(
            bit_data.info.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 4,
                height: 1,
                channels: 3,
                pixel_grouping: 4,
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
            pixel_grouping: 4,
        };
        let bit_data = filter.process(path.clone()).unwrap();

        assert_eq!(bit_data.num_rows(), 1);
        assert_eq!(bit_data.feature_bits(0), 32);
        assert_eq!(feature_bits_to_bytes(bit_data.get_feature(0, 0)), vec![10, 40, 70, 0]);
        assert_eq!(feature_bits_to_bytes(bit_data.get_feature(0, 1)), vec![20, 50, 80, 0]);
        assert_eq!(feature_bits_to_bytes(bit_data.get_feature(0, 2)), vec![30, 60, 90, 0]);

        let _ = std::fs::remove_file(path);
    }
}
