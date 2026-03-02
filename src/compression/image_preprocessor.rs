use bitvec::prelude::*;
use image::DynamicImage;
use std::path::PathBuf;

use crate::compression::preprocessor::{
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
}

impl Default for BuildImageBitDataSet {
    fn default() -> Self {
        Self {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
        }
    }
}

impl Filter for BuildImageBitDataSet {
    type Input = PathBuf;
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let decoded = image::open(&input)?;
        match decoded {
            DynamicImage::ImageRgb8(img) => {
                build_image_bitdataset(img.width(), img.height(), 3, self.colorspace.as_u8(), img.into_raw())
            }
            DynamicImage::ImageRgba8(img) => {
                build_image_bitdataset(img.width(), img.height(), 4, self.colorspace.as_u8(), img.into_raw())
            }
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

    let rows = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| EntroGdError::InvalidMetadata {
            message: "image dimensions overflow row count".to_string(),
        })?;
    let channels_usize = channels as usize;
    let expected_len = rows
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

    let features = vec![FeatureSpec::new(FeatureDataType::UnsignedInt, 8); channels_usize];
    let reconstruction = BitDataReconstructionInfo::Image(ImageReconstructionInfo {
        width,
        height,
        channels,
        colorspace,
    });

    let chunk_size = channels_usize * 8;
    let total_bits = rows
        .checked_mul(chunk_size)
        .ok_or_else(|| EntroGdError::InvalidMetadata {
            message: "image bitdata size overflow".to_string(),
        })?;
    let info = BitDataInfo::new_with_reconstruction_info(features, total_bits, reconstruction)?;

    let mut bitstream = BitVec::<usize, Msb0>::with_capacity(total_bits);
    for channel_value in raw {
        push_bits_u8(&mut bitstream, channel_value);
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
                colorspace: 1
            })
        ));

        let _ = std::fs::remove_file(path);
    }
}
