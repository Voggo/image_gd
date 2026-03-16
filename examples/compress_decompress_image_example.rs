use bitvec::prelude::*;
use entro_gd::BitDataReconstructionInfo;
use entro_gd::ImageColorModel;
use entro_gd::ScopedTimer;
use entro_gd::prelude::*;
use entro_gd::{BitDataSet, EntroGdError, init_logging};
use image::{RgbImage, RgbaImage};
use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), EntroGdError> {
    let _log_handle = init_logging();
    let _timer =
        ScopedTimer::info("Total processing time for compressing and decompressing image(s)");
    let args: Vec<String> = env::args().collect();
    let input_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "data/images/rustacean.png".to_string()
    };

    let input = Path::new(&input_path);

    let files_to_process = if input.is_dir() {
        // If it's a directory, collect all image files
        let mut files = Vec::new();
        for entry in fs::read_dir(input)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if matches!(ext_str.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "gif") {
                        files.push(path);
                    }
                }
            }
        }
        files.sort();
        files
    } else {
        // If it's a file, process just that file
        vec![input.to_path_buf()]
    };

    if files_to_process.is_empty() {
        tracing::warn!("No image files found to process");
        return Ok(());
    }

    let data_folder = Path::new("data");
    let compressed_folder = data_folder.join("compressed");
    let decompressed_folder = data_folder.join("decompressed");

    fs::create_dir_all(&compressed_folder)?;
    fs::create_dir_all(&decompressed_folder)?;

    let compression_pipeline = BuildImageBitDataSet {
        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
        color_model: ImageColorModel::YCoCgR,
        pixel_grouping: 1,
        grouping_transform: ImageGroupingTransform::Raw,
    }
    .then(EntropyOptimized {})
    .then(GenCondensedSamples { m_max: 0 })
    .then(SelectBases { patience: 10 })
    .then(EncodeDataOptimized {});

    for image_file in files_to_process {
        tracing::info!("Processing: {}", image_file.display());

        let stem = image_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image_dataset");

        let igd_path = compressed_folder.join(format!("{}.igd", stem));
        let decompressed_path = decompressed_folder.join(format!("{}-decompressed.png", stem));

        tracing::info!("IGD output: {}", igd_path.display());
        tracing::info!("Decoded image output: {}", decompressed_path.display());

        let compressed = compression_pipeline.process(image_file.clone())?;
        let compressed_path = SaveIgdFile {
            output_path: igd_path.clone(),
        }
        .process(compressed.clone())?;

        let loaded_compressed = LoadIgdFile {}.process(compressed_path)?;
        let decompressed = (DecompressFileData {}).process(loaded_compressed.clone())?;

        write_bitdata_to_image(&decompressed, &decompressed_path)?;

        tracing::info!(
            "Compression done: original={} bits, encoded={} bits",
            compressed.metadata.original_size_bits(),
            compressed.encoded_data.get_encoded_size()
        );
        tracing::info!(
            "Compression ratio: {:.2}%",
            100.0 * compressed.encoded_data.get_encoded_size() as f64
                / compressed.metadata.original_size_bits() as f64
        );
        tracing::info!("Completed processing: {}", image_file.display());
    }

    tracing::info!("Done.");
    Ok(())
}

fn write_bitdata_to_image(bit_data: &BitDataSet, output_path: &Path) -> Result<(), EntroGdError> {
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
            let feature_bits = bit_data.get_feature(row, feature);
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
        output_path,
        image_info.width,
        image_info.height,
        image_info.channels,
        output_raw,
    )
}

fn byte_from_bits(bits: &BitSlice<usize, Msb0>) -> Result<u8, EntroGdError> {
    if bits.len() != 8 {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("expected 8 bits for image byte, got {}", bits.len()),
        });
    }

    Ok(bits
        .iter()
        .fold(0u8, |acc, bit| (acc << 1) | u8::from(*bit)))
}

fn bits_to_u16(bits: &BitSlice<usize, Msb0>) -> Result<u16, EntroGdError> {
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
    bits: &BitSlice<usize, Msb0>,
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

fn min_position_bits(pixels_per_group: usize) -> usize {
    if pixels_per_group <= 1 {
        0
    } else {
        usize::BITS as usize - (pixels_per_group - 1).leading_zeros() as usize
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

fn signed_half_wrapped(value: u8) -> u8 {
    ((value as i8) >> 1) as u8
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
