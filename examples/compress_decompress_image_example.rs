use entro_gd::BitDataReconstructionInfo;
use entro_gd::prelude::*;
use image::{RgbImage, RgbaImage};
use std::env;
use std::path::Path;

fn main() -> Result<(), EntroGdError> {
    init_logging();

    let args: Vec<String> = env::args().collect();
    let input_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "data/rustacean.png".to_string()
    };

    let input = Path::new(&input_path);
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image_dataset");
    let parent = input.parent().unwrap_or_else(|| Path::new("."));

    let igd_path = parent.join(format!("{}.igd", stem));
    let output_path = parent.join(format!("{}-decompressed.png", stem));

    tracing::info!("Image input: {}", input.display());
    tracing::info!("IGD output: {}", igd_path.display());
    tracing::info!("Decoded image output: {}", output_path.display());

    let compression_pipeline =
        build_image_compression_pipeline(ImageColorSpace::SrgbWithLinearAlpha, 50, 10);

    let compressed = compression_pipeline.process(input.to_path_buf())?;
    let compressed_path = SaveIgdFile {
        output_path: igd_path.clone(),
    }
    .process(compressed.clone())?;

    let loaded_compressed = LoadIgdFile {}.process(compressed_path)?;
    let decompressed = (DecompressFileData {}).process(loaded_compressed.clone())?;

    write_bitdata_to_image(&decompressed, &output_path)?;

    tracing::info!(
        "Compression done: original={} bits, encoded={} bits",
        compressed.metadata.original_size_bits(),
        compressed.encoded_data.get_encoded_size()
    );
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
    if bit_data.num_features() != channels {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "feature count {} does not match image channel count {}",
                bit_data.num_features(),
                channels
            ),
        });
    }

    let mut raw = Vec::with_capacity(bit_data.num_rows() * channels);
    for row in 0..bit_data.num_rows() {
        for feature in 0..channels {
            let feature_bits = bit_data.get_feature(row, feature);
            let spec = bit_data.info.feature_spec(feature);
            let value = decode_value_from_bits(feature_bits, spec);
            match value {
                DataValue::Unsigned(v) if v <= u8::MAX as u64 => raw.push(v as u8),
                _ => {
                    return Err(EntroGdError::InvalidMetadata {
                        message: format!(
                            "invalid decoded value for image channel at row {}, feature {}",
                            row, feature
                        ),
                    });
                }
            }
        }
    }

    save_raw_image(
        output_path,
        image_info.width,
        image_info.height,
        image_info.channels,
        raw,
    )
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
