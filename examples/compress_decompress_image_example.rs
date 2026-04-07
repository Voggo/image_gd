use entro_gd::ImageColorModel;
use entro_gd::ScopedTimer;
use entro_gd::compression::preprocessor::DEFAULT_ALIGN_ROWS_TO_WORD;
use entro_gd::prelude::*;
use entro_gd::{EntroGdError, decompress_igd_to_image, init_logging};
use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), EntroGdError> {
    unsafe {
        env::set_var("ENTRO_GD_LOG_TO_STDERR", "1");
        env::set_var("RUST_LOG", "trace");
    }
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
            if path.is_file()
                && let Some(ext) = path.extension()
            {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if matches!(ext_str.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "gif") {
                    files.push(path);
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
    let base_debug_folder = data_folder.join("base_selection_debug");
    let _base_selection_debug_csv_paths = files_to_process.iter().map(|file| {
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image_dataset");
        base_debug_folder.join(format!("{}_base_selection_debug.csv", stem))
    });

    fs::create_dir_all(&compressed_folder)?;
    fs::create_dir_all(&decompressed_folder)?;

    let compression_pipeline = BuildImageBitDataSet {
        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
        color_model: ImageColorModel::YCoCgR,
        pixel_grouping: 3,
        grouping_transform: ImageGroupingTransform::ForFirstPixel,
        pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
    }
    .then(EntropyNaive {})
    .then(SelectBasesProfileAllBits {
        split_into_batches: 1,
        base_bit_impl: BaseBitImpl::Naive,
    })
    .then(EncodeDataFusedDictionary {});

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

        decompress_igd_to_image(&compressed_path, &decompressed_path)?;

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
