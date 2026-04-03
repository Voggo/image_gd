use entro_gd::data_loader::{CsvDataLoader, DataLoader, DataValue, FloatStorage};
use entro_gd::prelude::*;
use entro_gd::{
    BitDataSet, DecompressAnalytics, DecompressFileData, EntroGdError, LoadEgdFile, SaveEgdFile,
    ScopedTimer, decode_value_from_bits, init_logging, write_bitdata_as_csv,
};
use std::env;
use std::path::Path;

fn main() -> Result<(), EntroGdError> {
    let _log_handle = init_logging();

    let _timer = ScopedTimer::info("Total compression-decompression process");

    // Get input path from command line argument
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        tracing::info!("Usage: {} <input_csv_file>", args[0]);
        tracing::info!(
            "Example: {} data/tabular/data-10000-4-8bit-sparse.csv",
            args[0]
        );
        return Ok(());
    }

    let input_path = &args[1];

    // Generate output path by inserting "-decompressed" before the file extension
    let output_path = {
        let path = Path::new(input_path);
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let ext = path.extension().unwrap().to_str().unwrap();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        parent
            .join(format!("{}-decompressed.{}", stem, ext))
            .to_str()
            .unwrap()
            .to_string()
    };

    tracing::info!("Loading CSV data from: {}", input_path);
    let loader = CsvDataLoader::new(true).with_float_storage(FloatStorage::F32);
    let loaded = loader.load(input_path)?;
    let dataset = loaded.dataset;
    let metadata = loaded.metadata;

    tracing::info!("Dataset loaded successfully!");
    tracing::info!("  Rows: {}", dataset.num_rows());
    tracing::info!("  Columns: {}", dataset.num_columns());

    let bit_data = BitDataSet::from_dataset(&dataset)?;

    tracing::info!("\nOriginal BitData:");
    tracing::info!("  Total bits: {}", bit_data.data.total_bits());
    tracing::info!("  Chunk size: {}", bit_data.data.chunk_size);
    tracing::info!("  Num rows: {}", bit_data.data.num_rows);
    tracing::info!("  Num features: {}", bit_data.info.num_features());

    // construct EGD file path by replacing input file extension with .egd
    let egd_file_path = {
        let path = Path::new(input_path);
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        parent
            .join(format!("{}.egd", stem))
            .to_str()
            .unwrap()
            .to_string()
    };

    let _compression_timer =
        ScopedTimer::info("Compression process without loading .csv file and parsing");
    // Compress the data with filters (pipe-and-filter style)
    tracing::info!("\nCompressing data...");
    let compression_pipeline = EntropyBatched {}
        .then(GenCondensedSamples { m_max: 50 })
        .then(SelectBasesOptimized {
            patience: 10,
            base_bit_impl: BaseBitImpl::BatchGroups,
        })
        .then(EncodeDataOptimized {});
    let compressed = compression_pipeline.process(bit_data)?;

    let compressed_path = SaveEgdFile {
        output_path: egd_file_path.into(),
    }
    .process(compressed.clone())?;

    // Drop the compression timer here to exclude file saving time from the compression timing
    drop(_compression_timer);

    // Load the compressed data back from the EGD file to verify it was saved correctly
    let loaded_compressed = LoadEgdFile {}.process(compressed_path)?;

    assert_eq!(
        compressed.metadata.original_size_bits(),
        loaded_compressed.metadata.original_size_bits()
    );
    assert_eq!(
        compressed.metadata.num_features(),
        loaded_compressed.metadata.num_features()
    );
    assert_eq!(
        compressed.base_table.len(),
        loaded_compressed.base_table.len()
    );
    // Base bit positions can differ in order, so we skip exact ordering checks here.
    // assert_eq!(
    //     compressed.base_bit_positions,
    //     loaded_compressed.base_bit_positions
    // );
    assert_eq!(
        compressed.encoded_data.get_num_samples(),
        loaded_compressed.encoded_data.get_num_samples()
    );

    tracing::info!("Compression complete!");
    tracing::info!(
        "  Original size: {} bits",
        compressed.metadata.original_size_bits()
    );
    tracing::info!(
        "  Encoded stream size: {} bits",
        compressed.encoded_data.get_encoded_size()
    );
    tracing::info!("  Base table entries: {}", compressed.base_table.len());
    tracing::info!("  Base bit positions: {:?}", compressed.base_bit_positions);

    // Calculate compression ratio
    let base_table_size = loaded_compressed
        .base_table
        .iter()
        .map(|(pattern, _)| pattern.len())
        .sum::<usize>();
    let total_compressed_size = loaded_compressed.encoded_data.get_encoded_size() + base_table_size;
    let compression_ratio =
        total_compressed_size as f64 / loaded_compressed.metadata.original_size_bits() as f64;
    tracing::info!("  Approximate compression ratio: {:.2}", compression_ratio);

    // Decompress the data through filters
    tracing::info!("\nDecompressing data...");
    let decompressed = (DecompressFileData {}).process(loaded_compressed.clone())?;
    if let Some(analytics) = (DecompressAnalytics {}).process(loaded_compressed.clone())? {
        tracing::info!("\nDecompression analytics:");
        for (i, (sample, weight)) in analytics
            .samples
            .iter()
            .zip(analytics.weights.iter())
            .enumerate()
        {
            tracing::info!("  Sample {}: {} weight", i, weight);

            for feature_idx in 0..loaded_compressed.metadata.num_features() {
                let feature_start = loaded_compressed.metadata.feature_offset(feature_idx);
                let feature_end =
                    feature_start + loaded_compressed.metadata.feature_bits(feature_idx);
                let feature_bits = &sample[feature_start..feature_end];
                let spec = loaded_compressed.metadata.feature_spec(feature_idx);

                let formatted = match decode_value_from_bits(feature_bits, spec) {
                    DataValue::Unsigned(v) => v.to_string(),
                    DataValue::Signed(v) => v.to_string(),
                    DataValue::F32(v) => v.to_string(),
                    DataValue::F64(v) => v.to_string(),
                };

                tracing::info!("    Feature {}: {}", feature_idx, formatted);
            }
        }
    }

    tracing::info!("Decompression complete!");
    tracing::info!("  Total bits: {}", decompressed.data.total_bits());
    tracing::info!("  Num rows: {}", decompressed.data.num_rows);
    tracing::info!("  Num features: {}", decompressed.info.num_features());

    // Verify data integrity
    // let data_matches = bit_data.data == decompressed.data;
    // println!("\nData integrity check: {}", if data_matches { "PASSED ✓" } else { "FAILED ✗" });

    // if !data_matches {
    //     // Count mismatches for debugging
    //     let mismatches: usize = bit_data.data.iter()
    //         .zip(decompressed.data.iter())
    //         .filter(|(a, b)| *a != *b)
    //         .count();
    //     println!("  Number of bit mismatches: {} out of {}", mismatches, bit_data.total_bits());
    // }

    // Convert decompressed BitData back to CSV
    tracing::info!("\nWriting decompressed data to: {}", output_path);
    write_bitdata_as_csv(&decompressed, &output_path, metadata.headers.as_deref())?;
    tracing::info!("Done!");

    Ok(())
}
