use entro_gd::compression::compress::{compress, decompress_analytics, decompress_file};
use entro_gd::compression::file_format::{save_compressed_as_egd, load_compressed_from_egd};
use entro_gd::data_loader::{CsvDataLoader, DataLoader, DataValue, FeatureDataType};
use entro_gd::error::EntroGdError;
use entro_gd::preprocessor::{decode_value_from_bits, BitDataSet};
use entro_gd::init_logging;
use entro_gd::timing::ScopedTimer;
use std::fs::File;
use std::io::Write;
use std::{env, u64};
use std::path::Path;

fn main() -> Result<(), EntroGdError> {
    init_logging();

    let _timer = ScopedTimer::info("Total compression-decompression process");

    // Get input path from command line argument
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        log::info!("Usage: {} <input_csv_file>", args[0]);
        log::info!("Example: {} data/data-10000-4-8bit-sparse.csv", args[0]);
        std::process::exit(1);
    }
    
    let input_path = &args[1];
    
    // Generate output path by inserting "-decompressed" before the file extension
    let output_path = {
        let path = Path::new(input_path);
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let ext = path.extension().unwrap().to_str().unwrap();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        parent.join(format!("{}-decompressed.{}", stem, ext))
            .to_str().unwrap().to_string()
    };
    
    log::info!("Loading CSV data from: {}", input_path);
    let loader = CsvDataLoader::new(true).with_float_type(FeatureDataType::F32);
    let entro_gd::data_loader::LoadedDataset { dataset, metadata } = loader.load(input_path)?;
    
    log::info!("Dataset loaded successfully!");
    log::info!("  Rows: {}", dataset.num_rows());
    log::info!("  Columns: {}", dataset.num_columns());
    
    let bit_data = BitDataSet::from_dataset(&dataset)?;
    
    log::info!("\nOriginal BitData:");
    log::info!("  Total bits: {}", bit_data.data.total_bits());
    log::info!("  Chunk size: {}", bit_data.data.chunk_size);
    log::info!("  Num rows: {}", bit_data.data.num_rows);
    log::info!("  Num features: {}", bit_data.info.num_features());
    
    // Compress the data
    log::info!("\nCompressing data...");
    let compressed = compress(&bit_data);
    
    // Save compressed data to EGD file format
    let egd_file_path = {
        let path = Path::new(input_path);
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        parent.join(format!("{}.egd", stem))
            .to_str().unwrap().to_string()
    };
    save_compressed_as_egd(&compressed, &egd_file_path)?;

    // Load the compressed data back from the EGD file to verify it was saved correctly
    let loaded_compressed = load_compressed_from_egd(&egd_file_path)?;

    assert_eq!(compressed.metadata.original_size_bits, loaded_compressed.metadata.original_size_bits);
    assert_eq!(compressed.metadata.num_features(), loaded_compressed.metadata.num_features());
    assert_eq!(compressed.base_table.len(), loaded_compressed.base_table.len());
    assert_eq!(compressed.encoded_data.get_num_samples(), loaded_compressed.encoded_data.get_num_samples());
    
    log::info!("Compression complete!");
    log::info!("  Original size: {} bits", compressed.metadata.original_size_bits);
    log::info!("  Encoded stream size: {} bits", compressed.encoded_data.get_encoded_size());
    log::info!("  Base table entries: {}", compressed.base_table.len());
    log::info!("  Base bit positions: {:?}", compressed.base_bit_positions);
    
    // Calculate compression ratio
    let base_table_size = loaded_compressed.base_table.iter()
        .map(|(pattern, _)| pattern.len())
        .sum::<usize>();
    let total_compressed_size = loaded_compressed.encoded_data.get_encoded_size() + base_table_size;
    let compression_ratio = total_compressed_size as f64 / loaded_compressed.metadata.original_size_bits as f64;
    log::info!("  Approximate compression ratio: {:.2}", compression_ratio);
    // Decompress the data
    log::info!("\nDecompressing data...");
    let decompressed = decompress_file(&loaded_compressed)?;
    if let Some(analytics) = decompress_analytics(&loaded_compressed) {
        log::info!("\nDecompression analytics:");
        for (i, (sample, weight)) in analytics.samples.iter().zip(analytics.weights.iter()).enumerate() {
            let mut sample_int = 0u64;
            for bit in sample.iter().take(64) {
                sample_int = (sample_int << 1) | (*bit as u64);
            }
            log::info!("  Sample {}: {} weight, {} sample int", i, weight, sample_int);
        }
    }
    
    log::info!("Decompression complete!");
    log::info!("  Total bits: {}", decompressed.data.total_bits());
    log::info!("  Num rows: {}", decompressed.data.num_rows);
    log::info!("  Num features: {}", decompressed.info.num_features());
    
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
    log::info!("\nWriting decompressed data to: {}", output_path);
    write_bitdata_to_csv(&decompressed, &output_path, metadata.headers.as_deref())?;
    log::info!("Done!");
    
    Ok(())
}

/// Convert BitData back to a CSV file
fn write_bitdata_to_csv(
    bit_data: &BitDataSet,
    path: &str,
    headers: Option<&[String]>,
) -> Result<(), EntroGdError> {
    let mut file = File::create(path)?;
    
    // Write headers if available
    if let Some(headers) = headers {
        writeln!(file, "{}", headers.join(","))?;
    }
    
    // Write each row
    for row in 0..bit_data.data.num_rows {
        let mut values: Vec<String> = Vec::with_capacity(bit_data.info.num_features());
        for feature in 0..bit_data.info.num_features() {
            let feature_bits = bit_data.get_feature(row, feature);
            let spec = &bit_data.info.features[feature];
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
