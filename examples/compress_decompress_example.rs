use entro_gd::compression::compress::{compress, decompress_analytics, decompress_file};
use entro_gd::data_loader::{CsvDataLoader, DataLoader, FeatureDataType};
use entro_gd::error::EntroGdError;
use entro_gd::preprocessor::BitDataSet;
use std::fs::File;
use std::io::Write;
use std::{env, u64};
use std::path::Path;

fn main() -> Result<(), EntroGdError> {
    // Get input path from command line argument
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_csv_file>", args[0]);
        eprintln!("Example: {} data/data-10000-4-8bit-sparse.csv", args[0]);
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
    
    println!("Loading CSV data from: {}", input_path);
    let loader = CsvDataLoader::new(true).with_float_type(FeatureDataType::F32);
    let entro_gd::data_loader::LoadedDataset { dataset, metadata } = loader.load(input_path)?;
    
    println!("Dataset loaded successfully!");
    println!("  Rows: {}", dataset.num_rows());
    println!("  Columns: {}", dataset.num_columns());
    
    let bit_data = BitDataSet::from_dataset(&dataset)?;
    
    println!("\nOriginal BitData:");
    println!("  Total bits: {}", bit_data.data.total_bits());
    println!("  Chunk size: {}", bit_data.data.chunk_size);
    println!("  Num rows: {}", bit_data.data.num_rows);
    println!("  Num features: {}", bit_data.info.num_features());
    
    // Compress the data
    println!("\nCompressing data...");
    let compressed = compress(&bit_data);
    
    println!("Compression complete!");
    println!("  Original size: {} bits", compressed.metadata.original_size_bits);
    println!("  Encoded stream size: {} bits", compressed.encoded_data.get_encoded_size());
    println!("  Base table entries: {}", compressed.base_table.len());
    println!("  Base bit positions: {:?}", compressed.base_bit_positions);
    
    // Calculate compression ratio
    let base_table_size = compressed.base_table.iter()
        .map(|(pattern, _)| pattern.len())
        .sum::<usize>();
    let total_compressed_size = compressed.encoded_data.get_encoded_size() + base_table_size;
    let compression_ratio = total_compressed_size as f64 / compressed.metadata.original_size_bits as f64;
    println!("  Approximate compression ratio: {:.2}", compression_ratio); 
    // Decompress the data
    println!("\nDecompressing data...");
    let decompressed = decompress_file(&compressed)?;

    if let Some(analytics) = decompress_analytics(&compressed) {
        println!("\nDecompression analytics:");
        for (i, (sample, weight)) in analytics.samples.iter().zip(analytics.weights.iter()).enumerate() {
            let mut sample_int = 0u64;
            for bit in sample.iter().take(64) {
                sample_int = (sample_int << 1) | (*bit as u64);
            }
            println!("  Sample {}: {} weight, {} sample int", i, weight, sample_int);
        }
    }
    
    println!("Decompression complete!");
    println!("  Total bits: {}", decompressed.data.total_bits());
    println!("  Num rows: {}", decompressed.data.num_rows);
    println!("  Num features: {}", decompressed.info.num_features());
    
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
    println!("\nWriting decompressed data to: {}", output_path);
    write_bitdata_to_csv(&decompressed, &output_path, metadata.headers.as_deref())?;
    
    println!("Done!");
    
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
            let mut value: u64 = 0;
            for (i, bit) in feature_bits.iter().enumerate() {
                if *bit {
                    value |= 1 << (spec.bits - 1 - i);
                }
            }
            let formatted = match spec.data_type {
                FeatureDataType::UnsignedInt => value.to_string(),
                FeatureDataType::SignedInt => {
                    let signed = if spec.bits == 64 {
                        value as i64
                    } else {
                        let shift = 64 - spec.bits;
                        ((value << shift) as i64) >> shift
                    };
                    signed.to_string()
                }
                FeatureDataType::F32 => {
                    let f = f32::from_bits(value as u32);
                    f.to_string()
                }
                FeatureDataType::F64 => {
                    let f = f64::from_bits(value);
                    f.to_string()
                }
            };
            values.push(formatted);
        }
        writeln!(file, "{}", values.join(","))?;
    }
    
    Ok(())
}
