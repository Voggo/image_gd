use bitvec::field::BitField;
use entro_gd::data_loader::Dataset;
use entro_gd::preprocessor::BitData;
use entro_gd::compression::compress::{compress, decompress_file, decompress_analytics};
use std::fs::File;
use std::io::Write;
use std::{env, u64};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let dataset = Dataset::load_csv(input_path, true)?;
    
    println!("Dataset loaded successfully!");
    println!("  Rows: {}", dataset.num_rows());
    println!("  Columns: {}", dataset.num_columns());
    
    // Convert dataset to BitData with 8 bits per feature
    let bits_per_feature = 8;
    let mut bit_data = BitData::from_dataset(&dataset, bits_per_feature);
    
    println!("\nOriginal BitData:");
    println!("  Total bits: {}", bit_data.total_bits());
    println!("  Chunk size: {}", bit_data.chunk_size);
    println!("  Num rows: {}", bit_data.num_rows);
    println!("  Num features: {}", bit_data.num_features);
    
    // Compress the data
    println!("\nCompressing data...");
    let compressed = compress(&mut bit_data);
    
    println!("Compression complete!");
    println!("  Original size: {} bits", compressed.metadata.original_size);
    println!("  Encoded stream size: {} bits", compressed.encoded_data.get_encoded_size());
    println!("  Base table entries: {}", compressed.base_table.len());
    println!("  Base bit positions: {:?}", compressed.base_bit_positions);
    
    // Calculate compression ratio
    let base_table_size = compressed.base_table.iter()
        .map(|(pattern, _)| pattern.len())
        .sum::<usize>();
    let total_compressed_size = compressed.encoded_data.get_encoded_size() + base_table_size;
    let compression_ratio = total_compressed_size as f64 / compressed.metadata.original_size as f64;
    println!("  Approximate compression ratio: {:.2}", compression_ratio); 
    // Decompress the data
    println!("\nDecompressing data...");
    let decompressed = decompress_file(&compressed)?;

    if let Some(analytics) = decompress_analytics(&compressed) {
        println!("\nDecompression analytics:");
        for (i, (sample, weight)) in analytics.samples.iter().zip(analytics.weights.iter()).enumerate() {
            let sample_int = sample.load_be::<u64>();
            println!("  Sample {}: {} weight, {} sample int", i, weight, sample_int);
        }
    }
    
    println!("Decompression complete!");
    println!("  Total bits: {}", decompressed.total_bits());
    println!("  Num rows: {}", decompressed.num_rows);
    println!("  Num features: {}", decompressed.num_features);
    
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
    write_bitdata_to_csv(&decompressed, &output_path, dataset.headers())?;
    
    println!("Done!");
    
    Ok(())
}

/// Convert BitData back to a CSV file
fn write_bitdata_to_csv(
    bit_data: &BitData,
    path: &str,
    headers: Option<&Vec<String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    
    // Write headers if available
    if let Some(headers) = headers {
        writeln!(file, "{}", headers.join(","))?;
    }
    
    // Write each row
    for row in 0..bit_data.num_rows {
        let mut values: Vec<String> = Vec::with_capacity(bit_data.num_features);
        for feature in 0..bit_data.num_features {
            // Extract the bits for this feature and convert to u8
            let feature_bits = bit_data.get_feature(row, feature);
            let mut value: u64 = 0;
            // Use bits_per_feature to reconstruct the value, assuming MSB first
            for (i, bit) in feature_bits.iter().enumerate() {
                if *bit {
                    value |= 1 << (bit_data.bits_per_feature - 1 - i);
                }
            }
            values.push(value.to_string());
        }
        writeln!(file, "{}", values.join(","))?;
    }
    
    Ok(())
}
