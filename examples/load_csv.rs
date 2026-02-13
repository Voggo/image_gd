use entro_gd::data_loader::{CsvDataLoader, DataLoader};
use entro_gd::error::EntroGdError;
use entro_gd::preprocessor::BitDataSet;
use entro_gd::compression::entropy::calculate_entropy;

fn main() -> Result<(), EntroGdError> {
    // Load the CSV file with headers
    let loader = CsvDataLoader::new(true);
    let dataset = loader.load("data/data-10000-8-int.csv")?.dataset;
    
    println!("Dataset loaded successfully!");
    println!("  Rows: {}", dataset.num_rows());
    println!("  Columns: {}", dataset.num_columns());
    
    // Convert dataset to BitData based on column data types
    let bit_data = BitDataSet::from_dataset(&dataset)?;
    
    println!("\nBitData representation:");
    println!("{}", bit_data);
    
    // Calculate entropy for each bit position
    let entropies = calculate_entropy(&bit_data);
    
    println!("\nEntropy Analysis:");
    println!("  Total bits: {}", entropies.len());
    println!("  Entropies by bit position:");
    for (i, entropy) in entropies.iter() {
        println!("    Bit {}: {:.6}", i, entropy);
    }
    
    // Calculate average entropy
    let avg_entropy = entropies.iter().map(|(_, entropy)| entropy).sum::<f64>() / entropies.len() as f64;
    println!("\n  Average entropy: {:.6}", avg_entropy);
    
    // Calculate entropy per feature
    println!("\n  Entropy per feature:");
    for feature in 0..bit_data.info.num_features() {
        let start = bit_data.info.feature_offset(feature);
        let end = start + bit_data.info.feature_bits(feature);
        let feature_entropy: f64 = entropies[start..end].iter().map(|(_, entropy)| entropy).sum();
        let feature_avg_entropy = feature_entropy / bit_data.info.feature_bits(feature) as f64;
        println!("    Feature {}: avg = {:.6}, total = {:.6}", feature, feature_avg_entropy, feature_entropy);
    }
    
    Ok(())
}

