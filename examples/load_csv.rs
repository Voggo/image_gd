use entro_gd::data_loader::Dataset;
use entro_gd::preprocessor::BitData;
use entro_gd::compression::entropy::calculate_entropy;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the CSV file with headers
    let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true)?;
    
    println!("Dataset loaded successfully!");
    println!("  Rows: {}", dataset.num_rows());
    println!("  Columns: {}", dataset.num_columns());
    
    // Convert dataset to BitData with 8 bits per feature
    let bits_per_feature = 8;
    let bit_data = BitData::from_dataset(&dataset, bits_per_feature);
    
    println!("\nBitData representation:");
    println!("{}", bit_data);
    
    // Calculate entropy for each bit position
    let entropies = calculate_entropy(&bit_data);
    
    println!("\nEntropy Analysis:");
    println!("  Total bits: {}", entropies.len());
    println!("  Entropies by bit position:");
    for (i, &entropy) in entropies.iter().enumerate() {
        println!("    Bit {}: {:.6}", i, entropy);
    }
    
    // Calculate average entropy
    let avg_entropy = entropies.iter().sum::<f64>() / entropies.len() as f64;
    println!("\n  Average entropy: {:.6}", avg_entropy);
    
    // Calculate entropy per feature
    println!("\n  Entropy per feature:");
    for feature in 0..bit_data.num_features {
        let start = feature * bits_per_feature;
        let end = start + bits_per_feature;
        let feature_entropy: f64 = entropies[start..end].iter().sum();
        let feature_avg_entropy = feature_entropy / bits_per_feature as f64;
        println!("    Feature {}: avg = {:.6}, total = {:.6}", feature, feature_avg_entropy, feature_entropy);
    }
    
    Ok(())
}

