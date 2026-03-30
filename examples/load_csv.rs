use entro_gd::data_loader::{CsvDataLoader, DataLoader};
use entro_gd::{BitDataSet, EntroGdError, calculate_entropy, init_logging};

fn main() -> Result<(), EntroGdError> {
    let _log_handle = init_logging();

    // Load the CSV file with headers
    let loader = CsvDataLoader::new(true);
    let dataset = loader.load("data/tabular/data-10000-8-int.csv")?.dataset;

    tracing::info!("Dataset loaded successfully!");
    tracing::info!("  Rows: {}", dataset.num_rows());
    tracing::info!("  Columns: {}", dataset.num_columns());

    // Convert dataset to BitData based on column data types
    let bit_data = BitDataSet::from_dataset(&dataset)?;

    tracing::info!("\nBitData representation:");
    tracing::info!("{}", bit_data);

    // Calculate entropy for each bit position
    let entropies = calculate_entropy(&bit_data);

    tracing::info!("\nEntropy Analysis:");
    tracing::info!("  Total bits: {}", entropies.len());
    tracing::info!("  Entropies by bit position:");
    for (i, entropy) in entropies.iter() {
        tracing::info!("    Bit {}: {:.6}", i, entropy);
    }

    // Calculate average entropy
    let avg_entropy =
        entropies.iter().map(|(_, entropy)| entropy).sum::<f64>() / entropies.len() as f64;
    tracing::info!("\n  Average entropy: {:.6}", avg_entropy);

    // Calculate entropy per feature
    tracing::info!("\n  Entropy per feature:");
    for feature in 0..bit_data.info.num_features() {
        let start = bit_data.info.feature_offset(feature);
        let end = start + bit_data.info.feature_bits(feature);
        let feature_entropy: f64 = entropies[start..end]
            .iter()
            .map(|(_, entropy)| entropy)
            .sum();
        let feature_avg_entropy = feature_entropy / bit_data.info.feature_bits(feature) as f64;
        tracing::info!(
            "    Feature {}: avg = {:.6}, total = {:.6}",
            feature,
            feature_avg_entropy,
            feature_entropy
        );
    }

    Ok(())
}
