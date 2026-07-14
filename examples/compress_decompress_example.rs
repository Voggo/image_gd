use image_gd::data_loader::{CsvDataLoader, DataLoader, DataValue, FloatStorage};
use image_gd::prelude::*;
use image_gd::{
    BitDataSet, EntroGdError, ScopedTimer, decode_value_from_bits, init_logging,
    write_bitdata_as_csv,
};
use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), EntroGdError> {
    unsafe {
        env::set_var("ENTRO_GD_LOG_TO_STDERR", "1");
        env::set_var("RUST_LOG", "debug");
    }
    let _log_handle = init_logging();
    let _timer = ScopedTimer::info("Total compression-decompression process");

    let args: Vec<String> = env::args().collect();
    let input_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "data/tabular/aarhus-citylab.csv".to_string()
    };

    let input = Path::new(&input_path);
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset");
    let parent = input.parent().unwrap_or_else(|| Path::new("."));

    let egd_path = parent.join(format!("{}.egd", stem));
    let decompressed_path = parent.join(format!("{}-decompressed.csv", stem));

    tracing::info!("Loading CSV: {}", input.display());
    let loader = CsvDataLoader::new(true).with_float_storage(FloatStorage::F32);
    let loaded = loader.load(&input_path)?;
    let dataset = loaded.dataset;
    let metadata = loaded.metadata;

    tracing::info!(
        "Loaded: {} rows, {} columns",
        dataset.num_rows(),
        dataset.num_columns()
    );

    let bit_data = BitDataSet::from_dataset(&dataset)?;
    let original_size_bits = bit_data.data.total_bits();

    let compression_pipeline = Entropy {}
        .then(GenCondensedSamples { m_max: 100 })
        .then(SelectBasesAdaptive {
            width_decay: 0.45,
            patience: 5,
            base_bit_impl: BaseBitImpl::HyperLogLogCount,
        })
        .then(BuildSortedBaseTable {})
        .then(EncodeData {})
        .then(DeltaEncodeBaseTableFixed {});

    tracing::info!("Compressing...");
    let compressed = compression_pipeline.process(bit_data)?;

    tracing::info!("Saving EGD: {}", egd_path.display());
    let compressed_path = SaveEgdFile {
        output_path: egd_path,
    }
    .process(compressed.clone())?;

    let egd_size_bytes = fs::metadata(&compressed_path)?.len() as usize;
    let egd_size_bits = egd_size_bytes.saturating_mul(8);

    tracing::info!("Compression complete:");
    tracing::info!("  Original: {} bits", original_size_bits);
    tracing::info!(
        "  Compressed file: {} bytes ({} bits)",
        egd_size_bytes,
        egd_size_bits
    );
    tracing::info!(
        "  Encoded stream: {} bits",
        compressed.encoded_data.get_encoded_size()
    );
    tracing::info!("  Base table entries: {}", compressed.base_table.len());
    tracing::info!(
        "  Base bit positions: {:?}",
        compressed.layout.selected_base_bit_positions()
    );

    if original_size_bits > 0 {
        tracing::info!(
            "  Compression ratio: {:.2}% ({:.1} KB on disk)",
            100.0 * egd_size_bits as f64 / original_size_bits as f64,
            egd_size_bytes as f64 / 1024.0
        );
    }

    tracing::info!("Decompressing...");
    let compressed_data = LoadEgdFile {}.process(compressed_path.clone())?;
    let decompressed = DecompressFileData {}.process(compressed_data.clone())?;

    tracing::info!(
        "Decompressed: {} rows, {} features",
        decompressed.data.num_rows,
        decompressed.info.num_features()
    );

    if let Some(analytics) = (DecompressAnalytics {}).process(compressed_data.clone())? {
        tracing::debug!("Analytics: {} condensed samples", analytics.samples.len());
        for (i, (sample, weight)) in analytics
            .samples
            .iter()
            .zip(analytics.weights.iter())
            .enumerate()
        {
            tracing::trace!("  Sample {}: {} weight", i, weight);
            for feature_idx in 0..compressed_data.metadata.num_features() {
                let feature_start = compressed_data.metadata.feature_offset(feature_idx);
                let feature_end =
                    feature_start + compressed_data.metadata.feature_bits(feature_idx);
                let feature_bits = &sample[feature_start..feature_end];
                let spec = compressed_data.metadata.feature_spec(feature_idx);

                let formatted = match decode_value_from_bits(feature_bits, spec) {
                    DataValue::Unsigned(v) => v.to_string(),
                    DataValue::Signed(v) => v.to_string(),
                    DataValue::F32(v) => v.to_string(),
                    DataValue::F64(v) => v.to_string(),
                };
                tracing::trace!("    Feature {}: {}", feature_idx, formatted);
            }
        }
    }

    tracing::info!("Writing CSV: {}", decompressed_path.display());
    write_bitdata_as_csv(
        &decompressed,
        &decompressed_path,
        metadata.headers.as_deref(),
    )?;

    tracing::info!("Done.");
    Ok(())
}
