pub mod error;
pub mod data_loader;
pub mod compression;
pub mod timing;
pub mod filter_pipeline;

pub use compression::{
	calculate_entropy, decode_value_from_bits, BitData, BitDataInfo, BitDataSet,
	CompressedData, CondensedSamples, DecompressAnalytics, DecompressFileData, DeviationData,
	DeviationSample, EgdFile, EncodeData, EntropyNaive, FeatureSpec, FeatureTransform,
	GenCondensedSamples, LoadEgdFile, SaveEgdFile, SelectBases, FORMAT_VERSION, MAGIC_BYTES,
};
pub use data_loader::{
	ColumnData, CsvDataLoader, DataLoader, DataValue, Dataset, DatasetMetadata, FeatureDataType,
	LoadedDataset,
};
pub use error::EntroGdError;
pub use filter_pipeline::{Chain, Filter, FilterExt};
pub use timing::ScopedTimer;

pub mod prelude {
	pub use crate::{
		decode_value_from_bits, init_logging, BitDataSet, CsvDataLoader, DataLoader, DataValue,
		DecompressAnalytics, DecompressFileData, EncodeData, EntroGdError, EntropyNaive,
		FeatureDataType, Filter, FilterExt, GenCondensedSamples, LoadEgdFile, SaveEgdFile,
		ScopedTimer, SelectBases,
	};
}

use flexi_logger::{Duplicate, FileSpec, Logger, WriteMode};
use std::sync::Once;

static LOGGING_INIT: Once = Once::new();

/// Initialize logging to a file.
///
/// Configuration via environment variables:
/// - `RUST_LOG` (default: `info`)
/// - `ENTRO_GD_LOG_DIR` (default: `logs`)
/// - `ENTRO_GD_LOG_TO_STDERR` (`1`/`true` to mirror `warn+` to stderr)
///
/// Log filename: `entro_gd.log`.
///
/// Safe to call multiple times.
pub fn init_logging() {
	LOGGING_INIT.call_once(|| {
		let level_spec = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
		let log_dir = std::env::var("ENTRO_GD_LOG_DIR").unwrap_or_else(|_| "logs".to_string());
		let mirror_stderr = std::env::var("ENTRO_GD_LOG_TO_STDERR")
			.map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
			.unwrap_or(false);

		let mut logger = match Logger::try_with_env_or_str(level_spec.as_str()) {
			Ok(logger) => logger
				.log_to_file(
					FileSpec::default()
						.directory(&log_dir)
						.basename(format!("entro_gd_{}", chrono::Local::now().format("%Y%m%d_%H%M%S")))
						.suffix("log")
						.suppress_timestamp(),
				)
				.write_mode(WriteMode::Direct),
			Err(_) => {
				eprintln!("Failed to initialize logger from RUST_LOG; falling back to default logger settings");
				let mut fallback = Logger::try_with_str("info")
					.expect("default logger configuration should be valid")
					.log_to_file(
						FileSpec::default()
							.directory(&log_dir)
							.basename(format!("entro_gd_{}", chrono::Local::now().format("%Y%m%d_%H%M%S")))
							.suffix("log")
							.suppress_timestamp(),
					)
					.write_mode(WriteMode::Direct);
				if mirror_stderr {
					fallback = fallback.duplicate_to_stderr(Duplicate::Warn);
				}
				if fallback.start().is_err() {
					eprintln!("Failed to start file logger fallback");
				}
				return;
			}
		};

		if mirror_stderr {
			logger = logger.duplicate_to_stderr(Duplicate::Warn);
		}

		if logger.start().is_err() {
			eprintln!("Failed to start file logger");
		} else {
			log::info!(
				"logging initialized (dir={}, level={}, mirror_stderr={})",
				log_dir,
				level_spec,
				mirror_stderr
			);
		}
	});
}
