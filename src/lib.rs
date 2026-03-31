pub mod compression;
pub mod data_loader;
pub mod error;
pub mod filter_pipeline;
#[cfg(feature = "experiment-runner")]
pub mod pipeline_profiles;
#[cfg(feature = "experiment-runner")]
pub mod pipeline_runner;
pub mod timing;
mod utils;

pub type BitOrder = bitvec::prelude::Lsb0;
pub type BitStream = bitvec::prelude::BitVec<usize, BitOrder>;
pub type BitView = bitvec::prelude::BitSlice<usize, BitOrder>;

pub use compression::{
    BaseBitImpl, BitData, BitDataCompressionInfo, BitDataInfo, BitDataReconstructionInfo,
    BitDataSet, BuildBitDataSet, BuildImageBitDataSet, CompressedData, CondensedSamples,
    DecompressAnalytics, DecompressFileData, DecompressRowsData, DeviationData, DeviationSample,
    EgdFile, EncodeData, EncodeDataFusedDictionary, EncodeDataHuffman, EncodeDataHuffmanBaseIdOnly,
    EncodeDataOptimized, EncodeDataRLE, EncodedData, EntropyNaive, EntropyOptimized,
    EntropyStrideSampled, FORMAT_VERSION, FeatureSpec, FeatureTransform, FloatScalingMode,
    GenCondensedSamples, IMAGE_FORMAT_VERSION, IMAGE_MAGIC_BYTES, IgdFile, ImageColorModel,
    ImageColorSpace, ImageGroupingTransform, ImageReconstructionInfo, InferFeatureSpecs,
    LoadEgdFile, LoadIgdFile, MAGIC_BYTES, PreprocessOptions, RleDeviationData, SaveEgdFile,
    SaveIgdFile, SelectBases, SelectBasesDebug, SelectBasesOptimized, SelectBasesProfileAllBits,
    calculate_entropy, calculate_entropy_stride_sampled, decode_value_from_bits,
    decompress_egd_to_csv, decompress_igd_to_image, load_and_decompress_egd,
    load_and_decompress_igd, write_bitdata_as_csv, write_bitdata_as_image, write_bitdata_to_output,
};
pub use data_loader::{
    ColumnData, CsvDataLoader, DataLoader, DataValue, Dataset, DatasetMetadata, FeatureDataType,
    FloatStorage, LoadedDataset, MissingValuePolicy,
};
pub use error::EntroGdError;
pub use filter_pipeline::{Chain, Filter, FilterExt};
#[cfg(feature = "experiment-runner")]
pub use pipeline_profiles::{
    ConfigBaseBitImpl, ConfigEncodeImpl, ConfigImageColorModel, ConfigImageColorSpace,
    ConfigImageGroupingTransform, ConfigMissingValuePolicy, ConfigSelectBasesImpl,
    CsvPipelineProfile, CsvPipelineProfileConfig, CsvProfileGroupConfig, EncodeImpl,
    ExperimentRunnerConfigFile, ImageBuildConfig, ImageBuildSweepConfig, ImagePipelineProfile,
    ImagePipelineProfileConfig, ImageProfileGroupConfig, IntegerRangeU8, IntegerRangeU32,
    IntegerRangeUsize, IntegerSweepU8, IntegerSweepU32, IntegerSweepUsize, PipelineProfileSet,
    SelectBasesImpl,
};
#[cfg(feature = "experiment-runner")]
pub use pipeline_runner::{
    CompressedSizeBreakdownBits, CompressionStageDurationsMs, ExperimentRecord, ExperimentReport,
    ExperimentRunOptions, InputKind, collect_input_files, detect_input_kind,
    run_experiments_on_path, write_report_csv,
};
pub use timing::ScopedTimer;

pub mod prelude {
    pub use crate::{
        BaseBitImpl, BuildBitDataSet, BuildImageBitDataSet, DecompressAnalytics,
        DecompressFileData, DecompressRowsData, EncodeData, EncodeDataFusedDictionary,
        EncodeDataHuffman, EncodeDataHuffmanBaseIdOnly, EncodeDataOptimized, EncodeDataRLE,
        EntropyNaive, EntropyOptimized, EntropyStrideSampled, Filter, FilterExt, FloatScalingMode,
        GenCondensedSamples, ImageColorModel, ImageColorSpace, ImageGroupingTransform,
        InferFeatureSpecs, LoadEgdFile, LoadIgdFile, PreprocessOptions, SaveEgdFile, SaveIgdFile,
        SelectBases, SelectBasesDebug, SelectBasesOptimized, SelectBasesProfileAllBits,
    };
}

use std::sync::Once;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

static LOGGING_INIT: Once = Once::new();

/// RAII handle for the async logging worker.
///
/// Keep this value alive for as long as you want logging to remain active.
/// Dropping it flushes pending log events.
pub struct LogHandle {
    _guard: WorkerGuard,
}

/// Initialize logging to a file.
///
/// Configuration via environment variables:
/// - `LOG` or `RUST_LOG` (default: `info`)
/// - `ENTRO_GD_LOG_DIR` (default: `logs`)
/// - `ENTRO_GD_LOG_TO_STDERR` (`1`/`true` to mirror `warn+` to stderr)
///
/// Log filename: `entro_gd_YYYYMMDD_HHMMSS.log`.
///
/// Safe to call multiple times. Only the first successful call initializes
/// the global subscriber and returns a [`LogHandle`].
pub fn init_logging() -> Option<LogHandle> {
    let mut log_handle = None;
    LOGGING_INIT.call_once(|| {
        let level_spec = std::env::var("LOG")
            .or_else(|_| std::env::var("RUST_LOG"))
            .unwrap_or_else(|_| "info".to_string());
        let log_dir = std::env::var("ENTRO_GD_LOG_DIR").unwrap_or_else(|_| "logs".to_string());
        let mirror_stderr = std::env::var("ENTRO_GD_LOG_TO_STDERR")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);

        if std::fs::create_dir_all(&log_dir).is_err() {
            tracing::warn!(log_dir = %log_dir, "failed to create log directory");
            return;
        }

        let file_name = format!(
            "entro_gd_{}.log",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        );
        let file_appender = tracing_appender::rolling::never(&log_dir, file_name);
        let (file_writer, guard) = tracing_appender::non_blocking(file_appender);

        let init_result = if mirror_stderr {
            let env_filter =
                EnvFilter::try_new(level_spec.clone()).unwrap_or_else(|_| EnvFilter::new("info"));
            let file_layer = fmt::layer().with_ansi(false).with_writer(file_writer);
            let stderr_layer = fmt::layer()
                .with_ansi(true)
                .with_writer(std::io::stderr)
                .with_filter(LevelFilter::WARN);
            tracing::subscriber::set_global_default(
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(file_layer)
                    .with(stderr_layer),
            )
        } else {
            let env_filter =
                EnvFilter::try_new(level_spec.clone()).unwrap_or_else(|_| EnvFilter::new("info"));
            let file_layer = fmt::layer().with_ansi(false).with_writer(file_writer);
            tracing::subscriber::set_global_default(
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(file_layer),
            )
        };

        if init_result.is_err() {
            tracing::warn!("failed to initialize tracing subscriber");
            return;
        }

        log_handle = Some(LogHandle { _guard: guard });

        tracing::info!(
            log_dir = %log_dir,
            level = %level_spec,
            mirror_stderr,
            "logging initialized"
        );
    });
    log_handle
}
