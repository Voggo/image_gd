pub mod base_bits;
pub mod compress;
pub mod entropy;
pub mod file_format;
pub mod preprocessor;

pub use compress::{
    CompressedData, CondensedSamples, DecompressAnalytics, DecompressFileData, DeviationData,
    DeviationSample, EncodeData, EncodeDataOptimized, GenCondensedSamples, SelectBases, build_compression_pipeline,
};
pub use entropy::{EntropyNaive, EntropyOptimized, calculate_entropy};
pub use file_format::{EgdFile, FORMAT_VERSION, LoadEgdFile, MAGIC_BYTES, SaveEgdFile};
pub use preprocessor::{
    BitData, BitDataInfo, BitDataSet, FeatureSpec, FeatureTransform, decode_value_from_bits,
};
