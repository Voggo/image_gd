pub mod base_bits;
pub mod compress;
pub mod entropy;
pub mod file_format;
pub mod preprocessor;

pub use base_bits::{BaseBit, BaseBitBatchGroups, BaseBitGroups, BaseBitSignatureGroups};
pub use compress::{
    CompressedData, CondensedSamples, DecompressAnalytics, DecompressFileData, DeviationData,
    DeviationSample, EncodeData, EncodeDataOptimized, GenCondensedSamples, SelectBases,
    SelectBasesOptimized, build_compression_pipeline, build_compression_pipeline_optimized,
    build_compression_pipeline_with_preprocessing,
};
pub use entropy::{EntropyNaive, EntropyOptimized, calculate_entropy};
pub use file_format::{EgdFile, FORMAT_VERSION, LoadEgdFile, MAGIC_BYTES, SaveEgdFile};
pub use preprocessor::{
    BitData, BitDataInfo, BitDataSet, BuildBitDataSet, FeatureSpec, FeatureTransform,
    FloatScalingMode, InferFeatureSpecs, PreprocessOptions, decode_value_from_bits,
};
