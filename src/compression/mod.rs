pub mod base_bits;
pub mod base_selection;
pub mod compress;
pub mod condensed_samples;
pub mod decompression;
pub mod encoding;
pub mod entropy;
pub mod file_format;
pub mod image_preprocessor;
pub mod preprocessor;
pub mod tabular_preprocessor;

pub use base_bits::{BaseBit, BaseBitBatchGroups, BaseBitGroups, BaseBitSignatureGroups};
pub use base_selection::{
    SelectBases, SelectBasesOptimizedv1, SelectBasesOptimizedv2, SelectBasesOptimizedv3,
};
pub use compress::{
    CompressedData, CondensedSamples, DeviationData, DeviationSample, EncodedData,
    RleDeviationData, build_compression_pipeline, build_compression_pipeline_optimized,
    build_compression_pipeline_with_preprocessing, build_image_compression_pipeline,
};
pub use condensed_samples::GenCondensedSamples;
pub use decompression::{
    DecompressAnalytics, DecompressFileData, DecompressRowsData, decompress_analytics,
    decompress_file,
};
pub use encoding::{EncodeData, EncodeDataOptimized, EncodeDataRLE};
pub use entropy::{EntropyNaive, EntropyOptimized, calculate_entropy};
pub use file_format::{
    EgdFile, FORMAT_VERSION, IMAGE_FORMAT_VERSION, IMAGE_MAGIC_BYTES, IgdFile, LoadEgdFile,
    LoadIgdFile, MAGIC_BYTES, SaveEgdFile, SaveIgdFile,
};
pub use image_preprocessor::{BuildImageBitDataSet, ImageColorSpace};
pub use preprocessor::{
    BitData, BitDataCompressionInfo, BitDataInfo, BitDataReconstructionInfo, BitDataSet,
    FeatureSpec, FeatureTransform, FloatScalingMode, ImageReconstructionInfo, PreprocessOptions,
    decode_value_from_bits,
};
pub use tabular_preprocessor::{
    BuildBitDataSet, InferFeatureSpecs,
};
