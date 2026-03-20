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
    BatchedBaseBitImpl, SelectBases, SelectBasesOptimized,
};
pub use compress::{
    CompressedData, CondensedSamples, DeviationData, DeviationSample, EncodedData, RleDeviationData,
};
pub use condensed_samples::GenCondensedSamples;
pub use decompression::{
    DecompressAnalytics, DecompressFileData, DecompressRowsData, decompress_analytics,
    decompress_file, write_bitdata_as_csv, write_bitdata_as_image, write_bitdata_to_output,
};
pub use encoding::{
    EncodeData, EncodeDataHuffman, EncodeDataHuffmanBaseIdOnly, EncodeDataOptimized,
    EncodeDataRLE,
};
pub use entropy::{EntropyNaive, EntropyOptimized, calculate_entropy};
pub use file_format::{
    EgdFile, FORMAT_VERSION, IMAGE_FORMAT_VERSION, IMAGE_MAGIC_BYTES, IgdFile, LoadEgdFile,
    LoadIgdFile, MAGIC_BYTES, SaveEgdFile, SaveIgdFile, decompress_egd_to_csv,
    decompress_igd_to_image, load_and_decompress_egd, load_and_decompress_igd,
};
pub use image_preprocessor::{
    BuildImageBitDataSet, ImageColorModel, ImageColorSpace, ImageGroupingTransform,
};
pub use preprocessor::{
    BitData, BitDataCompressionInfo, BitDataInfo, BitDataReconstructionInfo, BitDataSet,
    FeatureSpec, FeatureTransform, FloatScalingMode, ImageReconstructionInfo, PreprocessOptions,
    decode_value_from_bits,
};
pub use tabular_preprocessor::{BuildBitDataSet, InferFeatureSpecs};
