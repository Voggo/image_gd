pub mod base_bits;
pub mod base_selection;
pub mod base_table;
pub mod condensed_samples;
pub mod decompression;
#[path = "encoding/mod.rs"]
pub mod encoding;
pub mod entropy;
#[path = "file_format/mod.rs"]
pub mod file_format;
pub mod image_preprocessor;
pub mod preprocessor;

pub use base_bits::{
    BaseBit, BaseBitBatchGroups, BaseBitGroups, BaseBitHyperLogLogCount, BaseBitSignatureGroups,
};
pub use base_selection::{
    BaseBitImpl, BaseSelectionContext, SelectBases, SelectBasesDebug, SelectBasesOptimized,
    SelectBasesProfileAllBits,
};
pub use base_table::{BuildBaseTable, BuildSortedBaseTable, PreEncodeContext};
pub use condensed_samples::GenCondensedSamples;
pub use decompression::{
    DecompressAnalytics, DecompressFileData, DecompressRandomAccessHandle, decompress_analytics,
    decompress_file, write_bitdata_as_csv, write_bitdata_as_image, write_bitdata_to_output,
};
pub use encoding::{
    BaseTable, CompressedData, CondensedSamples, DeltaBaseTableData, DeviationData,
    DeviationSample, EncodedData, RleDeviationData, RleDeviationOffsetData,
};
pub use encoding::{
    DeltaEncodeBaseTable, DeltaEncodeBaseTableFixed, EncodeData, EncodeDataFusedDictionary,
    EncodeDataHuffman, EncodeDataOffsetRLE, EncodeDataOptimized, EncodeDataRLE,
};
pub use entropy::{
    EntropyBatched, EntropyBitScore, EntropyNaive, EntropyScoredContext, EntropyStrideSampled,
    EntropyStrideSampledBatched, calculate_entropy, calculate_entropy_stride_sampled,
};
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
    BuildBitDataSet, DEFAULT_ALIGN_ROWS_TO_WORD, FeatureSpec, FeatureTransform, FloatScalingMode,
    ImageReconstructionInfo, InferFeatureSpecs, PixelGrouping, PreprocessOptions,
    decode_value_from_bits,
};
