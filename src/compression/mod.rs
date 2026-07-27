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

pub use base_bits::{BaseBit, BaseBitGroups, BaseBitHyperLogLogCount};
pub use base_selection::{
    BaseBitImpl, BaseSelectionContext, SelectBases, SelectBasesAdaptive, SelectBasesDebug,
    SelectBasesProfileAllBits, SelectBasesThreshold,
};
pub use base_table::{BuildBaseTable, BuildSortedBaseTable, PreEncodeContext};
pub use condensed_samples::GenCondensedSamples;
pub use decompression::{
    DecompressAnalytics, DecompressFileData, DecompressRandomAccessHandle, decompress_analytics,
    decompress_file, write_bitdata_as_csv, write_bitdata_as_image, write_bitdata_to_output,
};
pub use encoding::{
    BaseTable, CompressedData, CondensedSamples, DeltaBaseTableData, DeviationData,
    DeviationSample, EncodedData, RleDeviationOffsetData,
};
pub use encoding::{
    DeltaEncodeBaseTable, DeltaEncodeBaseTableFixed, EncodeData, EncodeDataHuffman,
    EncodeDataOffsetRLE,
};
pub use entropy::{Entropy, EntropyBitScore, EntropyScoredContext, calculate_entropy};
pub use file_format::{
    DecodeDeltaBaseTable, EgdFile, FORMAT_VERSION, IMAGE_FORMAT_VERSION, IMAGE_MAGIC_BYTES,
    IgdFile, LoadEgdFile, LoadIgdFile, MAGIC_BYTES, SaveEgdFile, SaveIgdFile,
    decompress_egd_to_csv, decompress_igd_to_image, load_and_decompress_egd,
    load_and_decompress_igd,
};
pub use image_preprocessor::{
    BuildImageBitDataSet, ImageColorModel, ImageColorSpace, ImageGroupingTransform, OpenImage,
};
pub use preprocessor::{
    BitData, BitDataCompressionInfo, BitDataInfo, BitDataReconstructionInfo, BitDataSet,
    BuildBitDataSet, DEFAULT_ALIGN_ROWS_TO_WORD, FeatureDataType, FeatureSpec, FeatureTransform,
    FloatScalingMode, ImageReconstructionInfo, PixelGrouping, PreprocessOptions,
    reconstruct_feature_value,
};
