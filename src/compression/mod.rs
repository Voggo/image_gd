pub mod base_bit_selection;
pub mod compress;
pub mod entropy;
pub mod file_format;
pub mod preprocessor;

pub use compress::{
	build_compression_pipeline,
	CompressedData, CondensedSamples, DecompressAnalytics, DecompressFileData, DeviationData,
	DeviationSample, EncodeData, GenCondensedSamples, SelectBases,
};
pub use entropy::{calculate_entropy, EntropyNaive, EntropyOptimized};
pub use file_format::{EgdFile, LoadEgdFile, SaveEgdFile, FORMAT_VERSION, MAGIC_BYTES};
pub use preprocessor::{
	decode_value_from_bits, BitData, BitDataInfo, BitDataSet, FeatureSpec, FeatureTransform,
};