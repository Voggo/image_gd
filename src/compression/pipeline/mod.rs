pub mod interface;
pub mod implementation;

pub use interface::{
    BaseBitGroupOptimizer, CompressionParams, CompressionRunner, CondensedSampleSelector, Encoder,
    EntropyCalculator,
};
pub use implementation::{
    CompressionPipeline, CompressionPipelineBuilder, DefaultBaseBitGroupOptimizer,
    DefaultCondensedSampleSelector, DefaultEncoder, DefaultEntropyCalculator,
};
