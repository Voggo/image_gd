use crate::compression::base_bit_groups::BaseBitGroups;
use crate::compression::compress::{CompressedData, CondensedSamples, DeviationData};
use crate::preprocessor::{BitDataSet, BitDataView};

#[derive(Debug, Clone)]
pub struct CompressionParams {
    pub condensed_sample_max_bases: usize,
    pub base_optimization_patience: usize,
    pub enable_condensed_samples: bool,
}

impl CompressionParams {
    pub fn new(
        condensed_sample_max_bases: usize,
        base_optimization_patience: usize,
        enable_condensed_samples: bool,
    ) -> Self {
        CompressionParams {
            condensed_sample_max_bases,
            base_optimization_patience,
            enable_condensed_samples,
        }
    }
}

pub trait EntropyCalculator: Send + Sync {
    fn calculate(&self, bit_data: &BitDataSet) -> Vec<(usize, f64)>;
}

pub trait CondensedSampleSelector: Send + Sync {
    fn select(
        &self,
        bit_data: &BitDataSet,
        entropy: Vec<(usize, f64)>,
        max_bases: usize,
    ) -> CondensedSamples;
}

pub trait BaseBitGroupOptimizer: Send + Sync {
    fn optimize<'a>(
        &self,
        bit_data: &'a dyn BitDataView,
        entropy: Vec<(usize, f64)>,
        patience: usize,
    ) -> BaseBitGroups<'a>;
}

pub trait Encoder: Send + Sync {
    fn encode(&self, bit_data: &dyn BitDataView, base_bit_groups: &BaseBitGroups<'_>)
        -> DeviationData;
}

pub trait CompressionRunner: Send + Sync {
    fn compress(&self, bit_data: &BitDataSet) -> CompressedData;
}
