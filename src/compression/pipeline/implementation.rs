use crate::compression::base_bit_groups::BaseBitGroups;
use crate::compression::compress::{
    CompressedData, CondensedSamples, DeviationData, ExtendedBitData,
};
use crate::compression::entropy;
use crate::compression::pipeline::interface::{
    BaseBitGroupOptimizer, CompressionParams, CompressionRunner, CondensedSampleSelector, Encoder,
    EntropyCalculator,
};
use crate::preprocessor::{BitDataSet, BitDataView};
use crate::timing::ScopedTimer;
use bitvec::prelude::*;
use log::debug;

pub struct DefaultEntropyCalculator;

impl EntropyCalculator for DefaultEntropyCalculator {
    fn calculate(&self, bit_data: &BitDataSet) -> Vec<(usize, f64)> {
        entropy::calculate_entropy(bit_data)
    }
}

pub struct DefaultCondensedSampleSelector;

impl DefaultCondensedSampleSelector {
    /// Organize entropy values by feature and return as flattened vector.
    /// Orders bits by alternating through features (feature 1, 2, 3, ... n, repeat).
    /// Within each feature, bits are sorted from lowest to highest entropy.
    /// Zero entropy bits are placed first.
    fn organize_entropy_by_feature(
        &self,
        entropy: Vec<(usize, f64)>,
        bit_data: &dyn BitDataView,
    ) -> Vec<(usize, f64)> {
        let num_features = bit_data.num_features();
        let mut entropy_by_feature: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_features];
        let mut zero_entropy = Vec::new();

        for (bit_pos, entropy_val) in entropy {
            if let Some(feature_idx) = bit_data.feature_index_for_bit(bit_pos) {
                if entropy_val == 0.0 {
                    zero_entropy.push((bit_pos, entropy_val));
                } else {
                    entropy_by_feature[feature_idx].push((bit_pos, entropy_val));
                }
            }
        }

        for feature_entropy in &mut entropy_by_feature {
            feature_entropy.sort_by(|a, b| a.1.total_cmp(&b.1));
        }

        let mut result = zero_entropy;
        let max_bits_per_feature = entropy_by_feature
            .iter()
            .map(|f| f.len())
            .max()
            .unwrap_or(0);

        for i in 0..max_bits_per_feature {
            for feature_idx in 0..num_features {
                if i < entropy_by_feature[feature_idx].len() {
                    result.push(entropy_by_feature[feature_idx][i]);
                }
            }
        }

        result
    }

    fn get_condensed_sample<'a>(
        &self,
        bit_data: &'a dyn BitDataView,
        condensed_bit_groups: &mut BaseBitGroups<'a>,
        entropy: Vec<(usize, f64)>,
        m_max: usize,
    ) -> CondensedSamples {
        // IMPORTANT: there is a difference between the chunks function used in this function and the get_chunk function in BitDataView.
        // The chunks function is used to iterate over the bits of a sample in groups of 64 bits,
        // while the get_chunk function retrieves the entire chunk for a given sample index.
        // The chunks function is used to calculate the deviation from the base pattern for each sample,
        // while the get_chunk function is used to retrieve the original chunk for each sample
        // when calculating the deviation sum and creating the condensed sample.

        let mut samples = Vec::new();
        let mut weights = Vec::new();

        let organized_entropy_by_feature = self.organize_entropy_by_feature(entropy, bit_data);

        for &(bit_position, entropy_val) in organized_entropy_by_feature.iter() {
            if condensed_bit_groups.get_num_bases() >= m_max {
                break;
            }
            if entropy_val == 0.0 {
                condensed_bit_groups.add_constant_bit_position(bit_position);
            } else {
                condensed_bit_groups.add_bit_position(bit_position);
            }
        }
        let bases = condensed_bit_groups.get_bases();
        let base_mask_chunked_int = condensed_bit_groups
            .get_base_bit_mask()
            .chunks(64)
            .map(|chunk| chunk.load_be::<u64>())
            .collect::<Vec<u64>>();

        for (base_group, base) in condensed_bit_groups.get_groups().iter().zip(bases.iter()) {
            // Convert Vec<u64> back to BitVec<usize, Msb0>
            let mut condensed_bitvec = BitVec::<usize, Msb0>::with_capacity(bit_data.chunk_size());
            let mut acumulators = vec![0u128; base_mask_chunked_int.len()];

            for &sample_idx in base_group.iter() {
                let sample = bit_data.get_chunk(sample_idx);
                for (i, (sample_chunk, base_mask_chunk)) in sample
                    .chunks(64)
                    .zip(base_mask_chunked_int.iter())
                    .enumerate()
                {
                    let chunk_deviation = sample_chunk.load_be::<u64>() & !base_mask_chunk;
                    acumulators[i] += chunk_deviation as u128;
                }
            }
            let base_chunks = base
                .0
                .chunks(64)
                .map(|chunk| chunk.load_be::<u64>())
                .collect::<Vec<u64>>();
            let averaged_chunks: Vec<u64> = acumulators
                .iter()
                .zip(base_chunks.iter())
                .map(|(&acc, &base)| base + (acc as f64 / base_group.len() as f64) as u64)
                .collect();

            // Add all complete 64-bit chunks
            averaged_chunks.iter().for_each(|&chunk| {
                condensed_bitvec.extend_from_bitslice(BitSlice::<u64, Msb0>::from_element(&chunk));
            });
            condensed_bitvec.truncate(bit_data.chunk_size());
            samples.push(condensed_bitvec);
            weights.push(base_group.len());
        }
        CondensedSamples { samples, weights }
    }
}

impl CondensedSampleSelector for DefaultCondensedSampleSelector {
    fn select(
        &self,
        bit_data: &BitDataSet,
        entropy: Vec<(usize, f64)>,
        max_bases: usize,
    ) -> CondensedSamples {
        let mut condensed_base_bit_groups = BaseBitGroups::new(bit_data);
        self.get_condensed_sample(bit_data, &mut condensed_base_bit_groups, entropy, max_bases)
    }
}

pub struct DefaultBaseBitGroupOptimizer;

impl DefaultBaseBitGroupOptimizer {
    pub(crate) fn calculate_compressed_size(
        &self,
        bit_data: &dyn BitDataView,
        base_bit_groups: &BaseBitGroups<'_>,
    ) -> usize {
        let n_b = base_bit_groups.get_num_bases(); // number of bases
        let l_b = base_bit_groups.get_num_bits_per_base(); // bits per base
        let n = bit_data.num_rows(); // number of rows/samples
        let m = 0usize; // number of bases in condensed sample
        let chunk_size = bit_data.chunk_size();

        // Ensure l_b doesn't exceed chunk_size to prevent underflow
        let l_b = l_b.min(chunk_size);
        let l_d = chunk_size - l_b; // bits per deviation
        let l_id = (n_b as f64).log2().ceil() as usize; // bits per id
        let s_params = 0usize; // size of additional parameters in bits (not implemented yet)

        n_b * l_b + (n + m) * (l_d + l_id) + m * 0 + s_params
    }

    fn optimize_with_tau<'a>(
        &self,
        bit_data: &'a dyn BitDataView,
        mut entropy: Vec<(usize, f64)>,
        tau: usize,
    ) -> BaseBitGroups<'a> {
        let mut tau_count = 0usize; // count of non-improving additions
        let mut base_bit_groups = BaseBitGroups::new(bit_data);

        entropy.sort_by(|a, b| a.1.total_cmp(&b.1));
        while let Some((bit_pos, 0.0)) = entropy.first() {
            base_bit_groups.add_constant_bit_position(*bit_pos);
            entropy.remove(0);
        }

        let mut best_base_bit_groups = base_bit_groups.clone();
        let mut best_compressed_size =
            self.calculate_compressed_size(bit_data, &best_base_bit_groups);
        let mut trial_base_bit_groups = base_bit_groups.clone();
        for &(bit_position, _) in entropy.iter() {
            trial_base_bit_groups.add_bit_position(bit_position);
            let trial_compressed_size =
                self.calculate_compressed_size(bit_data, &trial_base_bit_groups);
            debug!(
                "Trial bit position: {}, Trial compressed size: {}, Best compressed size: {}",
                bit_position, trial_compressed_size, best_compressed_size
            );
            if trial_compressed_size < best_compressed_size {
                best_compressed_size = trial_compressed_size;
                best_base_bit_groups = trial_base_bit_groups.clone();
                tau_count = 0;
            } else {
                tau_count += 1;
            }
            if tau_count >= tau {
                break;
            }
        }
        best_base_bit_groups
    }
}

impl BaseBitGroupOptimizer for DefaultBaseBitGroupOptimizer {
    fn optimize<'a>(
        &self,
        bit_data: &'a dyn BitDataView,
        entropy: Vec<(usize, f64)>,
        patience: usize,
    ) -> BaseBitGroups<'a> {
        self.optimize_with_tau(bit_data, entropy, patience)
    }
}

pub struct DefaultEncoder;

impl DefaultEncoder {
    /// Encode the actual data using the base bit groups and base table.
    fn encode_data(
        &self,
        bit_data: &dyn BitDataView,
        base_bit_groups: &BaseBitGroups<'_>,
    ) -> DeviationData {
        let mut encoded_bit_stream = BitVec::<usize, Msb0>::new();
        let base_bit_mask = base_bit_groups.get_base_bit_mask();

        // Calculate l_id: number of bits needed to represent base group IDs
        // If there are no bases, we still need at least 1 bit to represent ID 0
        let num_bases = base_bit_groups.get_num_bases();
        let l_id = {
            let bits = (num_bases as f64).log2().ceil() as usize;
            if bits == 0 { 1 } else { bits }
        };

        let chunk_size = bit_data.chunk_size();
        let num_bits_per_base = base_bit_groups.get_num_bits_per_base();

        let num_deviation_bits = chunk_size.saturating_sub(num_bits_per_base);

        let num_rows = bit_data.num_rows();
        let mut row_to_group_id = vec![0usize; num_rows];
        for (id, group) in base_bit_groups.get_groups().iter().enumerate() {
            for &row in group.iter() {
                row_to_group_id[row] = id;
            }
        }

        for row in 0..num_rows {
            let id = row_to_group_id[row];

            let chunk = bit_data.get_chunk(row);
            for bit_pos in 0..chunk_size {
                if !base_bit_mask[bit_pos] {
                    encoded_bit_stream.push(chunk[bit_pos]);
                }
            }
            if l_id > 0 {
                for shift in (0..l_id).rev() {
                    encoded_bit_stream.push(((id >> shift) & 1) == 1);
                }
            }
        }

        DeviationData::new(
            encoded_bit_stream,
            bit_data.num_rows(),
            num_deviation_bits,
            l_id,
        )
    }
}

impl Encoder for DefaultEncoder {
    fn encode(
        &self,
        bit_data: &dyn BitDataView,
        base_bit_groups: &BaseBitGroups<'_>,
    ) -> DeviationData {
        self.encode_data(bit_data, base_bit_groups)
    }
}

pub struct CompressionPipeline {
    params: CompressionParams,
    entropy_calculator: Box<dyn EntropyCalculator>,
    condensed_sample_selector: Box<dyn CondensedSampleSelector>,
    base_optimizer: Box<dyn BaseBitGroupOptimizer>,
    encoder: Box<dyn Encoder>,
}

impl CompressionPipeline {
    pub fn new(params: CompressionParams) -> Self {
        CompressionPipeline {
            params,
            entropy_calculator: Box::new(DefaultEntropyCalculator),
            condensed_sample_selector: Box::new(DefaultCondensedSampleSelector),
            base_optimizer: Box::new(DefaultBaseBitGroupOptimizer),
            encoder: Box::new(DefaultEncoder),
        }
    }

    pub fn builder(params: CompressionParams) -> CompressionPipelineBuilder {
        CompressionPipelineBuilder::new(params)
    }

    pub fn params(&self) -> &CompressionParams {
        &self.params
    }

    pub fn compress(&self, bit_data: &BitDataSet) -> CompressedData {
        <Self as CompressionRunner>::compress(self, bit_data)
    }
}

impl CompressionRunner for CompressionPipeline {
    fn compress(&self, bit_data: &BitDataSet) -> CompressedData {
        let _total_timer = ScopedTimer::info("compression pipeline");

        let entropy = {
            let _timer = ScopedTimer::debug("entropy calculation");
            self.entropy_calculator.calculate(bit_data)
        };
        debug!("Initial entropy per bit position: {:?}", entropy);

        let condensed_samples =
            if self.params.enable_condensed_samples && self.params.condensed_sample_max_bases > 0 {
                let _timer = ScopedTimer::debug("condensed sample selection");
                self.condensed_sample_selector.select(
                    bit_data,
                    entropy.clone(),
                    self.params.condensed_sample_max_bases,
                )
            } else {
                CondensedSamples {
                    samples: Vec::new(),
                    weights: Vec::new(),
                }
            };

        let extended = ExtendedBitData::new(bit_data, condensed_samples.samples.clone());

        let best_base_bit_groups = {
            let _timer = ScopedTimer::debug("base bit group optimization");
            self.base_optimizer.optimize(
                &extended,
                entropy,
                self.params.base_optimization_patience,
            )
        };

        let base_table = best_base_bit_groups.get_bases();

        debug!(
            "Optimized to {} base groups with {} bits per base.",
            best_base_bit_groups.get_num_bases(),
            best_base_bit_groups.get_num_bits_per_base()
        );
        debug!("Base table size: {}", base_table.len());
        debug!(
            "Base bit positions: {:?}",
            best_base_bit_groups.get_base_bit_positions()
        );

        let encoded_data = {
            let _timer = ScopedTimer::debug("encoding");
            self.encoder.encode(&extended, &best_base_bit_groups)
        };
        let base_bit_positions = best_base_bit_groups.get_base_bit_positions().to_owned();

        CompressedData {
            encoded_data,
            condensed_sample_weights: if self.params.enable_condensed_samples {
                Some(condensed_samples.weights)
            } else {
                None
            },
            base_table,
            base_bit_positions,
            metadata: bit_data.info.clone(),
        }
    }
}

pub struct CompressionPipelineBuilder {
    pipeline: CompressionPipeline,
}

impl CompressionPipelineBuilder {
    pub fn new(params: CompressionParams) -> Self {
        CompressionPipelineBuilder {
            pipeline: CompressionPipeline::new(params),
        }
    }

    pub fn with_entropy_calculator(
        mut self,
        entropy_calculator: impl EntropyCalculator + 'static,
    ) -> Self {
        self.pipeline.entropy_calculator = Box::new(entropy_calculator);
        self
    }

    pub fn with_condensed_sample_selector(
        mut self,
        condensed_sample_selector: impl CondensedSampleSelector + 'static,
    ) -> Self {
        self.pipeline.condensed_sample_selector = Box::new(condensed_sample_selector);
        self
    }

    pub fn with_base_optimizer(
        mut self,
        base_optimizer: impl BaseBitGroupOptimizer + 'static,
    ) -> Self {
        self.pipeline.base_optimizer = Box::new(base_optimizer);
        self
    }

    pub fn with_encoder(mut self, encoder: impl Encoder + 'static) -> Self {
        self.pipeline.encoder = Box::new(encoder);
        self
    }

    pub fn build(self) -> CompressionPipeline {
        self.pipeline
    }
}
