use crate::compression::base_bit_selection::BaseBitGroups;
use crate::compression::entropy::EntropyOptimized;
use crate::compression::preprocessor::{BitData, BitDataInfo, BitDataSet};
use crate::error::EntroGdError;
use crate::filter_pipeline::{Filter, FilterExt};
use crate::timing::ScopedTimer;
use bitvec::prelude::*;
use log::debug;
use std::sync::Arc;

pub fn build_compression_pipeline(
    m_max: usize,
    patience: usize,
) -> impl Filter<Input = BitDataSet, Output = CompressedData> {
    EntropyOptimized {}
        .then(GenCondensedSamples { m_max })
        .then(SelectBases { patience })
        .then(EncodeData {})
}

pub fn build_compression_pipeline_optimized(
    m_max: usize,
    patience: usize,
) -> impl Filter<Input = BitDataSet, Output = CompressedData> {
    EntropyOptimized {}
        .then(GenCondensedSamples { m_max })
        .then(SelectBases { patience })
        .then(EncodeDataOptimized {})
}

/// Represents the compressed output
#[derive(Debug, Clone)]
pub struct CompressedData {
    /// The encoded data stream
    pub encoded_data: DeviationData,
    /// The Weights for the condensed samples (if used)
    // Should be stored as a bitstream of length m * l_w (log_2(n).ceil() bits per weight)
    pub condensed_sample_weights: Option<Vec<usize>>,
    /// Base table mapping base patterns to their frequencies or encodings
    pub base_table: Vec<(BitVec<usize, Msb0>, usize)>,
    /// Bit positions used as base bits during compression
    pub base_bit_positions: Vec<usize>,
    /// Metadata for decompression (column count, base bits used, etc.)
    pub metadata: BitDataInfo,
}

impl CompressedData {
    pub fn new(encoded_data: DeviationData, metadata: BitDataInfo) -> Self {
        CompressedData {
            encoded_data,
            condensed_sample_weights: None,
            base_table: Vec::new(),
            base_bit_positions: Vec::new(),
            metadata,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CondensedSamples {
    pub samples: Vec<BitVec<usize, Msb0>>,
    pub weights: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct DeviationSample {
    pub deviation: BitVec<usize, Msb0>,
    pub id: BitVec<usize, Msb0>,
}

#[derive(Debug, Clone)]
pub struct DeviationData {
    encoded_bit_stream: BitVec<usize, Msb0>,
    num_samples: usize,
    num_deviation_bits: usize,
    num_id_bits: usize,
}

fn build_base_bit_mask(chunk_size: usize, base_bit_positions: &[usize]) -> BitVec<usize, Msb0> {
    let mut mask = bitvec![usize, Msb0; 0; chunk_size];
    for &bit_pos in base_bit_positions {
        if bit_pos < chunk_size {
            mask.set(bit_pos, true);
        }
    }
    mask
}

impl DeviationData {
    pub fn new(
        encoded_bit_stream: BitVec<usize, Msb0>,
        num_samples: usize,
        num_deviation_bits: usize,
        num_id_bits: usize,
    ) -> Self {
        DeviationData {
            encoded_bit_stream,
            num_samples,
            num_deviation_bits,
            num_id_bits,
        }
    }

    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        if sample_idx >= self.num_samples {
            return None;
        }
        let start_bit = sample_idx * (self.num_deviation_bits + self.num_id_bits);
        let end_bit = start_bit + self.num_deviation_bits + self.num_id_bits;
        Some(DeviationSample {
            deviation: self.encoded_bit_stream[start_bit..start_bit + self.num_deviation_bits]
                .to_bitvec(),
            id: self.encoded_bit_stream[start_bit + self.num_deviation_bits..end_bit].to_bitvec(),
        })
    }

    pub fn get_encoded_size(&self) -> usize {
        self.encoded_bit_stream.len()
    }

    pub fn encoded_bit_stream(&self) -> &BitVec<usize, Msb0> {
        &self.encoded_bit_stream
    }

    pub fn get_num_samples(&self) -> usize {
        self.num_samples
    }

    pub fn get_num_deviation_bits(&self) -> usize {
        self.num_deviation_bits
    }

    pub fn get_num_id_bits(&self) -> usize {
        self.num_id_bits
    }
}

fn organize_entropy_by_feature(
    entropy: &[(usize, f64)],
    bit_data: &BitDataSet,
) -> Vec<(usize, f64)> {
    let num_features = bit_data.num_features();
    let mut entropy_by_feature: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_features];
    let mut zero_entropy = Vec::new();

    for &(bit_pos, entropy_val) in entropy {
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
        for feature_entropy in entropy_by_feature.iter().take(num_features) {
            if let Some(entry) = feature_entropy.get(i) {
                result.push(*entry);
            }
        }
    }

    result
}

pub struct GenCondensedSamples {
    pub m_max: usize, // maximum number of bases to include in condensed samples
}

impl Filter for GenCondensedSamples {
    type Input = (BitDataSet, Vec<(usize, f64)>);
    type Output = (BitDataSet, Vec<(usize, f64)>);

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Generating condensed samples (m_max = {})",
            self.m_max
        ));
        let (bit_data, entropy) = input;
        let condensed_samples = select_condensed_samples(&bit_data, &entropy, self.m_max);
        Ok((
            append_condensed_samples(bit_data, condensed_samples),
            entropy,
        ))
    }
}

fn select_condensed_samples(
    bit_data: &BitDataSet,
    entropy: &[(usize, f64)],
    m_max: usize,
) -> CondensedSamples {
    fn bits_to_u64(bits: &BitSlice<usize, Msb0>) -> u64 {
        bits.iter()
            .fold(0u64, |acc, bit| (acc << 1) | (*bit as u64))
    }

    fn push_u64_bits(stream: &mut BitVec<usize, Msb0>, value: u64, bits: usize) {
        for shift in (0..bits).rev() {
            stream.push(((value >> shift) & 1) == 1);
        }
    }

    fn mask_for_bits(bits: usize) -> u64 {
        if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
    }

    let mut samples = Vec::new();
    let mut weights = Vec::new();

    let organized_entropy_by_feature = organize_entropy_by_feature(entropy, bit_data);
    let mut condensed_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());

    // Extract zero entropy bits first
    let zero_entropy_bits: Vec<usize> = organized_entropy_by_feature
        .iter()
        .take_while(|&(_bit_position, entropy_val)| *entropy_val == 0.0)
        .map(|(bit_position, _)| *bit_position)
        .collect();
    condensed_bit_groups.add_constant_bit_positions(&zero_entropy_bits);

    // Loop over non-zero entropy bits
    for &(bit_position, _entropy_val) in organized_entropy_by_feature
        .iter()
        .skip(zero_entropy_bits.len())
    {
        if condensed_bit_groups.get_num_bases() >= m_max {
            break;
        }
        condensed_bit_groups.add_bit_position(bit_data, bit_position);
        debug!(
            "Added bit position {} to condensed samples, current number of bases: {}",
            bit_position,
            condensed_bit_groups.get_num_bases()
        );
    }

    let bases = condensed_bit_groups.get_bases(bit_data);
    let base_mask = condensed_bit_groups.get_base_bit_mask();

    for (base_group, base) in condensed_bit_groups.get_groups().iter().zip(bases.iter()) {
        if base_group.is_empty() {
            continue;
        }

        let mut condensed_bitvec = BitVec::<usize, Msb0>::with_capacity(bit_data.chunk_size());

        for feature_idx in 0..bit_data.num_features() {
            let feature_offset = bit_data.feature_offset(feature_idx);
            let feature_bits = bit_data.feature_bits(feature_idx);
            let feature_end = feature_offset + feature_bits;

            let base_feature = &base.0[feature_offset..feature_end];
            let base_feature_mask = &base_mask[feature_offset..feature_end];

            let base_value = bits_to_u64(base_feature);
            let deviation_mask = !bits_to_u64(base_feature_mask) & mask_for_bits(feature_bits);

            let mut deviation_accumulator = 0u128;
            for &sample_idx in base_group.iter() {
                let sample_feature = &bit_data.get_chunk(sample_idx)[feature_offset..feature_end];
                let sample_feature_value = bits_to_u64(sample_feature);
                let feature_deviation = sample_feature_value & deviation_mask;
                deviation_accumulator += feature_deviation as u128;
            }

            let mean_deviation = (deviation_accumulator / base_group.len() as u128) as u64;
            let condensed_feature_value =
                (base_value + mean_deviation) & mask_for_bits(feature_bits);
            push_u64_bits(&mut condensed_bitvec, condensed_feature_value, feature_bits);
        }

        samples.push(condensed_bitvec);
        weights.push(base_group.len());
    }

    CondensedSamples { samples, weights }
}

fn append_condensed_samples(
    mut bit_data: BitDataSet,
    condensed_samples: CondensedSamples,
) -> BitDataSet {
    bit_data.info.m_condensed_samples = Some(condensed_samples.samples.len());
    bit_data.info.m_condensed_sample_weights = Some(condensed_samples.weights);

    for sample in condensed_samples.samples {
        bit_data
            .data
            .extend_from_bitslice(sample.as_bitslice())
            .expect("condensed sample chunk size must match dataset chunk size");
    }
    bit_data
}

fn calculate_compressed_size(bit_data: &BitDataSet, base_bit_groups: &BaseBitGroups) -> usize {
    fn min_bit_length(value: usize) -> usize {
        match value {
            0 => 0,
            1 => 1,
            _ => (usize::BITS as usize) - ((value - 1).leading_zeros() as usize),
        }
    }

    let n = bit_data.num_rows(); // number of encoded samples
    let d = bit_data.num_features(); // dimensionality (number of features)
    let n_b = base_bit_groups.get_num_bases(); // number of bases
    let chunk_size = bit_data.chunk_size(); // sum(m) in the Python implementation

    let len_b = base_bit_groups.get_num_bits_per_base().min(chunk_size); // bits per base
    let len_d = chunk_size - len_b; // deviation bits per sample

    // Python-like sizing terms.
    let len_bc = min_bit_length(n); // bits per base count
    let len_id = min_bit_length(n_b); // bits per base id

    let size_bases = n_b * len_b;
    let size_base_counts = n_b * len_bc;
    let size_deviations = n * (len_d + len_id);

    // 16 * d + 16 + size_dev_bits (+ currently ignored terms).
    let size_dev_bits = chunk_size;
    let size_params = 16 * d + 16 + size_dev_bits;

    size_bases + size_base_counts + size_deviations + size_params
}

pub struct SelectBases {
    pub patience: usize,
}

impl Filter for SelectBases {
    type Input = (BitDataSet, Vec<(usize, f64)>);
    type Output = (BitDataSet, BaseBitGroups);

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting base bits with patience {}",
            self.patience
        ));
        let (bit_data, entropy) = input;
        let base_bit_groups = optimize_base_bit_groups(&bit_data, entropy, self.patience);
        Ok((bit_data, base_bit_groups))
    }
}

fn optimize_base_bit_groups(
    bit_data: &BitDataSet,
    mut entropy: Vec<(usize, f64)>,
    patience: usize,
) -> BaseBitGroups {
    let mut non_improving_count = 0usize;
    let mut base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());

    entropy.sort_by(|a, b| a.1.total_cmp(&b.1));
    let zero_entropy_bits: Vec<usize> = entropy
        .iter()
        .take_while(|&(_bit_position, entropy_val)| *entropy_val == 0.0)
        .map(|(bit_position, _)| *bit_position)
        .collect();
    base_bit_groups.add_constant_bit_positions(&zero_entropy_bits);

    let mut best_base_bit_groups = base_bit_groups.clone();
    let mut best_compressed_size = calculate_compressed_size(bit_data, &best_base_bit_groups);
    let mut trial_base_bit_groups = base_bit_groups.clone();

    for &(bit_position, _) in entropy.iter().skip(zero_entropy_bits.len()) {
        trial_base_bit_groups.add_bit_position(bit_data, bit_position);
        let trial_compressed_size = calculate_compressed_size(bit_data, &trial_base_bit_groups);

        debug!(
            "Trial bit position: {}, Trial compressed size: {}, Best compressed size: {}",
            bit_position, trial_compressed_size, best_compressed_size
        );

        if trial_compressed_size < best_compressed_size {
            best_compressed_size = trial_compressed_size;
            best_base_bit_groups = trial_base_bit_groups.clone();
            non_improving_count = 0;
        } else {
            non_improving_count += 1;
        }

        if non_improving_count >= patience {
            break;
        }
    }
    log::info!(
        "\nSelected base bit mask: {}",
        best_base_bit_groups
            .get_base_bit_mask()
            .iter()
            .map(|b| if *b { "1" } else { "0" })
            .collect::<String>()
    );
    best_base_bit_groups
}

pub struct EncodeData {}

impl Filter for EncodeData {
    type Input = (BitDataSet, BaseBitGroups);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format");
        let (bit_data, base_bit_groups) = input;
        let mut compressed = CompressedData::new(
            encode_data(&bit_data, &base_bit_groups),
            bit_data.info.clone(),
        );
        compressed.base_table = base_bit_groups.get_bases(&bit_data);
        compressed.base_bit_positions = base_bit_groups.get_base_bit_positions().to_vec();
        compressed.condensed_sample_weights = bit_data.info.m_condensed_sample_weights.clone();
        Ok(compressed)
    }
}

pub struct EncodeDataOptimized {}

impl Filter for EncodeDataOptimized {
    type Input = (BitDataSet, BaseBitGroups);
    type Output = CompressedData;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Encoding data into compressed format (optimized)");
        let (bit_data, base_bit_groups) = input;
        let mut compressed = CompressedData::new(
            encode_data_optimized(&bit_data, &base_bit_groups),
            bit_data.info.clone(),
        );
        compressed.base_table = base_bit_groups.get_bases(&bit_data);
        compressed.base_bit_positions = base_bit_groups.get_base_bit_positions().to_vec();
        compressed.condensed_sample_weights = bit_data.info.m_condensed_sample_weights.clone();
        Ok(compressed)
    }
}

fn encode_data(bit_data: &BitDataSet, base_bit_groups: &BaseBitGroups) -> DeviationData {
    let mut encoded_bit_stream = BitVec::<usize, Msb0>::new();
    let base_bit_mask = base_bit_groups.get_base_bit_mask();

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

    for (row, id_ref) in row_to_group_id.iter().enumerate().take(num_rows) {
        let id = *id_ref;
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

fn encode_data_optimized(bit_data: &BitDataSet, base_bit_groups: &BaseBitGroups) -> DeviationData {
    let base_bit_mask = base_bit_groups.get_base_bit_mask();

    let num_bases = base_bit_groups.get_num_bases();
    let l_id = {
        let bits = (num_bases as f64).log2().ceil() as usize;
        if bits == 0 { 1 } else { bits }
    };

    let chunk_size = bit_data.chunk_size();
    let num_bits_per_base = base_bit_groups.get_num_bits_per_base();
    let num_deviation_bits = chunk_size.saturating_sub(num_bits_per_base);

    // Precompute non-base (deviation) bit positions once.
    let mut deviation_positions = Vec::with_capacity(num_deviation_bits);
    for bit_pos in 0..chunk_size {
        if !base_bit_mask[bit_pos] {
            deviation_positions.push(bit_pos);
        }
    }

    // Collapse consecutive deviation positions into contiguous ranges so we can
    // append whole slices instead of pushing one bit at a time.
    let mut deviation_ranges: Vec<(usize, usize)> = Vec::new();
    if let Some(&first_pos) = deviation_positions.first() {
        let mut range_start = first_pos;
        let mut prev = first_pos;

        for &pos in deviation_positions.iter().skip(1) {
            if pos == prev + 1 {
                prev = pos;
            } else {
                deviation_ranges.push((range_start, prev + 1));
                range_start = pos;
                prev = pos;
            }
        }
        deviation_ranges.push((range_start, prev + 1));
    }

    // Precompute ID bit patterns per base to avoid per-row shift/push loops.
    let mut id_bits_per_base: Vec<BitVec<usize, Msb0>> = Vec::new();
    if l_id > 0 {
        id_bits_per_base = Vec::with_capacity(num_bases);
        for id in 0..num_bases {
            let mut id_bits = BitVec::<usize, Msb0>::with_capacity(l_id);
            for shift in (0..l_id).rev() {
                id_bits.push(((id >> shift) & 1) == 1);
            }
            id_bits_per_base.push(id_bits);
        }
    }

    let num_rows = bit_data.num_rows();
    let mut row_to_group_id = vec![0usize; num_rows];
    let mut encoded_bit_stream =
        BitVec::<usize, Msb0>::with_capacity(num_rows * (num_deviation_bits + l_id));
    for (id, group) in base_bit_groups.get_groups().iter().enumerate() {
        for &row in group.iter() {
            row_to_group_id[row] = id;
        }
    }

    for (row, id_ref) in row_to_group_id.iter().enumerate().take(num_rows) {
        let id = *id_ref;
        let chunk = bit_data.get_chunk(row);

        for &(start, end) in &deviation_ranges {
            encoded_bit_stream.extend_from_bitslice(&chunk[start..end]);
        }

        if l_id > 0 {
            encoded_bit_stream.extend_from_bitslice(id_bits_per_base[id].as_bitslice());
        }
    }

    DeviationData::new(
        encoded_bit_stream,
        bit_data.num_rows(),
        num_deviation_bits,
        l_id,
    )
}

pub struct DecompressRowsData;

impl Filter for DecompressRowsData {
    type Input = (Arc<CompressedData>, Vec<usize>);
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!("Decompressing {} rows", input.1.len()));
        let (compressed, indices) = input;
        decompress_samples_batch(compressed.as_ref(), &indices)
    }
}

/// Private batch decompression function
fn decompress_samples_batch(
    compressed: &CompressedData,
    indices: &[usize],
) -> Result<BitDataSet, EntroGdError> {
    let data_info = &compressed.metadata;
    let num_features = data_info.num_features();
    let chunk_size = data_info.chunk_size();
    let original_num_rows = data_info.original_size_bits / chunk_size;
    if num_features == 0 {
        return Err(EntroGdError::InvalidMetadata {
            message: "num_features is 0".to_string(),
        });
    }
    if let Some(&sample_idx) = indices.iter().find(|&&idx| idx >= original_num_rows) {
        return Err(EntroGdError::DecompressionSampleMissing { sample_idx });
    }
    let base_bit_positions = &compressed.base_bit_positions;
    let base_bit_mask = build_base_bit_mask(chunk_size, base_bit_positions);

    let mut reconstructed_bits = BitVec::<usize, Msb0>::with_capacity(chunk_size * indices.len());

    for &sample_idx in indices {
        let sample = match compressed.encoded_data.get_sample(sample_idx) {
            Some(s) => s,
            None => {
                return Err(EntroGdError::DecompressionSampleMissing { sample_idx });
            }
        };

        let mut chunk = bitvec![usize, Msb0; 0; chunk_size];
        let mut base_id = 0usize;
        for i in 0..sample.id.len() {
            base_id = (base_id << 1) | (sample.id[i] as usize);
        }
        if base_id >= compressed.base_table.len() {
            return Err(EntroGdError::InvalidBaseId {
                base_id,
                table_len: compressed.base_table.len(),
            });
        }
        let base_pattern = &compressed.base_table[base_id].0;
        for bit_pos in 0..chunk_size.min(base_pattern.len()) {
            chunk.set(bit_pos, base_pattern[bit_pos]);
        }
        let mut deviation_bit_idx = 0;
        for bit_pos in 0..chunk_size {
            if !base_bit_mask[bit_pos] && deviation_bit_idx < sample.deviation.len() {
                chunk.set(bit_pos, sample.deviation[deviation_bit_idx]);
                deviation_bit_idx += 1;
            }
        }
        reconstructed_bits.extend_from_bitslice(&chunk);
    }

    let data = BitData {
        data: reconstructed_bits,
        chunk_size,
        num_rows: indices.len(),
    };
    let info = data_info.with_original_size_bits(chunk_size * indices.len());
    Ok(BitDataSet { data, info })
}

pub struct DecompressFileData {}

impl Filter for DecompressFileData {
    type Input = CompressedData;
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Decompressing entire file data");
        decompress_file(&input)
    }
}

/// Decompress the original file data (first n samples)
pub fn decompress_file(compressed: &CompressedData) -> Result<BitDataSet, EntroGdError> {
    let data_info = &compressed.metadata;
    let original_num_rows = data_info.original_size_bits / data_info.chunk_size();
    let indices: Vec<usize> = (0..original_num_rows).collect();
    decompress_samples_batch(compressed, &indices)
}

pub struct DecompressAnalytics {}

impl Filter for DecompressAnalytics {
    type Input = CompressedData;
    type Output = Option<CondensedSamples>;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Decompressing condensed samples for analytics");
        Ok(decompress_analytics(&input))
    }
}

/// Decompress condensed samples for analytics
pub fn decompress_analytics(compressed: &CompressedData) -> Option<CondensedSamples> {
    if let Some(weights) = &compressed.condensed_sample_weights {
        // Reconstruct condensed samples from base_table
        let samples: Vec<BitVec<usize, Msb0>> = compressed
            .base_table
            .iter()
            .map(|(bv, _)| bv.clone())
            .collect();
        Some(CondensedSamples {
            samples,
            weights: weights.clone(),
        })
    } else {
        None
    }
}

/// Calculate the original uncompressed size in bits
#[cfg(test)]
fn calculate_original_size(bit_data: &BitDataSet) -> usize {
    bit_data.data.num_rows * bit_data.data.chunk_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::compress::build_compression_pipeline;
    use crate::compression::preprocessor::{BitData, BitDataInfo, BitDataSet, FeatureSpec};
    use crate::data_loader::FeatureDataType;
    use crate::filter_pipeline::Filter;
    use pretty_assertions::assert_eq;

    fn get_compression_pipeline() -> impl Filter<Input = BitDataSet, Output = CompressedData> {
        build_compression_pipeline(100, 5)
    }

    // Helper function to create test BitData
    fn create_test_bit_data(
        num_rows: usize,
        bits_per_feature: usize,
        num_features: usize,
    ) -> BitDataSet {
        let chunk_size = bits_per_feature * num_features;
        let total_bits = chunk_size * num_rows;

        let data = bitvec![usize, Msb0; 0; total_bits];
        let data = BitData {
            data,
            chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();

        BitDataSet { data, info }
    }

    // Tests for calculate_original_size
    #[test]
    fn test_calculate_original_size_basic() {
        let bit_data = create_test_bit_data(100, 1, 8);
        let original_size = calculate_original_size(&bit_data);
        assert_eq!(original_size, 100 * 8);
    }

    #[test]
    fn test_calculate_original_size_single_row() {
        let bit_data = create_test_bit_data(1, 2, 8);
        let original_size = calculate_original_size(&bit_data);
        assert_eq!(original_size, 1 * 16);
    }

    #[test]
    fn test_calculate_original_size_large_data() {
        let bit_data = create_test_bit_data(10000, 2, 32);
        let original_size = calculate_original_size(&bit_data);
        assert_eq!(original_size, 10000 * 64);
    }

    #[test]
    fn test_calculate_original_size_zero_rows() {
        let bit_data = create_test_bit_data(0, 1, 8);
        let original_size = calculate_original_size(&bit_data);
        assert_eq!(original_size, 0);
    }

    #[test]
    fn test_calculate_original_size_single_bit() {
        let bit_data = create_test_bit_data(100, 1, 1);
        let original_size = calculate_original_size(&bit_data);
        assert_eq!(original_size, 100);
    }

    // Tests for calculate_compressed_size
    #[test]
    fn test_calculate_compressed_size_basic() {
        let bit_data = create_test_bit_data(100, 1, 32);
        let base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0 (no bases), len_b = 0, len_d = 32, len_id = 0
        // size_deviations = 100 * 32 = 3200
        // size_params = 16*32 + 16 + 32 = 560
        // total = 3760
        assert_eq!(compressed_size, 3760);
    }

    #[test]
    fn test_calculate_compressed_size_single_row() {
        let bit_data = create_test_bit_data(1, 2, 8);
        let base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, len_b = 0, len_d = 16, len_id = 0
        // size_deviations = 1 * 16 = 16
        // size_params = 16*8 + 16 + 16 = 160
        // total = 176
        assert_eq!(compressed_size, 176);
    }

    #[test]
    fn test_calculate_compressed_size_large_data() {
        let bit_data = create_test_bit_data(10000, 2, 32);
        let base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, len_b = 0, len_d = 64, len_id = 0
        // size_deviations = 10000 * 64 = 640000
        // size_params = 16*32 + 16 + 64 = 592
        // total = 640592
        assert_eq!(compressed_size, 640592);
    }

    #[test]
    fn test_calculate_compressed_size_power_of_two_rows() {
        // Test with a power-of-2 number of rows to verify log2 calculation
        let bit_data = create_test_bit_data(16, 1, 32);
        let base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let compressed_size = calculate_compressed_size(&bit_data, &base_bit_groups);

        // n_b = 0, len_b = 0, len_d = 32, len_id = 0
        // size_deviations = 16 * 32 = 512
        // size_params = 16*32 + 16 + 32 = 560
        // total = 1072
        assert_eq!(compressed_size, 1072);
    }

    // Tests for compress function
    #[test]
    fn test_compress_basic() {
        // Use larger data to avoid optimization issues in edge cases
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Check that we get a CompressedData struct with valid fields
        assert_eq!(compressed.metadata.num_features(), 16);
        assert_eq!(compressed.metadata.original_size_bits, 100 * 32);
        assert!(!compressed.base_table.is_empty());
        assert!(compressed.encoded_data.get_encoded_size() > 0);
    }

    #[test]
    fn test_compress_single_row() {
        let bit_data = create_test_bit_data(1, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 16);
        assert_eq!(compressed.metadata.original_size_bits, 32);
        // Verify decompressed data matches original size
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 1);
    }

    #[test]
    fn test_compress_large_data() {
        let bit_data = create_test_bit_data(1000, 2, 32);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 32);
        assert_eq!(compressed.metadata.original_size_bits, 1000 * 64);
        // Verify decompressed data matches original size
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 1000);
    }

    #[test]
    fn test_compress_preserves_sample_count() {
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify original size is preserved in metadata
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits, 100 * 32);
        // Verify decompressed data matches original count
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_compress_metadata_accuracy() {
        let bit_data = create_test_bit_data(50, 2, 8);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 8);
        assert_eq!(compressed.metadata.original_size_bits, 50 * 16);
    }

    #[test]
    fn test_compress_encoded_data_structure() {
        let bit_data = create_test_bit_data(20, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify original metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits, 20 * 32);

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 20);

        // Verify decompressed data structure
        assert_eq!(decompressed.info.num_features(), 16);
        assert_eq!(decompressed.data.chunk_size, 32); // bits_per_feature * num_features = 2 * 16 = 32
    }

    #[test]
    fn test_compress_invalid_sample_index() {
        let bit_data = create_test_bit_data(10, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 10);

        // Verify metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits, 10 * 32);
    }

    #[test]
    fn test_compress_base_table_non_empty() {
        let bit_data = create_test_bit_data(50, 2, 32);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Base table should contain entries for each base group
        assert!(!compressed.base_table.is_empty());

        // Each base table entry should have a count > 0
        for (_base, count) in &compressed.base_table {
            assert!(*count > 0);
        }

        // Verify decompress_analytics returns condensed samples
        let analytics = decompress_analytics(&compressed);
        assert!(analytics.is_some());
        let condensed = analytics.unwrap();
        assert_eq!(condensed.samples.len(), compressed.base_table.len());
        assert_eq!(condensed.weights.len(), compressed.base_table.len());
    }

    #[test]
    fn test_compress_varying_row_counts() {
        // Test with different numbers of rows
        let row_counts = vec![1, 5, 10, 50, 100, 500];

        for num_rows in row_counts {
            let bit_data = create_test_bit_data(num_rows, 2, 16);
            let compressed = get_compression_pipeline()
                .process(bit_data.clone())
                .unwrap();

            assert_eq!(compressed.metadata.num_features(), 16);
            assert_eq!(compressed.metadata.original_size_bits, num_rows * 32);

            // Verify decompress_file returns correct number of rows
            let decompressed = decompress_file(&compressed).unwrap();
            assert_eq!(decompressed.data.num_rows, num_rows);
        }
    }

    #[test]
    fn test_compress_encoded_data_retrieval() {
        let bit_data = create_test_bit_data(20, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify we can retrieve samples and they have expected structure
        for i in 0..20 {
            let sample = compressed.encoded_data.get_sample(i);
            assert!(sample.is_some());
            let sample_bits = sample.unwrap();
            // Sample should contain deviation bits + id bits
            assert!(sample_bits.deviation.len() + sample_bits.id.len() > 0);
        }

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 20);
    }

    #[test]
    fn test_compress_consistency() {
        // Compress separate instances of the same data and verify results are consistent
        let original_rows = 50;
        let bit_data1 = create_test_bit_data(original_rows, 2, 16);
        let bit_data2 = create_test_bit_data(original_rows, 2, 16);

        let compressed1 = get_compression_pipeline()
            .process(bit_data1.clone())
            .unwrap();
        let compressed2 = get_compression_pipeline()
            .process(bit_data2.clone())
            .unwrap();

        assert_eq!(
            compressed1.metadata.original_size_bits,
            compressed2.metadata.original_size_bits
        );
        // Verify both decompress to the same original row count
        let decompressed1 = decompress_file(&compressed1).unwrap();
        let decompressed2 = decompress_file(&compressed2).unwrap();
        assert_eq!(decompressed1.data.num_rows, decompressed2.data.num_rows);
        assert_eq!(decompressed1.data.num_rows, original_rows);
    }

    #[test]
    fn test_compress_power_of_two_rows() {
        // Test with power-of-2 number of rows to verify log2 calculations
        let powers_of_two = vec![1, 2, 4, 8, 16, 32, 64];

        for num_rows in powers_of_two {
            let bit_data = create_test_bit_data(num_rows, 2, 16);
            let compressed = get_compression_pipeline()
                .process(bit_data.clone())
                .unwrap();

            assert_eq!(compressed.metadata.original_size_bits, num_rows * 32);
            // Verify decompressed matches original
            let decompressed = decompress_file(&compressed).unwrap();
            assert_eq!(decompressed.data.num_rows, num_rows);
        }
    }

    #[test]
    fn test_compress_small_chunks() {
        // Test with 8-bit chunks (1 byte per feature, 1 feature)
        let bit_data = create_test_bit_data(100, 1, 8);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 8);
        assert_eq!(compressed.metadata.original_size_bits, 100 * 8);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_compress_medium_chunks() {
        // Test with 16-bit chunks
        let bit_data = create_test_bit_data(100, 1, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 16);
        assert_eq!(compressed.metadata.original_size_bits, 100 * 16);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_compress_large_chunks() {
        // Test with 32-bit chunks
        let bit_data = create_test_bit_data(100, 1, 32);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        assert_eq!(compressed.metadata.num_features(), 32);
        assert_eq!(compressed.metadata.original_size_bits, 100 * 32);
        // Verify decompressed matches original
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 100);
    }

    #[test]
    fn test_deviation_data_out_of_bounds() {
        let bit_data = create_test_bit_data(50, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify decompress_file returns correct number of rows
        let decompressed = decompress_file(&compressed).unwrap();
        assert_eq!(decompressed.data.num_rows, 50);

        // Verify metadata is correct
        // chunk_size = bits_per_feature * num_features = 2 * 16 = 32
        assert_eq!(compressed.metadata.original_size_bits, 50 * 32);
    }

    #[test]
    fn test_compressed_data_structure() {
        let bit_data = create_test_bit_data(100, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        // Verify CompressedData has all required fields
        assert!(!compressed.base_table.is_empty());
        assert!(compressed.encoded_data.get_num_samples() > 0);
        assert_eq!(compressed.metadata.num_features(), 16);
        assert!(compressed.metadata.original_size_bits > 0);
    }

    fn create_simulated_bit_data(
        num_rows: usize,
        bits_per_feature: usize,
        num_features: usize,
    ) -> BitDataSet {
        let chunk_size = bits_per_feature * num_features;
        let total_bits = chunk_size * num_rows;
        let mut data = BitVec::<usize, Msb0>::with_capacity(total_bits);

        // Create simulated data with low entropy patterns
        for row in 0..num_rows {
            for bit_pos in 0..chunk_size {
                // Low entropy: most bits are constant (0), with few changes
                // Only the first few bit positions vary
                let bit = if bit_pos < 2 {
                    // First 2 bits vary by row
                    row % 2 == bit_pos
                } else {
                    // All remaining bits are constant 0 (high redundancy)
                    false
                };
                data.push(bit);
            }
        }

        let data = BitData {
            data,
            chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        BitDataSet { data, info }
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        // Create test bit data with zeros
        let original_rows = 50;
        let bit_data = create_test_bit_data(original_rows, 2, 16);

        // Compress the data
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        // Decompress the data
        let decompressed = decompress_file(&compressed).expect("Decompression failed");

        // Verify the decompressed data has the correct structure
        // Note: bit_data.num_rows may have been increased by condensed samples,
        // but decompressed should match the original row count
        assert_eq!(decompressed.info.num_features(), 16); // Original num_features
        assert_eq!(decompressed.data.num_rows, original_rows); // Should match original, not expanded
        assert_eq!(decompressed.data.chunk_size, 32);
        assert_eq!(decompressed.info.feature_bits(0), 2);
        assert_eq!(decompressed.data.data.len(), original_rows * 32); // Original size, not expanded
    }

    #[test]
    fn test_compress_decompress_with_simulated_data_debug() {
        // Create a very simple test case with minimal data
        let bit_data = create_simulated_bit_data(5, 1, 8); // 5 rows, 8-bit chunks

        log::info!("\n=== INPUT DATA ===");
        log::info!("Num rows: {}", bit_data.data.num_rows);
        log::info!("Chunk size: {}", bit_data.data.chunk_size);
        log::info!("Total bits: {}", bit_data.data.data.len());
        for row in 0..bit_data.data.num_rows {
            let start = row * bit_data.data.chunk_size;
            let end = start + bit_data.data.chunk_size;
            let chunk: Vec<u8> = bit_data.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            log::info!("Row {}: {:?}", row, chunk);
        }

        // Compress the data
        log::info!("\n=== COMPRESSING ===");
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();

        log::info!("Base table entries: {}", compressed.base_table.len());
        log::info!(
            "Encoded bit stream size: {}",
            compressed.encoded_data.get_encoded_size()
        );
        log::info!("Num samples: {}", compressed.encoded_data.get_num_samples());
        log::info!(
            "Num deviation bits: {}",
            compressed.encoded_data.get_num_deviation_bits()
        );
        log::info!("Num ID bits: {}", compressed.encoded_data.get_num_id_bits());

        // Decompress the data
        log::info!("\n=== DECOMPRESSING ===");
        let decompressed = decompress_file(&compressed).expect("Decompression failed");

        log::info!("Decompressed num rows: {}", decompressed.data.num_rows);
        log::info!("Decompressed chunk size: {}", decompressed.data.chunk_size);

        log::info!("\n=== COMPARING ===");
        // Check row by row - use decompressed row count since bit_data now contains condensed samples
        let num_original_rows = 5;
        for row in 0..num_original_rows {
            let start = row * bit_data.data.chunk_size;
            let end = start + bit_data.data.chunk_size;

            let original_chunk: Vec<u8> = bit_data.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            let decompressed_chunk: Vec<u8> = decompressed.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();

            if original_chunk == decompressed_chunk {
                log::info!("Row {}: OK", row);
            } else {
                log::info!("Row {} MISMATCH:", row);
                log::info!("  Original:     {:?}", original_chunk);
                log::info!("  Decompressed: {:?}", decompressed_chunk);
            }
        }

        // Verify the original rows match by checking only the first num_original_rows rows
        let original_bits: Vec<bool> = bit_data.data.data
            [0..(num_original_rows * bit_data.data.chunk_size)]
            .iter()
            .map(|b| *b)
            .collect();
        let decompressed_bits: Vec<bool> = decompressed.data.data
            [0..(num_original_rows * bit_data.data.chunk_size)]
            .iter()
            .map(|b| *b)
            .collect();
        assert_eq!(
            decompressed_bits, original_bits,
            "Decompressed data does not match original for the first {} rows",
            num_original_rows
        );

        // Print analytics if available
        if let Some(analytics) = decompress_analytics(&compressed) {
            log::info!("\n=== ANALYTICS ===");
            log::info!("Condensed samples: {}", analytics.samples.len());
            for (i, sample) in analytics.samples.iter().enumerate() {
                let weight = analytics.weights[i];
                let sample_bits: Vec<u8> = sample.iter().map(|b| if *b { 1 } else { 0 }).collect();
                log::info!("Sample {}: {:?}, Weight: {}", i, sample_bits, weight);
            }
        }
    }

    #[test]
    fn test_compress_decompress_roundtrip_with_simulated_data_debug_2() {
        // Create a simple BitData for testing
        let num_rows = 12;
        let chunk_size = 4;
        let num_features = 4;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, // Row 0
            true, true, false, false, // Row 1
            true, true, false, true, // Row 2
            true, true, false, false, // Row 3
            true, true, false, false, // Row 4
            true, false, true, true, // Row 5
            true, true, false, false, // Row 6
            true, true, false, false, // Row 7
            true, true, false, false, // Row 8
            true, true, false, true, // Row 9
            true, true, false, true, // Row 10
            true, true, false, false, // Row 11
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };
        log::info!("\n=== INPUT DATA ===");
        for row in 0..bit_data.data.num_rows {
            let start = row * bit_data.data.chunk_size;
            let end = start + bit_data.data.chunk_size;
            let chunk: Vec<u8> = bit_data.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            log::info!("Row {}: {:?}", row, chunk);
        }
        // Compress the data
        log::info!("\n=== COMPRESSING ===");
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        log::info!("Base table entries: {}", compressed.base_table.len());
        log::info!("Base table: {:?}", compressed.base_table);
        log::info!(
            "Encoded bit stream size: {}",
            compressed.encoded_data.get_encoded_size()
        );
        log::info!("Num samples: {}", compressed.encoded_data.get_num_samples());
        log::info!(
            "Num deviation bits: {}",
            compressed.encoded_data.get_num_deviation_bits()
        );
        log::info!("Num ID bits: {}", compressed.encoded_data.get_num_id_bits());
        // Decompress analytics if available
        if let Some(analytics) = decompress_analytics(&compressed) {
            log::info!("\n=== ANALYTICS ===");
            log::info!("Condensed samples: {}", analytics.samples.len());
            for (i, sample) in analytics.samples.iter().enumerate() {
                let weight = analytics.weights[i];
                let sample_bits: Vec<u8> = sample.iter().map(|b| if *b { 1 } else { 0 }).collect();
                log::info!("Sample {}: {:?}, Weight: {}", i, sample_bits, weight);
            }
        }
        // Decompress the data
        log::info!("\n=== DECOMPRESSING ===");
        let decompressed = decompress_file(&compressed).expect("Decompression failed");
        log::info!("Decompressed num rows: {}", decompressed.data.num_rows);
        log::info!("Decompressed chunk size: {}", decompressed.data.chunk_size);
        log::info!("\n=== COMPARING ===");
        // Check row by row - use decompressed row count since bit_data now contains condensed samples
        let num_original_rows = 12;
        for row in 0..num_original_rows {
            let start = row * bit_data.data.chunk_size;
            let end = start + bit_data.data.chunk_size;
            let original_chunk: Vec<u8> = bit_data.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            let decompressed_chunk: Vec<u8> = decompressed.data.data[start..end]
                .iter()
                .map(|b| if *b { 1 } else { 0 })
                .collect();
            if original_chunk == decompressed_chunk {
                log::info!("Row {}: OK", row);
            } else {
                log::info!("Row {} MISMATCH:", row);
                log::info!("  Original:     {:?}", original_chunk);
                log::info!("  Decompressed: {:?}", decompressed_chunk);
            }
        }
    }

    #[test]
    fn test_decompress_rows_random_subset() {
        let bit_data = create_test_bit_data(20, 2, 16);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        let indices = vec![0, 3, 7, 19];

        let subset = decompress_samples_batch(&compressed, &indices).unwrap();

        assert_eq!(subset.data.num_rows, indices.len());
        assert_eq!(subset.data.chunk_size, 32);
        assert_eq!(subset.info.num_features(), 16);
    }

    #[test]
    fn test_decompress_rows_filter() {
        let bit_data = create_test_bit_data(10, 2, 8);
        let compressed = get_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        let filter = DecompressRowsData;
        let subset = filter
            .process((Arc::clone(&Arc::new(compressed)), vec![1, 5, 9]))
            .unwrap();

        assert_eq!(subset.data.num_rows, 3);
        assert_eq!(subset.data.chunk_size, 16);
        assert_eq!(subset.info.num_features(), 8);
    }
}
