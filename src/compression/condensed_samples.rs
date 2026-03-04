use crate::compression::base_bits::BaseBitGroups;
use crate::compression::compress::CondensedSamples;
use crate::compression::tabular_preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use bitvec::prelude::*;

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
    pub m_max: usize,
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

    let zero_entropy_bits: Vec<usize> = organized_entropy_by_feature
        .iter()
        .take_while(|&(_bit_position, entropy_val)| *entropy_val == 0.0)
        .map(|(bit_position, _)| *bit_position)
        .collect();
    condensed_bit_groups.add_constant_bit_positions(&zero_entropy_bits);

    for &(bit_position, _entropy_val) in organized_entropy_by_feature
        .iter()
        .skip(zero_entropy_bits.len())
    {
        if condensed_bit_groups.get_num_bases() >= m_max {
            break;
        }
        condensed_bit_groups.add_bit_position(bit_data, bit_position);
        tracing::debug!(
            bit_position,
            current_num_bases = condensed_bit_groups.get_num_bases(),
            m_max,
            "added bit position to condensed samples"
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
    bit_data
        .info
        .set_condensed_sample_weights(Some(condensed_samples.weights));

    for sample in condensed_samples.samples {
        bit_data
            .data
            .extend_from_bitslice(sample.as_bitslice())
            .expect("condensed sample chunk size must match dataset chunk size");
    }
    bit_data
}
