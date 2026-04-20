use crate::compression::preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;

#[derive(Debug, Clone)]
pub struct EntropyScoredContext {
    pub bit_data: BitDataSet,
    pub entropy_scores: Vec<(usize, f64)>,
}

impl EntropyScoredContext {
    pub fn new(bit_data: BitDataSet, entropy_scores: Vec<(usize, f64)>) -> Self {
        Self {
            bit_data,
            entropy_scores,
        }
    }
}

pub struct EntropyNaive;

impl Filter for EntropyNaive {
    type Input = BitDataSet;
    type Output = EntropyScoredContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Calculating entropy for each bit position");
        let entropy = calculate_entropy(&input);
        Ok(EntropyScoredContext::new(input, entropy))
    }
}

pub fn calculate_entropy(bit_data: &BitDataSet) -> Vec<(usize, f64)> {
    calculate_entropy_stride_sampled_naive(bit_data, 0)
}

/// [Deprecated] Too cheap to compute and limits future use of entropy
/// Entropy filter that computes entropy from a stride-sampled subset of rows
/// using the naive per-bit/per-row approach.
///
/// `skip_rows` controls how many rows are skipped between sampled rows.
/// - `skip_rows = 0` samples every row.
/// - `skip_rows = 1` samples rows `0, 2, 4, ...`.
/// - `skip_rows = n` samples every `n + 1`th row.
pub struct EntropyStrideSampled {
    pub skip_rows: usize,
}

impl EntropyStrideSampled {
    pub fn new(skip_rows: usize) -> Self {
        Self { skip_rows }
    }
}

impl Filter for EntropyStrideSampled {
    type Input = BitDataSet;
    type Output = EntropyScoredContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        tracing::warn!("This filter is deprecated and will not be used as it limits the ability to use entropy for future calculations. 
        (constant bit layout and base table column entropy ordering).
        Use the non stride sampled versions instead");
        let _timer =
            ScopedTimer::info("Calculating entropy for each bit position (stride sampled naive)");
        let entropy = calculate_entropy_stride_sampled_naive(&input, self.skip_rows);
        Ok(EntropyScoredContext::new(input, entropy))
    }
}

pub fn calculate_entropy_stride_sampled_naive(
    bit_data: &BitDataSet,
    skip_rows: usize,
) -> Vec<(usize, f64)> {
    let num_rows = bit_data.num_rows();
    let chunk_size = bit_data.chunk_size();
    if num_rows == 0 {
        return (0..chunk_size).map(|bit| (bit, 0.0)).collect();
    }

    let stride = skip_rows.saturating_add(1);
    let mut sampled_rows = 0usize;

    for _ in (0..num_rows).step_by(stride) {
        sampled_rows += 1;
    }

    if sampled_rows == 0 {
        return (0..chunk_size).map(|bit| (bit, 0.0)).collect();
    }

    let inv_rows = 1.0 / sampled_rows as f64;
    let mut out = Vec::with_capacity(chunk_size);

    for bit in 0..chunk_size {
        let mut count_ones = 0usize;
        for row in (0..num_rows).step_by(stride) {
            if unsafe { bit_data.get_bit_unchecked(row, bit) } {
                count_ones += 1;
            }
        }

        let entropy = if count_ones == 0 || count_ones == sampled_rows {
            0.0
        } else {
            let p = count_ones as f64 * inv_rows;
            -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
        };
        out.push((bit, entropy));
    }

    out
}

pub struct EntropyBatched {}

impl Filter for EntropyBatched {
    type Input = BitDataSet;
    type Output = EntropyScoredContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Calculating entropy for each bit position (optimized)");
        let entropy = calculate_entropy_optimized(&input);
        Ok(EntropyScoredContext::new(input, entropy))
    }
}

pub fn calculate_entropy_optimized(bit_data: &BitDataSet) -> Vec<(usize, f64)> {
    calculate_entropy_with_stride(bit_data, 1)
}

/// [Deprecated] Too cheap to compute and limits future use of entropy
/// Entropy filter that computes entropy from a stride-sampled subset of rows.
///
/// `skip_rows` controls how many rows are skipped between sampled rows.
/// - `skip_rows = 0` samples every row.
/// - `skip_rows = 1` samples rows `0, 2, 4, ...`.
/// - `skip_rows = n` samples every `n + 1`th row.
pub struct EntropyStrideSampledBatched {
    pub skip_rows: usize,
}

impl EntropyStrideSampledBatched {
    pub fn new(skip_rows: usize) -> Self {
        Self { skip_rows }
    }
}

impl Filter for EntropyStrideSampledBatched {
    type Input = BitDataSet;
    type Output = EntropyScoredContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        tracing::warn!("This filter is deprecated and will not be used as it limits the ability to use entropy for future calculations. 
        (constant bit layout and base table column entropy ordering).
        Use the non stride sampled versions instead");
        let _timer =
            ScopedTimer::info("Calculating entropy for each bit position (stride sampled)");
        let entropy = calculate_entropy_stride_sampled(&input, self.skip_rows);
        Ok(EntropyScoredContext::new(input, entropy))
    }
}

pub fn calculate_entropy_stride_sampled(
    bit_data: &BitDataSet,
    skip_rows: usize,
) -> Vec<(usize, f64)> {
    let stride = skip_rows.saturating_add(1);
    calculate_entropy_with_stride(bit_data, stride)
}

fn calculate_entropy_with_stride(bit_data: &BitDataSet, stride: usize) -> Vec<(usize, f64)> {
    let num_rows = bit_data.num_rows();
    let chunk_size = bit_data.chunk_size();
    if num_rows == 0 {
        return (0..chunk_size).map(|bit| (bit, 0.0)).collect();
    }
    let stride = stride.max(1);

    let mut ones_count = vec![0usize; chunk_size];

    // Trying to minimize bounds checks by using iterators
    for row in (0..num_rows).step_by(stride) {
        let row_bits = unsafe { bit_data.get_chunk_unchecked(row) };
        for (count, bit) in ones_count.iter_mut().zip(row_bits.iter()) {
            *count += *bit as usize;
        }
    }
    let sampled_rows = (num_rows + stride - 1) / stride;
    let inv_rows = 1.0 / sampled_rows as f64;
    ones_count
        .into_iter()
        .enumerate()
        .map(|(bit, count)| {
            let entropy = if count == 0 || count == sampled_rows {
                0.0
            } else {
                let p = count as f64 * inv_rows;
                -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
            };
            (bit, entropy)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::preprocessor::{BitData, BitDataInfo, BitDataSet, FeatureSpec};
    use crate::data_loader::FeatureDataType;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_entropy_calculation_1() {
        // Create a simple BitData for testing
        let num_rows = 4;
        let chunk_size = 3;
        let num_features = 3;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, // Row 0
            true, true, false, // Row 1
            false, false, true, // Row 2
            false, true, false, // Row 3
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropies = calculate_entropy(&bit_data);
        tracing::info!("Entropies: {:?}", entropies);
        assert_eq!(entropies.len(), chunk_size);
        // Add more assertions based on expected entropy values
    }

    #[test]
    fn test_entropy_calculation_2() {
        // Create a simple BitData for testing
        let num_rows = 6;
        let chunk_size = 8;
        let num_features = 8;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, false, true, true, false, // Row 0
            true, true, false, true, true, true, false, false, // Row 1
            false, true, true, false, true, false, true, true, // Row 2
            true, false, false, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false, // Row 5
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropies = calculate_entropy(&bit_data);
        let expected_entropies = vec![
            (0, 0.9183),
            (1, 0.9183),
            (2, 0.9183),
            (3, 0.9183),
            (4, 0.65002),
            (5, 0.65002),
            (6, 0.9183),
            (7, 1.0),
        ];
        tracing::info!("Entropies: {:?}", entropies);
        assert_eq!(entropies.len(), chunk_size);
        assert_eq!(
            entropies
                .iter()
                .map(|(i, e)| (*i, (e * 100_000.0).round() / 100_000.0))
                .collect::<Vec<_>>(),
            expected_entropies
        );
    }

    #[test]
    fn test_pipeline_with_entropy() {
        // Create a simple BitData for testing
        let num_rows = 6;
        let chunk_size = 8;
        let num_features = 8;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, false, true, true, false, // Row 0
            true, true, false, true, true, true, false, false, // Row 1
            false, true, true, false, true, false, true, true, // Row 2
            true, false, false, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false, // Row 5
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropy_filter = EntropyNaive;
        let entropy_scored = entropy_filter.process(bit_data).unwrap();
        tracing::info!("Entropies from pipeline: {:?}", entropy_scored.entropy_scores);
        assert_eq!(entropy_scored.entropy_scores.len(), chunk_size);
    }

    #[test]
    fn test_stride_sampled_entropy_skip_zero_matches_optimized() {
        let num_rows = 6;
        let chunk_size = 8;
        let num_features = 8;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, false, true, true, false, // Row 0
            true, true, false, true, true, true, false, false, // Row 1
            false, true, true, false, true, false, true, true, // Row 2
            true, false, false, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false, // Row 5
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let expected = calculate_entropy_optimized(&bit_data);
        let sampled = calculate_entropy_stride_sampled(&bit_data, 0);
        assert_eq!(sampled, expected);
    }

    #[test]
    fn test_stride_sampled_entropy_skip_one_expected_values() {
        let num_rows = 6;
        let chunk_size = 8;
        let num_features = 8;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, false, true, true, false, // Row 0
            true, true, false, true, true, true, false, false, // Row 1
            false, true, true, false, true, false, true, true, // Row 2
            true, false, false, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false, // Row 5
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        // skip_rows = 1 samples rows: 0, 2, 4
        let entropies = calculate_entropy_stride_sampled(&bit_data, 1);
        let expected = vec![
            (0, 0.9183),
            (1, 0.9183),
            (2, 0.9183),
            (3, 0.0),
            (4, 0.9183),
            (5, 0.9183),
            (6, 0.9183),
            (7, 0.9183),
        ];

        assert_eq!(entropies.len(), chunk_size);
        assert_eq!(
            entropies
                .iter()
                .map(|(i, e)| (*i, (e * 100_000.0).round() / 100_000.0))
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn test_pipeline_with_stride_sampled_entropy() {
        let num_rows = 6;
        let chunk_size = 8;
        let num_features = 8;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, false, true, true, false, // Row 0
            true, true, false, true, true, true, false, false, // Row 1
            false, true, true, false, true, false, true, true, // Row 2
            true, false, false, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false, // Row 5
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropy_filter = EntropyStrideSampledBatched::new(1);
        let entropy_scored = entropy_filter.process(bit_data).unwrap();
        assert_eq!(entropy_scored.entropy_scores.len(), chunk_size);
    }

    #[test]
    fn test_stride_sampled_naive_skip_zero_matches_naive() {
        let num_rows = 6;
        let chunk_size = 8;
        let num_features = 8;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, false, true, true, false, // Row 0
            true, true, false, true, true, true, false, false, // Row 1
            false, true, true, false, true, false, true, true, // Row 2
            true, false, false, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false, // Row 5
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let expected = calculate_entropy(&bit_data);
        let sampled = calculate_entropy_stride_sampled_naive(&bit_data, 0);
        assert_eq!(sampled, expected);
    }

    #[test]
    fn test_pipeline_with_stride_sampled_naive_entropy() {
        let num_rows = 6;
        let chunk_size = 8;
        let num_features = 8;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true, false, false, true, true, false, // Row 0
            true, true, false, true, true, true, false, false, // Row 1
            false, true, true, false, true, false, true, true, // Row 2
            true, false, false, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false, // Row 5
        ];
        let data = BitData {
            data: data.into_iter().collect(),
            chunk_size,
            stride: chunk_size,
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropy_filter = EntropyStrideSampled::new(1);
        let entropy_scored = entropy_filter.process(bit_data).unwrap();
        assert_eq!(entropy_scored.entropy_scores.len(), chunk_size);
    }
}
