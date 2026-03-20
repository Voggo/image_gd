use crate::compression::base_bits::{
    BaseBit, BaseBitBatchGroups, BaseBitGroups, BaseBitIncSignatureGroups, BaseBitSignatureGroups,
};
use crate::compression::tabular_preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use crate::utils::bits_needed_nonzero;

pub(crate) fn calculate_compressed_size<B: BaseBit>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> usize {
    let n = bit_data.num_rows();
    let d = bit_data.num_features();
    let n_b = base_bit_groups.get_num_bases();
    let chunk_size = bit_data.chunk_size();

    let len_b = base_bit_groups.get_num_bits_per_base().min(chunk_size);
    let len_d = chunk_size - len_b;

    let len_bc = if n == 0 { 0 } else { bits_needed_nonzero(n) };
    let len_id = if n_b == 0 {
        0
    } else {
        bits_needed_nonzero(n_b)
    };

    let size_bases = n_b * len_b;
    let size_base_counts = n_b * len_bc;
    let size_deviations = n * (len_d + len_id);

    let size_dev_bits = chunk_size;
    let size_params = 16 * d + 16 + size_dev_bits;

    size_bases + size_base_counts + size_deviations + size_params
}

pub struct SelectBases {
    pub patience: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchedBaseBitImpl {
    BatchGroups,
    IncSignatureGroups,
    SignatureGroups,
}

impl Filter for SelectBases {
    type Input = (BitDataSet, Vec<(usize, f64)>);
    type Output = (BitDataSet, Box<dyn BaseBit>);

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting base bits with patience {}",
            self.patience
        ));
        let (bit_data, entropy) = input;
        let base_bit_groups = select_base_bits(&bit_data, entropy, self.patience);
        Ok((bit_data, Box::new(base_bit_groups)))
    }
}

fn select_base_bits(
    bit_data: &BitDataSet,
    mut entropy: Vec<(usize, f64)>,
    patience: usize,
) -> BaseBitGroups {
    let mut non_improving_count = 0usize;
    let mut base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());

    entropy.sort_by(|a, b| a.1.total_cmp(&b.1));
    tracing::debug!(
        entropy = ?entropy,
        "Sorted entropies (bit position, entropy value)"
    );
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
        if best_base_bit_groups.get_num_bits_per_base()
            >= (bit_data.chunk_size() as f64 * 1.0) as usize
        {
            break;
        }

        trial_base_bit_groups.add_bit_position(bit_data, bit_position);
        let trial_compressed_size = calculate_compressed_size(bit_data, &trial_base_bit_groups);

        tracing::debug!(
            bit_position,
            trial_compressed_size,
            best_compressed_size,
            "evaluated trial base bit"
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
    let selected_mask = best_base_bit_groups
        .get_base_bit_mask()
        .iter()
        .map(|b| if *b { "1" } else { "0" })
        .collect::<String>();
    tracing::info!(
        selected_num_bases = best_base_bit_groups.get_num_bases(),
        selected_num_bits_per_base = best_base_bit_groups.get_num_bits_per_base(),
        selected_mask = %selected_mask,
        selected_compressed_size_bytes = best_compressed_size / 8,
        "selected base bit mask (compressed size in bytes)"
    );
    best_base_bit_groups
}

pub struct SelectBasesOptimized {
    pub patience: usize,
    pub base_bit_impl: BatchedBaseBitImpl,
}

impl Filter for SelectBasesOptimized {
    type Input = (BitDataSet, Vec<(usize, f64)>);
    type Output = (BitDataSet, Box<dyn BaseBit>);

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting base bits in batches with patience {}",
            self.patience
        ));
        let (bit_data, entropy) = input;
        let base_bit_groups = match self.base_bit_impl {
            BatchedBaseBitImpl::BatchGroups => select_base_bits_threshold_optimized(
                &bit_data,
                BaseBitBatchGroups::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy,
                0.80,
                self.patience,
            ),
            BatchedBaseBitImpl::IncSignatureGroups => select_base_bits_threshold_optimized(
                &bit_data,
                BaseBitIncSignatureGroups::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy,
                0.80,
                self.patience,
            ),
            BatchedBaseBitImpl::SignatureGroups => select_base_bits_threshold_optimized(
                &bit_data,
                BaseBitSignatureGroups::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy,
                0.80,
                self.patience,
            ),
        };
        Ok((bit_data, base_bit_groups))
    }
}

#[allow(dead_code)]
fn select_base_bits_optimized(
    bit_data: &BitDataSet,
    mut base_bit_groups: impl BaseBit + Clone + 'static,
    mut entropy: Vec<(usize, f64)>,
    patience: usize,
) -> Box<dyn BaseBit> {
    let mut non_improving_count = 0usize;

    entropy.sort_by(|a, b| a.1.total_cmp(&b.1));
    tracing::debug!(
        entropy = ?entropy,
        "Sorted entropies (bit position, entropy value)"
    );
    let zero_entropy_bits: Vec<usize> = entropy
        .iter()
        .take_while(|&(_bit_position, entropy_val)| *entropy_val == 0.0)
        .map(|(bit_position, _)| *bit_position)
        .collect();
    base_bit_groups.add_constant_bit_positions(&zero_entropy_bits);

    let mut best_base_bit_groups = base_bit_groups.clone();
    let mut best_compressed_size = calculate_compressed_size(bit_data, &best_base_bit_groups);
    let mut trial_base_bit_groups = base_bit_groups.clone();

    for i in (0..entropy.len()).step_by(3) {
        let bit_positions: Vec<usize> = entropy
            .iter()
            .skip(zero_entropy_bits.len())
            .skip(i)
            .take(3)
            .map(|(pos, _)| *pos)
            .collect();

        if bit_positions.is_empty() {
            break;
        }

        trial_base_bit_groups.add_bit_positions(bit_data, &bit_positions);
        let trial_compressed_size = calculate_compressed_size(bit_data, &trial_base_bit_groups);

        tracing::debug!(
            bit_positions = ?bit_positions,
            trial_compressed_size,
            best_compressed_size,
            "evaluated trial base-bit batch"
        );

        if trial_compressed_size < best_compressed_size {
            best_compressed_size = trial_compressed_size;
            best_base_bit_groups = trial_base_bit_groups.clone();
            non_improving_count = 0;
        } else {
            non_improving_count += 1;
        }

        if non_improving_count >= patience / 3 {
            break;
        }
    }
    let selected_mask = best_base_bit_groups
        .get_base_bit_mask()
        .iter()
        .map(|b| if *b { "1" } else { "0" })
        .collect::<String>();
    tracing::info!(
        selected_num_bases = best_base_bit_groups.get_num_bases(),
        selected_num_bits_per_base = best_base_bit_groups.get_num_bits_per_base(),
        selected_mask = %selected_mask,
        selected_compressed_size_bytes = best_compressed_size / 8,
        "selected base bit mask (optimized, compressed size in bytes)"
    );
    Box::new(best_base_bit_groups)
}

fn select_base_bits_threshold_optimized(
    bit_data: &BitDataSet,
    mut base_bit_groups: impl BaseBit + Clone + 'static,
    mut entropy: Vec<(usize, f64)>,
    entropy_threshold: f64,
    patience: usize,
) -> Box<dyn BaseBit> {
    let mut non_improving_count = 0usize;

    entropy.sort_by(|a, b| a.1.total_cmp(&b.1));
    tracing::debug!(
        entropy = ?entropy,
        "Sorted entropies (bit position, entropy value)"
    );

    let threshold_bits: Vec<usize> = entropy
        .iter()
        .take_while(|&(_bit_position, entropy_val)| *entropy_val <= entropy_threshold)
        .map(|(bit_position, _)| *bit_position)
        .collect();
    let num_threshold_bits = threshold_bits.len();

    let zero_entropy_bits: Vec<usize> = threshold_bits
        .iter()
        .copied()
        .take_while(|&pos| {
            entropy
                .iter()
                .find(|(p, _)| *p == pos)
                .map(|(_, e)| *e == 0.0)
                .unwrap_or(false)
        })
        .collect();
    base_bit_groups.add_constant_bit_positions(&zero_entropy_bits);

    let low_entropy_bits: Vec<usize> = threshold_bits
        .iter()
        .copied()
        .skip(zero_entropy_bits.len())
        .collect();
    if !low_entropy_bits.is_empty() {
        base_bit_groups.add_bit_positions(bit_data, &low_entropy_bits);
    }

    tracing::info!(
        num_threshold_bits,
        entropy_threshold,
        "bulk-added bits at or below entropy threshold"
    );

    let mut best_base_bit_groups = base_bit_groups.clone();
    let mut best_compressed_size = calculate_compressed_size(bit_data, &best_base_bit_groups);
    let mut trial_base_bit_groups = base_bit_groups.clone();

    for &(bit_position, _) in entropy.iter().skip(num_threshold_bits) {
        trial_base_bit_groups.add_bit_positions(bit_data, &[bit_position]);
        let trial_compressed_size = calculate_compressed_size(bit_data, &trial_base_bit_groups);

        tracing::debug!(
            bit_position,
            trial_compressed_size,
            best_compressed_size,
            "evaluated trial base-bit"
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
    let selected_mask = best_base_bit_groups
        .get_base_bit_mask()
        .iter()
        .map(|b| if *b { "1" } else { "0" })
        .collect::<String>();
    tracing::info!(
        selected_num_bases = best_base_bit_groups.get_num_bases(),
        selected_num_bits_per_base = best_base_bit_groups.get_num_bits_per_base(),
        selected_mask = %selected_mask,
        selected_compressed_size_bytes = best_compressed_size / 8,
        "selected base bit mask (threshold optimized, compressed size in bytes)"
    );
    Box::new(best_base_bit_groups)
}
