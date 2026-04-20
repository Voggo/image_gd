use crate::compression::base_bits::{
    BaseBit, BaseBitBatchGroups, BaseBitGroups, BaseBitHyperLogLogCount, BaseBitIncSignatureGroups,
    BaseBitSignatureGroups,
};
use crate::compression::entropy::EntropyScoredContext;
use crate::compression::preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use crate::utils::bits_needed_nonzero;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompressedSizeBreakdown {
    pub total_size: usize,
    pub num_bases: usize,
    pub size_bases: usize,
    pub size_deviations: usize,
    pub size_ids: usize,
    pub size_params: usize,
}

pub(crate) fn calculate_compressed_size_breakdown<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> CompressedSizeBreakdown {
    let n = bit_data.num_rows();
    let d = bit_data.num_features();
    let chunk_size = bit_data.chunk_size();
    let n_b = base_bit_groups.get_num_bases();

    let len_b = base_bit_groups.get_num_bits_per_base();
    let len_d = chunk_size - len_b;

    let len_id = if n_b == 0 {
        0
    } else {
        bits_needed_nonzero(n_b)
    };

    let size_bases = n_b * len_b;
    let size_deviations = n * len_d;
    let size_ids = n * len_id;
    let size_dev_bits = chunk_size;
    let size_params = 16 * d + 16 + size_dev_bits; // should revisit this
    let total_size = size_bases + size_deviations + size_ids + size_params;

    CompressedSizeBreakdown {
        total_size,
        num_bases: n_b,
        size_bases,
        size_deviations,
        size_ids,
        size_params,
    }
}

pub(crate) fn calculate_compressed_size<B: BaseBit + ?Sized>(
    bit_data: &BitDataSet,
    base_bit_groups: &B,
) -> usize {
    calculate_compressed_size_breakdown(bit_data, base_bit_groups).total_size
}

struct SelectBasesCsvLogger {
    writer: Option<BufWriter<File>>,
}

impl SelectBasesCsvLogger {
    fn new(output_path: Option<&Path>) -> Self {
        let Some(path) = output_path else {
            return Self { writer: None };
        };

        let selected_path = Self::next_available_path(path);
        let file = match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&selected_path)
        {
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(path = %selected_path.display(), ?error, "failed to create select-bases debug csv");
                return Self { writer: None };
            }
        };

        let mut writer = BufWriter::new(file);
        if let Err(error) = writeln!(
            writer,
            "bit_position,num_bases,size_bases,size_deviations,size_ids,size_params,total_size"
        ) {
            tracing::warn!(path = %selected_path.display(), ?error, "failed to write select-bases csv header");
            return Self { writer: None };
        }

        if selected_path != path {
            tracing::info!(
                requested_path = %path.display(),
                selected_path = %selected_path.display(),
                "select-bases debug csv path already existed; using incremented suffix"
            );
        }

        Self {
            writer: Some(writer),
        }
    }

    fn next_available_path(path: &Path) -> PathBuf {
        if !path.exists() {
            return path.to_path_buf();
        }

        let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "select_bases_debug".to_string());
        let extension = path.extension().map(|e| e.to_string_lossy().into_owned());

        for suffix in 1..=usize::MAX {
            let file_name = match &extension {
                Some(ext) => format!("{}_{}.{}", stem, suffix, ext),
                None => format!("{}_{}", stem, suffix),
            };
            let candidate = parent.join(file_name);
            if !candidate.exists() {
                return candidate;
            }
        }

        path.to_path_buf()
    }

    fn log_row(&mut self, bit_position: usize, breakdown: CompressedSizeBreakdown) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };

        if let Err(error) = writeln!(
            writer,
            "{},{},{},{},{},{},{}",
            bit_position,
            breakdown.num_bases,
            breakdown.size_bases,
            breakdown.size_deviations,
            breakdown.size_ids,
            breakdown.size_params,
            breakdown.total_size
        ) {
            tracing::warn!(
                ?error,
                "failed to write select-bases csv row; disabling csv logging"
            );
            self.writer = None;
        }
    }
}

pub struct SelectBases {
    pub patience: usize,
}

pub struct BaseSelectionContext {
    pub bit_data: BitDataSet,
    pub base_bits: Box<dyn BaseBit>,
}

impl BaseSelectionContext {
    pub fn new(bit_data: BitDataSet, base_bits: Box<dyn BaseBit>) -> Self {
        Self {
            bit_data,
            base_bits,
        }
    }
}

/// Profiling-oriented selector that adds every bit position as base bits.
///
/// By default (`split_into_batches = 1`), all positions are added in a single
/// batch call. Set `split_into_batches > 1` to split the full set of bit
/// positions into that many smaller additions.
pub struct SelectBasesProfileAllBits {
    /// Desired number of additions used to add all bit positions.
    ///
    /// - `0` and `1` both behave as a single batch add.
    /// - Values larger than the number of bit positions are clamped.
    pub split_into_batches: usize,
    pub base_bit_impl: BaseBitImpl,
}

pub struct SelectBasesDebug {
    pub patience: usize,
    pub debug_csv_paths: Mutex<Vec<PathBuf>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseBitImpl {
    Naive,
    BatchGroups,
    IncSignatureGroups,
    SignatureGroups,
    HyperLogLogCount,
}

impl Filter for SelectBases {
    type Input = EntropyScoredContext;
    type Output = BaseSelectionContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting base bits with patience {}",
            self.patience
        ));
        let EntropyScoredContext {
            bit_data,
            entropy_scores,
        } = input;
        let base_bit_groups = select_base_bits(&bit_data, entropy_scores, self.patience);
        Ok(BaseSelectionContext::new(
            bit_data,
            Box::new(base_bit_groups),
        ))
    }
}

impl Filter for SelectBasesProfileAllBits {
    type Input = EntropyScoredContext;
    type Output = BaseSelectionContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting all base bits for profiling (split_into_batches={}, base_bit_impl={:?})",
            self.split_into_batches, self.base_bit_impl
        ));

        let EntropyScoredContext { bit_data, .. } = input;
        let chunk_size = bit_data.chunk_size();
        let all_bit_positions: Vec<usize> = (0..chunk_size).collect();
        let mut base_bit_groups: Box<dyn BaseBit> = match self.base_bit_impl {
            BaseBitImpl::Naive => Box::new(BaseBitGroups::new(bit_data.num_rows(), chunk_size)),
            BaseBitImpl::BatchGroups => {
                Box::new(BaseBitBatchGroups::new(bit_data.num_rows(), chunk_size))
            }
            BaseBitImpl::IncSignatureGroups => Box::new(BaseBitIncSignatureGroups::new(
                bit_data.num_rows(),
                chunk_size,
            )),
            BaseBitImpl::SignatureGroups => {
                Box::new(BaseBitSignatureGroups::new(bit_data.num_rows(), chunk_size))
            }
            BaseBitImpl::HyperLogLogCount => Box::new(BaseBitHyperLogLogCount::new(
                bit_data.num_rows(),
                chunk_size,
            )),
        };

        if !all_bit_positions.is_empty() {
            let requested_batches = self.split_into_batches.max(1);
            let batch_count = requested_batches.min(all_bit_positions.len());

            if batch_count == 1 {
                base_bit_groups
                    .as_mut()
                    .add_bit_positions(&bit_data, &all_bit_positions);
            } else {
                let batch_size = all_bit_positions.len().div_ceil(batch_count);
                for batch in all_bit_positions.chunks(batch_size) {
                    base_bit_groups.as_mut().add_bit_positions(&bit_data, batch);
                }
            }
        }

        let compressed_size = calculate_compressed_size(&bit_data, base_bit_groups.as_ref());
        let selected_mask = base_bit_groups
            .get_base_bit_mask()
            .iter()
            .map(|b| if *b { "1" } else { "0" })
            .collect::<String>();
        tracing::info!(
            selected_num_bases = base_bit_groups.get_num_bases(),
            selected_num_bits_per_base = base_bit_groups.get_num_bits_per_base(),
            selected_mask = %selected_mask,
            selected_compressed_size_bytes = compressed_size / 8,
            "selected base bit mask (profile all bits, compressed size in bytes)"
        );

        Ok(BaseSelectionContext::new(bit_data, base_bit_groups))
    }
}

impl Filter for SelectBasesDebug {
    type Input = EntropyScoredContext;
    type Output = BaseSelectionContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting base bits with CSV debug and patience {}",
            self.patience
        ));
        let EntropyScoredContext {
            bit_data,
            entropy_scores,
        } = input;

        let selected_debug_csv_path = {
            let guard = self
                .debug_csv_paths
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.first().cloned()
        };

        if selected_debug_csv_path.is_none() {
            tracing::warn!(
                "SelectBasesDebug has no remaining debug CSV paths; running without CSV logging"
            );
        }

        let base_bit_groups = select_base_bits_debug(
            &bit_data,
            entropy_scores,
            self.patience,
            selected_debug_csv_path.as_deref(),
        );

        if let Some(selected_path) = selected_debug_csv_path {
            let mut guard = self
                .debug_csv_paths
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(pos) = guard.iter().position(|p| *p == selected_path) {
                guard.remove(pos);
            }
        }

        Ok(BaseSelectionContext::new(
            bit_data,
            Box::new(base_bit_groups),
        ))
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

fn select_base_bits_debug(
    bit_data: &BitDataSet,
    mut entropy: Vec<(usize, f64)>,
    patience: usize,
    debug_csv_path: Option<&Path>,
) -> BaseBitGroups {
    let mut non_improving_count = 0usize;
    let mut base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
    let mut csv_logger = SelectBasesCsvLogger::new(debug_csv_path);

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
    for &bit_position in &zero_entropy_bits {
        base_bit_groups.add_constant_bit_positions(&[bit_position]);
        let breakdown = calculate_compressed_size_breakdown(bit_data, &base_bit_groups);
        csv_logger.log_row(bit_position, breakdown);
    }

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
        let trial_breakdown = calculate_compressed_size_breakdown(bit_data, &trial_base_bit_groups);
        csv_logger.log_row(bit_position, trial_breakdown);
        let trial_compressed_size = trial_breakdown.total_size;

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
        "selected base bit mask (debug csv, compressed size in bytes)"
    );
    best_base_bit_groups
}

pub struct SelectBasesOptimized {
    pub patience: usize,
    pub base_bit_impl: BaseBitImpl,
}

impl Filter for SelectBasesOptimized {
    type Input = EntropyScoredContext;
    type Output = BaseSelectionContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!(
            "Selecting base bits in batches with patience {}",
            self.patience
        ));
        let EntropyScoredContext {
            bit_data,
            entropy_scores,
        } = input;
        let base_bit_groups = match self.base_bit_impl {
            BaseBitImpl::Naive => select_base_bits_threshold_optimized(
                &bit_data,
                BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy_scores,
                0.80,
                self.patience,
            ),
            BaseBitImpl::BatchGroups => select_base_bits_threshold_optimized(
                &bit_data,
                BaseBitBatchGroups::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy_scores,
                0.80,
                self.patience,
            ),
            BaseBitImpl::IncSignatureGroups => select_base_bits_threshold_optimized(
                &bit_data,
                BaseBitIncSignatureGroups::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy_scores,
                0.80,
                self.patience,
            ),
            BaseBitImpl::SignatureGroups => select_base_bits_threshold_optimized(
                &bit_data,
                BaseBitSignatureGroups::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy_scores,
                0.80,
                self.patience,
            ),
            BaseBitImpl::HyperLogLogCount => select_base_bits_threshold_optimized(
                &bit_data,
                BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size()),
                entropy_scores,
                0.80,
                self.patience,
            ),
        };
        Ok(BaseSelectionContext::new(bit_data, base_bit_groups))
    }
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
