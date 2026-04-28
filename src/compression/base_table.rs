use std::cmp::Ordering;

use crate::compression::base_bits::BaseBit;
use crate::compression::base_selection::BaseSelectionContext;
use crate::compression::entropy::ConstantBitPolarity;
use crate::compression::preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use bitvec::prelude::*;

#[derive(Clone)]
pub struct PreEncodeContext {
    pub bit_data: BitDataSet,
    pub row_to_base_id: Vec<usize>,
    pub layout: BaseLayoutInfo,
    pub variable_base_table: Vec<(BitVec<usize, Lsb0>, usize)>,
    /// Column indices into `variable_base_table` rows, ordered by ascending
    /// unweighted entropy as used by `BuildSortedBaseTable`.
    pub entropy_sorted_column_order: Option<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseLayoutInfo {
    pub bit_states: Vec<BaseBitLayoutState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseBitLayoutState {
    Deviation,
    Variable,
    ConstantZero,
    ConstantOne,
}

impl BaseLayoutInfo {
    pub fn from_bit_states(bit_states: Vec<BaseBitLayoutState>) -> Self {
        Self { bit_states }
    }

    pub fn chunk_size(&self) -> usize {
        self.bit_states.len()
    }

    pub fn state_at(&self, bit_position: usize) -> BaseBitLayoutState {
        self.bit_states
            .get(bit_position)
            .copied()
            .unwrap_or(BaseBitLayoutState::Deviation)
    }

    pub fn selected_base_bit_positions(&self) -> Vec<usize> {
        self.bit_states
            .iter()
            .enumerate()
            .filter_map(|(bit_position, state)| {
                if matches!(
                    state,
                    BaseBitLayoutState::Variable
                        | BaseBitLayoutState::ConstantZero
                        | BaseBitLayoutState::ConstantOne
                ) {
                    Some(bit_position)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn deviation_bit_positions(&self) -> Vec<usize> {
        self.bit_states
            .iter()
            .enumerate()
            .filter_map(|(bit_position, state)| {
                if *state == BaseBitLayoutState::Deviation {
                    Some(bit_position)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn variable_base_bit_positions(&self) -> Vec<usize> {
        self.bit_states
            .iter()
            .enumerate()
            .filter_map(|(bit_position, state)| {
                if *state == BaseBitLayoutState::Variable {
                    Some(bit_position)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn constant_zero_bit_positions(&self) -> Vec<usize> {
        self.bit_states
            .iter()
            .enumerate()
            .filter_map(|(bit_position, state)| {
                if *state == BaseBitLayoutState::ConstantZero {
                    Some(bit_position)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn constant_one_bit_positions(&self) -> Vec<usize> {
        self.bit_states
            .iter()
            .enumerate()
            .filter_map(|(bit_position, state)| {
                if *state == BaseBitLayoutState::ConstantOne {
                    Some(bit_position)
                } else {
                    None
                }
            })
            .collect()
    }
}

pub struct BuildBaseTable {}

impl Filter for BuildBaseTable {
    type Input = BaseSelectionContext;
    type Output = PreEncodeContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Building base table and layout from selected bases");
        let BaseSelectionContext {
            bit_data,
            base_bits,
            constant_bit_polarity,
        } = input;
        Ok(build_encode_context(
            bit_data,
            base_bits.as_ref(),
            &constant_bit_polarity,
            false,
        ))
    }
}

pub struct BuildSortedBaseTable {}

impl Filter for BuildSortedBaseTable {
    type Input = BaseSelectionContext;
    type Output = PreEncodeContext;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Building and sorting base table");
        let BaseSelectionContext {
            bit_data,
            base_bits,
            constant_bit_polarity,
        } = input;
        Ok(build_encode_context(
            bit_data,
            base_bits.as_ref(),
            &constant_bit_polarity,
            true,
        ))
    }
}

fn build_encode_context(
    bit_data: BitDataSet,
    base_bit_groups: &dyn BaseBit,
    constant_bit_polarity: &ConstantBitPolarity,
    sort_bases: bool,
) -> PreEncodeContext {
    let selected_base_table = base_bit_groups.get_bases(&bit_data);
    let base_bit_positions = base_bit_groups.get_base_bit_positions().to_vec();
    let layout = build_base_layout_from_constant_polarity(
        bit_data.chunk_size(),
        &base_bit_positions,
        &constant_bit_polarity.constant_zero_bit_positions,
        &constant_bit_polarity.constant_one_bit_positions,
    );
    let variable_positions = layout.variable_base_bit_positions();
    let variable_base_table = project_selected_bases_to_variable(
        &base_bit_positions,
        &variable_positions,
        &selected_base_table,
    );

    let mut row_to_base_id = vec![0usize; bit_data.num_rows()];
    for (id, group) in base_bit_groups.get_groups().iter().enumerate() {
        for &row in group {
            row_to_base_id[row] = id;
        }
    }

    let mut context = PreEncodeContext {
        bit_data,
        row_to_base_id,
        layout,
        variable_base_table,
        entropy_sorted_column_order: None,
    };

    if sort_bases {
        sort_context_base_tables(&mut context);
    }

    context
}

pub(crate) fn build_base_layout_from_constant_polarity(
    chunk_size: usize,
    selected_positions: &[usize],
    constant_zero_positions: &[usize],
    constant_one_positions: &[usize],
) -> BaseLayoutInfo {
    let mut bit_states = vec![BaseBitLayoutState::Deviation; chunk_size];

    for &bit_position in selected_positions {
        if bit_position >= bit_states.len() {
            continue;
        }
        bit_states[bit_position] = if constant_one_positions.contains(&bit_position) {
            BaseBitLayoutState::ConstantOne
        } else if constant_zero_positions.contains(&bit_position) {
            BaseBitLayoutState::ConstantZero
        } else {
            BaseBitLayoutState::Variable
        };
    }

    BaseLayoutInfo::from_bit_states(bit_states)
}

pub(crate) fn project_selected_bases_to_variable(
    selected_positions: &[usize],
    variable_positions: &[usize],
    selected_bases: &[(BitVec<usize, Lsb0>, usize)],
) -> Vec<(BitVec<usize, Lsb0>, usize)> {
    let variable_indices: Vec<usize> = variable_positions
        .iter()
        .filter_map(|bit_pos| selected_positions.iter().position(|p| p == bit_pos))
        .collect();

    selected_bases
        .iter()
        .map(|(selected_bits, count)| {
            let mut variable_bits = BitVec::with_capacity(variable_indices.len());
            for selected_idx in variable_indices.iter().copied() {
                variable_bits.push(
                    selected_bits
                        .get(selected_idx)
                        .map(|bit| *bit)
                        .unwrap_or(false),
                );
            }
            (variable_bits, *count)
        })
        .collect()
}

fn sort_context_base_tables(context: &mut PreEncodeContext) {
    let _timer = ScopedTimer::trace("Sorting base table entries and remapping IDs");
    let base_count = context.variable_base_table.len();
    let column_order = column_order_by_unweighted_entropy(&context.variable_base_table);
    context.entropy_sorted_column_order = Some(column_order.clone());

    if base_count <= 1 {
        return;
    }

    let mut indices: Vec<usize> = (0..base_count).collect();
    indices.sort_by(|&lhs, &rhs| {
        compare_rows_by_column_order(
            &context.variable_base_table[lhs].0,
            &context.variable_base_table[rhs].0,
            &column_order,
        )
        .then_with(|| lhs.cmp(&rhs))
    });

    let mut old_to_new = vec![0usize; base_count];
    for (new_id, old_id) in indices.iter().copied().enumerate() {
        old_to_new[old_id] = new_id;
    }

    context.variable_base_table = indices
        .iter()
        .map(|&old_id| context.variable_base_table[old_id].clone())
        .collect();

    for id in &mut context.row_to_base_id {
        if let Some(&new_id) = old_to_new.get(*id) {
            *id = new_id;
        }
    }
}

fn column_order_by_unweighted_entropy(
    variable_base_table: &[(BitVec<usize, Lsb0>, usize)],
) -> Vec<usize> {
    let Some((first_bits, _)) = variable_base_table.first() else {
        return Vec::new();
    };

    let row_count = variable_base_table.len();
    let bit_len = first_bits.len();

    let mut columns_with_entropy: Vec<(usize, f64)> = (0..bit_len)
        .map(|column_idx| {
            let ones = variable_base_table
                .iter()
                .filter(|(bits, _)| bits.get(column_idx).map(|b| *b).unwrap_or(false))
                .count();
            let entropy = binary_entropy_from_counts(ones, row_count);
            (column_idx, entropy)
        })
        .collect();

    columns_with_entropy.sort_by(|(lhs_idx, lhs_entropy), (rhs_idx, rhs_entropy)| {
        lhs_entropy
            .partial_cmp(rhs_entropy)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs_idx.cmp(rhs_idx))
    });

    columns_with_entropy
        .into_iter()
        .map(|(column_idx, _)| column_idx)
        .collect()
}

fn compare_rows_by_column_order(
    lhs: &BitSlice<usize, Lsb0>,
    rhs: &BitSlice<usize, Lsb0>,
    column_order: &[usize],
) -> Ordering {
    for &column_idx in column_order {
        let l = lhs.get(column_idx).map(|bit| *bit).unwrap_or(false);
        let r = rhs.get(column_idx).map(|bit| *bit).unwrap_or(false);
        match r.cmp(&l) {
            Ordering::Equal => continue,
            non_equal => return non_equal,
        }
    }

    let len = lhs.len().min(rhs.len());
    for idx in 0..len {
        let l = lhs.get(idx).map(|bit| *bit).unwrap_or(false);
        let r = rhs.get(idx).map(|bit| *bit).unwrap_or(false);
        match r.cmp(&l) {
            Ordering::Equal => continue,
            non_equal => return non_equal,
        }
    }

    lhs.len().cmp(&rhs.len())
}

fn binary_entropy_from_counts(ones: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }

    if ones == 0 || ones == total {
        return 0.0;
    }

    let p = ones as f64 / total as f64;
    let q = 1.0 - p;
    -(p * p.log2()) - (q * q.log2())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::base_bits::BaseBitGroups;
    use crate::compression::entropy::ConstantBitPolarity;
    use crate::compression::preprocessor::{
        BitData, BitDataInfo, FeatureDataType, FeatureSpec, FeatureTransform,
    };

    fn create_test_bit_data_set(rows_and_bits: Vec<Vec<bool>>) -> BitDataSet {
        assert!(!rows_and_bits.is_empty(), "Must have at least one row");
        let chunk_size = rows_and_bits[0].len();
        let num_rows = rows_and_bits.len();

        let mut data = BitVec::with_capacity(num_rows * chunk_size);
        for row in &rows_and_bits {
            assert_eq!(row.len(), chunk_size, "All rows must have same bit width");
            for &bit in row {
                data.push(bit);
            }
        }

        let bit_data = BitData {
            data,
            chunk_size,
            stride: chunk_size,
            num_rows,
        };

        let features = vec![FeatureSpec {
            data_type: FeatureDataType::UnsignedInt,
            bits: chunk_size,
            transform: FeatureTransform::None,
        }];

        let info = BitDataInfo::new(features, num_rows * chunk_size)
            .expect("Failed to construct BitDataInfo");

        BitDataSet {
            data: bit_data,
            info,
        }
    }

    #[test]
    fn test_build_base_table_populates_layout_and_rows() {
        let bit_data = create_test_bit_data_set(vec![
            vec![false, false, false, true],
            vec![false, true, false, true],
            vec![true, false, true, true],
            vec![true, true, true, true],
        ]);

        let mut groups = BaseBitGroups::new(4, 4);
        groups.add_bit_position(&bit_data, 0);
        groups.add_bit_position(&bit_data, 1);
        groups.add_bit_position(&bit_data, 3);

        let context = BuildBaseTable {}
            .process(
                crate::compression::base_selection::BaseSelectionContext::new(
                    bit_data,
                    Box::new(groups),
                    ConstantBitPolarity {
                        constant_zero_bit_positions: Vec::new(),
                        constant_one_bit_positions: vec![3],
                    },
                ),
            )
            .expect("BuildBaseTable should succeed");

        assert_eq!(context.variable_base_table.len(), 4);
        assert_eq!(context.layout.selected_base_bit_positions(), vec![0, 1, 3]);
        assert_eq!(context.layout.variable_base_bit_positions(), vec![0, 1]);
        assert_eq!(context.layout.constant_one_bit_positions(), vec![3]);
        assert!(context.layout.constant_zero_bit_positions().is_empty());
        assert!(context.entropy_sorted_column_order.is_none());
        let mut ids = context.row_to_base_id.clone();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_build_sorted_base_table_remaps_ids_to_sorted_order() {
        let bit_data = create_test_bit_data_set(vec![
            vec![true, true, false, false],
            vec![false, false, false, false],
            vec![true, false, false, false],
            vec![false, true, false, false],
        ]);

        let mut groups = BaseBitGroups::new(4, 4);
        groups.add_bit_position(&bit_data, 0);
        groups.add_bit_position(&bit_data, 1);

        let unsorted = BuildBaseTable {}
            .process(
                crate::compression::base_selection::BaseSelectionContext::new(
                    bit_data.clone(),
                    Box::new(groups.clone()),
                    ConstantBitPolarity::default(),
                ),
            )
            .expect("BuildBaseTable should succeed");
        let sorted = BuildSortedBaseTable {}
            .process(
                crate::compression::base_selection::BaseSelectionContext::new(
                    bit_data,
                    Box::new(groups),
                    ConstantBitPolarity::default(),
                ),
            )
            .expect("BuildSortedBaseTable should succeed");

        assert_eq!(
            sorted.variable_base_table.len(),
            unsorted.variable_base_table.len()
        );
        assert!(unsorted.entropy_sorted_column_order.is_none());
        assert!(sorted.entropy_sorted_column_order.is_some());

        let mut unsorted_patterns = unsorted
            .variable_base_table
            .iter()
            .map(|(bits, _)| bits.clone())
            .collect::<Vec<_>>();
        let mut sorted_patterns = sorted
            .variable_base_table
            .iter()
            .map(|(bits, _)| bits.clone())
            .collect::<Vec<_>>();
        unsorted_patterns.sort();
        sorted_patterns.sort();
        assert_eq!(sorted_patterns, unsorted_patterns);
    }

    #[test]
    fn test_sorted_base_table_uses_column_entropy_priority_lexicographic_order() {
        let bit_data =
            create_test_bit_data_set(vec![vec![true], vec![true], vec![true], vec![true]]);

        let make_bits = |bits: &[bool]| {
            let mut out = BitVec::with_capacity(bits.len());
            for bit in bits {
                out.push(*bit);
            }
            out
        };

        let mut context = PreEncodeContext {
            bit_data,
            row_to_base_id: vec![0, 1, 2, 3],
            layout: BaseLayoutInfo::from_bit_states(vec![
                BaseBitLayoutState::Variable,
                BaseBitLayoutState::Variable,
                BaseBitLayoutState::Variable,
                BaseBitLayoutState::Variable,
            ]),
            variable_base_table: vec![
                (make_bits(&[true, false, true, false]), 1),  // 1010
                (make_bits(&[false, false, false, true]), 1), // 0001
                (make_bits(&[true, false, true, true]), 1),   // 1011
                (make_bits(&[false, true, true, true]), 1),   // 0111
            ],
            entropy_sorted_column_order: None,
        };

        sort_context_base_tables(&mut context);

        let as_vec = |bits: &BitVec<usize, Lsb0>| -> Vec<bool> { bits.iter().by_vals().collect() };
        let sorted_rows: Vec<Vec<bool>> = context
            .variable_base_table
            .iter()
            .map(|(bits, _)| as_vec(bits))
            .collect();

        assert_eq!(
            sorted_rows,
            vec![
                vec![false, true, true, true],
                vec![true, false, true, true],
                vec![true, false, true, false],
                vec![false, false, false, true],
            ]
        );
        assert_eq!(context.row_to_base_id, vec![2, 3, 1, 0]);
        assert_eq!(context.entropy_sorted_column_order, Some(vec![1, 2, 3, 0]));
    }
}
