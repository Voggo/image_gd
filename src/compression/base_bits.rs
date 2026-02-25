use crate::compression::preprocessor::BitDataSet;
use crate::timing::ScopedTimer;
use bitvec::prelude::*;

pub trait BaseBit {
    fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize;
    fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        let mut num_bases = self.get_num_bases();
        for &bit_position in bit_positions {
            num_bases = self.add_bit_position(bit_data, bit_position);
        }
        num_bases
    }
    fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize;
    fn get_bases(&self, bit_data: &BitDataSet) -> Vec<(BitVec<usize, Msb0>, usize)>;
    fn get_groups(&self) -> &[Vec<usize>];
    fn get_num_bases(&self) -> usize;
    fn get_num_bits_per_base(&self) -> usize;
    fn get_base_bit_mask(&self) -> &BitSlice<usize, Msb0>;
    fn get_base_bit_positions(&self) -> &[usize];
}

#[derive(Clone)]
pub struct BaseBitGroups {
    groups: Vec<Vec<usize>>,
    base_bit_mask: BitVec<usize, Msb0>,
    base_bit_positions: Vec<usize>,
    num_bases: usize,
    num_bits_per_base: usize,
}

impl std::fmt::Debug for BaseBitGroups {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("BaseBitGroups")
            .field("groups", &self.groups)
            .field("base_bit_mask", &self.base_bit_mask)
            .field("base_bit_positions", &self.base_bit_positions)
            .field("num_bases", &self.num_bases)
            .field("num_bits_per_base", &self.num_bits_per_base)
            .finish()
    }
}

impl BaseBitGroups {
    pub fn new(num_rows: usize, chunk_size: usize) -> Self {
        let groups: Vec<Vec<usize>> = vec![(0..num_rows).collect()];
        let base_bit_mask = bitvec![usize, Msb0; 0; chunk_size];
        let base_bit_positions = Vec::new();
        let num_bits_per_base = 0;
        BaseBitGroups {
            groups,
            base_bit_mask,
            base_bit_positions,
            num_bases: num_bits_per_base, // Initially, because they are constant bit positions
            num_bits_per_base,
        }
    }

    pub fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        let _timer = ScopedTimer::trace(format!("Adding bit position {}", bit_position));

        // Skip duplicate work if this bit has already been selected.
        if self.base_bit_mask[bit_position] {
            return self.num_bases;
        }

        self.base_bit_mask.set(bit_position, true);
        self.num_bits_per_base += 1;
        self.base_bit_positions.push(bit_position);

        // Collect split-off groups and append after iteration.
        // This avoids repeated growth/reallocation of `self.groups` while iterating.
        let mut new_groups = Vec::new();

        for group in &mut self.groups {
            if group.len() <= 1 {
                continue;
            }

            // Keep zero-bit rows in place; move one-bit rows into a side vector.
            // This avoids allocating a second vector for zeros.
            let mut group_ones = Vec::with_capacity(group.len() / 2 + 1);
            group.retain(|&row| {
                if bit_data.get_bit(row, bit_position) {
                    group_ones.push(row);
                    false
                } else {
                    true
                }
            });

            if group.is_empty() {
                // All rows had bit=1.
                *group = group_ones;
            } else if !group_ones.is_empty() {
                // Mixed group: keep zeros in place and append ones as a new group.
                new_groups.push(group_ones);
            }
        }

        self.groups.extend(new_groups);
        self.num_bases = self.groups.len();
        self.num_bases
    }

    pub fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        let _timer =
            ScopedTimer::debug(format!("Adding constant bit positions {:?}", bit_positions));
        for &bit_position in bit_positions {
            self.base_bit_mask.set(bit_position, true);
        }
        self.num_bits_per_base += bit_positions.len();
        self.base_bit_positions.extend_from_slice(bit_positions);
        self.num_bases = self.groups.len(); // Number of bases doesn't change for constant bits
        self.num_bases
    }
    /// Get the bases as BitVecs along with their counts
    pub fn get_bases(&self, bit_data: &BitDataSet) -> Vec<(BitVec<usize, Msb0>, usize)> {
        let _timer = ScopedTimer::debug("Getting bases");
        let mut bases = Vec::with_capacity(self.num_bases);
        for group in &self.groups {
            if group.is_empty() {
                continue;
            }
            let base: BitVec<usize, Msb0> = bit_data.get_chunk(group[0]).to_bitvec();
            bases.push((base & &self.base_bit_mask, group.len()));
        }
        bases
    }

    pub fn get_groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    pub fn get_num_bases(&self) -> usize {
        self.num_bases
    }

    pub fn get_num_bits_per_base(&self) -> usize {
        self.num_bits_per_base
    }

    pub fn get_base_bit_mask(&self) -> &BitSlice<usize, Msb0> {
        &self.base_bit_mask
    }

    pub fn get_base_bit_positions(&self) -> &[usize] {
        &self.base_bit_positions
    }
}

impl BaseBit for BaseBitGroups {
    fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        BaseBitGroups::add_bit_position(self, bit_data, bit_position)
    }

    fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        BaseBitGroups::add_constant_bit_positions(self, bit_positions)
    }

    fn get_bases(&self, bit_data: &BitDataSet) -> Vec<(BitVec<usize, Msb0>, usize)> {
        BaseBitGroups::get_bases(self, bit_data)
    }

    fn get_groups(&self) -> &[Vec<usize>] {
        BaseBitGroups::get_groups(self)
    }

    fn get_num_bases(&self) -> usize {
        BaseBitGroups::get_num_bases(self)
    }

    fn get_num_bits_per_base(&self) -> usize {
        BaseBitGroups::get_num_bits_per_base(self)
    }

    fn get_base_bit_mask(&self) -> &BitSlice<usize, Msb0> {
        BaseBitGroups::get_base_bit_mask(self)
    }

    fn get_base_bit_positions(&self) -> &[usize] {
        BaseBitGroups::get_base_bit_positions(self)
    }
}

#[derive(Clone)]
pub struct BaseBitBatchGroups {
    groups: Vec<Vec<usize>>,
    base_bit_mask: BitVec<usize, Msb0>,
    base_bit_positions: Vec<usize>,
    num_bases: usize,
    num_bits_per_base: usize,
}

impl std::fmt::Debug for BaseBitBatchGroups {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("BaseBitBatchGroups")
            .field("groups", &self.groups)
            .field("base_bit_mask", &self.base_bit_mask)
            .field("base_bit_positions", &self.base_bit_positions)
            .field("num_bases", &self.num_bases)
            .field("num_bits_per_base", &self.num_bits_per_base)
            .finish()
    }
}

impl BaseBitBatchGroups {
    pub fn new(num_rows: usize, chunk_size: usize) -> Self {
        let groups: Vec<Vec<usize>> = vec![(0..num_rows).collect()];
        let base_bit_mask = bitvec![usize, Msb0; 0; chunk_size];
        let base_bit_positions = Vec::new();
        let num_bits_per_base = 0;
        BaseBitBatchGroups {
            groups,
            base_bit_mask,
            base_bit_positions,
            num_bases: num_bits_per_base,
            num_bits_per_base,
        }
    }

    pub fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        let _timer = ScopedTimer::trace(format!("Adding bit positions {:?}", bit_positions));

        if bit_positions.is_empty() {
            return self.num_bases;
        }

        // Avoid pathological allocations when too many bits are requested at once.
        if bit_positions.len() >= usize::BITS as usize || bit_positions.len() > 16 {
            for &bit_position in bit_positions {
                self.add_bit_position(bit_data, bit_position);
            }
            return self.num_bases;
        }

        let bucket_count = 1usize << bit_positions.len();
        let mut counts = vec![0usize; bucket_count];
        let mut offsets = vec![0usize; bucket_count];
        let mut write_positions = vec![0usize; bucket_count];
        let mut scratch_rows: Vec<usize> = Vec::new();
        let mut scratch_bucket_ids: Vec<usize> = Vec::new();

        let mut next_groups: Vec<Vec<usize>> = Vec::with_capacity(self.groups.len());

        for group in &self.groups {
            if group.is_empty() {
                continue;
            }

            if group.len() <= 1 {
                next_groups.push(group.clone());
                continue;
            }

            counts.fill(0);

            if scratch_bucket_ids.len() < group.len() {
                scratch_bucket_ids.resize(group.len(), 0);
            }

            for (idx, &row) in group.iter().enumerate() {
                let mut bucket_id = 0usize;
                for (bit_idx, &bit_position) in bit_positions.iter().enumerate() {
                    bucket_id |= (bit_data.get_bit(row, bit_position) as usize) << bit_idx;
                }
                scratch_bucket_ids[idx] = bucket_id;
                counts[bucket_id] += 1;
            }

            let mut running = 0usize;
            for bucket_id in 0..bucket_count {
                offsets[bucket_id] = running;
                running += counts[bucket_id];
            }

            if scratch_rows.len() < group.len() {
                scratch_rows.resize(group.len(), 0);
            }

            write_positions.copy_from_slice(&offsets);

            for (idx, &row) in group.iter().enumerate() {
                let bucket_id = scratch_bucket_ids[idx];
                let write_idx = write_positions[bucket_id];
                scratch_rows[write_idx] = row;
                write_positions[bucket_id] += 1;
            }

            for bucket_id in 0..bucket_count {
                let count = counts[bucket_id];
                if count == 0 {
                    continue;
                }
                let start = offsets[bucket_id];
                let end = start + count;
                next_groups.push(scratch_rows[start..end].to_vec());
            }
        }

        for &bit_position in bit_positions {
            self.base_bit_mask.set(bit_position, true);
        }
        self.base_bit_positions.extend_from_slice(bit_positions);
        self.num_bits_per_base += bit_positions.len();
        self.groups = next_groups;
        self.num_bases = self.groups.len();
        self.num_bases
    }

    pub fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        self.add_bit_positions(bit_data, &[bit_position])
    }

    pub fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        let _timer =
            ScopedTimer::debug(format!("Adding constant bit positions {:?}", bit_positions));
        for &bit_position in bit_positions {
            self.base_bit_mask.set(bit_position, true);
        }
        self.num_bits_per_base += bit_positions.len();
        self.base_bit_positions.extend_from_slice(bit_positions);
        self.num_bases = self.groups.len();
        self.num_bases
    }

    pub fn get_bases(&self, bit_data: &BitDataSet) -> Vec<(BitVec<usize, Msb0>, usize)> {
        let _timer = ScopedTimer::debug("Getting bases");
        let mut bases = Vec::with_capacity(self.num_bases);
        for group in &self.groups {
            if group.is_empty() {
                continue;
            }
            let base: BitVec<usize, Msb0> = bit_data.get_chunk(group[0]).to_bitvec();
            bases.push((base & &self.base_bit_mask, group.len()));
        }
        bases
    }

    pub fn get_groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    pub fn get_num_bases(&self) -> usize {
        self.num_bases
    }

    pub fn get_num_bits_per_base(&self) -> usize {
        self.num_bits_per_base
    }

    pub fn get_base_bit_mask(&self) -> &BitSlice<usize, Msb0> {
        &self.base_bit_mask
    }

    pub fn get_base_bit_positions(&self) -> &[usize] {
        &self.base_bit_positions
    }
}

impl BaseBit for BaseBitBatchGroups {
    fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        BaseBitBatchGroups::add_bit_position(self, bit_data, bit_position)
    }

    fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        BaseBitBatchGroups::add_bit_positions(self, bit_data, bit_positions)
    }

    fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        BaseBitBatchGroups::add_constant_bit_positions(self, bit_positions)
    }

    fn get_bases(&self, bit_data: &BitDataSet) -> Vec<(BitVec<usize, Msb0>, usize)> {
        BaseBitBatchGroups::get_bases(self, bit_data)
    }

    fn get_groups(&self) -> &[Vec<usize>] {
        BaseBitBatchGroups::get_groups(self)
    }

    fn get_num_bases(&self) -> usize {
        BaseBitBatchGroups::get_num_bases(self)
    }

    fn get_num_bits_per_base(&self) -> usize {
        BaseBitBatchGroups::get_num_bits_per_base(self)
    }

    fn get_base_bit_mask(&self) -> &BitSlice<usize, Msb0> {
        BaseBitBatchGroups::get_base_bit_mask(self)
    }

    fn get_base_bit_positions(&self) -> &[usize] {
        BaseBitBatchGroups::get_base_bit_positions(self)
    }
}

#[derive(Clone)]
pub struct BaseBitSignatureGroups {
    groups: Vec<Vec<usize>>,
    base_bit_mask: BitVec<usize, Msb0>,
    base_bit_positions: Vec<usize>,
    num_bases: usize,
    num_bits_per_base: usize,
}

impl std::fmt::Debug for BaseBitSignatureGroups {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("BaseBitSignatureGroups")
            .field("groups", &self.groups)
            .field("base_bit_mask", &self.base_bit_mask)
            .field("base_bit_positions", &self.base_bit_positions)
            .field("num_bases", &self.num_bases)
            .field("num_bits_per_base", &self.num_bits_per_base)
            .finish()
    }
}

impl BaseBitSignatureGroups {
    pub fn new(num_rows: usize, chunk_size: usize) -> Self {
        let groups: Vec<Vec<usize>> = vec![(0..num_rows).collect()];
        let base_bit_mask = bitvec![usize, Msb0; 0; chunk_size];
        let base_bit_positions = Vec::new();
        let num_bits_per_base = 0;
        BaseBitSignatureGroups {
            groups,
            base_bit_mask,
            base_bit_positions,
            num_bases: num_bits_per_base,
            num_bits_per_base,
        }
    }

    pub fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        let _timer = ScopedTimer::trace(format!(
            "Adding bit positions (signature) {:?}",
            bit_positions
        ));

        if bit_positions.is_empty() {
            return self.num_bases;
        }

        let mut added_count = 0usize;
        for &bit_position in bit_positions {
            if !self.base_bit_mask[bit_position] {
                self.base_bit_mask.set(bit_position, true);
                self.base_bit_positions.push(bit_position);
                added_count += 1;
            }
        }

        if added_count == 0 {
            return self.num_bases;
        }

        self.num_bits_per_base += added_count;

        let groups = if self.base_bit_positions.len() <= 128 {
            let mut signature_rows: Vec<(u128, usize)> = Vec::with_capacity(bit_data.num_rows());
            for row in 0..bit_data.num_rows() {
                let mut signature = 0u128;
                for &bit_position in &self.base_bit_positions {
                    signature <<= 1;
                    signature |= bit_data.get_bit(row, bit_position) as u128;
                }
                signature_rows.push((signature, row));
            }

            signature_rows.sort_unstable_by_key(|(signature, _row)| *signature);

            let mut groups: Vec<Vec<usize>> = Vec::new();
            let mut current_signature: Option<u128> = None;
            for (signature, row) in signature_rows {
                if current_signature == Some(signature) {
                    groups.last_mut().expect("group exists").push(row);
                } else {
                    current_signature = Some(signature);
                    groups.push(vec![row]);
                }
            }
            groups
        } else {
            log::warn!(
                "More than 128 selected bits; using lexicographic fallback for grouping"
            );
            let mut rows: Vec<usize> = (0..bit_data.num_rows()).collect();
            rows.sort_unstable_by(|&lhs, &rhs| {
                for &bit_position in &self.base_bit_positions {
                    let lhs_bit = bit_data.get_bit(lhs, bit_position);
                    let rhs_bit = bit_data.get_bit(rhs, bit_position);
                    match lhs_bit.cmp(&rhs_bit) {
                        std::cmp::Ordering::Equal => {}
                        ordering => return ordering,
                    }
                }
                std::cmp::Ordering::Equal
            });

            let mut groups: Vec<Vec<usize>> = Vec::new();
            'rows_loop: for row in rows {
                if let Some(last_group) = groups.last_mut() {
                    let last_row = last_group[0];
                    for &bit_position in &self.base_bit_positions {
                        if bit_data.get_bit(last_row, bit_position)
                            != bit_data.get_bit(row, bit_position)
                        {
                            groups.push(vec![row]);
                            continue 'rows_loop;
                        }
                    }
                    last_group.push(row);
                } else {
                    groups.push(vec![row]);
                }
            }
            groups
        };

        self.groups = groups;
        self.num_bases = self.groups.len();
        self.num_bases
    }

    pub fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        self.add_bit_positions(bit_data, &[bit_position])
    }

    pub fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        let _timer =
            ScopedTimer::debug(format!("Adding constant bit positions {:?}", bit_positions));
        for &bit_position in bit_positions {
            if !self.base_bit_mask[bit_position] {
                self.base_bit_mask.set(bit_position, true);
                self.base_bit_positions.push(bit_position);
                self.num_bits_per_base += 1;
            }
        }
        self.num_bases = self.groups.len();
        self.num_bases
    }

    pub fn get_bases(&self, bit_data: &BitDataSet) -> Vec<(BitVec<usize, Msb0>, usize)> {
        let _timer = ScopedTimer::debug("Getting bases");
        let mut bases = Vec::with_capacity(self.num_bases);
        for group in &self.groups {
            if group.is_empty() {
                continue;
            }
            let base: BitVec<usize, Msb0> = bit_data.get_chunk(group[0]).to_bitvec();
            bases.push((base & &self.base_bit_mask, group.len()));
        }
        bases
    }

    pub fn get_groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    pub fn get_num_bases(&self) -> usize {
        self.num_bases
    }

    pub fn get_num_bits_per_base(&self) -> usize {
        self.num_bits_per_base
    }

    pub fn get_base_bit_mask(&self) -> &BitSlice<usize, Msb0> {
        &self.base_bit_mask
    }

    pub fn get_base_bit_positions(&self) -> &[usize] {
        &self.base_bit_positions
    }
}

impl BaseBit for BaseBitSignatureGroups {
    fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        BaseBitSignatureGroups::add_bit_position(self, bit_data, bit_position)
    }

    fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        BaseBitSignatureGroups::add_bit_positions(self, bit_data, bit_positions)
    }

    fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        BaseBitSignatureGroups::add_constant_bit_positions(self, bit_positions)
    }

    fn get_bases(&self, bit_data: &BitDataSet) -> Vec<(BitVec<usize, Msb0>, usize)> {
        BaseBitSignatureGroups::get_bases(self, bit_data)
    }

    fn get_groups(&self) -> &[Vec<usize>] {
        BaseBitSignatureGroups::get_groups(self)
    }

    fn get_num_bases(&self) -> usize {
        BaseBitSignatureGroups::get_num_bases(self)
    }

    fn get_num_bits_per_base(&self) -> usize {
        BaseBitSignatureGroups::get_num_bits_per_base(self)
    }

    fn get_base_bit_mask(&self) -> &BitSlice<usize, Msb0> {
        BaseBitSignatureGroups::get_base_bit_mask(self)
    }

    fn get_base_bit_positions(&self) -> &[usize] {
        BaseBitSignatureGroups::get_base_bit_positions(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::preprocessor::{BitData, BitDataInfo, BitDataSet, FeatureSpec};
    use crate::data_loader::FeatureDataType;

    fn print_bases(base_bit_groups: &BaseBitGroups, bit_data: &BitDataSet) {
        log::info!("Bases after adding bit:");
        for (i, (base, count)) in base_bit_groups.get_bases(bit_data).iter().enumerate() {
            log::info!("Base {}: {:?}, Count: {}", i, base, count);
        }
    }

    #[test]
    fn test_base_bit_groups() {
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
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };
        log::info!("BitData: {}", bit_data);
        let mut base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let num_bases = base_bit_groups.add_bit_position(&bit_data, 4);
        assert_eq!(num_bases, 2);
        print_bases(&base_bit_groups, &bit_data);
        let num_bases = base_bit_groups.add_bit_position(&bit_data, 5);
        assert_eq!(num_bases, 3);
        print_bases(&base_bit_groups, &bit_data);
    }

    #[test]
    fn test_base_bit_groups_with_initial_bits() {
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
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };
        log::info!("BitData: {}", bit_data);
        let mut base_bit_groups = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let num_bases = base_bit_groups.add_bit_position(&bit_data, 4);
        assert_eq!(num_bases, 2);
        print_bases(&base_bit_groups, &bit_data);
        let num_bases = base_bit_groups.add_bit_position(&bit_data, 5);
        assert_eq!(num_bases, 3);
        print_bases(&base_bit_groups, &bit_data);
    }

    #[test]
    fn test_base_bit_batch_groups_add_multiple_bits() {
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
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let mut batch_groups = BaseBitBatchGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let num_bases = batch_groups.add_bit_positions(&bit_data, &[4, 5]);
        assert_eq!(num_bases, 3);
        assert_eq!(batch_groups.get_num_bits_per_base(), 2);
        assert_eq!(batch_groups.get_groups().iter().map(|g| g.len()).sum::<usize>(), num_rows);
    }

    #[test]
    fn test_base_bit_signature_groups_add_multiple_bits() {
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
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let mut signature_groups =
            BaseBitSignatureGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let num_bases = signature_groups.add_bit_positions(&bit_data, &[4, 5]);
        assert_eq!(num_bases, 3);
        assert_eq!(signature_groups.get_num_bits_per_base(), 2);
        assert_eq!(
            signature_groups
                .get_groups()
                .iter()
                .map(|g| g.len())
                .sum::<usize>(),
            num_rows
        );
    }
}
