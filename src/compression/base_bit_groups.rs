use crate::preprocessor::BitDataView;
use bitvec::prelude::*;

#[derive(Clone)]
pub struct BaseBitGroups<'a> {
    bit_data: &'a dyn BitDataView,
    groups: Vec<Vec<usize>>,
    base_bit_mask: BitVec<usize, Msb0>,
    base_bit_positions: Vec<usize>,
    num_bases: usize,
    num_bits_per_base: usize,
}

impl std::fmt::Debug for BaseBitGroups<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseBitGroups")
            .field("groups", &self.groups)
            .field("base_bit_mask", &self.base_bit_mask)
            .field("base_bit_positions", &self.base_bit_positions)
            .field("num_bases", &self.num_bases)
            .field("num_bits_per_base", &self.num_bits_per_base)
            .finish()
    }
}

impl<'a> BaseBitGroups<'a> {
    pub fn new(bit_data: &'a impl BitDataView) -> Self {
        let groups: Vec<Vec<usize>> = vec![(0..bit_data.num_rows()).collect()];
        let base_bit_mask = bitvec![usize, Msb0; 0; bit_data.chunk_size()];
        let base_bit_positions = Vec::new();
        let num_bits_per_base = 0;
        BaseBitGroups {
            bit_data,
            groups,
            base_bit_mask,
            base_bit_positions,
            num_bases: num_bits_per_base, // Initially, because they are constant bit positions
            num_bits_per_base,
        }
    }

    pub fn add_bit_position(&mut self, bit_position: usize) -> usize {
        self.base_bit_mask.set(bit_position, true);
        self.num_bits_per_base += 1;
        self.base_bit_positions.push(bit_position);

        for group_idx in 0..self.groups.len() {
            let mut group_zeros = Vec::new();
            let mut group_ones = Vec::new();
            for &row in self.groups[group_idx].iter() {
                if self.bit_data.get_bit(row, bit_position) {
                    group_ones.push(row);
                } else {
                    group_zeros.push(row);
                }
            }
            if group_zeros.is_empty() && !group_ones.is_empty() {
                self.groups[group_idx] = group_ones;
            } else if group_ones.is_empty() && !group_zeros.is_empty() {
                self.groups[group_idx] = group_zeros;
            } else if !group_zeros.is_empty() && !group_ones.is_empty() {
                self.groups[group_idx] = group_zeros;
                self.groups.push(group_ones);
            }
        }
        self.num_bases = self.groups.len();
        self.num_bases
    }

    /// Get the bases as BitVecs along with their counts
    pub fn get_bases(&self) -> Vec<(BitVec<usize, Msb0>, usize)> {
        let mut bases = Vec::with_capacity(self.num_bases);
        for group in &self.groups {
            if group.is_empty() {
                continue;
            }
            let base: BitVec<usize, Msb0> = self.bit_data.get_chunk(group[0]).to_bitvec();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::BitData;

    fn print_bases(base_bit_groups: &BaseBitGroups) {
        println!("Bases after adding bit:");
        for (i, (base, count)) in base_bit_groups.get_bases().iter().enumerate() {
            println!("Base {}: {:?}, Count: {}", i, base, count);
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
        let bit_data = BitData {
            data: data.into_iter().collect(),
            bits_per_feature,
            num_features,
            num_rows,
            chunk_size,
        };
        println!("BitData: {}", bit_data);
        let mut base_bit_groups = BaseBitGroups::new(&bit_data);
        let num_bases = base_bit_groups.add_bit_position(4);
        assert_eq!(num_bases, 2);
        print_bases(&base_bit_groups);
        let num_bases = base_bit_groups.add_bit_position(5);
        assert_eq!(num_bases, 3);
        print_bases(&base_bit_groups);
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
        let bit_data = BitData {
            data: data.into_iter().collect(),
            bits_per_feature,
            num_features,
            num_rows,
            chunk_size,
        };
        println!("BitData: {}", bit_data);
        let mut base_bit_groups = BaseBitGroups::new(&bit_data);
        let num_bases = base_bit_groups.add_bit_position(4);
        assert_eq!(num_bases, 2);
        print_bases(&base_bit_groups);
        let num_bases = base_bit_groups.add_bit_position(5);
        assert_eq!(num_bases, 3);
        print_bases(&base_bit_groups);   
    }


}
