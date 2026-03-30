use crate::compression::preprocessor::BitDataSet;
use crate::timing::ScopedTimer;
use bitvec::prelude::*;

use fxhash::FxHashMap;
use once_cell::unsync::OnceCell;

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

        const BIT_LIMIT: usize = 8;

        // Process in recursive batches, but ensure this frame only handles <= BIT_LIMIT bits.
        if bit_positions.len() > BIT_LIMIT {
            let (head, tail) = bit_positions.split_at(BIT_LIMIT);
            self.add_bit_positions(bit_data, head);
            return self.add_bit_positions(bit_data, tail);
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
pub struct BaseBitIncSignatureGroups {
    /// Per-row signature class id over selected (non-constant) base bits.
    ///
    /// `row_signatures[row]` is not a packed bit pattern; it is a stable class id.
    /// When new positions are added, classes are refined incrementally from
    /// `(old_class_id, new_bits_batch)` to `new_class_id` without rebuilding groups.
    row_signatures: Vec<u64>,
    /// Frequency table for signatures. Maintained incrementally during updates.
    signature_counts: FxHashMap<u64, usize>,
    /// Monotonic id generator for new signature classes.
    next_signature_id: u64,
    /// Lazily materialized groups cache, invalidated when signatures change.
    ///
    /// This keeps `add_bit_positions()` allocation-free per row and defers
    /// `Vec<Vec<usize>>` construction to the rare `get_groups()` call.
    groups_cache: OnceCell<Vec<Vec<usize>>>,
    base_bit_mask: BitVec<usize, Msb0>,
    base_bit_positions: Vec<usize>,
    num_bits_per_base: usize,
}

impl std::fmt::Debug for BaseBitIncSignatureGroups {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("BaseBitBatchGroups")
            .field("row_signatures", &self.row_signatures)
            .field("signature_counts", &self.signature_counts)
            .field("next_signature_id", &self.next_signature_id)
            .field("base_bit_mask", &self.base_bit_mask)
            .field("base_bit_positions", &self.base_bit_positions)
            .field("num_bases", &self.signature_counts.len())
            .field("num_bits_per_base", &self.num_bits_per_base)
            .finish()
    }
}

impl BaseBitIncSignatureGroups {
    pub fn new(num_rows: usize, chunk_size: usize) -> Self {
        let base_bit_mask = bitvec![usize, Msb0; 0; chunk_size];
        let base_bit_positions = Vec::new();
        let row_signatures = vec![0u64; num_rows];
        let mut signature_counts = FxHashMap::with_capacity_and_hasher(1, Default::default());
        signature_counts.insert(0u64, num_rows);
        let num_bits_per_base = 0;
        BaseBitIncSignatureGroups {
            row_signatures,
            signature_counts,
            next_signature_id: 1,
            groups_cache: OnceCell::new(),
            base_bit_mask,
            base_bit_positions,
            num_bits_per_base,
        }
    }

    /// Refine current signature classes using one batch of at most 64 bit positions.
    fn refine_signatures_batch(&mut self, bit_data: &BitDataSet, bit_positions_batch: &[usize]) {
        debug_assert!(bit_positions_batch.len() <= 64);

        // Key packs `(old_signature, mini_signature)` to reduce tuple-hash overhead.
        let mut transitions = FxHashMap::<u128, (u64, usize)>::with_capacity_and_hasher(
            self.signature_counts.len(),
            Default::default(),
        );

        for row in 0..self.row_signatures.len() {
            let old_signature = self.row_signatures[row];
            let row_chunk = bit_data.get_chunk(row);

            let mut mini_signature = 0u64;
            for (i, &bit_position) in bit_positions_batch.iter().enumerate() {
                let bit = if row_chunk[bit_position] { 1 } else { 0 };
                mini_signature |= bit << i;
            }

            let transition_key = ((old_signature as u128) << 64) | (mini_signature as u128);
            let transition_entry = transitions.entry(transition_key).or_insert_with(|| {
                let new_id = self.next_signature_id;
                self.next_signature_id = self
                    .next_signature_id
                    .checked_add(1)
                    .expect("BaseBitBatchGroups signature id overflow");
                (new_id, 0)
            });
            transition_entry.1 += 1;

            self.row_signatures[row] = transition_entry.0;
        }

        // Apply count deltas in aggregate to avoid per-row hash-map churn.
        for (transition_key, (new_signature, transitioned_count)) in transitions {
            let old_signature = (transition_key >> 64) as u64;

            let remove_old = {
                let count = self
                    .signature_counts
                    .get_mut(&old_signature)
                    .expect("old signature must exist in signature_counts");
                *count -= transitioned_count;
                *count == 0
            };
            if remove_old {
                self.signature_counts.remove(&old_signature);
            }

            *self.signature_counts.entry(new_signature).or_insert(0) += transitioned_count;
        }
    }

    pub fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        let _timer = ScopedTimer::trace(format!("Adding bit positions {:?}", bit_positions));

        if bit_positions.is_empty() {
            return self.get_num_bases();
        }

        for batch in bit_positions.chunks(64) {
            self.refine_signatures_batch(bit_data, batch);
        }

        for &bit_position in bit_positions {
            self.base_bit_mask.set(bit_position, true);
        }
        self.base_bit_positions.extend_from_slice(bit_positions);
        self.num_bits_per_base += bit_positions.len();

        // Signatures changed; lazy groups must be rebuilt on demand.
        self.groups_cache.take();

        self.get_num_bases()
    }

    pub fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        self.add_bit_positions(bit_data, &[bit_position])
    }

    pub fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        let _timer =
            ScopedTimer::debug(format!("Adding constant bit positions {:?}", bit_positions));

        let mut added_count = 0usize;
        for &bit_position in bit_positions {
            assert!(
                bit_position < self.base_bit_mask.len(),
                "bit position {} out of bounds for chunk size {}",
                bit_position,
                self.base_bit_mask.len()
            );
            if self.base_bit_mask[bit_position] {
                continue;
            }
            self.base_bit_mask.set(bit_position, true);
            self.base_bit_positions.push(bit_position);
            added_count += 1;
        }
        self.num_bits_per_base += added_count;

        // Constants do not change row signatures; base count is unchanged.
        self.get_num_bases()
    }

    pub fn get_bases(&self, bit_data: &BitDataSet) -> Vec<(BitVec<usize, Msb0>, usize)> {
        let _timer = ScopedTimer::debug("Getting bases");

        // Preserve stable ordering by first row occurrence for consistency with
        // `get_groups()` and downstream row->base-id mappings.
        let mut signature_order = Vec::with_capacity(self.signature_counts.len());
        let mut seen = FxHashMap::<u64, usize>::with_capacity_and_hasher(
            self.signature_counts.len(),
            Default::default(),
        );
        for (row, &signature) in self.row_signatures.iter().enumerate() {
            seen.entry(signature).or_insert_with(|| {
                signature_order.push((signature, row));
                row
            });
        }

        let mut bases = Vec::with_capacity(signature_order.len());
        for (signature, representative_row) in signature_order {
            let count = *self
                .signature_counts
                .get(&signature)
                .expect("signature must exist in signature_counts");
            let base: BitVec<usize, Msb0> = bit_data.get_chunk(representative_row).to_bitvec();
            bases.push((base & &self.base_bit_mask, count));
        }
        bases
    }

    pub fn get_groups(&self) -> &[Vec<usize>] {
        self.groups_cache
            .get_or_init(|| {
                let mut group_index_by_signature =
                    FxHashMap::<u64, usize>::with_capacity_and_hasher(
                        self.signature_counts.len(),
                        Default::default(),
                    );
                let mut groups: Vec<Vec<usize>> = Vec::with_capacity(self.signature_counts.len());

                for (row, &signature) in self.row_signatures.iter().enumerate() {
                    let group_idx = if let Some(&idx) = group_index_by_signature.get(&signature) {
                        idx
                    } else {
                        let idx = groups.len();
                        groups.push(Vec::new());
                        group_index_by_signature.insert(signature, idx);
                        idx
                    };
                    groups[group_idx].push(row);
                }
                groups
            })
            .as_slice()
    }

    pub fn get_num_bases(&self) -> usize {
        self.signature_counts.len()
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

impl BaseBit for BaseBitIncSignatureGroups {
    fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        BaseBitIncSignatureGroups::add_bit_position(self, bit_data, bit_position)
    }

    fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        BaseBitIncSignatureGroups::add_bit_positions(self, bit_data, bit_positions)
    }

    fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        BaseBitIncSignatureGroups::add_constant_bit_positions(self, bit_positions)
    }

    fn get_bases(&self, bit_data: &BitDataSet) -> Vec<(BitVec<usize, Msb0>, usize)> {
        BaseBitIncSignatureGroups::get_bases(self, bit_data)
    }

    fn get_groups(&self) -> &[Vec<usize>] {
        BaseBitIncSignatureGroups::get_groups(self)
    }

    fn get_num_bases(&self) -> usize {
        BaseBitIncSignatureGroups::get_num_bases(self)
    }

    fn get_num_bits_per_base(&self) -> usize {
        BaseBitIncSignatureGroups::get_num_bits_per_base(self)
    }

    fn get_base_bit_mask(&self) -> &BitSlice<usize, Msb0> {
        BaseBitIncSignatureGroups::get_base_bit_mask(self)
    }

    fn get_base_bit_positions(&self) -> &[usize] {
        BaseBitIncSignatureGroups::get_base_bit_positions(self)
    }
}

#[derive(Clone)]
pub struct BaseBitHyperLogLogCount {
    base_bit_mask: BitVec<usize, Msb0>,
    base_bit_positions: Vec<usize>,
    num_bits_per_base: usize,
    num_bases_estimate: usize,
    row_hashes: Vec<u64>,
    bit_hash_words: Vec<u64>,
    registers: Vec<u8>,
}

impl std::fmt::Debug for BaseBitHyperLogLogCount {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("BaseBitHyperLogLogCount")
            .field("base_bit_mask", &self.base_bit_mask)
            .field("base_bit_positions", &self.base_bit_positions)
            .field("num_bits_per_base", &self.num_bits_per_base)
            .field("num_bases_estimate", &self.num_bases_estimate)
            .finish()
    }
}

impl BaseBitHyperLogLogCount {
    const HLL_PRECISION: u8 = 8;
    const SPLITMIX64_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

    pub fn new(num_rows: usize, chunk_size: usize) -> Self {
        let base_bit_mask = bitvec![usize, Msb0; 0; chunk_size];
        let base_bit_positions = Vec::new();

        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut bit_hash_words = Vec::with_capacity(chunk_size);
        for _ in 0..chunk_size {
            bit_hash_words.push(Self::splitmix64(&mut state));
        }

        BaseBitHyperLogLogCount {
            base_bit_mask,
            base_bit_positions,
            num_bits_per_base: 0,
            num_bases_estimate: 0,
            row_hashes: vec![0u64; num_rows],
            bit_hash_words,
            registers: vec![0u8; 1usize << Self::HLL_PRECISION],
        }
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(Self::SPLITMIX64_INCREMENT);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn avalanche_hash(mut value: u64) -> u64 {
        value ^= value >> 33;
        value = value.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        value ^= value >> 33;
        value = value.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        value ^ (value >> 33)
    }

    fn hll_alpha(num_registers: usize) -> f64 {
        match num_registers {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => {
                let m = num_registers as f64;
                0.7213 / (1.0 + 1.079 / m)
            }
        }
    }

    fn estimate_from_registers(registers: &[u8], num_rows: usize) -> usize {
        if num_rows == 0 {
            return 0;
        }

        let m = registers.len() as f64;
        let alpha = Self::hll_alpha(registers.len());

        let mut harmonic_sum = 0.0f64;
        let mut zero_count = 0usize;
        for &register in registers {
            if register == 0 {
                zero_count += 1;
            }
            harmonic_sum += 2f64.powi(-(register as i32));
        }

        let mut estimate = alpha * m * m / harmonic_sum;
        if estimate <= 2.5 * m && zero_count > 0 {
            estimate = m * (m / zero_count as f64).ln();
        }

        estimate.round().clamp(1.0, num_rows as f64).max(1.0) as usize
    }

    pub fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        let _timer = ScopedTimer::trace(format!("Adding bit positions {:?}", bit_positions));

        if bit_positions.is_empty() {
            return self.num_bases_estimate;
        }

        let mut new_bit_positions = Vec::with_capacity(bit_positions.len());
        for &bit_position in bit_positions {
            assert!(
                bit_position < self.base_bit_mask.len(),
                "bit position {} out of bounds for chunk size {}",
                bit_position,
                self.base_bit_mask.len()
            );
            if self.base_bit_mask[bit_position] {
                continue;
            }

            new_bit_positions.push(bit_position);
            self.base_bit_mask.set(bit_position, true);
            self.base_bit_positions.push(bit_position);
            self.num_bits_per_base += 1;
        }

        if new_bit_positions.is_empty() {
            return self.num_bases_estimate;
        }

        let raw_bits = bit_data.data.raw();
        let chunk_size = bit_data.chunk_size();
        self.registers.fill(0);
        let bucket_shift = 64 - Self::HLL_PRECISION as usize;

        for (row, row_hash) in self.row_hashes.iter_mut().enumerate() {
            let row_start = row * chunk_size;
            for &bit_position in &new_bit_positions {
                if raw_bits[row_start + bit_position] {
                    *row_hash ^= self.bit_hash_words[bit_position];
                }
            }

            let mixed = Self::avalanche_hash(*row_hash ^ 0xA24B_AED4_963E_E407);
            let bucket = (mixed >> bucket_shift) as usize;

            let suffix = mixed << Self::HLL_PRECISION;
            let rank = (suffix.leading_zeros() as usize + 1)
                .min((64 - Self::HLL_PRECISION as usize) + 1) as u8;

            self.registers[bucket] = self.registers[bucket].max(rank);
        }

        self.num_bases_estimate =
            Self::estimate_from_registers(&self.registers, self.row_hashes.len());
        self.num_bases_estimate
    }

    pub fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        self.add_bit_positions(bit_data, &[bit_position])
    }

    pub fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        let _timer =
            ScopedTimer::debug(format!("Adding constant bit positions {:?}", bit_positions));

        let mut added_count = 0usize;
        for &bit_position in bit_positions {
            assert!(
                bit_position < self.base_bit_mask.len(),
                "bit position {} out of bounds for chunk size {}",
                bit_position,
                self.base_bit_mask.len()
            );
            if self.base_bit_mask[bit_position] {
                continue;
            }
            self.base_bit_mask.set(bit_position, true);
            self.base_bit_positions.push(bit_position);
            added_count += 1;
        }

        self.num_bits_per_base += added_count;
        self.num_bases_estimate
    }

    pub fn get_num_bases(&self) -> usize {
        self.num_bases_estimate
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

impl BaseBit for BaseBitHyperLogLogCount {
    fn add_bit_position(&mut self, bit_data: &BitDataSet, bit_position: usize) -> usize {
        BaseBitHyperLogLogCount::add_bit_position(self, bit_data, bit_position)
    }

    fn add_bit_positions(&mut self, bit_data: &BitDataSet, bit_positions: &[usize]) -> usize {
        BaseBitHyperLogLogCount::add_bit_positions(self, bit_data, bit_positions)
    }

    fn add_constant_bit_positions(&mut self, bit_positions: &[usize]) -> usize {
        BaseBitHyperLogLogCount::add_constant_bit_positions(self, bit_positions)
    }

    fn get_bases(&self, _bit_data: &BitDataSet) -> Vec<(BitVec<usize, Msb0>, usize)> {
        panic!(
            "BaseBitHyperLogLogCount does not support get_bases(); use a different BaseBit implementation"
        )
    }

    fn get_groups(&self) -> &[Vec<usize>] {
        panic!(
            "BaseBitHyperLogLogCount does not support get_groups(); use a different BaseBit implementation"
        )
    }

    fn get_num_bases(&self) -> usize {
        BaseBitHyperLogLogCount::get_num_bases(self)
    }

    fn get_num_bits_per_base(&self) -> usize {
        BaseBitHyperLogLogCount::get_num_bits_per_base(self)
    }

    fn get_base_bit_mask(&self) -> &BitSlice<usize, Msb0> {
        BaseBitHyperLogLogCount::get_base_bit_mask(self)
    }

    fn get_base_bit_positions(&self) -> &[usize] {
        BaseBitHyperLogLogCount::get_base_bit_positions(self)
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
            tracing::warn!(
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

    /// Helper function to create standard test data with 6 rows and 8 bits per row
    fn create_test_bit_data() -> BitDataSet {
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
            chunk_size,
            num_rows,
        };

        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();

        BitDataSet {
            data: bit_data,
            info,
        }
    }

    fn print_bases(base_bit_groups: &BaseBitGroups, bit_data: &BitDataSet) {
        tracing::info!("Bases after adding bit:");
        for (i, (base, count)) in base_bit_groups.get_bases(bit_data).iter().enumerate() {
            tracing::info!("Base {}: {:?}, Count: {}", i, base, count);
        }
    }

    #[test]
    fn test_base_bit_groups() {
        let bit_data = create_test_bit_data();
        tracing::info!("BitData: {}", bit_data);

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
        let bit_data = create_test_bit_data();
        tracing::info!("BitData: {}", bit_data);

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
        let bit_data = create_test_bit_data();

        let mut batch_groups = BaseBitBatchGroups::new(bit_data.num_rows(), bit_data.chunk_size());
        let num_bases = batch_groups.add_bit_positions(&bit_data, &[4, 5]);
        assert_eq!(num_bases, 3);
        assert_eq!(batch_groups.get_num_bits_per_base(), 2);
        assert_eq!(
            batch_groups
                .get_groups()
                .iter()
                .map(|g| g.len())
                .sum::<usize>(),
            bit_data.num_rows()
        );
    }

    #[test]
    fn test_base_bit_signature_groups_add_multiple_bits() {
        let bit_data = create_test_bit_data();

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
            bit_data.num_rows()
        );
    }

    #[test]
    fn test_base_bit_hll_count_add_multiple_bits() {
        let bit_data = create_test_bit_data();

        let mut hll_groups =
            BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size());
        let num_bases = hll_groups.add_bit_positions(&bit_data, &[4, 5]);
        assert!(num_bases >= 1);
        assert!(num_bases <= bit_data.num_rows());
        assert_eq!(hll_groups.get_num_bits_per_base(), 2);
    }

    #[test]
    #[should_panic(expected = "does not support get_groups")]
    fn test_base_bit_hll_count_get_groups_panics() {
        let bit_data = create_test_bit_data();
        let hll_groups = BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size());
        let _ = hll_groups.get_groups();
    }

    #[test]
    #[should_panic(expected = "does not support get_bases")]
    fn test_base_bit_hll_count_get_bases_panics() {
        let bit_data = create_test_bit_data();
        let hll_groups = BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size());
        let _ = hll_groups.get_bases(&bit_data);
    }
}
