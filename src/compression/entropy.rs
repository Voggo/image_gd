use crate::compression::preprocessor::BitDataSet;
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;

pub struct EntropyNaive;

impl Filter for EntropyNaive {
    type Input = BitDataSet;
    type Output = (BitDataSet, Vec<(usize, f64)>);

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Calculating entropy for each bit position");
        let entropy = calculate_entropy(&input);
        Ok((input, entropy))
    }
}

pub fn calculate_entropy(bit_data: &BitDataSet) -> Vec<(usize, f64)> {
    (0..bit_data.chunk_size())
        .map(|bit| {
            let count_ones = (0..bit_data.num_rows())
                .filter(|&row| bit_data.get_bit(row, bit))
                .count();
            let p = count_ones as f64 / bit_data.num_rows() as f64;
            let entropy = if p == 0.0 || p == 1.0 {
                0.0
            } else {
                -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
            };
            (bit, entropy)
        })
        .collect()
}

pub struct EntropyOptimized {}

impl Filter for EntropyOptimized {
    type Input = BitDataSet;
    type Output = (BitDataSet, Vec<(usize, f64)>);

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Calculating entropy for each bit position (optimized)");
        let entropy = calculate_entropy_optimized(&input);
        Ok((input, entropy))
    }
}

pub fn calculate_entropy_optimized(bit_data: &BitDataSet) -> Vec<(usize, f64)> {
    let num_rows = bit_data.num_rows();
    let chunk_size = bit_data.chunk_size();
    let inv_rows = 1.0 / num_rows as f64;
    const WORD_SIZE: usize = 64; // Assuming 64 bit, should be read from cpu architecture

    let words_per_row = (chunk_size + WORD_SIZE - 1) / WORD_SIZE; 

    let mut ones_count = vec![0usize; chunk_size];

    let raw = bit_data.data.raw().as_raw_slice();

    for row in 0..num_rows {
        let row_offset = row * words_per_row;

        for word_idx in 0..words_per_row {
            // read the word for this as a usize (assuming 64-bit architecture)
            let word = raw[row_offset + word_idx];

            let base_bit = word_idx * WORD_SIZE;

            let mut w = word;

            // This loop is optimized for counting sparse bits (1s) in the word
            // runs in O(number of 1s) instead of O(chunk_size) for each row
            while w != 0 {
                let tz = w.trailing_zeros() as usize;

                // Flip for Msb0
                let bit_in_word = (usize::BITS as usize - 1) - tz;
                let bit_index = base_bit + bit_in_word;

                if bit_index < chunk_size {
                    ones_count[bit_index] += 1;
                }
                // This is magic:
                // the w - 1 flips the lsb that is set to 1, and all bits below it to 1
                // using and to remove the new 1's 
                // w       = 1001000
                // w - 1   = 1000111
                // ------------------
                // w&(w-1) = 1000000
                w &= w - 1;
            }
        }
    }

    ones_count
        .into_iter()
        .enumerate()
        .map(|(bit, count)| {
            let entropy = if count == 0 || count == num_rows {
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
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropies = calculate_entropy(&bit_data);
        log::info!("Entropies: {:?}", entropies);
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
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropies = calculate_entropy(&bit_data);
        log::info!("Entropies: {:?}", entropies);
        assert_eq!(entropies.len(), chunk_size);
        // Add more assertions based on expected entropy values
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
            num_rows,
        };
        let features =
            vec![FeatureSpec::new(FeatureDataType::UnsignedInt, bits_per_feature); num_features];
        let info = BitDataInfo::new(features, chunk_size * num_rows).unwrap();
        let bit_data = BitDataSet { data, info };

        let entropy_filter = EntropyNaive;
        let (_bit_data, entropies) = entropy_filter.process(bit_data).unwrap();
        log::info!("Entropies from pipeline: {:?}", entropies);
        assert_eq!(entropies.len(), chunk_size);
    }
}
