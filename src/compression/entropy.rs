use crate::preprocessor::BitData;

pub fn calculate_entropy(bit_data: &BitData) -> Vec<(usize, f64)> {
    (0..bit_data.chunk_size)
        .map(|bit| {
            let count_ones = (0..bit_data.num_rows)
                .filter(|&row| bit_data.get_bit(row, bit))
                .count();
            let p = count_ones as f64 / bit_data.num_rows as f64;
            let entropy = if p == 0.0 || p == 1.0 {
                0.0
            } else {
                -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
            };
            (bit, entropy)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::BitData;

    #[test]
    fn test_entropy_calculation_1() {
        // Create a simple BitData for testing
        let num_rows = 4;
        let chunk_size = 3;
        let num_features = 3;
        let bits_per_feature = 1;
        let data = vec![
            true, false, true,  // Row 0
            true, true, false,  // Row 1
            false, false, true, // Row 2
            false, true, false, // Row 3
        ];
        let bit_data = BitData {
            data: data.into_iter().collect(),
            bits_per_feature,
            num_features,
            num_rows,
            chunk_size,
        };

        let entropies = calculate_entropy(&bit_data);
        println!("Entropies: {:?}", entropies);
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
            true, false, true, false, false, true, true, false,  // Row 0
            true, true, false, true, true, true, false, false,  // Row 1
            false, true, true, false, true, false, true, true, // Row 2
            true, false, false, false, true, true, false, true, // Row 3
            false, true, false, false, true, true, false, true, // Row 4
            true, true, false, true, true, true, false, false,  // Row 5
        ];
        let bit_data = BitData {
            data: data.into_iter().collect(),
            bits_per_feature,
            num_features,
            num_rows,
            chunk_size,
        };

        let entropies = calculate_entropy(&bit_data);
        println!("Entropies: {:?}", entropies);
        assert_eq!(entropies.len(), chunk_size);
        // Add more assertions based on expected entropy values
    }
}