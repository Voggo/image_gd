use crate::compression::compress::{CompressedData, CondensedSamples, build_base_bit_mask};
use crate::compression::tabular_preprocessor::{BitData, BitDataSet};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;
use bitvec::prelude::*;
use std::sync::Arc;

pub struct DecompressRowsData;

impl Filter for DecompressRowsData {
    type Input = (Arc<CompressedData>, Vec<usize>);
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info(format!("Decompressing {} rows", input.1.len()));
        let (compressed, indices) = input;
        decompress_samples_batch(compressed.as_ref(), &indices)
    }
}

pub(crate) fn decompress_samples_batch(
    compressed: &CompressedData,
    indices: &[usize],
) -> Result<BitDataSet, EntroGdError> {
    let data_info = &compressed.metadata;
    let num_features = data_info.num_features();
    let chunk_size = data_info.chunk_size();
    let original_num_rows = data_info.original_size_bits() / chunk_size;
    if num_features == 0 {
        return Err(EntroGdError::InvalidMetadata {
            message: "num_features is 0".to_string(),
        });
    }
    if let Some(&sample_idx) = indices.iter().find(|&&idx| idx >= original_num_rows) {
        return Err(EntroGdError::DecompressionSampleMissing { sample_idx });
    }
    let base_bit_positions = &compressed.base_bit_positions;
    let base_bit_mask = build_base_bit_mask(chunk_size, base_bit_positions);

    let mut reconstructed_bits = BitVec::<usize, Msb0>::with_capacity(chunk_size * indices.len());

    for &sample_idx in indices {
        let sample = match compressed.encoded_data.get_sample(sample_idx) {
            Some(s) => s,
            None => {
                return Err(EntroGdError::DecompressionSampleMissing { sample_idx });
            }
        };

        let mut chunk = bitvec![usize, Msb0; 0; chunk_size];
        let mut base_id = 0usize;
        for i in 0..sample.id.len() {
            base_id = (base_id << 1) | (sample.id[i] as usize);
        }
        if base_id >= compressed.base_table.len() {
            return Err(EntroGdError::InvalidBaseId {
                base_id,
                table_len: compressed.base_table.len(),
            });
        }
        let base_pattern = &compressed.base_table[base_id].0;
        for bit_pos in 0..chunk_size.min(base_pattern.len()) {
            chunk.set(bit_pos, base_pattern[bit_pos]);
        }
        let mut deviation_bit_idx = 0;
        for bit_pos in 0..chunk_size {
            if !base_bit_mask[bit_pos] && deviation_bit_idx < sample.deviation.len() {
                chunk.set(bit_pos, sample.deviation[deviation_bit_idx]);
                deviation_bit_idx += 1;
            }
        }
        reconstructed_bits.extend_from_bitslice(&chunk);
    }

    let data = BitData {
        data: reconstructed_bits,
        chunk_size,
        num_rows: indices.len(),
    };
    let info = data_info.with_original_size_bits(chunk_size * indices.len());
    Ok(BitDataSet { data, info })
}

pub struct DecompressFileData {}

impl Filter for DecompressFileData {
    type Input = CompressedData;
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Decompressing entire file data");
        decompress_file(&input)
    }
}

pub fn decompress_file(compressed: &CompressedData) -> Result<BitDataSet, EntroGdError> {
    let data_info = &compressed.metadata;
    let original_num_rows = data_info.original_size_bits() / data_info.chunk_size();
    let indices: Vec<usize> = (0..original_num_rows).collect();
    decompress_samples_batch(compressed, &indices)
}

pub struct DecompressAnalytics {}

impl Filter for DecompressAnalytics {
    type Input = CompressedData;
    type Output = Option<CondensedSamples>;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Decompressing condensed samples for analytics");
        Ok(decompress_analytics(&input))
    }
}

pub fn decompress_analytics(compressed: &CompressedData) -> Option<CondensedSamples> {
    if let Some(weights) = &compressed.condensed_sample_weights {
        let samples: Vec<BitVec<usize, Msb0>> = compressed
            .base_table
            .iter()
            .map(|(bv, _)| bv.clone())
            .collect();
        Some(CondensedSamples {
            samples,
            weights: weights.clone(),
        })
    } else {
        None
    }
}
