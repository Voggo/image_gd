use super::{DeviationData, DeviationSample, DeviationSampleRef, RleDeviationData};
use crate::error::EntroGdError;

pub(crate) const RLE_SHORT_MAX: u8 = 7;
pub(crate) const RLE_LONG_MIN: u8 = 8;
pub(crate) const RLE_LONG_MAX: u8 = 134;
pub(crate) const RLE_TERMINATOR_PAYLOAD: u8 = 0x7F;

impl RleDeviationData {
    pub fn new(
        symbol_bit_stream: crate::BitStream,
        rm_values: Vec<(u8, u8)>,
        num_samples: usize,
        num_deviation_bits: usize,
        num_id_bits: usize,
    ) -> Self {
        let mut rm_control_stream = crate::BitStream::new();
        for &(r, m) in &rm_values {
            write_rle_control_value(&mut rm_control_stream, r);
            write_rle_control_value(&mut rm_control_stream, m);
        }
        // Terminator packet: 1-bit long-packet flag + 7-bit reserved payload (= 127) -> 0xFF.
        rm_control_stream.push(true);
        for shift in (0..7).rev() {
            rm_control_stream.push(((RLE_TERMINATOR_PAYLOAD >> shift) & 1) == 1);
        }

        RleDeviationData {
            symbol_bit_stream,
            rm_values,
            rm_control_stream,
            num_samples,
            num_deviation_bits,
            num_id_bits,
        }
    }

    pub fn get_sample(&self, sample_idx: usize) -> Option<DeviationSample> {
        if sample_idx >= self.num_samples {
            return None;
        }

        let symbol_width = self.num_deviation_bits + self.num_id_bits;
        let mut symbol_cursor = 0usize;
        let mut logical_idx = 0usize;

        for &(r_encoded, m_count) in &self.rm_values {
            if r_encoded > 0 {
                if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return None;
                }
                let run_symbol =
                    self.symbol_bit_stream[symbol_cursor..symbol_cursor + symbol_width].to_bitvec();
                symbol_cursor += symbol_width;

                let run_len = (r_encoded as usize) + 1;
                if sample_idx < logical_idx + run_len {
                    return Some(DeviationSample {
                        deviation: run_symbol[0..self.num_deviation_bits].to_bitvec(),
                        id: run_symbol[self.num_deviation_bits..].to_bitvec(),
                    });
                }
                logical_idx += run_len;
                if logical_idx >= self.num_samples {
                    break;
                }
            }

            let literals = m_count as usize;
            if sample_idx < logical_idx + literals {
                let offset = sample_idx - logical_idx;
                let start = symbol_cursor + offset * symbol_width;
                let end = start + symbol_width;
                if end > self.symbol_bit_stream.len() {
                    return None;
                }
                let symbol = self.symbol_bit_stream[start..end].to_bitvec();
                return Some(DeviationSample {
                    deviation: symbol[0..self.num_deviation_bits].to_bitvec(),
                    id: symbol[self.num_deviation_bits..].to_bitvec(),
                });
            }

            symbol_cursor += literals * symbol_width;
            logical_idx += literals;
            if logical_idx >= self.num_samples {
                break;
            }
        }

        None
    }

    pub fn symbol_bit_stream(&self) -> &crate::BitStream {
        &self.symbol_bit_stream
    }

    pub fn rm_control_stream(&self) -> &crate::BitStream {
        &self.rm_control_stream
    }

    pub fn rm_values(&self) -> &[(u8, u8)] {
        &self.rm_values
    }

    pub fn get_num_samples(&self) -> usize {
        self.num_samples
    }

    pub fn get_num_deviation_bits(&self) -> usize {
        self.num_deviation_bits
    }

    pub fn get_num_id_bits(&self) -> usize {
        self.num_id_bits
    }

    pub fn get_encoded_size(&self) -> usize {
        self.symbol_bit_stream.len() + self.rm_control_stream.len()
    }

    pub(crate) fn for_each_sample(
        &self,
        mut f: impl FnMut(DeviationSampleRef<'_>) -> Result<(), EntroGdError>,
    ) -> Result<(), EntroGdError> {
        let symbol_width = self.num_deviation_bits + self.num_id_bits;
        let mut symbol_cursor = 0usize;
        let mut decoded_samples = 0usize;

        for &(r_encoded, m_count) in &self.rm_values {
            if r_encoded > 0 {
                let run_len = (r_encoded as usize) + 1;
                if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "RLE run symbol exceeds symbol stream length".to_string(),
                    });
                }

                let run_symbol =
                    &self.symbol_bit_stream[symbol_cursor..symbol_cursor + symbol_width];
                let sample = DeviationSampleRef {
                    deviation: &run_symbol[..self.num_deviation_bits],
                    id: &run_symbol[self.num_deviation_bits..],
                };
                for _ in 0..run_len {
                    f(DeviationSampleRef {
                        deviation: sample.deviation,
                        id: sample.id,
                    })?;
                }
                symbol_cursor += symbol_width;
                decoded_samples += run_len;
            }

            let literal_count = m_count as usize;
            for _ in 0..literal_count {
                if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "RLE literal symbol exceeds symbol stream length".to_string(),
                    });
                }

                let literal = &self.symbol_bit_stream[symbol_cursor..symbol_cursor + symbol_width];
                f(DeviationSampleRef {
                    deviation: &literal[..self.num_deviation_bits],
                    id: &literal[self.num_deviation_bits..],
                })?;
                symbol_cursor += symbol_width;
            }
            decoded_samples += literal_count;
        }

        if decoded_samples != self.num_samples {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "RLE decoded sample count mismatch: expected {}, got {}",
                    self.num_samples, decoded_samples
                ),
            });
        }

        if symbol_cursor != self.symbol_bit_stream.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "RLE symbol stream not fully consumed: consumed {} of {} bits",
                    symbol_cursor,
                    self.symbol_bit_stream.len()
                ),
            });
        }

        Ok(())
    }

    pub fn to_deviation_data(&self) -> Result<DeviationData, EntroGdError> {
        let symbol_width = self.num_deviation_bits + self.num_id_bits;
        let expected_raw_bits = self.num_samples.checked_mul(symbol_width).ok_or_else(|| {
            EntroGdError::InvalidMetadata {
                message: "raw deviation stream length overflow".to_string(),
            }
        })?;

        let mut raw = crate::BitStream::with_capacity(expected_raw_bits);
        let mut symbol_cursor = 0usize;
        let mut decoded_samples = 0usize;

        for &(r_encoded, m_count) in &self.rm_values {
            if r_encoded > 0 {
                let run_len = (r_encoded as usize) + 1;
                if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "RLE run symbol exceeds symbol stream length".to_string(),
                    });
                }

                let run_symbol =
                    &self.symbol_bit_stream[symbol_cursor..symbol_cursor + symbol_width];
                for _ in 0..run_len {
                    raw.extend_from_bitslice(run_symbol);
                }
                symbol_cursor += symbol_width;
                decoded_samples += run_len;
            }

            let literal_count = m_count as usize;
            for _ in 0..literal_count {
                if symbol_cursor + symbol_width > self.symbol_bit_stream.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "RLE literal symbol exceeds symbol stream length".to_string(),
                    });
                }

                raw.extend_from_bitslice(
                    &self.symbol_bit_stream[symbol_cursor..symbol_cursor + symbol_width],
                );
                symbol_cursor += symbol_width;
            }
            decoded_samples += literal_count;
        }

        if decoded_samples != self.num_samples {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "RLE decoded sample count mismatch: expected {}, got {}",
                    self.num_samples, decoded_samples
                ),
            });
        }

        if symbol_cursor != self.symbol_bit_stream.len() {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "RLE symbol stream not fully consumed: consumed {} of {} bits",
                    symbol_cursor,
                    self.symbol_bit_stream.len()
                ),
            });
        }

        Ok(DeviationData::new(
            raw,
            self.num_samples,
            self.num_deviation_bits,
            self.num_id_bits,
        ))
    }
}

pub(super) fn write_rle_control_value(out: &mut crate::BitStream, value: u8) {
    assert!(
        value <= RLE_LONG_MAX,
        "RLE control value {} out of range (max {})",
        value,
        RLE_LONG_MAX
    );

    if value <= RLE_SHORT_MAX {
        // 4-bit packet: 0 + 3-bit payload
        out.push(false);
        for shift in (0..3).rev() {
            out.push(((value >> shift) & 1) == 1);
        }
    } else {
        // 8-bit packet: 1 + 7-bit payload with bias -8 (stored payload = value - 8)
        out.push(true);
        let payload = value - RLE_LONG_MIN;
        for shift in (0..7).rev() {
            out.push(((payload >> shift) & 1) == 1);
        }
    }
}
