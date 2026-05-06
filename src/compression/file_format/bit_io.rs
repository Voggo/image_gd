use crate::error::EntroGdError;
use bitvec::prelude::*;

#[derive(Debug, Default)]
pub(super) struct BitWriter {
    bytes: Vec<u8>,
    bit_len: usize,
}

#[derive(Debug)]
pub(super) struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    pub(super) fn read_bit(&mut self) -> Result<bool, EntroGdError> {
        if self.bit_pos >= self.bytes.len() * 8 {
            return Err(EntroGdError::InvalidMetadata {
                message: "unexpected end of EGD bitstream".to_string(),
            });
        }
        let byte_index = self.bit_pos / 8;
        let bit_in_byte = self.bit_pos % 8;
        let bit = ((self.bytes[byte_index] >> bit_in_byte) & 1) == 1;
        self.bit_pos += 1;
        Ok(bit)
    }

    pub(super) fn read_u8(&mut self) -> Result<u8, EntroGdError> {
        let mut value = 0u8;
        for i in 0..8 {
            if self.read_bit()? {
                value |= 1 << i;
            }
        }
        Ok(value)
    }

    pub(super) fn read_u16(&mut self) -> Result<u16, EntroGdError> {
        let mut value = 0u16;
        for i in 0..16 {
            if self.read_bit()? {
                value |= 1 << i;
            }
        }
        Ok(value)
    }

    pub(super) fn read_u32(&mut self) -> Result<u32, EntroGdError> {
        let mut value = 0u32;
        for i in 0..32 {
            if self.read_bit()? {
                value |= 1 << i;
            }
        }
        Ok(value)
    }

    pub(super) fn read_u64(&mut self) -> Result<u64, EntroGdError> {
        let mut value = 0u64;
        for i in 0..64 {
            if self.read_bit()? {
                value |= 1 << i;
            }
        }
        Ok(value)
    }

    pub(super) fn read_u64_bits(&mut self, width: usize) -> Result<u64, EntroGdError> {
        if width > 64 {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("cannot read {} bits into u64", width),
            });
        }
        let mut value = 0u64;
        for i in 0..width {
            if self.read_bit()? {
                value |= 1 << i;
            }
        }
        Ok(value)
    }

    pub(super) fn read_usize_bits(&mut self, width: usize) -> Result<usize, EntroGdError> {
        if width > usize::BITS as usize {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("cannot read {} bits into usize", width),
            });
        }
        let mut value = 0usize;
        for i in 0..width {
            if self.read_bit()? {
                value |= 1 << i;
            }
        }
        Ok(value)
    }

    pub(super) fn align_to_byte(&mut self) {
        while !self.bit_pos.is_multiple_of(8) {
            self.bit_pos += 1;
        }
    }

    pub(super) fn remaining_bits(&self) -> usize {
        self.bytes.len() * 8 - self.bit_pos
    }

    pub(super) fn read_bits(&mut self, len: usize) -> Result<BitVec<usize, Lsb0>, EntroGdError> {
        if len > self.remaining_bits() {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "requested {} bits, but only {} remain",
                    len,
                    self.remaining_bits()
                ),
            });
        }
        let mut out = BitVec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_bit()?);
        }
        Ok(out)
    }
}

impl BitWriter {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn write_bit(&mut self, bit: bool) {
        let byte_index = self.bit_len / 8;
        let bit_in_byte = self.bit_len % 8;
        if byte_index == self.bytes.len() {
            self.bytes.push(0);
        }
        if bit {
            self.bytes[byte_index] |= 1 << bit_in_byte;
        }
        self.bit_len += 1;
    }

    pub(super) fn write_bitslice(&mut self, bits: &BitSlice<usize, Lsb0>) {
        for bit in bits {
            self.write_bit(*bit);
        }
    }

    pub(super) fn write_u8(&mut self, value: u8) {
        self.write_usize_bits(value as usize, 8);
    }

    pub(super) fn write_u16(&mut self, value: u16) {
        self.write_usize_bits(value as usize, 16);
    }

    pub(super) fn write_u32(&mut self, value: u32) {
        self.write_usize_bits(value as usize, 32);
    }

    pub(super) fn write_u64(&mut self, value: u64) {
        for shift in 0..64 {
            self.write_bit(((value >> shift) & 1) == 1);
        }
    }

    pub(super) fn write_u64_bits(&mut self, value: u64, width: usize) {
        for shift in 0..width {
            self.write_bit(((value >> shift) & 1) == 1);
        }
    }

    pub(super) fn write_usize_bits(&mut self, value: usize, width: usize) {
        for shift in 0..width {
            self.write_bit(((value >> shift) & 1) == 1);
        }
    }

    pub(super) fn align_to_byte(&mut self) {
        while !self.bit_len.is_multiple_of(8) {
            self.write_bit(false);
        }
    }

    pub(super) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
