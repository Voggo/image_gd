pub(crate) const fn bits_needed_nonzero(value: usize) -> usize {
    if value <= 1 {
        1
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

pub(crate) const fn min_position_bits(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

pub(crate) const fn signed_half_wrapped(value: u8) -> u8 {
    ((value as i8) >> 1) as u8
}
