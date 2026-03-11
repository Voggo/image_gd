pub(crate) const fn bits_needed_nonzero(value: usize) -> usize {
    if value <= 1 {
        1
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}