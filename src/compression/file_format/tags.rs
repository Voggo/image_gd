use crate::data_loader::FeatureDataType;
use crate::error::EntroGdError;

pub(super) const ENCODING_TAG_NORMAL: u8 = 0;
pub(super) const ENCODING_TAG_RLE_RM_PACKED: u8 = 1;
pub(super) const ENCODING_TAG_HUFFMAN_BASE_ID_ONLY: u8 = 3;
pub(super) const HUFFMAN_CODE_LENGTH_BITS: usize = 5;

pub(super) const BASE_TABLE_TAG_RAW: u8 = 0;
pub(super) const BASE_TABLE_TAG_DELTA: u8 = 1;

pub(super) fn encode_data_type(data_type: FeatureDataType) -> u8 {
    match data_type {
        FeatureDataType::SignedInt => 0,
        FeatureDataType::UnsignedInt => 1,
        FeatureDataType::F32 => 2,
        FeatureDataType::F64 => 3,
    }
}

pub(super) fn decode_data_type(tag: u8) -> Result<FeatureDataType, EntroGdError> {
    match tag {
        0 => Ok(FeatureDataType::SignedInt),
        1 => Ok(FeatureDataType::UnsignedInt),
        2 => Ok(FeatureDataType::F32),
        3 => Ok(FeatureDataType::F64),
        _ => Err(EntroGdError::InvalidMetadata {
            message: format!("unsupported feature data type tag {}", tag),
        }),
    }
}
