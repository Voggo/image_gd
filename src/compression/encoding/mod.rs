mod delta_base_table;
mod encoding_core;
mod fused_dictionary;
mod huffman;
mod rle;

pub use self::delta_base_table::{
    DeltaEncodeBaseTable, DeltaEncodeBaseTableFixed, get_delta_codec, get_delta_codec_fixed,
};
pub use self::encoding_core::{
    BaseTable, CompressedData, CondensedSamples, DeltaBaseTableData, DeviationData,
    DeviationSample, EncodeData, EncodeDataOffsetRLE, EncodeDataRLE, EncodedData,
    HuffmanDeviationData, RleDeviationData, RleDeviationOffsetData,
};
pub use self::fused_dictionary::EncodeDataFusedDictionary;
pub use self::huffman::EncodeDataHuffman;
pub(crate) use self::rle::{RLE_LONG_MAX, RLE_SHORT_MAX};
