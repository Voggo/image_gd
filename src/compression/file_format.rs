mod bit_io;
mod egd;
mod igd;
mod path_utils;
mod tags;

pub use egd::{
    EgdFile, FORMAT_VERSION, LoadEgdFile, MAGIC_BYTES, SaveEgdFile, decompress_egd_to_csv,
    load_and_decompress_egd, load_compressed_from_egd, save_compressed_as_egd,
};
pub use igd::{
    IMAGE_FORMAT_VERSION, IMAGE_MAGIC_BYTES, IgdFile, LoadIgdFile, SaveIgdFile,
    decompress_igd_to_image, load_and_decompress_igd, load_compressed_from_igd,
    save_compressed_as_igd,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::base_selection::SelectBases;
    use crate::compression::compress::{CompressedData, EncodedData};
    use crate::compression::condensed_samples::GenCondensedSamples;
    use crate::compression::decompression::decompress_file;
    use crate::compression::encoding::{EncodeData, EncodeDataHuffman, EncodeDataRLE};
    use crate::compression::entropy::EntropyOptimized;
    use crate::compression::preprocessor::{
        BitData, BitDataInfo, BitDataReconstructionInfo, BitDataSet, FeatureSpec, ImageColorModel,
        ImageGroupingTransform, ImageReconstructionInfo,
    };
    use crate::data_loader::FeatureDataType;
    use crate::filter_pipeline::{Filter, FilterExt};

    fn get_compression_pipeline() -> impl Filter<Input = BitDataSet, Output = CompressedData> {
        EntropyOptimized {}
            .then(GenCondensedSamples { m_max: 50 })
            .then(SelectBases { patience: 10 })
            .then(EncodeData {})
    }

    fn get_rle_compression_pipeline() -> impl Filter<Input = BitDataSet, Output = CompressedData> {
        EntropyOptimized {}
            .then(GenCondensedSamples { m_max: 100 })
            .then(SelectBases { patience: 5 })
            .then(EncodeDataRLE {})
    }

    fn get_huffman_compression_pipeline() -> impl Filter<Input = BitDataSet, Output = CompressedData>
    {
        EntropyOptimized {}
            .then(GenCondensedSamples { m_max: 100 })
            .then(SelectBases { patience: 5 })
            .then(EncodeDataHuffman {})
    }

    #[test]
    fn test_build_and_save_egd() {
        let data = BitData {
            data: bitvec::bitvec![usize, bitvec::order::Msb0; 0; 64],
            num_rows: 1,
            chunk_size: 64,
        };
        let features = vec![FeatureSpec::new(FeatureDataType::UnsignedInt, 8); 8];
        let info = BitDataInfo::new(features, 64).unwrap();
        let bit_data = BitDataSet { data, info };
        let pipeline = get_compression_pipeline();
        let compressed = pipeline.process(bit_data).unwrap();
        let egd = EgdFile::from_compressed_data(&compressed).unwrap();

        assert!(egd.as_bytes().len() >= 4);
        assert_eq!(&egd.as_bytes()[0..3], &MAGIC_BYTES);
        assert_eq!(egd.as_bytes()[3], FORMAT_VERSION);

        let output = std::env::temp_dir().join("entro_gd_test_output");
        let saved = egd.save(&output).unwrap();
        assert_eq!(saved.extension().and_then(|s| s.to_str()), Some("egd"));
        assert!(saved.exists());
        let _ = std::fs::remove_file(saved);
    }

    #[test]
    fn test_roundtrip_egd_to_compressed_data() {
        let data = BitData {
            data: bitvec::bitvec![usize, bitvec::order::Msb0; 0; 320],
            num_rows: 5,
            chunk_size: 64,
        };
        let features = vec![
            FeatureSpec::new(FeatureDataType::UnsignedInt, 32),
            FeatureSpec::new(FeatureDataType::UnsignedInt, 32),
        ];
        let info = BitDataInfo::new(features, 320).unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_compression_pipeline().process(bit_data).unwrap();

        let egd = EgdFile::from_compressed_data(&compressed).unwrap();
        let loaded = egd.to_compressed_data().unwrap();

        assert_eq!(loaded.metadata, compressed.metadata);
        assert_eq!(loaded.base_bit_positions, compressed.base_bit_positions);
        assert_eq!(
            loaded.condensed_sample_weights,
            compressed.condensed_sample_weights
        );
        assert_eq!(
            loaded.encoded_data.encoded_bit_stream(),
            compressed.encoded_data.encoded_bit_stream()
        );
        assert_eq!(
            loaded.encoded_data.get_num_samples(),
            compressed.encoded_data.get_num_samples()
        );
        assert_eq!(
            loaded.encoded_data.get_num_deviation_bits(),
            compressed.encoded_data.get_num_deviation_bits()
        );
        assert_eq!(
            loaded.encoded_data.get_num_id_bits(),
            compressed.encoded_data.get_num_id_bits()
        );
        assert_eq!(loaded.base_table.len(), compressed.base_table.len());
        for (lhs, rhs) in loaded.base_table.iter().zip(compressed.base_table.iter()) {
            assert_eq!(lhs.0, rhs.0);
            assert_eq!(lhs.1, rhs.1);
        }
    }

    #[test]
    fn test_roundtrip_igd_to_compressed_data_with_image_metadata() {
        let data = BitData {
            data: bitvec::bitvec![usize, bitvec::order::Msb0; 0; 96],
            num_rows: 4,
            chunk_size: 24,
        };
        let features = vec![FeatureSpec::new(FeatureDataType::UnsignedInt, 8); 3];
        let info = BitDataInfo::new_with_reconstruction_info(
            features,
            96,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 2,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: 1,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0,
            }),
        )
        .unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_compression_pipeline().process(bit_data).unwrap();

        let igd = IgdFile::from_compressed_data(&compressed).unwrap();
        let loaded = igd.to_compressed_data().unwrap();

        assert_eq!(loaded.metadata.num_features(), 3);
        assert_eq!(loaded.metadata.original_size_bits(), 96);
        assert!(matches!(
            loaded.metadata.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 2,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: 1,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0
            })
        ));
    }

    #[test]
    fn test_roundtrip_igd_to_compressed_data_with_for_min_metadata() {
        let data = BitData {
            data: bitvec::bitvec![usize, bitvec::order::Msb0; 0; 111],
            num_rows: 1,
            chunk_size: 111,
        };
        let features = vec![FeatureSpec::new(FeatureDataType::UnsignedInt, 37); 3];
        let info = BitDataInfo::new_with_reconstruction_info(
            features,
            111,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 4,
                height: 1,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: 4,
                grouping_transform: ImageGroupingTransform::ForMin,
                colorspace: 0,
            }),
        )
        .unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_compression_pipeline().process(bit_data).unwrap();

        let igd = IgdFile::from_compressed_data(&compressed).unwrap();
        let loaded = igd.to_compressed_data().unwrap();

        assert!(matches!(
            loaded.metadata.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 4,
                height: 1,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: 4,
                grouping_transform: ImageGroupingTransform::ForMin,
                colorspace: 0
            })
        ));
    }

    #[test]
    fn test_roundtrip_egd_with_rle_payload() {
        let data = BitData {
            data: bitvec::bitvec![usize, bitvec::order::Msb0; 0; 512],
            num_rows: 8,
            chunk_size: 64,
        };
        let features = vec![FeatureSpec::new(FeatureDataType::UnsignedInt, 8); 8];
        let info = BitDataInfo::new(features, 512).unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_rle_compression_pipeline().process(bit_data).unwrap();
        let egd = EgdFile::from_compressed_data(&compressed).unwrap();
        let loaded = egd.to_compressed_data().unwrap();

        match (&compressed.encoded_data, &loaded.encoded_data) {
            (EncodedData::Rle(src), EncodedData::Normal(dst)) => {
                let src_as_raw = src.to_deviation_data().unwrap();
                assert_eq!(src_as_raw.encoded_bit_stream(), dst.encoded_bit_stream());
                assert_eq!(src_as_raw.get_num_samples(), dst.get_num_samples());
                assert_eq!(
                    src_as_raw.get_num_deviation_bits(),
                    dst.get_num_deviation_bits()
                );
                assert_eq!(src_as_raw.get_num_id_bits(), dst.get_num_id_bits());
            }
            _ => panic!("expected source RLE data and loaded normalized data"),
        }
    }

    #[test]
    fn test_roundtrip_egd_with_huffman_payload() {
        let data = BitData {
            data: bitvec::bitvec![usize, bitvec::order::Msb0;
                0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 1, 0, 0, 1, 1, 0,
                1, 0, 0, 1, 1, 0, 0, 1,
                1, 1, 0, 0, 1, 1, 0, 0,
            ],
            num_rows: 4,
            chunk_size: 8,
        };
        let features = vec![FeatureSpec::new(FeatureDataType::UnsignedInt, 8)];
        let info = BitDataInfo::new(features, 32).unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_huffman_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        let egd = EgdFile::from_compressed_data(&compressed).unwrap();
        let loaded = egd.to_compressed_data().unwrap();

        assert!(matches!(loaded.encoded_data, EncodedData::Huffman(_)));
        assert_eq!(loaded.metadata, compressed.metadata);
        assert_eq!(
            decompress_file(&loaded).unwrap().data.data,
            bit_data.data.data[0..32]
        );
    }

    #[test]
    fn test_roundtrip_igd_with_huffman_image_payload() {
        let data = BitData {
            data: bitvec::bitvec![usize, bitvec::order::Msb0;
                0,0,0,1, 0,0,1,0, 0,0,1,1,
                0,1,0,0, 0,1,0,1, 0,1,1,0,
                0,1,1,1, 1,0,0,0, 1,0,0,1,
                1,0,1,0, 1,0,1,1, 1,1,0,0,
            ],
            num_rows: 4,
            chunk_size: 12,
        };
        let features = vec![FeatureSpec::new(FeatureDataType::UnsignedInt, 4); 3];
        let info = BitDataInfo::new_with_reconstruction_info(
            features,
            48,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 2,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: 1,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0,
            }),
        )
        .unwrap();
        let bit_data = BitDataSet { data, info };

        let compressed = get_huffman_compression_pipeline()
            .process(bit_data.clone())
            .unwrap();
        let igd = IgdFile::from_compressed_data(&compressed).unwrap();
        let loaded = igd.to_compressed_data().unwrap();

        assert!(matches!(loaded.encoded_data, EncodedData::Huffman(_)));
        assert!(matches!(
            loaded.metadata.reconstruction,
            BitDataReconstructionInfo::Image(ImageReconstructionInfo {
                width: 2,
                height: 2,
                channels: 3,
                color_model: ImageColorModel::Rgb,
                pixel_grouping: 1,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                colorspace: 0
            })
        ));
        assert_eq!(
            decompress_file(&loaded).unwrap().data.data,
            bit_data.data.data[0..48]
        );
    }
}
