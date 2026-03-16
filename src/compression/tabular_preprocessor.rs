pub use crate::compression::preprocessor::{
    BitData, BitDataCompressionInfo, BitDataInfo, BitDataReconstructionInfo, BitDataSet,
    FeatureSpec, FeatureTransform, FloatScalingMode, ImageColorModel, ImageGroupingTransform,
    ImageReconstructionInfo, PreprocessOptions,
    decode_value_from_bits, FeatureDataType,
};

use std::path::Path;
use tracing::{debug, info, trace};

use crate::data_loader::{DataLoader, DataValue, Dataset, DatasetMetadata};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;

const MAX_DECIMAL_SCALE: u8 = 9;

/// Filter for inferring feature specifications from a tabular dataset.
pub struct InferFeatureSpecs {
    pub options: PreprocessOptions,
}

impl Filter for InferFeatureSpecs {
    type Input = Dataset;
    type Output = (Dataset, Vec<FeatureSpec>);

    fn process(&self, dataset: Self::Input) -> Result<Self::Output, EntroGdError> {
        let specs = infer_feature_specs(&dataset, self.options);
        Ok((dataset, specs))
    }
}

/// Filter for building BitDataSet from a Dataset with feature specifications.
pub struct BuildBitDataSet;

impl Filter for BuildBitDataSet {
    type Input = (Dataset, Vec<FeatureSpec>);
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let (dataset, features) = input;
        BitDataSet::from_dataset_with_schema(&dataset, features)
    }
}

impl BitDataSet {
    /// Create BitDataSet from a Dataset using column data types to derive the schema.
    pub fn from_dataset(dataset: &Dataset) -> Result<Self, EntroGdError> {
        Self::from_dataset_with_options(dataset, PreprocessOptions::default())
    }

    /// Create BitDataSet from a Dataset using configurable preprocessing options.
    pub fn from_dataset_with_options(
        dataset: &Dataset,
        options: PreprocessOptions,
    ) -> Result<Self, EntroGdError> {
        let _timer = ScopedTimer::info("Converting Dataset to BitDataSet");
        let num_features = dataset.num_columns();
        info!(
            rows = dataset.num_rows(),
            columns = num_features,
            "starting dataset preprocessing"
        );
        let features = infer_feature_specs(dataset, options);
        Self::from_dataset_with_schema(dataset, features)
    }

    /// Load data via a DataLoader and build BitDataSet using column data types.
    pub fn from_loader<L: DataLoader, P: AsRef<Path>>(
        loader: &L,
        path: P,
    ) -> Result<(Self, DatasetMetadata), EntroGdError> {
        info!(path = %path.as_ref().display(), "loading dataset from path");
        let loaded = loader.load(path)?;
        debug!(
            rows = loaded.dataset.num_rows(),
            columns = loaded.dataset.num_columns(),
            "loaded dataset metadata"
        );
        let bit_data = Self::from_dataset(&loaded.dataset)?;
        Ok((bit_data, loaded.metadata))
    }

    /// Load data via a DataLoader and build BitDataSet using a provided schema.
    pub fn from_loader_with_schema<L: DataLoader, P: AsRef<Path>>(
        loader: &L,
        path: P,
        features: Vec<FeatureSpec>,
    ) -> Result<(Self, DatasetMetadata), EntroGdError> {
        info!(
            path = %path.as_ref().display(),
            schema_features = features.len(),
            "loading dataset from path with user-provided schema"
        );
        let loaded = loader.load(path)?;
        debug!(
            rows = loaded.dataset.num_rows(),
            columns = loaded.dataset.num_columns(),
            "loaded dataset metadata"
        );
        let bit_data = Self::from_dataset_with_schema(&loaded.dataset, features)?;
        Ok((bit_data, loaded.metadata))
    }
}

fn infer_feature_specs(dataset: &Dataset, options: PreprocessOptions) -> Vec<FeatureSpec> {
    let num_features = dataset.num_columns();
    let mut features = Vec::with_capacity(num_features);
    for feature_idx in 0..num_features {
        let data_type = dataset
            .column_type(feature_idx)
            .expect("feature index should be in bounds while inferring schema");
        let spec = match data_type {
            FeatureDataType::F32 | FeatureDataType::F64 => {
                infer_float_feature_spec(dataset, feature_idx, data_type, options)
            }
            FeatureDataType::SignedInt | FeatureDataType::UnsignedInt => {
                infer_integer_feature_spec(dataset, feature_idx, data_type, options)
            }
        };
        debug!(
            "Inferred feature {} schema: type={:?}, bits={}, transform={:?}",
            feature_idx, spec.data_type, spec.bits, spec.transform
        );
        features.push(spec);
    }
    features
}

fn infer_float_feature_spec(
    dataset: &Dataset,
    column: usize,
    data_type: FeatureDataType,
    options: PreprocessOptions,
) -> FeatureSpec {
    let fallback_bits = match data_type {
        FeatureDataType::F32 => 32,
        FeatureDataType::F64 => 64,
        _ => unreachable!("infer_float_feature_spec called with non-float type"),
    };

    if matches!(options.float_scaling, FloatScalingMode::Disabled) {
        return FeatureSpec::new(data_type, fallback_bits);
    }

    debug!(
        "Inferring float feature spec for column {} with type {:?} and fallback_bits={}",
        column, data_type, fallback_bits
    );

    let mut best_scale: Option<u8> = None;
    for decimal_scale in 0..=options.max_decimal_scale.min(MAX_DECIMAL_SCALE) {
        trace!(
            "Trying decimal_scale={} for float column {}",
            decimal_scale, column
        );
        if scaled_int_range(dataset, column, data_type, decimal_scale).is_some() {
            best_scale = Some(decimal_scale);
            break;
        }
    }

    if let Some(decimal_scale) = best_scale {
        let (min_value, max_value) = scaled_int_range(dataset, column, data_type, decimal_scale)
            .expect("scale should have a valid integer range");
        match options.float_scaling {
            FloatScalingMode::ScaledOffsetSignedInt => {
                let shifted_max = shifted_max_signed(min_value, max_value);
                let bits = storage_bucket_bits(shifted_max);
                debug!(
                    "Selected scaled+offset transform for column {}: decimal_scale={}, min={}, max={}, shifted_max={}, bits={}",
                    column, decimal_scale, min_value, max_value, shifted_max, bits
                );
                FeatureSpec {
                    data_type,
                    bits,
                    transform: FeatureTransform::ScaledOffsetSignedInt {
                        decimal_scale,
                        min_value,
                    },
                }
            }
            FloatScalingMode::ScaledSignedInt => {
                let bits = signed_storage_bucket_bits(min_value, max_value);
                debug!(
                    "Selected scaled signed transform for column {}: decimal_scale={}, min={}, max={}, bits={}",
                    column, decimal_scale, min_value, max_value, bits
                );
                FeatureSpec {
                    data_type,
                    bits,
                    transform: FeatureTransform::ScaledSignedInt { decimal_scale },
                }
            }
            FloatScalingMode::Disabled => FeatureSpec::new(data_type, fallback_bits),
        }
    } else {
        debug!(
            "No lossless scaled-int transform found for column {}; using raw float bits",
            column
        );
        FeatureSpec::new(data_type, fallback_bits)
    }
}

fn infer_integer_feature_spec(
    dataset: &Dataset,
    column: usize,
    data_type: FeatureDataType,
    options: PreprocessOptions,
) -> FeatureSpec {
    match data_type {
        FeatureDataType::SignedInt => {
            let mut min_value = i64::MAX;
            let mut max_value = i64::MIN;

            for row in 0..dataset.num_rows() {
                let value = match dataset.value_at(row, column).expect("row/column should be valid") {
                    DataValue::Signed(v) => v,
                    _ => unreachable!("signed column type mismatch while inferring schema"),
                };
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }

            if dataset.num_rows() == 0 {
                min_value = 0;
                max_value = 0;
            }

            if options.integer_zero_normalization {
                let shifted_max = shifted_max_signed(min_value, max_value);
                let bits = storage_bucket_bits(shifted_max);

                debug!(
                    "Inferred signed integer column {} (offset): min={}, max={}, shifted_max={}, bits={}",
                    column, min_value, max_value, shifted_max, bits
                );

                FeatureSpec {
                    data_type,
                    bits,
                    transform: FeatureTransform::OffsetSignedInt { min_value },
                }
            } else {
                let bits = signed_storage_bucket_bits(min_value, max_value);
                debug!(
                    "Inferred signed integer column {} (no offset): min={}, max={}, bits={}",
                    column, min_value, max_value, bits
                );
                FeatureSpec {
                    data_type,
                    bits,
                    transform: FeatureTransform::None,
                }
            }
        }
        FeatureDataType::UnsignedInt => {
            let mut min_value = u64::MAX;
            let mut max_value = u64::MIN;

            for row in 0..dataset.num_rows() {
                let value = match dataset.value_at(row, column).expect("row/column should be valid") {
                    DataValue::Unsigned(v) => v,
                    _ => unreachable!("unsigned column type mismatch while inferring schema"),
                };
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }

            if dataset.num_rows() == 0 {
                min_value = 0;
                max_value = 0;
            }

            if options.integer_zero_normalization {
                let shifted_max = max_value.saturating_sub(min_value);
                let bits = storage_bucket_bits(shifted_max);

                debug!(
                    "Inferred unsigned integer column {} (offset): min={}, max={}, shifted_max={}, bits={}",
                    column, min_value, max_value, shifted_max, bits
                );

                FeatureSpec {
                    data_type,
                    bits,
                    transform: FeatureTransform::OffsetUnsignedInt { min_value },
                }
            } else {
                let bits = storage_bucket_bits(max_value);
                debug!(
                    "Inferred unsigned integer column {} (no offset): min={}, max={}, bits={}",
                    column, min_value, max_value, bits
                );
                FeatureSpec {
                    data_type,
                    bits,
                    transform: FeatureTransform::None,
                }
            }
        }
        _ => unreachable!("infer_integer_feature_spec called with non-integer type"),
    }
}

fn shifted_max_signed(min_value: i64, max_value: i64) -> u64 {
    ((max_value as i128) - (min_value as i128)) as u64
}

fn storage_bucket_bits(max_value: u64) -> usize {
    if max_value <= u8::MAX as u64 {
        8
    } else if max_value <= u16::MAX as u64 {
        16
    } else if max_value <= u32::MAX as u64 {
        32
    } else {
        64
    }
}

fn signed_storage_bucket_bits(min_value: i64, max_value: i64) -> usize {
    if min_value >= i8::MIN as i64 && max_value <= i8::MAX as i64 {
        8
    } else if min_value >= i16::MIN as i64 && max_value <= i16::MAX as i64 {
        16
    } else if min_value >= i32::MIN as i64 && max_value <= i32::MAX as i64 {
        32
    } else {
        64
    }
}

fn scaled_int_range(
    dataset: &Dataset,
    column: usize,
    data_type: FeatureDataType,
    decimal_scale: u8,
) -> Option<(i64, i64)> {
    if dataset.num_rows() == 0 {
        trace!(
            "Column {} has zero rows, treating scaled range as [0,0]",
            column
        );
        return Some((0, 0));
    }

    let factor = 10f64.powi(decimal_scale as i32);
    if !matches!(data_type, FeatureDataType::F32 | FeatureDataType::F64) {
        trace!(
            "Column {} has non-float type {:?}; cannot scale",
            column, data_type
        );
        return None;
    }

    let mut min_value = i64::MAX;
    let mut max_value = i64::MIN;

    for row in 0..dataset.num_rows() {
        let value = match dataset.value_at(row, column).expect("row/column should be valid") {
            DataValue::F32(v) => v as f64,
            DataValue::F64(v) => v,
            _ => {
                trace!(
                    "Column {} row {} type mismatch while computing scaled range",
                    column, row
                );
                return None;
            }
        };
        if !value.is_finite() {
            trace!(
                "Column {} row {} is non-finite ({}), cannot apply scaling",
                column, row, value
            );
            return None;
        }

        let scaled = value * factor;
        let rounded = scaled.round();
        if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
            trace!(
                "Column {} row {} overflows i64 after scaling (value={}, scale={})",
                column, row, value, decimal_scale
            );
            return None;
        }

        let reconstructed_ok = match data_type {
            FeatureDataType::F32 => {
                let original = value as f32;
                let reconstructed = (rounded / factor) as f32;
                reconstructed.to_bits() == original.to_bits()
            }
            FeatureDataType::F64 => {
                let original = value;
                let reconstructed = rounded / factor;
                reconstructed.to_bits() == original.to_bits()
            }
            _ => false,
        };
        if !reconstructed_ok {
            trace!(
                "Column {} row {} failed lossless reconstruction at scale {}",
                column, row, decimal_scale
            );
            return None;
        }

        let int_value = rounded as i64;
        min_value = min_value.min(int_value);
        max_value = max_value.max(int_value);
    }

    trace!(
        "Column {} scale {} accepted with integer range [{}, {}]",
        column, decimal_scale, min_value, max_value
    );

    Some((min_value, max_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::{ColumnData, CsvDataLoader};

    #[test]
    fn print_bitdata_head() {
        let loader = CsvDataLoader::new(true);
        let loaded = loader
            .load("data/data-10000-8-int.csv")
            .expect("Failed to load dataset");
        let dataset = loaded.dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        tracing::info!("\nFirst 5 rows in bit representation (MSB first):");
        tracing::info!("==============================================");

        // Print the first 5 chunks with a delimiter at each feature boundary
        for row_index in 0..5 {
            tracing::info!("\nRow {}:", row_index);

            for feat_idx in 0..bit_data.info.num_features() {
                let feature_bits = bit_data.get_feature(row_index, feat_idx);
                let bits: String = feature_bits
                    .iter()
                    .map(|bit| if *bit { '1' } else { '0' })
                    .collect();
                tracing::info!("  Feature {}: {}", feat_idx, bits);
            }
        }
    }

    #[test]
    fn test_bitdata_from_dataset() {
        let loader = CsvDataLoader::new(true);
        let dataset = loader.load("data/data-10000-8-int.csv").unwrap().dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        assert_eq!(bit_data.info.num_features(), 8);
        assert_eq!(bit_data.info.feature_bits(0), 8);
        assert_eq!(bit_data.data.chunk_size, 8 * 8);
        assert_eq!(bit_data.data.num_rows, 10000);
        assert_eq!(bit_data.data.total_bits(), 10000 * 64);
    }

    #[test]
    fn test_chunk_access() {
        let loader = CsvDataLoader::new(true);
        let dataset = loader.load("data/data-10000-8-int.csv").unwrap().dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        let chunk = bit_data.data.get_chunk(0);
        assert_eq!(chunk.len(), 64); // 8 features * 8 bits

        let feature = bit_data.get_feature(0, 0);
        assert_eq!(feature.len(), 8);
    }

    #[test]
    fn test_get_bit_individual_access() {
        let loader = CsvDataLoader::new(true);
        let dataset = loader.load("data/data-10000-8-int.csv").unwrap().dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        // Verify get_bit is consistent with get_feature
        let feature_bits = bit_data.info.feature_bits(0);
        for row in 0..10 {
            for feat in 0..8 {
                let feature_slice = bit_data.get_feature(row, feat);
                for bit in 0..feature_bits {
                    assert_eq!(
                        bit_data.get_bit_by_feature(row, feat, bit),
                        feature_slice[bit],
                        "Mismatch at row={}, feat={}, bit={}",
                        row,
                        feat,
                        bit
                    );
                }
            }
        }
    }

    #[test]
    fn test_raw_access() {
        let loader = CsvDataLoader::new(true);
        let dataset = loader.load("data/data-10000-8-int.csv").unwrap().dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        let raw = bit_data.data.raw();
        assert_eq!(raw.len(), bit_data.data.total_bits());

        // Verify raw access matches get_bit
        let feature_bits = bit_data.info.feature_bits(0);
        for i in 0..100 {
            let row = i / bit_data.data.chunk_size;
            let within_chunk = i % bit_data.data.chunk_size;
            let feat = within_chunk / feature_bits;
            let bit = within_chunk % feature_bits;
            assert_eq!(raw[i], bit_data.get_bit_by_feature(row, feat, bit));
        }
    }

    #[test]
    fn test_single_bit_precision() {
        let loader = CsvDataLoader::new(true);
        let dataset = loader.load("data/data-10000-8-int.csv").unwrap().dataset;
        let bit_data =
            BitDataSet::from_dataset(&dataset).expect("Failed to create BitData from dataset");

        // With 1 bit, values should only be 0 or 1
        assert_eq!(bit_data.info.feature_bits(0), 8);
        assert_eq!(bit_data.data.chunk_size, 8 * 8);
    }

    #[test]
    fn test_schema_with_floats() {
        let columns = vec![
            ColumnData::Unsigned(vec![1, 2]),
            ColumnData::F64(vec![3.5, -1.0]),
            ColumnData::Signed(vec![-2, 5]),
            ColumnData::F64(vec![1.25, 0.5]),
        ];
        let dataset = Dataset::from_columns(columns).unwrap();
        let features = vec![
            FeatureSpec::new(FeatureDataType::UnsignedInt, 8),
            FeatureSpec::new(FeatureDataType::F64, 64),
            FeatureSpec::new(FeatureDataType::SignedInt, 8),
            FeatureSpec::new(FeatureDataType::F64, 64),
        ];

        let bit_data = BitDataSet::from_dataset_with_schema(&dataset, features).unwrap();
        assert_eq!(bit_data.info.num_features(), 4);
        assert_eq!(bit_data.data.chunk_size, 144);
        assert_eq!(bit_data.data.num_rows, 2);
    }

    #[test]
    fn test_auto_scale_float_feature_spec() {
        let columns = vec![
            ColumnData::F64(vec![1.25, 2.50, -0.75]),
            ColumnData::Unsigned(vec![1, 2, 3]),
        ];
        let dataset = Dataset::from_columns(columns).unwrap();

        let bit_data = BitDataSet::from_dataset(&dataset).unwrap();
        let spec = bit_data.info.feature_spec(0);

        assert_eq!(spec.data_type, FeatureDataType::F64);
        assert!(matches!(
            spec.transform,
            FeatureTransform::ScaledOffsetSignedInt { .. }
        ));
        assert!(spec.bits <= 64);
    }

    #[test]
    fn test_integer_columns_are_offset_and_bucketed() {
        let columns = vec![
            ColumnData::Signed(vec![-5, -1, 10]),
            ColumnData::Unsigned(vec![1000, 1001, 1005]),
        ];
        let dataset = Dataset::from_columns(columns).unwrap();
        let bit_data = BitDataSet::from_dataset(&dataset).unwrap();

        let spec0 = bit_data.info.feature_spec(0);
        assert_eq!(spec0.bits, 8);
        assert!(matches!(
            spec0.transform,
            FeatureTransform::OffsetSignedInt { min_value: -5 }
        ));

        let spec1 = bit_data.info.feature_spec(1);
        assert_eq!(spec1.bits, 8);
        assert!(matches!(
            spec1.transform,
            FeatureTransform::OffsetUnsignedInt { min_value: 1000 }
        ));
    }

    #[test]
    fn test_integer_columns_can_disable_zero_normalization() {
        let columns = vec![
            ColumnData::Signed(vec![-5, -1, 10]),
            ColumnData::Unsigned(vec![1000, 1001, 1005]),
        ];
        let dataset = Dataset::from_columns(columns).unwrap();
        let options = PreprocessOptions {
            integer_zero_normalization: false,
            ..PreprocessOptions::default()
        };

        let bit_data = BitDataSet::from_dataset_with_options(&dataset, options).unwrap();

        let spec0 = bit_data.info.feature_spec(0);
        assert_eq!(spec0.bits, 8);
        assert!(matches!(spec0.transform, FeatureTransform::None));

        let spec1 = bit_data.info.feature_spec(1);
        assert_eq!(spec1.bits, 16);
        assert!(matches!(spec1.transform, FeatureTransform::None));
    }

    #[test]
    fn test_float_columns_can_use_scaled_signed_transform() {
        let columns = vec![ColumnData::F64(vec![-1.25, 0.0, 2.50])];
        let dataset = Dataset::from_columns(columns).unwrap();
        let options = PreprocessOptions {
            float_scaling: FloatScalingMode::ScaledSignedInt,
            max_decimal_scale: 3,
            integer_zero_normalization: true,
        };

        let bit_data = BitDataSet::from_dataset_with_options(&dataset, options).unwrap();
        let spec = bit_data.info.feature_spec(0);
        assert!(matches!(
            spec.transform,
            FeatureTransform::ScaledSignedInt { .. }
        ));
    }
}
