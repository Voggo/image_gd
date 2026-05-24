use crate::compression::image_preprocessor::ImageColorSpace;
use crate::compression::preprocessor::{
    FloatScalingMode, ImageColorModel, ImageGroupingTransform, PixelGrouping, PreprocessOptions,
};
use crate::data_loader::{FloatStorage, MissingValuePolicy};
use crate::error::EntroGdError;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectBasesImpl {
    Naive,
    Optimized,
    ProfileAllBits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseBitImpl {
    Naive,
    BatchGroups,
    IncSignatureGroups,
    SignatureGroups,
    HyperLogLogCount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeImpl {
    Naive,
    FusedDictionary,
    FusedDictionarySinglePass,
    Rle,
    OffsetRle,
    HuffmanBaseIdOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyImpl {
    Naive,
    Batched,
    StrideSampled,
    StrideSampledBatched,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseTableImpl {
    Raw,
    Sorted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaCodecImpl {
    None,
    Unary,
    Fixed,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSelectBasesImpl {
    Naive,
    #[serde(alias = "optimized_v1", alias = "optimized_v2", alias = "optimized_v3")]
    Optimized,
    #[serde(alias = "profile")]
    ProfileAllBits,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigBaseBitImpl {
    Naive,
    BatchGroups,
    IncSignatureGroups,
    SignatureGroups,
    HyperLogLogCount,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigEncodeImpl {
    Naive,
    FusedDictionary,
    FusedDictionarySinglePass,
    Rle,
    OffsetRle,
    Huffman,
    HuffmanBaseIdOnly,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigEntropyImpl {
    Naive,
    #[serde(alias = "optimized")]
    Batched,
    StrideSampled,
    StrideSampledBatched,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigBaseTableImpl {
    Raw,
    Sorted,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigDeltaCodecImpl {
    None,
    Unary,
    Fixed,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigFloatStorage {
    F32,
    F64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigMissingValuePolicy {
    Error,
    Zero,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigFloatScalingMode {
    Disabled,
    ScaledSignedInt,
    ScaledOffsetSignedInt,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigImageColorSpace {
    SrgbWithLinearAlpha,
    Linear,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigImageColorModel {
    Rgb,
    YCoCg,
    YCoCgR,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigImageGroupingTransform {
    Raw,
    ForFirstPixel,
    ForMin,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CsvPreprocessConfig {
    pub float_scaling: Option<ConfigFloatScalingMode>,
    pub max_decimal_scale: Option<u8>,
    pub integer_zero_normalization: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CsvPipelineProfileConfig {
    pub name: String,
    pub has_headers: Option<bool>,
    pub float_storage: Option<ConfigFloatStorage>,
    pub missing_value_policy: Option<ConfigMissingValuePolicy>,
    pub preprocess: Option<CsvPreprocessConfig>,
    pub m_max: usize,
    pub patience: usize,
    pub entropy_threshold: Option<f64>,
    pub select_impl: ConfigSelectBasesImpl,
    pub base_bit_impl: Option<ConfigBaseBitImpl>,
    pub entropy_impl: Option<ConfigEntropyImpl>,
    pub entropy_skip_rows: Option<usize>,
    pub use_condensed_samples: Option<bool>,
    pub base_table_impl: Option<ConfigBaseTableImpl>,
    pub delta_codec_impl: Option<ConfigDeltaCodecImpl>,
    pub encode_impl: ConfigEncodeImpl,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageBuildConfigSerializable {
    pub colorspace: ConfigImageColorSpace,
    pub color_model: ConfigImageColorModel,
    pub pixel_grouping: ImageGroupingConfig,
    pub grouping_transform: ConfigImageGroupingTransform,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageGroupingConfig {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImagePipelineProfileConfig {
    pub name: String,
    pub build: ImageBuildConfigSerializable,
    pub m_max: usize,
    pub patience: usize,
    pub entropy_threshold: Option<f64>,
    pub select_impl: ConfigSelectBasesImpl,
    pub base_bit_impl: Option<ConfigBaseBitImpl>,
    pub entropy_impl: Option<ConfigEntropyImpl>,
    pub entropy_skip_rows: Option<usize>,
    pub use_condensed_samples: Option<bool>,
    pub base_table_impl: Option<ConfigBaseTableImpl>,
    pub delta_codec_impl: Option<ConfigDeltaCodecImpl>,
    pub encode_impl: ConfigEncodeImpl,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ExperimentRunnerConfigFile {
    pub input_path: Option<String>,
    pub recursive: Option<bool>,
    pub compare_png: Option<bool>,
    pub csv_profiles: Option<Vec<CsvPipelineProfileConfig>>,
    pub image_profiles: Option<Vec<ImagePipelineProfileConfig>>,
    pub csv_profile_groups: Option<Vec<CsvProfileGroupConfig>>,
    pub image_profile_groups: Option<Vec<ImageProfileGroupConfig>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntegerRangeUsize {
    pub start: usize,
    pub end: usize,
    pub step: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntegerRangeU32 {
    pub start: u32,
    pub end: u32,
    pub step: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntegerRangeU8 {
    pub start: u8,
    pub end: u8,
    pub step: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntegerSweepUsize {
    pub values: Option<Vec<usize>>,
    pub range: Option<IntegerRangeUsize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntegerSweepU32 {
    pub values: Option<Vec<u32>>,
    pub range: Option<IntegerRangeU32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntegerSweepU8 {
    pub values: Option<Vec<u8>>,
    pub range: Option<IntegerRangeU8>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CsvPreprocessSweepConfig {
    pub float_scaling: Option<Vec<ConfigFloatScalingMode>>,
    pub max_decimal_scale: Option<IntegerSweepU8>,
    pub integer_zero_normalization: Option<Vec<bool>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CsvProfileGroupConfig {
    pub name: String,
    pub has_headers: Option<Vec<bool>>,
    pub float_storage: Option<Vec<ConfigFloatStorage>>,
    pub missing_value_policy: Option<Vec<ConfigMissingValuePolicy>>,
    pub preprocess: Option<CsvPreprocessSweepConfig>,
    pub m_max: Option<IntegerSweepUsize>,
    pub patience: Option<IntegerSweepUsize>,
    pub entropy_threshold: Option<Vec<f64>>,
    pub select_impl: Option<Vec<ConfigSelectBasesImpl>>,
    pub base_bit_impl: Option<Vec<ConfigBaseBitImpl>>,
    pub entropy_impl: Option<Vec<ConfigEntropyImpl>>,
    pub entropy_skip_rows: Option<IntegerSweepUsize>,
    pub use_condensed_samples: Option<Vec<bool>>,
    pub base_table_impl: Option<Vec<ConfigBaseTableImpl>>,
    pub delta_codec_impl: Option<Vec<ConfigDeltaCodecImpl>>,
    pub encode_impl: Option<Vec<ConfigEncodeImpl>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageBuildSweepConfig {
    pub colorspace: Option<Vec<ConfigImageColorSpace>>,
    pub color_model: Option<Vec<ConfigImageColorModel>>,
    pub pixel_grouping: Option<Vec<ImageGroupingConfig>>,
    pub grouping_transform: Option<Vec<ConfigImageGroupingTransform>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageProfileGroupConfig {
    pub name: String,
    pub build: Option<ImageBuildSweepConfig>,
    pub m_max: Option<IntegerSweepUsize>,
    pub patience: Option<IntegerSweepUsize>,
    pub entropy_threshold: Option<Vec<f64>>,
    pub select_impl: Option<Vec<ConfigSelectBasesImpl>>,
    pub base_bit_impl: Option<Vec<ConfigBaseBitImpl>>,
    pub entropy_impl: Option<Vec<ConfigEntropyImpl>>,
    pub entropy_skip_rows: Option<IntegerSweepUsize>,
    pub use_condensed_samples: Option<Vec<bool>>,
    pub base_table_impl: Option<Vec<ConfigBaseTableImpl>>,
    pub delta_codec_impl: Option<Vec<ConfigDeltaCodecImpl>>,
    pub encode_impl: Option<Vec<ConfigEncodeImpl>>,
}

#[derive(Debug, Clone)]
pub struct CsvPipelineProfile {
    pub name: String,
    pub has_headers: bool,
    pub float_storage: FloatStorage,
    pub missing_value_policy: MissingValuePolicy,
    pub preprocess: PreprocessOptions,
    pub m_max: usize,
    pub patience: usize,
    pub entropy_threshold: f64,
    pub select_impl: SelectBasesImpl,
    pub base_bit_impl: BaseBitImpl,
    pub entropy_impl: EntropyImpl,
    pub entropy_skip_rows: usize,
    pub use_condensed_samples: bool,
    pub base_table_impl: BaseTableImpl,
    pub delta_codec_impl: DeltaCodecImpl,
    pub encode_impl: EncodeImpl,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageBuildConfig {
    pub colorspace: ImageColorSpace,
    pub color_model: ImageColorModel,
    pub pixel_grouping: PixelGrouping,
    pub grouping_transform: ImageGroupingTransform,
}

#[derive(Debug, Clone)]
pub struct ImagePipelineProfile {
    pub name: String,
    pub build: ImageBuildConfig,
    pub m_max: usize,
    pub patience: usize,
    pub entropy_threshold: f64,
    pub select_impl: SelectBasesImpl,
    pub base_bit_impl: BaseBitImpl,
    pub entropy_impl: EntropyImpl,
    pub entropy_skip_rows: usize,
    pub use_condensed_samples: bool,
    pub base_table_impl: BaseTableImpl,
    pub delta_codec_impl: DeltaCodecImpl,
    pub encode_impl: EncodeImpl,
}

#[derive(Debug, Clone)]
pub struct PipelineProfileSet {
    pub csv: Vec<CsvPipelineProfile>,
    pub image: Vec<ImagePipelineProfile>,
}

impl PipelineProfileSet {
    pub fn default_profiles() -> Self {
        Self {
            csv: vec![
                CsvPipelineProfile {
                    name: "csv_fast".to_string(),
                    has_headers: true,
                    float_storage: FloatStorage::F32,
                    missing_value_policy: MissingValuePolicy::Error,
                    preprocess: PreprocessOptions::default(),
                    m_max: 0,
                    patience: 6,
                    entropy_threshold: 0.70,
                    select_impl: SelectBasesImpl::Optimized,
                    base_bit_impl: BaseBitImpl::SignatureGroups,
                    entropy_impl: EntropyImpl::Naive,
                    entropy_skip_rows: 0,
                    use_condensed_samples: true,
                    base_table_impl: BaseTableImpl::Raw,
                    delta_codec_impl: DeltaCodecImpl::None,
                    encode_impl: EncodeImpl::Naive,
                },
                CsvPipelineProfile {
                    name: "csv_balanced".to_string(),
                    has_headers: true,
                    float_storage: FloatStorage::F32,
                    missing_value_policy: MissingValuePolicy::Error,
                    preprocess: PreprocessOptions::default(),
                    m_max: 25,
                    patience: 10,
                    entropy_threshold: 0.70,
                    select_impl: SelectBasesImpl::Optimized,
                    base_bit_impl: BaseBitImpl::IncSignatureGroups,
                    entropy_impl: EntropyImpl::Naive,
                    entropy_skip_rows: 0,
                    use_condensed_samples: true,
                    base_table_impl: BaseTableImpl::Raw,
                    delta_codec_impl: DeltaCodecImpl::None,
                    encode_impl: EncodeImpl::Rle,
                },
                CsvPipelineProfile {
                    name: "csv_ratio".to_string(),
                    has_headers: true,
                    float_storage: FloatStorage::F32,
                    missing_value_policy: MissingValuePolicy::Error,
                    preprocess: PreprocessOptions::default(),
                    m_max: 75,
                    patience: 20,
                    entropy_threshold: 0.70,
                    select_impl: SelectBasesImpl::Optimized,
                    base_bit_impl: BaseBitImpl::BatchGroups,
                    entropy_impl: EntropyImpl::Naive,
                    entropy_skip_rows: 0,
                    use_condensed_samples: true,
                    base_table_impl: BaseTableImpl::Raw,
                    delta_codec_impl: DeltaCodecImpl::None,
                    encode_impl: EncodeImpl::Rle,
                },
            ],
            image: vec![
                ImagePipelineProfile {
                    name: "img_fast".to_string(),
                    build: ImageBuildConfig {
                        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
                        color_model: ImageColorModel::Rgb,
                        pixel_grouping: PixelGrouping::new(1, 1),
                        grouping_transform: ImageGroupingTransform::Raw,
                    },
                    m_max: 0,
                    patience: 6,
                    entropy_threshold: 0.70,
                    select_impl: SelectBasesImpl::Optimized,
                    base_bit_impl: BaseBitImpl::SignatureGroups,
                    entropy_impl: EntropyImpl::Naive,
                    entropy_skip_rows: 0,
                    use_condensed_samples: true,
                    base_table_impl: BaseTableImpl::Raw,
                    delta_codec_impl: DeltaCodecImpl::None,
                    encode_impl: EncodeImpl::Naive,
                },
                ImagePipelineProfile {
                    name: "img_balanced".to_string(),
                    build: ImageBuildConfig {
                        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
                        color_model: ImageColorModel::YCoCgR,
                        pixel_grouping: PixelGrouping::new(1, 1),
                        grouping_transform: ImageGroupingTransform::ForFirstPixel,
                    },
                    m_max: 25,
                    patience: 10,
                    entropy_threshold: 0.70,
                    select_impl: SelectBasesImpl::Optimized,
                    base_bit_impl: BaseBitImpl::IncSignatureGroups,
                    entropy_impl: EntropyImpl::Naive,
                    entropy_skip_rows: 0,
                    use_condensed_samples: true,
                    base_table_impl: BaseTableImpl::Raw,
                    delta_codec_impl: DeltaCodecImpl::None,
                    encode_impl: EncodeImpl::Rle,
                },
                ImagePipelineProfile {
                    name: "img_ratio".to_string(),
                    build: ImageBuildConfig {
                        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
                        color_model: ImageColorModel::YCoCgR,
                        pixel_grouping: PixelGrouping::new(2, 1),
                        grouping_transform: ImageGroupingTransform::ForMin,
                    },
                    m_max: 75,
                    patience: 20,
                    entropy_threshold: 0.70,
                    select_impl: SelectBasesImpl::Optimized,
                    base_bit_impl: BaseBitImpl::BatchGroups,
                    entropy_impl: EntropyImpl::Naive,
                    entropy_skip_rows: 0,
                    use_condensed_samples: true,
                    base_table_impl: BaseTableImpl::Raw,
                    delta_codec_impl: DeltaCodecImpl::None,
                    encode_impl: EncodeImpl::HuffmanBaseIdOnly,
                },
            ],
        }
    }

    pub fn from_config_file(
        path: &Path,
    ) -> Result<(Self, ExperimentRunnerConfigFile), EntroGdError> {
        let raw = fs::read_to_string(path)?;
        let config: ExperimentRunnerConfigFile = match path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref()
        {
            Some("json") => {
                serde_json::from_str(&raw).map_err(|err| EntroGdError::InvalidMetadata {
                    message: format!("failed to parse JSON config '{}': {}", path.display(), err),
                })?
            }
            _ => {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "unsupported config extension for '{}'; use .json",
                        path.display()
                    ),
                });
            }
        };

        let mut profiles = Self::default_profiles();
        let has_csv_config = config.csv_profiles.is_some() || config.csv_profile_groups.is_some();
        let has_image_config =
            config.image_profiles.is_some() || config.image_profile_groups.is_some();

        if has_csv_config {
            profiles.csv.clear();
        }
        if has_image_config {
            profiles.image.clear();
        }

        if let Some(csv_profiles) = &config.csv_profiles {
            let converted: Vec<CsvPipelineProfile> = csv_profiles
                .iter()
                .cloned()
                .map(CsvPipelineProfile::try_from)
                .collect::<Result<_, EntroGdError>>()?;
            profiles.csv.extend(converted);
        }
        if let Some(image_profiles) = &config.image_profiles {
            let converted: Vec<ImagePipelineProfile> = image_profiles
                .iter()
                .cloned()
                .map(ImagePipelineProfile::try_from)
                .collect::<Result<_, EntroGdError>>()?;
            profiles.image.extend(converted);
        }

        if let Some(csv_groups) = &config.csv_profile_groups {
            for group in csv_groups {
                profiles.csv.extend(expand_csv_group(group)?);
            }
        }

        if let Some(image_groups) = &config.image_profile_groups {
            for group in image_groups {
                profiles.image.extend(expand_image_group(group)?);
            }
        }

        if profiles.csv.is_empty() && profiles.image.is_empty() {
            return Err(EntroGdError::InvalidMetadata {
                message: "config produced no profiles".to_string(),
            });
        }

        Ok((profiles, config))
    }
}

fn expand_csv_group(
    group: &CsvProfileGroupConfig,
) -> Result<Vec<CsvPipelineProfile>, EntroGdError> {
    let has_headers = group.has_headers.clone().unwrap_or_else(|| vec![true]);
    let float_storage = group
        .float_storage
        .clone()
        .unwrap_or_else(|| vec![ConfigFloatStorage::F32]);
    let missing_value_policy = group
        .missing_value_policy
        .clone()
        .unwrap_or_else(|| vec![ConfigMissingValuePolicy::Error]);
    let m_max = expand_usize_sweep(group.m_max.as_ref(), 0)?;
    let patience = expand_usize_sweep(group.patience.as_ref(), 10)?;
    let select_impl = group
        .select_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigSelectBasesImpl::Optimized]);
    let base_bit_impl = group
        .base_bit_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigBaseBitImpl::BatchGroups]);
    let encode_impl = group
        .encode_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigEncodeImpl::Naive]);
    let entropy_impl = group
        .entropy_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigEntropyImpl::Naive]);
    let entropy_skip_rows = expand_usize_sweep(group.entropy_skip_rows.as_ref(), 0)?;
    let entropy_threshold = group
        .entropy_threshold
        .clone()
        .unwrap_or_else(|| vec![0.70]);
    let use_condensed_samples = group
        .use_condensed_samples
        .clone()
        .unwrap_or_else(|| vec![true]);
    let base_table_impl = group
        .base_table_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigBaseTableImpl::Raw]);
    let delta_codec_impl = group
        .delta_codec_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigDeltaCodecImpl::None]);

    if encode_impl
        .iter()
        .any(|encoding| matches!(encoding, ConfigEncodeImpl::Huffman))
    {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "CSV group '{}' uses legacy 'huffman'; use 'huffman_base_id_only'",
                group.name
            ),
        });
    }

    let preprocess = group
        .preprocess
        .clone()
        .unwrap_or(CsvPreprocessSweepConfig {
            float_scaling: None,
            max_decimal_scale: None,
            integer_zero_normalization: None,
        });
    let float_scaling = preprocess
        .float_scaling
        .unwrap_or_else(|| vec![ConfigFloatScalingMode::ScaledOffsetSignedInt]);
    let max_decimal_scale = expand_u8_sweep(preprocess.max_decimal_scale.as_ref(), 9)?;
    let integer_zero_normalization = preprocess
        .integer_zero_normalization
        .unwrap_or_else(|| vec![true]);

    let mut profiles = Vec::new();
    let axis_lengths = [
        has_headers.len(),
        float_storage.len(),
        missing_value_policy.len(),
        float_scaling.len(),
        max_decimal_scale.len(),
        integer_zero_normalization.len(),
        m_max.len(),
        patience.len(),
        entropy_threshold.len(),
        select_impl.len(),
        base_bit_impl.len(),
        entropy_impl.len(),
        entropy_skip_rows.len(),
        use_condensed_samples.len(),
        base_table_impl.len(),
        delta_codec_impl.len(),
        encode_impl.len(),
    ];

    for_each_combination(&axis_lengths, |indices| {
        let has_headers_item = has_headers[indices[0]];
        let float_storage_item = &float_storage[indices[1]];
        let missing_item = &missing_value_policy[indices[2]];
        let float_scaling_item = &float_scaling[indices[3]];
        let max_decimal_scale_item = max_decimal_scale[indices[4]];
        let int_zero_item = integer_zero_normalization[indices[5]];
        let m_max_item = m_max[indices[6]];
        let patience_item = patience[indices[7]];
        let entropy_threshold_item = entropy_threshold[indices[8]];
        let select_item = &select_impl[indices[9]];
        let base_bit_item = &base_bit_impl[indices[10]];
        let entropy_item = &entropy_impl[indices[11]];
        let entropy_skip_rows_item = entropy_skip_rows[indices[12]];
        let use_condensed_item = use_condensed_samples[indices[13]];
        let base_table_item = &base_table_impl[indices[14]];
        let delta_codec_item = &delta_codec_impl[indices[15]];
        let encode_item = &encode_impl[indices[16]];
        let run_idx = profiles.len();

        let name = format!(
            "{}__{:03}_m{}_p{}_th{:.3}_sel{:?}_bb{:?}_ent{:?}_sk{}_cond{}_tbl{:?}_dc{:?}_enc{:?}",
            group.name,
            run_idx,
            m_max_item,
            patience_item,
            entropy_threshold_item,
            select_item,
            base_bit_item,
            entropy_item,
            entropy_skip_rows_item,
            use_condensed_item,
            base_table_item,
            delta_codec_item,
            encode_item
        );

        profiles.push(CsvPipelineProfile {
            name,
            has_headers: has_headers_item,
            float_storage: float_storage_item.clone().into(),
            missing_value_policy: missing_item.clone().into(),
            preprocess: PreprocessOptions {
                float_scaling: float_scaling_item.clone().into(),
                max_decimal_scale: max_decimal_scale_item,
                integer_zero_normalization: int_zero_item,
            },
            m_max: m_max_item,
            patience: patience_item,
            entropy_threshold: entropy_threshold_item,
            select_impl: select_item.clone().into(),
            base_bit_impl: base_bit_item.clone().into(),
            entropy_impl: entropy_item.clone().into(),
            entropy_skip_rows: entropy_skip_rows_item,
            use_condensed_samples: use_condensed_item,
            base_table_impl: base_table_item.clone().into(),
            delta_codec_impl: delta_codec_item.clone().into(),
            encode_impl: convert_encode_impl(encode_item.clone())?,
        });

        Ok::<(), EntroGdError>(())
    })?;

    if profiles.is_empty() {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("CSV group '{}' produced no profiles", group.name),
        });
    }

    Ok(profiles)
}

fn expand_image_group(
    group: &ImageProfileGroupConfig,
) -> Result<Vec<ImagePipelineProfile>, EntroGdError> {
    let build = group.build.clone().unwrap_or(ImageBuildSweepConfig {
        colorspace: None,
        color_model: None,
        pixel_grouping: None,
        grouping_transform: None,
    });

    let colorspace = build
        .colorspace
        .unwrap_or_else(|| vec![ConfigImageColorSpace::SrgbWithLinearAlpha]);
    let color_model = build
        .color_model
        .unwrap_or_else(|| vec![ConfigImageColorModel::Rgb]);
    let pixel_grouping = build.pixel_grouping.unwrap_or_else(|| {
        vec![ImageGroupingConfig {
            width: 1,
            height: 1,
        }]
    });
    let grouping_transform = build
        .grouping_transform
        .unwrap_or_else(|| vec![ConfigImageGroupingTransform::Raw]);

    let m_max = expand_usize_sweep(group.m_max.as_ref(), 0)?;
    let patience = expand_usize_sweep(group.patience.as_ref(), 10)?;
    let entropy_threshold = group
        .entropy_threshold
        .clone()
        .unwrap_or_else(|| vec![0.70]);
    let select_impl = group
        .select_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigSelectBasesImpl::Optimized]);
    let base_bit_impl = group
        .base_bit_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigBaseBitImpl::BatchGroups]);
    let encode_impl = group
        .encode_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigEncodeImpl::Naive]);
    let entropy_impl = group
        .entropy_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigEntropyImpl::Naive]);
    let entropy_skip_rows = expand_usize_sweep(group.entropy_skip_rows.as_ref(), 0)?;
    let use_condensed_samples = group
        .use_condensed_samples
        .clone()
        .unwrap_or_else(|| vec![true]);
    let base_table_impl = group
        .base_table_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigBaseTableImpl::Raw]);
    let delta_codec_impl = group
        .delta_codec_impl
        .clone()
        .unwrap_or_else(|| vec![ConfigDeltaCodecImpl::None]);

    if encode_impl
        .iter()
        .any(|encoding| matches!(encoding, ConfigEncodeImpl::Huffman))
    {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "image group '{}' uses legacy 'huffman'; use 'huffman_base_id_only'",
                group.name
            ),
        });
    }

    if pixel_grouping
        .iter()
        .any(|grouping| grouping.width == 0 || grouping.height == 0)
    {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "image group '{}' has invalid pixel_grouping dimensions",
                group.name
            ),
        });
    }

    let mut profiles = Vec::new();
    let axis_lengths = [
        colorspace.len(),
        color_model.len(),
        pixel_grouping.len(),
        grouping_transform.len(),
        m_max.len(),
        patience.len(),
        entropy_threshold.len(),
        select_impl.len(),
        base_bit_impl.len(),
        entropy_impl.len(),
        entropy_skip_rows.len(),
        use_condensed_samples.len(),
        base_table_impl.len(),
        delta_codec_impl.len(),
        encode_impl.len(),
    ];

    for_each_combination(&axis_lengths, |indices| {
        let colorspace_item = &colorspace[indices[0]];
        let color_model_item = &color_model[indices[1]];
        let pixel_grouping_item = &pixel_grouping[indices[2]];
        let grouping_transform_item = &grouping_transform[indices[3]];
        let m_max_item = m_max[indices[4]];
        let patience_item = patience[indices[5]];
        let entropy_threshold_item = entropy_threshold[indices[6]];
        let select_item = &select_impl[indices[7]];
        let base_bit_item = &base_bit_impl[indices[8]];
        let entropy_item = &entropy_impl[indices[9]];
        let entropy_skip_rows_item = entropy_skip_rows[indices[10]];
        let use_condensed_item = use_condensed_samples[indices[11]];
        let base_table_item = &base_table_impl[indices[12]];
        let delta_codec_item = &delta_codec_impl[indices[13]];
        let encode_item = &encode_impl[indices[14]];
        let run_idx = profiles.len();

        let name = format!(
            "{}__{:03}_cm{:?}_pg{}x{}_gt{:?}_th{:.3}_sel{:?}_bb{:?}_ent{:?}_sk{}_cond{}_tbl{:?}_dc{:?}_enc{:?}",
            group.name,
            run_idx,
            color_model_item,
            pixel_grouping_item.width,
            pixel_grouping_item.height,
            grouping_transform_item,
            entropy_threshold_item,
            select_item,
            base_bit_item,
            entropy_item,
            entropy_skip_rows_item,
            use_condensed_item,
            base_table_item,
            delta_codec_item,
            encode_item
        );

        profiles.push(ImagePipelineProfile {
            name,
            build: ImageBuildConfig {
                colorspace: colorspace_item.clone().into(),
                color_model: color_model_item.clone().into(),
                pixel_grouping: PixelGrouping::new(
                    pixel_grouping_item.width,
                    pixel_grouping_item.height,
                ),
                grouping_transform: grouping_transform_item.clone().into(),
            },
            m_max: m_max_item,
            patience: patience_item,
            entropy_threshold: entropy_threshold_item,
            select_impl: select_item.clone().into(),
            base_bit_impl: base_bit_item.clone().into(),
            entropy_impl: entropy_item.clone().into(),
            entropy_skip_rows: entropy_skip_rows_item,
            use_condensed_samples: use_condensed_item,
            base_table_impl: base_table_item.clone().into(),
            delta_codec_impl: delta_codec_item.clone().into(),
            encode_impl: convert_encode_impl(encode_item.clone())?,
        });

        Ok::<(), EntroGdError>(())
    })?;

    if profiles.is_empty() {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("image group '{}' produced no profiles", group.name),
        });
    }

    Ok(profiles)
}

fn for_each_combination<E>(
    axis_lengths: &[usize],
    mut callback: impl FnMut(&[usize]) -> Result<(), E>,
) -> Result<(), E> {
    if axis_lengths.contains(&0) {
        return Ok(());
    }

    let mut index = vec![0usize; axis_lengths.len()];

    loop {
        callback(&index)?;

        let mut carry_position = axis_lengths.len();
        while carry_position > 0 {
            carry_position -= 1;
            index[carry_position] += 1;
            if index[carry_position] < axis_lengths[carry_position] {
                break;
            }
            index[carry_position] = 0;
            if carry_position == 0 {
                return Ok(());
            }
        }
    }
}

macro_rules! impl_expand_numeric_sweep {
    ($sweep_ty:ty, $range_ty:ty, $value_ty:ty, $expand_sweep_fn:ident, $expand_range_fn:ident) => {
        fn $expand_sweep_fn(
            sweep: Option<&$sweep_ty>,
            default_value: $value_ty,
        ) -> Result<Vec<$value_ty>, EntroGdError> {
            let Some(sweep) = sweep else {
                return Ok(vec![default_value]);
            };

            match (&sweep.values, &sweep.range) {
                (Some(values), None) => {
                    if values.is_empty() {
                        Err(EntroGdError::InvalidMetadata {
                            message: "empty integer values list is not allowed".to_string(),
                        })
                    } else {
                        Ok(values.clone())
                    }
                }
                (None, Some(range)) => $expand_range_fn(range),
                (Some(_), Some(_)) => Err(EntroGdError::InvalidMetadata {
                    message: "integer sweep cannot contain both 'values' and 'range'".to_string(),
                }),
                (None, None) => Ok(vec![default_value]),
            }
        }

        fn $expand_range_fn(range: &$range_ty) -> Result<Vec<$value_ty>, EntroGdError> {
            if range.step == 0 {
                return Err(EntroGdError::InvalidMetadata {
                    message: "integer range step must be > 0".to_string(),
                });
            }
            if range.end < range.start {
                return Err(EntroGdError::InvalidMetadata {
                    message: "integer range end must be >= start".to_string(),
                });
            }

            let mut values = Vec::new();
            let mut current = range.start;
            while current <= range.end {
                values.push(current);
                current = current.saturating_add(range.step);
                if current == <$value_ty>::MAX {
                    break;
                }
            }

            if values.is_empty() {
                return Err(EntroGdError::InvalidMetadata {
                    message: "integer range expansion produced no values".to_string(),
                });
            }

            Ok(values)
        }
    };
}

impl_expand_numeric_sweep!(
    IntegerSweepUsize,
    IntegerRangeUsize,
    usize,
    expand_usize_sweep,
    expand_usize_range
);
impl_expand_numeric_sweep!(
    IntegerSweepU8,
    IntegerRangeU8,
    u8,
    expand_u8_sweep,
    expand_u8_range
);

macro_rules! impl_from_enum {
    ($src:ty => $dst:ty { $($src_variant:ident => $dst_variant:ident),+ $(,)? }) => {
        impl From<$src> for $dst {
            fn from(value: $src) -> Self {
                match value {
                    $(<$src>::$src_variant => <$dst>::$dst_variant,)+
                }
            }
        }
    };
}

impl_from_enum!(ConfigSelectBasesImpl => SelectBasesImpl {
    Naive => Naive,
    Optimized => Optimized,
    ProfileAllBits => ProfileAllBits,
});

impl_from_enum!(ConfigBaseBitImpl => BaseBitImpl {
    Naive => Naive,
    BatchGroups => BatchGroups,
    IncSignatureGroups => IncSignatureGroups,
    SignatureGroups => SignatureGroups,
    HyperLogLogCount => HyperLogLogCount,
});

impl_from_enum!(ConfigEntropyImpl => EntropyImpl {
    Naive => Naive,
    Batched => Batched,
    StrideSampled => StrideSampled,
    StrideSampledBatched => StrideSampledBatched,
});

impl_from_enum!(ConfigBaseTableImpl => BaseTableImpl {
    Raw => Raw,
    Sorted => Sorted,
});

impl_from_enum!(ConfigFloatStorage => FloatStorage {
    F32 => F32,
    F64 => F64,
});

impl_from_enum!(ConfigMissingValuePolicy => MissingValuePolicy {
    Error => Error,
    Zero => Zero,
});

impl_from_enum!(ConfigFloatScalingMode => FloatScalingMode {
    Disabled => Disabled,
    ScaledSignedInt => ScaledSignedInt,
    ScaledOffsetSignedInt => ScaledOffsetSignedInt,
});

impl_from_enum!(ConfigImageColorSpace => ImageColorSpace {
    SrgbWithLinearAlpha => SrgbWithLinearAlpha,
    Linear => Linear,
});

impl_from_enum!(ConfigImageColorModel => ImageColorModel {
    Rgb => Rgb,
    YCoCg => YCoCg,
    YCoCgR => YCoCgR,
});

impl_from_enum!(ConfigImageGroupingTransform => ImageGroupingTransform {
    Raw => Raw,
    ForFirstPixel => ForFirstPixel,
    ForMin => ForMin,
});

impl_from_enum!(ConfigDeltaCodecImpl => DeltaCodecImpl {
    None => None,
    Unary => Unary,
    Fixed => Fixed,
});

impl TryFrom<CsvPipelineProfileConfig> for CsvPipelineProfile {
    type Error = EntroGdError;

    fn try_from(value: CsvPipelineProfileConfig) -> Result<Self, Self::Error> {
        if matches!(value.encode_impl, ConfigEncodeImpl::Huffman) {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "CSV profile '{}' uses legacy 'huffman'; use 'huffman_base_id_only'",
                    value.name
                ),
            });
        }

        let mut preprocess = PreprocessOptions::default();
        if let Some(config) = value.preprocess {
            if let Some(float_scaling) = config.float_scaling {
                preprocess.float_scaling = float_scaling.into();
            }
            if let Some(max_decimal_scale) = config.max_decimal_scale {
                preprocess.max_decimal_scale = max_decimal_scale;
            }
            if let Some(integer_zero_normalization) = config.integer_zero_normalization {
                preprocess.integer_zero_normalization = integer_zero_normalization;
            }
        }

        Ok(Self {
            name: value.name,
            has_headers: value.has_headers.unwrap_or(true),
            float_storage: value
                .float_storage
                .unwrap_or(ConfigFloatStorage::F32)
                .into(),
            missing_value_policy: value
                .missing_value_policy
                .unwrap_or(ConfigMissingValuePolicy::Error)
                .into(),
            preprocess,
            m_max: value.m_max,
            patience: value.patience,
            entropy_threshold: value.entropy_threshold.unwrap_or(0.70),
            select_impl: value.select_impl.into(),
            base_bit_impl: value
                .base_bit_impl
                .unwrap_or(ConfigBaseBitImpl::BatchGroups)
                .into(),
            entropy_impl: value
                .entropy_impl
                .unwrap_or(ConfigEntropyImpl::Naive)
                .into(),
            entropy_skip_rows: value.entropy_skip_rows.unwrap_or(0),
            use_condensed_samples: value.use_condensed_samples.unwrap_or(true),
            base_table_impl: value
                .base_table_impl
                .unwrap_or(ConfigBaseTableImpl::Raw)
                .into(),
            delta_codec_impl: value
                .delta_codec_impl
                .unwrap_or(ConfigDeltaCodecImpl::None)
                .into(),
            encode_impl: convert_encode_impl(value.encode_impl)?,
        })
    }
}

impl TryFrom<ImagePipelineProfileConfig> for ImagePipelineProfile {
    type Error = EntroGdError;

    fn try_from(value: ImagePipelineProfileConfig) -> Result<Self, Self::Error> {
        if matches!(value.encode_impl, ConfigEncodeImpl::Huffman) {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "image profile '{}' uses legacy 'huffman'; use 'huffman_base_id_only'",
                    value.name
                ),
            });
        }

        if value.build.pixel_grouping.width == 0 || value.build.pixel_grouping.height == 0 {
            return Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "image profile '{}' has invalid pixel_grouping dimensions",
                    value.name
                ),
            });
        }

        Ok(Self {
            name: value.name,
            build: ImageBuildConfig {
                colorspace: value.build.colorspace.into(),
                color_model: value.build.color_model.into(),
                pixel_grouping: PixelGrouping::new(
                    value.build.pixel_grouping.width,
                    value.build.pixel_grouping.height,
                ),
                grouping_transform: value.build.grouping_transform.into(),
            },
            m_max: value.m_max,
            patience: value.patience,
            entropy_threshold: value.entropy_threshold.unwrap_or(0.70),
            select_impl: value.select_impl.into(),
            base_bit_impl: value
                .base_bit_impl
                .unwrap_or(ConfigBaseBitImpl::BatchGroups)
                .into(),
            entropy_impl: value
                .entropy_impl
                .unwrap_or(ConfigEntropyImpl::Naive)
                .into(),
            entropy_skip_rows: value.entropy_skip_rows.unwrap_or(0),
            use_condensed_samples: value.use_condensed_samples.unwrap_or(true),
            base_table_impl: value
                .base_table_impl
                .unwrap_or(ConfigBaseTableImpl::Raw)
                .into(),
            delta_codec_impl: value
                .delta_codec_impl
                .unwrap_or(ConfigDeltaCodecImpl::None)
                .into(),
            encode_impl: convert_encode_impl(value.encode_impl)?,
        })
    }
}

fn convert_encode_impl(value: ConfigEncodeImpl) -> Result<EncodeImpl, EntroGdError> {
    match value {
        ConfigEncodeImpl::Naive => Ok(EncodeImpl::Naive),
        ConfigEncodeImpl::FusedDictionary => Ok(EncodeImpl::FusedDictionary),
        ConfigEncodeImpl::FusedDictionarySinglePass => Ok(EncodeImpl::FusedDictionarySinglePass),
        ConfigEncodeImpl::Rle => Ok(EncodeImpl::Rle),
        ConfigEncodeImpl::OffsetRle => Ok(EncodeImpl::OffsetRle),
        ConfigEncodeImpl::Huffman => Err(EntroGdError::InvalidMetadata {
            message: "legacy 'huffman' is no longer supported; use 'huffman_base_id_only'"
                .to_string(),
        }),
        ConfigEncodeImpl::HuffmanBaseIdOnly => Ok(EncodeImpl::HuffmanBaseIdOnly),
    }
}
