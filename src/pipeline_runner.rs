use crate::BitDataSet;
use crate::compression::base_bits::BaseBit;
use crate::compression::compress::{CompressedData, EncodedData};
use crate::compression::image_preprocessor::BuildImageBitDataSet;
use crate::compression::tabular_preprocessor::{BuildBitDataSet, InferFeatureSpecs};
use crate::data_loader::{CsvDataLoader, DataLoader};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::pipeline_profiles::{
    BaseBitImpl, CsvPipelineProfile, EncodeImpl, ImagePipelineProfile, PipelineProfileSet,
    SelectBasesImpl,
};
use crate::prelude::*;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Csv,
    Image,
    Unsupported,
}

#[derive(Debug, Clone, Copy)]
pub struct ExperimentRunOptions {
    pub recursive: bool,
}

impl Default for ExperimentRunOptions {
    fn default() -> Self {
        Self { recursive: false }
    }
}

#[derive(Debug, Clone)]
pub struct CompressionStageDurationsMs {
    pub load_input: f64,
    pub preprocess: f64,
    pub entropy: f64,
    pub condensed_samples: f64,
    pub select_bases: f64,
    pub encode: f64,
    pub total: f64,
}

#[derive(Debug, Clone)]
pub struct CompressedSizeBreakdownBits {
    pub encoded_stream_total: usize,
    pub encoded_payload_bits: usize,
    pub normal_symbol_stream_bits: usize,
    pub rle_symbol_stream_bits: usize,
    pub rle_control_stream_bits: usize,
    pub rle_packet_count: usize,
    pub huffman_pixel_stream_bits: usize,
    pub huffman_row_offsets_bits: usize,
    pub huffman_symbol_table_bits: usize,
    pub huffman_code_lengths_bits: usize,
    pub base_table_patterns: usize,
    pub base_bit_positions: usize,
    pub condensed_weights: usize,
    pub estimated_total: usize,
}

#[derive(Debug, Clone)]
pub struct ExperimentRecord {
    pub file_path: PathBuf,
    pub kind: InputKind,
    pub preset_name: String,
    pub config: ExperimentConfigColumns,
    pub select_impl: SelectBasesImpl,
    pub base_bit_impl: BaseBitImpl,
    pub encode_impl: EncodeImpl,
    pub m_max: usize,
    pub patience: usize,
    pub stage_ms: CompressionStageDurationsMs,
    pub original_bits: usize,
    pub size_breakdown_bits: CompressedSizeBreakdownBits,
}

#[derive(Debug, Clone, Default)]
pub struct ExperimentConfigColumns {
    pub csv_has_headers: Option<bool>,
    pub csv_float_storage: Option<String>,
    pub csv_missing_value_policy: Option<String>,
    pub csv_float_scaling: Option<String>,
    pub csv_max_decimal_scale: Option<u8>,
    pub csv_integer_zero_normalization: Option<bool>,
    pub image_colorspace: Option<String>,
    pub image_color_model: Option<String>,
    pub image_pixel_grouping: Option<u32>,
    pub image_grouping_transform: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ExperimentReport {
    pub discovered_files: usize,
    pub processed_files: usize,
    pub skipped_files: Vec<PathBuf>,
    pub records: Vec<ExperimentRecord>,
}

pub fn detect_input_kind(path: &Path) -> InputKind {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return InputKind::Unsupported;
    };

    match ext.to_ascii_lowercase().as_str() {
        "csv" => InputKind::Csv,
        "png" | "jpg" | "jpeg" | "bmp" | "gif" => InputKind::Image,
        _ => InputKind::Unsupported,
    }
}

pub fn collect_input_files(path: &Path, recursive: bool) -> Result<Vec<PathBuf>, EntroGdError> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("input path does not exist: {}", path.display()),
        });
    }

    let mut files = Vec::new();
    collect_dir(path, recursive, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_dir(path: &Path, recursive: bool, files: &mut Vec<PathBuf>) -> Result<(), EntroGdError> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if entry_path.is_file() {
            files.push(entry_path);
        } else if recursive && entry_path.is_dir() {
            collect_dir(&entry_path, recursive, files)?;
        }
    }
    Ok(())
}

pub fn run_experiments_on_path(
    path: &Path,
    profiles: &PipelineProfileSet,
    options: ExperimentRunOptions,
) -> Result<ExperimentReport, EntroGdError> {
    let files = collect_input_files(path, options.recursive)?;
    let mut records = Vec::new();
    let mut skipped_files = Vec::new();

    for file in &files {
        let kind = detect_input_kind(file);
        match kind {
            InputKind::Csv => {
                let mut file_records = Vec::new();
                for profile in &profiles.csv {
                    let record = run_csv_profile(file, profile)?;
                    file_records.push(record);
                }
                records.extend(file_records);
            }
            InputKind::Image => {
                let mut file_records = Vec::new();
                for profile in &profiles.image {
                    let record = run_image_profile(file, profile)?;
                    file_records.push(record);
                }
                records.extend(file_records);
            }
            InputKind::Unsupported => skipped_files.push(file.clone()),
        }
    }

    Ok(ExperimentReport {
        discovered_files: files.len(),
        processed_files: records
            .iter()
            .map(|record| record.file_path.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        skipped_files,
        records,
    })
}

pub fn write_report_csv(path: &Path, records: &[ExperimentRecord]) -> Result<(), EntroGdError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "file_path,input_kind,preset,select_impl,base_bit_impl,encode_impl,m_max,patience,csv_has_headers,csv_float_storage,csv_missing_value_policy,csv_float_scaling,csv_max_decimal_scale,csv_integer_zero_normalization,image_colorspace,image_color_model,image_pixel_grouping,image_grouping_transform,original_bits,load_ms,preprocess_ms,entropy_ms,condensed_ms,select_ms,encode_ms,total_ms,encoded_stream_total_bits,encoded_payload_bits,normal_symbol_stream_bits,rle_symbol_stream_bits,rle_control_stream_bits,rle_packet_count,huffman_pixel_stream_bits,huffman_row_offsets_bits,huffman_symbol_table_bits,huffman_code_lengths_bits,base_table_pattern_bits,base_bit_positions_bits,condensed_weights_bits,estimated_total_bits"
    )?;

    for record in records {
        writeln!(
            file,
            "{},{:?},{},{:?},{:?},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            record.file_path.display(),
            record.kind,
            record.preset_name,
            record.select_impl,
            record.base_bit_impl,
            record.encode_impl,
            record.m_max,
            record.patience,
            record
                .config
                .csv_has_headers
                .map(|v| v.to_string())
                .unwrap_or_default(),
            record.config.csv_float_storage.as_deref().unwrap_or(""),
            record
                .config
                .csv_missing_value_policy
                .as_deref()
                .unwrap_or(""),
            record.config.csv_float_scaling.as_deref().unwrap_or(""),
            record
                .config
                .csv_max_decimal_scale
                .map(|v| v.to_string())
                .unwrap_or_default(),
            record
                .config
                .csv_integer_zero_normalization
                .map(|v| v.to_string())
                .unwrap_or_default(),
            record.config.image_colorspace.as_deref().unwrap_or(""),
            record.config.image_color_model.as_deref().unwrap_or(""),
            record
                .config
                .image_pixel_grouping
                .map(|v| v.to_string())
                .unwrap_or_default(),
            record
                .config
                .image_grouping_transform
                .as_deref()
                .unwrap_or(""),
            record.original_bits,
            record.stage_ms.load_input,
            record.stage_ms.preprocess,
            record.stage_ms.entropy,
            record.stage_ms.condensed_samples,
            record.stage_ms.select_bases,
            record.stage_ms.encode,
            record.stage_ms.total,
            record.size_breakdown_bits.encoded_stream_total,
            record.size_breakdown_bits.encoded_payload_bits,
            record.size_breakdown_bits.normal_symbol_stream_bits,
            record.size_breakdown_bits.rle_symbol_stream_bits,
            record.size_breakdown_bits.rle_control_stream_bits,
            record.size_breakdown_bits.rle_packet_count,
            record.size_breakdown_bits.huffman_pixel_stream_bits,
            record.size_breakdown_bits.huffman_row_offsets_bits,
            record.size_breakdown_bits.huffman_symbol_table_bits,
            record.size_breakdown_bits.huffman_code_lengths_bits,
            record.size_breakdown_bits.base_table_patterns,
            record.size_breakdown_bits.base_bit_positions,
            record.size_breakdown_bits.condensed_weights,
            record.size_breakdown_bits.estimated_total,
        )?;
    }

    Ok(())
}

fn run_csv_profile(
    file: &Path,
    profile: &CsvPipelineProfile,
) -> Result<ExperimentRecord, EntroGdError> {
    let total_t0 = Instant::now();

    let load_t0 = Instant::now();
    let loader = CsvDataLoader::new(profile.has_headers)
        .with_float_storage(profile.float_storage)
        .with_missing_value_policy(profile.missing_value_policy);
    let loaded = loader.load(file)?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1_000.0;

    let preprocess_t0 = Instant::now();
    let (dataset, features) = InferFeatureSpecs {
        options: profile.preprocess,
    }
    .process(loaded.dataset)?;
    let bit_data = BuildBitDataSet.process((dataset, features))?;
    let preprocess_ms = preprocess_t0.elapsed().as_secs_f64() * 1_000.0;

    let entropy_t0 = Instant::now();
    let entropy_out = EntropyOptimized {}.process(bit_data)?;
    let entropy_ms = entropy_t0.elapsed().as_secs_f64() * 1_000.0;

    let condensed_t0 = Instant::now();
    let condensed_out = GenCondensedSamples {
        m_max: profile.m_max,
    }
    .process(entropy_out)?;
    let condensed_ms = condensed_t0.elapsed().as_secs_f64() * 1_000.0;

    let select_t0 = Instant::now();
    let selected = select_bases(
        profile.select_impl,
        profile.base_bit_impl,
        condensed_out,
        profile.patience,
    )?;
    let select_ms = select_t0.elapsed().as_secs_f64() * 1_000.0;

    let encode_t0 = Instant::now();
    let compressed = encode_data(profile.encode_impl, selected)?;
    let encode_ms = encode_t0.elapsed().as_secs_f64() * 1_000.0;

    let size_breakdown_bits = estimate_size_breakdown_bits(&compressed);

    Ok(ExperimentRecord {
        file_path: file.to_path_buf(),
        kind: InputKind::Csv,
        preset_name: profile.name.to_string(),
        config: ExperimentConfigColumns {
            csv_has_headers: Some(profile.has_headers),
            csv_float_storage: Some(format!("{:?}", profile.float_storage)),
            csv_missing_value_policy: Some(format!("{:?}", profile.missing_value_policy)),
            csv_float_scaling: Some(format!("{:?}", profile.preprocess.float_scaling)),
            csv_max_decimal_scale: Some(profile.preprocess.max_decimal_scale),
            csv_integer_zero_normalization: Some(profile.preprocess.integer_zero_normalization),
            ..ExperimentConfigColumns::default()
        },
        select_impl: profile.select_impl,
        base_bit_impl: profile.base_bit_impl,
        encode_impl: profile.encode_impl,
        m_max: profile.m_max,
        patience: profile.patience,
        stage_ms: CompressionStageDurationsMs {
            load_input: load_ms,
            preprocess: preprocess_ms,
            entropy: entropy_ms,
            condensed_samples: condensed_ms,
            select_bases: select_ms,
            encode: encode_ms,
            total: total_t0.elapsed().as_secs_f64() * 1_000.0,
        },
        original_bits: compressed.metadata.original_size_bits(),
        size_breakdown_bits,
    })
}

fn run_image_profile(
    file: &Path,
    profile: &ImagePipelineProfile,
) -> Result<ExperimentRecord, EntroGdError> {
    let total_t0 = Instant::now();

    let load_t0 = Instant::now();
    let bit_data = BuildImageBitDataSet {
        colorspace: profile.build.colorspace,
        color_model: profile.build.color_model,
        pixel_grouping: profile.build.pixel_grouping,
        grouping_transform: profile.build.grouping_transform,
    }
    .process(file.to_path_buf())?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1_000.0;

    let entropy_t0 = Instant::now();
    let entropy_out = EntropyOptimized {}.process(bit_data)?;
    let entropy_ms = entropy_t0.elapsed().as_secs_f64() * 1_000.0;

    let condensed_t0 = Instant::now();
    let condensed_out = GenCondensedSamples {
        m_max: profile.m_max,
    }
    .process(entropy_out)?;
    let condensed_ms = condensed_t0.elapsed().as_secs_f64() * 1_000.0;

    let select_t0 = Instant::now();
    let selected = select_bases(
        profile.select_impl,
        profile.base_bit_impl,
        condensed_out,
        profile.patience,
    )?;
    let select_ms = select_t0.elapsed().as_secs_f64() * 1_000.0;

    let encode_t0 = Instant::now();
    let compressed = encode_data(profile.encode_impl, selected)?;
    let encode_ms = encode_t0.elapsed().as_secs_f64() * 1_000.0;

    let size_breakdown_bits = estimate_size_breakdown_bits(&compressed);

    Ok(ExperimentRecord {
        file_path: file.to_path_buf(),
        kind: InputKind::Image,
        preset_name: profile.name.to_string(),
        config: ExperimentConfigColumns {
            image_colorspace: Some(format!("{:?}", profile.build.colorspace)),
            image_color_model: Some(format!("{:?}", profile.build.color_model)),
            image_pixel_grouping: Some(profile.build.pixel_grouping),
            image_grouping_transform: Some(format!("{:?}", profile.build.grouping_transform)),
            ..ExperimentConfigColumns::default()
        },
        select_impl: profile.select_impl,
        base_bit_impl: profile.base_bit_impl,
        encode_impl: profile.encode_impl,
        m_max: profile.m_max,
        patience: profile.patience,
        stage_ms: CompressionStageDurationsMs {
            load_input: load_ms,
            preprocess: 0.0,
            entropy: entropy_ms,
            condensed_samples: condensed_ms,
            select_bases: select_ms,
            encode: encode_ms,
            total: total_t0.elapsed().as_secs_f64() * 1_000.0,
        },
        original_bits: compressed.metadata.original_size_bits(),
        size_breakdown_bits,
    })
}

fn select_bases(
    implementation: SelectBasesImpl,
    base_bit_impl: BaseBitImpl,
    input: (BitDataSet, Vec<(usize, f64)>),
    patience: usize,
) -> Result<(BitDataSet, Box<dyn BaseBit>), EntroGdError> {
    match implementation {
        SelectBasesImpl::Naive => SelectBases { patience }.process(input),
        SelectBasesImpl::Optimized => SelectBasesOptimized {
            patience,
            base_bit_impl: match base_bit_impl {
                BaseBitImpl::Naive => crate::compression::base_selection::BaseBitImpl::Naive,
                BaseBitImpl::BatchGroups => {
                    crate::compression::base_selection::BaseBitImpl::BatchGroups
                }
                BaseBitImpl::IncSignatureGroups => {
                    crate::compression::base_selection::BaseBitImpl::IncSignatureGroups
                }
                BaseBitImpl::SignatureGroups => {
                    crate::compression::base_selection::BaseBitImpl::SignatureGroups
                }
            },
        }
        .process(input),
    }
}

fn encode_data(
    implementation: EncodeImpl,
    input: (BitDataSet, Box<dyn BaseBit>),
) -> Result<CompressedData, EntroGdError> {
    match implementation {
        EncodeImpl::Naive => EncodeData {}.process(input),
        EncodeImpl::Optimized => EncodeDataOptimized {}.process(input),
        EncodeImpl::Rle => EncodeDataRLE {}.process(input),
        EncodeImpl::Huffman => EncodeDataHuffman {}.process(input),
        EncodeImpl::HuffmanBaseIdOnly => EncodeDataHuffmanBaseIdOnly {}.process(input),
    }
}

fn bits_needed_nonzero(value: usize) -> usize {
    if value == 0 {
        1
    } else {
        (usize::BITS as usize) - value.leading_zeros() as usize
    }
}

fn estimate_size_breakdown_bits(compressed: &CompressedData) -> CompressedSizeBreakdownBits {
    let encoded_stream_total = compressed.encoded_data.get_encoded_size();
    let base_table_patterns = compressed
        .base_table
        .iter()
        .map(|(pattern, _)| pattern.len())
        .sum::<usize>();

    let position_width_bits = bits_needed_nonzero(compressed.metadata.chunk_size().max(1));
    let base_bit_positions = compressed.base_bit_positions.len() * position_width_bits;

    let condensed_weights = compressed
        .condensed_sample_weights
        .as_ref()
        .map(|weights| {
            let weight_width = bits_needed_nonzero(compressed.metadata.n_data_samples().max(1));
            weights.len() * weight_width
        })
        .unwrap_or(0);

    let (
        encoded_payload_bits,
        normal_symbol_stream_bits,
        rle_symbol_stream_bits,
        rle_control_stream_bits,
        rle_packet_count,
        huffman_pixel_stream_bits,
        huffman_row_offsets_bits,
        huffman_symbol_table_bits,
        huffman_code_lengths_bits,
    ) = match &compressed.encoded_data {
        EncodedData::Normal(data) => (
            data.encoded_bit_stream().len(),
            data.encoded_bit_stream().len(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        EncodedData::Rle(data) => (
            data.symbol_bit_stream().len(),
            0,
            data.symbol_bit_stream().len(),
            data.rm_control_stream().len(),
            data.rm_values().len(),
            0,
            0,
            0,
            0,
        ),
        EncodedData::Huffman(data) => {
            let symbol_width = data.get_num_deviation_bits() + data.get_num_id_bits();
            let symbol_table_bits = data.canonical_symbols().len() * symbol_width;
            let code_lengths_bits = data.canonical_code_lengths().len() * 5;
            (
                data.pixel_bit_stream().len(),
                0,
                0,
                0,
                0,
                data.pixel_bit_stream().len(),
                data.row_offsets().len() * 32,
                symbol_table_bits,
                code_lengths_bits,
            )
        }
    };

    let estimated_total = encoded_stream_total
        + base_table_patterns
        + base_bit_positions
        + condensed_weights;

    CompressedSizeBreakdownBits {
        encoded_stream_total,
        encoded_payload_bits,
        normal_symbol_stream_bits,
        rle_symbol_stream_bits,
        rle_control_stream_bits,
        rle_packet_count,
        huffman_pixel_stream_bits,
        huffman_row_offsets_bits,
        huffman_symbol_table_bits,
        huffman_code_lengths_bits,
        base_table_patterns,
        base_bit_positions,
        condensed_weights,
        estimated_total,
    }
}
