use crate::BitDataSet;
use crate::compression::base_selection::BaseSelectionContext;
use crate::compression::encoding::{CompressedData, EncodedData};
use crate::compression::entropy::EntropyScoredContext;
use crate::compression::image_preprocessor::BuildImageBitDataSet;
use crate::compression::preprocessor::{
    BuildBitDataSet, DEFAULT_ALIGN_ROWS_TO_WORD, InferFeatureSpecs,
};
use crate::data_loader::{CsvDataLoader, DataLoader};
use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::pipeline_profiles::{
    BaseBitImpl, BaseTableImpl, CsvPipelineProfile, EncodeImpl, EntropyImpl, ImagePipelineProfile,
    PipelineProfileSet, SelectBasesImpl,
};
use crate::prelude::*;
use crate::utils::bits_needed_nonzero;
use std::fs;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Csv,
    Image,
    Unsupported,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExperimentRunOptions {
    pub recursive: bool,
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
    pub entropy_impl: Option<String>,
    pub entropy_skip_rows: Option<usize>,
    pub use_condensed_samples: Option<bool>,
    pub base_table_impl: Option<String>,
    pub delta_encode_base_table: Option<bool>,
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

struct TerminalProgress {
    enabled: bool,
    start: Instant,
    total_files: usize,
    total_runs: usize,
    completed_runs: usize,
    spinner_index: usize,
}

impl TerminalProgress {
    const SPINNER_FRAMES: [&'static str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

    fn new(total_files: usize, total_runs: usize) -> Self {
        Self {
            enabled: std::io::stderr().is_terminal(),
            start: Instant::now(),
            total_files,
            total_runs,
            completed_runs: 0,
            spinner_index: 0,
        }
    }

    fn begin(&mut self) {
        if self.enabled {
            let _ = write!(
                std::io::stderr(),
                "\x1b[2K\r🚀 Starting experiment run • files: {} • jobs: {}",
                self.total_files,
                self.total_runs,
            );
            let _ = std::io::stderr().flush();
        }
    }

    fn file_started(
        &mut self,
        file_idx: usize,
        path: &Path,
        kind: InputKind,
        profile_count: usize,
    ) {
        if self.enabled {
            let _ = write!(
                std::io::stderr(),
                "\x1b[2K\r📂 File {}/{} [{:?}] {} • profiles: {}",
                file_idx,
                self.total_files,
                kind,
                path.display(),
                profile_count,
            );
            let _ = std::io::stderr().flush();
        }
    }

    fn job_finished(&mut self, file_idx: usize, profile_name: &str, stage_total_ms: f64) {
        self.completed_runs += 1;
        if !self.enabled {
            return;
        }

        self.spinner_index = (self.spinner_index + 1) % Self::SPINNER_FRAMES.len();
        let spinner = Self::SPINNER_FRAMES[self.spinner_index];

        let percent = if self.total_runs == 0 {
            100.0
        } else {
            (self.completed_runs as f64 / self.total_runs as f64) * 100.0
        };

        let bar_width = 24usize;
        let filled = if self.total_runs == 0 {
            bar_width
        } else {
            (self.completed_runs * bar_width) / self.total_runs
        }
        .min(bar_width);
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(bar_width - filled));

        let elapsed = self.start.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            self.completed_runs as f64 / elapsed
        } else {
            0.0
        };
        let remaining = self.total_runs.saturating_sub(self.completed_runs) as f64;
        let eta_seconds = if rate > 0.0 {
            (remaining / rate).max(0.0)
        } else {
            0.0
        };
        let eta_mins = (eta_seconds as u64) / 60;
        let eta_secs = (eta_seconds as u64) % 60;

        let _ = write!(
            std::io::stderr(),
            "\x1b[2K\r{} [{}] {}/{} ({:>5.1}%) • file {}/{} • {} • {:.1} it/s • η {:02}:{:02} • last {:.1} ms",
            spinner,
            bar,
            self.completed_runs,
            self.total_runs,
            percent,
            file_idx,
            self.total_files,
            profile_name,
            rate,
            eta_mins,
            eta_secs,
            stage_total_ms,
        );
        let _ = std::io::stderr().flush();
    }

    fn finished(&mut self, skipped_files: usize) {
        if !self.enabled {
            return;
        }

        let elapsed = self.start.elapsed().as_secs_f64();
        let _ = writeln!(
            std::io::stderr(),
            "\x1b[2K\r✅ Completed {} jobs in {:.2}s • skipped files: {}",
            self.completed_runs,
            elapsed,
            skipped_files,
        );
    }
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
    let total_runs = files
        .iter()
        .map(|file| match detect_input_kind(file) {
            InputKind::Csv => profiles.csv.len(),
            InputKind::Image => profiles.image.len(),
            InputKind::Unsupported => 0,
        })
        .sum::<usize>();

    let mut progress = TerminalProgress::new(files.len(), total_runs);
    progress.begin();

    let mut records = Vec::new();
    let mut skipped_files = Vec::new();

    for (file_idx, file) in files.iter().enumerate() {
        let display_idx = file_idx + 1;
        let kind = detect_input_kind(file);
        match kind {
            InputKind::Csv => {
                progress.file_started(display_idx, file, kind, profiles.csv.len());
                let mut file_records = Vec::new();
                for profile in &profiles.csv {
                    let record = run_csv_profile(file, profile)?;
                    progress.job_finished(display_idx, &profile.name, record.stage_ms.total);
                    file_records.push(record);
                }
                records.extend(file_records);
            }
            InputKind::Image => {
                progress.file_started(display_idx, file, kind, profiles.image.len());
                let mut file_records = Vec::new();
                for profile in &profiles.image {
                    let record = run_image_profile(file, profile)?;
                    progress.job_finished(display_idx, &profile.name, record.stage_ms.total);
                    file_records.push(record);
                }
                records.extend(file_records);
            }
            InputKind::Unsupported => skipped_files.push(file.clone()),
        }
    }

    progress.finished(skipped_files.len());

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
        "file_path,input_kind,preset,select_impl,base_bit_impl,entropy_impl,entropy_skip_rows,use_condensed_samples,base_table_impl,delta_encode_base_table,encode_impl,m_max,patience,csv_has_headers,csv_float_storage,csv_missing_value_policy,csv_float_scaling,csv_max_decimal_scale,csv_integer_zero_normalization,image_colorspace,image_color_model,image_pixel_grouping,image_grouping_transform,original_bits,load_ms,preprocess_ms,entropy_ms,condensed_ms,select_ms,encode_ms,total_ms,encoded_stream_total_bits,encoded_payload_bits,normal_symbol_stream_bits,rle_symbol_stream_bits,rle_control_stream_bits,rle_packet_count,huffman_pixel_stream_bits,huffman_row_offsets_bits,huffman_symbol_table_bits,huffman_code_lengths_bits,base_table_pattern_bits,base_bit_positions_bits,condensed_weights_bits,estimated_total_bits"
    )?;

    for record in records {
        writeln!(
            file,
            "{},{:?},{},{:?},{:?},{},{},{},{},{},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            record.file_path.display(),
            record.kind,
            record.preset_name,
            record.select_impl,
            record.base_bit_impl,
            record.config.entropy_impl.as_deref().unwrap_or(""),
            record
                .config
                .entropy_skip_rows
                .map(|v| v.to_string())
                .unwrap_or_default(),
            record
                .config
                .use_condensed_samples
                .map(|v| v.to_string())
                .unwrap_or_default(),
            record.config.base_table_impl.as_deref().unwrap_or(""),
            record
                .config
                .delta_encode_base_table
                .map(|v| v.to_string())
                .unwrap_or_default(),
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
    let bit_data = BuildBitDataSet::default().process((dataset, features))?;
    let preprocess_ms = preprocess_t0.elapsed().as_secs_f64() * 1_000.0;

    let entropy_t0 = Instant::now();
    let entropy_out = run_entropy(profile.entropy_impl, profile.entropy_skip_rows, bit_data)?;
    let entropy_ms = entropy_t0.elapsed().as_secs_f64() * 1_000.0;

    let (select_input, condensed_ms) = if profile.use_condensed_samples {
        let condensed_t0 = Instant::now();
        let condensed_out = GenCondensedSamples {
            m_max: profile.m_max,
        }
        .process(entropy_out)?;
        (
            condensed_out,
            condensed_t0.elapsed().as_secs_f64() * 1_000.0,
        )
    } else {
        (entropy_out, 0.0)
    };

    let select_t0 = Instant::now();
    let selected = select_bases(
        profile.select_impl,
        profile.base_bit_impl,
        select_input,
        profile.patience,
    )?;
    let select_ms = select_t0.elapsed().as_secs_f64() * 1_000.0;

    let encode_t0 = Instant::now();
    let compressed = encode_data(
        profile.encode_impl,
        profile.base_table_impl,
        profile.delta_encode_base_table,
        selected,
    )?;
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
            entropy_impl: Some(format!("{:?}", profile.entropy_impl)),
            entropy_skip_rows: Some(profile.entropy_skip_rows),
            use_condensed_samples: Some(profile.use_condensed_samples),
            base_table_impl: Some(format!("{:?}", profile.base_table_impl)),
            delta_encode_base_table: Some(profile.delta_encode_base_table),
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
        pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
    }
    .process(file.to_path_buf())?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1_000.0;

    let entropy_t0 = Instant::now();
    let entropy_out = run_entropy(profile.entropy_impl, profile.entropy_skip_rows, bit_data)?;
    let entropy_ms = entropy_t0.elapsed().as_secs_f64() * 1_000.0;

    let (select_input, condensed_ms) = if profile.use_condensed_samples {
        let condensed_t0 = Instant::now();
        let condensed_out = GenCondensedSamples {
            m_max: profile.m_max,
        }
        .process(entropy_out)?;
        (
            condensed_out,
            condensed_t0.elapsed().as_secs_f64() * 1_000.0,
        )
    } else {
        (entropy_out, 0.0)
    };

    let select_t0 = Instant::now();
    let selected = select_bases(
        profile.select_impl,
        profile.base_bit_impl,
        select_input,
        profile.patience,
    )?;
    let select_ms = select_t0.elapsed().as_secs_f64() * 1_000.0;

    let encode_t0 = Instant::now();
    let compressed = encode_data(
        profile.encode_impl,
        profile.base_table_impl,
        profile.delta_encode_base_table,
        selected,
    )?;
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
            entropy_impl: Some(format!("{:?}", profile.entropy_impl)),
            entropy_skip_rows: Some(profile.entropy_skip_rows),
            use_condensed_samples: Some(profile.use_condensed_samples),
            base_table_impl: Some(format!("{:?}", profile.base_table_impl)),
            delta_encode_base_table: Some(profile.delta_encode_base_table),
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
    input: EntropyScoredContext,
    patience: usize,
) -> Result<BaseSelectionContext, EntroGdError> {
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
                BaseBitImpl::HyperLogLogCount => {
                    crate::compression::base_selection::BaseBitImpl::HyperLogLogCount
                }
            },
        }
        .process(input),
        SelectBasesImpl::ProfileAllBits => SelectBasesProfileAllBits {
            split_into_batches: patience,
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
                BaseBitImpl::HyperLogLogCount => {
                    crate::compression::base_selection::BaseBitImpl::HyperLogLogCount
                }
            },
        }
        .process(input),
    }
}

fn encode_data(
    implementation: EncodeImpl,
    base_table_impl: BaseTableImpl,
    delta_encode_base_table: bool,
    input: BaseSelectionContext,
) -> Result<CompressedData, EntroGdError> {
    match implementation {
        EncodeImpl::FusedDictionary => EncodeDataFusedDictionary {}.process(input),
        _ => {
            let base_table_ctx = match base_table_impl {
                BaseTableImpl::Raw => BuildBaseTable {}.process(input)?,
                BaseTableImpl::Sorted => BuildSortedBaseTable {}.process(input)?,
            };

            let compressed = match implementation {
                EncodeImpl::Naive => EncodeData {}.process(base_table_ctx),
                EncodeImpl::Optimized => EncodeDataOptimized {}.process(base_table_ctx),
                EncodeImpl::Rle => EncodeDataRLE {}.process(base_table_ctx),
                EncodeImpl::Huffman => EncodeDataHuffman {}.process(base_table_ctx),
                EncodeImpl::HuffmanBaseIdOnly => {
                    EncodeDataHuffmanBaseIdOnly {}.process(base_table_ctx)
                }
                EncodeImpl::FusedDictionary => unreachable!(),
            }?;

            if delta_encode_base_table {
                DeltaEncodeBaseTable {}.process(compressed)
            } else {
                Ok(compressed)
            }
        }
    }
}

fn run_entropy(
    implementation: EntropyImpl,
    skip_rows: usize,
    input: BitDataSet,
) -> Result<EntropyScoredContext, EntroGdError> {
    match implementation {
        EntropyImpl::Naive => EntropyNaive {}.process(input),
        EntropyImpl::Batched => EntropyBatched {}.process(input),
        EntropyImpl::StrideSampled => EntropyStrideSampled { skip_rows }.process(input),
        EntropyImpl::StrideSampledBatched => {
            EntropyStrideSampledBatched { skip_rows }.process(input)
        }
    }
}

fn estimate_size_breakdown_bits(compressed: &CompressedData) -> CompressedSizeBreakdownBits {
    let encoded_stream_total = compressed.encoded_data.get_encoded_size();
    let variable_base_bits = compressed.layout.variable_base_bit_positions().len();
    let base_table_payload_bits = match &compressed.base_table {
        crate::compression::encoding::BaseTable::Raw(rows) => {
            // Matches EgdFile::from_compressed_data raw base-table layout:
            // - base table tag (u8)
            // - num_bases (u64)
            // - packed row bits in metadata variable-base order (lb bits per row)
            let num_bases = rows.len();
            8 + 64 + num_bases * variable_base_bits
        }
        crate::compression::encoding::BaseTable::Delta(delta) => {
            // Matches EgdFile::from_compressed_data delta base-table layout:
            // - base table tag (u8)
            // - num_bases (u64)
            // - sort order len (u64)
            // - sort order indices (index_bits each)
            // - first sort key (lb bits, only if num_bases > 0)
            // - delta_count (u64)
            // - delta_bit_len (u64)
            // - delta bitstream payload
            let num_bases = delta.raw_rows.len();
            let index_bits = bits_needed_nonzero(variable_base_bits.max(1));
            let order_bits = delta.sort_column_order.len() * index_bits;
            let first_key_bits = if num_bases > 0 { variable_base_bits } else { 0 };
            8 + 64 + 64 + order_bits + first_key_bits + 64 + 64 + delta.delta_bit_stream.len()
        }
    };

    // EgdFile::from_compressed_data aligns to byte after base table.
    let base_table_padding_bits = (8 - (base_table_payload_bits % 8)) % 8;
    let base_table_patterns = base_table_payload_bits + base_table_padding_bits;

    let position_width_bits = bits_needed_nonzero(compressed.metadata.chunk_size().max(1));
    let base_bit_positions =
        compressed.layout.selected_base_bit_positions().len() * position_width_bits;

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

    let estimated_total =
        encoded_stream_total + base_table_patterns + base_bit_positions + condensed_weights;

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
