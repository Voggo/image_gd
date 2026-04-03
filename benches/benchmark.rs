use std::path::PathBuf;
use std::sync::Arc;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use std::hint::black_box;

use entro_gd::compression::preprocessor::DEFAULT_BITDATA_ROW_PADDING;
use entro_gd::data_loader::{CsvDataLoader, DataLoader, FloatStorage};
use entro_gd::prelude::*;
use entro_gd::{BitDataSet, CompressedData, Dataset, DecompressRowsData};

#[derive(Clone, Copy)]
enum CompressionVariant {
    BaselineV1,
    ImageBaselineV1,
}

impl CompressionVariant {
    fn label(self) -> &'static str {
        match self {
            CompressionVariant::BaselineV1 => "baseline-v1",
            CompressionVariant::ImageBaselineV1 => "image-baseline-v1",
        }
    }
}

#[derive(Clone, Copy)]
struct RoundtripCase {
    data_file_path: &'static str,
    variant: CompressionVariant,
    input: RoundtripInput,
    m_max: usize,
    patience: usize,
}

#[derive(Clone, Copy)]
enum RoundtripInput {
    Csv {
        float_storage: FloatStorage,
    },
    Image {
        colorspace: ImageColorSpace,
        color_model: ImageColorModel,
        pixel_grouping: u32,
        grouping_transform: ImageGroupingTransform,
    },
}

impl RoundtripCase {
    fn csv(
        data_file_path: &'static str,
        variant: CompressionVariant,
        float_storage: FloatStorage,
        m_max: usize,
        patience: usize,
    ) -> Self {
        Self {
            data_file_path,
            variant,
            input: RoundtripInput::Csv { float_storage },
            m_max,
            patience,
        }
    }

    fn image(
        data_file_path: &'static str,
        variant: CompressionVariant,
        m_max: usize,
        patience: usize,
    ) -> Self {
        Self {
            data_file_path,
            variant,
            input: RoundtripInput::Image {
                colorspace: ImageColorSpace::SrgbWithLinearAlpha,
                color_model: ImageColorModel::YCoCgR,
                pixel_grouping: 1,
                grouping_transform: ImageGroupingTransform::Raw,
            },
            m_max,
            patience,
        }
    }

    fn is_csv(self) -> bool {
        matches!(self.input, RoundtripInput::Csv { .. })
    }

    fn file_extension(self) -> &'static str {
        match self.input {
            RoundtripInput::Csv { .. } => "egd",
            RoundtripInput::Image { .. } => "igd",
        }
    }

    fn input_label(self) -> &'static str {
        match self.input {
            RoundtripInput::Csv { .. } => "csv",
            RoundtripInput::Image { .. } => "image",
        }
    }
}

fn roundtrip_cases() -> Vec<RoundtripCase> {
    vec![
        RoundtripCase::csv(
            "data/tabular/aarhus-citylab.csv",
            CompressionVariant::BaselineV1,
            FloatStorage::F32,
            50,
            10,
        ),
        RoundtripCase::image(
            "data/images/kodim10.png",
            CompressionVariant::ImageBaselineV1,
            0,
            10,
        ),
    ]
}

fn dataset_label(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn benchmark_label(case: RoundtripCase) -> String {
    format!(
        "{}/{}/{}-m{}-p{}",
        case.variant.label(),
        case.input_label(),
        dataset_label(case.data_file_path),
        case.m_max,
        case.patience
    )
}

fn compressed_output_path(case: RoundtripCase) -> PathBuf {
    format!(
        "target/bench-artifacts/{}.{}.{}",
        dataset_label(case.data_file_path),
        case.variant.label(),
        case.file_extension(),
    )
    .into()
}

fn build_loader(case: RoundtripCase) -> CsvDataLoader {
    match case.input {
        RoundtripInput::Csv { float_storage } => CsvDataLoader::new(true).with_float_storage(float_storage),
        RoundtripInput::Image { .. } => {
            panic!("build_loader called for image case: {}", case.data_file_path)
        }
    }
}

fn build_image_bit_data(case: RoundtripCase, path: PathBuf) -> BitDataSet {
    match case.input {
        RoundtripInput::Image {
            colorspace,
            color_model,
            pixel_grouping,
            grouping_transform,
        } => BuildImageBitDataSet {
            colorspace,
            color_model,
            pixel_grouping,
            grouping_transform,
            pad_rows_to_word: DEFAULT_BITDATA_ROW_PADDING,
        }
        .process(path)
        .unwrap(),
        RoundtripInput::Csv { .. } => {
            panic!(
                "build_image_bit_data called for csv case: {}",
                case.data_file_path
            )
        }
    }
}

fn run_compression_core(case: RoundtripCase, bit_data: BitDataSet) -> CompressedData {
    match case.variant {
        CompressionVariant::BaselineV1 => EntropyOptimized {}
            .then(GenCondensedSamples { m_max: case.m_max })
            .then(SelectBases {
                patience: case.patience,
            })
            .then(EncodeDataOptimized {})
            .process(bit_data)
            .unwrap(),
        CompressionVariant::ImageBaselineV1 => EntropyOptimized {}
            .then(SelectBases {
                patience: case.patience,
            })
            .then(EncodeDataOptimized {})
            .process(bit_data)
            .unwrap(),
    }
}

struct PreparedRoundtripCase {
    case: RoundtripCase,
    name: String,
    source_size: u64,
    dataset_seed: Option<Dataset>,
    bit_data_seed: BitDataSet,
    compressed_seed: CompressedData,
    compressed_path: PathBuf,
    row_indices: Vec<usize>,
}

fn prepare_roundtrip_case(case: RoundtripCase) -> PreparedRoundtripCase {
    let _ = std::fs::create_dir_all("target/bench-artifacts");
    let source_size = std::fs::metadata(case.data_file_path).unwrap().len() as u64;
    let dataset_seed = if case.is_csv() {
        let loader = build_loader(case);
        Some(loader.load(case.data_file_path).unwrap().dataset)
    } else {
        None
    };
    let bit_data_seed = match &dataset_seed {
        Some(dataset) => BitDataSet::from_dataset(dataset).unwrap(),
        None => build_image_bit_data(case, PathBuf::from(case.data_file_path)),
    };
    let compressed_seed = run_compression_core(case, bit_data_seed.clone());
    let compressed_path = match case.input {
        RoundtripInput::Csv { .. } => SaveEgdFile {
            output_path: compressed_output_path(case),
        }
        .process(compressed_seed.clone())
        .unwrap(),
        RoundtripInput::Image { .. } => SaveIgdFile {
            output_path: compressed_output_path(case),
        }
        .process(compressed_seed.clone())
        .unwrap(),
    };
    let row_indices = (0..bit_data_seed.num_rows()).collect::<Vec<usize>>();

    PreparedRoundtripCase {
        case,
        name: benchmark_label(case),
        source_size,
        dataset_seed,
        bit_data_seed,
        compressed_seed,
        compressed_path,
        row_indices,
    }
}

fn benchmark_load_csv(c: &mut Criterion) {
    let mut group = c.benchmark_group("LoadCsv");

    for case in roundtrip_cases() {
        if !case.is_csv() {
            continue;
        }
        group.throughput(Throughput::Bytes(
            std::fs::metadata(case.data_file_path).unwrap().len() as u64,
        ));

        group.bench_function(BenchmarkId::from_parameter(benchmark_label(case)), |b| {
            b.iter(|| {
                let loader = build_loader(case);
                let loaded = loader.load(case.data_file_path).unwrap();
                black_box(loaded);
            });
        });
    }

    group.finish();
}

fn benchmark_preprocess_to_bitdata(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("PreprocessToBitData");

    for prepared in &prepared_cases {
        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            match prepared.case.input {
                RoundtripInput::Csv { .. } => b.iter_batched(
                    || prepared.dataset_seed.clone().unwrap(),
                    |dataset| {
                        let bit_data = BitDataSet::from_dataset(&dataset).unwrap();
                        black_box(bit_data);
                    },
                    BatchSize::SmallInput,
                ),
                RoundtripInput::Image { .. } => b.iter_batched(
                    || PathBuf::from(prepared.case.data_file_path),
                    |path| {
                        let bit_data = build_image_bit_data(prepared.case, path);
                        black_box(bit_data);
                    },
                    BatchSize::SmallInput,
                ),
            }
        });
    }

    group.finish();
}

fn benchmark_compression_core(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("CompressionCore");

    for prepared in &prepared_cases {
        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            b.iter_batched(
                || prepared.bit_data_seed.clone(),
                |bit_data| {
                    let compressed = run_compression_core(prepared.case, bit_data);
                    black_box(compressed);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn benchmark_save_compressed(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("SaveCompressedFile");

    for prepared in &prepared_cases {
        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            b.iter_batched(
                || prepared.compressed_seed.clone(),
                |compressed| {
                    let path = match prepared.case.input {
                        RoundtripInput::Csv { .. } => SaveEgdFile {
                            output_path: prepared.compressed_path.clone(),
                        }
                        .process(compressed)
                        .unwrap(),
                        RoundtripInput::Image { .. } => SaveIgdFile {
                            output_path: prepared.compressed_path.clone(),
                        }
                        .process(compressed)
                        .unwrap(),
                    };
                    black_box(path);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn benchmark_decompression_warm(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("DecompressionWarm");

    for prepared in &prepared_cases {
        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            b.iter_batched(
                || {
                    (
                        Arc::new(prepared.compressed_seed.clone()),
                        prepared.row_indices.clone(),
                    )
                },
                |input| {
                    let decompressed = DecompressRowsData {}.process(input).unwrap();
                    black_box(decompressed);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn benchmark_decompress_file_warm(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("DecompressFileWarm");

    for prepared in &prepared_cases {
        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            b.iter_batched(
                || prepared.compressed_seed.clone(),
                |compressed| {
                    let decompressed = DecompressFileData {}.process(compressed).unwrap();
                    black_box(decompressed);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn benchmark_decompression_cold(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("DecompressionCold");

    for prepared in &prepared_cases {
        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            b.iter(|| {
                let loaded_compressed = match prepared.case.input {
                    RoundtripInput::Csv { .. } => LoadEgdFile {}
                        .process(prepared.compressed_path.clone())
                        .unwrap(),
                    RoundtripInput::Image { .. } => LoadIgdFile {}
                        .process(prepared.compressed_path.clone())
                        .unwrap(),
                };
                let decompressed = DecompressRowsData {}
                    .process((Arc::new(loaded_compressed), prepared.row_indices.clone()))
                    .unwrap();
                black_box(decompressed);
            });
        });
    }

    group.finish();
}

fn benchmark_decompress_file_cold(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("DecompressFileCold");

    for prepared in &prepared_cases {
        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            b.iter(|| {
                let loaded_compressed = match prepared.case.input {
                    RoundtripInput::Csv { .. } => LoadEgdFile {}
                        .process(prepared.compressed_path.clone())
                        .unwrap(),
                    RoundtripInput::Image { .. } => LoadIgdFile {}
                        .process(prepared.compressed_path.clone())
                        .unwrap(),
                };
                let decompressed = DecompressFileData {}.process(loaded_compressed).unwrap();
                black_box(decompressed);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_load_csv,
    benchmark_preprocess_to_bitdata,
    benchmark_compression_core,
    benchmark_save_compressed,
    benchmark_decompression_warm,
    benchmark_decompress_file_warm,
    benchmark_decompression_cold,
    benchmark_decompress_file_cold
);
criterion_main!(benches);
