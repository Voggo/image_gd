use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use image::GenericImageView;
use png::{BitDepth, ColorType, Compression, Decoder, Encoder, FilterType};
use std::hint::black_box;

use entro_gd::compression::preprocessor::DEFAULT_ALIGN_ROWS_TO_WORD;
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

fn png_output_path(case: RoundtripCase) -> PathBuf {
    format!(
        "target/bench-artifacts/{}.{}.png",
        dataset_label(case.data_file_path),
        case.variant.label(),
    )
    .into()
}

fn build_loader(case: RoundtripCase) -> CsvDataLoader {
    match case.input {
        RoundtripInput::Csv { float_storage } => {
            CsvDataLoader::new(true).with_float_storage(float_storage)
        }
        RoundtripInput::Image { .. } => {
            panic!(
                "build_loader called for image case: {}",
                case.data_file_path
            )
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
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
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
        CompressionVariant::BaselineV1 => EntropyBatched {}
            .then(GenCondensedSamples { m_max: case.m_max })
            .then(SelectBases {
                patience: case.patience,
            })
            .then(EncodeDataOptimized {})
            .process(bit_data)
            .unwrap(),
        CompressionVariant::ImageBaselineV1 => EntropyNaive {}
            .then(SelectBases {
                patience: case.patience,
            })
            .then(EncodeDataOptimized {})
            .process(bit_data)
            .unwrap(),
    }
}

#[derive(Clone)]
struct RawImageData {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

fn load_raw_image_rgba(path: &str) -> RawImageData {
    let image = image::open(path).unwrap();
    let (width, height) = image.dimensions();
    let rgba = image.to_rgba8().into_raw();
    RawImageData {
        width,
        height,
        rgba,
    }
}

fn encode_png_bytes(raw: &RawImageData) -> Vec<u8> {
    let mut output = Vec::new();
    let mut encoder = Encoder::new(&mut output, raw.width, raw.height);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    encoder.set_compression(Compression::Default);
    encoder.set_filter(FilterType::Sub);

    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&raw.rgba).unwrap();
    drop(writer);
    output
}

fn decode_png_bytes(input: &[u8]) -> RawImageData {
    let decoder = Decoder::new(Cursor::new(input));
    let mut reader = decoder.read_info().unwrap();
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer).unwrap();
    buffer.truncate(info.buffer_size());

    RawImageData {
        width: info.width,
        height: info.height,
        rgba: buffer,
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
    raw_image_seed: Option<RawImageData>,
    png_bytes_seed: Option<Vec<u8>>,
    png_path: Option<PathBuf>,
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
    let (raw_image_seed, png_bytes_seed, png_path) = match case.input {
        RoundtripInput::Image { .. } => {
            let raw_image = load_raw_image_rgba(case.data_file_path);
            let png_bytes = encode_png_bytes(&raw_image);
            let png_path = png_output_path(case);
            std::fs::write(&png_path, &png_bytes).unwrap();
            (Some(raw_image), Some(png_bytes), Some(png_path))
        }
        RoundtripInput::Csv { .. } => (None, None, None),
    };

    PreparedRoundtripCase {
        case,
        name: benchmark_label(case),
        source_size,
        dataset_seed,
        bit_data_seed,
        compressed_seed,
        compressed_path,
        row_indices,
        raw_image_seed,
        png_bytes_seed,
        png_path,
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
        group.bench_function(
            BenchmarkId::from_parameter(&prepared.name),
            |b| match prepared.case.input {
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
            },
        );
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

fn benchmark_png_compress_file_warm(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("PngCompressFileWarm");

    for prepared in &prepared_cases {
        let Some(raw_image_seed) = &prepared.raw_image_seed else {
            continue;
        };

        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            b.iter_batched(
                || raw_image_seed.clone(),
                |raw_image| {
                    let encoded = encode_png_bytes(&raw_image);
                    black_box(encoded);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn benchmark_png_decompress_file_warm(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("PngDecompressFileWarm");

    for prepared in &prepared_cases {
        let Some(png_bytes_seed) = &prepared.png_bytes_seed else {
            continue;
        };

        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            b.iter_batched(
                || png_bytes_seed.clone(),
                |png_bytes| {
                    let decoded = decode_png_bytes(&png_bytes);
                    black_box(decoded);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn benchmark_png_decompress_file_cold(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("PngDecompressFileCold");

    for prepared in &prepared_cases {
        let Some(png_path) = &prepared.png_path else {
            continue;
        };

        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            b.iter(|| {
                let png_bytes = std::fs::read(png_path).unwrap();
                let decoded = decode_png_bytes(&png_bytes);
                black_box(decoded);
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
    benchmark_decompress_file_cold,
    benchmark_png_compress_file_warm,
    benchmark_png_decompress_file_warm,
    benchmark_png_decompress_file_cold
);
criterion_main!(benches);
