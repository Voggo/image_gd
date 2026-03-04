use std::path::PathBuf;
use std::sync::Arc;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use std::hint::black_box;

use entro_gd::CompressedData;
use entro_gd::Dataset;
use entro_gd::compression::compress::DecompressRowsData;
use entro_gd::prelude::*;

#[derive(Clone, Copy)]
enum CompressionVariant {
    BaselineV1,
}

impl CompressionVariant {
    fn label(self) -> &'static str {
        match self {
            CompressionVariant::BaselineV1 => "baseline-v1",
        }
    }
}

#[derive(Clone, Copy)]
struct RoundtripCase {
    data_file_path: &'static str,
    variant: CompressionVariant,
    float_type: FeatureDataType,
    m_max: usize,
    patience: usize,
}

fn roundtrip_cases() -> Vec<RoundtripCase> {
    vec![RoundtripCase {
        data_file_path: "data/aarhus-citylab.csv",
        variant: CompressionVariant::BaselineV1,
        float_type: FeatureDataType::F32,
        m_max: 50,
        patience: 10,
    }]
}

fn dataset_label(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn benchmark_label(case: RoundtripCase) -> String {
    format!(
        "{}/{}-m{}-p{}",
        case.variant.label(),
        dataset_label(case.data_file_path),
        case.m_max,
        case.patience
    )
}

fn compressed_output_path(case: RoundtripCase) -> PathBuf {
    format!(
        "target/bench-artifacts/{}.{}.egd",
        dataset_label(case.data_file_path),
        case.variant.label()
    )
    .into()
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
    }
}

struct PreparedRoundtripCase {
    case: RoundtripCase,
    name: String,
    source_size: u64,
    dataset_seed: Dataset,
    bit_data_seed: BitDataSet,
    compressed_seed: CompressedData,
    compressed_path: PathBuf,
    row_indices: Vec<usize>,
}

fn prepare_roundtrip_case(case: RoundtripCase) -> PreparedRoundtripCase {
    let _ = std::fs::create_dir_all("target/bench-artifacts");
    let loader = CsvDataLoader::new(true).with_float_type(case.float_type);
    let loaded = loader.load(case.data_file_path).unwrap();
    let dataset_seed = loaded.dataset;
    let source_size = std::fs::metadata(case.data_file_path).unwrap().len() as u64;
    let bit_data_seed = BitDataSet::from_dataset(&dataset_seed).unwrap();
    let compressed_seed = run_compression_core(case, bit_data_seed.clone());
    let compressed_path = SaveEgdFile {
        output_path: compressed_output_path(case),
    }
    .process(compressed_seed.clone())
    .unwrap();
    let row_indices = (0..dataset_seed.num_rows()).collect::<Vec<usize>>();

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
        group.throughput(Throughput::Bytes(
            std::fs::metadata(case.data_file_path).unwrap().len() as u64,
        ));

        group.bench_function(BenchmarkId::from_parameter(benchmark_label(case)), |b| {
            b.iter(|| {
                let loader = CsvDataLoader::new(true).with_float_type(case.float_type);
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
            b.iter_batched(
                || prepared.dataset_seed.clone(),
                |dataset| {
                    let bit_data = BitDataSet::from_dataset(&dataset).unwrap();
                    black_box(bit_data);
                },
                BatchSize::SmallInput,
            );
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

fn benchmark_save_egd(c: &mut Criterion) {
    let prepared_cases: Vec<PreparedRoundtripCase> = roundtrip_cases()
        .into_iter()
        .map(prepare_roundtrip_case)
        .collect();
    let mut group = c.benchmark_group("SaveEgdFile");

    for prepared in &prepared_cases {
        group.throughput(Throughput::Bytes(prepared.source_size));
        group.bench_function(BenchmarkId::from_parameter(&prepared.name), |b| {
            b.iter_batched(
                || prepared.compressed_seed.clone(),
                |compressed| {
                    let path = SaveEgdFile {
                        output_path: prepared.compressed_path.clone(),
                    }
                    .process(compressed)
                    .unwrap();
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
                let loaded_compressed = LoadEgdFile {}
                    .process(prepared.compressed_path.clone())
                    .unwrap();
                let decompressed = DecompressRowsData {}
                    .process((Arc::new(loaded_compressed), prepared.row_indices.clone()))
                    .unwrap();
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
    benchmark_save_egd,
    benchmark_decompression_warm,
    benchmark_decompression_cold
);
criterion_main!(benches);
