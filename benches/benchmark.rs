use std::sync::Arc;
use std::path::PathBuf;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};

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
        "{}.{}.egd",
        case.data_file_path.trim_end_matches(".csv"),
        case.variant.label()
    )
    .into()
}

fn run_compression_pipeline(case: RoundtripCase, bit_data: BitDataSet) -> PathBuf {
    let output_path = compressed_output_path(case);

    match case.variant {
        CompressionVariant::BaselineV1 => EntropyOptimized {}
            .then(GenCondensedSamples { m_max: case.m_max })
            .then(SelectBases {
                patience: case.patience,
            })
            .then(EncodeDataOptimized {})
            .then(SaveEgdFile { output_path })
            .process(bit_data)
            .unwrap(),
    }
}

fn benchmark_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compression");

    for case in roundtrip_cases() {
        let loader = CsvDataLoader::new(true).with_float_type(case.float_type);
        let loaded = loader.load(case.data_file_path).unwrap();
        let dataset = loaded.dataset;

        group.throughput(Throughput::Bytes(
            std::fs::metadata(case.data_file_path).unwrap().len() as u64,
        ));

        group.bench_function(BenchmarkId::from_parameter(benchmark_label(case)), |b| {
            b.iter(|| {
                let bit_data = BitDataSet::from_dataset(&dataset).unwrap();
                let _compressed_path = run_compression_pipeline(case, bit_data);
            });
        });
    }

    group.finish();
}

fn benchmark_decompression(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decompression");

    for case in roundtrip_cases() {
        let loader = CsvDataLoader::new(true).with_float_type(case.float_type);
        let loaded = loader.load(case.data_file_path).unwrap();
        let dataset = loaded.dataset;
        let row_indices = (0..dataset.num_rows()).collect::<Vec<usize>>();

        group.throughput(Throughput::Bytes(
            std::fs::metadata(case.data_file_path).unwrap().len() as u64,
        ));

        let compressed_path = run_compression_pipeline(
            case,
            BitDataSet::from_dataset(&dataset).unwrap(),
        );

        group.bench_function(BenchmarkId::from_parameter(benchmark_label(case)), |b| {
            b.iter(|| {
                let loaded_compressed_data = LoadEgdFile {}.process(compressed_path.clone()).unwrap();
                let _decompressed_file_data = DecompressRowsData {}
                    .process((Arc::new(loaded_compressed_data), row_indices.clone()))
                    .unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_compression, benchmark_decompression);
criterion_main!(benches);
