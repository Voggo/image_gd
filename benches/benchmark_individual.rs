use std::path::PathBuf;
use std::sync::Arc;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use std::hint::black_box;

use entro_gd::CompressedData;
use entro_gd::compression::compress::{
    DecompressRowsData, SelectBasesOptimizedv1, SelectBasesOptimizedv2, SelectBasesOptimizedv3,
};
use entro_gd::prelude::*;

type DynBaseBit = Box<dyn entro_gd::compression::base_bits::BaseBit>;

#[derive(Clone, Copy)]
enum EntropyImpl {
    Naive,
    Optimized,
}

impl EntropyImpl {
    fn label(self) -> &'static str {
        match self {
            EntropyImpl::Naive => "naive",
            EntropyImpl::Optimized => "optimized",
        }
    }

    fn entropy(
        self,
        bit_data: BitDataSet,
    ) -> Result<(BitDataSet, Vec<(usize, f64)>), EntroGdError> {
        match self {
            EntropyImpl::Naive => EntropyNaive {}.process(bit_data),
            EntropyImpl::Optimized => EntropyOptimized {}.process(bit_data),
        }
    }
}

#[derive(Clone, Copy)]
enum GenCondensedImpl {
    Current,
}

impl GenCondensedImpl {
    fn label(self) -> &'static str {
        match self {
            GenCondensedImpl::Current => "current",
        }
    }

    fn process(
        self,
        input: (BitDataSet, Vec<(usize, f64)>),
        m_max: usize,
    ) -> Result<(BitDataSet, Vec<(usize, f64)>), EntroGdError> {
        match self {
            GenCondensedImpl::Current => GenCondensedSamples { m_max }.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum SelectBasesImpl {
    Naive,
    Optimizedv1,
    Optimizedv2,
    Optimizedv3,
}

impl SelectBasesImpl {
    fn label(self) -> &'static str {
        match self {
            SelectBasesImpl::Naive => "naive",
            SelectBasesImpl::Optimizedv1 => "optimizedv1",
            SelectBasesImpl::Optimizedv2 => "optimizedv2",
            SelectBasesImpl::Optimizedv3 => "optimizedv3",
        }
    }

    fn process(
        self,
        input: (BitDataSet, Vec<(usize, f64)>),
        patience: usize,
    ) -> Result<(BitDataSet, DynBaseBit), EntroGdError> {
        match self {
            SelectBasesImpl::Naive => SelectBases { patience }.process(input),
            SelectBasesImpl::Optimizedv1 => SelectBasesOptimizedv1 { patience }.process(input),
            SelectBasesImpl::Optimizedv2 => SelectBasesOptimizedv2 { patience }.process(input),
            SelectBasesImpl::Optimizedv3 => SelectBasesOptimizedv3 { patience }.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum EncodeImpl {
    Naive,
    Optimized,
}

impl EncodeImpl {
    fn label(self) -> &'static str {
        match self {
            EncodeImpl::Naive => "naive",
            EncodeImpl::Optimized => "optimized",
        }
    }

    fn process(self, input: (BitDataSet, DynBaseBit)) -> Result<CompressedData, EntroGdError> {
        match self {
            EncodeImpl::Naive => EncodeData {}.process(input),
            EncodeImpl::Optimized => EncodeDataOptimized {}.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum SaveImpl {
    Current,
}

impl SaveImpl {
    fn label(self) -> &'static str {
        match self {
            SaveImpl::Current => "current",
        }
    }

    fn process(self, input: CompressedData, output_path: PathBuf) -> Result<PathBuf, EntroGdError> {
        match self {
            SaveImpl::Current => SaveEgdFile { output_path }.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum LoadImpl {
    Current,
}

impl LoadImpl {
    fn label(self) -> &'static str {
        match self {
            LoadImpl::Current => "current",
        }
    }

    fn process(self, input: PathBuf) -> Result<CompressedData, EntroGdError> {
        match self {
            LoadImpl::Current => LoadEgdFile {}.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum DecompressRowsImpl {
    Current,
}

impl DecompressRowsImpl {
    fn label(self) -> &'static str {
        match self {
            DecompressRowsImpl::Current => "current",
        }
    }

    fn process(self, input: (Arc<CompressedData>, Vec<usize>)) -> Result<BitDataSet, EntroGdError> {
        match self {
            DecompressRowsImpl::Current => DecompressRowsData.process(input),
        }
    }
}

#[derive(Clone, Copy)]
struct StepBenchCase {
    data_file_path: &'static str,
    float_type: FeatureDataType,
    m_max: usize,
    patience: usize,
}

fn step_bench_cases() -> Vec<StepBenchCase> {
    vec![
        StepBenchCase {
            data_file_path: "data/aarhus-citylab.csv",
            float_type: FeatureDataType::F32,
            m_max: 50,
            patience: 10,
        },
        StepBenchCase {
            data_file_path: "data/aarhus-citylab_duplicated.csv",
            float_type: FeatureDataType::F32,
            m_max: 50,
            patience: 10,
        },
    ]
}

const ENTROPY_IMPLS: [EntropyImpl; 2] = [EntropyImpl::Naive, EntropyImpl::Optimized];
const GEN_CONDENSED_IMPLS: [GenCondensedImpl; 1] = [GenCondensedImpl::Current];
const SELECT_BASES_IMPLS: [SelectBasesImpl; 4] = [
    SelectBasesImpl::Naive,
    SelectBasesImpl::Optimizedv1,
    SelectBasesImpl::Optimizedv2,
    SelectBasesImpl::Optimizedv3,
];
const ENCODE_IMPLS: [EncodeImpl; 2] = [EncodeImpl::Naive, EncodeImpl::Optimized];
const SAVE_IMPLS: [SaveImpl; 1] = [SaveImpl::Current];
const LOAD_IMPLS: [LoadImpl; 1] = [LoadImpl::Current];
const DECOMPRESS_ROWS_IMPLS: [DecompressRowsImpl; 1] = [DecompressRowsImpl::Current];

fn dataset_label(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn case_label(case: StepBenchCase) -> String {
    format!(
        "{}-m{}-p{}",
        dataset_label(case.data_file_path),
        case.m_max,
        case.patience
    )
}

fn output_path(case: StepBenchCase) -> PathBuf {
    PathBuf::from(format!(
        "target/bench-artifacts/{}.egd",
        dataset_label(case.data_file_path),
    ))
}

struct PreparedCase {
    name: String,
    source_size: u64,
    m_max: usize,
    patience: usize,
    bit_data_seed: BitDataSet,
    entropy_seed: (BitDataSet, Vec<(usize, f64)>),
    condensed_seed: (BitDataSet, Vec<(usize, f64)>),
    compressed_seed: CompressedData,
    egd_path: PathBuf,
    rows_input_seed: (Arc<CompressedData>, Vec<usize>),
}

fn prepare_case(case: StepBenchCase) -> PreparedCase {
    let loader = CsvDataLoader::new(true).with_float_type(case.float_type);
    let loaded = loader.load(case.data_file_path).unwrap();
    let dataset = loaded.dataset;
    let source_size = std::fs::metadata(case.data_file_path).unwrap().len() as u64;

    let bit_data_seed = BitDataSet::from_dataset(&dataset).unwrap();
    let entropy_seed = EntropyOptimized {}.process(bit_data_seed.clone()).unwrap();
    let condensed_seed = GenCondensedSamples { m_max: case.m_max }
        .process(entropy_seed.clone())
        .unwrap();
    let selected_seed = SelectBasesOptimizedv1 {
        patience: case.patience,
    }
    .process(condensed_seed.clone())
    .unwrap();
    let compressed_seed = EncodeDataOptimized {}.process(selected_seed).unwrap();

    let egd_path = output_path(case);
    let _saved_once = SaveEgdFile {
        output_path: egd_path.clone(),
    }
    .process(compressed_seed.clone())
    .unwrap();
    let loaded_compressed_seed = LoadEgdFile {}.process(egd_path.clone()).unwrap();
    let rows_seed = (0..dataset.num_rows()).collect::<Vec<usize>>();
    let rows_input_seed = (Arc::new(loaded_compressed_seed), rows_seed);

    PreparedCase {
        name: case_label(case),
        source_size,
        m_max: case.m_max,
        patience: case.patience,
        bit_data_seed,
        entropy_seed,
        condensed_seed,
        compressed_seed,
        egd_path,
        rows_input_seed,
    }
}

fn bench_step_group<ImplType, Input, Output, LabelFn, InputFn, RunFn>(
    c: &mut Criterion,
    group_name: &str,
    prepared_cases: &[PreparedCase],
    implementations: &[ImplType],
    impl_label: LabelFn,
    make_input: InputFn,
    run_impl: RunFn,
) where
    ImplType: Copy,
    LabelFn: Fn(ImplType) -> &'static str + Copy,
    InputFn: Fn(&PreparedCase) -> Input + Copy,
    RunFn: Fn(ImplType, Input, &PreparedCase) -> Result<Output, EntroGdError> + Copy,
{
    let mut group = c.benchmark_group(group_name);
    for case in prepared_cases {
        group.throughput(Throughput::Bytes(case.source_size));
        for implementation in implementations.iter().copied() {
            group.bench_function(
                BenchmarkId::from_parameter(format!(
                    "{}/{}",
                    impl_label(implementation),
                    case.name
                )),
                |b| {
                    b.iter_batched(
                        || make_input(case),
                        |input| {
                            let output = run_impl(implementation, input, case).unwrap();
                            black_box(output);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

fn benchmark_filter_steps(c: &mut Criterion) {
    let _ = std::fs::create_dir_all("target/bench-artifacts");
    let prepared_cases: Vec<PreparedCase> =
        step_bench_cases().into_iter().map(prepare_case).collect();

    bench_step_group(
        c,
        "Step/Entropy.process",
        &prepared_cases,
        &ENTROPY_IMPLS,
        EntropyImpl::label,
        |case| case.bit_data_seed.clone(),
        |implementation, input, _case| implementation.entropy(input),
    );

    bench_step_group(
        c,
        "Step/GenCondensedSamples.process",
        &prepared_cases,
        &GEN_CONDENSED_IMPLS,
        GenCondensedImpl::label,
        |case| case.entropy_seed.clone(),
        |implementation, input, case| implementation.process(input, case.m_max),
    );

    bench_step_group(
        c,
        "Step/SelectBases.process",
        &prepared_cases,
        &SELECT_BASES_IMPLS,
        SelectBasesImpl::label,
        |case| case.condensed_seed.clone(),
        |implementation, input, case| implementation.process(input, case.patience),
    );

    bench_step_group(
        c,
        "Step/EncodeData.process",
        &prepared_cases,
        &ENCODE_IMPLS,
        EncodeImpl::label,
        |case| {
            SelectBasesOptimizedv1 {
                patience: case.patience,
            }
            .process(case.condensed_seed.clone())
            .unwrap()
        },
        |implementation, input, _case| implementation.process(input),
    );

    bench_step_group(
        c,
        "Step/SaveEgdFile.process",
        &prepared_cases,
        &SAVE_IMPLS,
        SaveImpl::label,
        |case| case.compressed_seed.clone(),
        |implementation, input, case| implementation.process(input, case.egd_path.clone()),
    );

    bench_step_group(
        c,
        "Step/LoadEgdFile.process",
        &prepared_cases,
        &LOAD_IMPLS,
        LoadImpl::label,
        |case| case.egd_path.clone(),
        |implementation, input, _case| implementation.process(input),
    );

    bench_step_group(
        c,
        "Step/DecompressRowsData.process",
        &prepared_cases,
        &DECOMPRESS_ROWS_IMPLS,
        DecompressRowsImpl::label,
        |case| {
            (
                case.rows_input_seed.0.clone(),
                case.rows_input_seed.1.clone(),
            )
        },
        |implementation, input, _case| implementation.process(input),
    );
}

criterion_group!(benches, benchmark_filter_steps);
criterion_main!(benches);
