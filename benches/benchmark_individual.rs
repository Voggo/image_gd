use std::path::PathBuf;
use std::sync::Arc;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use std::hint::black_box;

use entro_gd::data_loader::{CsvDataLoader, DataLoader, FloatStorage};
use entro_gd::prelude::*;
use entro_gd::{BitDataSet, CompressedData, DecompressRowsData, EntroGdError};

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
    Optimized(BaseBitImpl),
    ProfileAllBits(BaseBitImpl),
}

impl SelectBasesImpl {
    fn label(self) -> &'static str {
        match self {
            SelectBasesImpl::Naive => "naive",
            SelectBasesImpl::Optimized(BaseBitImpl::Naive) => "optimized_naive",
            SelectBasesImpl::Optimized(BaseBitImpl::BatchGroups) => "optimized_batch_groups",
            SelectBasesImpl::Optimized(BaseBitImpl::IncSignatureGroups) => {
                "optimized_inc_signature_groups"
            }
            SelectBasesImpl::Optimized(BaseBitImpl::SignatureGroups) => {
                "optimized_signature_groups"
            }
            SelectBasesImpl::Optimized(BaseBitImpl::HyperLogLogCount) => {
                "optimized_hyper_log_log_count"
            }
            SelectBasesImpl::ProfileAllBits(BaseBitImpl::Naive) => "profile_all_bits_naive",
            SelectBasesImpl::ProfileAllBits(BaseBitImpl::BatchGroups) => {
                "profile_all_bits_batch_groups"
            }
            SelectBasesImpl::ProfileAllBits(BaseBitImpl::IncSignatureGroups) => {
                "profile_all_bits_inc_signature_groups"
            }
            SelectBasesImpl::ProfileAllBits(BaseBitImpl::SignatureGroups) => {
                "profile_all_bits_signature_groups"
            }
            SelectBasesImpl::ProfileAllBits(BaseBitImpl::HyperLogLogCount) => {
                "profile_all_bits_hyper_log_log_count"
            }
        }
    }

    fn process(
        self,
        input: (BitDataSet, Vec<(usize, f64)>),
        patience: usize,
    ) -> Result<(BitDataSet, DynBaseBit), EntroGdError> {
        match self {
            SelectBasesImpl::Naive => SelectBases { patience }.process(input),
            SelectBasesImpl::Optimized(base_bit_impl) => SelectBasesOptimized {
                patience,
                base_bit_impl,
            }
            .process(input),
            SelectBasesImpl::ProfileAllBits(base_bit_impl) => SelectBasesProfileAllBits {
                split_into_batches: patience,
                base_bit_impl,
            }
            .process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum EncodeImpl {
    Naive,
    Optimized,
    FusedDictionary,
}

impl EncodeImpl {
    fn label(self) -> &'static str {
        match self {
            EncodeImpl::Naive => "naive",
            EncodeImpl::Optimized => "optimized",
            EncodeImpl::FusedDictionary => "fused_dictionary",
        }
    }

    fn process(self, input: (BitDataSet, DynBaseBit)) -> Result<CompressedData, EntroGdError> {
        match self {
            EncodeImpl::Naive => EncodeData {}.process(input),
            EncodeImpl::Optimized => EncodeDataOptimized {}.process(input),
            EncodeImpl::FusedDictionary => EncodeDataFusedDictionary {}.process(input),
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
            DecompressRowsImpl::Current => DecompressRowsData {}.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum DecompressFileImpl {
    Current,
}

impl DecompressFileImpl {
    fn label(self) -> &'static str {
        match self {
            DecompressFileImpl::Current => "current",
        }
    }

    fn process(self, input: CompressedData) -> Result<BitDataSet, EntroGdError> {
        match self {
            DecompressFileImpl::Current => DecompressFileData {}.process(input),
        }
    }
}

#[derive(Clone, Copy)]
struct StepBenchCase {
    data_file_path: &'static str,
    input: StepBenchInput,
    m_max: usize,
    patience: usize,
}

#[derive(Clone, Copy)]
enum StepBenchInput {
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

impl StepBenchCase {
    fn csv(
        data_file_path: &'static str,
        float_storage: FloatStorage,
        m_max: usize,
        patience: usize,
    ) -> Self {
        Self {
            data_file_path,
            input: StepBenchInput::Csv { float_storage },
            m_max,
            patience,
        }
    }

    fn image(
        data_file_path: &'static str,
        m_max: usize,
        patience: usize,
    ) -> Self {
        Self {
            data_file_path,
            input: StepBenchInput::Image {
                colorspace: ImageColorSpace::SrgbWithLinearAlpha,
                color_model: ImageColorModel::YCoCgR,
                pixel_grouping: 3,
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
            },
            m_max,
            patience,
        }
    }
}

fn step_bench_cases() -> Vec<StepBenchCase> {
    vec![
        StepBenchCase::csv(
            "data/tabular/aarhus-citylab_duplicated.csv",
            FloatStorage::F32,
            50,
            10,
        ),
        StepBenchCase::image("data/images/kodim10.png", 0, 10),
    ]
}

const ENTROPY_IMPLS: [EntropyImpl; 2] = [EntropyImpl::Naive, EntropyImpl::Optimized];
const GEN_CONDENSED_IMPLS: [GenCondensedImpl; 1] = [GenCondensedImpl::Current];
const SELECT_BASES_IMPLS: [SelectBasesImpl; 4] = [
    // SelectBasesImpl::Naive,
    // SelectBasesImpl::Optimized(BaseBitImpl::BatchGroups),
    // SelectBasesImpl::Optimized(BaseBitImpl::IncSignatureGroups),
    // SelectBasesImpl::Optimized(BaseBitImpl::SignatureGroups),
    // SelectBasesImpl::Optimized(BaseBitImpl::HyperLogLogCount),
    SelectBasesImpl::ProfileAllBits(BaseBitImpl::Naive),
    SelectBasesImpl::ProfileAllBits(BaseBitImpl::BatchGroups),
    SelectBasesImpl::ProfileAllBits(BaseBitImpl::IncSignatureGroups),
    // SelectBasesImpl::ProfileAllBits(BaseBitImpl::SignatureGroups),
    SelectBasesImpl::ProfileAllBits(BaseBitImpl::HyperLogLogCount),
];
const ENCODE_IMPLS: [EncodeImpl; 3] = [
    EncodeImpl::Naive,
    EncodeImpl::Optimized,
    EncodeImpl::FusedDictionary,
];
const SAVE_IMPLS: [SaveImpl; 1] = [SaveImpl::Current];
const LOAD_IMPLS: [LoadImpl; 1] = [LoadImpl::Current];
const DECOMPRESS_ROWS_IMPLS: [DecompressRowsImpl; 1] = [DecompressRowsImpl::Current];
const DECOMPRESS_FILE_IMPLS: [DecompressFileImpl; 1] = [DecompressFileImpl::Current];

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
        "target/bench-artifacts/{}-m{}-p{}.egd",
        dataset_label(case.data_file_path),
        case.m_max,
        case.patience,
    ))
}

fn build_loader(case: StepBenchCase) -> CsvDataLoader {
    let float_storage = match case.input {
        StepBenchInput::Csv { float_storage } => float_storage,
        StepBenchInput::Image { .. } => {
            panic!("build_loader called for image case: {}", case.data_file_path)
        }
    };
    CsvDataLoader::new(true).with_float_storage(float_storage)
}

fn build_bit_data_seed(case: StepBenchCase) -> BitDataSet {
    match case.input {
        StepBenchInput::Csv { .. } => {
            let loader = build_loader(case);
            let loaded = loader.load(case.data_file_path).unwrap();
            BitDataSet::from_dataset(&loaded.dataset).unwrap()
        }
        StepBenchInput::Image {
            colorspace,
            color_model,
            pixel_grouping,
            grouping_transform,
        } => BuildImageBitDataSet {
            colorspace,
            color_model,
            pixel_grouping,
            grouping_transform,
        }
        .process(PathBuf::from(case.data_file_path))
        .unwrap(),
    }
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
    loaded_compressed_seed: CompressedData,
    egd_path: PathBuf,
    rows_input_seed: (Arc<CompressedData>, Vec<usize>),
}

fn prepare_case(case: StepBenchCase) -> PreparedCase {
    let source_size = std::fs::metadata(case.data_file_path).unwrap().len() as u64;

    let bit_data_seed = build_bit_data_seed(case);
    let entropy_seed = EntropyOptimized {}.process(bit_data_seed.clone()).unwrap();
    let condensed_seed = GenCondensedSamples { m_max: case.m_max }
        .process(entropy_seed.clone())
        .unwrap();
    let selected_seed = SelectBasesOptimized {
        patience: case.patience,
        base_bit_impl: BaseBitImpl::BatchGroups,
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
    let rows_seed = (0..bit_data_seed.data.num_rows).collect::<Vec<usize>>();
    let rows_input_seed = (Arc::new(loaded_compressed_seed.clone()), rows_seed);

    PreparedCase {
        name: case_label(case),
        source_size,
        m_max: case.m_max,
        patience: case.patience,
        bit_data_seed,
        entropy_seed,
        condensed_seed,
        compressed_seed,
        loaded_compressed_seed,
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
            SelectBasesOptimized {
                patience: case.patience,
                base_bit_impl: BaseBitImpl::BatchGroups,
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

    bench_step_group(
        c,
        "Step/DecompressFileData.process",
        &prepared_cases,
        &DECOMPRESS_FILE_IMPLS,
        DecompressFileImpl::label,
        |case| case.loaded_compressed_seed.clone(),
        |implementation, input, _case| implementation.process(input),
    );
}

criterion_group!(benches, benchmark_filter_steps);
criterion_main!(benches);
