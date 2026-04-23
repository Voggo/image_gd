use std::hint::black_box;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};

use entro_gd::compression::preprocessor::DEFAULT_ALIGN_ROWS_TO_WORD;
use entro_gd::data_loader::{CsvDataLoader, DataLoader, FloatStorage};
use entro_gd::prelude::*;
use entro_gd::{
    BaseSelectionContext, BitDataSet, CompressedData, CondensedSamples, Dataset, EntroGdError,
    EntropyScoredContext, FeatureSpec,
};

#[allow(unused)]
#[derive(Clone, Copy)]
enum InferFeatureSpecsImpl {
    Current,
}

impl InferFeatureSpecsImpl {
    fn label(self) -> &'static str {
        match self {
            InferFeatureSpecsImpl::Current => "current",
        }
    }

    fn process(self, input: Dataset) -> Result<(Dataset, Vec<FeatureSpec>), EntroGdError> {
        match self {
            InferFeatureSpecsImpl::Current => InferFeatureSpecs {
                options: PreprocessOptions::default(),
            }
            .process(input),
        }
    }
}

#[allow(unused)]
#[derive(Clone, Copy)]
enum BuildBitDataSetImpl {
    Current,
}

impl BuildBitDataSetImpl {
    fn label(self) -> &'static str {
        match self {
            BuildBitDataSetImpl::Current => "current",
        }
    }

    fn process(self, input: (Dataset, Vec<FeatureSpec>)) -> Result<BitDataSet, EntroGdError> {
        match self {
            BuildBitDataSetImpl::Current => BuildBitDataSet::default().process(input),
        }
    }
}

#[allow(unused)]
#[derive(Clone, Copy)]
enum BuildImageBitDataSetImpl {
    Current,
}

impl BuildImageBitDataSetImpl {
    fn label(self) -> &'static str {
        match self {
            BuildImageBitDataSetImpl::Current => "current",
        }
    }

    fn process(
        self,
        input: PathBuf,
        image_input: StepBenchInput,
    ) -> Result<BitDataSet, EntroGdError> {
        match (self, image_input) {
            (
                BuildImageBitDataSetImpl::Current,
                StepBenchInput::Image {
                    colorspace,
                    color_model,
                    pixel_grouping,
                    grouping_transform,
                },
            ) => BuildImageBitDataSet {
                colorspace,
                color_model,
                pixel_grouping,
                grouping_transform,
                pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
            }
            .process(input),
            (BuildImageBitDataSetImpl::Current, StepBenchInput::Csv { .. }) => {
                panic!("BuildImageBitDataSet.process called with csv case")
            }
        }
    }
}

#[allow(unused)]
#[derive(Clone, Copy)]
enum EntropyImpl {
    Naive,
    Batched,
    StrideSampledNaive,
    StrideSampledBatched,
}

impl EntropyImpl {
    fn label(self) -> &'static str {
        match self {
            EntropyImpl::Naive => "naive",
            EntropyImpl::Batched => "batched",
            EntropyImpl::StrideSampledNaive => "stride_sampled_naive",
            EntropyImpl::StrideSampledBatched => "stride_sampled_batched",
        }
    }

    fn entropy(self, bit_data: BitDataSet) -> Result<EntropyScoredContext, EntroGdError> {
        match self {
            EntropyImpl::Naive => EntropyNaive {}.process(bit_data),
            EntropyImpl::Batched => EntropyBatched {}.process(bit_data),
            EntropyImpl::StrideSampledNaive => {
                EntropyStrideSampled { skip_rows: 1 }.process(bit_data)
            }
            EntropyImpl::StrideSampledBatched => {
                EntropyStrideSampledBatched { skip_rows: 1 }.process(bit_data)
            }
        }
    }
}

#[allow(unused)]
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
        input: EntropyScoredContext,
        m_max: usize,
    ) -> Result<EntropyScoredContext, EntroGdError> {
        match self {
            GenCondensedImpl::Current => GenCondensedSamples { m_max }.process(input),
        }
    }
}

#[allow(unused)]
#[derive(Clone, Copy)]
enum SelectBasesImpl {
    Current,
    ProfileAllBits(BaseBitImpl),
    DebugNoCsv,
    Optimized(BaseBitImpl),
}

impl SelectBasesImpl {
    fn label(self) -> &'static str {
        match self {
            SelectBasesImpl::Current => "current",
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
            SelectBasesImpl::DebugNoCsv => "debug_no_csv",
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
        }
    }

    fn process(
        self,
        input: EntropyScoredContext,
        patience: usize,
    ) -> Result<BaseSelectionContext, EntroGdError> {
        match self {
            SelectBasesImpl::Current => SelectBases { patience }.process(input),
            SelectBasesImpl::ProfileAllBits(base_bit_impl) => SelectBasesProfileAllBits {
                split_into_batches: patience,
                base_bit_impl,
            }
            .process(input),
            SelectBasesImpl::DebugNoCsv => SelectBasesDebug {
                patience,
                debug_csv_paths: Mutex::new(Vec::new()),
            }
            .process(input),
            SelectBasesImpl::Optimized(base_bit_impl) => SelectBasesOptimized {
                patience,
                base_bit_impl,
            }
            .process(input),
        }
    }
}

#[allow(unused)]
#[derive(Clone, Copy)]
enum EncodeImpl {
    Naive,
    Optimized,
    Rle,
    FusedDictionary,
    HuffmanBaseIdOnly,
}

impl EncodeImpl {
    fn label(self) -> &'static str {
        match self {
            EncodeImpl::Naive => "naive",
            EncodeImpl::Optimized => "optimized",
            EncodeImpl::Rle => "rle",
            EncodeImpl::FusedDictionary => "fused_dictionary",
            EncodeImpl::HuffmanBaseIdOnly => "huffman_base_id_only",
        }
    }

    fn process(self, input: BaseSelectionContext) -> Result<CompressedData, EntroGdError> {
        match self {
            EncodeImpl::FusedDictionary => EncodeDataFusedDictionary {}.process(input),
            _ => {
                let base_table_ctx = BuildBaseTable {}.process(input)?;
                match self {
                    EncodeImpl::Naive => EncodeData {}.process(base_table_ctx),
                    EncodeImpl::Optimized => EncodeDataOptimized {}.process(base_table_ctx),
                    EncodeImpl::Rle => EncodeDataRLE {}.process(base_table_ctx),
                    EncodeImpl::HuffmanBaseIdOnly => EncodeDataHuffman {}.process(base_table_ctx),
                    EncodeImpl::FusedDictionary => unreachable!(),
                }
            }
        }
    }
}

fn id_bits_needed(num_bases: usize) -> usize {
    if num_bases <= 1 {
        1
    } else {
        usize::BITS as usize - (num_bases - 1).leading_zeros() as usize
    }
}

fn huffman_symbol_width(input: &BaseSelectionContext) -> usize {
    let num_bases = input.base_bits.get_num_bases();
    id_bits_needed(num_bases)
}

#[allow(unused)]
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

#[allow(unused)]
#[derive(Clone, Copy)]
enum SaveIgdImpl {
    Current,
}

impl SaveIgdImpl {
    fn label(self) -> &'static str {
        match self {
            SaveIgdImpl::Current => "current",
        }
    }

    fn process(self, input: CompressedData, output_path: PathBuf) -> Result<PathBuf, EntroGdError> {
        match self {
            SaveIgdImpl::Current => SaveIgdFile { output_path }.process(input),
        }
    }
}

#[allow(unused)]
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

#[allow(unused)]
#[derive(Clone, Copy)]
enum LoadIgdImpl {
    Current,
}

impl LoadIgdImpl {
    fn label(self) -> &'static str {
        match self {
            LoadIgdImpl::Current => "current",
        }
    }

    fn process(self, input: PathBuf) -> Result<CompressedData, EntroGdError> {
        match self {
            LoadIgdImpl::Current => LoadIgdFile {}.process(input),
        }
    }
}

#[allow(unused)]
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

#[allow(unused)]
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

#[allow(unused)]
#[derive(Clone, Copy)]
enum DecompressAnalyticsImpl {
    Current,
}

impl DecompressAnalyticsImpl {
    fn label(self) -> &'static str {
        match self {
            DecompressAnalyticsImpl::Current => "current",
        }
    }

    fn process(self, input: CompressedData) -> Result<Option<CondensedSamples>, EntroGdError> {
        match self {
            DecompressAnalyticsImpl::Current => DecompressAnalytics {}.process(input),
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

    fn image_with_transform(
        data_file_path: &'static str,
        color_model: ImageColorModel,
        pixel_grouping: u32,
        grouping_transform: ImageGroupingTransform,
        m_max: usize,
        patience: usize,
    ) -> Self {
        Self {
            data_file_path,
            input: StepBenchInput::Image {
                colorspace: ImageColorSpace::SrgbWithLinearAlpha,
                color_model,
                pixel_grouping,
                grouping_transform,
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
            20,
        ),
        StepBenchCase::image_with_transform(
            "data/images/kodim10.png",
            ImageColorModel::YCoCgR,
            4,
            ImageGroupingTransform::ForFirstPixel,
            0,
            20,
        ),
    ]
}

const INFER_FEATURE_SPECS_IMPLS: [InferFeatureSpecsImpl; 1] = [InferFeatureSpecsImpl::Current];
const BUILD_BIT_DATA_SET_IMPLS: [BuildBitDataSetImpl; 1] = [BuildBitDataSetImpl::Current];
const BUILD_IMAGE_BIT_DATA_SET_IMPLS: [BuildImageBitDataSetImpl; 1] =
    [BuildImageBitDataSetImpl::Current];
const ENTROPY_IMPLS: [EntropyImpl; 4] = [
    EntropyImpl::Naive,
    EntropyImpl::Batched,
    EntropyImpl::StrideSampledNaive,
    EntropyImpl::StrideSampledBatched,
];
const GEN_CONDENSED_IMPLS: [GenCondensedImpl; 1] = [GenCondensedImpl::Current];
const SELECT_BASES_IMPLS: [SelectBasesImpl; 9] = [
    SelectBasesImpl::Current,
    SelectBasesImpl::ProfileAllBits(BaseBitImpl::Naive),
    SelectBasesImpl::ProfileAllBits(BaseBitImpl::BatchGroups),
    SelectBasesImpl::ProfileAllBits(BaseBitImpl::IncSignatureGroups),
    SelectBasesImpl::ProfileAllBits(BaseBitImpl::HyperLogLogCount),
    SelectBasesImpl::Optimized(BaseBitImpl::Naive),
    SelectBasesImpl::Optimized(BaseBitImpl::BatchGroups),
    SelectBasesImpl::Optimized(BaseBitImpl::IncSignatureGroups),
    SelectBasesImpl::Optimized(BaseBitImpl::HyperLogLogCount),
];
const ENCODE_IMPLS: [EncodeImpl; 5] = [
    EncodeImpl::Naive,
    EncodeImpl::Optimized,
    EncodeImpl::Rle,
    EncodeImpl::FusedDictionary,
    EncodeImpl::HuffmanBaseIdOnly,
];
const SAVE_IMPLS: [SaveImpl; 1] = [SaveImpl::Current];
const SAVE_IGD_IMPLS: [SaveIgdImpl; 1] = [SaveIgdImpl::Current];
const LOAD_IMPLS: [LoadImpl; 1] = [LoadImpl::Current];
const LOAD_IGD_IMPLS: [LoadIgdImpl; 1] = [LoadIgdImpl::Current];
const DECOMPRESS_ROWS_IMPLS: [DecompressRowsImpl; 1] = [DecompressRowsImpl::Current];
const DECOMPRESS_FILE_IMPLS: [DecompressFileImpl; 1] = [DecompressFileImpl::Current];
const DECOMPRESS_ANALYTICS_IMPLS: [DecompressAnalyticsImpl; 1] = [DecompressAnalyticsImpl::Current];

fn dataset_label(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn case_label(case: StepBenchCase) -> String {
    let input_label = match case.input {
        StepBenchInput::Csv { .. } => "csv".to_string(),
        StepBenchInput::Image {
            color_model,
            pixel_grouping,
            grouping_transform,
            ..
        } => format!(
            "img-{:?}-g{}-{:?}",
            color_model, pixel_grouping, grouping_transform
        ),
    };
    format!(
        "{}-{}-m{}-p{}",
        dataset_label(case.data_file_path),
        input_label,
        case.m_max,
        case.patience
    )
}

fn output_egd_path(case: StepBenchCase) -> PathBuf {
    PathBuf::from(format!(
        "target/bench-artifacts/{}-m{}-p{}.egd",
        dataset_label(case.data_file_path),
        case.m_max,
        case.patience,
    ))
}

fn output_igd_path(case: StepBenchCase) -> PathBuf {
    PathBuf::from(format!(
        "target/bench-artifacts/{}-m{}-p{}.igd",
        dataset_label(case.data_file_path),
        case.m_max,
        case.patience,
    ))
}

fn build_loader(case: StepBenchCase) -> CsvDataLoader {
    let float_storage = match case.input {
        StepBenchInput::Csv { float_storage } => float_storage,
        StepBenchInput::Image { .. } => {
            panic!(
                "build_loader called for image case: {}",
                case.data_file_path
            )
        }
    };
    CsvDataLoader::new(true).with_float_storage(float_storage)
}

fn build_csv_dataset_seed(case: StepBenchCase) -> Dataset {
    match case.input {
        StepBenchInput::Csv { .. } => {
            let loader = build_loader(case);
            loader.load(case.data_file_path).unwrap().dataset
        }
        StepBenchInput::Image { .. } => {
            panic!(
                "build_csv_dataset_seed called for image case: {}",
                case.data_file_path
            )
        }
    }
}

fn build_bit_data_seed(case: StepBenchCase, dataset_seed: Option<&Dataset>) -> BitDataSet {
    match case.input {
        StepBenchInput::Csv { .. } => BitDataSet::from_dataset(
            dataset_seed
                .expect("dataset seed should be available for csv case when building bit data"),
        )
        .unwrap(),
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
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        }
        .process(PathBuf::from(case.data_file_path))
        .unwrap(),
    }
}

fn infer_feature_specs_seed(dataset: &Dataset) -> Vec<FeatureSpec> {
    InferFeatureSpecs {
        options: PreprocessOptions::default(),
    }
    .process(dataset.clone())
    .unwrap()
    .1
}

struct CompressedDataWrapper {
    normal: CompressedData,
    rle: CompressedData,
    huffman_base_id_only: CompressedData,
}

struct CompressedRowsWrapper {
    normal: (Arc<CompressedData>, Vec<usize>),
    rle: (Arc<CompressedData>, Vec<usize>),
    huffman_base_id_only: (Arc<CompressedData>, Vec<usize>),
}

struct PreparedCase {
    name: String,
    data_file_path: &'static str,
    input: StepBenchInput,
    source_size: u64,
    m_max: usize,
    patience: usize,
    dataset_seed: Option<Dataset>,
    inferred_feature_specs_seed: Option<Vec<FeatureSpec>>,
    bit_data_seed: BitDataSet,
    entropy_seed: EntropyScoredContext,
    condensed_seed: EntropyScoredContext,
    huffman_symbol_width_seed: usize,
    compressed_seeds: CompressedDataWrapper,
    loaded_compressed_seeds: CompressedDataWrapper,
    egd_path: PathBuf,
    igd_path: Option<PathBuf>,
    rows_input_seeds: CompressedRowsWrapper,
}

impl PreparedCase {
    fn is_csv(&self) -> bool {
        matches!(self.input, StepBenchInput::Csv { .. })
    }

    fn is_image(&self) -> bool {
        matches!(self.input, StepBenchInput::Image { .. })
    }

    fn supports_huffman_encoding(&self) -> bool {
        self.huffman_symbol_width_seed <= 64
    }
}

fn encode_impl_supported(implementation: EncodeImpl, case: &PreparedCase) -> bool {
    match implementation {
        EncodeImpl::HuffmanBaseIdOnly => case.supports_huffman_encoding(),
        _ => true,
    }
}

fn prepare_case(case: StepBenchCase) -> PreparedCase {
    let dataset_seed = match case.input {
        StepBenchInput::Csv { .. } => Some(build_csv_dataset_seed(case)),
        StepBenchInput::Image { .. } => None,
    };
    let inferred_feature_specs_seed = dataset_seed.as_ref().map(infer_feature_specs_seed);

    let bit_data_seed = build_bit_data_seed(case, dataset_seed.as_ref());
    let source_size = (bit_data_seed.data.num_rows * bit_data_seed.data.chunk_size / 8) as u64;
    let entropy_seed = EntropyBatched {}.process(bit_data_seed.clone()).unwrap();
    let condensed_seed = GenCondensedSamples { m_max: case.m_max }
        .process(entropy_seed.clone())
        .unwrap();
    let selected_seed = SelectBasesOptimized {
        patience: case.patience,
        base_bit_impl: BaseBitImpl::BatchGroups,
    }
    .process(condensed_seed.clone())
    .unwrap();
    let huffman_symbol_width_seed = huffman_symbol_width(&selected_seed);
    let base_table_ctx_seed = BuildBaseTable {}.process(selected_seed).unwrap();
    let compressed_seed = EncodeDataOptimized {}
        .process(base_table_ctx_seed.clone())
        .unwrap();
    let compressed_seed_rle = EncodeDataRLE {}
        .process(base_table_ctx_seed.clone())
        .unwrap();
    let compressed_seed_huffman = EncodeDataHuffman {}
        .process(base_table_ctx_seed.clone())
        .unwrap();
    let compressed_seeds = CompressedDataWrapper {
        normal: compressed_seed.clone(),
        rle: compressed_seed_rle.clone(),
        huffman_base_id_only: compressed_seed_huffman.clone(),
    };

    let egd_path = output_egd_path(case);
    let _saved_once = SaveEgdFile {
        output_path: egd_path.clone(),
    }
    .process(compressed_seed.clone())
    .unwrap();
    let loaded_compressed_seed = LoadEgdFile {}.process(egd_path.clone()).unwrap();
    let _saved_once = SaveEgdFile {
        output_path: egd_path.clone(),
    }
    .process(compressed_seed_rle.clone())
    .unwrap();
    let loaded_compressed_seed_rle = LoadEgdFile {}.process(egd_path.clone()).unwrap();
    let _saved_once = SaveEgdFile {
        output_path: egd_path.clone(),
    }
    .process(compressed_seed_huffman.clone())
    .unwrap();
    let loaded_compressed_seed_huffman = LoadEgdFile {}.process(egd_path.clone()).unwrap();
    let loaded_compressed_seeds = CompressedDataWrapper {
        normal: loaded_compressed_seed.clone(),
        rle: loaded_compressed_seed_rle.clone(),
        huffman_base_id_only: loaded_compressed_seed_huffman.clone(),
    };
    let igd_path = match case.input {
        StepBenchInput::Image { .. } => {
            let path = output_igd_path(case);
            let _saved_once = SaveIgdFile {
                output_path: path.clone(),
            }
            .process(compressed_seed.clone())
            .unwrap();
            Some(path)
        }
        StepBenchInput::Csv { .. } => None,
    };

    let rows_seed = (0..bit_data_seed.data.num_rows).collect::<Vec<usize>>();
    let rows_input_seed = (Arc::new(loaded_compressed_seed.clone()), rows_seed.clone());
    let rows_input_rle_seed = (
        Arc::new(loaded_compressed_seed_rle.clone()),
        rows_seed.clone(),
    );
    let rows_input_huffman_seed = (
        Arc::new(loaded_compressed_seed_huffman.clone()),
        rows_seed.clone(),
    );
    let rows_input_seeds = CompressedRowsWrapper {
        normal: rows_input_seed,
        rle: rows_input_rle_seed,
        huffman_base_id_only: rows_input_huffman_seed,
    };

    PreparedCase {
        name: case_label(case),
        data_file_path: case.data_file_path,
        input: case.input,
        source_size,
        m_max: case.m_max,
        patience: case.patience,
        dataset_seed,
        inferred_feature_specs_seed,
        bit_data_seed,
        entropy_seed,
        condensed_seed,
        huffman_symbol_width_seed,
        compressed_seeds,
        loaded_compressed_seeds,
        egd_path,
        igd_path,
        rows_input_seeds,
    }
}

fn bench_step_group<ImplType, Input, Output, LabelFn, InputFn, RunFn, CaseFilter, ImplCaseFilter>(
    c: &mut Criterion,
    group_name: &str,
    prepared_cases: &[PreparedCase],
    implementations: &[ImplType],
    impl_label: LabelFn,
    make_input: InputFn,
    run_impl: RunFn,
    case_filter: CaseFilter,
    impl_case_filter: ImplCaseFilter,
) where
    ImplType: Copy,
    LabelFn: Fn(ImplType) -> &'static str + Copy,
    InputFn: Fn(&PreparedCase) -> Input + Copy,
    RunFn: Fn(ImplType, Input, &PreparedCase) -> Result<Output, EntroGdError> + Copy,
    CaseFilter: Fn(&PreparedCase) -> bool + Copy,
    ImplCaseFilter: Fn(ImplType, &PreparedCase) -> bool + Copy,
{
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10).warm_up_time(Duration::from_secs(1));
    for case in prepared_cases {
        if !case_filter(case) {
            continue;
        }
        group.throughput(Throughput::Bytes(case.source_size));
        for implementation in implementations.iter().copied() {
            if !impl_case_filter(implementation, case) {
                continue;
            }
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
        "Step/InferFeatureSpecs.process",
        &prepared_cases,
        &INFER_FEATURE_SPECS_IMPLS,
        InferFeatureSpecsImpl::label,
        |case| {
            case.dataset_seed
                .clone()
                .expect("dataset seed should be available for csv cases")
        },
        |implementation, input, _case| implementation.process(input),
        PreparedCase::is_csv,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/BuildBitDataSet.process",
        &prepared_cases,
        &BUILD_BIT_DATA_SET_IMPLS,
        BuildBitDataSetImpl::label,
        |case| {
            (
                case.dataset_seed
                    .clone()
                    .expect("dataset seed should be available for csv cases"),
                case.inferred_feature_specs_seed
                    .clone()
                    .expect("inferred feature specs should be available for csv cases"),
            )
        },
        |implementation, input, _case| implementation.process(input),
        PreparedCase::is_csv,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/BuildImageBitDataSet.process",
        &prepared_cases,
        &BUILD_IMAGE_BIT_DATA_SET_IMPLS,
        BuildImageBitDataSetImpl::label,
        |case| PathBuf::from(case.data_file_path),
        |implementation, input, case| implementation.process(input, case.input),
        PreparedCase::is_image,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/Entropy.process",
        &prepared_cases,
        &ENTROPY_IMPLS,
        EntropyImpl::label,
        |case| case.bit_data_seed.clone(),
        |implementation, input, _case| implementation.entropy(input),
        |_| true,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/GenCondensedSamples.process",
        &prepared_cases,
        &GEN_CONDENSED_IMPLS,
        GenCondensedImpl::label,
        |case| case.entropy_seed.clone(),
        |implementation, input, case| implementation.process(input, case.m_max),
        |_| true,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/SelectBases.process",
        &prepared_cases,
        &SELECT_BASES_IMPLS,
        SelectBasesImpl::label,
        |case| case.condensed_seed.clone(),
        |implementation, input, case| implementation.process(input, case.patience),
        |_| true,
        |_, _| true,
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
        |_| true,
        encode_impl_supported,
    );

    bench_step_group(
        c,
        "Step/SaveEgdFileNormalEncode.process",
        &prepared_cases,
        &SAVE_IMPLS,
        SaveImpl::label,
        |case| case.compressed_seeds.normal.clone(),
        |implementation, input, case| implementation.process(input, case.egd_path.clone()),
        |_| true,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/SaveEgdFileRleEncode.process",
        &prepared_cases,
        &SAVE_IMPLS,
        SaveImpl::label,
        |case| case.compressed_seeds.rle.clone(),
        |implementation, input, case| implementation.process(input, case.egd_path.clone()),
        |_| true,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/SaveEgdFileHuffmanBaseIdOnlyEncode.process",
        &prepared_cases,
        &SAVE_IMPLS,
        SaveImpl::label,
        |case| case.compressed_seeds.huffman_base_id_only.clone(),
        |implementation, input, case| implementation.process(input, case.egd_path.clone()),
        |_| true,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/LoadEgdFile.process",
        &prepared_cases,
        &LOAD_IMPLS,
        LoadImpl::label,
        |case| case.egd_path.clone(),
        |implementation, input, _case| implementation.process(input),
        |_| true,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/SaveIgdFileNormalEncode.process",
        &prepared_cases,
        &SAVE_IGD_IMPLS,
        SaveIgdImpl::label,
        |case| case.compressed_seeds.normal.clone(),
        |implementation, input, case| {
            implementation.process(
                input,
                case.igd_path
                    .clone()
                    .expect("igd path should be available for image cases"),
            )
        },
        PreparedCase::is_image,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/SaveIgdFileRleEncode.process",
        &prepared_cases,
        &SAVE_IGD_IMPLS,
        SaveIgdImpl::label,
        |case| case.compressed_seeds.rle.clone(),
        |implementation, input, case| {
            implementation.process(
                input,
                case.igd_path
                    .clone()
                    .expect("igd path should be available for image cases"),
            )
        },
        PreparedCase::is_image,
        |_, _| true,
    );
    bench_step_group(
        c,
        "Step/SaveIgdFileHuffmanBaseIdOnlyEncode.process",
        &prepared_cases,
        &SAVE_IGD_IMPLS,
        SaveIgdImpl::label,
        |case| case.compressed_seeds.huffman_base_id_only.clone(),
        |implementation, input, case| {
            implementation.process(
                input,
                case.igd_path
                    .clone()
                    .expect("igd path should be available for image cases"),
            )
        },
        PreparedCase::is_image,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/LoadIgdFile.process",
        &prepared_cases,
        &LOAD_IGD_IMPLS,
        LoadIgdImpl::label,
        |case| {
            case.igd_path
                .clone()
                .expect("igd path should be available for image cases")
        },
        |implementation, input, _case| implementation.process(input),
        PreparedCase::is_image,
        |_, _| true,
    );

    // Needs refactoring perhaps...
    bench_step_group(
        c,
        "Step/DecompressRowsDataNormalEncode.process",
        &prepared_cases,
        &DECOMPRESS_ROWS_IMPLS,
        DecompressRowsImpl::label,
        |case| {
            (
                case.rows_input_seeds.normal.0.clone(),
                case.rows_input_seeds.normal.1.clone(),
            )
        },
        |implementation, input, _case| implementation.process(input),
        |_| true,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/DecompressRowsDataRleEncode.process",
        &prepared_cases,
        &DECOMPRESS_ROWS_IMPLS,
        DecompressRowsImpl::label,
        |case| {
            (
                case.rows_input_seeds.rle.0.clone(),
                case.rows_input_seeds.rle.1.clone(),
            )
        },
        |implementation, input, _case| implementation.process(input),
        |_| true,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/DecompressRowsDataHuffmanBaseIdOnlyEncode.process",
        &prepared_cases,
        &DECOMPRESS_ROWS_IMPLS,
        DecompressRowsImpl::label,
        |case| {
            (
                case.rows_input_seeds.huffman_base_id_only.0.clone(),
                case.rows_input_seeds.huffman_base_id_only.1.clone(),
            )
        },
        |implementation, input, _case| implementation.process(input),
        |_| true,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/DecompressFileDataNormalEncode.process",
        &prepared_cases,
        &DECOMPRESS_FILE_IMPLS,
        DecompressFileImpl::label,
        |case| case.loaded_compressed_seeds.normal.clone(),
        |implementation, input, _case| implementation.process(input),
        |_| true,
        |_, _| true,
    );
    bench_step_group(
        c,
        "Step/DecompressFileDataRleEncode.process",
        &prepared_cases,
        &DECOMPRESS_FILE_IMPLS,
        DecompressFileImpl::label,
        |case| case.loaded_compressed_seeds.rle.clone(),
        |implementation, input, _case| implementation.process(input),
        |_| true,
        |_, _| true,
    );
    bench_step_group(
        c,
        "Step/DecompressFileDataHuffmanBaseIdOnlyEncode.process",
        &prepared_cases,
        &DECOMPRESS_FILE_IMPLS,
        DecompressFileImpl::label,
        |case| case.loaded_compressed_seeds.huffman_base_id_only.clone(),
        |implementation, input, _case| implementation.process(input),
        |_| true,
        |_, _| true,
    );

    bench_step_group(
        c,
        "Step/DecompressAnalytics.process",
        &prepared_cases,
        &DECOMPRESS_ANALYTICS_IMPLS,
        DecompressAnalyticsImpl::label,
        |case| case.compressed_seeds.normal.clone(),
        |implementation, input, _case| implementation.process(input),
        |_| true,
        |_, _| true,
    );
}

criterion_group!(benches, benchmark_filter_steps);
criterion_main!(benches);
