use std::hint::black_box;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use entro_gd::compression::preprocessor::DEFAULT_ALIGN_ROWS_TO_WORD;
use entro_gd::prelude::*;
use entro_gd::{
    BaseSelectionContext, BitDataSet, CompressedData, DecompressRandomAccessHandle, EntroGdError,
    EntropyScoredContext, PreEncodeContext,
};

// ── CSV collection ────────────────────────────────────────────────────────────

struct BenchRecord {
    stage: &'static str,
    impl_name: String,
    file: String,
    sample_time_ns: u128,
    source_bytes: u64,
}

static BENCH_RECORDS: LazyLock<Mutex<Vec<BenchRecord>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

// ── Harness config ────────────────────────────────────────────────────────────

const N_RUNS: usize = 3;

// ── Canonical pipeline config ─────────────────────────────────────────────────

const CANONICAL_M_MAX: usize = 0;
const CANONICAL_PATIENCE: usize = 10;
const CANONICAL_ENTROPY_THRESHOLD: f64 = 0.70;
const CANONICAL_BASE_BIT_IMPL: BaseBitImpl = BaseBitImpl::BatchGroups;
const IMAGE_DIR: &str = "data/bench_datasets/kodak_dataset";

// ── Implementation enums ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum BuildImageBitDataSetImpl {
    Native,
    YCoCgR,
    Group4x1,
    ForFirstPixel,
    ForMin,
}

impl BuildImageBitDataSetImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::YCoCgR => "ycocgr",
            Self::Group4x1 => "group_2x2",
            Self::ForFirstPixel => "forfirstpixel_2x2",
            Self::ForMin => "formin_2x2",
        }
    }

    fn process(self, path: PathBuf) -> Result<BitDataSet, EntroGdError> {
        let image = OpenImage.process(path)?;
        let (color_model, pixel_grouping, grouping_transform) = match self {
            Self::Native => (
                ImageColorModel::Rgb,
                PixelGrouping::new(1, 1),
                ImageGroupingTransform::Raw,
            ),
            Self::YCoCgR => (
                ImageColorModel::YCoCgR,
                PixelGrouping::new(1, 1),
                ImageGroupingTransform::Raw,
            ),
            Self::Group4x1 => (
                ImageColorModel::YCoCgR,
                PixelGrouping::new(2, 2),
                ImageGroupingTransform::Raw,
            ),
            Self::ForFirstPixel => (
                ImageColorModel::YCoCgR,
                PixelGrouping::new(2, 2),
                ImageGroupingTransform::ForFirstPixel,
            ),
            Self::ForMin => (
                ImageColorModel::YCoCgR,
                PixelGrouping::new(2, 2),
                ImageGroupingTransform::ForMin,
            ),
        };
        BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model,
            pixel_grouping,
            grouping_transform,
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        }
        .process(image)
    }
}

#[derive(Clone, Copy)]
enum EntropyImpl {
    Naive,
    Batched,
}

impl EntropyImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Batched => "batched",
        }
    }

    fn process(self, input: BitDataSet) -> Result<EntropyScoredContext, EntroGdError> {
        match self {
            Self::Naive => EntropyNaive {}.process(input),
            Self::Batched => EntropyBatched {}.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum GenCondensedImpl {
    Current,
}

impl GenCondensedImpl {
    fn label(self) -> &'static str {
        "current"
    }

    fn process(self, input: EntropyScoredContext) -> Result<EntropyScoredContext, EntroGdError> {
        GenCondensedSamples {
            m_max: CANONICAL_M_MAX,
        }
        .process(input)
    }
}

#[derive(Clone, Copy)]
enum SelectBasesImpl {
    Naive {
        patience: usize,
    },
    Threshold {
        base_bit_impl: BaseBitImpl,
        patience: usize,
    },
    Adaptive {
        base_bit_impl: BaseBitImpl,
        patience: usize,
    },
}

impl SelectBasesImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Naive { .. } => "naive",
            Self::Threshold {
                base_bit_impl: BaseBitImpl::Naive,
                ..
            } => "threshold_naive",
            Self::Threshold {
                base_bit_impl: BaseBitImpl::BatchGroups,
                ..
            } => "threshold_batch_groups",
            Self::Threshold {
                base_bit_impl: BaseBitImpl::IncSignatureGroups,
                ..
            } => "threshold_inc_signature_groups",
            Self::Threshold {
                base_bit_impl: BaseBitImpl::SignatureGroups,
                ..
            } => "threshold_signature_groups",
            Self::Threshold {
                base_bit_impl: BaseBitImpl::HyperLogLogCount,
                ..
            } => "threshold_hyper_log_log_count",
            Self::Adaptive {
                base_bit_impl: BaseBitImpl::Naive,
                ..
            } => "adaptive_naive",
            Self::Adaptive {
                base_bit_impl: BaseBitImpl::BatchGroups,
                ..
            } => "adaptive_batch_groups",
            Self::Adaptive {
                base_bit_impl: BaseBitImpl::IncSignatureGroups,
                ..
            } => "adaptive_inc_signature_groups",
            Self::Adaptive {
                base_bit_impl: BaseBitImpl::SignatureGroups,
                ..
            } => "adaptive_signature_groups",
            Self::Adaptive {
                base_bit_impl: BaseBitImpl::HyperLogLogCount,
                ..
            } => "adaptive_hyper_log_log_count",
        }
    }

    fn process(self, input: EntropyScoredContext) -> Result<BaseSelectionContext, EntroGdError> {
        match self {
            Self::Naive { patience } => SelectBases { patience }.process(input),
            Self::Threshold {
                base_bit_impl,
                patience,
            } => SelectBasesThreshold {
                patience,
                base_bit_impl,
                entropy_threshold: CANONICAL_ENTROPY_THRESHOLD,
            }
            .process(input),
            Self::Adaptive {
                base_bit_impl,
                patience,
            } => SelectBasesAdaptive {
                width_decay: 0.3,
                patience,
                base_bit_impl,
            }
            .process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum BuildBaseTableImpl {
    Unsorted,
    Sorted,
}

impl BuildBaseTableImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Unsorted => "build_base_table",
            Self::Sorted => "build_sorted_base_table",
        }
    }

    fn process(self, input: BaseSelectionContext) -> Result<PreEncodeContext, EntroGdError> {
        match self {
            Self::Unsorted => BuildBaseTable {}.process(input),
            Self::Sorted => BuildSortedBaseTable {}.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum EncodeImpl {
    Normal,
    Rle,
    RleOffset,
    Huffman,
}

impl EncodeImpl {
    fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Rle => "rle",
            Self::RleOffset => "rle_offset",
            Self::Huffman => "huffman",
        }
    }

    fn process(self, input: PreEncodeContext) -> Result<CompressedData, EntroGdError> {
        match self {
            Self::Normal => EncodeDataOptimized {}.process(input),
            Self::Rle => EncodeDataRLE {}.process(input),
            Self::RleOffset => EncodeDataOffsetRLE {}.process(input),
            Self::Huffman => EncodeDataHuffman {}.process(input),
        }
    }
}

#[derive(Clone, Copy)]
enum DeltaEncodeImpl {
    Current,
}

impl DeltaEncodeImpl {
    fn label(self) -> &'static str {
        "current"
    }

    fn process(self, input: CompressedData) -> Result<CompressedData, EntroGdError> {
        DeltaEncodeBaseTable {}.process(input)
    }
}

#[derive(Clone, Copy)]
enum SaveIgdImpl {
    Current,
}

impl SaveIgdImpl {
    fn label(self) -> &'static str {
        "current"
    }

    fn process(self, compressed: CompressedData, path: PathBuf) -> Result<PathBuf, EntroGdError> {
        SaveIgdFile { output_path: path }.process(compressed)
    }
}

#[derive(Clone, Copy)]
enum LoadIgdImpl {
    Current,
}

impl LoadIgdImpl {
    fn label(self) -> &'static str {
        "current"
    }

    fn process(self, path: PathBuf) -> Result<CompressedData, EntroGdError> {
        LoadIgdFile {}.process(path)
    }
}

#[derive(Clone, Copy)]
enum DecompressFileImpl {
    Current,
}

impl DecompressFileImpl {
    fn label(self) -> &'static str {
        "current"
    }

    fn process(self, input: CompressedData) -> Result<BitDataSet, EntroGdError> {
        DecompressFileData {}.process(input)
    }
}

#[derive(Clone, Copy)]
enum DecompressRowsImpl {
    Current,
}

impl DecompressRowsImpl {
    fn label(self) -> &'static str {
        "current"
    }

    fn process(
        self,
        handle: Rc<DecompressRandomAccessHandle>,
        indices: Vec<usize>,
    ) -> Result<BitDataSet, EntroGdError> {
        (*handle).clone().decompress_samples(&indices)
    }
}

// ── Standard arrays ───────────────────────────────────────────────────────────

const BUILD_IMAGE_IMPLS: [BuildImageBitDataSetImpl; 5] = [
    BuildImageBitDataSetImpl::Native,
    BuildImageBitDataSetImpl::YCoCgR,
    BuildImageBitDataSetImpl::Group4x1,
    BuildImageBitDataSetImpl::ForFirstPixel,
    BuildImageBitDataSetImpl::ForMin,
];

const ENTROPY_IMPLS: [EntropyImpl; 2] = [EntropyImpl::Naive, EntropyImpl::Batched];

const GEN_CONDENSED_IMPLS: [GenCondensedImpl; 1] = [GenCondensedImpl::Current];

const SELECT_BASES_IMPLS: [SelectBasesImpl; 9] = [
    SelectBasesImpl::Naive { patience: 10 },
    SelectBasesImpl::Threshold {
        base_bit_impl: BaseBitImpl::Naive,
        patience: 10,
    },
    SelectBasesImpl::Threshold {
        base_bit_impl: BaseBitImpl::BatchGroups,
        patience: 10,
    },
    SelectBasesImpl::Threshold {
        base_bit_impl: BaseBitImpl::IncSignatureGroups,
        patience: 10,
    },
    SelectBasesImpl::Threshold {
        base_bit_impl: BaseBitImpl::HyperLogLogCount,
        patience: 10,
    },
    SelectBasesImpl::Adaptive {
        base_bit_impl: BaseBitImpl::Naive,
        patience: 10,
    },
    SelectBasesImpl::Adaptive {
        base_bit_impl: BaseBitImpl::BatchGroups,
        patience: 10,
    },
    SelectBasesImpl::Adaptive {
        base_bit_impl: BaseBitImpl::IncSignatureGroups,
        patience: 10,
    },
    SelectBasesImpl::Adaptive {
        base_bit_impl: BaseBitImpl::HyperLogLogCount,
        patience: 10,
    },
];

const BUILD_BASE_TABLE_IMPLS: [BuildBaseTableImpl; 2] =
    [BuildBaseTableImpl::Unsorted, BuildBaseTableImpl::Sorted];

const ENCODE_IMPLS: [EncodeImpl; 4] = [
    EncodeImpl::Normal,
    EncodeImpl::Rle,
    EncodeImpl::RleOffset,
    EncodeImpl::Huffman,
];

const DELTA_ENCODE_IMPLS: [DeltaEncodeImpl; 1] = [DeltaEncodeImpl::Current];
const SAVE_IGD_IMPLS: [SaveIgdImpl; 1] = [SaveIgdImpl::Current];
const LOAD_IGD_IMPLS: [LoadIgdImpl; 1] = [LoadIgdImpl::Current];
const DECOMPRESS_FILE_IMPLS: [DecompressFileImpl; 1] = [DecompressFileImpl::Current];
const DECOMPRESS_ROWS_IMPLS: [DecompressRowsImpl; 1] = [DecompressRowsImpl::Current];

// ── Case preparation ──────────────────────────────────────────────────────────

struct CompressedDataSeeds {
    normal: CompressedData,
    rle: CompressedData,
    rle_offset: CompressedData,
    huffman: CompressedData,
}

struct IgdPaths {
    normal: PathBuf,
    rle: PathBuf,
    rle_offset: PathBuf,
    huffman: PathBuf,
}

struct DecompressRowsSeeds {
    normal: (Rc<DecompressRandomAccessHandle>, Vec<usize>),
    rle: (Rc<DecompressRandomAccessHandle>, Vec<usize>),
    rle_offset: (Rc<DecompressRandomAccessHandle>, Vec<usize>),
    huffman: (Rc<DecompressRandomAccessHandle>, Vec<usize>),
}

struct PreparedCase {
    name: String,
    data_file_path: String,
    source_size: u64,
    bit_data_seed: BitDataSet,
    entropy_seed: EntropyScoredContext,
    condensed_seed: EntropyScoredContext,
    base_table_ctx_seed: PreEncodeContext,
    pre_delta_compressed_seed: CompressedData,
    compressed_seeds: CompressedDataSeeds,
    loaded_compressed_seeds: CompressedDataSeeds,
    igd_paths: IgdPaths,
    rows_context_seeds: DecompressRowsSeeds,
}

fn canonical_select_bases(input: EntropyScoredContext) -> BaseSelectionContext {
    SelectBasesThreshold {
        patience: CANONICAL_PATIENCE,
        base_bit_impl: CANONICAL_BASE_BIT_IMPL,
        entropy_threshold: CANONICAL_ENTROPY_THRESHOLD,
    }
    .process(input)
    .unwrap()
}

fn igd_artifact_path(name: &str, encoding: &str) -> PathBuf {
    PathBuf::from(format!(
        "target/bench-artifacts/{}-{}.igd",
        name, encoding
    ))
}

fn prepare_case(image_path: PathBuf) -> PreparedCase {
    let name = image_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();

    let bit_data_seed = BuildImageBitDataSetImpl::ForFirstPixel
        .process(image_path.clone())
        .unwrap();
    let source_size = (bit_data_seed.data.num_rows * bit_data_seed.data.chunk_size / 8) as u64;

    let entropy_seed = EntropyBatched {}.process(bit_data_seed.clone()).unwrap();
    let condensed_seed = GenCondensedSamples {
        m_max: CANONICAL_M_MAX,
    }
    .process(entropy_seed.clone())
    .unwrap();

    let base_table_ctx_seed = {
        let sel = canonical_select_bases(condensed_seed.clone());
        BuildBaseTable {}.process(sel).unwrap()
    };
    let sorted_ctx = {
        let sel = canonical_select_bases(condensed_seed.clone());
        BuildSortedBaseTable {}.process(sel).unwrap()
    };

    let normal = EncodeDataOptimized {}
        .process(base_table_ctx_seed.clone())
        .unwrap();
    let rle = EncodeDataRLE {}
        .process(base_table_ctx_seed.clone())
        .unwrap();
    let rle_offset = EncodeDataOffsetRLE {}
        .process(base_table_ctx_seed.clone())
        .unwrap();
    let huffman = EncodeDataHuffman {}
        .process(sorted_ctx)
        .unwrap();
    let pre_delta_compressed_seed = normal.clone();

    let igd_paths = IgdPaths {
        normal: igd_artifact_path(&name, "normal"),
        rle: igd_artifact_path(&name, "rle"),
        rle_offset: igd_artifact_path(&name, "rle_offset"),
        huffman: igd_artifact_path(&name, "huffman"),
    };

    SaveIgdFile {
        output_path: igd_paths.normal.clone(),
    }
    .process(normal.clone())
    .unwrap();
    SaveIgdFile {
        output_path: igd_paths.rle.clone(),
    }
    .process(rle.clone())
    .unwrap();
    SaveIgdFile {
        output_path: igd_paths.rle_offset.clone(),
    }
    .process(rle_offset.clone())
    .unwrap();
    SaveIgdFile {
        output_path: igd_paths.huffman.clone(),
    }
    .process(huffman.clone())
    .unwrap();

    let loaded_normal = LoadIgdFile {}.process(igd_paths.normal.clone()).unwrap();
    let loaded_rle = LoadIgdFile {}.process(igd_paths.rle.clone()).unwrap();
    let loaded_rle_offset = LoadIgdFile {}
        .process(igd_paths.rle_offset.clone())
        .unwrap();
    let loaded_huffman = LoadIgdFile {}.process(igd_paths.huffman.clone()).unwrap();

    let rows: Vec<usize> = (0..bit_data_seed.data.num_rows).collect();

    let ctx_normal = DecompressRandomAccessHandle::new(loaded_normal.clone()).unwrap();
    let ctx_rle = DecompressRandomAccessHandle::new(loaded_rle.clone()).unwrap();
    let ctx_rle_offset = DecompressRandomAccessHandle::new(loaded_rle_offset.clone()).unwrap();
    let ctx_huffman = DecompressRandomAccessHandle::new(loaded_huffman.clone()).unwrap();

    PreparedCase {
        name,
        data_file_path: image_path.to_string_lossy().into_owned(),
        source_size,
        bit_data_seed,
        entropy_seed,
        condensed_seed,
        base_table_ctx_seed,
        pre_delta_compressed_seed,
        compressed_seeds: CompressedDataSeeds {
            normal: normal.clone(),
            rle: rle.clone(),
            rle_offset: rle_offset.clone(),
            huffman: huffman.clone(),
        },
        loaded_compressed_seeds: CompressedDataSeeds {
            normal: loaded_normal,
            rle: loaded_rle,
            rle_offset: loaded_rle_offset,
            huffman: loaded_huffman,
        },
        igd_paths,
        rows_context_seeds: DecompressRowsSeeds {
            normal: (Rc::new(ctx_normal), rows.clone()),
            rle: (Rc::new(ctx_rle), rows.clone()),
            rle_offset: (Rc::new(ctx_rle_offset), rows.clone()),
            huffman: (Rc::new(ctx_huffman), rows),
        },
    }
}

fn discover_cases() -> Vec<PreparedCase> {
    let _ = std::fs::create_dir_all("target/bench-artifacts");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(IMAGE_DIR)
        .unwrap_or_else(|e| panic!("cannot read {IMAGE_DIR}: {e}"))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let ext = path.extension()?.to_string_lossy().to_lowercase();
            (ext == "png" || ext == "jpg" || ext == "jpeg").then_some(path)
        })
        .collect();
    paths.sort();
    paths.into_iter().map(prepare_case).collect()
}

// ── Generic benchmark harness ─────────────────────────────────────────────────

fn bench_step_group<ImplType, Input, Output, LabelFn, InputFn, RunFn>(
    stage_name: &'static str,
    prepared_cases: &[PreparedCase],
    implementations: &[ImplType],
    impl_label: LabelFn,
    make_input: InputFn,
    run_impl: RunFn,
) where
    ImplType: Copy,
    LabelFn: Fn(ImplType) -> &'static str + Copy,
    InputFn: Fn(&PreparedCase) -> Input + Copy,
    RunFn: Fn(ImplType, Input) -> Result<Output, EntroGdError> + Copy,
{
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    if !filter.is_empty() && !stage_name.contains(filter.as_str()) {
        return;
    }
    let total = prepared_cases.len() * implementations.len();
    let mut done = 0;
    for case in prepared_cases {
        for &implementation in implementations {
            let label = impl_label(implementation);
            done += 1;
            eprintln!("[{done}/{total}] {stage_name}/{label}/{}", case.name);

            // warmup: avoids measuring cold-start paging / branch-predictor effects
            let _ = black_box(run_impl(implementation, make_input(case)));

            for _ in 0..N_RUNS {
                let input = make_input(case);
                let start = Instant::now();
                let output = run_impl(implementation, input).unwrap();
                let elapsed = start.elapsed();
                black_box(output);
                BENCH_RECORDS.lock().unwrap().push(BenchRecord {
                    stage: stage_name,
                    impl_name: label.to_string(),
                    file: case.name.clone(),
                    sample_time_ns: elapsed.as_nanos(),
                    source_bytes: case.source_size,
                });
            }
        }
    }
}

// ── All benchmark groups ──────────────────────────────────────────────────────

fn benchmark_filter_steps() {
    let prepared_cases = discover_cases();

    bench_step_group(
        "Step/BuildImageBitDataSet.process",
        &prepared_cases,
        &BUILD_IMAGE_IMPLS,
        BuildImageBitDataSetImpl::label,
        |case| PathBuf::from(&case.data_file_path),
        |implementation, path| implementation.process(path),
    );

    bench_step_group(
        "Step/Entropy.process",
        &prepared_cases,
        &ENTROPY_IMPLS,
        EntropyImpl::label,
        |case| case.bit_data_seed.clone(),
        |implementation, input| implementation.process(input),
    );

    bench_step_group(
        "Step/GenCondensedSamples.process",
        &prepared_cases,
        &GEN_CONDENSED_IMPLS,
        GenCondensedImpl::label,
        |case| case.entropy_seed.clone(),
        |implementation, input| implementation.process(input),
    );

    bench_step_group(
        "Step/SelectBases.process",
        &prepared_cases,
        &SELECT_BASES_IMPLS,
        SelectBasesImpl::label,
        |case| case.condensed_seed.clone(),
        |implementation, input| implementation.process(input),
    );

    bench_step_group(
        "Step/BuildBaseTable.process",
        &prepared_cases,
        &BUILD_BASE_TABLE_IMPLS,
        BuildBaseTableImpl::label,
        |case| canonical_select_bases(case.condensed_seed.clone()),
        |implementation, input| implementation.process(input),
    );

    bench_step_group(
        "Step/EncodeData.process",
        &prepared_cases,
        &ENCODE_IMPLS,
        EncodeImpl::label,
        |case| case.base_table_ctx_seed.clone(),
        |implementation, input| implementation.process(input),
    );

    bench_step_group(
        "Step/DeltaEncodeBaseTable.process",
        &prepared_cases,
        &DELTA_ENCODE_IMPLS,
        DeltaEncodeImpl::label,
        |case| case.pre_delta_compressed_seed.clone(),
        |implementation, input| implementation.process(input),
    );

    bench_step_group(
        "Step/SaveIgdFile-normal.process",
        &prepared_cases,
        &SAVE_IGD_IMPLS,
        SaveIgdImpl::label,
        |case| (case.compressed_seeds.normal.clone(), case.igd_paths.normal.clone()),
        |implementation, (compressed, path)| implementation.process(compressed, path),
    );
    bench_step_group(
        "Step/SaveIgdFile-rle.process",
        &prepared_cases,
        &SAVE_IGD_IMPLS,
        SaveIgdImpl::label,
        |case| (case.compressed_seeds.rle.clone(), case.igd_paths.rle.clone()),
        |implementation, (compressed, path)| implementation.process(compressed, path),
    );
    bench_step_group(
        "Step/SaveIgdFile-rle_offset.process",
        &prepared_cases,
        &SAVE_IGD_IMPLS,
        SaveIgdImpl::label,
        |case| {
            (
                case.compressed_seeds.rle_offset.clone(),
                case.igd_paths.rle_offset.clone(),
            )
        },
        |implementation, (compressed, path)| implementation.process(compressed, path),
    );
    bench_step_group(
        "Step/SaveIgdFile-huffman.process",
        &prepared_cases,
        &SAVE_IGD_IMPLS,
        SaveIgdImpl::label,
        |case| {
            (
                case.compressed_seeds.huffman.clone(),
                case.igd_paths.huffman.clone(),
            )
        },
        |implementation, (compressed, path)| implementation.process(compressed, path),
    );

    bench_step_group(
        "Step/LoadIgdFile-normal.process",
        &prepared_cases,
        &LOAD_IGD_IMPLS,
        LoadIgdImpl::label,
        |case| case.igd_paths.normal.clone(),
        |implementation, path| implementation.process(path),
    );
    bench_step_group(
        "Step/LoadIgdFile-rle.process",
        &prepared_cases,
        &LOAD_IGD_IMPLS,
        LoadIgdImpl::label,
        |case| case.igd_paths.rle.clone(),
        |implementation, path| implementation.process(path),
    );
    bench_step_group(
        "Step/LoadIgdFile-rle_offset.process",
        &prepared_cases,
        &LOAD_IGD_IMPLS,
        LoadIgdImpl::label,
        |case| case.igd_paths.rle_offset.clone(),
        |implementation, path| implementation.process(path),
    );
    bench_step_group(
        "Step/LoadIgdFile-huffman.process",
        &prepared_cases,
        &LOAD_IGD_IMPLS,
        LoadIgdImpl::label,
        |case| case.igd_paths.huffman.clone(),
        |implementation, path| implementation.process(path),
    );

    bench_step_group(
        "Step/DecompressFileData-normal.process",
        &prepared_cases,
        &DECOMPRESS_FILE_IMPLS,
        DecompressFileImpl::label,
        |case| case.loaded_compressed_seeds.normal.clone(),
        |implementation, input| implementation.process(input),
    );
    bench_step_group(
        "Step/DecompressFileData-rle.process",
        &prepared_cases,
        &DECOMPRESS_FILE_IMPLS,
        DecompressFileImpl::label,
        |case| case.loaded_compressed_seeds.rle.clone(),
        |implementation, input| implementation.process(input),
    );
    bench_step_group(
        "Step/DecompressFileData-rle_offset.process",
        &prepared_cases,
        &DECOMPRESS_FILE_IMPLS,
        DecompressFileImpl::label,
        |case| case.loaded_compressed_seeds.rle_offset.clone(),
        |implementation, input| implementation.process(input),
    );
    bench_step_group(
        "Step/DecompressFileData-huffman.process",
        &prepared_cases,
        &DECOMPRESS_FILE_IMPLS,
        DecompressFileImpl::label,
        |case| case.loaded_compressed_seeds.huffman.clone(),
        |implementation, input| implementation.process(input),
    );

    bench_step_group(
        "Step/DecompressRowsData-normal.process",
        &prepared_cases,
        &DECOMPRESS_ROWS_IMPLS,
        DecompressRowsImpl::label,
        |case| {
            (
                case.rows_context_seeds.normal.0.clone(),
                case.rows_context_seeds.normal.1.clone(),
            )
        },
        |implementation, (handle, indices)| implementation.process(handle, indices),
    );
    bench_step_group(
        "Step/DecompressRowsData-rle.process",
        &prepared_cases,
        &DECOMPRESS_ROWS_IMPLS,
        DecompressRowsImpl::label,
        |case| {
            (
                case.rows_context_seeds.rle.0.clone(),
                case.rows_context_seeds.rle.1.clone(),
            )
        },
        |implementation, (handle, indices)| implementation.process(handle, indices),
    );
    bench_step_group(
        "Step/DecompressRowsData-rle_offset.process",
        &prepared_cases,
        &DECOMPRESS_ROWS_IMPLS,
        DecompressRowsImpl::label,
        |case| {
            (
                case.rows_context_seeds.rle_offset.0.clone(),
                case.rows_context_seeds.rle_offset.1.clone(),
            )
        },
        |implementation, (handle, indices)| implementation.process(handle, indices),
    );
    bench_step_group(
        "Step/DecompressRowsData-huffman.process",
        &prepared_cases,
        &DECOMPRESS_ROWS_IMPLS,
        DecompressRowsImpl::label,
        |case| {
            (
                case.rows_context_seeds.huffman.0.clone(),
                case.rows_context_seeds.huffman.1.clone(),
            )
        },
        |implementation, (handle, indices)| implementation.process(handle, indices),
    );
}

// ── CSV output ────────────────────────────────────────────────────────────────

fn write_bench_csv() {
    let records = BENCH_RECORDS.lock().unwrap();
    if records.is_empty() {
        return;
    }
    let _ = std::fs::create_dir_all("target/bench-results");
    let path = format!("target/bench-results/steps_{}.csv", IMAGE_DIR.replace('/', "_"));
    let mut out = String::from("stage,impl,file,sample_time_ns,throughput_bytes_s\n");
    for r in records.iter() {
        let throughput = if r.sample_time_ns > 0 {
            r.source_bytes as u128 * 1_000_000_000 / r.sample_time_ns
        } else {
            0
        };
        out.push_str(&format!(
            "{},{},{},{},{}\n",
            r.stage, r.impl_name, r.file, r.sample_time_ns, throughput
        ));
    }
    std::fs::write(&path, out).unwrap_or_else(|e| eprintln!("failed to write {path}: {e}"));
    println!("bench results written to {path} ({} rows)", records.len());
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    benchmark_filter_steps();
    write_bench_csv();
}
