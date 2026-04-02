use entro_gd::compression::BaseBitBatchGroups;
use entro_gd::compression::base_bits::BaseBitHyperLogLogCount;
use entro_gd::compression::preprocessor::DEFAULT_BITDATA_ROW_PADDING;
use entro_gd::data_loader::{CsvDataLoader, DataLoader};
use entro_gd::{
    BitDataSet, BuildImageBitDataSet, EntroGdError, Filter, ImageColorModel, ImageColorSpace,
    ImageGroupingTransform, calculate_entropy, init_logging,
};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;

#[derive(Clone, Copy)]
enum InputKind {
    Auto,
    Csv,
    Image,
}

struct CliOptions {
    input: PathBuf,
    output: PathBuf,
    has_headers: bool,
    limit_bits: Option<usize>,
    input_kind: InputKind,
}

fn parse_bool(value: &str) -> Result<bool, EntroGdError> {
    match value {
        "1" | "true" | "TRUE" | "yes" | "YES" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" => Ok(false),
        _ => Err(EntroGdError::InvalidMetadata {
            message: format!("invalid bool value '{}'; use true/false", value),
        }),
    }
}

fn parse_args() -> Result<CliOptions, EntroGdError> {
    let mut input: Option<PathBuf> = None;
    let mut output = PathBuf::from("target/basebit_hll_precision.csv");
    let mut has_headers = true;
    let mut limit_bits: Option<usize> = None;
    let mut input_kind = InputKind::Auto;

    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --input".to_string(),
                    });
                }
                input = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--output" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --output".to_string(),
                    });
                }
                output = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--headers" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --headers".to_string(),
                    });
                }
                has_headers = parse_bool(&args[i + 1])?;
                i += 2;
            }
            "--limit-bits" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --limit-bits".to_string(),
                    });
                }
                let value =
                    args[i + 1]
                        .parse::<usize>()
                        .map_err(|err| EntroGdError::InvalidMetadata {
                            message: format!(
                                "invalid --limit-bits value '{}': {}",
                                args[i + 1],
                                err
                            ),
                        })?;
                limit_bits = Some(value);
                i += 2;
            }
            "--input-kind" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --input-kind".to_string(),
                    });
                }
                input_kind = match args[i + 1].as_str() {
                    "auto" => InputKind::Auto,
                    "csv" => InputKind::Csv,
                    "image" => InputKind::Image,
                    other => {
                        return Err(EntroGdError::InvalidMetadata {
                            message: format!(
                                "invalid --input-kind value '{}'; use auto|csv|image",
                                other
                            ),
                        });
                    }
                };
                i += 2;
            }
            other => {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("unknown argument: {}", other),
                });
            }
        }
    }

    let input = input.ok_or_else(|| EntroGdError::InvalidMetadata {
        message: "--input is required".to_string(),
    })?;

    Ok(CliOptions {
        input,
        output,
        has_headers,
        limit_bits,
        input_kind,
    })
}

fn infer_input_kind(path: &Path) -> InputKind {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("png" | "jpg" | "jpeg" | "bmp" | "gif") => InputKind::Image,
        _ => InputKind::Csv,
    }
}

fn load_bit_data(options: &CliOptions) -> Result<BitDataSet, EntroGdError> {
    let resolved_kind = match options.input_kind {
        InputKind::Auto => infer_input_kind(&options.input),
        explicit => explicit,
    };

    match resolved_kind {
        InputKind::Csv => {
            let loader = CsvDataLoader::new(options.has_headers);
            let dataset = loader.load(&options.input)?.dataset;
            BitDataSet::from_dataset(&dataset)
        }
        InputKind::Image => BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::YCoCgR,
            pixel_grouping: 1,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: DEFAULT_BITDATA_ROW_PADDING,
        }
        .process(options.input.clone()),
        InputKind::Auto => unreachable!("auto input kind should be resolved before load"),
    }
}

fn main() -> Result<(), EntroGdError> {
    let _log_handle = init_logging();
    let options = parse_args()?;

    let bit_data = load_bit_data(&options)?;

    let mut entropy = calculate_entropy(&bit_data);
    entropy.sort_by(|a, b| a.1.total_cmp(&b.1));

    let max_steps = options
        .limit_bits
        .map(|limit| limit.min(entropy.len()))
        .unwrap_or(entropy.len());

    if let Some(parent) = options.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let output_file = File::create(&options.output)?;
    let mut writer = BufWriter::new(output_file);

    writeln!(
        writer,
        "step,bit_position,entropy,is_constant,exact_num_bases,approx_num_bases,abs_error,relative_error"
    )?;

    let mut exact = BaseBitBatchGroups::new(bit_data.num_rows(), bit_data.chunk_size());
    let mut approx = BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size());

    let mut sum_abs_rel = 0.0f64;
    let mut sum_sq_rel = 0.0f64;
    let mut sum_rel = 0.0f64;
    let mut measured = 0usize;

    for (step, (bit_position, entropy_value)) in entropy.into_iter().take(max_steps).enumerate() {
        let is_constant = entropy_value == 0.0;
        if is_constant {
            exact.add_constant_bit_positions(&[bit_position]);
            approx.add_constant_bit_positions(&[bit_position]);
        } else {
            exact.add_bit_position(&bit_data, bit_position);
            approx.add_bit_position(&bit_data, bit_position);
        }

        let exact_num_bases = exact.get_num_bases();
        let approx_num_bases = approx.get_num_bases();
        let abs_error = approx_num_bases.abs_diff(exact_num_bases);

        let relative_error = if exact_num_bases == 0 {
            0.0
        } else {
            (approx_num_bases as f64 - exact_num_bases as f64) / exact_num_bases as f64
        };

        if !is_constant {
            sum_abs_rel += relative_error.abs();
            sum_sq_rel += relative_error * relative_error;
            sum_rel += relative_error;
            measured += 1;
        }

        writeln!(
            writer,
            "{},{},{:.8},{},{},{},{},{:.8}",
            step + 1,
            bit_position,
            entropy_value,
            is_constant,
            exact_num_bases,
            approx_num_bases,
            abs_error,
            relative_error
        )?;
    }

    writer.flush()?;

    if measured > 0 {
        let n = measured as f64;
        let mape = sum_abs_rel / n;
        let rmse = (sum_sq_rel / n).sqrt();
        let bias = sum_rel / n;

        tracing::info!(
            input = %options.input.display(),
            output = %options.output.display(),
            measured_steps = measured,
            mape,
            rmse,
            bias,
            "HLL precision evaluation completed"
        );
    } else {
        tracing::warn!("No steps evaluated; check input data and --limit-bits");
    }

    Ok(())
}
