use entro_gd::PipelineProfileSet;
use entro_gd::pipeline_runner::{ExperimentRunOptions, run_experiments_on_path, write_report_csv};
use entro_gd::{EntroGdError, init_logging};
use std::env;
use std::path::PathBuf;

const DEFAULT_CONFIG_PATH: &str = "configs/experiment_profiles.json";

fn main() -> Result<(), EntroGdError> {
    let _log_handle = init_logging();

    let args: Vec<String> = env::args().collect();
    let mut input_path: Option<PathBuf> = None;
    let mut config_path: PathBuf = PathBuf::from(DEFAULT_CONFIG_PATH);
    let mut recursive = false;
    let mut compare_png = false;
    let mut output_path = PathBuf::from("target/experiment-dashboard.csv");

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --path".to_string(),
                    });
                }
                input_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--recursive" => {
                recursive = true;
                i += 1;
            }
            "--compare-png" => {
                compare_png = true;
                i += 1;
            }
            "--config" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --config".to_string(),
                    });
                }
                config_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--output" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --output".to_string(),
                    });
                }
                output_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            other => {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("unknown argument: {}", other),
                });
            }
        }
    }

    let (profiles, file_config) = PipelineProfileSet::from_config_file(&config_path)?;

    let input_path = input_path
        .or_else(|| file_config.input_path.as_ref().map(PathBuf::from))
        .ok_or_else(|| EntroGdError::InvalidMetadata {
            message: "--path is required (or set input_path in config)".to_string(),
        })?;

    let options = ExperimentRunOptions {
        recursive: file_config.recursive.unwrap_or(recursive),
        compare_png: file_config.compare_png.unwrap_or(compare_png),
    };

    let report = run_experiments_on_path(&input_path, &profiles, options)?;
    write_report_csv(&output_path, &report.records)?;

    tracing::info!("Discovered files: {}", report.discovered_files);
    tracing::info!("Processed files: {}", report.processed_files);
    tracing::info!("Skipped files: {}", report.skipped_files.len());
    tracing::info!("Result rows: {}", report.records.len());
    tracing::info!("PNG comparison enabled: {}", options.compare_png);
    tracing::info!("Wrote dashboard report to: {}", output_path.display());

    Ok(())
}
