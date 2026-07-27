use polars::prelude::*;
use std::path::Path;

use crate::error::EntroGdError;
use crate::timing::ScopedTimer;

pub fn load_csv(
    path: impl AsRef<Path>,
    has_header: bool,
    null_strategy: Option<FillNullStrategy>,
) -> Result<DataFrame, EntroGdError> {
    let _timer = ScopedTimer::info(format!("Loading dataset from CSV: {:?}", path.as_ref()));

    let mut df = CsvReadOptions::default()
        .with_has_header(has_header)
        .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
        .finish()?;

    if let Some(strategy) = null_strategy {
        df = df.fill_null(strategy)?;
    }

    Ok(df)
}
