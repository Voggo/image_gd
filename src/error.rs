use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum EntroGdError {
    Io(std::io::Error),
    Csv(csv::Error),
    Image(image::ImageError),
    ParseValue {
        value: String,
        row: usize,
        column: usize,
        source: std::num::ParseIntError,
    },
    ParseFloatValue {
        value: String,
        row: usize,
        column: usize,
        source: std::num::ParseFloatError,
    },
    BitSliceLengthMismatch {
        expected: usize,
        actual: usize,
    },
    DecompressionSampleMissing {
        sample_idx: usize,
    },
    InvalidBaseId {
        base_id: usize,
        table_len: usize,
    },
    InvalidMetadata {
        message: String,
    },
    InvalidFeatureSpec {
        message: String,
    },
}

impl Display for EntroGdError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            EntroGdError::Io(err) => write!(f, "IO error: {}", err),
            EntroGdError::Csv(err) => write!(f, "CSV error: {}", err),
            EntroGdError::Image(err) => write!(f, "Image error: {}", err),
            EntroGdError::ParseValue {
                value,
                row,
                column,
                source,
            } => write!(
                f,
                "Failed to parse value '{}' at row {}, column {}: {}",
                value, row, column, source
            ),
            EntroGdError::ParseFloatValue {
                value,
                row,
                column,
                source,
            } => write!(
                f,
                "Failed to parse float value '{}' at row {}, column {}: {}",
                value, row, column, source
            ),
            EntroGdError::BitSliceLengthMismatch { expected, actual } => write!(
                f,
                "Bit slice length mismatch (expected {}, got {})",
                expected, actual
            ),
            EntroGdError::DecompressionSampleMissing { sample_idx } => {
                write!(f, "Failed to retrieve sample at index {}", sample_idx)
            }
            EntroGdError::InvalidBaseId { base_id, table_len } => write!(
                f,
                "Invalid base ID: {} (base table has {} entries)",
                base_id, table_len
            ),
            EntroGdError::InvalidMetadata { message } => {
                write!(f, "Invalid compression metadata: {}", message)
            }
            EntroGdError::InvalidFeatureSpec { message } => {
                write!(f, "Invalid feature specification: {}", message)
            }
        }
    }
}

impl Error for EntroGdError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            EntroGdError::Io(err) => Some(err),
            EntroGdError::Csv(err) => Some(err),
            EntroGdError::Image(err) => Some(err),
            EntroGdError::ParseValue { source, .. } => Some(source),
            EntroGdError::ParseFloatValue { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<std::io::Error> for EntroGdError {
    fn from(err: std::io::Error) -> Self {
        EntroGdError::Io(err)
    }
}

impl From<csv::Error> for EntroGdError {
    fn from(err: csv::Error) -> Self {
        EntroGdError::Csv(err)
    }
}

impl From<image::ImageError> for EntroGdError {
    fn from(err: image::ImageError) -> Self {
        EntroGdError::Image(err)
    }
}
