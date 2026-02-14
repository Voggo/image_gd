use std::fs::File;
use std::path::Path;

use csv::ReaderBuilder;

use crate::error::EntroGdError;

/// Supported feature data types for parsing and bit packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureDataType {
    SignedInt,
    UnsignedInt,
    F32,
    F64,
}

/// Strongly-typed value used when accessing row/column data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataValue {
    Signed(i64),
    Unsigned(u64),
    F32(f32),
    F64(f64),
}

/// Column-oriented storage of a single data type.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    Signed(Vec<i64>),
    Unsigned(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Signed(values) => values.len(),
            ColumnData::Unsigned(values) => values.len(),
            ColumnData::F32(values) => values.len(),
            ColumnData::F64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data_type(&self) -> FeatureDataType {
        match self {
            ColumnData::Signed(_) => FeatureDataType::SignedInt,
            ColumnData::Unsigned(_) => FeatureDataType::UnsignedInt,
            ColumnData::F32(_) => FeatureDataType::F32,
            ColumnData::F64(_) => FeatureDataType::F64,
        }
    }

    pub fn value_at(&self, row: usize) -> DataValue {
        match self {
            ColumnData::Signed(values) => DataValue::Signed(values[row]),
            ColumnData::Unsigned(values) => DataValue::Unsigned(values[row]),
            ColumnData::F32(values) => DataValue::F32(values[row]),
            ColumnData::F64(values) => DataValue::F64(values[row]),
        }
    }
}

/// Column-oriented dataset where each column has a single data type.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    columns: Vec<ColumnData>,
    num_rows: usize,
}

impl Dataset {
    /// Create a dataset from typed columns. All columns must have the same length.
    pub fn from_columns(columns: Vec<ColumnData>) -> Result<Self, EntroGdError> {
        let num_rows = columns.first().map(|c| c.len()).unwrap_or(0);
        for (idx, col) in columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!(
                        "column {} has length {}, expected {}",
                        idx,
                        col.len(),
                        num_rows
                    ),
                });
            }
        }
        Ok(Dataset { columns, num_rows })
    }

    /// Get the number of rows
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get the number of columns/features
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Access all columns
    pub fn columns(&self) -> &[ColumnData] {
        &self.columns
    }

    /// Get the data type of a column
    pub fn column_type(&self, column: usize) -> FeatureDataType {
        self.columns[column].data_type()
    }

    /// Access a value by row and column
    pub fn value_at(&self, row: usize, column: usize) -> DataValue {
        self.columns[column].value_at(row)
    }
}

/// Metadata stripped from the dataset (e.g., headers).
#[derive(Debug, Clone, Default)]
pub struct DatasetMetadata {
    pub headers: Option<Vec<String>>,
}

/// Loaded dataset plus any associated metadata.
#[derive(Debug, Clone)]
pub struct LoadedDataset {
    pub dataset: Dataset,
    pub metadata: DatasetMetadata,
}

/// Generic data loader trait to support multiple input formats.
pub trait DataLoader {
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<LoadedDataset, EntroGdError>;
}

/// CSV loader with simple type inference for columns.
#[derive(Debug, Clone, Copy)]
pub struct CsvDataLoader {
    has_headers: bool,
    float_type: FeatureDataType,
}

impl CsvDataLoader {
    pub fn new(has_headers: bool) -> Self {
        CsvDataLoader {
            has_headers,
            float_type: FeatureDataType::F64,
        }
    }

    pub fn with_float_type(mut self, float_type: FeatureDataType) -> Self {
        if matches!(float_type, FeatureDataType::F32 | FeatureDataType::F64) {
            self.float_type = float_type;
        }
        self
    }
}

impl DataLoader for CsvDataLoader {
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<LoadedDataset, EntroGdError> {
        let file = File::open(path)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(self.has_headers)
            .from_reader(file);

        let headers = if self.has_headers {
            Some(reader.headers()?.iter().map(|s| s.to_string()).collect())
        } else {
            None
        };

        let mut builders: Vec<ColumnBuilder> = Vec::new();
        let mut num_rows = 0usize;

        for record_result in reader.records() {
            let record = record_result?;
            let row_len = record.len();

            if row_len > builders.len() {
                for _ in builders.len()..row_len {
                    builders.push(ColumnBuilder::new_with_len(num_rows));
                }
            }

            let total_cols = builders.len();
            for col_idx in 0..total_cols {
                if col_idx < row_len {
                    let raw = record.get(col_idx).unwrap_or("").trim();
                    if raw.is_empty() {
                        builders[col_idx].push_missing();
                    } else {
                        builders[col_idx].push_value(raw, num_rows, col_idx)?;
                    }
                } else {
                    builders[col_idx].push_missing();
                }
            }

            num_rows += 1;
        }

        let float_type = self.float_type;
        let columns = builders
            .into_iter()
            .map(|builder| builder.into_column(float_type))
            .collect();

        let dataset = Dataset::from_columns(columns)?;
        let metadata = DatasetMetadata { headers };

        Ok(LoadedDataset { dataset, metadata })
    }
}

#[derive(Debug, Clone)]
enum ColumnBuilder {
    Unsigned(Vec<u64>),
    Signed(Vec<i64>),
    Float(Vec<f64>),
}

impl ColumnBuilder {
    fn new_with_len(len: usize) -> Self {
        ColumnBuilder::Unsigned(vec![0u64; len])
    }

    fn push_missing(&mut self) {
        match self {
            ColumnBuilder::Unsigned(values) => values.push(0),
            ColumnBuilder::Signed(values) => values.push(0),
            ColumnBuilder::Float(values) => values.push(0.0),
        }
    }

    fn push_value(&mut self, raw: &str, row: usize, column: usize) -> Result<(), EntroGdError> {
        if looks_float(raw) {
            let value: f64 = raw
                .parse()
                .map_err(|source| EntroGdError::ParseFloatValue {
                    value: raw.to_string(),
                    row,
                    column,
                    source,
                })?;
            self.push_float(value);
            return Ok(());
        }

        if raw.starts_with('-') {
            let value: i64 = raw
                .parse()
                .map_err(|source| EntroGdError::ParseValue {
                    value: raw.to_string(),
                    row,
                    column,
                    source,
                })?;
            self.push_signed(value)?;
            return Ok(());
        }

        let value: u64 = raw
            .parse()
            .map_err(|source| EntroGdError::ParseValue {
                value: raw.to_string(),
                row,
                column,
                source,
            })?;
        self.push_unsigned(value);
        Ok(())
    }

    fn push_unsigned(&mut self, value: u64) {
        match self {
            ColumnBuilder::Unsigned(values) => values.push(value),
            ColumnBuilder::Signed(values) => values.push(value as i64),
            ColumnBuilder::Float(values) => values.push(value as f64),
        }
    }

    fn push_signed(&mut self, value: i64) -> Result<(), EntroGdError> {
        match self {
            ColumnBuilder::Unsigned(values) => {
                let mut signed = Vec::with_capacity(values.len() + 1);
                for &v in values.iter() {
                    if v > i64::MAX as u64 {
                        return Err(EntroGdError::InvalidMetadata {
                            message: "unsigned value too large for signed column".to_string(),
                        });
                    }
                    signed.push(v as i64);
                }
                signed.push(value);
                *self = ColumnBuilder::Signed(signed);
                Ok(())
            }
            ColumnBuilder::Signed(values) => {
                values.push(value);
                Ok(())
            }
            ColumnBuilder::Float(values) => {
                values.push(value as f64);
                Ok(())
            }
        }
    }

    fn push_float(&mut self, value: f64) {
        match self {
            ColumnBuilder::Unsigned(values) => {
                let mut floats = Vec::with_capacity(values.len() + 1);
                for &v in values.iter() {
                    floats.push(v as f64);
                }
                floats.push(value);
                *self = ColumnBuilder::Float(floats);
            }
            ColumnBuilder::Signed(values) => {
                let mut floats = Vec::with_capacity(values.len() + 1);
                for &v in values.iter() {
                    floats.push(v as f64);
                }
                floats.push(value);
                *self = ColumnBuilder::Float(floats);
            }
            ColumnBuilder::Float(values) => values.push(value),
        }
    }

    fn into_column(self, float_type: FeatureDataType) -> ColumnData {
        match self {
            ColumnBuilder::Unsigned(values) => ColumnData::Unsigned(values),
            ColumnBuilder::Signed(values) => ColumnData::Signed(values),
            ColumnBuilder::Float(values) => match float_type {
                FeatureDataType::F32 => {
                    ColumnData::F32(values.into_iter().map(|v| v as f32).collect())
                }
                FeatureDataType::F64 => ColumnData::F64(values),
                _ => ColumnData::F64(values),
            },
        }
    }
}

fn looks_float(value: &str) -> bool {
    value.contains('.') || value.contains('e') || value.contains('E')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_dataset() {
        let dataset = Dataset::from_columns(Vec::new()).unwrap();
        assert_eq!(dataset.num_rows(), 0);
        assert_eq!(dataset.num_columns(), 0);
    }

    #[test]
    fn test_from_rows() {
        let columns = vec![
            ColumnData::Unsigned(vec![1, 4]),
            ColumnData::Unsigned(vec![2, 5]),
            ColumnData::Unsigned(vec![3, 6]),
        ];
        let dataset = Dataset::from_columns(columns).unwrap();
        assert_eq!(dataset.num_rows(), 2);
        assert_eq!(dataset.num_columns(), 3);
    }

    #[test]
    fn test_row_access() {
        let columns = vec![
            ColumnData::Signed(vec![10, 20]),
            ColumnData::F64(vec![1.5, 2.5]),
        ];
        let dataset = Dataset::from_columns(columns).unwrap();

        assert_eq!(dataset.value_at(0, 0), DataValue::Signed(10));
        assert_eq!(dataset.value_at(1, 0), DataValue::Signed(20));
        assert_eq!(dataset.value_at(0, 1), DataValue::F64(1.5));
        assert_eq!(dataset.value_at(1, 1), DataValue::F64(2.5));
    }

    #[test]
    fn test_set_headers() {
        let dataset = Dataset::from_columns(vec![ColumnData::Unsigned(vec![10])]).unwrap();
        assert_eq!(dataset.num_rows(), 1);

        let headers = vec!["feature1".to_string()];
        let metadata = DatasetMetadata {
            headers: Some(headers.clone()),
        };

        assert_eq!(metadata.headers.as_ref().unwrap()[0], "feature1");
        assert_eq!(dataset.value_at(0, 0), DataValue::Unsigned(10));
    }

    #[test]
    fn test_from_rows_with_headers() {
        let dataset = Dataset::from_columns(vec![ColumnData::Unsigned(vec![5])]).unwrap();
        let metadata = DatasetMetadata {
            headers: Some(vec!["col".to_string()]),
        };

        assert_eq!(dataset.num_rows(), 1);
        assert_eq!(metadata.headers.unwrap()[0], "col");
    }
}

