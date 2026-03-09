use csv::{Reader, ReaderBuilder};
use std::fs::File;
use std::path::Path;

use crate::error::EntroGdError;
use crate::timing::ScopedTimer;

/// Supported feature data types for parsing and bit packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureDataType {
    SignedInt,
    UnsignedInt,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FloatStorage {
    F32,
    #[default]
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingValuePolicy {
    #[default]
    Error,
    Zero,
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

    pub fn get(&self, row: usize) -> Option<DataValue> {
        match self {
            ColumnData::Signed(values) => values.get(row).copied().map(DataValue::Signed),
            ColumnData::Unsigned(values) => values.get(row).copied().map(DataValue::Unsigned),
            ColumnData::F32(values) => values.get(row).copied().map(DataValue::F32),
            ColumnData::F64(values) => values.get(row).copied().map(DataValue::F64),
        }
    }

    pub fn value_at(&self, row: usize) -> Option<DataValue> {
        self.get(row)
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

    pub fn column(&self, column: usize) -> Option<&ColumnData> {
        self.columns.get(column)
    }

    /// Get the data type of a column
    pub fn column_type(&self, column: usize) -> Option<FeatureDataType> {
        self.column(column).map(ColumnData::data_type)
    }

    /// Access a value by row and column
    pub fn value_at(&self, row: usize, column: usize) -> Option<DataValue> {
        self.column(column)?.get(row)
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

/// CSV loader with explicit float storage and missing-value handling.
#[derive(Debug, Clone, Copy)]
pub struct CsvDataLoader {
    has_headers: bool,
    float_storage: FloatStorage,
    missing_value_policy: MissingValuePolicy,
}

impl CsvDataLoader {
    pub fn new(has_headers: bool) -> Self {
        CsvDataLoader {
            has_headers,
            float_storage: FloatStorage::F64,
            missing_value_policy: MissingValuePolicy::Error,
        }
    }

    pub fn with_float_storage(mut self, float_storage: FloatStorage) -> Self {
        self.float_storage = float_storage;
        self
    }

    pub fn with_missing_value_policy(mut self, missing_value_policy: MissingValuePolicy) -> Self {
        self.missing_value_policy = missing_value_policy;
        self
    }

    pub fn with_float_type(mut self, float_type: FeatureDataType) -> Result<Self, EntroGdError> {
        self.float_storage = match float_type {
            FeatureDataType::F32 => FloatStorage::F32,
            FeatureDataType::F64 => FloatStorage::F64,
            FeatureDataType::SignedInt | FeatureDataType::UnsignedInt => {
                return Err(EntroGdError::InvalidCsvConfiguration {
                    message: format!(
                        "float storage must be F32 or F64, got {:?}",
                        float_type
                    ),
                });
            }
        };

        Ok(self)
    }
}

impl DataLoader for CsvDataLoader {
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<LoadedDataset, EntroGdError> {
        let path = path.as_ref();
        let _timer = ScopedTimer::info(format!("Loading dataset from CSV: {:?}", path));

        let scan = scan_csv(path, self.has_headers)?;
        let mut reader = open_csv_reader(path, self.has_headers)?;
        let mut columns = scan
            .column_types
            .iter()
            .copied()
            .map(|column_type| ColumnBuffer::new(column_type, self.float_storage, scan.num_rows))
            .collect::<Vec<_>>();

        for (row_idx, record_result) in reader.records().enumerate() {
            let record = record_result?;
            for (col_idx, column) in columns.iter_mut().enumerate() {
                let raw = record.get(col_idx).map(str::trim);
                column.push_field(raw, row_idx, col_idx, self.missing_value_policy)?;
            }
        }

        let dataset =
            Dataset::from_columns(columns.into_iter().map(ColumnBuffer::into_column).collect())?;
        let metadata = DatasetMetadata {
            headers: scan.headers,
        };

        Ok(LoadedDataset { dataset, metadata })
    }
}

#[derive(Debug, Clone, Copy, Default)]
enum InferredColumnType {
    #[default]
    Unsigned,
    Signed,
    Float,
}

#[derive(Debug)]
struct CsvScanSummary {
    headers: Option<Vec<String>>,
    num_rows: usize,
    column_types: Vec<InferredColumnType>,
}

#[derive(Debug)]
enum ColumnBuffer {
    Unsigned(Vec<u64>),
    Signed(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl ColumnBuffer {
    fn new(column_type: InferredColumnType, float_storage: FloatStorage, capacity: usize) -> Self {
        match column_type {
            InferredColumnType::Unsigned => ColumnBuffer::Unsigned(Vec::with_capacity(capacity)),
            InferredColumnType::Signed => ColumnBuffer::Signed(Vec::with_capacity(capacity)),
            InferredColumnType::Float => match float_storage {
                FloatStorage::F32 => ColumnBuffer::F32(Vec::with_capacity(capacity)),
                FloatStorage::F64 => ColumnBuffer::F64(Vec::with_capacity(capacity)),
            },
        }
    }

    fn push_field(
        &mut self,
        raw: Option<&str>,
        row: usize,
        column: usize,
        missing_value_policy: MissingValuePolicy,
    ) -> Result<(), EntroGdError> {
        match raw {
            Some(raw) if !raw.is_empty() => self.push_value(raw, row, column),
            _ => self.push_missing(row, column, missing_value_policy),
        }
    }

    fn push_missing(
        &mut self,
        row: usize,
        column: usize,
        missing_value_policy: MissingValuePolicy,
    ) -> Result<(), EntroGdError> {
        if matches!(missing_value_policy, MissingValuePolicy::Error) {
            return Err(EntroGdError::MissingCsvValue { row, column });
        }

        match self {
            ColumnBuffer::Unsigned(values) => values.push(0),
            ColumnBuffer::Signed(values) => values.push(0),
            ColumnBuffer::F32(values) => values.push(0.0),
            ColumnBuffer::F64(values) => values.push(0.0),
        }

        Ok(())
    }

    fn push_value(&mut self, raw: &str, row: usize, column: usize) -> Result<(), EntroGdError> {
        match self {
            ColumnBuffer::Unsigned(values) => {
                let value = raw.parse().map_err(|source| EntroGdError::ParseValue {
                    value: raw.to_string(),
                    row,
                    column,
                    source,
                })?;
                values.push(value);
            }
            ColumnBuffer::Signed(values) => {
                let value = raw.parse().map_err(|source| EntroGdError::ParseValue {
                    value: raw.to_string(),
                    row,
                    column,
                    source,
                })?;
                values.push(value);
            }
            ColumnBuffer::F32(values) => {
                let value = raw
                    .parse::<f32>()
                    .map_err(|source| EntroGdError::ParseFloatValue {
                        value: raw.to_string(),
                        row,
                        column,
                        source,
                    })?;
                values.push(value);
            }
            ColumnBuffer::F64(values) => {
                let value = raw
                    .parse::<f64>()
                    .map_err(|source| EntroGdError::ParseFloatValue {
                        value: raw.to_string(),
                        row,
                        column,
                        source,
                    })?;
                values.push(value);
            }
        }

        Ok(())
    }

    fn into_column(self) -> ColumnData {
        match self {
            ColumnBuffer::Unsigned(values) => ColumnData::Unsigned(values),
            ColumnBuffer::Signed(values) => ColumnData::Signed(values),
            ColumnBuffer::F32(values) => ColumnData::F32(values),
            ColumnBuffer::F64(values) => ColumnData::F64(values),
        }
    }
}

fn open_csv_reader(path: &Path, has_headers: bool) -> Result<Reader<File>, EntroGdError> {
    let file = File::open(path)?;
    Ok(ReaderBuilder::new()
        .has_headers(has_headers)
        .from_reader(file))
}

fn scan_csv(path: &Path, has_headers: bool) -> Result<CsvScanSummary, EntroGdError> {
    let mut reader = open_csv_reader(path, has_headers)?;
    let headers = if has_headers {
        Some(reader.headers()?.iter().map(|s| s.to_string()).collect())
    } else {
        None
    };

    let mut column_types = Vec::new();
    let mut num_rows = 0usize;

    for record_result in reader.records() {
        let record = record_result?;
        num_rows += 1;

        if record.len() > column_types.len() {
            column_types.resize(record.len(), InferredColumnType::default());
        }

        for (column, field) in record.iter().enumerate() {
            let raw = field.trim();
            if raw.is_empty() {
                continue;
            }

            column_types[column] = merge_column_types(column_types[column], infer_column_type(raw));
        }
    }

    Ok(CsvScanSummary {
        headers,
        num_rows,
        column_types,
    })
}

fn infer_column_type(value: &str) -> InferredColumnType {
    if looks_float(value) {
        InferredColumnType::Float
    } else if value.starts_with('-') {
        InferredColumnType::Signed
    } else {
        InferredColumnType::Unsigned
    }
}

fn merge_column_types(lhs: InferredColumnType, rhs: InferredColumnType) -> InferredColumnType {
    match (lhs, rhs) {
        (InferredColumnType::Float, _) | (_, InferredColumnType::Float) => InferredColumnType::Float,
        (InferredColumnType::Signed, _) | (_, InferredColumnType::Signed) => InferredColumnType::Signed,
        _ => InferredColumnType::Unsigned,
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

        assert_eq!(dataset.value_at(0, 0), Some(DataValue::Signed(10)));
        assert_eq!(dataset.value_at(1, 0), Some(DataValue::Signed(20)));
        assert_eq!(dataset.value_at(0, 1), Some(DataValue::F64(1.5)));
        assert_eq!(dataset.value_at(1, 1), Some(DataValue::F64(2.5)));
        assert_eq!(dataset.value_at(2, 1), None);
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
        assert_eq!(dataset.value_at(0, 0), Some(DataValue::Unsigned(10)));
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

    #[test]
    fn test_missing_value_policy_zero() {
        let path = std::env::temp_dir().join("entro_gd_missing_zero.csv");
        std::fs::write(&path, "a,b\n1,\n2,3\n").unwrap();

        let loaded = CsvDataLoader::new(true)
            .with_missing_value_policy(MissingValuePolicy::Zero)
            .load(&path)
            .unwrap();

        assert_eq!(loaded.dataset.value_at(0, 1), Some(DataValue::Unsigned(0)));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_missing_value_policy_error() {
        let path = std::env::temp_dir().join("entro_gd_missing_error.csv");
        std::fs::write(&path, "a,b\n1,\n").unwrap();

        let err = CsvDataLoader::new(true).load(&path).unwrap_err();
        assert!(matches!(err, EntroGdError::MissingCsvValue { row: 0, column: 1 }));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_with_float_type_rejects_integer_type() {
        let err = CsvDataLoader::new(true)
            .with_float_type(FeatureDataType::UnsignedInt)
            .unwrap_err();
        assert!(matches!(err, EntroGdError::InvalidCsvConfiguration { .. }));
    }
}
