use std::error::Error;
use std::fs::File;
use std::path::Path;
use csv::ReaderBuilder;

/// A simple dataset where each row is a vector of string representations
/// Each feature in a row gets its own string
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Row-oriented storage: each row is a Vec<String> where each string represents a feature
    rows: Vec<Vec<String>>,
    /// Optional column names/headers
    headers: Option<Vec<String>>,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new() -> Self {
        Dataset {
            rows: Vec::new(),
            headers: None,
        }
    }

    /// Create a dataset from rows
    pub fn from_rows(rows: Vec<Vec<String>>) -> Self {
        Dataset {
            rows,
            headers: None,
        }
    }

    /// Create a dataset from rows with headers
    pub fn from_rows_with_headers(rows: Vec<Vec<String>>, headers: Vec<String>) -> Self {
        Dataset {
            rows,
            headers: Some(headers),
        }
    }

    /// Load a dataset from a CSV file
    pub fn load_csv<P: AsRef<Path>>(path: P, has_headers: bool) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut reader = ReaderBuilder::new().has_headers(has_headers).from_reader(file);

        let mut rows = Vec::new();
        for result in reader.records() {
            let record = result?;
            let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
            rows.push(row);
        }

        Ok(Dataset {
            rows,
            headers: if has_headers {
                Some(reader.headers()?.iter().map(|s| s.to_string()).collect())
            } else {
                None
            },
        })
    }

    /// Get the number of rows
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Get the number of columns/features (from first row, or 0 if empty)
    pub fn num_columns(&self) -> usize {
        self.rows.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Get a reference to a specific row
    pub fn row(&self, idx: usize) -> Option<&Vec<String>> {
        self.rows.get(idx)
    }

    /// Get a reference to all rows
    pub fn rows(&self) -> &[Vec<String>] {
        &self.rows
    }

    /// Set headers
    pub fn set_headers(&mut self, headers: Vec<String>) {
        self.headers = Some(headers);
    }

    /// Get headers
    pub fn headers(&self) -> Option<&Vec<String>> {
        self.headers.as_ref()
    }
}

impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_dataset() {
        let dataset = Dataset::new();
        assert_eq!(dataset.num_rows(), 0);
        assert_eq!(dataset.num_columns(), 0);
    }

    #[test]
    fn test_from_rows() {
        let rows = vec![
            vec!["1".to_string(), "2".to_string(), "3".to_string()],
            vec!["4".to_string(), "5".to_string(), "6".to_string()],
        ];
        let dataset = Dataset::from_rows(rows);
        assert_eq!(dataset.num_rows(), 2);
        assert_eq!(dataset.num_columns(), 3);
    }

    #[test]
    fn test_row_access() {
        let rows = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string(), "d".to_string()],
        ];
        let dataset = Dataset::from_rows(rows);
        
        let row1 = dataset.row(0).unwrap();
        assert_eq!(row1[0], "a");
        assert_eq!(row1[1], "b");
        
        let row2 = dataset.row(1).unwrap();
        assert_eq!(row2[0], "c");
        assert_eq!(row2[1], "d");
        
        assert!(dataset.row(2).is_none());
    }

    #[test]
    fn test_set_headers() {
        let rows = vec![vec!["10".to_string()]];
        let mut dataset = Dataset::from_rows(rows);
        
        assert!(dataset.headers().is_none());
        
        let headers = vec!["feature1".to_string()];
        dataset.set_headers(headers);
        
        assert!(dataset.headers().is_some());
        assert_eq!(dataset.headers().unwrap().len(), 1);

        assert_eq!(dataset.headers().unwrap()[0], "feature1");
        assert_eq!(dataset.row(0), Some(&vec!["10".to_string()]));
    }

    #[test]
    fn test_from_rows_with_headers() {
        let rows = vec![vec!["5".to_string()]];
        let headers = vec!["col".to_string()];
        let dataset = Dataset::from_rows_with_headers(rows, headers);
        
        assert_eq!(dataset.num_rows(), 1);
        assert_eq!(dataset.headers().unwrap()[0], "col");
    }
}

