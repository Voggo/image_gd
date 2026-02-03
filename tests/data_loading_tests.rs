use entro_gd::data_loader::Dataset;

#[test]
fn test_load_csv_from_file() {
    let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true);
    assert!(dataset.is_ok(), "Failed to load CSV file");
    
    let dataset = dataset.unwrap();
    assert_eq!(dataset.num_rows(), 10000, "Expected 10000 rows");
    assert_eq!(dataset.num_columns(), 8, "Expected 8 columns");
}

#[test]
fn test_dataset_structure() {
    let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
    
    // Test first row exists and has correct number of features
    let first_row = dataset.row(0);
    assert!(first_row.is_some(), "First row should exist");
    assert_eq!(first_row.unwrap().len(), 8, "First row should have 8 features");
}

#[test]
fn test_dataset_row_access() {
    let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
    
    // Test accessing specific rows
    let row_0 = dataset.row(0);
    assert!(row_0.is_some());
    
    let row_100 = dataset.row(100);
    assert!(row_100.is_some());
    
    let row_9999 = dataset.row(9999);
    assert!(row_9999.is_some(), "Last row (9999) should exist");
    
    let row_10000 = dataset.row(10000);
    assert!(row_10000.is_none(), "Row 10000 should not exist (out of bounds)");
}

#[test]
fn test_dataset_values_are_numeric() {
    let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
    
    // Parse first few rows to ensure all values are valid u64 integers
    for row_idx in 0..10 {
        if let Some(row) = dataset.row(row_idx) {
            for (col_idx, value_str) in row.iter().enumerate() {
                let parsed: Result<u64, _> = value_str.parse();
                assert!(parsed.is_ok(), 
                    "Row {}, Column {} has non-numeric value: '{}'", 
                    row_idx, col_idx, value_str);
            }
        }
    }
}

#[test]
fn test_dataset_values_in_u8_range() {
    let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
    
    // Check that all values fit in u8 range (0-255)
    for row_idx in 0..100 {
        if let Some(row) = dataset.row(row_idx) {
            for (col_idx, value_str) in row.iter().enumerate() {
                let value: u8 = value_str.parse()
                    .expect(&format!("Failed to parse row {}, col {}", row_idx, col_idx));
                assert!(value <= 255, "Value should fit in u8 range");
            }
        }
    }
}

#[test]
fn test_dataset_rows_slice() {
    let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
    
    let rows = dataset.rows();
    assert_eq!(rows.len(), 10000, "Rows slice should have 10000 elements");
    
    // Test that we can access rows through the slice
    assert_eq!(rows[0].len(), 8, "First row in slice should have 8 features");
    assert_eq!(rows[9999].len(), 8, "Last row in slice should have 8 features");
}

#[test]
fn test_dataset_create_from_rows() {
    let rows = vec![
        vec!["1".to_string(), "2".to_string(), "3".to_string(), "4".to_string(),
             "5".to_string(), "6".to_string(), "7".to_string(), "8".to_string()],
        vec!["10".to_string(), "20".to_string(), "30".to_string(), "40".to_string(),
             "50".to_string(), "60".to_string(), "70".to_string(), "80".to_string()],
    ];
    
    let dataset = Dataset::from_rows(rows);
    assert_eq!(dataset.num_rows(), 2);
    assert_eq!(dataset.num_columns(), 8);
    
    let row_0 = dataset.row(0).unwrap();
    assert_eq!(row_0[0], "1");
    assert_eq!(row_0[7], "8");
}

#[test]
fn test_dataset_with_headers() {
    let rows = vec![
        vec!["255".to_string(), "128".to_string(), "64".to_string(), "32".to_string(),
             "16".to_string(), "8".to_string(), "4".to_string(), "2".to_string()],
    ];
    let headers = vec![
        "col0".to_string(), "col1".to_string(), "col2".to_string(), "col3".to_string(),
        "col4".to_string(), "col5".to_string(), "col6".to_string(), "col7".to_string(),
    ];
    
    let dataset = Dataset::from_rows_with_headers(rows, headers);
    
    assert_eq!(dataset.num_rows(), 1);
    assert_eq!(dataset.num_columns(), 8);
    assert!(dataset.headers().is_some());
    
    if let Some(hdrs) = dataset.headers() {
        assert_eq!(hdrs.len(), 8);
        assert_eq!(hdrs[0], "col0");
    }
}

#[test]
fn test_dataset_empty() {
    let empty_dataset = Dataset::new();
    assert_eq!(empty_dataset.num_rows(), 0);
    assert_eq!(empty_dataset.num_columns(), 0);
    assert!(empty_dataset.row(0).is_none());
}

#[test]
fn test_dataset_consistent_column_count() {
    let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
    
    // Verify all rows have the same number of columns
    let expected_cols = dataset.num_columns();
    for row_idx in 0..dataset.num_rows() {
        if let Some(row) = dataset.row(row_idx) {
            assert_eq!(row.len(), expected_cols,
                "Row {} has {} columns, expected {}", 
                row_idx, row.len(), expected_cols);
        }
    }
}

#[test]
fn test_sample_data_values() {
    let dataset = Dataset::load_csv("data/data-10000-8-int.csv", true).unwrap();
    
    // Test a specific row to ensure data loads correctly
    // (these values should match what's in the CSV)
    if let Some(row) = dataset.row(0) {
        assert_eq!(row.len(), 8);
        // All values should be parseable as integers
        for value_str in row.iter() {
            let _: u64 = value_str.parse()
                .expect(&format!("Failed to parse: {}", value_str));
        }
    }
}
