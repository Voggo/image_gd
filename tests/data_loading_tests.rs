use entro_gd::data_loader::{
    ColumnData, CsvDataLoader, DataLoader, DataValue, Dataset, DatasetMetadata,
};

#[test]
fn test_load_csv_from_file() {
    let loader = CsvDataLoader::new(true);
    let dataset = loader.load("data/tabular/data-10000-8-int.csv");
    assert!(dataset.is_ok(), "Failed to load CSV file");

    let dataset = dataset.unwrap().dataset;
    assert_eq!(dataset.num_rows(), 10000, "Expected 10000 rows");
    assert_eq!(dataset.num_columns(), 8, "Expected 8 columns");
}

#[test]
fn test_dataset_structure() {
    let loader = CsvDataLoader::new(true);
    let dataset = loader
        .load("data/tabular/data-10000-8-int.csv")
        .unwrap()
        .dataset;

    assert_eq!(dataset.num_columns(), 8, "Expected 8 columns");
    assert_eq!(dataset.num_rows(), 10000, "Expected 10000 rows");

    for column in dataset.columns() {
        assert_eq!(column.len(), dataset.num_rows());
    }
}

#[test]
fn test_dataset_row_access() {
    let loader = CsvDataLoader::new(true);
    let dataset = loader
        .load("data/tabular/data-10000-8-int.csv")
        .unwrap()
        .dataset;

    let value_0_0 = dataset.value_at(0, 0);
    let value_100_0 = dataset.value_at(100, 0);
    let value_9999_0 = dataset.value_at(9999, 0);

    assert!(matches!(value_0_0, Some(DataValue::Unsigned(_))));
    assert!(matches!(value_100_0, Some(DataValue::Unsigned(_))));
    assert!(matches!(value_9999_0, Some(DataValue::Unsigned(_))));
}

#[test]
fn test_dataset_values_are_numeric() {
    let loader = CsvDataLoader::new(true);
    let dataset = loader
        .load("data/tabular/data-10000-8-int.csv")
        .unwrap()
        .dataset;

    for row_idx in 0..10 {
        for col_idx in 0..dataset.num_columns() {
            let value = dataset.value_at(row_idx, col_idx);
            assert!(matches!(value, Some(DataValue::Unsigned(_))));
        }
    }
}

#[test]
fn test_dataset_values_in_u8_range() {
    let loader = CsvDataLoader::new(true);
    let dataset = loader
        .load("data/tabular/data-10000-8-int.csv")
        .unwrap()
        .dataset;

    for row_idx in 0..100 {
        for col_idx in 0..dataset.num_columns() {
            match dataset.value_at(row_idx, col_idx) {
                Some(DataValue::Unsigned(value)) => assert!(value <= 255),
                other => panic!("Unexpected value type: {:?}", other),
            }
        }
    }
}

#[test]
fn test_dataset_rows_slice() {
    let loader = CsvDataLoader::new(true);
    let dataset = loader
        .load("data/tabular/data-10000-8-int.csv")
        .unwrap()
        .dataset;

    assert_eq!(dataset.num_rows(), 10000);
    assert_eq!(dataset.num_columns(), 8);

    for column in dataset.columns() {
        assert_eq!(column.len(), 10000);
    }
}

#[test]
fn test_dataset_create_from_rows() {
    let columns = vec![
        ColumnData::Unsigned(vec![1, 10]),
        ColumnData::Unsigned(vec![2, 20]),
        ColumnData::Unsigned(vec![3, 30]),
        ColumnData::Unsigned(vec![4, 40]),
        ColumnData::Unsigned(vec![5, 50]),
        ColumnData::Unsigned(vec![6, 60]),
        ColumnData::Unsigned(vec![7, 70]),
        ColumnData::Unsigned(vec![8, 80]),
    ];

    let dataset = Dataset::from_columns(columns).unwrap();
    assert_eq!(dataset.num_rows(), 2);
    assert_eq!(dataset.num_columns(), 8);

    assert_eq!(dataset.value_at(0, 0), Some(DataValue::Unsigned(1)));
    assert_eq!(dataset.value_at(0, 7), Some(DataValue::Unsigned(8)));
}

#[test]
fn test_dataset_with_headers() {
    let columns = vec![
        ColumnData::Unsigned(vec![255]),
        ColumnData::Unsigned(vec![128]),
        ColumnData::Unsigned(vec![64]),
        ColumnData::Unsigned(vec![32]),
        ColumnData::Unsigned(vec![16]),
        ColumnData::Unsigned(vec![8]),
        ColumnData::Unsigned(vec![4]),
        ColumnData::Unsigned(vec![2]),
    ];
    let headers = vec![
        "col0".to_string(),
        "col1".to_string(),
        "col2".to_string(),
        "col3".to_string(),
        "col4".to_string(),
        "col5".to_string(),
        "col6".to_string(),
        "col7".to_string(),
    ];

    let dataset = Dataset::from_columns(columns).unwrap();
    let metadata = DatasetMetadata {
        headers: Some(headers),
    };

    assert_eq!(dataset.num_rows(), 1);
    assert_eq!(dataset.num_columns(), 8);
    assert!(metadata.headers.is_some());

    if let Some(hdrs) = metadata.headers {
        assert_eq!(hdrs.len(), 8);
        assert_eq!(hdrs[0], "col0");
    }
}

#[test]
fn test_dataset_empty() {
    let empty_dataset = Dataset::from_columns(Vec::new()).unwrap();
    assert_eq!(empty_dataset.num_rows(), 0);
    assert_eq!(empty_dataset.num_columns(), 0);
}

#[test]
fn test_dataset_consistent_column_count() {
    let loader = CsvDataLoader::new(true);
    let dataset = loader
        .load("data/tabular/data-10000-8-int.csv")
        .unwrap()
        .dataset;

    let expected_rows = dataset.num_rows();
    for column in dataset.columns() {
        assert_eq!(column.len(), expected_rows);
    }
}

#[test]
fn test_sample_data_values() {
    let loader = CsvDataLoader::new(true);
    let dataset = loader
        .load("data/tabular/data-10000-8-int.csv")
        .unwrap()
        .dataset;

    for col_idx in 0..dataset.num_columns() {
        let value = dataset.value_at(0, col_idx);
        assert!(matches!(value, Some(DataValue::Unsigned(_))));
    }
}
