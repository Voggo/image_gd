import csv
import sys

def compare_csv(file1, file2):
    """
    Compare two CSV files and print the differences.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
    """
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        differences = []

        for row_num, (row1, row2) in enumerate(zip(reader1, reader2), start=1):
            for col_num, (cell1, cell2) in enumerate(zip(row1, row2), start=1):
                if cell1 != cell2:
                    differences.append((row_num, col_num, cell1, cell2))

        # Check if one file has extra rows
        extra_rows_file1 = list(reader1)
        extra_rows_file2 = list(reader2)

        if extra_rows_file1:
            for row_num, row in enumerate(extra_rows_file1, start=row_num + 1):
                differences.append((row_num, None, row, None))

        if extra_rows_file2:
            for row_num, row in enumerate(extra_rows_file2, start=row_num + 1):
                differences.append((row_num, None, None, row))

    if differences:
        print("Differences found:")
        for diff in differences:
            row, col, val1, val2 = diff
            if col is None:
                if val1 is not None:
                    print(f"Extra row in file1 at row {row}: {val1}")
                if val2 is not None:
                    print(f"Extra row in file2 at row {row}: {val2}")
            else:
                print(f"Difference at row {row}, column {col}: file1='{val1}' file2='{val2}'")
    else:
        print("The files are identical.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_csv.py <file1.csv> <file2.csv>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    compare_csv(file1, file2)
