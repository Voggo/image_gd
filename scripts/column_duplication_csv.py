#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def detect_csv_dialect(file_path: Path) -> csv.Dialect:
	with file_path.open("r", newline="", encoding="utf-8") as f:
		sample = f.read(4096)
		f.seek(0)
		try:
			return csv.Sniffer().sniff(sample)
		except csv.Error:
			return csv.get_dialect("excel")


def parse_column_selection(user_input: str, header: list[str]) -> list[int]:
	selections = [item.strip() for item in user_input.split(",") if item.strip()]
	if not selections:
		raise ValueError("No columns were selected.")

	indices: list[int] = []
	for item in selections:
		if item.isdigit():
			idx = int(item)
			if idx < 0 or idx >= len(header):
				raise ValueError(f"Column index out of range: {idx}")
			indices.append(idx)
			continue

		if item not in header:
			raise ValueError(f"Column name not found: {item}")
		indices.append(header.index(item))

	return list(dict.fromkeys(indices))


def ask_duplication_counts(selected_indices: list[int], header: list[str]) -> dict[int, int]:
	counts: dict[int, int] = {}
	for idx in selected_indices:
		col_name = header[idx]
		while True:
			raw = input(f"How many copies of '{col_name}'? ").strip()
			if not raw.isdigit():
				print("Please enter a non-negative integer.")
				continue

			count = int(raw)
			if count < 0:
				print("Please enter a non-negative integer.")
				continue

			counts[idx] = count
			break

	return counts


def duplicate_columns(rows: list[list[str]], header: list[str], counts: dict[int, int]) -> list[list[str]]:
	new_header = header.copy()
	for idx, count in counts.items():
		for i in range(1, count + 1):
			new_header.append(f"{header[idx]}_copy{i}")

	new_rows: list[list[str]] = [new_header]
	for row in rows:
		expanded = row.copy()
		for idx, count in counts.items():
			value = row[idx] if idx < len(row) else ""
			for _ in range(count):
				expanded.append(value)
		new_rows.append(expanded)

	return new_rows


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Duplicate selected CSV columns interactively."
	)
	parser.add_argument("input_csv", help="Path to input CSV file")
	parser.add_argument(
		"-o",
		"--output",
		help="Path to output CSV file (default: <input_stem>_duplicated.csv)",
	)
	args = parser.parse_args()

	input_path = Path(args.input_csv)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	output_path = Path(args.output) if args.output else input_path.with_name(
		f"{input_path.stem}_duplicated{input_path.suffix}"
	)

	dialect = detect_csv_dialect(input_path)

	with input_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.reader(f, dialect)
		all_rows = list(reader)

	if not all_rows:
		raise ValueError("Input CSV is empty.")

	header = all_rows[0]
	data_rows = all_rows[1:]

	print("Available columns:")
	for i, name in enumerate(header):
		print(f"  [{i}] {name}")

	selection_input = input(
		"Enter columns to duplicate (comma-separated names or indices): "
	)
	selected_indices = parse_column_selection(selection_input, header)

	counts = ask_duplication_counts(selected_indices, header)
	updated_rows = duplicate_columns(data_rows, header, counts)

	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f, dialect)
		writer.writerows(updated_rows)

	print(f"Done. Wrote duplicated CSV to: {output_path}")


if __name__ == "__main__":
	main()
