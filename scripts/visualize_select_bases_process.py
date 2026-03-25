#!/usr/bin/env python3
"""Visualize base selection growth from debug CSV output.

Creates a stacked bar chart where:
- x-axis: bit positions in *order of addition* (row order in the CSV)
- y-axis: size breakdown (`size_bases`, `size_deviations`, `size_ids`, `size_params`)

Example (single file):
  python scripts/visualize_select_bases_process.py \
	  --input data/base_selection_debug_6.csv \
	  --output logs/base_selection_debug_6.png

Example (folder):
  python scripts/visualize_select_bases_process.py \
	  --input data \
	  --output logs/base-selection-plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SIZE_COLUMNS = ["size_bases", "size_deviations", "size_ids", "size_params"]


def format_size(value: int) -> str:
	"""Format an integer size value using SI suffixes (B, KB, MB, ...)."""
	units = ["B", "KB", "MB", "GB", "TB", "PB"]
	value_f = float(value)
	unit_idx = 0
	while value_f >= 1000.0 and unit_idx < len(units) - 1:
		value_f /= 1000.0
		unit_idx += 1

	if unit_idx == 0:
		return f"{int(value_f)} {units[unit_idx]}"
	return f"{value_f:.2f} {units[unit_idx]}"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot size breakdown vs bit-position addition order"
	)
	parser.add_argument(
		"--input",
		default="data/base_selection_debug_6.csv",
		help="Path to base selection debug CSV or a folder of CSV files",
	)
	parser.add_argument(
		"--output",
		default="",
		help=(
			"Optional output path. For file input: output image path. "
			"For directory input: output directory for generated images."
		),
	)
	parser.add_argument("--dpi", type=int, default=150, help="Figure DPI for saved output")
	parser.add_argument(
		"--no-total-line",
		action="store_true",
		help="Disable overlay line for total_size",
	)
	return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, int]]:
	with path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		rows: list[dict[str, int]] = []
		for raw in reader:
			rows.append(
				{
					"bit_position": int(raw["bit_position"]),
					"num_bases": int(raw["num_bases"]),
					"size_bases": int(raw["size_bases"]),
					"size_deviations": int(raw["size_deviations"]),
					"size_ids": int(raw["size_ids"]),
					"size_params": int(raw["size_params"]),
					"total_size": int(raw["total_size"]),
				}
			)
	return rows


def make_plot(rows: list[dict[str, int]], show_total_line: bool = True):
	try:
		import matplotlib.pyplot as plt
		from matplotlib.ticker import FuncFormatter
	except Exception as exc:  # pragma: no cover
		raise RuntimeError("matplotlib is required for plotting") from exc

	x = list(range(len(rows)))
	x_labels = [str(r["bit_position"]) for r in rows]
	totals = [r["total_size"] for r in rows]
	min_idx = min(range(len(rows)), key=lambda i: totals[i])

	fig, ax = plt.subplots(figsize=(14, 7))
	bottom = [0] * len(rows)
	segment_values: dict[str, list[int]] = {}

	for col in SIZE_COLUMNS:
		values = [r[col] for r in rows]
		segment_values[col] = values
		bars = ax.bar(x, values, bottom=bottom, width=0.85, label=col)

		# Highlight the smallest-total bar across all stacked segments.
		bars[min_idx].set_edgecolor("crimson")
		bars[min_idx].set_linewidth(2.4)
		bars[min_idx].set_hatch("///")
		bottom = [b + v for b, v in zip(bottom, values)]

	if show_total_line:
		ax.plot(x, totals, color="black", linewidth=2, marker="o", label="total_size")

	# Explicitly mark the smallest total point.
	ax.scatter(
		x[min_idx],
		totals[min_idx],
		color="crimson",
		edgecolor="black",
		zorder=5,
		s=80,
		label="smallest total",
	)

	# Annotate total and per-component breakdown for the smallest bar.
	bit_pos = rows[min_idx]["bit_position"]
	annotation_lines = [
		f"smallest @ bit {bit_pos}",
		f"total_size={format_size(totals[min_idx])} ({totals[min_idx]})",
	]
	for col in SIZE_COLUMNS:
		annotation_lines.append(
			f"{col}={format_size(segment_values[col][min_idx])} ({segment_values[col][min_idx]})"
		)

	ax.annotate(
		"\n".join(annotation_lines),
		xy=(x[min_idx], totals[min_idx]),
		xytext=(20, 20),
		textcoords="offset points",
		bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "crimson", "alpha": 0.95},
		arrowprops={"arrowstyle": "->", "color": "crimson", "lw": 1.2},
	)

	# Also label the segment sizes directly inside the highlighted bar where possible.
	segment_bottom = 0
	for col in SIZE_COLUMNS:
		value = segment_values[col][min_idx]
		if value > 0:
			ax.text(
				x[min_idx],
				segment_bottom + (value / 2),
				format_size(value),
				ha="center",
				va="center",
				fontsize=8,
				fontweight="bold",
				color="black",
			)
		segment_bottom += value

	ax.set_title("Base Selection Process: Size Breakdown by Addition Order")
	ax.set_xlabel("Bit position (in order of addition)")
	ax.set_ylabel("Size (B/KB/MB)")
	ax.yaxis.set_major_formatter(
		FuncFormatter(lambda v, _pos: format_size(int(max(v, 0))))
	)
	ax.set_xticks(x)
	ax.set_xticklabels(x_labels, rotation=45, ha="right")
	ax.grid(axis="y", alpha=0.25)
	ax.legend()

	plt.tight_layout()
	return fig


def _process_single_file(input_csv: Path, output_path: Path | None, show_total_line: bool, dpi: int) -> int:
	rows = read_rows(input_csv)
	if not rows:
		print(f"Skipping empty CSV: {input_csv}")
		return 0

	fig = make_plot(rows, show_total_line=show_total_line)

	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=dpi)
		print(f"Saved figure to: {output_path}")
	else:
		import matplotlib.pyplot as plt

		plt.show()

	return 0


def main() -> int:
	args = parse_args()
	input_path = Path(args.input)
	if not input_path.exists():
		print(f"Input file not found: {input_path}")
		return 1

	if input_path.is_file():
		output_path = Path(args.output) if args.output else None
		return _process_single_file(
			input_csv=input_path,
			output_path=output_path,
			show_total_line=not args.no_total_line,
			dpi=args.dpi,
		)

	# Directory mode: process all CSV files in the folder.
	csv_files = sorted(input_path.glob("*.csv"))
	if not csv_files:
		print(f"No .csv files found in directory: {input_path}")
		return 1

	if args.output:
		output_dir = Path(args.output)
	else:
		output_dir = input_path / "plots"

	output_dir.mkdir(parents=True, exist_ok=True)

	for csv_file in csv_files:
		out_file = output_dir / f"{csv_file.stem}.png"
		_process_single_file(
			input_csv=csv_file,
			output_path=out_file,
			show_total_line=not args.no_total_line,
			dpi=args.dpi,
		)

	print(f"Processed {len(csv_files)} CSV files")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
