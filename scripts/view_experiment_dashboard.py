#!/usr/bin/env python3
"""Quick dashboard viewer for experiment CSV output.

Usage:
  python scripts/view_experiment_dashboard.py --input target/experiment-dashboard.csv
  python scripts/view_experiment_dashboard.py --input target/experiment-dashboard.csv --plot
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _normalize_header(name: str) -> str:
    return name.strip()


def _stitch_rows(path: Path) -> list[dict]:
    """Read CSV rows while tolerating whitespace-padded headers and split physical lines.

    Some generated dashboard files can contain rows split across multiple physical
    lines (e.g. from unexpected newlines in one field). This function stitches
    partial rows until the expected column count is reached.
    """
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            return []

        header = [_normalize_header(col) for col in raw_header]
        expected = len(header)
        rows: list[dict] = []
        pending: list[str] = []

        for raw_row in reader:
            row = [cell.strip() for cell in raw_row]

            if not row:
                continue

            if pending:
                pending.extend(row)
            else:
                pending = row

            if len(pending) < expected:
                continue

            if len(pending) > expected:
                # Keep extra fragments in the last column so we don't drop data.
                pending = pending[: expected - 1] + [" ".join(pending[expected - 1 :])]

            rows.append(dict(zip(header, pending)))
            pending = []

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View compression experiment dashboard data")
    parser.add_argument("--input", required=True, help="Path to dashboard CSV")
    parser.add_argument("--top", type=int, default=5, help="Top-N rows for per-file summaries")
    parser.add_argument("--plot", action="store_true", help="Show a quick scatter plot")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = _stitch_rows(path)

    if not rows:
        return rows

    for row in rows:
        for key in (
            "total_ms",
            "load_ms",
            "preprocess_ms",
            "entropy_ms",
            "condensed_ms",
            "select_ms",
            "encode_ms",
        ):
            row[key] = float(row.get(key, "0") or "0")

        for key in (
            "original_bits",
            "estimated_total_bits",
            "encoded_stream_total_bits",
            "encoded_payload_bits",
            "normal_symbol_stream_bits",
            "rle_symbol_stream_bits",
            "rle_control_stream_bits",
            "rle_packet_count",
            "huffman_pixel_stream_bits",
            "huffman_row_offsets_bits",
            "huffman_symbol_table_bits",
            "huffman_code_lengths_bits",
            "base_table_pattern_bits",
            "base_table_value_bits",
            "base_bit_positions_bits",
            "condensed_weights_bits",
        ):
            row[key] = int(row.get(key, "0") or "0")

        row["compression_ratio"] = row["estimated_total_bits"] / max(row["original_bits"], 1)

    return rows


def format_bits(bits: int) -> str:
    """Format bit counts with byte-scaled units (B/KB/MB/GB/...)."""
    bytes_value = bits / 8.0
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_idx = 0
    while bytes_value >= 1000.0 and unit_idx < len(units) - 1:
        bytes_value /= 1000.0
        unit_idx += 1
    return f"{bytes_value:.2f} {units[unit_idx]}"


def print_table_header(preset_width: int) -> None:
    print(
        f"  {'preset':<{preset_width}} {'enc':<20} {'time(ms)':>10} {'size':>12} {'bits':>14} {'ratio':>8}"
    )
    print(
        f"  {'-' * preset_width} {'-' * 20} {'-' * 10} {'-' * 12} {'-' * 14} {'-' * 8}"
    )


def print_table_row(row: dict, preset_width: int) -> None:
    print(
        f"  {row['preset']:<{preset_width}} "
        f"{row['encode_impl']:<20} "
        f"{row['total_ms']:>10.2f} "
        f"{format_bits(row['estimated_total_bits']):>12} "
        f"{row['estimated_total_bits']:>14} "
        f"{row['compression_ratio']:>8.4f}"
    )


def print_summary(rows: list[dict], top_n: int) -> None:
    print(f"Rows: {len(rows)}")

    by_kind = defaultdict(int)
    for row in rows:
        by_kind[row["input_kind"]] += 1
    print("By input kind:")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind}: {count}")

    by_file = defaultdict(list)
    for row in rows:
        by_file[row["file_path"]].append(row)

    print(f"\nFiles: {len(by_file)}")
    for file_path in sorted(by_file.keys()):
        file_rows = by_file[file_path]
        preset_width = max(32, max(len(row["preset"]) for row in file_rows))
        print(f"\n=== {file_path} ({len(file_rows)} runs) ===")

        fastest = min(file_rows, key=lambda r: r["total_ms"])
        smallest = min(file_rows, key=lambda r: r["estimated_total_bits"])
        print(
            "Best (time): "
            f"{fastest['preset']} | {fastest['encode_impl']} | total_ms={fastest['total_ms']:.2f} | "
            f"size={format_bits(fastest['estimated_total_bits'])} ({fastest['estimated_total_bits']} bits) | "
            f"ratio={fastest['compression_ratio']:.4f}"
        )
        print(
            "Best (size): "
            f"{smallest['preset']} | {smallest['encode_impl']} | total_ms={smallest['total_ms']:.2f} | "
            f"size={format_bits(smallest['estimated_total_bits'])} ({smallest['estimated_total_bits']} bits) | "
            f"ratio={smallest['compression_ratio']:.4f}"
        )

        print("Top by time:")
        print_table_header(preset_width)
        for row in sorted(file_rows, key=lambda r: r["total_ms"])[:top_n]:
            print_table_row(row, preset_width)

        print("Top by size:")
        print_table_header(preset_width)
        for row in sorted(file_rows, key=lambda r: r["estimated_total_bits"])[:top_n]:
            print_table_row(row, preset_width)

        # print("Direct compare (preset -> time/size):")
        # print_table_header(preset_width)
        # for row in sorted(file_rows, key=lambda r: (r["preset"], r["total_ms"])):
        #     print_table_row(row, preset_width)


def plot(rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception:
        print("matplotlib not available; skipping plot")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["input_kind"], row["file_path"])].append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    for (kind, file_path), kind_rows in grouped.items():
        x = [r["total_ms"] for r in kind_rows]
        y = [r["estimated_total_bits"] for r in kind_rows]
        label = f"{kind}:{Path(file_path).name}"
        ax.scatter(x, y, label=label, alpha=0.75)

    ax.set_title("Compression Tradeoff: total_ms vs estimated_total_bits")
    ax.set_xlabel("Total compression time [ms]")
    ax.set_ylabel("Estimated compressed size [bits / scaled bytes]")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value:.0f}b ({format_bits(int(max(value, 0)))})")
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> int:
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        print(f"Input file not found: {path}")
        return 1

    rows = load_rows(path)
    if not rows:
        print("No rows found")
        return 0

    print_summary(rows, args.top)
    if args.plot:
        plot(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
