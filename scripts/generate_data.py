#!/usr/bin/env python3
"""
Generate synthetic CSV data files with configurable parameters.

Usage:
    python generate_data.py --rows 10000 --features 8 --output data.csv
    python generate_data.py --rows 5000 --features 4 --labels 5 --dtype int --output int_data.csv
"""

import argparse
import csv
import random
import math
from pathlib import Path


def generate_gaussian(num_rows: int, num_features: int) -> list:
    """Generate data with Gaussian distribution."""
    data = []
    for _ in range(num_rows):
        features = [random.gauss(mu=-15 + i, sigma=2.0) for i in range(num_features)]
        data.append(features)
    return data


def generate_uniform(num_rows: int, num_features: int) -> list:
    """Generate data with uniform distribution."""
    data = []
    for _ in range(num_rows):
        features = [random.uniform(-50, 50) for _ in range(num_features)]
        data.append(features)
    return data


def generate_integer(num_rows: int, num_features: int) -> list:
    """Generate integer data."""
    data = []
    for _ in range(num_rows):
        features = [random.randint(0, 255) for _ in range(num_features)]
        data.append(features)
    return data


def generate_binary(num_rows: int, num_features: int) -> list:
    """Generate binary (0/1) data."""
    data = []
    for _ in range(num_rows):
        features = [random.randint(0, 1) for _ in range(num_features)]
        data.append(features)
    return data


def generate_sparse(num_rows: int, num_features: int, sparsity: float = 0.8) -> list:
    """Generate sparse data (mostly zeros)."""
    data = []
    for _ in range(num_rows):
        features = [
            random.gauss(0, 10) if random.random() > sparsity else 0.0
            for _ in range(num_features)
        ]
        data.append(features)
    return data


def generate_categorical(
    num_rows: int, num_features: int, categories: int = 10
) -> list:
    """Generate categorical integer data (like encoded categories)."""
    data = []
    for _ in range(num_rows):
        features = [random.randint(0, categories - 1) for _ in range(num_features)]
        data.append(features)
    return data


def generate_mixed(num_rows: int, num_features: int) -> list:
    """Generate mixed data types (alternating float/int)."""
    data = []
    for _ in range(num_rows):
        features = []
        for i in range(num_features):
            if i % 2 == 0:
                features.append(random.gauss(0, 10))
            else:
                features.append(random.randint(-100, 100))
        data.append(features)
    return data


def generate_sinusoidal(num_rows: int, num_features: int) -> list:
    """Generate data with sinusoidal patterns."""
    data = []
    for row in range(num_rows):
        t = row / num_rows * 2 * math.pi * 10  # 10 full cycles
        features = [
            math.sin(t + i * math.pi / num_features) * 10 + random.gauss(0, 1)
            for i in range(num_features)
        ]
        data.append(features)
    return data


GENERATORS = {
    "gaussian": generate_gaussian,
    "uniform": generate_uniform,
    "int": generate_integer,
    "binary": generate_binary,
    "sparse": generate_sparse,
    "categorical": generate_categorical,
    "mixed": generate_mixed,
    "sinusoidal": generate_sinusoidal,
}


def write_csv(data: list, output_path: Path, precision: int = 4):
    """Write data to CSV file."""
    num_features = len(data[0])
    header = [f"f{i+1}" for i in range(num_features)]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for features in data:
            row = []
            for feat in features:
                if isinstance(feat, float):
                    row.append(round(feat, precision))
                else:
                    row.append(feat)
            writer.writerow(row)

    print(f"Generated {len(data)} rows with {num_features} features -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CSV data files")
    parser.add_argument(
        "--rows",
        "-r",
        type=int,
        default=10000,
        help="Number of rows to generate (default: 10000)",
    )
    parser.add_argument(
        "--features", "-f", type=int, default=8, help="Number of features (default: 8)"
    )
    parser.add_argument(
        "--dtype",
        "-d",
        type=str,
        default="gaussian",
        choices=list(GENERATORS.keys()),
        help=f"Data type/distribution (default: gaussian)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: data-{rows}-{features}.csv)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--precision",
        "-p",
        type=int,
        default=4,
        help="Decimal precision for floats (default: 4)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Generate output filename if not provided
    if args.output is None:
        output_path = Path(f"data/data-{args.rows}-{args.features}-{args.dtype}.csv")
    else:
        output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate data
    generator = GENERATORS[args.dtype]
    data = generator(args.rows, args.features)

    # Write to file
    write_csv(data, output_path, args.precision)


if __name__ == "__main__":
    main()
