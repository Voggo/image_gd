#!/usr/bin/env python3
"""
Generate synthetic integer CSV data with configurable bit widths and distributions.

Usage:
    python generate_data.py --rows 10000 --features 8 --bits 8 --distribution uniform
    python generate_data.py --rows 5000 --features 4 --bits "8,16,8,16" --distribution gaussian
    python generate_data.py --rows 10000 --features 8 --bits 8 --distribution poisson --output data.csv
"""

import argparse
import csv
import random
import math
from pathlib import Path
from typing import List, Callable


def generate_uniform_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate data with uniform distribution across bit range for each feature."""
    data = []
    for _ in range(num_rows):
        features = [
            random.randint(0, (1 << bits) - 1) for bits in bits_per_feature
        ]
        data.append(features)
    return data


def generate_gaussian_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate data with Gaussian distribution, clipped to bit range."""
    data = []
    for feature_idx, bits in enumerate(bits_per_feature):
        max_val = (1 << bits) - 1
        mean = max_val / 2
        sigma = max_val / 6  # ~99.7% within range
    
    for _ in range(num_rows):
        features = []
        for bits in bits_per_feature:
            max_val = (1 << bits) - 1
            mean = max_val / 2
            sigma = max_val / 6
            val = int(random.gauss(mean, sigma))
            val = max(0, min(val, max_val))  # Clamp to valid range
            features.append(val)
        data.append(features)
    return data


def generate_exponential_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate data with exponential distribution."""
    data = []
    for _ in range(num_rows):
        features = []
        for bits in bits_per_feature:
            max_val = (1 << bits) - 1
            val = int(random.expovariate(1.0 / (max_val / 2)) % (max_val + 1))
            features.append(val)
        data.append(features)
    return data


def generate_poisson_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate data with Poisson distribution."""
    data = []
    for _ in range(num_rows):
        features = []
        for bits in bits_per_feature:
            max_val = (1 << bits) - 1
            lambda_param = max_val / 2
            val = min(random.randint(0, max_val), int(random.expovariate(1.0 / lambda_param)))
            features.append(val)
        data.append(features)
    return data


def generate_beta_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate data with Beta distribution (biased towards extremes)."""
    data = []
    for _ in range(num_rows):
        features = []
        for bits in bits_per_feature:
            max_val = (1 << bits) - 1
            val = int(random.betavariate(0.5, 0.5) * max_val)
            features.append(val)
        data.append(features)
    return data


def generate_zipf_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate data with Zipf distribution (power-law, biased towards small values)."""
    data = []
    for _ in range(num_rows):
        features = []
        for bits in bits_per_feature:
            max_val = (1 << bits) - 1
            # Simple Zipf approximation
            z = random.zipfian(1.5, max_val) if hasattr(random, 'zipfian') else random.randint(0, max_val)
            features.append(int(z))
        data.append(features)
    return data


def generate_sparse_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate sparse data with 80% zeros and 20% small values (high compression potential)."""
    data = []
    for _ in range(num_rows):
        features = []
        for bits in bits_per_feature:
            max_val = (1 << bits) - 1
            # 80% chance of 0, 20% chance of small value
            if random.random() < 0.8:
                val = 0
            else:
                # When non-zero, use only low bits (0-15 for 8-bit)
                val = random.randint(1, min(15, max_val))
            features.append(val)
        data.append(features)
    return data


def generate_skewed_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate heavily skewed data concentrated in lower values."""
    data = []
    for _ in range(num_rows):
        features = []
        for bits in bits_per_feature:
            max_val = (1 << bits) - 1
            # Use exponential with bias towards 0
            val = int(random.expovariate(2.0 / max_val))
            val = min(val, max_val)
            features.append(val)
        data.append(features)
    return data


def generate_binary_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate binary-like data: mostly 0 or max_val."""
    data = []
    for _ in range(num_rows):
        features = []
        for bits in bits_per_feature:
            max_val = (1 << bits) - 1
            # 70% zeros, 30% max value
            val = max_val if random.random() < 0.3 else 0
            features.append(val)
        data.append(features)
    return data


def generate_low_entropy_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate data using only a few distinct patterns for minimal entropy."""
    data = []
    # Define a small set of allowed values (e.g., only 4-5 patterns per feature)
    patterns = {}
    for bits in bits_per_feature:
        max_val = (1 << bits) - 1
        # Use only 4 evenly spaced patterns
        patterns[bits] = [0, max_val // 3, 2 * max_val // 3, max_val]
    
    for _ in range(num_rows):
        features = []
        for bits in bits_per_feature:
            val = random.choice(patterns[bits])
            features.append(val)
        data.append(features)
    return data


def generate_gradient_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate data with repeating gradient patterns."""
    data = []
    pattern_length = 20  # Repeat every 20 rows
    
    for row in range(num_rows):
        features = []
        pattern_pos = row % pattern_length
        
        for bits in bits_per_feature:
            max_val = (1 << bits) - 1
            # Create a gradient value that repeats
            val = int((pattern_pos / pattern_length) * max_val)
            # Add small noise
            val = max(0, min(val + random.randint(-2, 2), max_val))
            features.append(val)
        data.append(features)
    return data


def generate_repeated_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate data with many repeated values and common patterns."""
    data = []
    # Create a small set of "templates" that repeat often
    template_size = min(10, num_rows // 100)
    templates = []
    for _ in range(template_size):
        template = [random.randint(0, (1 << bits) - 1) for bits in bits_per_feature]
        templates.append(template)
    
    for i in range(num_rows):
        # 70% of time, repeat one of the templates; 30% of time, create new data
        if random.random() < 0.7 or len(templates) == 0:
            features = random.choice(templates).copy() if templates else [random.randint(0, (1 << bits) - 1) for bits in bits_per_feature]
            # Add minor variations to repeated templates
            features = [max(0, min(f + random.randint(-1, 1), (1 << bits) - 1)) for f, bits in zip(features, bits_per_feature)]
        else:
            features = [random.randint(0, (1 << bits) - 1) for bits in bits_per_feature]
        data.append(features)
    return data


def generate_clustered_dist(num_rows: int, num_features: int, bits_per_feature: List[int]) -> list:
    """Generate clustered data around random centers."""
    data = []
    # Pick random cluster centers
    centers = [[random.randint(0, (1 << bits) - 1) for bits in bits_per_feature] for _ in range(min(5, num_features))]
    
    for _ in range(num_rows):
        center = random.choice(centers)
        features = []
        for bits, center_val in zip(bits_per_feature, center):
            max_val = (1 << bits) - 1
            # Add noise around center
            val = int(random.gauss(center_val, max_val / 8))
            val = max(0, min(val, max_val))
            features.append(val)
        data.append(features)
    return data


DISTRIBUTIONS = {
    "uniform": generate_uniform_dist,
    "gaussian": generate_gaussian_dist,
    "exponential": generate_exponential_dist,
    "poisson": generate_poisson_dist,
    "beta": generate_beta_dist,
    "zipf": generate_zipf_dist,
    "clustered": generate_clustered_dist,
    "sparse": generate_sparse_dist,
    "skewed": generate_skewed_dist,
    "binary": generate_binary_dist,
    "low_entropy": generate_low_entropy_dist,
    "gradient": generate_gradient_dist,
    "repeated": generate_repeated_dist,
}


def parse_bits_argument(bits_arg: str, num_features: int) -> List[int]:
    """
    Parse bits argument which can be:
    - Single integer: "8" -> [8, 8, 8, ...]
    - Comma-separated: "8,16,8" -> [8, 16, 8]
    """
    if "," in bits_arg:
        bits_list = [int(b.strip()) for b in bits_arg.split(",")]
        if len(bits_list) != num_features:
            raise ValueError(
                f"Number of bit specifications ({len(bits_list)}) must match number of features ({num_features})"
            )
        return bits_list
    else:
        bits = int(bits_arg)
        if bits < 1 or bits > 64:
            raise ValueError("Bits per feature must be between 1 and 64")
        return [bits] * num_features


def write_csv(data: list, output_path: Path, bits_per_feature: List[int] = None):
    """Write integer data to CSV file."""
    num_features = len(data[0])
    header = [f"f{i+1}" for i in range(num_features)]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    bits_info = f" ({', '.join(map(str, bits_per_feature))} bits)" if bits_per_feature else ""
    print(f"Generated {len(data)} rows with {num_features} features{bits_info} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic integer CSV data with configurable bit widths and distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_data.py --rows 10000 --features 8 --bits 8 --distribution uniform
  python generate_data.py --rows 5000 --features 4 --bits "8,16,8,16" --distribution gaussian
  python generate_data.py --rows 10000 --features 8 --bits 8 --distribution poisson --output custom.csv

Available distributions: """ + ", ".join(DISTRIBUTIONS.keys())
    )
    
    parser.add_argument(
        "--rows",
        "-r",
        type=int,
        default=10000,
        help="Number of rows to generate (default: 10000)",
    )
    parser.add_argument(
        "--features",
        "-f",
        type=int,
        default=8,
        help="Number of features (default: 8)",
    )
    parser.add_argument(
        "--bits",
        "-b",
        type=str,
        default="8",
        help="Bits per feature: single value '8' or comma-separated '8,16,8' (default: 8)",
    )
    parser.add_argument(
        "--distribution",
        "-d",
        type=str,
        default="uniform",
        choices=list(DISTRIBUTIONS.keys()),
        help="Root distribution (default: uniform)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: data/data-{rows}-{features}-{bits}-{distribution}.csv)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Parse bits argument
    try:
        bits_per_feature = parse_bits_argument(args.bits, args.features)
    except ValueError as e:
        parser.error(str(e))

    # Generate output filename if not provided
    if args.output is None:
        bits_str = "-".join(map(str, bits_per_feature)) if len(set(bits_per_feature)) > 1 else str(bits_per_feature[0])
        output_path = Path(
            f"data/data-{args.rows}-{args.features}-{bits_str}bit-{args.distribution}.csv"
        )
    else:
        output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate data
    generator = DISTRIBUTIONS[args.distribution]
    data = generator(args.rows, args.features, bits_per_feature)

    # Write to file
    write_csv(data, output_path, bits_per_feature)


if __name__ == "__main__":
    main()
