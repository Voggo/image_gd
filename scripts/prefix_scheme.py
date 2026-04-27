#!/usr/bin/env python3
"""Analyze tiered prefix+payload schemes from delta bit-length histograms.

This script helps choose an approximate optimal prefix encoding by combining:
- exact DP minimization of expected coded bits for each tier count,
- diminishing-returns (knee) detection,
- a complexity-aware objective that discourages too many tiers.

Input CSV format must include at least:
  bit_len,count
Extra columns are ignored.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class TierSegment:
    tier_index: int
    bit_min: int
    bit_max: int
    count: int


@dataclass(frozen=True)
class PlanResult:
    mode: str  # "unary" or "fixed"
    prefix_bits: int  # fixed prefix bits, 0 for unary
    n_tiers: int
    total_bits: int
    ratio_vs_lb: float
    avg_bits_per_delta: float
    payload_maxima: Tuple[int, ...]
    segments: Tuple[TierSegment, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find approximate optimal tiered prefix encodings from bit-length histograms."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=(
            "target/base_table_delta_compression/"
            "joel-filipe-187166.png__sorted__unsigned__cs-SrgbWithLinearAlpha"
            "__cm-YCoCgR__pg-4__gt-ForFirstPixel.csv"
        ),
        help="Path to histogram CSV with bit_len,count columns.",
    )
    parser.add_argument(
        "--max-unary-tiers",
        type=int,
        default=16,
        help="Maximum unary tier count to evaluate.",
    )
    parser.add_argument(
        "--max-fixed-prefix-bits",
        type=int,
        default=6,
        help="Evaluate fixed prefixes from 1..this value.",
    )
    parser.add_argument(
        "--knee-min-improvement-pct",
        type=float,
        default=0.20,
        help=(
            "Knee threshold in percent: when incremental improvement drops below this percent, "
            "extra tiers are likely not worth it."
        ),
    )
    parser.add_argument(
        "--complexity-penalty-pct",
        type=float,
        default=0.05,
        help=(
            "Percent of lower-bound bits to penalize per tier in complexity-aware ranking. "
            "Higher -> prefers fewer tiers."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many top candidates to print in each recommendation table.",
    )
    parser.add_argument(
        "--show-segments-for",
        type=str,
        default="best",
        choices=["none", "best", "knee", "all"],
        help="Print tier coverage segments for selected solutions.",
    )
    return parser.parse_args()


def load_histogram(path: Path) -> Dict[int, int]:
    histogram: Dict[int, int] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"bit_len", "count"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            bit_len = int(row["bit_len"])
            count = int(row["count"])
            if count <= 0:
                continue
            histogram[bit_len] = histogram.get(bit_len, 0) + count

    if not histogram:
        raise ValueError("Histogram is empty after parsing counts.")
    return histogram


def build_prefix_sums(values: Sequence[int]) -> List[int]:
    out = [0]
    s = 0
    for v in values:
        s += v
        out.append(s)
    return out


def range_sum(prefix: Sequence[int], left: int, right: int) -> int:
    return prefix[right + 1] - prefix[left]


def optimize_for_tiers(
    bit_lens: Sequence[int],
    counts: Sequence[int],
    count_prefix: Sequence[int],
    n_tiers: int,
    mode: str,
    fixed_prefix_bits: int,
) -> Tuple[int, Tuple[int, ...], Tuple[TierSegment, ...]]:
    n = len(bit_lens)
    if n_tiers < 1 or n_tiers > n:
        raise ValueError("n_tiers must be between 1 and number of distinct bit lengths")

    inf = 10**30

    def prefix_cost(tier_idx: int) -> int:
        if mode == "unary":
            return tier_idx + 1
        return fixed_prefix_bits

    def group_cost(i: int, j: int, tier_idx: int) -> int:
        payload_bits = bit_lens[j]
        group_count = range_sum(count_prefix, i, j)
        return (prefix_cost(tier_idx) + payload_bits) * group_count

    dp = [[inf] * n for _ in range(n_tiers)]
    prev = [[-1] * n for _ in range(n_tiers)]

    for j in range(n):
        dp[0][j] = group_cost(0, j, 0)

    for k in range(1, n_tiers):
        for j in range(k, n):
            best = inf
            best_i = -1
            for i in range(k, j + 1):
                c = dp[k - 1][i - 1] + group_cost(i, j, k)
                if c < best:
                    best = c
                    best_i = i
            dp[k][j] = best
            prev[k][j] = best_i

    boundaries: List[Tuple[int, int]] = []
    j = n - 1
    for k in range(n_tiers - 1, -1, -1):
        i = prev[k][j] if k > 0 else 0
        boundaries.append((i, j))
        j = i - 1
    boundaries.reverse()

    payloads: List[int] = []
    segments: List[TierSegment] = []
    for tier_idx, (i, j) in enumerate(boundaries):
        payloads.append(bit_lens[j])
        segments.append(
            TierSegment(
                tier_index=tier_idx,
                bit_min=bit_lens[i],
                bit_max=bit_lens[j],
                count=range_sum(count_prefix, i, j),
            )
        )

    return dp[n_tiers - 1][n - 1], tuple(payloads), tuple(segments)


def evaluate_candidates(
    bit_lens: Sequence[int],
    counts: Sequence[int],
    lb_bits: int,
    total_count: int,
    max_unary_tiers: int,
    max_fixed_prefix_bits: int,
) -> List[PlanResult]:
    count_prefix = build_prefix_sums(counts)
    n = len(bit_lens)
    results: List[PlanResult] = []

    unary_limit = min(max_unary_tiers, n)
    for tiers in range(1, unary_limit + 1):
        bits, payloads, segments = optimize_for_tiers(
            bit_lens,
            counts,
            count_prefix,
            tiers,
            mode="unary",
            fixed_prefix_bits=0,
        )
        results.append(
            PlanResult(
                mode="unary",
                prefix_bits=0,
                n_tiers=tiers,
                total_bits=bits,
                ratio_vs_lb=bits / lb_bits,
                avg_bits_per_delta=bits / total_count,
                payload_maxima=payloads,
                segments=segments,
            )
        )

    for pb in range(1, max_fixed_prefix_bits + 1):
        fixed_limit = min(1 << pb, n)
        for tiers in range(1, fixed_limit + 1):
            bits, payloads, segments = optimize_for_tiers(
                bit_lens,
                counts,
                count_prefix,
                tiers,
                mode="fixed",
                fixed_prefix_bits=pb,
            )
            results.append(
                PlanResult(
                    mode="fixed",
                    prefix_bits=pb,
                    n_tiers=tiers,
                    total_bits=bits,
                    ratio_vs_lb=bits / lb_bits,
                    avg_bits_per_delta=bits / total_count,
                    payload_maxima=payloads,
                    segments=segments,
                )
            )

    return results


def describe_mode(plan: PlanResult) -> str:
    if plan.mode == "unary":
        return "unary"
    return f"fixed/{plan.prefix_bits}b"


def find_knee_by_family(
    plans: Sequence[PlanResult],
    min_improvement_pct: float,
) -> Dict[Tuple[str, int], PlanResult]:
    families: Dict[Tuple[str, int], List[PlanResult]] = {}
    for p in plans:
        key = (p.mode, p.prefix_bits)
        families.setdefault(key, []).append(p)

    knees: Dict[Tuple[str, int], PlanResult] = {}
    for key, family in families.items():
        family_sorted = sorted(family, key=lambda x: x.n_tiers)
        best_so_far = family_sorted[0]
        knee = best_so_far

        for idx in range(1, len(family_sorted)):
            prev = family_sorted[idx - 1]
            cur = family_sorted[idx]
            gain = prev.total_bits - cur.total_bits
            gain_pct = 100.0 * gain / prev.total_bits if prev.total_bits > 0 else 0.0
            if gain_pct < min_improvement_pct:
                knee = prev
                break
            if cur.total_bits < best_so_far.total_bits:
                best_so_far = cur
                knee = cur

        knees[key] = knee

    return knees


def complexity_aware_best(
    plans: Sequence[PlanResult],
    lb_bits: int,
    penalty_pct_per_tier: float,
) -> List[Tuple[float, PlanResult]]:
    penalty_bits_per_tier = lb_bits * (penalty_pct_per_tier / 100.0)
    scored: List[Tuple[float, PlanResult]] = []
    for p in plans:
        score = p.total_bits + penalty_bits_per_tier * p.n_tiers
        scored.append((score, p))
    scored.sort(key=lambda x: x[0])
    return scored


def print_header(input_path: Path, histogram: Dict[int, int]) -> Tuple[int, int]:
    total = sum(histogram.values())
    lb_bits = sum(bit_len * c for bit_len, c in histogram.items())
    min_bit = min(histogram)
    max_bit = max(histogram)
    distinct = len(histogram)

    print(f"Input CSV      : {input_path}")
    print(f"Distinct bins  : {distinct} (range {min_bit}..{max_bit})")
    print(f"Total deltas   : {total}")
    print(f"Lower bound    : {lb_bits} bits ({lb_bits / 8.0:.1f} bytes)")
    print()

    return total, lb_bits


def print_top_by_bits(plans: Sequence[PlanResult], top_k: int) -> None:
    print("Best by expected coded bits:")
    print(
        "  {:<10} {:>5} {:>12} {:>9} {:>9}  {}".format(
            "scheme", "tiers", "bits", "ratio", "avg", "payload maxima"
        )
    )
    for p in sorted(plans, key=lambda x: x.total_bits)[:top_k]:
        print(
            "  {:<10} {:>5} {:>12} {:>9.4f} {:>9.3f}  {}".format(
                describe_mode(p),
                p.n_tiers,
                p.total_bits,
                p.ratio_vs_lb,
                p.avg_bits_per_delta,
                list(p.payload_maxima),
            )
        )
    print()


def print_knee_summary(
    plans: Sequence[PlanResult],
    knees: Dict[Tuple[str, int], PlanResult],
) -> None:
    print("Knee-point candidates (first diminishing-return point per scheme family):")
    print(
        "  {:<10} {:>5} {:>12} {:>9} {:>9}  {}".format(
            "scheme", "tiers", "bits", "ratio", "avg", "payload maxima"
        )
    )

    for key in sorted(knees.keys(), key=lambda x: (x[0], x[1])):
        p = knees[key]
        print(
            "  {:<10} {:>5} {:>12} {:>9.4f} {:>9.3f}  {}".format(
                describe_mode(p),
                p.n_tiers,
                p.total_bits,
                p.ratio_vs_lb,
                p.avg_bits_per_delta,
                list(p.payload_maxima),
            )
        )
    print()


def print_complexity_ranking(
    scored: Sequence[Tuple[float, PlanResult]],
    top_k: int,
) -> None:
    print("Complexity-aware ranking (bits + tier penalty):")
    print(
        "  {:<10} {:>5} {:>12} {:>12} {:>9}  {}".format(
            "scheme", "tiers", "bits", "score", "ratio", "payload maxima"
        )
    )
    for score, p in scored[:top_k]:
        print(
            "  {:<10} {:>5} {:>12} {:>12.1f} {:>9.4f}  {}".format(
                describe_mode(p),
                p.n_tiers,
                p.total_bits,
                score,
                p.ratio_vs_lb,
                list(p.payload_maxima),
            )
        )
    print()


def print_plan_segments(plan: PlanResult) -> None:
    print(
        f"Tier segments for {describe_mode(plan)} with {plan.n_tiers} tiers "
        f"(bits={plan.total_bits}, ratio={plan.ratio_vs_lb:.4f}):"
    )
    print("  {:>5} {:>12} {:>12} {:>12}".format("tier", "bit_min", "bit_max", "count"))
    for seg in plan.segments:
        print(
            "  {:>5} {:>12} {:>12} {:>12}".format(
                seg.tier_index, seg.bit_min, seg.bit_max, seg.count
            )
        )
    print()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)

    histogram = load_histogram(input_path)
    total_count, lb_bits = print_header(input_path, histogram)

    bit_lens = sorted(histogram)
    counts = [histogram[b] for b in bit_lens]

    plans = evaluate_candidates(
        bit_lens=bit_lens,
        counts=counts,
        lb_bits=lb_bits,
        total_count=total_count,
        max_unary_tiers=args.max_unary_tiers,
        max_fixed_prefix_bits=args.max_fixed_prefix_bits,
    )

    if not plans:
        raise RuntimeError("No plans generated. Check your input and limits.")

    knees = find_knee_by_family(plans, args.knee_min_improvement_pct)
    complexity_scored = complexity_aware_best(
        plans,
        lb_bits=lb_bits,
        penalty_pct_per_tier=args.complexity_penalty_pct,
    )

    best_by_bits = min(plans, key=lambda p: p.total_bits)
    best_by_complexity = complexity_scored[0][1]

    print_top_by_bits(plans, args.top_k)
    print_knee_summary(plans, knees)
    print_complexity_ranking(complexity_scored, args.top_k)

    print("Recommended starting points:")
    print(
        f"  1) Pure-bit optimum      : {describe_mode(best_by_bits)} "
        f"tiers={best_by_bits.n_tiers}, ratio={best_by_bits.ratio_vs_lb:.4f}"
    )
    print(
        f"  2) Complexity-aware best : {describe_mode(best_by_complexity)} "
        f"tiers={best_by_complexity.n_tiers}, ratio={best_by_complexity.ratio_vs_lb:.4f}"
    )

    unary_knee = knees.get(("unary", 0))
    if unary_knee is not None:
        print(
            f"  3) Unary knee            : tiers={unary_knee.n_tiers}, "
            f"ratio={unary_knee.ratio_vs_lb:.4f}"
        )

    fixed_knees = [knees[k] for k in sorted(knees) if k[0] == "fixed"]
    if fixed_knees:
        best_fixed_knee = min(fixed_knees, key=lambda p: p.total_bits)
        print(
            f"  4) Best fixed knee       : {describe_mode(best_fixed_knee)} "
            f"tiers={best_fixed_knee.n_tiers}, ratio={best_fixed_knee.ratio_vs_lb:.4f}"
        )
    print()

    if args.show_segments_for in ("best", "all"):
        print_plan_segments(best_by_bits)
    if args.show_segments_for in ("knee", "all"):
        if unary_knee is not None:
            print_plan_segments(unary_knee)
        if fixed_knees:
            best_fixed_knee = min(fixed_knees, key=lambda p: p.total_bits)
            print_plan_segments(best_fixed_knee)


if __name__ == "__main__":
    main()
