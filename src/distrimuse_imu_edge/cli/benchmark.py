from __future__ import annotations

import argparse

from distrimuse_imu_edge.evaluation.aggregate import aggregate_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate IMU edge benchmark results.")
    parser.add_argument("--results-dir", default="experiments/results")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = aggregate_results(args.results_dir)
    if df.empty:
        print("No runs found.")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
