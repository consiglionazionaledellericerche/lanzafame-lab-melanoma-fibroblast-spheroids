#!/usr/bin/env python3
"""
01b_consolidate_experiments.py

Concatenate the per-experiment outputs of 01_extract_spheroid_metrics.py into
the two consolidated files deposited with the Scientific Data Data Descriptor.
Rows are ordered by source_experiment, timepoint, sample, file_base.

Usage:
    python 01b_consolidate_experiments.py --inputs out/20250728_C out/20251021_C out/20251030_C \
        --output_dir deposit/03_processed_data --suffix SKMEL28_control_strategyC

CSV format: standard comma-separated, period decimal (pandas defaults)
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Per-experiment output folders.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--suffix", default="SKMEL28_control_strategyC")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for name in ("spheroid_metrics", "radial_profiles"):
        parts = [pd.read_csv(Path(d) / f"{name}.csv") for d in args.inputs]
        df = pd.concat(parts, ignore_index=True)
        keys = [k for k in ("source_experiment", "timepoint", "sample", "file_base", "bin") if k in df.columns]
        df = df.sort_values(keys, kind="stable").reset_index(drop=True)
        path = out / f"{name}_{args.suffix}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
