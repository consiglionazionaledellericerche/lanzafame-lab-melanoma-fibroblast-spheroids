#!/usr/bin/env python3
"""
02_compute_core_shell_metrics.py

Compute cell-type-resolved architecture metrics from radial profiles.

Derives:
    - Intensity-weighted radial centroids (mu_mel, mu_fb)
    - Centroid separation (delta_mu = mu_mel - mu_fb)
    - Core/shell enrichment fractions (r <= 0.30 / r >= 0.70)
    - Core enrichment score (FB_core - MEL_core)

Input:
    Radial profile CSV(s), either:
      - one file per experiment/strategy, named "{date}_{strategy_code}_radial_profiles.csv"
        (strategy inferred from the filename), or
      - a single consolidated file (e.g. the Scientific Data deposit's
        "radial_profiles_SKMEL28_control_strategyC.csv") carrying a
        'source_experiment' column that encodes "{date}_{strategy_code}"
        (strategy inferred from that column instead)
    Both layouts are produced, directly or after consolidation, by
    01_extract_spheroid_metrics.py.

Output:
    core_shell_metrics.csv — per-spheroid architecture summary
    (standard comma-separated, period decimal — same convention as the other
    deposited CSVs)

Reference:
    Veniali et al. (2026) Fluorescence imaging dataset of melanoma fibroblast
    co-spheroids with morphometric and radial profiling data. Scientific Data
    (Data Descriptor; submitted).

Usage:
    python 02_compute_core_shell_metrics.py --data_dir /path/to/csvs --output_dir /path/to/output
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


STRATEGY_MAP = {'C': 'Mixed'}  # strategy code in source_experiment labels; C = simultaneous (mixed) co-seeding
CORE_THRESHOLD = 0.30
SHELL_THRESHOLD = 0.70


def load_radial_profiles(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all radial profile CSVs with experiment/strategy metadata.

    Supports both the per-experiment filename convention
    ("{date}_{strategy_code}_radial_profiles.csv") and the consolidated
    Scientific Data deposit file, which instead carries a 'source_experiment'
    column encoding "{date}_{strategy_code}" per row.
    """
    files = sorted(data_dir.glob("*radial_profiles*.csv"))
    if not files:
        raise FileNotFoundError(f"No radial profile CSVs found in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'source_experiment' in df.columns:
            # Consolidated deposit file: derive experiment/strategy per row
            # from source_experiment ("{date}_{strategy_code}").
            split = df['source_experiment'].str.rsplit('_', n=1)
            df['experiment'] = split.str[0]
            strategy_code = split.str[1]
            df['strategy'] = strategy_code.map(STRATEGY_MAP).fillna(strategy_code)
        else:
            parts = f.stem.split('_')
            df['experiment'] = parts[0]
            strategy_code = parts[1]
            df['strategy'] = STRATEGY_MAP.get(strategy_code, strategy_code)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def compute_core_shell(radial_df: pd.DataFrame) -> pd.DataFrame:
    """Compute architecture metrics per spheroid from radial profiles.

    Parameters
    ----------
    radial_df : pd.DataFrame
        Combined radial profiles with columns: file_base, sample, group,
        timepoint, experiment, strategy, bin, r_mid, ring_area_pix,
        g_int_mean, r_int_mean.

    Returns
    -------
    pd.DataFrame
        Per-spheroid architecture metrics.
    """
    results: List[Dict] = []
    grouped = radial_df.groupby(['file_base', 'experiment', 'strategy'])

    for (fb, exp, strat), grp in grouped:
        grp = grp.sort_values('bin')
        r_mid = grp['r_mid'].values
        g_int = grp['g_int_mean'].values
        r_int = grp['r_int_mean'].values
        ring_area = grp['ring_area_pix'].values

        # Intensity-weighted radial centroids
        g_weight = g_int * ring_area
        r_weight = r_int * ring_area
        g_total = g_weight.sum()
        r_total = r_weight.sum()

        if g_total > 0 and r_total > 0:
            mu_mel = (r_mid * g_weight).sum() / g_total
            mu_fb = (r_mid * r_weight).sum() / r_total
            delta_mu = mu_mel - mu_fb
        else:
            mu_mel = mu_fb = delta_mu = np.nan

        # Core (r <= 0.30) and shell (r >= 0.70) fractions
        core_mask = r_mid <= CORE_THRESHOLD
        shell_mask = r_mid >= SHELL_THRESHOLD

        g_core = (g_int[core_mask] * ring_area[core_mask]).sum() / g_total if g_total > 0 else np.nan
        r_core = (r_int[core_mask] * ring_area[core_mask]).sum() / r_total if r_total > 0 else np.nan
        g_shell = (g_int[shell_mask] * ring_area[shell_mask]).sum() / g_total if g_total > 0 else np.nan
        r_shell = (r_int[shell_mask] * ring_area[shell_mask]).sum() / r_total if r_total > 0 else np.nan

        core_enrichment = (r_core - g_core) if np.isfinite(r_core) and np.isfinite(g_core) else np.nan

        row = {}
        if 'source_experiment' in grp.columns:
            row['source_experiment'] = grp['source_experiment'].iloc[0]
        row.update({
            'file_base': fb,
            'sample': grp['sample'].iloc[0],
            'group': grp['group'].iloc[0],
            'timepoint': grp['timepoint'].iloc[0],
        })
        # Optional provenance columns written by 01_extract_spheroid_metrics.py
        for extra in ('timepoint_nominal', 'acquisition_time', 'days_post_seeding'):
            if extra in grp.columns:
                row[extra] = grp[extra].iloc[0]
        row.update({
            'experiment': exp,
            'strategy': strat,
            'mu_mel': mu_mel,
            'mu_fb': mu_fb,
            'delta_mu': delta_mu,
            'mel_core_frac': g_core,
            'fb_core_frac': r_core,
            'mel_shell_frac': g_shell,
            'fb_shell_frac': r_shell,
            'core_enrichment': core_enrichment,
        })
        results.append(row)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compute core-shell architecture metrics from radial profiles."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing *_radial_profiles.csv files.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as data_dir).")
    parser.add_argument("--controls_only", action="store_true",
                        help="Restrict to control samples (C5BO, C3PV, Fb1609).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    print(f"Loading radial profiles from: {data_dir}")
    df_radial = load_radial_profiles(data_dir)

    if args.controls_only:
        df_radial = df_radial[df_radial['group'] == 'Control']
        print(f"Filtered to controls: {len(df_radial)} rows")

    print("Computing core-shell metrics...")
    df_cs = compute_core_shell(df_radial)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "core_shell_metrics.csv"
    df_cs.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Spheroid-timepoints: {len(df_cs)}")

    # Summary
    d7 = df_cs[df_cs['timepoint'] == 'D7']
    if len(d7) > 0:
        print("\n--- D7 endpoint summary ---")
        for strat in sorted(d7['strategy'].dropna().unique()):
            sub = d7[d7['strategy'] == strat]
            if len(sub) > 0:
                dm = sub['delta_mu'].dropna()
                pct_pos = (dm > 0).sum() / len(dm) * 100
                print(f"  {strat:12s}: n={len(sub):3d}, "
                      f"Dmu median={dm.median():.4f} "
                      f"(IQR {dm.quantile(0.25):.4f}-{dm.quantile(0.75):.4f}), "
                      f"{pct_pos:.0f}% Dmu>0")


if __name__ == "__main__":
    main()
