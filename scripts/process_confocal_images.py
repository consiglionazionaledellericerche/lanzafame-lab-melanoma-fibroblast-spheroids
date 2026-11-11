#!/usr/bin/env python3
"""
process_confocal_images.py

Image analysis pipeline for the six confocal z-stack co-spheroid images
(Leica .lif, control-donor / mixed-strategy subset). Companion to
01_extract_spheroid_metrics.py: extracts per-spheroid morphometrics and
radial intensity profiles, in the same conceptual format as the widefield
outputs, so this matched subset can support the widefield-vs-confocal
comparison described in the Data Descriptor (Methods, "Confocal fluorescence
imaging").

Raw stack selection
--------------------
Each .lif file contains one real 2-channel acquisition z-stack per spheroid
PLUS several LAS X-generated preview/3D-projection series (RGB renders at
various sizes, not raw data). The acquisition stack is identified by its
signature (exactly 2 channels, z > 1); preview series always have 3 channels
and are excluded. Channel order is read from the embedded LUTName metadata
(ChannelDescription elements) rather than assumed: channel with LUTName
"Green" = melanoma/GFP, channel with LUTName "Red" = fibroblast/PKH26 —
confirmed identical (Green first, Red second) across all six files.

Z-projection
------------
Each channel is reduced from a z-stack to a single 2D image by maximum-
intensity projection (MIP) before segmentation and radial profiling. This
makes the output directly comparable to the 2D widefield metrics, which is
the matched subset's stated purpose. Z-step and z-range vary per spheroid
(80-106 optical sections; not stored as one explicit field) — a genuine 3D
metric would require additional per-spheroid depth calibration that is out
of scope here and is not attempted.

Segmentation and morphometrics
-------------------------------
Follows 01_extract_spheroid_metrics.py (percentile normalisation, Otsu
threshold, morphological refinement, largest-component selection, 20-bin
normalised radial profiling, identical metric definitions) with ONE
deliberate difference in the segmentation step, required by the modality:
confocal MIPs resolve individual cells, so an Otsu threshold on the raw
summed image follows the bright fibroblast-rich core and leaves out the
dimmer, cell-textured melanoma rim (masks with circularity 0.06-0.14 in
testing). The confocal mask is therefore obtained from the per-pixel MAXIMUM
of the two normalised channels (either population marks the spheroid),
smoothed with a Gaussian (sigma = 4 px by default) before Otsu thresholding,
then closed with a disk (radius 12 px by default), hole-filled, and reduced to
the largest connected component. Parameters are exposed as --smooth_sigma and
--close_radius. Tested on the six deposited stacks: masks cover the whole
spheroid outline (circularity 0.47-0.65). Implemented with OpenCV + NumPy only
(no scikit-image dependency in this environment): Otsu thresholding is a
manual histogram-based implementation equivalent to skimage's; perimeter is
computed from an OpenCV contour (cv2.arcLength) rather than skimage's
crack-length formula, which gives a slightly different perimeter/circularity
convention than the widefield CSVs — see the data dictionary and Usage Notes.
This is a genuinely different modality and pipeline (2D confocal MIP vs.
direct 2D widefield), not a bug: absolute circularity values from the two
should not be treated as on the same numerical scale.

Sample / timepoint provenance
-----------------------------
The deposited .lif files are named {DONOR}_{n} (C3PV_1, C3PV_2, C5BO_1,
C5BO_2, Fb2294_1, Fb2294_2); the file stem is written as spheroid_id. Donor
and timepoint are not read from the file metadata. They are supplied on the
command line: --timepoint (one value for all stacks; the deposited spheroids
were all fixed at day 7) and --sample_map ("C3PV_1:C3PV,C5BO_1:C5BO,...");
unmapped IDs are written as "NA". --source_experiment labels the experiment
the spheroids came from; --experiment_map ("C3PV_1:20250728_C,...") overrides
it per spheroid when the stacks come from different experiments.

Outputs (standard comma-separated CSV, period decimal, matching the deposit convention):
    confocal_spheroid_metrics.csv   — one row per spheroid (6 total)
    confocal_radial_profiles.csv    — 20 rows per spheroid (120 total)

Usage:
    python process_confocal_images.py --input_dir /path/to/lif_files --output_dir /path/to/output
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from readlif.reader import LifFile


# ======================================================================
# OTSU THRESHOLD (manual, histogram-based — equivalent to skimage.filters.threshold_otsu)
# ======================================================================

def threshold_otsu_manual(values: np.ndarray, nbins: int = 256) -> float:
    """Histogram-based Otsu threshold, matching skimage's algorithm."""
    values = values[np.isfinite(values)]
    hist, bin_edges = np.histogram(values, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist = hist.astype(np.float64)

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    variance12 = np.nan_to_num(variance12, nan=-1.0)
    idx = int(np.argmax(variance12))
    return float(bin_centers[idx])


# ======================================================================
# LIF LOADING
# ======================================================================

def load_confocal_mip(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load a .lif file, select the true 2-channel z-stack, and return
    (mip_green, mip_red, pixel_size_um).

    Raises if zero or more than one image in the file matches the raw-stack
    signature (channels == 2 and z > 1), so an unexpected file layout fails
    loudly instead of silently picking the wrong series.
    """
    lif = LifFile(str(path))
    candidates = []
    for i, meta in enumerate(lif.image_list):
        img = lif.get_image(i)
        if img.channels == 2 and img.dims.z > 1:
            candidates.append((i, img))

    if len(candidates) != 1:
        raise RuntimeError(
            f"{path}: expected exactly 1 raw 2-channel z-stack series, found {len(candidates)}"
        )
    idx, img = candidates[0]

    # Confirm channel->dye mapping from embedded LUTName metadata rather than assuming.
    lut_names = []
    for elem in lif.xml_root.iter():
        if elem.tag.split('}')[-1] == 'ChannelDescription':
            lut_names.append(elem.attrib.get('LUTName', ''))
    if len(lut_names) < 2 or lut_names[0] != 'Green' or lut_names[1] != 'Red':
        raise RuntimeError(
            f"{path}: unexpected channel LUT order {lut_names!r} "
            f"(expected ['Green', 'Red', ...] for channel 0=melanoma, 1=fibroblast)"
        )

    z = img.dims.z
    mip_g = None
    mip_r = None
    for zi in range(z):
        fg = np.array(img.get_frame(z=zi, c=0))
        fr = np.array(img.get_frame(z=zi, c=1))
        if mip_g is None:
            mip_g = fg.copy()
            mip_r = fr.copy()
        else:
            np.maximum(mip_g, fg, out=mip_g)
            np.maximum(mip_r, fr, out=mip_r)

    scale = img.scale  # (px/um_x, px/um_y, px/um_z, None)
    pixel_size_um = 1.0 / scale[0]

    return mip_g.astype(np.float32), mip_r.astype(np.float32), pixel_size_um


# ======================================================================
# IMAGE PROCESSING (OpenCV port of 01_extract_spheroid_metrics.py's approach)
# ======================================================================

def normalize_percentile(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def segment_spheroid(g_norm: np.ndarray, r_norm: np.ndarray, min_obj_size: int = 500,
                     smooth_sigma: float = 4.0, close_radius: int = 12) -> np.ndarray:
    """Confocal-MIP segmentation (see module docstring): max of the two
    normalised channels -> Gaussian smoothing -> Otsu -> small-object removal
    -> closing -> hole filling -> largest component."""
    comb = np.maximum(g_norm, r_norm).astype(np.float32)
    if smooth_sigma > 0:
        comb = cv2.GaussianBlur(comb, (0, 0), smooth_sigma)
    positive = comb > 0
    if not np.any(positive):
        return np.zeros_like(comb, dtype=bool)

    thr = threshold_otsu_manual(comb[positive])
    mask = (comb >= thr).astype(np.uint8)

    k = 2 * int(close_radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # fill holes
    flood = mask.copy()
    h0, w0 = flood.shape
    ff_mask = np.zeros((h0 + 2, w0 + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    mask = (mask | (1 - flood)).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask, dtype=bool)

    # remove small objects
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < min_obj_size:
            labels[labels == lbl] = 0

    # clear border-touching components
    h, w = mask.shape
    border_labels = set(labels[0, :]) | set(labels[h - 1, :]) | set(labels[:, 0]) | set(labels[:, w - 1])
    border_labels.discard(0)
    for lbl in border_labels:
        labels[labels == lbl] = 0

    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return labels == keep


def positive_mask_otsu(norm_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = norm_img[mask]
    vals = vals[vals > 0]
    if vals.size > 10:
        thr = threshold_otsu_manual(vals)
        return (norm_img >= thr) & mask
    return (norm_img > 0) & mask


def contour_of(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# ======================================================================
# METRICS & RADIAL PROFILES
# ======================================================================

def compute_metrics_and_profiles(
    spheroid_id: str,
    g_img: np.ndarray,
    r_img: np.ndarray,
    pixel_size_um: float,
    core_r_cutoff: float = 0.5,
    n_radial_bins: int = 20,
    min_object_size: int = 500,
    smooth_sigma: float = 4.0,
    close_radius: int = 12,
    sample: str = "NA",
    timepoint: str = "NA",
    source_experiment: str = "NA",
) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
    g_norm = normalize_percentile(g_img)
    r_norm = normalize_percentile(r_img)
    sum_img = g_norm + r_norm

    mask = segment_spheroid(g_norm, r_norm, min_object_size, smooth_sigma, close_radius)
    if not np.any(mask):
        print(f"[WARN] No valid mask for spheroid {spheroid_id}")
        return None, None

    coords = np.column_stack(np.nonzero(mask))
    centroid = coords.mean(axis=0)
    dists = np.sqrt(((coords - centroid) ** 2).sum(axis=1))

    area_pix = int(mask.sum())
    radius_pix = float(np.sqrt(area_pix / np.pi))
    radius_um = radius_pix * pixel_size_um
    area_um2 = float(area_pix) * (pixel_size_um ** 2)

    r_norm_map = np.zeros_like(sum_img, dtype=np.float32)
    if radius_pix > 0:
        r_norm_map[mask] = dists / radius_pix
    r_norm_map = np.clip(r_norm_map, 0, 1.0)

    contour = contour_of(mask)
    if contour is not None and len(contour) >= 5:
        perim_pix = float(cv2.arcLength(contour, closed=True))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(area_pix / hull_area) if hull_area > 0 else np.nan
        try:
            (_, _), (ma, mb), _ = cv2.fitEllipse(contour)
            major, minor = max(ma, mb), min(ma, mb)
            aspect_ratio = major / minor if minor > 0 else np.nan
            eccentricity = float(np.sqrt(1 - (minor / major) ** 2)) if major > 0 else np.nan
        except cv2.error:
            aspect_ratio = eccentricity = np.nan
    else:
        perim_pix = np.nan
        solidity = eccentricity = aspect_ratio = np.nan

    perim_um = perim_pix * pixel_size_um if np.isfinite(perim_pix) else np.nan
    circularity = float(4.0 * np.pi * area_pix / (perim_pix ** 2)) if np.isfinite(perim_pix) and perim_pix > 0 else np.nan

    g_pos = positive_mask_otsu(g_norm, mask)
    r_pos = positive_mask_otsu(r_norm, mask)

    spheroid_area = float(area_pix)
    g_area_frac = float(g_pos.sum()) / spheroid_area if spheroid_area > 0 else np.nan
    r_area_frac = float(r_pos.sum()) / spheroid_area if spheroid_area > 0 else np.nan

    g_r_mean = float(r_norm_map[g_pos].mean()) if g_pos.any() else np.nan
    r_r_mean = float(r_norm_map[r_pos].mean()) if r_pos.any() else np.nan
    delta_r = (g_r_mean - r_r_mean) if np.isfinite(g_r_mean) and np.isfinite(r_r_mean) else np.nan

    core_mask = (r_norm_map < core_r_cutoff) & mask
    total_int = float((g_norm + r_norm)[mask].sum())
    core_int = float((g_norm + r_norm)[core_mask].sum())
    core_compaction = core_int / total_int if total_int > 0 else np.nan

    g_total = float(g_norm[mask].sum())
    r_total = float(r_norm[mask].sum())
    g_core = float(g_norm[core_mask].sum())
    r_core = float(r_norm[core_mask].sum())

    g_core_frac = g_core / g_total if g_total > 0 else np.nan
    r_core_frac = r_core / r_total if r_total > 0 else np.nan
    core_bias = (g_core_frac - r_core_frac) if np.isfinite(g_core_frac) and np.isfinite(r_core_frac) else np.nan

    rvals = r_norm_map[mask]
    gvals = g_norm[mask]
    rvals2 = r_norm[mask]

    g_r_wmean = float((rvals * gvals).sum() / gvals.sum()) if gvals.sum() > 0 else np.nan
    r_r_wmean = float((rvals * rvals2).sum() / rvals2.sum()) if rvals2.sum() > 0 else np.nan
    delta_r_w = (g_r_wmean - r_r_wmean) if np.isfinite(g_r_wmean) and np.isfinite(r_r_wmean) else np.nan

    inter = float((g_pos & r_pos).sum())
    union = float((g_pos | r_pos).sum())
    pos_jaccard = inter / union if union > 0 else np.nan

    shell_mask = mask & (~core_mask)
    g_mean_core = float(g_norm[core_mask].mean()) if core_mask.any() else np.nan
    g_mean_shell = float(g_norm[shell_mask].mean()) if shell_mask.any() else np.nan
    r_mean_core = float(r_norm[core_mask].mean()) if core_mask.any() else np.nan
    r_mean_shell = float(r_norm[shell_mask].mean()) if shell_mask.any() else np.nan

    group = "Control" if sample.upper() in ("C5BO", "C3PV", "FB2294", "2294") else ("NA" if sample == "NA" else "Other")
    metrics = {
        "source_experiment": source_experiment,
        "spheroid_id": spheroid_id,
        "sample": sample,
        "group": group,
        "timepoint": timepoint,
        "area_pix": area_pix,
        "area_um2": area_um2,
        "perim_pix": perim_pix,
        "perim_um": perim_um,
        "radius_um": radius_um,
        "circularity": circularity,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "aspect_ratio": aspect_ratio,
        "core_compaction": core_compaction,
        "g_r_mean": g_r_mean,
        "r_r_mean": r_r_mean,
        "delta_r": delta_r,
        "g_area_frac": g_area_frac,
        "r_area_frac": r_area_frac,
        "g_core_frac": g_core_frac,
        "r_core_frac": r_core_frac,
        "core_bias": core_bias,
        "g_r_wmean": g_r_wmean,
        "r_r_wmean": r_r_wmean,
        "delta_r_w": delta_r_w,
        "pos_jaccard": pos_jaccard,
        "g_mean_core": g_mean_core,
        "g_mean_shell": g_mean_shell,
        "r_mean_core": r_mean_core,
        "r_mean_shell": r_mean_shell,
    }

    edges = np.linspace(0.0, 1.0, n_radial_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    profiles: List[Dict] = []
    for i in range(n_radial_bins):
        lo, hi = edges[i], edges[i + 1]
        # r_norm_map is clipped to a max of 1.0 (any pixel farther from the
        # centroid than the equivalent circular radius collapses to exactly
        # 1.0 - this happens for essentially every non-perfectly-circular
        # mask). The final bin's upper edge is therefore inclusive (<=),
        # otherwise every pixel piled up at r_norm_map == 1.0 falls outside
        # all 20 bins and is silently dropped from the radial profile - not
        # just under-counted in the last bin, but absent from the profile
        # entirely. Bins 1-19 keep the exclusive '<' (adjacent-bin boundary
        # hits there are a measure-zero edge case, not a systematic effect).
        # Same fix as 01_extract_spheroid_metrics.py's widefield pipeline.
        is_last_bin = (i == n_radial_bins - 1)
        if is_last_bin:
            ring = mask & (r_norm_map >= lo) & (r_norm_map <= hi)
        else:
            ring = mask & (r_norm_map >= lo) & (r_norm_map < hi)
        ring_n = int(ring.sum())

        if ring_n == 0:
            g_pos_frac = r_pos_frac = np.nan
            g_int_mean = r_int_mean = np.nan
        else:
            g_pos_frac = float(g_pos[ring].sum()) / ring_n
            r_pos_frac = float(r_pos[ring].sum()) / ring_n
            g_int_mean = float(g_norm[ring].mean())
            r_int_mean = float(r_norm[ring].mean())

        profiles.append({
            "source_experiment": source_experiment,
            "spheroid_id": spheroid_id,
            "sample": sample,
            "group": group,
            "timepoint": timepoint,
            "bin": i + 1,
            "r_mid": float(mids[i]),
            "ring_area_pix": ring_n,
            "g_pos_frac": g_pos_frac,
            "r_pos_frac": r_pos_frac,
            "g_int_mean": g_int_mean,
            "r_int_mean": r_int_mean,
        })

    return metrics, profiles


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract spheroid metrics and radial profiles from confocal .lif z-stacks (MIP-based)."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--core_r_cutoff", type=float, default=0.5)
    parser.add_argument("--min_object_size", type=int, default=500)
    parser.add_argument("--smooth_sigma", type=float, default=4.0,
                        help="Gaussian sigma (px) applied before Otsu thresholding (default 4).")
    parser.add_argument("--close_radius", type=int, default=12,
                        help="Disk radius (px) of the morphological closing (default 12).")
    parser.add_argument("--timepoint", type=str, default="NA", help="Day label for all stacks, e.g. D7.")
    parser.add_argument("--sample_map", type=str, default="",
                        help="Comma-separated 'ID:DONOR' pairs, e.g. 'C3PV_1:C3PV,C5BO_1:C5BO'.")
    parser.add_argument("--source_experiment", type=str, default="NA")
    parser.add_argument("--experiment_map", type=str, default="",
                        help="Comma-separated 'ID:EXPERIMENT' pairs; overrides --source_experiment per spheroid.")
    args = parser.parse_args()
    sample_map = dict(kv.split(":") for kv in args.sample_map.split(",") if kv.strip())
    experiment_map = dict(kv.split(":") for kv in args.experiment_map.split(",") if kv.strip())

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    rows_metrics: List[Dict] = []
    rows_profiles: List[Dict] = []

    for f in sorted(input_dir.glob("*.lif"), key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.stem)):
        spheroid_id = f.stem
        print(f"Processing spheroid {spheroid_id} ({f.name})")
        g_img, r_img, pixel_size_um = load_confocal_mip(f)
        print(f"  pixel_size_um={pixel_size_um:.4f}")

        metrics, prof = compute_metrics_and_profiles(
            spheroid_id, g_img, r_img, pixel_size_um,
            core_r_cutoff=args.core_r_cutoff,
            n_radial_bins=args.n_bins,
            min_object_size=args.min_object_size,
            smooth_sigma=args.smooth_sigma,
            close_radius=args.close_radius,
            sample=sample_map.get(spheroid_id, "NA"),
            timepoint=args.timepoint,
            source_experiment=experiment_map.get(spheroid_id, args.source_experiment),
        )
        if metrics is not None:
            rows_metrics.append(metrics)
        if prof is not None:
            rows_profiles.extend(prof)

    if not rows_metrics:
        print("No spheroids processed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "confocal_spheroid_metrics.csv"
    profile_path = output_dir / "confocal_radial_profiles.csv"

    pd.DataFrame(rows_metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(rows_profiles).to_csv(profile_path, index=False)

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {profile_path}")
    print(f"Spheroids analyzed: {len(rows_metrics)}")
    print(f"Radial profile rows: {len(rows_profiles)} ({len(rows_metrics)} x {args.n_bins})")


if __name__ == "__main__":
    main()
