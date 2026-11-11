#!/usr/bin/env python
"""
03_quality_control.py - image-quality and segmentation-robustness checks for the
widefield TIFF dataset (Technical Validation of the Data Descriptor).

For every TIFF in the deposit:
  * plls            power log-log slope of the radially averaged power spectrum
                    (CellProfiler MeasureImageQuality definition; more negative = smoother/blurrier;
                    flagged as outlier if outside 1.5xIQR within experiment x channel)
  * pct_saturated   % pixels at the camera full scale (4095 for the 12-bit MetaMorph originals)
  * pct_above_4095  % pixels > 4095 (non-zero only for files rescaled to 16 bit)
  * max_value, p999 maximum and 99.9th percentile of the raw pixel values
For every GFP/PKH26 pair (one spheroid):
  * dice_alt        Dice coefficient between the pipeline mask (Otsu on the sum of the two
                    percentile-normalised channels, 01_extract_spheroid_metrics.py) and an
                    alternative mask (Gaussian sigma=4 on the per-pixel maximum of the two channels,
                    Otsu, closing r=8, hole filling, brightest object >=1000 px)
  * area_ratio_alt  area(alternative)/area(pipeline)
  * area_change_thr_m10 / _p10  relative area change when the Otsu threshold is scaled by 0.9 / 1.1
  * pair_time_gap_s  seconds between the acquisition timestamps of the GFP and PKH26 frames of the pair
  * pair_exposure_typical  True if both frames use the modal exposure of their experiment x day x channel group
  * channel_centroid_offset  distance between the centroids of the GFP-positive and PKH26-positive pixels inside the
                    mask, in units of the equivalent radius (large values = laterally segregated populations)
  * channel_offset_flag  True if channel_centroid_offset exceeds the 95th percentile of the dataset (informative flag)
A contact sheet of all masks (cyan contour on the merged image) is written per experiment.

Usage:
  python 03_quality_control.py --input_root 02_raw_images/widefield --inventory 05_metadata/widefield_image_inventory.csv \
         --metrics 03_processed_data/spheroid_metrics_SKMEL28_control_strategyC.csv --out_dir 03_processed_data
"""
import argparse, os, re
from pathlib import Path
import numpy as np, pandas as pd, tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology, measure
from skimage.segmentation import find_boundaries
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ex01", HERE / "01_extract_spheroid_metrics.py")
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)

FNAME = re.compile(r"^(d\d)_([a-z0-9]+)_(gfp|pkh26)_(\d{3})\.tif$", re.I)

def plls(img):
    a = img.astype(np.float64); a -= a.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    h, w = P.shape; cy, cx = h // 2, w // 2
    yy, xx = np.indices(P.shape); r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rad = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    rmax = min(cy, cx); f = np.arange(1, rmax); p = rad[1:rmax]; ok = p > 0
    return float(np.polyfit(np.log(f[ok]), np.log(p[ok]), 1)[0])

def alt_segment(gn, rn, sigma=4, close_r=8, min_size=200, gate=1000):
    comb = np.maximum(gn, rn); s = filters.gaussian(comb, sigma); m = s > filters.threshold_otsu(s)
    m = morphology.remove_small_objects(m, min_size); m = morphology.binary_closing(m, morphology.disk(close_r))
    m = ndi.binary_fill_holes(m); m = morphology.remove_small_objects(m, min_size)
    lab = measure.label(m)
    if lab.max() == 0: return m
    props = measure.regionprops(lab, intensity_image=comb); cand = [p for p in props if p.area >= gate] or props
    return lab == max(cand, key=lambda p: p.mean_intensity).label

def largest(m, min_size=200):
    m = morphology.remove_small_objects(m, min_size); m = morphology.binary_opening(m, morphology.disk(3))
    m = morphology.binary_closing(m, morphology.disk(3)); m = morphology.remove_small_objects(m, min_size)
    lab = measure.label(m)
    return (lab == max(measure.regionprops(lab), key=lambda p: p.area).label) if lab.max() else m

def dice(a, b): return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True); ap.add_argument("--inventory", required=True)
    ap.add_argument("--metrics", required=True); ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(); root = Path(a.input_root); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- per-image quality ----
    inv = pd.read_csv(a.inventory)
    recs = []
    for fo, fn in zip(inv.folder, inv.filename):
        img = tifffile.imread(root / fo / fn); mx = int(img.max()); full = 4095 if mx <= 4095 else 65535
        recs.append(dict(folder=fo, filename=fn, plls=plls(img), pct_saturated=100 * (img >= full).mean(),
                         pct_above_4095=100 * (img > 4095).mean(), max_value=mx, p999=float(np.percentile(img, 99.9))))
    qc = pd.DataFrame(recs)
    inv = inv.drop(columns=[c for c in qc.columns if c in inv.columns and c not in ("folder", "filename")], errors="ignore")
    inv = inv.merge(qc, on=["folder", "filename"])
    rescaled = inv.max_value > 4095
    inv.loc[rescaled, ["pct_saturated", "pct_above_4095"]] = np.nan   # full scale unknown after rescaling
    def flag(g):
        q1, q3 = g.plls.quantile([.25, .75]); iqr = q3 - q1
        return (g.plls < q1 - 1.5 * iqr) | (g.plls > q3 + 1.5 * iqr)
    inv["plls_outlier"] = inv.groupby(["source_experiment", "channel"], group_keys=False).apply(flag)
    inv.to_csv(a.inventory, index=False)
    print(f"{len(inv)} images; {int((inv.pct_saturated > 0).sum())} with pixels at full scale; "
          f"{int(inv.plls_outlier.sum())} PLLS outliers; {int(rescaled.sum())} rescaled file(s)")

    # ---- per-spheroid segmentation robustness ----
    idx = {}
    for fo in inv.folder.unique():
        for f in os.listdir(root / fo):
            m = FNAME.match(f)
            if m:
                tp, s, ch, rep = m.groups(); idx[(fo, f"{tp}_{s}_{rep}".upper(), ch.upper())] = root / fo / f
    sm = pd.read_csv(a.metrics)
    inv["exposure_mode"] = inv.groupby(["source_experiment", "timepoint", "channel"])["exposure"].transform(lambda s: s.mode().iloc[0])
    inv_idx = {(fo, fn): row for fo, fn, row in zip(inv.folder, inv.filename, inv.to_dict("records"))}
    folder_of = dict(zip(inv.source_experiment, inv.folder))
    rows = []; masks = {}
    for _, r in sm.iterrows():
        fo = folder_of[r.source_experiment]; fb = r.file_base.upper().replace("FB2294", "2294")
        kg = (fo, fb, "GFP") if (fo, fb, "GFP") in idx else (fo, r.file_base.upper(), "GFP")
        g = ex.load_gray(idx[kg]); rr = ex.load_gray(idx[(kg[0], kg[1], "PKH26")])
        gn, rn = ex.normalize_percentile(g), ex.normalize_percentile(rr)
        m1 = ex.segment_spheroid(gn + rn, 200); m2 = alt_segment(gn, rn)
        s = gn + rn; thr = filters.threshold_otsu(s)
        a_m, a_p = largest(s > 0.9 * thr).sum(), largest(s > 1.1 * thr).sum()
        gpos = (gn > filters.threshold_otsu(gn[m1])) & m1; rpos = (rn > filters.threshold_otsu(rn[m1])) & m1
        cg = np.array(np.nonzero(gpos)).mean(1); cr = np.array(np.nonzero(rpos)).mean(1)
        offset = float(np.hypot(*(cg - cr)) / np.sqrt(m1.sum() / np.pi))
        ir = inv_idx[(fo, os.path.basename(str(idx[(kg[0], kg[1], "PKH26")])))]
        ig = inv_idx[(fo, os.path.basename(str(idx[kg])))]
        gap = abs((pd.to_datetime(ir["acquisition_time"], format="%Y%m%d %H:%M:%S.%f") - pd.to_datetime(ig["acquisition_time"], format="%Y%m%d %H:%M:%S.%f")).total_seconds())
        typical = bool(ig["exposure"] == ig["exposure_mode"] and ir["exposure"] == ir["exposure_mode"])
        rows.append(dict(source_experiment=r.source_experiment, file_base=r.file_base, area_pix=int(m1.sum()),
                         dice_alt=dice(m1, m2), area_ratio_alt=m2.sum() / m1.sum(),
                         area_change_thr_m10=a_m / m1.sum() - 1, area_change_thr_p10=a_p / m1.sum() - 1,
                         pair_time_gap_s=gap, pair_exposure_typical=typical, channel_centroid_offset=offset))
        masks[(r.source_experiment, r.file_base)] = (gn, rn, m1)
    qs = pd.DataFrame(rows)
    qs["channel_offset_flag"] = qs.channel_centroid_offset > qs.channel_centroid_offset.quantile(0.95)
    qs.to_csv(out / "segmentation_robustness_per_spheroid.csv", index=False)
    print(f"pair time gap: median {qs.pair_time_gap_s.median():.0f} s, max {qs.pair_time_gap_s.max():.0f} s; "
          f"{int((~qs.pair_exposure_typical).sum())} pair(s) with non-modal exposure; "
          f"channel centroid offset median {qs.channel_centroid_offset.median():.3f}, 95th pct {qs.channel_centroid_offset.quantile(0.95):.3f}, "
          f"{int(qs.channel_offset_flag.sum())} flagged")
    print(f"Dice vs alternative: median {qs.dice_alt.median():.3f} (min {qs.dice_alt.min():.3f}); "
          f"median |area change| for +/-10% threshold: {qs[['area_change_thr_m10','area_change_thr_p10']].abs().median().mean():.3f}")

    # ---- contact sheets ----
    od = out / "qc_overlays_widefield"; od.mkdir(exist_ok=True); qsi = qs.set_index(["source_experiment", "file_base"])
    for e in sm.source_experiment.unique():
        sub = sm[sm.source_experiment == e].sort_values("file_base"); n = len(sub); nc = 8; nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(nc * 2.2, nr * 2.0 + 0.5))
        for ax, (_, r) in zip(axs.ravel(), sub.iterrows()):
            gn, rn, m1 = masks[(e, r.file_base)]
            rgb = np.dstack([np.clip(rn, 0, 1), np.clip(gn, 0, 1), np.zeros_like(gn)])
            ys, xs = np.where(m1); pad = 60
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, m1.shape[0]); x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, m1.shape[1])
            rgb = rgb[y0:y1, x0:x1].copy(); rgb[find_boundaries(m1[y0:y1, x0:x1], mode="outer")] = [0, 1, 1]
            flag = " ▲" if qsi.loc[(e, r.file_base), "channel_offset_flag"] else ""
            ax.imshow(rgb); ax.set_title(f"{r.file_base}  Dice {qsi.loc[(e, r.file_base), 'dice_alt']:.2f}{flag}", fontsize=6); ax.axis("off")
        for ax in axs.ravel()[n:]: ax.axis("off")
        fig.suptitle(f"{e}: pipeline mask (cyan) on merged GFP (green) / PKH26 (red), percentile-normalised; "
                     "Dice = agreement with the alternative segmentation rule; ▲ = channel centroid offset above the 95th percentile", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(od / f"qc_overlays_{e}.png", dpi=110); plt.close(fig)

if __name__ == "__main__":
    main()
