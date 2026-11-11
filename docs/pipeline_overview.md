# Pipeline Overview

Documentation of the code used for the Data Descriptor *Fluorescence imaging dataset of melanoma fibroblast co-spheroids with morphometric and radial profiling data* (Veniali et al., Scientific Data, submitted).

## Image Analysis Workflow

### Input
Paired widefield fluorescence images (.tif) of heterotypic spheroids:
- **GFP channel** (column prefix `g_`): SK-MEL-28-GFP melanoma cells
- **PKH26 channel** (column prefix `r_`): PKH26-labeled dermal fibroblasts

### Segmentation
1. Each channel is contrast-normalized using 1st-99th percentile clipping
2. Normalized channels are summed to produce a composite image
3. Global Otsu thresholding identifies the spheroid region
4. Morphological opening/closing and small-object removal refine the mask
5. Border-touching objects are cleared
6. The largest connected component is retained as the spheroid ROI

### Morphometrics
Per-spheroid shape descriptors computed from the binary mask:
- **Area** (pixels and um^2)
- **Perimeter** (pixels and um)
- **Equivalent radius** (um)
- **Circularity** = 4*pi*area/perimeter^2
- **Solidity** = area / convex hull area
- **Eccentricity** (0 = circle, 1 = line)
- **Aspect ratio** = major axis / minor axis

### Channel-resolved descriptors
- **Area fraction**: proportion of positive pixels per channel (Otsu within mask)
- **Radial mean**: mean normalized distance of positive pixels
- **Core compaction**: proportion of total signal in core region (r < 0.5)
- **Intensity-weighted radial mean**: centroid of intensity distribution
- **Jaccard overlap**: spatial overlap between channel-positive masks

### Radial Profiling
1. Spheroid centroid and equivalent radius define a normalized radial coordinate r in [0, 1]
2. The spheroid is partitioned into 20 concentric rings
3. Per ring: mean intensity for each channel, positive pixel fraction, ring area

## Architecture Metrics

Derived from radial profiles:

### Centroid Separation (Delta_mu)
Intensity-weighted radial centroid for each channel:
    mu = sum(r_mid * I_ring * A_ring) / sum(I_ring * A_ring)

Delta_mu = mu_MEL - mu_FB

Interpretation:
- Delta_mu > 0: fibroblast-enriched core, melanoma-enriched shell
- Delta_mu ~ 0: mixed/intermingled
- Delta_mu < 0: melanoma-enriched core

### Core Enrichment
Core fraction = integrated signal within r <= 0.30 / total signal
Core enrichment = FB_core_frac - MEL_core_frac

Positive values indicate preferential fibroblast accumulation in the spheroid interior.

## File Naming Convention

### Input images
```
{timepoint}_{sample}_{channel}_{replicate}.tif
```
Example: `D7_C5BO_GFP_002.tif` / `D7_C5BO_PKH26_002.tif` (case-insensitive; the donor tokens are C5BO, C3PV and Fb1609).

### Output CSVs
One `spheroid_metrics.csv` and one `radial_profiles.csv` per experiment (folder given with `--output_dir`), concatenated into the deposited files by `01b_consolidate_experiments.py` (see below).

CSV format: standard comma separator, period decimal.


## Reproducing the deposited processed data (Scientific Data deposit)

```bash
python scripts/01_extract_spheroid_metrics.py --input_dir 02_raw_images/widefield/20250728_D1-D7 --output_dir out/20250728_C --source_experiment 20250728_C --seeding_date 2025-07-28
python scripts/01_extract_spheroid_metrics.py --input_dir 02_raw_images/widefield/20251021_D1-D7 --output_dir out/20251021_C --source_experiment 20251021_C --seeding_date 2025-10-21 --nominal_map D6:D5
python scripts/01_extract_spheroid_metrics.py --input_dir 02_raw_images/widefield/20251030_D0-D7 --output_dir out/20251030_C --source_experiment 20251030_C --seeding_date 2025-10-30 --nominal_map D4:D3
python scripts/01b_consolidate_experiments.py --inputs out/20250728_C out/20251021_C out/20251030_C --output_dir 03_processed_data --suffix SKMEL28_control_strategyC
python scripts/02_compute_core_shell_metrics.py --data_dir 03_processed_data --output_dir 03_processed_data
mv 03_processed_data/core_shell_metrics.csv 03_processed_data/core_shell_metrics_SKMEL28_control_strategyC.csv
```

Expected output: 146 spheroids (41 + 32 + 73), 2,920 radial-profile rows; every deposited spheroid has one GFP and one PKH26 image (292 files). If a folder contains an unpaired single-channel image, the script skips it with a warning.

`03_quality_control.py` produces the Technical Validation numbers: per-image power log-log slope and saturation (written into `05_metadata/widefield_image_inventory.csv`, columns `plls`, `plls_outlier`, `pct_saturated`, `pct_above_4095`, `max_value`, `p999`), per-spheroid Dice agreement with an alternative segmentation rule, threshold sensitivity, GFP/PKH26 pair consistency (time gap, exposure) and channel centroid offset (`03_processed_data/segmentation_robustness_per_spheroid.csv`) and contact sheets of all masks (`03_processed_data/qc_overlays_widefield/`).

```bash
python scripts/03_quality_control.py --input_root 02_raw_images/widefield --inventory 05_metadata/widefield_image_inventory.csv \
       --metrics 03_processed_data/spheroid_metrics_SKMEL28_control_strategyC.csv --out_dir 03_processed_data
```

`process_confocal_images.py` processes the six Leica `.lif` confocal stacks (maximum-intensity projection, modality-specific segmentation: max of channels → Gaussian σ 4 px → Otsu → closing r 12 px → hole fill → largest component; metric definitions identical to the widefield pipeline). Deposited outputs:

```bash
python scripts/process_confocal_images.py --input_dir 02_raw_images/confocal --output_dir 03_processed_data/confocal \
    --timepoint D7 --sample_map "C3PV_1:C3PV,C3PV_2:C3PV,C5BO_1:C5BO,C5BO_2:C5BO,Fb1609_1:Fb1609,Fb1609_2:Fb1609" \
    --experiment_map "C3PV_1:20250728_C,C3PV_2:20250728_C,C5BO_1:20251021_C,C5BO_2:20251021_C,Fb1609_1:20251021_C,Fb1609_2:20251021_C"
```

`--experiment_map` writes the `source_experiment` of each stack: `20250728_C` for C3PV_1 and C3PV_2, `20251021_C` for the C5BO and Fb1609 stacks.

Perimeter is measured with an OpenCV contour, so `perim_*` and `circularity` are not on the same numerical scale as the widefield CSVs.

`convert_lif_to_ome_tiff.py` exports the same raw z-stack of each `.lif` (selected by the 2-channel, z > 1 signature; channel order checked from the LUT names) as a lossless OME-TIFF in `02_raw_images/confocal/ome_tiff/` (CZYX, 8-bit, zlib; PhysicalSizeX/Y/Z in µm, channel names, acquisition description). Re-running it reproduces the deposited files (identical pixels and OME-XML apart from the random UUID); `--verify` checks them without writing.

```bash
python scripts/convert_lif_to_ome_tiff.py --input_dir 02_raw_images/confocal --output_dir 02_raw_images/confocal/ome_tiff
```
