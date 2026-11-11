# melanoma-fibroblast-spheroids

Image-analysis pipeline for the dataset described in:

**Veniali G., Galli Balestreri E., Orioli D. & Lanzafame M. (2026). *Fluorescence imaging dataset of melanoma fibroblast co-spheroids with morphometric and radial profiling data.* Scientific Data (Data Descriptor, submitted).**

- Dataset: BioImage Archive, accession S-BIAD4346, <https://doi.org/10.6019/S-BIAD4346>
- Source code: <https://github.com/consiglionazionaledellericerche/lanzafame-lab-melanoma-fibroblast-spheroids>
- Archived code, version 1.0.0: Zenodo, <https://doi.org/10.5281/zenodo.24133753>
- Browser tool for single-spheroid analysis: <https://lanzafame-lab-spheroid-core-shell-metrics.onrender.com/> (separate publication in preparation)
- Laboratory web page: <https://www.igm.cnr.it/istituto-2/organizzazione/personale/lanzafame-manuela/>


---

## Scope

This repository contains the code used to generate the processed data deposited with the *Scientific Data* Data Descriptor: SK-MEL-28 melanoma cells co-seeded with primary dermal fibroblasts from three control donors (C5BO, C3PV, Fb2294), imaged by widefield (three experiments, days 1-7) and confocal microscopy. `01_extract_spheroid_metrics.py`, `01b_consolidate_experiments.py`, `02_compute_core_shell_metrics.py`, `process_confocal_images.py`, `convert_lif_to_ome_tiff.py` and `03_quality_control.py` reproduce the deposited files.

---

## Overview

This repository provides the complete computational workflow for:

1. **Image analysis** — segmenting multicellular spheroids from widefield fluorescence images and extracting per-spheroid morphometrics and cell-type-resolved radial intensity profiles.
2. **Core-shell quantification** — computing intensity-weighted radial centroids, centroid separation (Δμ), and core/shell enrichment scores from radial profiles.
3. **Confocal processing** — the same metrics on maximum-intensity projections of the six confocal stacks.
4. **Quality control** — image-quality and segmentation-robustness checks reported in Technical Validation.

## Repository structure

```
melanoma-fibroblast-spheroids/
├── scripts/
│   ├── 01_extract_spheroid_metrics.py   # Image analysis pipeline (widefield)
│   ├── 01b_consolidate_experiments.py   # Concatenate per-experiment outputs
│   ├── 02_compute_core_shell_metrics.py # Core-shell metrics from radial profiles
│   ├── 03_quality_control.py            # Image-quality (PLLS, saturation) and segmentation-robustness checks
│   ├── process_confocal_images.py       # Confocal .lif stacks → MIP metrics (see docs)
│   └── convert_lif_to_ome_tiff.py       # Confocal .lif stacks → lossless OME-TIFF
├── docs/
│   └── pipeline_overview.md             # Detailed pipeline documentation
├── requirements.txt
├── CITATION.cff                         # Citation metadata
├── publiccode.yml                       # Metadata for the Developers Italia reuse catalogue
├── LICENSE
└── README.md
```

## Pipeline

### Step 1: Image analysis (`01_extract_spheroid_metrics.py`)

Processes paired GFP/PKH26 `.tif` images from a specified directory (one experiment folder per run) and produces two CSV files per experiment. Channel images are paired by filename case-insensitively; every record carries the provenance fields `source_experiment`, `timepoint` (actual acquisition day, from the filename), `timepoint_nominal`, `acquisition_time` (from the MetaMorph TIFF header) and `days_post_seeding`.

| Output | Description |
|--------|-------------|
| `*_spheroid_metrics.csv` | Per-spheroid morphometrics (area, circularity, solidity, eccentricity, aspect ratio) and channel-resolved summary descriptors (area fractions, radial means, core compaction, Jaccard overlap) |
| `*_radial_profiles.csv` | Normalized concentric-ring (r = 0–1, 20 bins) intensity distributions for melanoma (GFP, column prefix `g_`) and fibroblast (PKH26, column prefix `r_`) channels |

**Usage:**
```bash
python scripts/01_extract_spheroid_metrics.py --input_dir /path/to/20251030_D0-D7 --output_dir out/20251030_C \
    --source_experiment 20251030_C --seeding_date 2025-10-30 --nominal_map D4:D3
```

The exact commands used for the deposited dataset are listed in `docs/pipeline_overview.md`. The per-experiment outputs are concatenated into the deposited files with:

```bash
python scripts/01b_consolidate_experiments.py --inputs out/20250728_C out/20251021_C out/20251030_C \
    --output_dir 03_processed_data --suffix SKMEL28_control_strategyC
```

**Key parameters:**
- `--pixel_size` : pixel size in µm (default: 1.62)
- `--n_bins` : number of radial bins (default: 20)
- `--core_r_cutoff` : core radius cutoff for legacy metrics (default: 0.5)
- `--min_object_size` : minimum object size in pixels (default: 200)
- `--source_experiment` : label written to the `source_experiment` column (default: input folder name)
- `--seeding_date` : seeding date (YYYY-MM-DD) for `days_post_seeding`
- `--nominal_map` : `ACTUAL:NOMINAL` day-label pairs, e.g. `D4:D3,D6:D5`, for `timepoint_nominal`

### Step 2: Core-shell metrics (`02_compute_core_shell_metrics.py`)

Derives architecture metrics from radial profiles:

- **Centroid separation** (Δμ = μ_MEL − μ_FB): intensity-weighted radial centroids; Δμ > 0 indicates fibroblast-core / melanoma-shell
- **Core enrichment** (FB_core − MEL_core): proportion of integrated signal within r ≤ 0.30
- **Shell enrichment**: proportion within r ≥ 0.70

**Usage:**
```bash
python scripts/02_compute_core_shell_metrics.py --data_dir /path/to/csvs --output_dir /path/to/output
```

The output is written as `core_shell_metrics.csv`; for the deposit it is renamed `core_shell_metrics_SKMEL28_control_strategyC.csv` (see `docs/pipeline_overview.md`).

### Confocal stacks (`process_confocal_images.py`)

Reads the primary two-channel z-stack of each Leica `.lif` file (requires `readlif`), computes maximum-intensity projections and applies the same metric definitions with a modality-specific segmentation (maximum of the normalised channels → Gaussian σ 4 px → Otsu → closing r 12 px → hole filling → largest component). Perimeter is measured from an OpenCV contour, so `perim_*` and `circularity` are not on the same numerical scale as the widefield files.

```bash
python scripts/process_confocal_images.py --input_dir 02_raw_images/confocal --output_dir 03_processed_data/confocal \
    --timepoint D7 --sample_map "C3PV_1:C3PV,C3PV_2:C3PV,C5BO_1:C5BO,C5BO_2:C5BO,Fb2294_1:Fb2294,Fb2294_2:Fb2294" \
    --experiment_map "C3PV_1:20250728_C,C3PV_2:20250728_C,C5BO_1:20251021_C,C5BO_2:20251021_C,Fb2294_1:20251021_C,Fb2294_2:20251021_C"
```

`convert_lif_to_ome_tiff.py` writes the same raw z-stack of each `.lif` as a lossless OME-TIFF (CZYX, 8-bit, zlib; physical pixel sizes, channel names and acquisition description in the OME-XML). `--verify` checks existing OME-TIFFs against the `.lif` files without writing anything.

```bash
python scripts/convert_lif_to_ome_tiff.py --input_dir 02_raw_images/confocal --output_dir 02_raw_images/confocal/ome_tiff
```

### Quality control (`03_quality_control.py`)

Computes, for every deposited TIFF, the power log-log slope (focus/blur indicator) and the percentage of pixels at the camera full scale (4095, 12-bit data), and flags within-group outliers; for every spheroid, the Dice agreement between the pipeline mask and an alternative segmentation rule, the sensitivity of the mask area to a ±10 % change of the Otsu threshold, the consistency of each GFP/PKH26 pair (acquisition time gap and exposure) and the offset between the centroids of the two channel signals (flagged above the 95th percentile). Writes the QC columns into `widefield_image_inventory.csv`, the table `segmentation_robustness_per_spheroid.csv` and one contact sheet of all masks per experiment (`qc_overlays_widefield/`). These are the numbers reported in the Technical Validation of the Data Descriptor.

```bash
python scripts/03_quality_control.py --input_root 02_raw_images/widefield --inventory 05_metadata/widefield_image_inventory.csv \
       --metrics 03_processed_data/spheroid_metrics_SKMEL28_control_strategyC.csv --out_dir 03_processed_data
```

## Requirements

```
numpy>=1.21
pandas>=1.3
matplotlib>=3.5
opencv-python>=4.5
scikit-image>=0.19
tifffile>=2021.0
scipy>=1.7
readlif>=0.6   # only for the confocal scripts
```

Install with:
```bash
pip install -r requirements.txt
```

## Image naming convention

The pipeline expects paired `.tif` images with the following naming pattern:

```
{timepoint}_{sample}_GFP_{replicate}.tif
{timepoint}_{sample}_PKH26_{replicate}.tif
```

Example: `D7_C5BO_GFP_002.tif`, `D7_C5BO_PKH26_002.tif`. Token matching is case-insensitive (`d7_c5bo_gfp_002.tif` pairs with `D7_C5BO_PKH26_002.tif`). The timepoint token is the actual day post-seeding.

Recognized sample identifiers: C5BO, C3PV, Fb2294 (the filename tokens `Fb2294`, `FB2294`, `2294` and the mistyped `FB2291` (experiment 3, day 1) all map to `Fb2294`).

## CSV output format

All CSV files are standard comma-separated files with period (`.`) decimal marks and open with the default settings of pandas, R and Excel (English locale).

## License

MIT License. See [LICENSE](LICENSE) for details.

Copyright © 2026 Institute of Molecular Genetics "Luigi Luca Cavalli-Sforza" - Consiglio Nazionale delle Ricerche, Italy.

Third-party dependencies keep their own licences. In particular, `readlif` (used only by the confocal scripts to read .lif files) is distributed under the GNU GPL v3; it is installed separately through `requirements.txt` and is not included in this repository.

## Citation

If you use this pipeline or the dataset, please cite the Data Descriptor (see also `CITATION.cff`):

```
Veniali G., Galli Balestreri E., Orioli D. & Lanzafame M. (2026).
Fluorescence imaging dataset of melanoma fibroblast co-spheroids with
morphometric and radial profiling data. Scientific Data (submitted).
```

## Contact

Donata Orioli and Manuela Lanzafame, CNR-IGM Institute of Molecular Genetics "Luigi Luca Cavalli-Sforza", Pavia, Italy — <https://www.igm.cnr.it/istituto-2/organizzazione/personale/lanzafame-manuela/>
