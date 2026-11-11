#!/usr/bin/env python3
"""
convert_lif_to_ome_tiff.py

Convert the confocal Leica .lif files of the dataset to lossless OME-TIFF.

Reference
---------
Veniali et al. (2026), "Fluorescence imaging dataset of melanoma fibroblast
co-spheroids with morphometric and radial profiling data", Scientific Data
(submitted).

What it does
------------
Each .lif file holds ONE raw two-channel z-stack plus several LAS X-generated
snapshots / 3D projections. The raw stack is selected by the same signature
used by process_confocal_images.py (exactly 2 channels, z > 1); the script
stops if zero or more than one series matches, and checks from the embedded
LUT names that channel 0 is Green (GFP, melanoma) and channel 1 is Red
(PKH26, fibroblasts).

The stack is written unchanged (8-bit, no rescaling) as a CZYX OME-TIFF with
zlib compression, physical pixel sizes from the .lif metadata
(PhysicalSizeX/Y/Z, µm), channel names and an acquisition description.

Output: <output_dir>/<stem>_GFP_PKH26_zstack.ome.tif for each <stem>.lif

Usage
-----
    python convert_lif_to_ome_tiff.py \\
        --input_dir 02_raw_images/confocal \\
        --output_dir 02_raw_images/confocal/ome_tiff

    # verify existing OME-TIFFs against the .lif files instead of writing
    python convert_lif_to_ome_tiff.py --input_dir ... --output_dir ... --verify

Requirements: numpy, tifffile, readlif
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
from readlif.reader import LifFile

CHANNEL_NAMES = ["GFP_SK-MEL-28_melanoma", "PKH26_fibroblasts"]

# Acquisition settings shared by all six stacks (Methods, confocal imaging)
ACQUISITION = ("Leica TCS SP8, HC Fluotar VISIR 25x/0.95 W, zoom 0.75, 8-bit; "
               "spheroid fixed at day 7")
CHANNELS_DESC = ("Channel 0: 488 nm ex / 500-550 nm em (GFP); "
                 "channel 1: 551 nm ex / 562-650 nm em (PKH26)")


def load_raw_stack(path: Path) -> Tuple[np.ndarray, str, float, float]:
    """Return (stack CZYX uint8, series name, pixel size xy µm, z step µm)."""
    lif = LifFile(str(path))
    candidates = []
    for i in range(len(lif.image_list)):
        img = lif.get_image(i)
        if img.channels == 2 and img.dims.z > 1:
            candidates.append(img)
    if len(candidates) != 1:
        raise RuntimeError(
            f"{path}: expected exactly 1 raw 2-channel z-stack series, found {len(candidates)}"
        )
    img = candidates[0]

    lut_names = [elem.attrib.get("LUTName", "") for elem in lif.xml_root.iter()
                 if elem.tag.split("}")[-1] == "ChannelDescription"]
    if len(lut_names) < 2 or lut_names[0] != "Green" or lut_names[1] != "Red":
        raise RuntimeError(
            f"{path}: unexpected channel LUT order {lut_names!r} "
            f"(expected ['Green', 'Red', ...] for channel 0=melanoma, 1=fibroblast)"
        )

    stack = np.stack([
        np.stack([np.asarray(img.get_frame(z=z, c=c)) for z in range(img.dims.z)])
        for c in range(2)
    ])
    if stack.dtype != np.uint8:
        raise RuntimeError(f"{path}: expected 8-bit data, got {stack.dtype}")

    scale = img.scale  # (px/µm x, px/µm y, px/µm z, t)
    return stack, img.name, 1.0 / scale[0], 1.0 / scale[2]


def ome_metadata(lif_name: str, series: str, px: float, pz: float) -> dict:
    return {
        "axes": "CZYX",
        "Name": "Image0",
        "Description": (f'Converted from {lif_name} series "{series}" ({ACQUISITION}). '
                        f"{CHANNELS_DESC}. Conversion: readlif + tifffile, lossless (zlib)."),
        "PhysicalSizeX": px, "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": px, "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": pz, "PhysicalSizeZUnit": "µm",
        "Channel": {"Name": CHANNEL_NAMES},
    }


def verify(out_path: Path, stack: np.ndarray, px: float, pz: float) -> list:
    """Compare an existing OME-TIFF with the .lif stack; return a list of problems."""
    problems = []
    with tifffile.TiffFile(out_path) as tif:
        if not tif.is_ome:
            return ["not an OME-TIFF"]
        data = tif.series[0].asarray()
        xml = tif.ome_metadata
    if data.shape != stack.shape or not np.array_equal(data, stack):
        problems.append(f"pixel data differ (file {data.shape}, lif {stack.shape})")
    for key, val in (("PhysicalSizeX", px), ("PhysicalSizeZ", pz)):
        if f'{key}="{val!r}"' not in xml and f'{key}="{val}"' not in xml:
            problems.append(f"{key} != {val}")
    for name in CHANNEL_NAMES:
        if f'Name="{name}"' not in xml:
            problems.append(f"channel {name} missing")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="Folder with the .lif files")
    ap.add_argument("--output_dir", required=True, help="Folder for the OME-TIFF files")
    ap.add_argument("--verify", action="store_true",
                    help="Check existing OME-TIFFs against the .lif files; write nothing")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    lifs = sorted(in_dir.glob("*.lif"))
    if not lifs:
        print(f"No .lif files in {in_dir}", file=sys.stderr)
        return 1
    if not args.verify:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_bad = 0
    for lif_path in lifs:
        stack, series, px, pz = load_raw_stack(lif_path)
        out_path = out_dir / f"{lif_path.stem}_GFP_PKH26_zstack.ome.tif"
        if args.verify:
            problems = verify(out_path, stack, px, pz) if out_path.exists() else ["missing"]
            n_bad += bool(problems)
            print(f"{out_path.name}: {'OK' if not problems else '; '.join(problems)}")
            continue
        tifffile.imwrite(out_path, stack, ome=True, photometric="minisblack",
                         compression="zlib",
                         metadata=ome_metadata(lif_path.name, series, px, pz))
        print(f"{lif_path.name} [{series}] -> {out_path.name}  "
              f"CZYX {stack.shape}, xy {px:.4f} µm, z {pz:.4f} µm")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
