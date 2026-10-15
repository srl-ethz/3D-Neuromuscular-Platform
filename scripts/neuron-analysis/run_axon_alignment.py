"""Axon alignment index.

Input: images in <data-dir>/Nogrooves/ and <data-dir>/Grooves/.
Output: <out-dir>/axon_alignment.csv

Usage:
    python scripts/neuron-analysis/run_axon_alignment.py --data-dir <input> --out-dir <output>
"""

import argparse
import csv
import os
import re

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure, filters, io, morphology
from skimage.feature import structure_tensor

SUBFOLDERS = ["Nogrooves", "Grooves"]

GAUSSIAN_SIGMA = 1.5
RIDGE_SIGMAS = (1, 2, 3, 4)
RIDGE_FRACTION = 0.05
TENSOR_SIGMA = 2.0
MIN_AXON_SIZE = 20
MIN_AXON_PIXELS = 100

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif")

NAME_RE = re.compile(r"^([A-Za-z])(\d+)_(\d+)\.", re.IGNORECASE)


def load_image(path):
    """Greyscale float image; the first channel is taken for RGB input."""
    img = io.imread(path)
    if img.ndim > 2:
        img = img[..., 0]
    return img.astype(float)


def preprocess(img):
    """Stretch to full range, then blur to suppress pixel noise."""
    return filters.gaussian(exposure.rescale_intensity(img), sigma=GAUSSIAN_SIGMA)


def segment_axons(img):
    """Binary mask of axon-like structure: line-like AND bright."""
    vesselness = filters.frangi(img, sigmas=RIDGE_SIGMAS, black_ridges=False)
    ridge = vesselness > RIDGE_FRACTION * vesselness.max()
    bright = img > filters.threshold_otsu(img)
    return morphology.remove_small_objects(ridge & bright, MIN_AXON_SIZE)


def alignment_index(axon_mask, img):
    """Directionality of the axons: 0 for random, 1 for perfectly aligned."""
    if np.sum(axon_mask) < MIN_AXON_PIXELS:
        return 0.0

    Axx, Axy, Ayy = structure_tensor(img, sigma=TENSOR_SIGMA, order="rc")
    orientations = np.degrees(0.5 * np.arctan2(2 * Axy, Axx - Ayy)) % 180.0

    doubled = np.radians(2 * orientations[axon_mask])
    return float(np.hypot(np.mean(np.cos(doubled)), np.mean(np.sin(doubled))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for folder_name in SUBFOLDERS:
        folder = os.path.join(args.data_dir, folder_name)
        for name in sorted(f for f in os.listdir(folder)
                           if f.lower().endswith(IMAGE_EXTENSIONS)):
            match = NAME_RE.match(name)
            if match is None:
                print(f"  {name}: SKIPPED (name does not parse)")
                continue
            letter, sample_no, replicate = match.groups()
            sample = f"{letter.upper()}{int(sample_no):02d}"

            img = preprocess(load_image(os.path.join(folder, name)))
            mask = segment_axons(img)
            if np.sum(mask) == 0:
                print(f"  {sample} replicate {int(replicate)}: SKIPPED (no signal)")
                continue
            index = alignment_index(mask, img)
            rows.append(dict(sample=sample, replicate=int(replicate),
                             alignment_index=f"{index:.6g}"))
            print(f"  {sample}  replicate {int(replicate)}  {index:.4f}")

    path = os.path.join(args.out_dir, "axon_alignment.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh,
                                fieldnames=["sample", "replicate", "alignment_index"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
