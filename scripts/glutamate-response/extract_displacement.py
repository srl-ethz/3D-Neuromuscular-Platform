#!/usr/bin/env python3
"""
Muscle displacement traces of neuromuscular constructs during glutamate
stimulation and after curare block.

Input: one folder of frames per recording, ordered by the frame number in the
file names.

Output: one CSV, one column per condition and construct (e.g. glutamate_1,
curare_1), one row per frame, displacement in µm.

Usage
-----
    python extract_displacement.py --um-per-px UM_PER_PX \
        --construct GLUTAMATE_DIR CURARE_DIR [--construct ...] \
        --out contraction_traces.csv
"""

import argparse
import os
import re

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from skimage import io as skio

CONDITIONS = ("Glutamate", "Curare")
MASK_PERCENTILE = 35
MASK_WINDOW = 25
# pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
FARNEBACK = (0.5, 4, 41, 6, 7, 1.5, 0)


def find_frames(folder):
    def key(name):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]
    names = sorted((n for n in os.listdir(folder)
                    if n.lower().endswith((".tif", ".tiff", ".bmp", ".png")) and not n.startswith(".")),
                   key=key)
    if not names:
        raise FileNotFoundError(f"No image files in {folder}")
    return [os.path.join(folder, n) for n in names]


def to_grey(raw):
    img = raw.astype(np.float32)
    return img[..., :3].mean(axis=-1) if img.ndim == 3 else img


def to_uint8(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)


def principal_axis_projection(vecs):
    centered = vecs - vecs.mean(axis=0)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    if axis[np.argmax(np.abs(axis))] < 0:   # sign convention (sign is arbitrary)
        axis = -axis
    return centered @ axis


def displacement_trace(folder, um_per_px):
    """Detrended displacement (µm), one value per frame."""
    files = find_frames(folder)
    ref = to_grey(skio.imread(files[0]))
    ref8 = to_uint8(ref)
    tissue = uniform_filter(ref, MASK_WINDOW) > np.percentile(ref, MASK_PERCENTILE)

    idx, vecs, prev = [], [], None
    for i, path in enumerate(files):
        raw = skio.imread(path)
        if prev is not None and np.array_equal(raw, prev):
            continue
        prev = raw
        flow = cv2.calcOpticalFlowFarneback(ref8, to_uint8(to_grey(raw)), None, *FARNEBACK)
        vecs.append((flow[..., 0][tissue].mean(), flow[..., 1][tissue].mean()))
        idx.append(i)

    um = np.interp(np.arange(len(files)), idx, principal_axis_projection(np.asarray(vecs))) * um_per_px
    t = np.arange(um.size)
    return um - np.polyval(np.polyfit(t, um, 1), t)


def main(argv=None):
    p = argparse.ArgumentParser(description="Extract muscle displacement traces to CSV.")
    p.add_argument("--construct", action="append", nargs=2, required=True,
                   metavar=("GLUTAMATE_DIR", "CURARE_DIR"), help="Repeat once per construct.")
    p.add_argument("--um-per-px", type=float, required=True, help="Pixel size (µm).")
    p.add_argument("--out", default="contraction_traces.csv")
    args = p.parse_args(argv)

    columns = {}
    for i, cond in enumerate(CONDITIONS):
        traces = [displacement_trace(folders[i], args.um_per_px) for folders in args.construct]
        n = min(x.size for x in traces)
        for k, x in enumerate(traces, 1):
            columns[f"{cond.lower()}_{k}"] = x[:n]
    rows = max(x.size for x in columns.values())
    with open(args.out, "w") as f:
        f.write(",".join(columns) + "\n")
        for r in range(rows):
            f.write(",".join(f"{x[r]:.4f}" if r < x.size else "" for x in columns.values()) + "\n")


if __name__ == "__main__":
    main()
