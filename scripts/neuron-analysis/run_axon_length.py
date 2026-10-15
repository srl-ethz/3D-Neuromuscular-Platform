"""Median traced axon length per sample.

Input: NeuronJ tracing CSVs in <length-dir>/Isotropic/ and <length-dir>/Anisotropic/,
calibrated by the magnification in the traced image name.
Output: <out-dir>/axon_length.csv

Usage:
    python scripts/neuron-analysis/run_axon_length.py --length-dir <input> --out-dir <output>
"""

import argparse
import csv
import os
import re

import numpy as np
import pandas as pd

CONDITION_PREFIX = {"Isotropic": "I", "Anisotropic": "G"}

PX_PER_MM_AT_1X = 110.0

MAG_RE = re.compile(r"(\d+)(?:[p.](\d+))?[xX]")


def parse_magnification(text):
    """Magnification factor parsed from a token such as 2X, 1p6X or 2.5X."""
    match = MAG_RE.search(text)
    if match is None:
        raise ValueError(f"no magnification token in: {text!r}")
    whole, frac = match.groups()
    return float(f"{whole}.{frac}") if frac else float(whole)


def find_column(df, keyword):
    """Column whose name contains ``keyword``, case-insensitively."""
    for col in df.columns:
        if keyword in str(col).strip().lower():
            return col
    raise ValueError(f"no column containing {keyword!r} in: {list(df.columns)}")


def read_lengths(path):
    """Traced lengths in mm, calibrated by the magnification of their image."""
    df = pd.read_csv(path, sep=None, engine="python")

    lengths_px = pd.to_numeric(df[find_column(df, "length")],
                               errors="coerce").dropna().to_numpy()

    names = df[find_column(df, "image")].dropna().unique()
    if len(names) == 0:
        raise ValueError(f"no image name in {os.path.basename(path)}")

    return lengths_px / (PX_PER_MM_AT_1X * parse_magnification(str(names[0])))


def find_condition_folder(root, name):
    """Subfolder matching ``name`` case-insensitively."""
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isdir(path) and entry.lower() == name.lower():
            return path
    raise FileNotFoundError(f"no '{name}' subfolder in {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--length-dir", required=True)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for condition in ["Isotropic", "Anisotropic"]:
        folder = find_condition_folder(args.length_dir, condition)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
        prefix = CONDITION_PREFIX[condition]

        for i, name in enumerate(files, start=1):
            lengths = read_lengths(os.path.join(folder, name))
            if len(lengths) == 0:
                print(f"WARNING: {name} holds no valid lengths -- skipped")
                continue
            sample = f"{prefix}{i:02d}"
            rows.append(dict(sample=sample, median_length_mm=f"{np.median(lengths):.6g}"))
            print(f"  {sample}  {np.median(lengths):6.3f} mm "
                  f"(from {len(lengths)} traces)")

    if not rows:
        raise SystemExit(f"no usable CSVs under {args.length_dir}")

    path = os.path.join(args.out_dir, "axon_length.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample", "median_length_mm"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
