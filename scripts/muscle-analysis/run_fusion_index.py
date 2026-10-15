"""Myotube metrics per field of view: fusion index, fiber density, fiber
alignment index and orientation distribution.

Input: .nd2 stacks in <data-dir>/Grooves/ and <data-dir>/Nogrooves/.

Usage:
    python scripts/muscle-analysis/run_fusion_index.py --data-dir <input> --out-dir <output>
"""

import argparse
import csv
import os
import time

import numpy as np

import segmentation as seg

GROUP_FOLDERS = {"Grooves": "G", "Nogrooves": "F"}

BIN_CENTERS = (seg.ALIGNMENT_HIST_BINS[:-1] + seg.ALIGNMENT_HIST_BINS[1:]) / 2
BIN_WIDTH = seg.ALIGNMENT_HIST_BINS[1] - seg.ALIGNMENT_HIST_BINS[0]


def process_fov(path):
    """All per-FOV measurements for a single stack."""
    fiber_img, nuclei_img, px_um = seg.load_fiber_and_nuclei_mip(path)

    fibers = seg.segment_fibers(fiber_img, px_um)
    nuclei = seg.segment_nuclei(nuclei_img, px_um)

    structure = seg.analyze_fiber_structure(fiber_img, fibers, px_um)
    return dict(
        fusion_index_pct=seg.fusion_index(fibers, nuclei),
        fiber_density_pct=structure["density_pct"],
        alignment_index=structure["alignment_index"],
        deviation_angles_deg=structure["deviation_angles_deg"],
    )


def _read(path):
    """Existing rows and the set of fields of view they cover."""
    if not os.path.exists(path):
        return [], set()
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return rows, {(r["sample"], int(r["replicate"])) for r in rows}


def _write(path, rows, fields):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--budget-s", type=float, default=float("inf"),
                        help="stop starting new fields of view after this many "
                             "seconds; rerun to continue where it stopped")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = {name: os.path.join(args.out_dir, f"{name}.csv")
             for name in ("fusion_index", "fiber_density", "alignment_index",
                          "orientation_distribution")}

    rows = {name: _read(path)[0] for name, path in paths.items()}
    done = _read(paths["fusion_index"])[1]
    deadline = time.time() + args.budget_s

    for folder_name, letter in GROUP_FOLDERS.items():
        folder = os.path.join(args.data_dir, folder_name)
        print(f"\n{folder}")
        files = seg.collect_files(folder)

        for number in sorted(files):
            sample = f"{letter}{number:02d}"
            for replicate, fov in enumerate(sorted(files[number]), start=1):
                if (sample, replicate) in done:
                    print(f"  {sample} replicate {replicate}: already done")
                    continue
                if time.time() > deadline:
                    print("  TIME BUDGET REACHED -- rerun to continue")
                    break

                r = process_fov(files[number][fov])
                key = dict(sample=sample, replicate=replicate)

                rows["fusion_index"].append(
                    dict(key, fusion_index_pct=f"{r['fusion_index_pct']:.6g}"))
                rows["fiber_density"].append(
                    dict(key, fiber_density_pct=f"{r['fiber_density_pct']:.6g}"))
                rows["alignment_index"].append(
                    dict(key, alignment_index=f"{r['alignment_index']:.6g}"))

                counts = seg.orientation_histogram(r["deviation_angles_deg"])
                if counts is not None:
                    for center, density in zip(BIN_CENTERS, counts):
                        rows["orientation_distribution"].append(
                            dict(key, deviation_deg=f"{center:g}",
                                 frequency_pct=f"{density * BIN_WIDTH * 100:.6g}"))

                print(f"  {sample} replicate {replicate}  "
                      f"FI={r['fusion_index_pct']:.2f}%  "
                      f"density={r['fiber_density_pct']:.1f}%  "
                      f"alignment={r['alignment_index']:.2f}")

    _write(paths["fusion_index"], rows["fusion_index"],
           ["sample", "replicate", "fusion_index_pct"])
    _write(paths["fiber_density"], rows["fiber_density"],
           ["sample", "replicate", "fiber_density_pct"])
    _write(paths["alignment_index"], rows["alignment_index"],
           ["sample", "replicate", "alignment_index"])
    _write(paths["orientation_distribution"], rows["orientation_distribution"],
           ["sample", "replicate", "deviation_deg", "frequency_pct"])


if __name__ == "__main__":
    main()
