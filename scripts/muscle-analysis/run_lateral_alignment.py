"""Alignment to the groove axis as a function of distance from the wall.

Input: .nd2 stacks in <data-dir>/Grooves/ and <data-dir>/Nogrooves/.

Usage:
    python scripts/muscle-analysis/run_lateral_alignment.py --data-dir <input> --out-dir <output>
"""

import argparse
import csv
import os
import time

import numpy as np
import nd2

import segmentation as seg
from groove_geometry import (distance_to_wall_um, find_walls, groove_angle,
                             rotate_to_vertical)

DIST_BIN_EDGES_UM = np.arange(0, 165, 15.0)
MIN_PIXELS_PER_BIN = 300
MIN_FIBER_PIXELS = 1000


def grooved_profile(path, mask_aware_threshold=False):
    """Mean alignment per distance bin for one grooved field of view."""
    with nd2.ND2File(path) as handle:
        px_um = seg.pixel_size_um(handle)
        arr = handle.asarray()

    n_z, n_channels = arr.shape[0], arr.shape[1]
    fiber_idx = seg.FIBER_CHANNEL_INDEX % n_channels

    angle, _ = groove_angle(arr[n_z // 2, 0].astype(float), px_um)
    if angle is None:
        print("    no scaffold edges found, skipped", flush=True)
        return None

    per_plane = []
    for z in range(n_z):
        walls, _, _ = find_walls(arr[z, 0].astype(float), angle, px_um)
        if walls.size < 2:
            continue

        rotated = rotate_to_vertical(arr[z, fiber_idx].astype(float), angle)
        orientations, finite = seg.orientation_field(rotated, px_um)

        filled = np.where(finite, rotated, 0.0)
        mask = seg.segment_fibers(
            filled, px_um, valid=finite if mask_aware_threshold else None) & finite
        if mask.sum() < MIN_FIBER_PIXELS:
            continue

        distance = distance_to_wall_um(rotated.shape, walls, px_um)
        values = []
        for i in range(len(DIST_BIN_EDGES_UM) - 1):
            lo, hi = DIST_BIN_EDGES_UM[i], DIST_BIN_EDGES_UM[i + 1]
            selected = mask & (distance >= lo) & (distance < hi)
            values.append(seg.nematic_S(orientations[selected], 90.0)
                          if selected.sum() >= MIN_PIXELS_PER_BIN else np.nan)
        per_plane.append(values)

    if not per_plane:
        return None
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.array(per_plane, dtype=float), axis=0)


def flat_value(path):
    """Mean self-referential alignment over the planes of one flat field."""
    with nd2.ND2File(path) as handle:
        px_um = seg.pixel_size_um(handle)
        arr = handle.asarray()

    fiber_idx = seg.FIBER_CHANNEL_INDEX % arr.shape[1]
    per_plane = []
    for z in range(arr.shape[0]):
        plane = arr[z, fiber_idx].astype(float)
        orientations, finite = seg.orientation_field(plane, px_um)
        mask = seg.segment_fibers(plane, px_um) & finite
        if mask.sum() < MIN_FIBER_PIXELS:
            continue
        per_plane.append(seg.resultant_S(orientations[mask]))
    return float(np.mean(per_plane)) if per_plane else None


def _read(path):
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
    parser.add_argument("--budget-s", type=float, default=float("inf"))
    parser.add_argument("--mask-aware-threshold", action="store_true",
                        help="compute the global Otsu threshold only over the "
                             "pixels carrying data, excluding the padding "
                             "introduced by rotation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    centers = (DIST_BIN_EDGES_UM[:-1] + DIST_BIN_EDGES_UM[1:]) / 2
    deadline = time.time() + args.budget_s

    profile_path = os.path.join(args.out_dir, "groove_alignment_profile.csv")
    flat_path = os.path.join(args.out_dir, "flat_alignment.csv")
    profile_rows, profile_done = _read(profile_path)
    flat_rows, flat_done = _read(flat_path)

    for folder_name, letter, done in (("Grooves", "G", profile_done),
                                      ("Nogrooves", "F", flat_done)):
        folder = os.path.join(args.data_dir, folder_name)
        print(f"\n{folder}")
        files = seg.collect_files(folder)

        for number in sorted(files):
            sample = f"{letter}{number:02d}"
            for replicate, fov in enumerate(sorted(files[number]), start=1):
                if (sample, replicate) in done:
                    print(f"  {sample} replicate {replicate}: already done", flush=True)
                    continue
                if time.time() > deadline:
                    print("  TIME BUDGET REACHED -- rerun to continue", flush=True)
                    break
                print(f"  {sample} replicate {replicate}", flush=True)
                try:
                    if letter == "G":
                        values = grooved_profile(files[number][fov],
                                                 args.mask_aware_threshold)
                        if values is None:
                            continue
                        for center, value in zip(centers, values):
                            profile_rows.append(dict(
                                sample=sample, replicate=replicate,
                                distance_um=f"{center:g}",
                                alignment_index=f"{value:.6g}"))
                    else:
                        value = flat_value(files[number][fov])
                        if value is None:
                            continue
                        flat_rows.append(dict(sample=sample, replicate=replicate,
                                              alignment_index=f"{value:.6g}"))
                except Exception as exc:                      # noqa: BLE001
                    print(f"    {type(exc).__name__}: {exc}", flush=True)

    _write(profile_path, profile_rows,
           ["sample", "replicate", "distance_um", "alignment_index"])
    _write(flat_path, flat_rows, ["sample", "replicate", "alignment_index"])


if __name__ == "__main__":
    main()
