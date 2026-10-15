#!/usr/bin/env python3
"""
Active Ca2+ area of neuromuscular constructs before and after glutamate.

Inputs
------
--data     folder containing the videos (one per recording; the frame rate is
           read from the file)
--samples  CSV, one row per recording: recording (video file, relative to
           --data), condition, construct, timepoint (Pre | Post)

Output: CSV with columns condition, construct, timepoint, active_area_pct.

Usage
-----
    python detect_active_area.py --data DATA_DIR --samples samples.csv --out active_area.csv
"""

import argparse
import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks
from skimage import measure, morphology
from skimage.registration import phase_cross_correlation

BIN = 2                          # spatial binning
BASELINE_PERCENTILE = 20         # F0 = this percentile of each pixel over time
SPATIAL_SIGMA = 1.0              # px
TEMPORAL_SIGMA = 1.0             # frames
DFF_THRESHOLD = 0.10             # peak ΔF/F0 of an active pixel (same for every recording)
EVENT_PROMINENCE = 0.08          # ΔF/F0 a blink must stand out
EVENT_MIN_WIDTH_S = 0.12         # minimum blink width (s)
EVENT_MIN_SEP_S = 0.5            # minimum spacing between blinks (s)
MIN_REGION_PX = 32               # region size gate (binned px)
MAX_REGION_PX = 161              # larger compact objects are dropped ...
FIBRE_ECCENTRICITY = 0.90        # ... unless elongated (fibres)
TISSUE_FRACTION = 0.4            # tissue = mean intensity >= this x field median
CLOSE_RADIUS_PX = 1              # merge regions closer than this
SYNCHRONY_MAD = 5.0              # artefact frame: blink count > median + 5 robust SD
SYNCHRONY_MIN_CANDIDATES = 20


def to_grey_binned(frame):
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3:
        frame = frame[..., :3].mean(axis=2)
    H, W = (frame.shape[0] // BIN) * BIN, (frame.shape[1] // BIN) * BIN
    return frame[:H, :W].reshape(H // BIN, BIN, W // BIN, BIN).mean(axis=(1, 3))


def load_video(path):
    """(grey, binned stack, frame rate) of a video; frames keep their bit depth."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_CONVERT_RGB, 0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(to_grey_binned(f))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames), fps


def register(stack):
    valid = np.ones(stack.shape[1:], dtype=bool)
    ref = np.median(stack[:10], axis=0)
    shifts = np.array([np.round(phase_cross_correlation(ref, f, normalization=None)[0]).astype(int) for f in stack])
    if not shifts.any():
        return stack, valid
    out = np.empty_like(stack)
    for i, (dy, dx) in enumerate(shifts):
        out[i] = np.roll(stack[i], (dy, dx), axis=(0, 1))
        if dy > 0: valid[:dy, :] = False
        elif dy < 0: valid[dy:, :] = False
        if dx > 0: valid[:, :dx] = False
        elif dx < 0: valid[:, dx:] = False
    return out, valid


def delta_f(stack):
    level = np.median(stack, axis=(1, 2))
    stack = stack / (level / np.median(level))[:, None, None]
    stack = gaussian_filter(stack, sigma=(0, SPATIAL_SIGMA, SPATIAL_SIGMA))
    stack = gaussian_filter1d(stack, sigma=TEMPORAL_SIGMA, axis=0)
    f0 = np.maximum(np.percentile(stack, BASELINE_PERCENTILE, axis=0), 1e-3)
    return (stack - f0) / f0


def candidate_regions(dff):
    mask = dff.max(axis=0) > DFF_THRESHOLD
    mask = morphology.remove_small_objects(mask, MIN_REGION_PX)
    mask = morphology.remove_small_holes(mask, MIN_REGION_PX)
    labeled = measure.label(mask)
    kept = np.zeros_like(labeled)
    n = 0
    for p in measure.regionprops(labeled):
        if p.area > MAX_REGION_PX and p.eccentricity < FIBRE_ECCENTRICITY:
            continue
        n += 1
        kept[labeled == p.label] = n
    return kept, n


def blinks(trace, fps):
    return find_peaks(trace, prominence=EVENT_PROMINENCE, width=max(1, int(EVENT_MIN_WIDTH_S * fps)),
                      distance=max(1, int(EVENT_MIN_SEP_S * fps)))[0]


def artefact_frames(blink_lists):
    if len(blink_lists) < SYNCHRONY_MIN_CANDIDATES:
        return set()
    counts = Counter(f for bl in blink_lists for f in bl)
    if not counts:
        return set()
    vals = np.array(list(counts.values()))
    med = np.median(vals)
    robust_sd = max(np.median(np.abs(vals - med)) * 1.4826, 1.0)
    return {f for f, c in counts.items() if c > med + SYNCHRONY_MAD * robust_sd}


def active_area_pct(path):
    stack, fps = load_video(path)
    stack, in_view = register(stack)
    dff = delta_f(stack)
    cand, n = candidate_regions(dff)
    mean_img = stack.mean(axis=0)
    tissue_floor = TISSUE_FRACTION * float(np.median(mean_img))
    traces = [dff[:, cand == k].mean(axis=1) for k in range(1, n + 1)]
    cand_blinks = [blinks(t, fps) for t in traces]
    artefacts = artefact_frames(cand_blinks)
    survivor = np.zeros(cand.shape, dtype=bool)
    for k, bl in enumerate(cand_blinks, start=1):
        region = cand == k
        if mean_img[region].mean() < tissue_floor or not in_view[region].all():
            continue
        if any(f not in artefacts for f in bl):
            survivor |= region
    survivor = morphology.binary_closing(survivor, morphology.disk(CLOSE_RADIUS_PX))
    merged = measure.label(survivor)
    active = np.zeros(cand.shape, dtype=bool)
    for k in range(1, int(merged.max()) + 1):
        m = merged == k
        if any(f not in artefacts for f in blinks(dff[:, m].mean(axis=1), fps)):
            active |= m
    return 100.0 * active.sum() / active.size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default="active_area.csv")
    args = ap.parse_args()
    samples = pd.read_csv(args.samples)
    rows = []
    for s in samples.itertuples(index=False):
        area = active_area_pct(os.path.join(args.data, s.recording))
        rows.append(dict(condition=s.condition, construct=s.construct, timepoint=s.timepoint, active_area_pct=area))
        print(f"{s.condition} construct {s.construct} {s.timepoint}: {area:.3f}% active area", flush=True)
    out = pd.DataFrame(rows)
    out["_tp"] = out.timepoint.map({"Pre": 0, "Post": 1})
    out = out.sort_values(["condition", "construct", "_tp"], ascending=[False, True, True]).drop(columns="_tp")
    out.to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
