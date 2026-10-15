#!/usr/bin/env python3
"""
Bending angle of electrically stimulated constructs, from image sequences to
the CSV used by plot_bending.py.

Input layout
------------
    <input>/<group>/<construct>/   one recording per construct: an image
                                   sequence (.tif/.tiff/.png), frames in
                                   natural sort order

Groups are processed in the order given by --groups (default: alphabetical),
constructs in natural sort order and numbered 1..n within each group.

Output CSV: group, construct, stim_hz (empty if no beating), bending_deg;
one row per frame.

Usage
-----
    python extract_bending.py INPUT --fps FPS --out bending_traces.csv         [--groups "Control" "Injury Grooves" "Injury Flat"]
"""

import argparse
import os
import re

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import (binary_fill_holes, distance_transform_edt, label as ndi_label,
                           sum as ndi_sum, uniform_filter)
from scipy.signal import butter, find_peaks, sosfiltfilt
from skimage import io as skio
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.graph import MCP_Geometric

FLOW_SCALE = 0.5            # optical flow at half resolution
END_FRAC = 0.15             # end region = 15% of the body area nearest each tip
HIGHPASS_HZ = 0.2
FREQ_BAND_HZ = (0.3, 5.0)   # beat frequencies searched
WINDOW_S, STEP_S = 6.0, 1.0
MIN_SEGMENT_S = 8.0
FREQ_TOLERANCE_PCT = 15.0
MIN_WINDOW_RHYTHM = 0.6
MIN_RHYTHM = 0.5
MIN_PEAK_DISTANCE_S = 0.2


# --------------------------------------------------------------------------- #
# Bending angle
# --------------------------------------------------------------------------- #
def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_frames(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff", ".png"))]
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    return [os.path.join(folder, f) for f in sorted(files, key=natural_key)]


def load_frame(path, scale=FLOW_SCALE):
    img = skio.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def norm8(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)


def tissue_mask(img):
    local = uniform_filter(img, 5)
    try:
        cut = threshold_multiotsu(local, classes=3)[0]
    except ValueError:
        cut = threshold_otsu(local)
    mask = local > cut
    labeled, n = ndi_label(mask)
    if n > 1:
        mask = labeled == (1 + int(np.argmax(ndi_sum(mask, labeled, index=np.arange(1, n + 1)))))
    return mask


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if xs.size else np.array([np.nan, np.nan])


def end_regions(mask, prev_centroids=None):
    """(region_a, region_b, centroid_a, centroid_b) or None; regions ordered"""
    if not mask.any():
        return None
    body = binary_fill_holes(mask)
    inner = distance_transform_edt(body)
    radius = 0.25 * inner.max()
    if radius >= 2:  # opening by a disk of this radius
        core = inner > radius
        trimmed = distance_transform_edt(~core) <= radius if core.any() else body
        lab, n = ndi_label(trimmed)
        if n:
            body = lab == (1 + int(np.argmax(ndi_sum(trimmed, lab, range(1, n + 1)))))
    small = cv2.resize(body.astype(np.uint8), (body.shape[1] // 2, body.shape[0] // 2),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    if small.sum() < 50:
        small = body
    cost = np.where(small, 1.0, np.inf)
    d0, _ = MCP_Geometric(cost).find_costs([tuple(np.argwhere(small)[0])])
    d0[~small] = -1
    tip_a = np.unravel_index(int(np.argmax(d0)), d0.shape)
    da, _ = MCP_Geometric(cost).find_costs([tip_a])
    da[~small] = -1
    tip_b = np.unravel_index(int(np.argmax(da)), da.shape)
    db, _ = MCP_Geometric(cost).find_costs([tip_b])
    size = (body.shape[1], body.shape[0])
    n_end = max(10, int(END_FRAC * body.sum()))
    regions = []
    for dist in (da, db):
        dist = cv2.resize(np.where(small, dist, 1e9).astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
        dist = np.where(body, dist, np.inf)
        regions.append(body & (dist <= np.partition(dist[body], n_end - 1)[n_end - 1]))
    cen = [centroid(regions[0]), centroid(regions[1])]
    if prev_centroids is not None:
        keep = np.hypot(*(cen[0] - prev_centroids[0])) + np.hypot(*(cen[1] - prev_centroids[1]))
        swap = np.hypot(*(cen[0] - prev_centroids[1])) + np.hypot(*(cen[1] - prev_centroids[0]))
        if swap < keep:
            regions, cen = regions[::-1], cen[::-1]
    return regions[0], regions[1], cen[0], cen[1]


def region_rotation(flow, region):
    """Rotation (rad) of a region from an affine fit to its optical flow."""
    ys, xs = np.nonzero(region)
    if xs.size < 10:
        return 0.0
    design = np.column_stack([np.ones(xs.size), xs - xs.mean(), ys - ys.mean()])
    cu = np.linalg.lstsq(design, flow[ys, xs, 0], rcond=None)[0]
    cv = np.linalg.lstsq(design, flow[ys, xs, 1], rcond=None)[0]
    return float((cv[1] - cu[2]) / 2.0)


def bending_angle(files):
    """Cumulative bending angle (degrees) over an image sequence."""
    win = max(9, int(41 * FLOW_SCALE) | 1)
    first = load_frame(files[0])
    prev8 = norm8(first)
    ends = end_regions(tissue_mask(first))
    dtheta = np.zeros(len(files))
    for i in range(1, len(files)):
        cur = load_frame(files[i])
        cur8 = norm8(cur)
        flow = cv2.calcOpticalFlowFarneback(prev8, cur8, None, 0.5, 4, win, 6, 7, 1.5, 0)
        dtheta[i] = region_rotation(flow, ends[1]) - region_rotation(flow, ends[0])
        nxt = end_regions(tissue_mask(cur), prev_centroids=(ends[2], ends[3]))
        if nxt is not None:  # otherwise keep the previous regions
            ends, prev8 = nxt, cur8
    return np.degrees(np.cumsum(dtheta))


# --------------------------------------------------------------------------- #
# Beating frequency and segment selection
# --------------------------------------------------------------------------- #
def detrend(y):
    t = np.arange(y.size)
    return y - np.polyval(np.polyfit(t, y, 1), t)


def power_spectrum(y, fps, pad_to_s=60.0):
    window = np.hanning(y.size)
    n_fft = max(y.size, int(round(pad_to_s * fps)))
    power = np.abs(np.fft.rfft((y - y.mean()) * window, n=n_fft)) ** 2 / np.sum(window ** 2)
    return np.fft.rfftfreq(n_fft, d=1.0 / fps), power


def dominant_frequency(y, fps):
    freqs, power = power_spectrum(y, fps)
    band = (freqs >= FREQ_BAND_HZ[0]) & (freqs <= FREQ_BAND_HZ[1])
    return float(freqs[band][np.argmax(power[band])])


def rhythm_score(y, fps, f0):
    """Fraction of in-band power at f0 and its 2nd/3rd harmonics."""
    freqs, power = power_spectrum(y, fps)
    band = (freqs >= FREQ_BAND_HZ[0]) & (freqs <= FREQ_BAND_HZ[1])
    tol = max(0.08, 2 * fps / y.size)
    near = band & np.any([np.abs(freqs - h * f0) <= tol for h in (1, 2, 3)], axis=0)
    return float(power[near].sum() / power[band].sum())


def nearest_freq(f, candidates):
    nearest = min(candidates, key=lambda c: abs(c - f))
    return nearest, (f - nearest) / nearest * 100.0


def frequency_segments(y, fps, candidates):
    """[(start, end, stim_hz or nan)] runs of regular beating at one expected"""
    n_win = int(round(WINDOW_S * fps))
    if y.size <= n_win:
        return [(0, y.size, float("nan"))], None
    starts = list(range(0, y.size - n_win + 1, max(1, int(round(STEP_S * fps)))))
    centers = np.array([(s + n_win / 2) / fps for s in starts])
    irregular = -1.0
    labels = np.full(len(starts), irregular)
    for i, s in enumerate(starts):
        w = y[s:s + n_win]
        f = dominant_frequency(w, fps)
        nearest, pct = nearest_freq(f, candidates)
        if abs(pct) <= FREQ_TOLERANCE_PCT and rhythm_score(w, fps, f) >= MIN_WINDOW_RHYTHM:
            labels[i] = nearest
    smoothed = labels.copy()
    for i in range(labels.size):  # 5-window majority filter
        vals, counts = np.unique(labels[max(0, i - 2):i + 3], return_counts=True)
        if counts.max() > 1:
            smoothed[i] = vals[np.argmax(counts)]

    def to_runs(lab):
        runs = []
        for i, v in enumerate(lab):
            if runs and runs[-1][2] == v:
                runs[-1][1] = i
            else:
                runs.append([i, i, v])
        return runs

    runs = to_runs(smoothed)
    for r in runs:
        if r[2] != irregular and (r[1] - r[0] + 1) * STEP_S < MIN_SEGMENT_S:
            r[2] = irregular
    runs = to_runs([r[2] for r in runs for _ in range(r[0], r[1] + 1)])
    segments = []
    for j, r in enumerate(runs):
        start = 0 if j == 0 else int(round((centers[runs[j - 1][1]] + centers[r[0]]) / 2 * fps))
        end = y.size if j == len(runs) - 1 else int(round((centers[r[1]] + centers[runs[j + 1][0]]) / 2 * fps))
        segments.append((start, end, float("nan") if r[2] == irregular else float(r[2])))
    return segments


def segment_stim(y, fps, label, candidates):
    """Stimulation frequency of a segment if it beats regularly, else None."""
    if label is None:
        f = dominant_frequency(y, fps)
    elif label != label:
        return None
    else:  # beat frequency = spectral peak near the window label
        freqs, power = power_spectrum(y, fps)
        tol = FREQ_TOLERANCE_PCT / 100.0
        near = (freqs >= label * (1 - tol)) & (freqs <= label * (1 + tol))
        f = float(freqs[near][np.argmax(power[near])]) if near.any() else dominant_frequency(y, fps)
    nearest, pct = nearest_freq(f, candidates)
    if rhythm_score(y, fps, f) >= MIN_RHYTHM and abs(pct) <= FREQ_TOLERANCE_PCT:
        return nearest
    return None


def beat_peaks(y, fps):
    x = y - np.median(y)
    prominence = 0.35 * (np.percentile(np.abs(x), 95) + 1e-12)
    return find_peaks(x, prominence=prominence, distance=max(1, int(round(MIN_PEAK_DISTANCE_S * fps))))[0]


def analysed_trace(files, fps, candidates):
    """(trace, stim_hz or None) of one construct: its longest regular segment,"""
    y = sosfiltfilt(butter(2, HIGHPASS_HZ, "highpass", fs=fps, output="sos"), bending_angle(files))
    if np.mean((y - y.mean()) ** 3) < 0:  # fast contraction phase up
        y = -y
    segments = frequency_segments(y, fps, candidates)
    if len(segments) == 1:
        label = segments[0][2]
        stim = segment_stim(y, fps, label if label == label else None, candidates)
        return y, stim
    regular = []
    for s, e, label in segments:
        seg = detrend(y[s:e])
        stim = segment_stim(seg, fps, label, candidates)
        if stim is not None:
            regular.append((seg, stim))
    if not regular:
        return y, None
    return max(regular, key=lambda r: r[0].size)


def aligned_start(y, stim, fps):
    """A quarter period before the first beat peak."""
    if stim is None:
        return 0
    quarter = int(round(fps / stim / 4))  # frames
    peaks = beat_peaks(y, fps)
    peaks = peaks[peaks >= quarter]
    return int(peaks[0] - quarter) if peaks.size else 0


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Image sequences -> bending-angle CSV.")
    p.add_argument("input", help="Folder of <group>/<construct>/ image sequences.")
    p.add_argument("--fps", type=float, required=True, help="Acquisition frame rate (frames/s).")
    p.add_argument("--out", default="bending_traces.csv")
    p.add_argument("--groups", nargs="+", help="Group folders in plotting order (default: alphabetical).")
    p.add_argument("--expected-freqs", type=float, nargs="+", default=[0.5, 0.7, 1.0, 2.0],
                   help="Stimulation frequencies (Hz).")
    args = p.parse_args()

    groups = args.groups or sorted(d for d in os.listdir(args.input)
                                   if os.path.isdir(os.path.join(args.input, d)) and not d.startswith("."))
    traces = []  # (group, construct, stim, trace)
    for g in groups:
        folders = sorted((d for d in os.listdir(os.path.join(args.input, g))
                          if os.path.isdir(os.path.join(args.input, g, d)) and not d.startswith(".")),
                         key=natural_key)
        for k, d in enumerate(folders, start=1):
            y, stim = analysed_trace(list_frames(os.path.join(args.input, g, d)), args.fps, args.expected_freqs)
            print(f"{g} {k}: {'no beating' if stim is None else f'{stim:g} Hz'}, {y.size} frames analysed")
            traces.append((g, k, stim, y))

    fps = args.fps
    starts = [aligned_start(y, stim, fps) for _, _, stim, y in traces]
    for g in groups:
        idx = [i for i, t in enumerate(traces) if t[0] == g and t[2] is not None]
        for i in idx[1:]:
            ref = next(j for j in idx if traces[j][2] == traces[i][2])
            if ref == i:
                continue
            y, y_ref = traces[i][3], traces[ref][3]
            period = int(round(fps / traces[i][2]))
            n = min(y.size - starts[i] - period, y_ref.size - starts[ref])
            a = y_ref[starts[ref]:starts[ref] + n]
            corr = [np.corrcoef(a, y[starts[i] + k:starts[i] + k + n])[0, 1] for k in range(period)]
            starts[i] += int(np.argmax(corr))
    n_win = min(t[3].size - s for t, s in zip(traces, starts))

    rows = []
    for (g, k, stim, y), s in zip(traces, starts):
        w = y[s:s + n_win]
        rows += [{"group": g, "construct": k, "stim_hz": stim, "bending_deg": v} for v in w - w.mean()]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.out} ({len(traces)} constructs x {n_win} frames)")


if __name__ == "__main__":
    main()
