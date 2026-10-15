"""Total axon area per sample.

Input: images and samples.csv (file, condition, region, rotate_deg,
magnification) in <image-dir>.
Output: <out-dir>/axon_area.csv

Usage:
    python scripts/neuron-analysis/run_axon_area.py --image-dir <input> --out-dir <output>
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

PX_PER_MM_AT_1X = 110.0

THRESH = 0.08

BACKGROUND_MAD_K = 1.5

MANIFEST = "samples.csv"


def to_gray(arr):
    """Normalised greyscale in 0-1, averaging channels for RGB input."""
    if arr.ndim == 3:
        return arr.mean(axis=2).astype(np.float32) / 255.0
    return arr.astype(np.float32) / 255.0


def find_split(gray):
    """Column of the darkest vertical seam, searched in the middle 40 %."""
    h, w = gray.shape
    roi = gray[:int(h * 0.6), :]
    lo, hi = int(w * 0.3), int(w * 0.7)
    return int(np.argmin(roi.mean(axis=0)[lo:hi])) + lo


def detect_spheroids(gray, min_area=2000):
    """Bounding boxes of the spheroids."""
    p995, p90 = np.percentile(gray, 99.5), np.percentile(gray, 90)
    high = max(p90 + 0.6 * (p995 - p90), np.percentile(gray, 97))

    binary = morphology.remove_small_objects(
        ndimage.binary_fill_holes(gray > high), min_size=min_area)
    distance = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=30, labels=binary)

    markers = np.zeros(gray.shape, dtype=int)
    for i, (r, c) in enumerate(coords):
        markers[r, c] = i + 1
    labelled = watershed(-distance, markers, mask=binary)

    return [dict(bbox=p.bbox) for p in measure.regionprops(labelled)
            if p.area >= min_area]


def load_samples(folder):
    """One dict per manifest row: condition, oriented array, pixel calibration."""
    with open(os.path.join(folder, MANIFEST)) as fh:
        manifest = list(csv.DictReader(fh))

    samples = []
    for row in manifest:
        gray = to_gray(np.array(Image.open(os.path.join(folder, row["file"]))))

        region = row["region"].strip().lower()
        if region in ("left", "right"):
            split = find_split(gray)
            gray = gray[:, :split] if region == "left" else gray[:, split:]

        turns = int(round(float(row["rotate_deg"]) / 90)) % 4
        if turns:
            gray = np.rot90(gray, k=turns)

        samples.append(dict(condition=row["condition"].strip().upper(), gray=gray,
                            px_per_mm=PX_PER_MM_AT_1X * float(row["magnification"])))
    return samples


def find_top_row(blurred, sph_top_y, thresh=THRESH):
    """Topmost row still carrying signal above the spheroids."""
    row, gap = sph_top_y, 0
    while row > 0:
        if (blurred[row - 1] > thresh).mean() < 0.05:
            gap += 1
        else:
            gap = 0
        if gap >= 2:
            return row
        row -= 1
    return 0


def find_secondary_cut(blurred, top_y, sph_top_y, thresh=THRESH):
    """Row of the first density dip following a density peak, if there is one."""
    fracs = np.array([(blurred[r] > thresh).mean() for r in range(top_y, sph_top_y)])
    n = len(fracs)
    if n < 20:
        return top_y

    smooth = np.convolve(fracs, np.ones(5) / 5, mode="same")
    peak_idx = None
    for i in range(5, n // 2):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > 0.25:
            peak_idx = i
            break
    if peak_idx is None:
        return top_y

    end = min(peak_idx + 60, n - 5)
    min_idx = peak_idx + int(np.argmin(smooth[peak_idx:end]))
    if smooth[min_idx] < smooth[peak_idx] * 0.85:
        return top_y + min_idx
    return top_y


def compute_prelim(gray_c):
    """Blurred image, effective threshold, spheroid mask and their top row."""
    gray_c = np.clip(gray_c - np.percentile(gray_c, 1), 0, None)
    blurred = ndimage.gaussian_filter(gray_c.astype(np.float32), sigma=1.5)

    spheroids = detect_spheroids(gray_c)
    sph_top_y = min(s["bbox"][0] for s in spheroids)
    sph_mask = np.zeros(gray_c.shape, dtype=bool)
    for s in spheroids:
        r0, c0, r1, c1 = s["bbox"]
        sph_mask[r0:r1, c0:c1] = True

    background = blurred[sph_top_y:][~sph_mask[sph_top_y:]]
    median = np.median(background)
    mad = np.median(np.abs(background - median)) * 1.4826
    eff_thresh = max(THRESH, median + BACKGROUND_MAD_K * mad)

    valid = blurred > 0
    if (blurred[valid] > eff_thresh).mean() > 0.35:
        eff_thresh = max(eff_thresh, threshold_otsu(blurred[valid & ~sph_mask]))

    return blurred, eff_thresh, sph_mask, sph_top_y


def compute_area_mask(gray_c, seed_dilation=15):
    """Mask of the outgrowth continuous with the spheroids."""
    blurred, eff_thresh, sph_mask, sph_top_y = compute_prelim(gray_c)

    axon_bin = morphology.remove_small_objects(blurred > eff_thresh, min_size=50)
    top_y = find_top_row(blurred, sph_top_y, thresh=eff_thresh)
    top_y = find_secondary_cut(blurred, top_y, sph_top_y, thresh=eff_thresh)
    axon_bin[:top_y, :] = False

    seed = ndimage.binary_dilation(sph_mask, iterations=seed_dilation)
    labelled = measure.label(axon_bin)
    connected = np.zeros_like(axon_bin)
    for label_id in np.unique(labelled[seed & axon_bin]):
        if label_id:
            connected |= (labelled == label_id)
    connected |= sph_mask

    relabelled = measure.label(connected)
    if relabelled.max() > 0:
        sizes = np.bincount(relabelled.ravel())
        sizes[0] = 0
        connected = relabelled == sizes.argmax()

    return connected


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    counters = {}
    rows = []
    for s in load_samples(args.image_dir):
        mask = compute_area_mask(s["gray"])
        counters[s["condition"]] = counters.get(s["condition"], 0) + 1
        sample = f"{s['condition']}{counters[s['condition']]:02d}"
        area = mask.sum() / s["px_per_mm"] ** 2
        rows.append(dict(sample=sample, area_mm2=f"{area:.6g}"))
        print(f"  {sample}  {area:7.3f} mm2")

    path = os.path.join(args.out_dir, "axon_area.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample", "area_mm2"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
