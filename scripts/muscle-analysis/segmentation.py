"""Image analysis shared by the fusion and lateral-alignment scripts."""

import os
import re

import numpy as np
import nd2
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from skimage.feature import structure_tensor
from skimage.filters import frangi, threshold_local, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (binary_closing, disk as morph_disk,
                                remove_small_holes, remove_small_objects)

FIBER_CHANNEL_INDEX = 1
NUCLEI_CHANNEL_INDEX = -2

PIXEL_SIZE_UM = 1.33
REFERENCE_PIXEL_SIZE_UM = 1.33

FIBER_RIDGE_FRACTION = 0.05

FOV_PATTERN = re.compile(r"(?:^|_)[GF](\d+)_(\d+)\.nd2$", re.IGNORECASE)

ALIGNMENT_HIST_BINS = np.linspace(-90, 90, 37)


def parse_filename(filename):
    """``(sample_number, fov)`` parsed from a filename, or None."""
    match = FOV_PATTERN.search(filename)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def collect_files(folder):
    """Map ``{replicate: {fov: path}}`` for every parseable ND2 in ``folder``."""
    files = {}
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".nd2"):
            continue
        parsed = parse_filename(name)
        if parsed is None:
            print(f"  [SKIP] could not parse: {name}")
            continue
        rep, fov = parsed
        files.setdefault(rep, {})[fov] = os.path.join(folder, name)
    return files


def pixel_size_um(nd2_file):
    """Lateral pixel size in um (PIXEL_SIZE_UM; the ND2 metadata is not used)."""
    return PIXEL_SIZE_UM


def _channel_axis_check(sizes, path):
    """Fail loudly if the array is not the expected ``(Z, C, Y, X)`` stack."""
    if sizes.get("T", 1) > 1:
        raise ValueError(f"{os.path.basename(path)}: time dimension present, "
                         "axis order assumption (Z, C, Y, X) does not hold")
    if sizes.get("Z", 1) < 2:
        raise ValueError(f"{os.path.basename(path)}: single Z plane, this "
                         "pipeline expects a stack to project")


def load_fiber_and_nuclei_mip(path):
    """Maximum-intensity projection of the fiber and nuclei channels over Z."""
    with nd2.ND2File(path) as handle:
        px_um = pixel_size_um(handle)
        _channel_axis_check(handle.sizes, path)
        n_channels = handle.sizes["C"]
        arr = handle.to_dask()

        fiber_idx = FIBER_CHANNEL_INDEX % n_channels
        nuclei_idx = NUCLEI_CHANNEL_INDEX % n_channels

        fiber = arr[:, fiber_idx].max(axis=0).compute().astype(np.float32)
        nuclei = arr[:, nuclei_idx].max(axis=0).compute().astype(np.float32)

    return fiber, nuclei, px_um


REFERENCE_PARAMS = dict(
    fiber_gaussian_sigma=1.1,
    fiber_local_block=51,
    fiber_local_offset=0.02,
    fiber_otsu_factor=0.6,
    fiber_close_radius=2,
    fiber_min_object_area=150,
    fiber_hole_area=150,
    fiber_frangi_sigmas=(1, 3, 5, 7, 9),
    orientation_tensor_sigma=2.0,
    nuclei_gaussian_sigma=0.8,
    nuclei_local_block=31,
    nuclei_min_object_area=12,
    nuclei_hole_area=12,
    nuclei_min_area=15,
    nuclei_max_area=1200,
)


def _odd(value, minimum=3):
    """Nearest odd integer, as required for local-threshold block sizes."""
    n = int(round(value))
    if n % 2 == 0:
        n += 1
    return max(n, minimum)


def scaled_params(px_um):
    """REFERENCE_PARAMS converted from reference pixels to this file's pixels."""
    s = REFERENCE_PIXEL_SIZE_UM / px_um
    r = REFERENCE_PARAMS
    return dict(
        fiber_gaussian_sigma=r["fiber_gaussian_sigma"] * s,
        fiber_local_block=_odd(r["fiber_local_block"] * s),
        fiber_local_offset=r["fiber_local_offset"],
        fiber_otsu_factor=r["fiber_otsu_factor"],
        fiber_close_radius=max(1, round(r["fiber_close_radius"] * s)),
        fiber_min_object_area=int(round(r["fiber_min_object_area"] * s ** 2)),
        fiber_hole_area=int(round(r["fiber_hole_area"] * s ** 2)),
        fiber_frangi_sigmas=tuple(v * s for v in r["fiber_frangi_sigmas"]),
        orientation_tensor_sigma=r["orientation_tensor_sigma"] * s,
        nuclei_gaussian_sigma=r["nuclei_gaussian_sigma"] * s,
        nuclei_local_block=_odd(r["nuclei_local_block"] * s),
        nuclei_min_object_area=int(round(r["nuclei_min_object_area"] * s ** 2)),
        nuclei_hole_area=int(round(r["nuclei_hole_area"] * s ** 2)),
        nuclei_min_area=r["nuclei_min_area"] * s ** 2,
        nuclei_max_area=r["nuclei_max_area"] * s ** 2,
    )


def segment_fibers(img, px_um, valid=None):
    """Binary mask of myosin-positive fiber structure."""
    p = scaled_params(px_um)

    img = img.astype(float)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    smoothed = gaussian_filter(img_norm, sigma=p["fiber_gaussian_sigma"])

    mask_local = smoothed > threshold_local(
        smoothed, block_size=p["fiber_local_block"], offset=p["fiber_local_offset"])
    otsu_source = smoothed if valid is None else smoothed[valid]
    mask_global = smoothed > threshold_otsu(otsu_source) * p["fiber_otsu_factor"]

    intensity_mask = mask_local & mask_global
    intensity_mask = binary_closing(intensity_mask, morph_disk(p["fiber_close_radius"]))
    intensity_mask = remove_small_objects(intensity_mask, min_size=p["fiber_min_object_area"])
    intensity_mask = remove_small_holes(intensity_mask, area_threshold=p["fiber_hole_area"])

    vesselness = frangi(smoothed, sigmas=p["fiber_frangi_sigmas"], black_ridges=False)
    ridge_mask = vesselness > FIBER_RIDGE_FRACTION * (vesselness.max() + 1e-12)

    labelled = label(intensity_mask)
    mask = np.zeros_like(intensity_mask)
    for region in regionprops(labelled):
        coords = region.coords
        if np.any(ridge_mask[coords[:, 0], coords[:, 1]]):
            mask[coords[:, 0], coords[:, 1]] = True

    mask = remove_small_objects(mask, min_size=p["fiber_min_object_area"])
    mask = remove_small_holes(mask, area_threshold=p["fiber_hole_area"])
    return mask


def segment_nuclei(img, px_um):
    """Labelled nuclei, filtered to a plausible nuclear area range."""
    p = scaled_params(px_um)

    smoothed = gaussian_filter(img.astype(float), sigma=p["nuclei_gaussian_sigma"])
    smoothed = rescale_intensity(smoothed, in_range="image", out_range=(0, 1))

    mask = smoothed > threshold_local(smoothed, block_size=p["nuclei_local_block"])
    mask = remove_small_objects(mask, min_size=p["nuclei_min_object_area"])
    mask = remove_small_holes(mask, area_threshold=p["nuclei_hole_area"])

    labels = label(mask)
    filtered = np.zeros_like(labels)
    next_id = 1
    for region in regionprops(labels):
        if p["nuclei_min_area"] <= region.area <= p["nuclei_max_area"]:
            coords = region.coords
            filtered[coords[:, 0], coords[:, 1]] = next_id
            next_id += 1
    return filtered


def fusion_index(fiber_mask, nuclei_labels):
    """Percentage of nuclei whose footprint overlaps the fiber mask."""
    nuclei = regionprops(nuclei_labels)
    if len(nuclei) == 0:
        return np.nan

    on_fiber = sum(
        1 for n in nuclei
        if np.any(fiber_mask[n.coords[:, 0], n.coords[:, 1]])
    )
    return on_fiber / len(nuclei) * 100.0


def orientation_field(img, px_um):
    """Local orientation in degrees (0-180) from the structure tensor."""
    p = scaled_params(px_um)
    arr = img.astype(float)
    finite = np.isfinite(arr)
    arr = np.where(finite, arr, np.nanmedian(arr))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    smoothed = gaussian_filter(arr, sigma=p["fiber_gaussian_sigma"])
    Axx, Axy, Ayy = structure_tensor(smoothed, sigma=p["orientation_tensor_sigma"],
                                     order="rc")
    return np.degrees(0.5 * np.arctan2(2 * Axy, Axx - Ayy)) % 180.0, finite


def resultant_S(angles_deg):
    """Self-referential nematic order parameter |<exp(2i theta)>|."""
    if angles_deg.size == 0:
        return np.nan
    doubled = np.radians(2.0 * angles_deg)
    return float(np.hypot(np.mean(np.cos(doubled)), np.mean(np.sin(doubled))))


def nematic_S(angles_deg, reference_deg):
    """Nematic order parameter about a given axis: <cos 2(theta - ref)>."""
    if angles_deg.size == 0:
        return np.nan
    return float(np.mean(np.cos(np.radians(2.0 * (angles_deg - reference_deg)))))


def analyze_fiber_structure(fiber_img, fiber_mask, px_um):
    """Per-FOV fiber area density and self-referential alignment."""
    result = dict(
        density_pct=float(fiber_mask.mean() * 100),
        alignment_index=np.nan,
        mean_orientation_deg=np.nan,
        deviation_angles_deg=np.array([]),
    )
    if not np.any(fiber_mask):
        return result

    orientations, _ = orientation_field(fiber_img, px_um)
    angles = orientations[fiber_mask]
    if angles.size:
        result["alignment_index"] = resultant_S(angles)
        doubled = np.radians(2 * angles)
        mean_angle = np.degrees(0.5 * np.arctan2(np.mean(np.sin(doubled)),
                                                 np.mean(np.cos(doubled)))) % 180
        result["mean_orientation_deg"] = float(mean_angle)
        result["deviation_angles_deg"] = (angles - mean_angle + 90) % 180 - 90

    return result


def orientation_histogram(deviation_angles_deg):
    """Probability density of the deviation angles over ALIGNMENT_HIST_BINS."""
    if deviation_angles_deg.size == 0:
        return None
    counts, _ = np.histogram(deviation_angles_deg, bins=ALIGNMENT_HIST_BINS,
                             density=True)
    return counts
