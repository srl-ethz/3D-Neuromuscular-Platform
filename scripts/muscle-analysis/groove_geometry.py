"""Groove geometry from the brightfield (Mono) channel of the ND2 stacks."""

import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from skimage.filters import sobel
from skimage.transform import hough_line


DESIGN_WALL_SPACING_UM = 300.0
DESIGN_PERIOD_UM = 600.0


def _wall_edge_map(mono, px_um):
    """Gradient magnitude of the scaffold, with cell texture suppressed."""
    img = mono.astype(float)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)
    smooth = gaussian_filter(img, sigma=max(1.5, 15.0 / px_um))
    gradient = sobel(smooth)
    return gradient > np.percentile(gradient, 90)


def groove_angle(mono, px_um):
    """Rotation angle in degrees that brings the grooves vertical."""
    edges = _wall_edge_map(mono, px_um)
    if edges.sum() < 200:
        return None, 0.0

    thetas = np.deg2rad(np.arange(-90.0, 90.0, 0.25))
    accumulator, theta, _ = hough_line(edges, theta=thetas)

    top_k = max(3, int(round(edges.shape[1] * px_um / DESIGN_WALL_SPACING_UM)))
    column = np.sort(accumulator, axis=0)[-top_k:, :].sum(axis=0).astype(float)
    column = gaussian_filter(column, sigma=2.0)
    best = int(np.argmax(column))
    return float(np.degrees(theta[best])), float(column[best] / (np.median(column) + 1e-9))


def find_walls(mono, angle_deg, px_um, min_sep_um=200.0, rel_threshold=0.35,
               min_coverage=0.5):
    """Wall x-positions in the rotated frame, plus the rotated edge profile."""
    edges = _wall_edge_map(mono, px_um).astype(float)
    rotated = rotate(edges, angle=angle_deg, reshape=True, order=1,
                     mode="constant", cval=np.nan)

    usable = np.isfinite(rotated).mean(axis=0) >= min_coverage
    if usable.sum() < 32:
        return np.array([]), np.array([]), rotated

    with np.errstate(invalid="ignore"):
        profile = np.nanmean(rotated, axis=0)
    profile = np.where(usable, profile, np.nan)

    filled = profile.copy()
    ok = np.isfinite(filled)
    filled[~ok] = 0.0
    filled = gaussian_filter(filled, sigma=max(1.0, 10.0 / px_um))
    filled[~ok] = np.nan

    finite = filled[np.isfinite(filled)]
    if finite.size == 0 or finite.max() <= 0:
        return np.array([]), profile, rotated

    threshold = rel_threshold * finite.max()
    left, right = np.roll(filled, 1), np.roll(filled, -1)
    candidates = np.where(np.isfinite(filled) & (filled > threshold)
                          & (filled >= np.nan_to_num(left, nan=-np.inf))
                          & (filled >= np.nan_to_num(right, nan=-np.inf)))[0]
    if candidates.size == 0:
        return np.array([]), profile, rotated

    min_sep = int(round(min_sep_um / px_um))
    kept = []
    for c in candidates[np.argsort(-filled[candidates])]:
        if all(abs(c - k) >= min_sep for k in kept):
            kept.append(int(c))
    return np.sort(np.array(kept)), profile, rotated


def distance_to_wall_um(shape_rot, walls_px, px_um):
    """Per-pixel distance to the nearest wall in um; NaN when no walls found."""
    height, width = shape_rot
    if walls_px.size == 0:
        return np.full((height, width), np.nan)
    x = np.arange(width)
    nearest = np.min(np.abs(x[None, :] - np.asarray(walls_px)[:, None]), axis=0)
    return np.repeat((nearest * px_um)[None, :], height, axis=0)


def rotate_to_vertical(img, angle_deg):
    """Rotate so the grooves run vertically, padding with NaN."""
    return rotate(img.astype(float), angle=angle_deg, reshape=True, order=1,
                  mode="constant", cval=np.nan)
