"""Dominant orientation estimation for RIFT2 keypoints.

The orientation histogram is built on the *phase-congruency* map gradient,
not on the raw image gradient, so that it inherits the same radiation
invariance as the rest of the pipeline. This mirrors the MATLAB reference
implementation.
"""

from __future__ import annotations

import numpy as np

from . import _orientation_kernel as _kernel

_ORI_PEAK_RATIO = 0.8
_ORI_BINS = 24


def _smooth_circular(hist: np.ndarray) -> np.ndarray:
    """Apply the 1-4-6-4-1 binomial smoother used by the MATLAB reference."""
    h = hist
    h_m2 = np.roll(h, 2)
    h_m1 = np.roll(h, 1)
    h_p1 = np.roll(h, -1)
    h_p2 = np.roll(h, -2)
    return (h_m2 + h_p2) / 16.0 + 4.0 * (h_m1 + h_p1) / 16.0 + h * 6.0 / 16.0


def compute_dominant_orientations(
    pc_map: np.ndarray,
    keypoints: np.ndarray,
    patch_size: int = 96,
) -> np.ndarray:
    """Compute one or more dominant orientations for each keypoint.

    Parameters
    ----------
    pc_map
        Normalized phase-congruency max-moment map (float, ``[0, 1]``).
    keypoints
        Input keypoints of shape ``(N, 2)`` in ``(x, y)`` order.
    patch_size
        Size of the square window (in pixels) used to build the orientation
        histogram.

    Returns
    -------
    oriented : ndarray, shape (K, 3)
        Columns are ``(x, y, angle_deg)``. ``K >= N`` because one keypoint
        may produce multiple dominant orientations, as in SIFT.
    """
    if keypoints.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Sobel-like gradients, matching the MATLAB 3x3 kernel used in the
    # reference (``[-1 0 1; -2 0 2; -1 0 1]``).
    gy, gx = np.gradient(pc_map.astype(np.float32))
    grad_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    grad_ang = np.rad2deg(np.arctan2(gy, gx)).astype(np.float32)
    np.add(grad_ang, 360.0, out=grad_ang, where=grad_ang < 0)

    H, W = pc_map.shape
    r = int(round(patch_size)) // 2
    sigma = r / 3.0

    # Precompute a circular mask and Gaussian weighting for the window.
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    circle_mask = (xx * xx + yy * yy) <= r * r
    gauss = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    base_weight = (circle_mask * gauss).astype(np.float32)

    if _kernel.HAS_NUMBA:
        return _assign_numba(grad_mag, grad_ang, base_weight, keypoints, r)

    out = []
    n = _ORI_BINS
    bin_scale = n / 360.0

    for kp in keypoints:
        x = int(round(float(kp[0])))
        y = int(round(float(kp[1])))

        x1, x2 = x - r, x + r
        y1, y2 = y - r, y + r
        if x1 < 0 or y1 < 0 or x2 >= W or y2 >= H:
            continue

        sub_mag = grad_mag[y1 : y2 + 1, x1 : x2 + 1]
        sub_ang = grad_ang[y1 : y2 + 1, x1 : x2 + 1]
        weight = base_weight * sub_mag

        bin_idx = np.floor(sub_ang * bin_scale).astype(np.int64)
        bin_idx %= n
        hist = np.bincount(bin_idx.ravel(), weights=weight.ravel(), minlength=n)
        hist = _smooth_circular(hist)

        peak = float(hist.max())
        if peak <= 0.0:
            continue
        threshold = peak * _ORI_PEAK_RATIO

        # Circular peak detection with parabolic interpolation.
        prev_h = np.roll(hist, 1)
        next_h = np.roll(hist, -1)
        peaks = np.where((hist > prev_h) & (hist > next_h) & (hist > threshold))[0]
        for kbin in peaks:
            h0 = hist[kbin]
            h_prev = prev_h[kbin]
            h_next = next_h[kbin]
            denom = h_prev + h_next - 2.0 * h0
            offset = 0.5 * (h_prev - h_next) / denom if denom != 0 else 0.0
            bin_f = kbin + offset
            if bin_f < 0:
                bin_f += n
            elif bin_f >= n:
                bin_f -= n
            angle = (360.0 / n) * bin_f
            out.append((float(x), float(y), float(angle)))

    if not out:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _assign_numba(
    grad_mag: np.ndarray,
    grad_ang: np.ndarray,
    base_weight: np.ndarray,
    keypoints: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Numba-accelerated orientation assignment path."""
    kpts = np.ascontiguousarray(keypoints[:, :2], dtype=np.float32)
    n = kpts.shape[0]
    max_per_kp = _kernel.MAX_PEAKS_PER_KP
    out_xy = np.zeros((n * max_per_kp, 2), dtype=np.float32)
    out_angle = np.zeros(n * max_per_kp, dtype=np.float32)
    out_index = np.full(n * max_per_kp, -1, dtype=np.int32)
    counter = np.zeros(1, dtype=np.int64)

    _kernel.assign_orientations_numba(
        grad_mag,
        grad_ang,
        base_weight,
        kpts,
        int(radius),
        out_xy,
        out_angle,
        out_index,
        counter,
    )

    mask = out_index >= 0
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32)
    out = np.empty((int(mask.sum()), 3), dtype=np.float32)
    out[:, 0] = out_xy[mask, 0]
    out[:, 1] = out_xy[mask, 1]
    out[:, 2] = out_angle[mask]
    return out
