"""Numba-accelerated dominant-orientation assignment.

This module mirrors the pure-NumPy implementation in
:mod:`rift2.orientation` but performs the entire per-keypoint loop inside
a single ``@njit(parallel=True)`` routine. It is imported lazily so that
environments without Numba simply fall back to the NumPy path.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange  # type: ignore

    HAS_NUMBA = True
except Exception:  # pragma: no cover - depends on the user environment.
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return _decorator

    def prange(*args, **kwargs):  # type: ignore
        return range(*args, **kwargs)


_ORI_PEAK_RATIO = 0.8
_ORI_BINS = 24


@njit(cache=True, fastmath=True, parallel=True, boundscheck=False)
def assign_orientations_numba(
    grad_mag: np.ndarray,    # (H, W) float32
    grad_ang: np.ndarray,    # (H, W) float32, degrees in [0, 360)
    base_weight: np.ndarray, # (2r+1, 2r+1) float32 (circular Gaussian)
    keypoints: np.ndarray,   # (N, 2) float32
    radius: int,
    out_xy: np.ndarray,      # (max_out, 2) float32
    out_angle: np.ndarray,   # (max_out,) float32
    out_index: np.ndarray,   # (max_out,) int32 -> index of source keypoint
    counter: np.ndarray,     # (1,) int64 -> atomic-ish write head
) -> None:
    """Per-keypoint orientation histogram + peak detection.

    Output buffers must be pre-allocated by the caller. Because Numba
    ``prange`` does not provide an atomic counter, we emit fixed-stride
    slots: keypoint ``k`` writes its (up to) ``MAX_PEAKS_PER_KP`` peaks at
    ``[k * MAX_PEAKS_PER_KP, (k+1) * MAX_PEAKS_PER_KP)``. The caller
    compacts the output afterwards using the ``out_index >= 0`` mask.
    """
    H = grad_mag.shape[0]
    W = grad_mag.shape[1]
    n = _ORI_BINS
    bin_scale = n / 360.0
    bin_step = 360.0 / n
    N = keypoints.shape[0]
    MAX_PEAKS_PER_KP = 4

    for k in prange(N):
        x = int(keypoints[k, 0] + 0.5)
        y = int(keypoints[k, 1] + 0.5)

        if x - radius < 0 or y - radius < 0 or x + radius >= W or y + radius >= H:
            continue

        # Local histogram in a thread-private buffer.
        hist = np.zeros(n, dtype=np.float32)

        for j in range(-radius, radius + 1):
            row = y + j
            wr = j + radius
            for i in range(-radius, radius + 1):
                col = x + i
                wc = i + radius
                w = base_weight[wr, wc]
                if w == 0.0:
                    continue
                a = grad_ang[row, col]
                b = int(math.floor(a * bin_scale))
                if b < 0:
                    b += n
                elif b >= n:
                    b -= n
                hist[b] += w * grad_mag[row, col]

        # 1-4-6-4-1 circular smoothing.
        smoothed = np.empty(n, dtype=np.float32)
        for b in range(n):
            bm2 = b - 2
            if bm2 < 0:
                bm2 += n
            bm1 = b - 1
            if bm1 < 0:
                bm1 += n
            bp1 = b + 1
            if bp1 >= n:
                bp1 -= n
            bp2 = b + 2
            if bp2 >= n:
                bp2 -= n
            smoothed[b] = (
                (hist[bm2] + hist[bp2]) / 16.0
                + 4.0 * (hist[bm1] + hist[bp1]) / 16.0
                + hist[b] * 6.0 / 16.0
            )

        peak = 0.0
        for b in range(n):
            if smoothed[b] > peak:
                peak = smoothed[b]
        if peak <= 0.0:
            continue
        threshold = peak * _ORI_PEAK_RATIO

        # Circular peak detection with parabolic interpolation.
        slot_base = k * MAX_PEAKS_PER_KP
        slot = 0
        for b in range(n):
            bm = b - 1
            if bm < 0:
                bm += n
            bp = b + 1
            if bp >= n:
                bp -= n
            h0 = smoothed[b]
            h_prev = smoothed[bm]
            h_next = smoothed[bp]
            if h0 > h_prev and h0 > h_next and h0 > threshold:
                denom = h_prev + h_next - 2.0 * h0
                offset = 0.0
                if denom != 0.0:
                    offset = 0.5 * (h_prev - h_next) / denom
                bin_f = b + offset
                if bin_f < 0.0:
                    bin_f += n
                elif bin_f >= n:
                    bin_f -= n
                if slot < MAX_PEAKS_PER_KP:
                    out_xy[slot_base + slot, 0] = float(x)
                    out_xy[slot_base + slot, 1] = float(y)
                    out_angle[slot_base + slot] = bin_step * bin_f
                    out_index[slot_base + slot] = k
                    slot += 1


MAX_PEAKS_PER_KP = 4
