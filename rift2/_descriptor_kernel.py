"""Optional Numba-accelerated descriptor kernel.

This module is imported lazily by :mod:`rift2.descriptor`. When Numba is
available, the JIT-compiled :func:`describe_batch_numba` replaces the
per-keypoint Python loop and the OpenCV ``warpAffine`` call with a single
fused, parallelized routine that does sampling, MIM circular shift, cell
histogramming and L2 normalization in one pass over each patch.

If Numba is missing (or fails to import for any reason), the public
:data:`HAS_NUMBA` flag is ``False`` and the descriptor module falls back to
its pure NumPy + OpenCV path with no behavioural difference.
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


@njit(cache=True, fastmath=True, parallel=True, boundscheck=False)
def describe_batch_numba(
    mim: np.ndarray,        # (H, W) uint8
    keypoints: np.ndarray,  # (N, 3) float32  -> (x, y, angle_deg)
    patch_size: int,
    norient: int,
    ncells: int,
    descriptors: np.ndarray,  # (N, ncells*ncells*norient) float32, pre-zeroed
    valid: np.ndarray,        # (N,) int8, set to 1 for kept keypoints
) -> None:
    """Fused per-keypoint descriptor extractor.

    Out-of-bounds keypoints are flagged in ``valid`` with a ``0``; the
    caller must compact the output afterwards.
    """
    H = mim.shape[0]
    W = mim.shape[1]
    radius = patch_size // 2
    out_size = patch_size + 1
    cell = out_size // ncells
    half = (out_size - 1) * 0.5
    scale = (2.0 * radius) / (out_size - 1) if out_size > 1 else 1.0
    D = ncells * ncells * norient
    N = keypoints.shape[0]
    deg2rad = math.pi / 180.0

    for k in prange(N):
        cx = keypoints[k, 0]
        cy = keypoints[k, 1]
        angle = keypoints[k, 2]

        if cx - radius < 0 or cy - radius < 0 or cx + radius >= W or cy + radius >= H:
            valid[k] = 0
            continue

        theta = angle * deg2rad
        ct = math.cos(theta)
        st = math.sin(theta)

        # First pass: sample patch (nearest neighbour) and accumulate
        # global per-channel counts so we can find the dominant index.
        counts = np.zeros(norient, dtype=np.int32)
        patch = np.empty((out_size, out_size), dtype=np.uint8)

        for v in range(out_size):
            dv = (v - half) * scale
            for u in range(out_size):
                du = (u - half) * scale
                xs = cx + ct * du - st * dv
                ys = cy + st * du + ct * dv
                ix = int(xs + 0.5) if xs >= 0 else int(xs - 0.5)
                iy = int(ys + 0.5) if ys >= 0 else int(ys - 0.5)
                if ix < 0:
                    ix = 0
                elif ix >= W:
                    ix = W - 1
                if iy < 0:
                    iy = 0
                elif iy >= H:
                    iy = H - 1
                m = mim[iy, ix]
                patch[v, u] = m
                counts[m] += 1

        dominant = 0
        best = counts[0]
        for i in range(1, norient):
            if counts[i] > best:
                best = counts[i]
                dominant = i

        # Second pass: build the (ncells x ncells x norient) histogram
        # with circular shift and accumulate directly into the output row.
        for i in range(D):
            descriptors[k, i] = 0.0

        for v in range(out_size):
            cy_idx = v // cell
            if cy_idx >= ncells:
                cy_idx = ncells - 1
            row_base = cy_idx * ncells * norient
            for u in range(out_size):
                cx_idx = u // cell
                if cx_idx >= ncells:
                    cx_idx = ncells - 1
                idx = patch[v, u] - dominant
                if idx < 0:
                    idx += norient
                descriptors[k, row_base + cx_idx * norient + idx] += 1.0

        # L2 normalization in place.
        n2 = 0.0
        for i in range(D):
            n2 += descriptors[k, i] * descriptors[k, i]
        if n2 > 0.0:
            inv = 1.0 / math.sqrt(n2)
            for i in range(D):
                descriptors[k, i] *= inv

        valid[k] = 1
