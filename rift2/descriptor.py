"""RIFT2 descriptor construction.

This module implements the "new rotation invariance technique" introduced by
RIFT2 (arXiv:2303.00319):

1. A Max Index Map (MIM) is obtained by summing the magnitude responses of
   the complex log-Gabor filters across all scales and taking the
   per-pixel ``argmax`` across orientations. The MIM therefore stores an
   integer channel index in ``[0, norient)`` at each pixel.
2. A square patch is sampled around each keypoint, optionally rotated by the
   keypoint's dominant orientation (which was estimated on the
   phase-congruency edge map).
3. Within the patch, the globally dominant MIM channel is found, and every
   index in the patch is circularly shifted so that this dominant channel
   becomes index ``0``. This gives a rotation-invariant patch encoding
   *without* having to recompute the log-Gabor convolutions, which is
   exactly the RIFT2 speedup over RIFT1.
4. The patch is divided into a ``no x no`` grid of cells; each cell yields a
   histogram of the ``no`` MIM indices. The concatenated histograms form a
   ``no * no * nbin``-D descriptor, L2-normalized.

All of this is vectorized with NumPy + OpenCV. The patch sampling is done
with a single ``cv2.warpAffine`` call per keypoint using ``INTER_NEAREST``,
which is correct because the MIM is a categorical map.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import _descriptor_kernel as _kernel


def build_max_index_map(eo: np.ndarray) -> np.ndarray:
    """Build the Max Index Map from complex log-Gabor responses.

    Parameters
    ----------
    eo : ndarray, shape (nscale, norient, H, W), complex
        Output of :func:`rift2.phase_congruency.phase_congruency`.

    Returns
    -------
    mim : ndarray, shape (H, W), uint8
        Index in ``[0, norient)`` of the orientation channel with the
        largest summed magnitude response at every pixel.
    """
    # Sum |EO| over scales -> (norient, H, W), then argmax over orientation.
    cs = np.abs(eo).sum(axis=0)
    mim = np.argmax(cs, axis=0).astype(np.uint8)
    return mim


def _sample_rotated_patch(
    mim: np.ndarray,
    cx: float,
    cy: float,
    radius: int,
    angle_deg: float,
    out_size: int,
) -> np.ndarray:
    """Sample an axis-aligned patch from ``mim`` at a rotated pose.

    The returned patch has shape ``(out_size, out_size)`` and is computed
    with nearest-neighbour interpolation, which is the only correct choice
    for a categorical index map.
    """
    # Build a 2x3 affine that maps destination pixel (u, v) in [0, out_size)
    # into source pixel coordinates in ``mim``. The destination origin is the
    # patch center, patches are rotated by ``-angle`` (inverse of keypoint
    # orientation), scaled so a unit destination step equals a full-
    # resolution source step.
    half = (out_size - 1) * 0.5
    scale = (2.0 * radius) / (out_size - 1) if out_size > 1 else 1.0
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    # Affine in the form expected by cv2.warpAffine with WARP_INVERSE_MAP.
    M = np.array(
        [
            [c * scale, -s * scale, cx - (c * scale * half - s * scale * half)],
            [s * scale, c * scale, cy - (s * scale * half + c * scale * half)],
        ],
        dtype=np.float64,
    )
    patch = cv2.warpAffine(
        mim,
        M,
        (out_size, out_size),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return patch


def compute_descriptors(
    mim: np.ndarray,
    oriented_keypoints: np.ndarray,
    patch_size: int = 96,
    norient: int = 6,
    ncells: int = 6,
) -> np.ndarray:
    """Compute RIFT2 descriptors for a set of oriented keypoints.

    Parameters
    ----------
    mim
        Max Index Map produced by :func:`build_max_index_map`.
    oriented_keypoints
        Array of shape ``(N, 3)`` with columns ``(x, y, angle_deg)``.
    patch_size
        Side length of the square patch used for description (pixels).
    norient
        Number of log-Gabor orientations, equal to the number of distinct
        values in ``mim`` and to the number of histogram bins per cell.
    ncells
        Grid resolution; the descriptor has ``ncells * ncells * norient``
        entries.

    Returns
    -------
    descriptors : ndarray, shape (K, ncells * ncells * norient), float32
        L2-normalized descriptors, row-major. ``K <= N`` because keypoints
        whose support window exceeds the image boundary are dropped.
    valid_idx : ndarray, shape (K,), int64
        Row indices into ``oriented_keypoints`` that produced descriptors.
    """
    if oriented_keypoints.size == 0:
        return (
            np.empty((0, ncells * ncells * norient), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    if _kernel.HAS_NUMBA:
        return _compute_descriptors_numba(
            mim, oriented_keypoints, patch_size, norient, ncells
        )

    H, W = mim.shape
    radius = patch_size // 2
    out_size = patch_size + 1
    cell = out_size // ncells

    descriptors = np.zeros(
        (oriented_keypoints.shape[0], ncells * ncells * norient),
        dtype=np.float32,
    )

    valid_idx = np.empty(oriented_keypoints.shape[0], dtype=np.int64)
    keep = 0
    for row, kp in enumerate(oriented_keypoints):
        cx, cy, angle = float(kp[0]), float(kp[1]), float(kp[2])
        if (
            cx - radius < 0
            or cy - radius < 0
            or cx + radius >= W
            or cy + radius >= H
        ):
            continue

        patch = _sample_rotated_patch(mim, cx, cy, radius, angle, out_size)

        # Rotation invariance: circularly shift MIM indices so the globally
        # dominant channel inside the patch becomes zero.
        counts = np.bincount(patch.ravel(), minlength=norient)
        dominant = int(np.argmax(counts[:norient]))
        patch_rot = (patch.astype(np.int16) - dominant) % norient

        # Aggregate 6x6 cell histograms in one vectorized pass.
        histo = np.zeros((ncells, ncells, norient), dtype=np.float32)
        for j in range(ncells):
            y0 = j * cell
            y1 = (j + 1) * cell if j < ncells - 1 else out_size
            for i in range(ncells):
                x0 = i * cell
                x1 = (i + 1) * cell if i < ncells - 1 else out_size
                clip = patch_rot[y0:y1, x0:x1].ravel()
                histo[j, i] = np.bincount(clip, minlength=norient)[:norient]

        vec = histo.ravel()
        n2 = np.linalg.norm(vec)
        if n2 > 0.0:
            vec /= n2
        descriptors[keep] = vec
        valid_idx[keep] = row
        keep += 1

    return descriptors[:keep], valid_idx[:keep]


def _compute_descriptors_numba(
    mim: np.ndarray,
    oriented_keypoints: np.ndarray,
    patch_size: int,
    norient: int,
    ncells: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Numba-accelerated path used when the JIT backend is available."""
    mim_u8 = np.ascontiguousarray(mim, dtype=np.uint8)
    kpts = np.ascontiguousarray(oriented_keypoints, dtype=np.float32)
    n = kpts.shape[0]
    descriptors = np.zeros((n, ncells * ncells * norient), dtype=np.float32)
    valid = np.zeros(n, dtype=np.int8)

    _kernel.describe_batch_numba(
        mim_u8, kpts, int(patch_size), int(norient), int(ncells), descriptors, valid
    )

    keep_mask = valid.astype(bool)
    valid_idx = np.nonzero(keep_mask)[0].astype(np.int64)
    return descriptors[keep_mask], valid_idx
