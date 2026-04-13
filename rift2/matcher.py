"""Descriptor matching utilities.

The RIFT2 reference uses MATLAB's ``matchFeatures`` with ``MaxRatio=1`` (i.e.
effectively nearest-neighbour matching). We expose a thin wrapper around
``cv2.BFMatcher`` that supports both crossCheck and a Lowe-style ratio test;
by default we reproduce the reference behavior.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def match_descriptors(
    des1: np.ndarray,
    des2: np.ndarray,
    ratio: float = 1.0,
    cross_check: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match two sets of RIFT2 descriptors.

    Parameters
    ----------
    des1, des2
        Float descriptor matrices of shape ``(N, D)`` / ``(M, D)``.
    ratio
        Lowe's ratio threshold. ``1.0`` disables the test and keeps the
        nearest neighbor unconditionally (the MATLAB reference behavior).
    cross_check
        If ``True``, only keep mutual nearest neighbors.

    Returns
    -------
    pairs : ndarray, shape (K, 2), int64
        Row indices into ``des1`` and ``des2`` of matched descriptors.
    distances : ndarray, shape (K,), float32
        L2 distances of the kept matches.
    """
    if des1.size == 0 or des2.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)

    d1 = np.ascontiguousarray(des1, dtype=np.float32)
    d2 = np.ascontiguousarray(des2, dtype=np.float32)

    if cross_check:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(d1, d2)
        if not matches:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)
        pairs = np.asarray([(m.queryIdx, m.trainIdx) for m in matches], dtype=np.int64)
        dists = np.asarray([m.distance for m in matches], dtype=np.float32)
        return pairs, dists

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    k = 2 if ratio < 1.0 else 1
    knn = matcher.knnMatch(d1, d2, k=k)
    pairs_list = []
    dists_list = []
    for neighbors in knn:
        if not neighbors:
            continue
        best = neighbors[0]
        if ratio < 1.0 and len(neighbors) >= 2:
            if best.distance > ratio * neighbors[1].distance:
                continue
        pairs_list.append((best.queryIdx, best.trainIdx))
        dists_list.append(best.distance)
    if not pairs_list:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)
    return (
        np.asarray(pairs_list, dtype=np.int64),
        np.asarray(dists_list, dtype=np.float32),
    )
