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


_GEMM_THRESHOLD = 256  # use the BLAS path once both sets exceed this size


def _match_gemm(
    d1: np.ndarray, d2: np.ndarray, ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact L2 nearest-neighbour matching via a single GEMM.

    RIFT2 descriptors are L2-normalized, so

        ||a - b||^2 = 2 - 2 * <a, b>

    and the nearest neighbour in L2 distance is the largest inner
    product. We therefore compute the full N x M cosine-similarity
    matrix with a single BLAS call and run argmax per row, which on a
    multi-core CPU is dramatically faster than a brute-force
    ``BFMatcher`` and still gives exact (non-approximate) matches.
    """
    sim = d1 @ d2.T  # (N, M) float32
    if ratio < 1.0 and sim.shape[1] >= 2:
        # Top-2 per row via argpartition (avoids a full argsort).
        idx_part = np.argpartition(-sim, kth=1, axis=1)[:, :2]
        rows = np.arange(sim.shape[0])[:, None]
        top2_sim = sim[rows, idx_part]
        order = np.argsort(-top2_sim, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)
        sim_sorted = np.take_along_axis(top2_sim, order, axis=1)
        best_sim = sim_sorted[:, 0]
        second_sim = sim_sorted[:, 1]
        # Convert similarities back to distances for the ratio test.
        best_d = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * best_sim))
        second_d = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * second_sim))
        keep = best_d <= ratio * second_d
        pairs = np.stack(
            [np.nonzero(keep)[0], idx_sorted[keep, 0]], axis=1
        ).astype(np.int64)
        dists = best_d[keep].astype(np.float32)
    else:
        idx = np.argmax(sim, axis=1)
        best_sim = sim[np.arange(sim.shape[0]), idx]
        dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * best_sim)).astype(np.float32)
        pairs = np.stack([np.arange(sim.shape[0], dtype=np.int64), idx.astype(np.int64)], axis=1)
    return pairs, dists


def match_descriptors(
    des1: np.ndarray,
    des2: np.ndarray,
    ratio: float = 1.0,
    cross_check: bool = False,
    backend: str = "auto",
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
        If ``True``, only keep mutual nearest neighbors. Forces the
        OpenCV ``BFMatcher`` backend.
    backend
        ``"auto"`` (default) picks the BLAS GEMM path when both descriptor
        sets are large enough to amortise the allocation, and the OpenCV
        ``BFMatcher`` otherwise. Pass ``"gemm"`` or ``"bf"`` to force one.

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

    use_gemm = backend == "gemm" or (
        backend == "auto"
        and d1.shape[0] >= _GEMM_THRESHOLD
        and d2.shape[0] >= _GEMM_THRESHOLD
    )
    if use_gemm:
        return _match_gemm(d1, d2, ratio)

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
