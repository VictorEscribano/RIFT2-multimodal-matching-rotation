"""Outlier removal for keypoint correspondences.

The original RIFT/RIFT2 code ships a custom RANSAC implementation (``FSC.m``
+ ``LSM.m``) that estimates a similarity, affine, or perspective transform.
For the Python port we defer to OpenCV's well-tested estimators:

* ``similarity`` -> :func:`cv2.estimateAffinePartial2D`
* ``affine``     -> :func:`cv2.estimateAffine2D`
* ``perspective``-> :func:`cv2.findHomography`

These are RANSAC-based, return a 3x3 matrix, and accept a pixel-space inlier
threshold. The ``similarity`` default matches the MATLAB demo.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _to_3x3(m: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = m
    return out


def estimate_similarity_ransac(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    model: str = "similarity",
    max_reprojection_error: float = 3.0,
    max_iters: int = 2000,
    confidence: float = 0.999,
) -> Tuple[np.ndarray, np.ndarray]:
    """Robustly fit a geometric transform between two keypoint sets.

    Parameters
    ----------
    src_points, dst_points
        Correspondence arrays of shape ``(N, 2)``.
    model
        Transform model: ``"similarity"``, ``"affine"`` or ``"perspective"``.
    max_reprojection_error
        Pixel threshold used by OpenCV's RANSAC.
    max_iters, confidence
        RANSAC stopping criteria.

    Returns
    -------
    H : ndarray of shape (3, 3)
        Transform mapping ``src`` into ``dst``.
    inlier_mask : ndarray of shape (N,), bool
        Mask of inlier correspondences.
    """
    if src_points.shape != dst_points.shape or src_points.ndim != 2 or src_points.shape[1] != 2:
        raise ValueError("src_points and dst_points must be (N, 2) arrays")
    if src_points.shape[0] < 2:
        raise ValueError("Need at least 2 correspondences")

    src = src_points.astype(np.float32).reshape(-1, 1, 2)
    dst = dst_points.astype(np.float32).reshape(-1, 1, 2)

    if model == "similarity":
        M, inliers = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=max_reprojection_error,
            maxIters=max_iters,
            confidence=confidence,
        )
        H = _to_3x3(M) if M is not None else np.eye(3)
    elif model == "affine":
        M, inliers = cv2.estimateAffine2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=max_reprojection_error,
            maxIters=max_iters,
            confidence=confidence,
        )
        H = _to_3x3(M) if M is not None else np.eye(3)
    elif model == "perspective":
        H, inliers = cv2.findHomography(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=max_reprojection_error,
            maxIters=max_iters,
            confidence=confidence,
        )
        if H is None:
            H = np.eye(3)
    else:
        raise ValueError(f"Unknown model: {model!r}")

    if inliers is None:
        mask = np.zeros(src_points.shape[0], dtype=bool)
    else:
        mask = inliers.ravel().astype(bool)
    return H.astype(np.float64), mask
