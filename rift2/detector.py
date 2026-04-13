"""FAST keypoint detection on the phase-congruency map."""

from __future__ import annotations

import cv2
import numpy as np


def detect_fast_keypoints(
    pc_map: np.ndarray,
    max_keypoints: int = 5000,
    fast_threshold: int = 1,
    nonmax_suppression: bool = True,
) -> np.ndarray:
    """Detect keypoints on a (normalized) phase-congruency map using FAST.

    The RIFT2 MATLAB reference normalizes the phase-congruency maximum-moment
    map to ``[0, 1]`` and calls ``detectFASTFeatures`` with a very low
    contrast threshold; we mirror that behavior using OpenCV's FAST, run on
    an 8-bit version of the map.

    Parameters
    ----------
    pc_map
        2-D ``float`` phase-congruency map. Will be internally normalized.
    max_keypoints
        Keep at most this many keypoints, sorted by response (descending).
    fast_threshold
        FAST intensity-difference threshold. Kept small because the PC map
        is smooth and the MATLAB reference uses an aggressive threshold.
    nonmax_suppression
        Whether FAST should apply non-maximum suppression.

    Returns
    -------
    keypoints : ndarray of shape (N, 2), float32
        Columns are ``(x, y)`` image coordinates.
    """
    if pc_map.ndim != 2:
        raise ValueError("pc_map must be 2-D")

    pc_min = float(pc_map.min())
    pc_max = float(pc_map.max())
    if pc_max - pc_min < 1e-12:
        return np.empty((0, 2), dtype=np.float32)

    norm = (pc_map - pc_min) / (pc_max - pc_min)
    img8 = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)

    detector = cv2.FastFeatureDetector_create(
        threshold=int(fast_threshold),
        nonmaxSuppression=bool(nonmax_suppression),
    )
    kps = detector.detect(img8, None)
    if not kps:
        return np.empty((0, 2), dtype=np.float32)

    kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:max_keypoints]
    return np.asarray([kp.pt for kp in kps], dtype=np.float32)
