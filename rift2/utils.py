"""Small helpers for visualization and image fusion."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Return a side-by-side visualization of a set of correspondences."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)

    def _to_bgr(img: np.ndarray) -> np.ndarray:
        return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    canvas[:h1, :w1] = _to_bgr(img1)
    canvas[:h2, w1 : w1 + w2] = _to_bgr(img2)

    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)) + w1, int(round(y2)))
        cv2.circle(canvas, p1, 3, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, p2, 3, color, thickness, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
    return canvas


def warp_and_blend(
    moving: np.ndarray,
    reference: np.ndarray,
    H: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Warp ``moving`` into ``reference`` using ``H`` and alpha-blend."""
    h, w = reference.shape[:2]
    warped = cv2.warpPerspective(moving, H, (w, h))
    if warped.ndim == 2:
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    ref = reference if reference.ndim == 3 else cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(ref, 1.0 - alpha, warped, alpha, 0.0)
