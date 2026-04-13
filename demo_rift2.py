"""End-to-end RIFT2 demo matching two multimodal image pairs.

Run from the repository root, for example::

    python demo_rift2.py --pair optical-optical
    python demo_rift2.py --pair sar-optical --output matches.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from rift2 import RIFT2
from rift2.utils import draw_matches, warp_and_blend


PAIRS = [
    "optical-optical",
    "infrared-optical",
    "sar-optical",
    "depth-optical",
    "map-optical",
    "inf_drone-optical_drone",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="RIFT2 Python demo")
    parser.add_argument("--pair", default="optical-optical", choices=PAIRS)
    parser.add_argument("--output", default="rift2_matches.png")
    parser.add_argument("--fusion", default="rift2_fusion.png")
    parser.add_argument("--max-keypoints", type=int, default=5000)
    args = parser.parse_args()

    base = Path(__file__).parent / args.pair
    im1 = cv2.imread(str(base / "pair1.jpg"))
    im2 = cv2.imread(str(base / "pair2.jpg"))
    if im1 is None or im2 is None:
        raise SystemExit(f"Could not read image pair from {base}")

    rift = RIFT2(max_keypoints=args.max_keypoints)

    t0 = time.perf_counter()
    r1 = rift.detect_and_describe(im1)
    r2 = rift.detect_and_describe(im2)
    t_desc = time.perf_counter() - t0

    t0 = time.perf_counter()
    H, pts1, pts2 = rift.match(r1, r2)
    t_match = time.perf_counter() - t0

    print(f"pair:          {args.pair}")
    print(f"keypoints:     {len(r1.keypoints)} / {len(r2.keypoints)}")
    print(f"inliers:       {len(pts1)}")
    print(f"describe time: {t_desc * 1000:.1f} ms")
    print(f"match time:    {t_match * 1000:.1f} ms")

    vis = draw_matches(im1, im2, pts1, pts2)
    cv2.imwrite(args.output, vis)
    fused = warp_and_blend(im1, im2, H)
    cv2.imwrite(args.fusion, fused)
    print(f"wrote {args.output} and {args.fusion}")


if __name__ == "__main__":
    main()
