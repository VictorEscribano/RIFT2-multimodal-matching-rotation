"""RIFT2 alignment benchmark over a folder of paired images.

Aligns every image in ``--moving-dir`` to its same-named counterpart in
``--reference-dir``, writes a 3-row diagnostic collage per pair, and
produces a JSON / TXT summary with timing and inlier statistics.

Layout of each collage::

    +---------------------------+---------------------------+
    | Reference (e.g. visible)  | Moving (e.g. infrared)    |   row 1
    +---------------------------+---------------------------+
    | RIFT2 matches (side by side, lines connecting inliers) |   row 2
    +--------------------------------------------------------+
    | Reference / warped-moving alpha blend                   |   row 3
    +--------------------------------------------------------+

Defaults are tuned for the Fixed-wing UAV dataset under ``Bottom_Up`` but
both folders, the output directory and the matching extensions can be
overridden from the command line.

Example
-------
    python benchmark_rift2.py \\
        --reference-dir "/home/aunav/Downloads/Fixed-wing-UAV-A'/Bottom_Up/Zoom_Imgs" \\
        --moving-dir   "/home/aunav/Downloads/Fixed-wing-UAV-A'/Bottom_Up/Infrared_Imgs"

The output directory ``RIFT_bench_alignment`` is created next to the two
input folders (i.e. one level above them) unless ``--output-dir`` is
explicitly given.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from rift2 import RIFT2
from rift2.utils import draw_matches


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_BG = (0, 0, 0)
_LABEL_FG = (255, 255, 255)
_BANNER_HEIGHT = 36


@dataclass
class PairResult:
    """Per-pair record stored in the benchmark summary."""

    name: str
    reference_shape: Tuple[int, int]
    moving_shape: Tuple[int, int]
    keypoints_reference: int
    keypoints_moving: int
    raw_matches: int
    inliers: int
    describe_ms: float
    match_ms: float
    total_ms: float
    success: bool
    error: Optional[str] = None


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / w
    return cv2.resize(img, (target_w, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Prepend a labelled banner above ``img``."""
    h, w = img.shape[:2]
    banner = np.full((_BANNER_HEIGHT, w, 3), _LABEL_BG, dtype=np.uint8)
    cv2.putText(banner, text, (12, 25), _FONT, 0.7, _LABEL_FG, 2, cv2.LINE_AA)
    return np.concatenate([banner, img], axis=0)


def _build_collage(
    reference: np.ndarray,
    moving: np.ndarray,
    matches_img: np.ndarray,
    fused: np.ndarray,
    pair_name: str,
    inliers: int,
    elapsed_ms: float,
) -> np.ndarray:
    """Stack the three diagnostic rows into a single image."""
    reference = _to_bgr(reference)
    moving = _to_bgr(moving)
    matches_img = _to_bgr(matches_img)
    fused = _to_bgr(fused)

    # Row 1: reference and moving at half the canvas width each.
    target_w = max(matches_img.shape[1], fused.shape[1])
    half_w = target_w // 2
    ref_panel = _resize_to_width(reference, half_w)
    mov_panel = _resize_to_width(moving, target_w - half_w)
    row1_h = max(ref_panel.shape[0], mov_panel.shape[0])

    def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return np.concatenate([img, pad], axis=0)

    ref_panel = _pad_to_height(_label(ref_panel, "REFERENCE"), row1_h + _BANNER_HEIGHT)
    mov_panel = _pad_to_height(_label(mov_panel, "MOVING"), row1_h + _BANNER_HEIGHT)
    row1 = np.concatenate([ref_panel, mov_panel], axis=1)

    # Row 2: matches.
    matches_img = _resize_to_width(matches_img, row1.shape[1])
    row2 = _label(matches_img, f"RIFT2 INLIERS: {inliers}   |   {elapsed_ms:.0f} ms")

    # Row 3: fused alpha blend.
    fused = _resize_to_width(fused, row1.shape[1])
    row3 = _label(fused, "WARPED MOVING / REFERENCE BLEND")

    collage = np.concatenate([row1, row2, row3], axis=0)

    # Top banner with the pair name.
    title = np.full((_BANNER_HEIGHT + 8, collage.shape[1], 3), _LABEL_BG, dtype=np.uint8)
    cv2.putText(title, f"PAIR  {pair_name}", (16, 28), _FONT, 0.8, _LABEL_FG, 2, cv2.LINE_AA)
    return np.concatenate([title, collage], axis=0)


def _warp_moving_into_reference(
    moving: np.ndarray, reference: np.ndarray, H: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Warp the moving image into the reference frame and alpha-blend.

    ``H`` maps moving-image coordinates into the reference frame because
    the matcher receives ``(moving, reference)`` correspondences.
    """
    h, w = reference.shape[:2]
    warped = cv2.warpPerspective(
        moving, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    if warped.ndim == 2:
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    ref_bgr = _to_bgr(reference)
    return cv2.addWeighted(ref_bgr, 1.0 - alpha, warped, alpha, 0.0)


def _scale_homography(H: np.ndarray, src_scale: float, dst_scale: float) -> np.ndarray:
    """Lift a homography computed at a reduced resolution back to full size.

    ``src_scale`` maps full-resolution source coordinates into the
    downsampled grid that was actually fed to RIFT2; likewise for
    ``dst_scale`` on the destination side. The full-resolution
    homography is therefore::

        H_full = diag(1/dst, 1/dst, 1) @ H @ diag(src, src, 1)
    """
    S_src = np.diag([src_scale, src_scale, 1.0])
    S_dst_inv = np.diag([1.0 / dst_scale, 1.0 / dst_scale, 1.0])
    return S_dst_inv @ H @ S_src


def _resize_max_side(img: np.ndarray, target_max_side: int) -> Tuple[np.ndarray, float]:
    """Resize ``img`` so its longer side equals ``target_max_side``.

    Returns the resized image and the applied scale factor. Images that
    are already small enough are returned unchanged.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= target_max_side:
        return img, 1.0
    scale = target_max_side / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _process_pair(
    rift: RIFT2,
    reference_path: Path,
    moving_path: Path,
    output_path: Path,
    target_side: int,
) -> PairResult:
    name = reference_path.name
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    moving = cv2.imread(str(moving_path), cv2.IMREAD_COLOR)
    if reference is None or moving is None:
        return PairResult(
            name=name,
            reference_shape=(0, 0),
            moving_shape=(0, 0),
            keypoints_reference=0,
            keypoints_moving=0,
            raw_matches=0,
            inliers=0,
            describe_ms=0.0,
            match_ms=0.0,
            total_ms=0.0,
            success=False,
            error="failed to read image",
        )

    try:
        # Two-pass FOV-matched alignment.
        #
        # Pass 1 (coarse): match at a small common resolution so that
        # even with the wide-vs-narrow FOV mismatch we recover a rough
        # homography H_coarse : moving -> reference.
        #
        # Pass 2 (refine): warp reference into the moving frame using
        # H_coarse^-1 at a resolution that matches the IR image. Both
        # pictures now cover the same scene footprint, so the refinement
        # match sees comparable content on both sides and yields an
        # order of magnitude more inliers than a single FOV-mismatched
        # pass. The two transforms are composed for the final H.
        coarse_side = 400
        ref_c, rs_c = _resize_max_side(reference, coarse_side)
        mov_c, ms_c = _resize_max_side(moving, coarse_side)

        t0 = time.perf_counter()
        r_ref_c = rift.detect_and_describe(ref_c)
        r_mov_c = rift.detect_and_describe(mov_c)
        H_c, coarse_mov, _ = rift.match(
            r_mov_c, r_ref_c, model="similarity", reprojection_threshold=6.0
        )
        if coarse_mov.shape[0] < 4:
            raise RuntimeError("coarse pass found too few inliers")
        H_coarse = _scale_homography(H_c, src_scale=ms_c, dst_scale=rs_c)

        # Warp reference into the moving frame so the refinement sees a
        # FOV-matched pair. Target the IR's native size to avoid
        # upsampling the visible unnecessarily.
        mov_h, mov_w = moving.shape[:2]
        ref_in_mov = cv2.warpPerspective(
            reference,
            np.linalg.inv(H_coarse),
            (mov_w, mov_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        ref_proc, ref_scale = _resize_max_side(ref_in_mov, target_side)
        mov_proc, mov_scale = _resize_max_side(moving, target_side)
        r_ref = rift.detect_and_describe(ref_proc)
        r_mov = rift.detect_and_describe(mov_proc)
        t_describe = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        H_refine_proc, pts_mov_proc, pts_ref_proc = rift.match(
            r_mov, r_ref, model="similarity", reprojection_threshold=6.0
        )
        t_match = (time.perf_counter() - t0) * 1000.0

        H_refine = _scale_homography(
            H_refine_proc, src_scale=mov_scale, dst_scale=ref_scale
        )
        # H_refine maps full-res moving -> ref_in_mov (still in the
        # moving frame); H_coarse then carries that into the original
        # reference frame.
        H = H_coarse @ H_refine

        pts_mov = pts_mov_proc / mov_scale if mov_scale else pts_mov_proc
        pts_ref_warped = (
            pts_ref_proc / ref_scale if ref_scale else pts_ref_proc
        )
        if pts_ref_warped.shape[0] > 0:
            pts_h = np.concatenate(
                [pts_ref_warped, np.ones((pts_ref_warped.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            mapped = (H_coarse @ pts_h.T).T
            pts_ref = (mapped[:, :2] / mapped[:, 2:3]).astype(np.float32)
        else:
            pts_ref = pts_ref_warped.astype(np.float32)

        inliers = int(pts_mov.shape[0])
        matches_img = draw_matches(moving, reference, pts_mov, pts_ref)
        fused = _warp_moving_into_reference(moving, reference, H)

        collage = _build_collage(
            reference=reference,
            moving=moving,
            matches_img=matches_img,
            fused=fused,
            pair_name=name,
            inliers=inliers,
            elapsed_ms=t_describe + t_match,
        )
        cv2.imwrite(str(output_path), collage, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return PairResult(
            name=name,
            reference_shape=tuple(reference.shape[:2]),
            moving_shape=tuple(moving.shape[:2]),
            keypoints_reference=int(len(r_ref.keypoints)),
            keypoints_moving=int(len(r_mov.keypoints)),
            raw_matches=int(min(len(r_ref.keypoints), len(r_mov.keypoints))),
            inliers=inliers,
            describe_ms=float(t_describe),
            match_ms=float(t_match),
            total_ms=float(t_describe + t_match),
            success=inliers >= 4,
        )
    except Exception as exc:  # pragma: no cover - exercised on bad data only.
        return PairResult(
            name=name,
            reference_shape=tuple(reference.shape[:2]),
            moving_shape=tuple(moving.shape[:2]),
            keypoints_reference=0,
            keypoints_moving=0,
            raw_matches=0,
            inliers=0,
            describe_ms=0.0,
            match_ms=0.0,
            total_ms=0.0,
            success=False,
            error=str(exc),
        )


def _summarise(results: List[PairResult]) -> dict:
    successes = [r for r in results if r.success]

    def _stats(values: List[float]) -> dict:
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "stdev": 0.0}
        return {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "stdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        }

    inliers = [r.inliers for r in successes]
    describe = [r.describe_ms for r in successes]
    match = [r.match_ms for r in successes]
    total = [r.total_ms for r in successes]

    return {
        "pairs_total": len(results),
        "pairs_succeeded": len(successes),
        "pairs_failed": len(results) - len(successes),
        "inliers": _stats(inliers),
        "describe_ms": _stats(describe),
        "match_ms": _stats(match),
        "total_ms_per_pair": _stats(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("/home/aunav/Downloads/Fixed-wing-UAV-A'/Bottom_Up/Wide_Imgs"),
        help="Directory of reference (e.g. visible / wide) images.",
    )
    parser.add_argument(
        "--moving-dir",
        type=Path,
        default=Path("/home/aunav/Downloads/Fixed-wing-UAV-A'/Bottom_Up/Zoom_Imgs"),
        help="Directory of moving (e.g. infrared) images, named identically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for the per-pair collages and summary file. "
             "Defaults to RIFT_bench_alignment next to the input folders.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="RIFT2 backend device.",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=25000,
        help="Cap on raw FAST keypoints per image (paper default: 5000).",
    )
    parser.add_argument(
        "--target-side",
        type=int,
        default=512,
        help="Common longer-side resolution to which both images are "
             "rescaled before feature extraction. Smaller values are "
             "faster and more robust to scale / FOV mismatch; larger "
             "values are slower but localise features more precisely. "
             "The final fused render is always produced at the original "
             "reference resolution.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N pairs (0 = all). Useful for smoke tests.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"],
        help="Image extensions to consider (case-insensitive).",
    )
    args = parser.parse_args()

    reference_dir: Path = args.reference_dir
    moving_dir: Path = args.moving_dir
    if not reference_dir.is_dir():
        print(f"reference-dir does not exist: {reference_dir}", file=sys.stderr)
        return 1
    if not moving_dir.is_dir():
        print(f"moving-dir does not exist: {moving_dir}", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        # Place RIFT_bench_alignment one level above the input folders
        # (which are siblings under Bottom_Up in the example dataset).
        output_dir = reference_dir.parent / "RIFT_bench_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {e.lower() for e in args.extensions}
    reference_files = sorted(
        p for p in reference_dir.iterdir() if p.suffix.lower() in extensions
    )
    if args.limit:
        reference_files = reference_files[: args.limit]

    if not reference_files:
        print(f"No images found in {reference_dir}", file=sys.stderr)
        return 1

    rift = RIFT2(max_keypoints=args.max_keypoints, device=args.device)

    print(f"reference: {reference_dir}")
    print(f"moving:    {moving_dir}")
    print(f"output:    {output_dir}")
    print(f"device:    {args.device}")
    print(f"pairs:     {len(reference_files)}")
    print()

    results: List[PairResult] = []
    wall_t0 = time.perf_counter()

    for i, ref_path in enumerate(reference_files, start=1):
        mov_path = moving_dir / ref_path.name
        if not mov_path.is_file():
            print(f"[{i:>4}/{len(reference_files)}] {ref_path.name}  SKIP (no moving)")
            continue

        out_path = output_dir / f"{ref_path.stem}_aligned.jpg"
        result = _process_pair(rift, ref_path, mov_path, out_path, args.target_side)
        results.append(result)

        flag = "OK " if result.success else "ERR"
        print(
            f"[{i:>4}/{len(reference_files)}] {result.name:32s} "
            f"{flag}  inliers={result.inliers:5d}  "
            f"desc={result.describe_ms:7.1f}ms  match={result.match_ms:7.1f}ms"
            + (f"  ({result.error})" if result.error else "")
        )

    wall_elapsed = time.perf_counter() - wall_t0

    summary = _summarise(results)
    summary["wall_time_seconds"] = float(wall_elapsed)
    summary["device"] = args.device
    summary["max_keypoints"] = int(args.max_keypoints)
    summary["reference_dir"] = str(reference_dir)
    summary["moving_dir"] = str(moving_dir)

    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps({"summary": summary, "pairs": [asdict(r) for r in results]}, indent=2)
    )

    summary_txt = output_dir / "summary.txt"
    with summary_txt.open("w") as fh:
        fh.write(f"RIFT2 benchmark summary\n")
        fh.write(f"=======================\n\n")
        fh.write(f"reference dir : {reference_dir}\n")
        fh.write(f"moving dir    : {moving_dir}\n")
        fh.write(f"output dir    : {output_dir}\n")
        fh.write(f"device        : {args.device}\n")
        fh.write(f"max keypoints : {args.max_keypoints}\n\n")
        fh.write(f"pairs total     : {summary['pairs_total']}\n")
        fh.write(f"pairs succeeded : {summary['pairs_succeeded']}\n")
        fh.write(f"pairs failed    : {summary['pairs_failed']}\n")
        fh.write(f"wall time       : {summary['wall_time_seconds']:.2f} s\n\n")

        def _row(label: str, stats: dict) -> str:
            return (
                f"{label:18s}"
                f"min={stats['min']:10.2f}  "
                f"max={stats['max']:10.2f}  "
                f"mean={stats['mean']:10.2f}  "
                f"median={stats['median']:10.2f}  "
                f"stdev={stats['stdev']:10.2f}\n"
            )

        fh.write(_row("inliers",     summary["inliers"]))
        fh.write(_row("describe ms", summary["describe_ms"]))
        fh.write(_row("match ms",    summary["match_ms"]))
        fh.write(_row("total ms",    summary["total_ms_per_pair"]))

    print()
    print(summary_txt.read_text())
    print(f"wrote {summary_json.name} and {summary_txt.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
