"""High-level RIFT2 pipeline.

Typical usage::

    import cv2
    from rift2 import RIFT2

    rift = RIFT2()
    im1 = cv2.imread("pair1.jpg")
    im2 = cv2.imread("pair2.jpg")

    r1 = rift.detect_and_describe(im1)
    r2 = rift.detect_and_describe(im2)

    H, matches1, matches2 = rift.match(r1, r2)

The :class:`RIFT2` object holds immutable parameters; call sites that
process many images of the same size can reuse one instance, which keeps
the log-Gabor filter bank cached internally.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import _cuda
from .descriptor import build_max_index_map, compute_descriptors
from .detector import detect_fast_keypoints
from .fsc import estimate_similarity_ransac
from .matcher import match_descriptors
from .orientation import compute_dominant_orientations
from .phase_congruency import _FilterBank, phase_congruency


@dataclass
class RIFT2Result:
    """Container for all intermediates of a single-image RIFT2 pass."""

    keypoints: np.ndarray        # (K, 3) -> (x, y, angle_deg)
    descriptors: np.ndarray      # (K, D) float32
    pc_map: np.ndarray           # (H, W) float64
    mim: np.ndarray              # (H, W) uint8


@dataclass
class RIFT2:
    """Configurable RIFT2 front-end.

    Parameters
    ----------
    nscale, norient
        Log-Gabor bank dimensions. ``norient`` is also the number of
        histogram bins per descriptor cell.
    max_keypoints
        Upper bound on the keypoints retained per image.
    patch_size
        Side length in pixels of the description window.
    ncells
        Grid resolution inside the description window.
    orientation
        If ``True`` (default), estimate dominant orientations. Otherwise
        all keypoints get angle 0 (upright descriptors).
    min_wavelength, mult, sigma_on_f, k, cut_off, g
        Phase-congruency parameters; defaults match the MATLAB reference.
    """

    nscale: int = 4
    norient: int = 6
    max_keypoints: int = 5000
    patch_size: int = 96
    ncells: int = 6
    orientation: bool = True
    min_wavelength: float = 3.0
    mult: float = 1.6
    sigma_on_f: float = 0.75
    k: float = 1.0
    cut_off: float = 0.5
    g: float = 3.0
    device: str = "cpu"

    _filter_bank: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device!r}")
        if self.device == "cuda" and not _cuda.HAS_CUPY:
            raise RuntimeError(
                "device='cuda' requested but CuPy is not installed. "
                "Install cupy-cudaXX matching your CUDA toolkit."
            )

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] in (3, 4):
            code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            return cv2.cvtColor(image, code)
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # ------------------------------------------------------------------ API

    def detect_and_describe(self, image: np.ndarray) -> RIFT2Result:
        """Run the full RIFT2 pipeline on a single image."""
        gray = self._to_gray(image).astype(np.float64)
        bank = self._ensure_bank(gray.shape)
        result, self._filter_bank = self._describe_with_bank(gray, bank)
        return result

    def describe_batch(
        self,
        images: Sequence[np.ndarray],
        max_workers: Optional[int] = None,
    ) -> List[RIFT2Result]:
        """Describe many images while amortising filter-bank construction.

        Images are grouped by shape and each group reuses a single
        log-Gabor bank. Within a group the per-image work runs on a thread
        pool. Threading only delivers wall-clock speedups when the GIL is
        released for most of the runtime, which is the case once the
        Numba descriptor kernel is active and SciPy's pocketfft is doing
        the FFT; with the pure-Python descriptor fallback the threaded
        path mostly serializes and you should pass ``max_workers=1``. The
        filter-bank reuse alone is still worth the call because it
        amortises construction across same-shape inputs.

        Parameters
        ----------
        images
            Iterable of BGR or grayscale images.
        max_workers
            Maximum number of worker threads. ``None`` lets
            :class:`concurrent.futures.ThreadPoolExecutor` choose.

        Returns
        -------
        list of RIFT2Result
            Results in the same order as ``images``.
        """
        prepared: List[Tuple[int, np.ndarray]] = []
        groups: dict[Tuple[int, int], List[int]] = {}
        for idx, image in enumerate(images):
            gray = self._to_gray(image).astype(np.float64)
            prepared.append((idx, gray))
            groups.setdefault(gray.shape, []).append(idx)

        results: List[Optional[RIFT2Result]] = [None] * len(prepared)

        for shape, indices in groups.items():
            bank = self._ensure_bank(shape)

            def _worker(i: int) -> None:
                _, gray = prepared[i]
                res, _ = self._describe_with_bank(gray, bank)
                results[i] = res

            if len(indices) == 1 or max_workers == 1:
                for i in indices:
                    _worker(i)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    list(ex.map(_worker, indices))

        return [r for r in results if r is not None]

    # ------------------------------------------------------------ internals

    def _ensure_bank(self, shape: Tuple[int, int]):
        bank = self._filter_bank
        builder = (
            _cuda._CudaFilterBank.build if self.device == "cuda" else _FilterBank.build
        )
        if (
            bank is None
            or bank.shape != shape
            or bank.nscale != self.nscale
            or bank.norient != self.norient
        ):
            bank = builder(
                shape,
                self.nscale,
                self.norient,
                self.min_wavelength,
                self.mult,
                self.sigma_on_f,
            )
            self._filter_bank = bank
        return bank

    def _describe_with_bank(
        self, gray: np.ndarray, bank: object
    ) -> Tuple[RIFT2Result, object]:
        """Phase-congruency + descriptor pass against a pre-built filter bank.

        Pure function with respect to ``self``; safe to invoke from worker
        threads as long as ``bank`` is treated as read-only (CPU path) or
        a CUDA stream is managed externally (GPU path).
        """
        pc_fn = _cuda.phase_congruency_cuda if self.device == "cuda" else phase_congruency
        M, EO, bank = pc_fn(
            gray,
            nscale=self.nscale,
            norient=self.norient,
            min_wavelength=self.min_wavelength,
            mult=self.mult,
            sigma_on_f=self.sigma_on_f,
            k=self.k,
            cut_off=self.cut_off,
            g=self.g,
            filter_bank=bank,
        )

        Mn = M - M.min()
        peak = Mn.max()
        if peak > 0:
            Mn /= peak

        raw_kps = detect_fast_keypoints(Mn, max_keypoints=self.max_keypoints)

        if self.orientation:
            oriented = compute_dominant_orientations(Mn, raw_kps, self.patch_size)
        else:
            if raw_kps.size == 0:
                oriented = np.empty((0, 3), dtype=np.float32)
            else:
                oriented = np.concatenate(
                    [raw_kps, np.zeros((raw_kps.shape[0], 1), dtype=np.float32)],
                    axis=1,
                )

        mim = build_max_index_map(EO)

        des, valid_idx = compute_descriptors(
            mim,
            oriented,
            patch_size=self.patch_size,
            norient=self.norient,
            ncells=self.ncells,
        )
        oriented = oriented[valid_idx]
        return (
            RIFT2Result(keypoints=oriented, descriptors=des, pc_map=Mn, mim=mim),
            bank,
        )

    # -------------------------------------------------------- matching helpers

    def match(
        self,
        r1: RIFT2Result,
        r2: RIFT2Result,
        model: str = "similarity",
        reprojection_threshold: float = 3.0,
        ratio: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match two :class:`RIFT2Result` objects and estimate a transform.

        Returns a tuple ``(H, pts1, pts2)`` where ``H`` is the ``3 x 3``
        transform mapping image-1 coordinates into image-2 coordinates, and
        ``pts1`` / ``pts2`` are ``(N, 2)`` arrays of inlier correspondences.
        """
        pairs, _ = match_descriptors(r1.descriptors, r2.descriptors, ratio=ratio)
        if pairs.shape[0] < 2:
            return np.eye(3), np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        p1 = r1.keypoints[pairs[:, 0], :2]
        p2 = r2.keypoints[pairs[:, 1], :2]

        # Deduplicate on the right side, as the MATLAB reference does.
        _, unique_idx = np.unique(p2, axis=0, return_index=True)
        unique_idx.sort()
        p1 = p1[unique_idx]
        p2 = p2[unique_idx]

        H, mask = estimate_similarity_ransac(
            p1, p2, model=model, max_reprojection_error=reprojection_threshold
        )
        return H, p1[mask], p2[mask]
