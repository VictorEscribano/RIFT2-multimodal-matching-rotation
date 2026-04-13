# RIFT2 — Multimodal Image Matching in Python

A pure-Python implementation of **RIFT2**, a radiation-variation insensitive
feature transform for multimodal image matching, built on top of NumPy,
SciPy and OpenCV.

This repository was originally a MATLAB reference implementation; it has
been fully rewritten in Python so that it can be used as a library from any
Python project and, in the near future, installed via `pip install rift2`.

> J. Li, Q. Hu, M. Ai. **RIFT: Multi-modal image matching based on
> radiation-variation insensitive feature transform.** *IEEE TIP*, 2020.
> [arXiv:1804.09493](https://arxiv.org/abs/1804.09493)
>
> J. Li, W. Xu, Q. Hu, Y. Zhang. **RIFT2: Speeding-up RIFT with A New
> Rotation-Invariance Technique.**
> [arXiv:2303.00319](https://arxiv.org/abs/2303.00319)

---

## 1. What RIFT2 does

RIFT2 matches keypoints between images captured with different sensors
(optical / infrared / SAR / depth / cartographic maps / ...) where
intensity statistics diverge and classical gradient-based descriptors
(SIFT, SURF, ORB) tend to fail.

The pipeline is:

1. **Phase congruency front-end (Kovesi).** A bank of log-Gabor filters is
   convolved with the image; the maximum-moment map is used as an
   illumination-invariant "edge-strength" image. This is the same
   representation used by the original RIFT.
2. **FAST keypoint detection** on the normalized phase-congruency map.
3. **Dominant orientation assignment** using a 24-bin Gaussian-weighted
   histogram of the phase-congruency gradient, with parabolic peak
   interpolation — analogous to SIFT but computed on the PC map.
4. **Max Index Map (MIM).** For every pixel, the orientation channel with
   the largest summed magnitude response is recorded as an integer index in
   `[0, norient)`. The MIM is the core rotation-invariance trick:
   - RIFT1 described each patch by re-running the whole filter bank at
     different orientations.
   - **RIFT2 instead circularly shifts the MIM indices** inside a patch so
     that the globally dominant channel becomes index 0. This makes the
     descriptor rotation-invariant **without** recomputing any convolution,
     which is the main contribution of the RIFT2 paper.
5. **Descriptor.** The rotated patch is partitioned into a `6 x 6` grid;
   each cell contributes a histogram over the `norient` MIM indices. The
   `6 * 6 * norient`-D vector is L2-normalized.
6. **Matching + outlier removal.** Nearest-neighbour matching (or Lowe's
   ratio test) followed by RANSAC estimation of a similarity, affine, or
   perspective transform.

---

## 2. Installation

### From source (current)

```bash
git clone <this-repo>
cd RIFT2-multimodal-matching-rotation
pip install -e .
```

Runtime dependencies:

- `numpy >= 1.21`
- `scipy >= 1.7` (faster FFT; optional at import time)
- `opencv-python >= 4.5`

`pip install rift2` support via PyPI will be added in a later release; the
project layout (`pyproject.toml`, `rift2/` package) is already compatible.

---

## 3. Quick start

```python
import cv2
from rift2 import RIFT2
from rift2.utils import draw_matches, warp_and_blend

im1 = cv2.imread("optical-optical/pair1.jpg")
im2 = cv2.imread("optical-optical/pair2.jpg")

rift = RIFT2(max_keypoints=5000)

res1 = rift.detect_and_describe(im1)
res2 = rift.detect_and_describe(im2)

H, pts1, pts2 = rift.match(res1, res2, model="similarity")

cv2.imwrite("matches.png", draw_matches(im1, im2, pts1, pts2))
cv2.imwrite("fused.png",   warp_and_blend(im1, im2, H))
```

Or run the bundled demo against any of the sample pairs:

```bash
python demo_rift2.py --pair optical-optical
python demo_rift2.py --pair sar-optical
python demo_rift2.py --pair infrared-optical
python demo_rift2.py --pair depth-optical
python demo_rift2.py --pair map-optical
```

---

## 4. API reference

### `rift2.RIFT2`

High-level pipeline. All parameters are keyword-only and have defaults
matching the MATLAB reference.

| Parameter        | Default | Meaning                                                     |
|------------------|---------|-------------------------------------------------------------|
| `nscale`         | 4       | Number of log-Gabor scales                                  |
| `norient`        | 6       | Number of log-Gabor orientations (and MIM bins per cell)    |
| `max_keypoints`  | 5000    | Upper bound on FAST keypoints per image                     |
| `patch_size`     | 96      | Side of the square description window (px)                  |
| `ncells`         | 6       | Grid resolution inside the description window               |
| `orientation`    | `True`  | Toggle dominant-orientation assignment                      |
| `min_wavelength` | 3.0     | Smallest log-Gabor wavelength                               |
| `mult`           | 1.6     | Scaling factor between successive filters                   |
| `sigma_on_f`     | 0.75    | Log-Gabor bandwidth                                         |
| `k`              | 1.0     | Noise-threshold multiplier                                  |
| `cut_off`        | 0.5     | Phase-congruency frequency-spread cutoff                    |
| `g`              | 3.0     | Sharpness of the spread-weighting sigmoid                   |

Methods:

- `detect_and_describe(image) -> RIFT2Result` — runs phase congruency,
  detection, orientation assignment, MIM construction, and descriptor
  extraction on a single BGR / grayscale image.
- `match(r1, r2, model="similarity", reprojection_threshold=3.0, ratio=1.0) -> (H, pts1, pts2)`
  — matches two results and robustly estimates a transform between them.

### `rift2.RIFT2Result`

Dataclass holding the intermediates of one pass: `keypoints` (`(K, 3)` with
`(x, y, angle_deg)`), `descriptors` (`(K, D) float32`), `pc_map` and `mim`.

### Lower-level entry points

```python
from rift2 import (
    phase_congruency,
    detect_fast_keypoints,
    compute_dominant_orientations,
    build_max_index_map,
    compute_descriptors,
    match_descriptors,
    estimate_similarity_ransac,
)
```

Each function is documented in its source file and can be composed
independently if you want to build a custom pipeline (e.g. to swap FAST for
a learned detector or to reuse the MIM for a different descriptor).

---

## 5. Performance notes

Time on RIFT2 is dominated by phase congruency, which performs
`nscale * norient` complex FFTs per image. The Python port exploits several
optimisations:

1. **Pre-computed filter bank.** The radial log-Gabor and angular spread
   filters depend only on image size and are cached inside the `RIFT2`
   instance, so consecutive images of the same size pay zero filter
   construction cost.
2. **Single FFT per image.** The Fourier transform of the input is computed
   once and reused across every scale / orientation combination.
3. **SciPy `pocketfft`.** When SciPy is available, it is used in place of
   `numpy.fft`; `pocketfft` is multi-threaded and releases the GIL.
4. **Vectorised descriptor arithmetic.** MIM rotation compensation is a
   single modulo operation on a `uint8` array; cell histograms use
   `np.bincount`, which is typically faster than any Python-level loop.
5. **OpenCV for sampling and matching.** Patch warping uses
   `cv2.warpAffine` with nearest-neighbour interpolation (correct for a
   categorical MIM); descriptor matching uses `cv2.BFMatcher`; RANSAC is
   performed by `cv2.estimateAffinePartial2D` / `estimateAffine2D` /
   `findHomography`, all of which release the GIL.
6. **Fast FAST.** Keypoint detection uses OpenCV's
   `FastFeatureDetector_create`, which is implemented in C++ SIMD.

If you process many images of the same resolution, **reuse one `RIFT2`
instance** to keep the filter bank hot.

---

## 6. Sample data

Five multimodal image pairs are provided for reproducing the paper results:

```
optical-optical/
infrared-optical/
sar-optical/
depth-optical/
map-optical/
```

Each directory contains `pair1.jpg` and `pair2.jpg`.

---

## 7. Citation

If you use this code, please cite the original RIFT and RIFT2 papers:

```bibtex
@article{li2020rift,
  title={RIFT: Multi-modal image matching based on radiation-variation
         insensitive feature transform},
  author={Li, Jiayuan and Hu, Qingwu and Ai, Mingyao},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={3296--3310},
  year={2020}
}

@article{li2023rift2,
  title={RIFT2: Speeding-up RIFT with A New Rotation-Invariance Technique},
  author={Li, Jiayuan and Xu, Wenpeng and Hu, Qingwu and Zhang, Yongjun},
  journal={arXiv preprint arXiv:2303.00319},
  year={2023}
}
```

---

## 8. License

The Python port is released under the MIT license. The log-Gabor phase
congruency code follows the same permissive terms as Peter Kovesi's
original MATLAB implementation.
