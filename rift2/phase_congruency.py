"""Phase congruency front-end used by RIFT / RIFT2.

This module is a vectorized NumPy port of Peter Kovesi's ``phasecong3`` MATLAB
routine, restricted to the quantities actually consumed downstream by the
RIFT2 pipeline: the maximum-moment map ``M`` (used for keypoint detection)
and the per-scale, per-orientation complex log-Gabor responses ``EO`` (used
to build the Max Index Map that feeds the descriptor).

Reference
---------
Peter Kovesi, "Image Features From Phase Congruency", Videre: A Journal of
Computer Vision Research, MIT Press, Vol. 1, No. 3, 1999.

Performance notes
-----------------
* Radial log-Gabor filters and angular spread functions are constructed on
  demand but only depend on image size and the fixed parameters, so callers
  that process many images of the same size can keep a single ``RIFT2``
  instance and amortize construction across calls (see :func:`_FilterBank`).
* The FFT is computed once per image. All ``nscale * norient`` filter
  responses reuse that single spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    # SciPy's pocketfft is typically faster than numpy.fft and releases the GIL.
    from scipy.fft import fft2 as _fft2, ifft2 as _ifft2
except Exception:  # pragma: no cover - SciPy is optional.
    from numpy.fft import fft2 as _fft2, ifft2 as _ifft2


_EPS = 1e-4


def _lowpass_filter(shape: Tuple[int, int], cutoff: float, order: int) -> np.ndarray:
    """Construct a Butterworth low-pass filter with origin at the FFT corner."""
    if not 0.0 < cutoff <= 0.5:
        raise ValueError("cutoff must lie in (0, 0.5]")
    rows, cols = shape

    def _range(n: int) -> np.ndarray:
        if n % 2:
            return np.arange(-(n - 1) / 2, (n - 1) / 2 + 1) / (n - 1)
        return np.arange(-n / 2, n / 2) / n

    x, y = np.meshgrid(_range(cols), _range(rows))
    radius = np.sqrt(x * x + y * y)
    f = 1.0 / (1.0 + (radius / cutoff) ** (2 * order))
    return np.fft.ifftshift(f)


@dataclass
class _FilterBank:
    """Pre-computed log-Gabor filter bank for a given image size."""

    log_gabor: np.ndarray       # (nscale, H, W) real
    spread: np.ndarray          # (norient, H, W) real
    shape: Tuple[int, int]
    nscale: int
    norient: int

    @classmethod
    def build(
        cls,
        shape: Tuple[int, int],
        nscale: int,
        norient: int,
        min_wavelength: float,
        mult: float,
        sigma_on_f: float,
    ) -> "_FilterBank":
        rows, cols = shape

        def _range(n: int) -> np.ndarray:
            if n % 2:
                return np.arange(-(n - 1) / 2, (n - 1) / 2 + 1) / (n - 1)
            return np.arange(-n / 2, n / 2) / n

        x, y = np.meshgrid(_range(cols), _range(rows))
        radius = np.sqrt(x * x + y * y)
        theta = np.arctan2(-y, x)

        radius = np.fft.ifftshift(radius)
        theta = np.fft.ifftshift(theta)
        radius[0, 0] = 1.0  # avoid log(0)

        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        lp = _lowpass_filter(shape, 0.45, 15)

        log_gabor = np.empty((nscale, rows, cols), dtype=np.float64)
        for s in range(nscale):
            wavelength = min_wavelength * mult ** s
            fo = 1.0 / wavelength
            lg = np.exp(-(np.log(radius / fo) ** 2) / (2.0 * np.log(sigma_on_f) ** 2))
            lg *= lp
            lg[0, 0] = 0.0
            log_gabor[s] = lg

        spread = np.empty((norient, rows, cols), dtype=np.float64)
        for o in range(norient):
            angl = o * np.pi / norient
            ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
            dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
            dtheta = np.abs(np.arctan2(ds, dc))
            dtheta = np.minimum(dtheta * norient / 2.0, np.pi)
            spread[o] = (np.cos(dtheta) + 1.0) / 2.0

        return cls(log_gabor, spread, shape, nscale, norient)


def phase_congruency(
    image: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wavelength: float = 3.0,
    mult: float = 1.6,
    sigma_on_f: float = 0.75,
    k: float = 1.0,
    cut_off: float = 0.5,
    g: float = 3.0,
    filter_bank: "_FilterBank | None" = None,
) -> Tuple[np.ndarray, np.ndarray, "_FilterBank"]:
    """Compute phase congruency and the per-orientation magnitude sums.

    Parameters
    ----------
    image
        2-D grayscale image.
    nscale, norient
        Number of log-Gabor scales and orientations. Defaults match the
        RIFT2 MATLAB reference (``nscale=4``, ``norient=6``).
    min_wavelength, mult, sigma_on_f, k, cut_off, g
        Kovesi phase-congruency parameters. Defaults reproduce the RIFT2
        MATLAB invocation (``mult=1.6``, ``sigmaOnf=0.75``, ``g=3``, ``k=1``).
    filter_bank
        Optional pre-built filter bank for amortized reuse across images of
        the same size.

    Returns
    -------
    M : ndarray, shape (H, W), float32
        Maximum-moment phase congruency map (edge strength).
    cs : ndarray, shape (norient, H, W), float32
        Per-orientation accumulated magnitude responses
        ``sum_s |EO[s, o]|``. The argmax across the first axis is the
        Max Index Map consumed by the descriptor stage. Returning ``cs``
        rather than the full 4-D ``EO`` complex array eliminates a large
        allocation that used to dominate FFT cache pressure.
    filter_bank : _FilterBank
        The (possibly newly created) filter bank, returned for reuse.
    """
    if image.ndim != 2:
        raise ValueError("phase_congruency expects a 2-D grayscale image")

    img = image.astype(np.float32, copy=False)
    rows, cols = img.shape

    if filter_bank is None or filter_bank.shape != (rows, cols) \
            or filter_bank.nscale != nscale or filter_bank.norient != norient:
        filter_bank = _FilterBank.build(
            (rows, cols), nscale, norient, min_wavelength, mult, sigma_on_f
        )

    image_fft = _fft2(img)  # complex64 because img is float32

    cs = np.zeros((norient, rows, cols), dtype=np.float32)

    covx2 = np.zeros((rows, cols), dtype=np.float32)
    covy2 = np.zeros((rows, cols), dtype=np.float32)
    covxy = np.zeros((rows, cols), dtype=np.float32)

    log_gabor = filter_bank.log_gabor
    spread = filter_bank.spread

    # Per-orientation work uses a small ring of (nscale+1) complex buffers
    # so we never materialize the full (nscale, norient, H, W) EO array.
    eo_stack_real = np.empty((nscale, rows, cols), dtype=np.float32)
    eo_stack_imag = np.empty((nscale, rows, cols), dtype=np.float32)

    for o in range(norient):
        angl = float(o * np.pi / norient)
        spread_o = spread[o]

        sum_e = np.zeros((rows, cols), dtype=np.float32)
        sum_o = np.zeros((rows, cols), dtype=np.float32)
        sum_an = np.zeros((rows, cols), dtype=np.float32)
        max_an = None
        tau = 0.0

        for s in range(nscale):
            flt = log_gabor[s] * spread_o
            eo = _ifft2(image_fft * flt).astype(np.complex64, copy=False)
            er = eo.real
            ei = eo.imag
            eo_stack_real[s] = er
            eo_stack_imag[s] = ei

            an = np.hypot(er, ei)
            sum_an += an
            sum_e += er
            sum_o += ei

            if s == 0:
                tau = float(np.median(sum_an)) / np.sqrt(np.log(4.0))
                max_an = an.copy()
            else:
                np.maximum(max_an, an, out=max_an)

        # Accumulate per-orientation magnitude sum for the MIM later.
        cs[o] = sum_an

        x_energy = np.sqrt(sum_e * sum_e + sum_o * sum_o) + _EPS
        mean_e = sum_e / x_energy
        mean_o = sum_o / x_energy

        energy = np.zeros_like(sum_e)
        for s in range(nscale):
            er = eo_stack_real[s]
            ei = eo_stack_imag[s]
            energy += er * mean_e + ei * mean_o - np.abs(er * mean_o - ei * mean_e)

        total_tau = tau * (1.0 - (1.0 / mult) ** nscale) / (1.0 - 1.0 / mult)
        est_mean = total_tau * np.sqrt(np.pi / 2.0)
        est_sigma = total_tau * np.sqrt((4.0 - np.pi) / 2.0)
        noise_thresh = float(est_mean + k * est_sigma)
        np.maximum(energy - noise_thresh, 0.0, out=energy)

        width = (sum_an / (max_an + _EPS) - 1.0) / (nscale - 1)
        weight = 1.0 / (1.0 + np.exp((cut_off - width) * g))

        pc = weight * energy / sum_an
        cangl = float(np.cos(angl))
        sangl = float(np.sin(angl))
        covx = pc * cangl
        covy = pc * sangl
        covx2 += covx * covx
        covy2 += covy * covy
        covxy += covx * covy

    covx2 *= 2.0 / norient
    covy2 *= 2.0 / norient
    covxy *= 4.0 / norient
    denom = np.sqrt(covxy * covxy + (covx2 - covy2) ** 2) + _EPS
    M = (covy2 + covx2 + denom) * 0.5

    return M.astype(np.float32, copy=False), cs, filter_bank
