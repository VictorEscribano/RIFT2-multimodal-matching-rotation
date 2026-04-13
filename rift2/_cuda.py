"""Optional CuPy GPU backend for the phase-congruency front-end.

Only the per-image FFT bottleneck is moved to the GPU; the rest of the
pipeline (FAST, orientation, descriptor) stays on the CPU because it works
on relatively small data and is dominated by integer logic that does not
benefit from GPU acceleration.

Usage is opt-in via ``RIFT2(device='cuda')``. If CuPy is not importable on
the host, :data:`HAS_CUPY` is ``False`` and the public API silently falls
back to the CPU path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore

    HAS_CUPY = True
except Exception:  # pragma: no cover - depends on the user environment.
    cp = None  # type: ignore
    HAS_CUPY = False


_EPS = 1e-4


@dataclass
class _CudaFilterBank:
    """GPU-resident log-Gabor filter bank."""

    log_gabor: "cp.ndarray"     # (nscale, H, W) float32
    spread: "cp.ndarray"        # (norient, H, W) float32
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
    ) -> "_CudaFilterBank":
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not installed; install cupy-cudaXX to use the GPU backend.")

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
        radius[0, 0] = 1.0
        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        # Reuse the CPU low-pass helper to avoid drift between backends.
        from .phase_congruency import _lowpass_filter

        lp = _lowpass_filter(shape, 0.45, 15)

        log_gabor = np.empty((nscale, rows, cols), dtype=np.float32)
        for s in range(nscale):
            wavelength = min_wavelength * mult ** s
            fo = 1.0 / wavelength
            lg = np.exp(-(np.log(radius / fo) ** 2) / (2.0 * np.log(sigma_on_f) ** 2))
            lg *= lp
            lg[0, 0] = 0.0
            log_gabor[s] = lg

        spread = np.empty((norient, rows, cols), dtype=np.float32)
        for o in range(norient):
            angl = o * np.pi / norient
            ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
            dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
            dtheta = np.abs(np.arctan2(ds, dc))
            dtheta = np.minimum(dtheta * norient / 2.0, np.pi)
            spread[o] = (np.cos(dtheta) + 1.0) / 2.0

        return cls(
            log_gabor=cp.asarray(log_gabor),
            spread=cp.asarray(spread),
            shape=shape,
            nscale=nscale,
            norient=norient,
        )


def phase_congruency_cuda(
    image: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wavelength: float = 3.0,
    mult: float = 1.6,
    sigma_on_f: float = 0.75,
    k: float = 1.0,
    cut_off: float = 0.5,
    g: float = 3.0,
    filter_bank: "_CudaFilterBank | None" = None,
) -> Tuple[np.ndarray, np.ndarray, "_CudaFilterBank"]:
    """GPU implementation of :func:`rift2.phase_congruency.phase_congruency`.

    Returns NumPy arrays so the rest of the pipeline can stay on the CPU
    without further code changes.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy is not installed; cannot run the CUDA backend.")
    if image.ndim != 2:
        raise ValueError("phase_congruency_cuda expects a 2-D grayscale image")

    rows, cols = image.shape
    if filter_bank is None or filter_bank.shape != (rows, cols) \
            or filter_bank.nscale != nscale or filter_bank.norient != norient:
        filter_bank = _CudaFilterBank.build(
            (rows, cols), nscale, norient, min_wavelength, mult, sigma_on_f
        )

    img = cp.asarray(image, dtype=cp.float32)
    image_fft = cp.fft.fft2(img)

    EO = cp.empty((nscale, norient, rows, cols), dtype=cp.complex64)
    energy_v0 = cp.zeros((rows, cols), dtype=cp.float32)
    energy_v1 = cp.zeros((rows, cols), dtype=cp.float32)
    energy_v2 = cp.zeros((rows, cols), dtype=cp.float32)
    covx2 = cp.zeros((rows, cols), dtype=cp.float32)
    covy2 = cp.zeros((rows, cols), dtype=cp.float32)
    covxy = cp.zeros((rows, cols), dtype=cp.float32)

    log_gabor = filter_bank.log_gabor
    spread = filter_bank.spread

    for o in range(norient):
        angl = o * np.pi / norient
        spread_o = spread[o]

        sum_e = cp.zeros((rows, cols), dtype=cp.float32)
        sum_o = cp.zeros((rows, cols), dtype=cp.float32)
        sum_an = cp.zeros((rows, cols), dtype=cp.float32)
        max_an = None
        tau = 0.0

        for s in range(nscale):
            flt = log_gabor[s] * spread_o
            eo = cp.fft.ifft2(image_fft * flt).astype(cp.complex64)
            EO[s, o] = eo
            an = cp.abs(eo)
            sum_an = sum_an + an
            sum_e = sum_e + eo.real
            sum_o = sum_o + eo.imag
            if s == 0:
                tau = float(cp.median(sum_an).get()) / float(np.sqrt(np.log(4.0)))
                max_an = an.copy()
            else:
                max_an = cp.maximum(max_an, an)

        energy_v0 += sum_e
        energy_v1 += float(np.cos(angl)) * sum_o
        energy_v2 += float(np.sin(angl)) * sum_o

        x_energy = cp.sqrt(sum_e * sum_e + sum_o * sum_o) + _EPS
        mean_e = sum_e / x_energy
        mean_o = sum_o / x_energy

        energy = cp.zeros_like(sum_e)
        for s in range(nscale):
            e = EO[s, o].real
            od = EO[s, o].imag
            energy = energy + e * mean_e + od * mean_o - cp.abs(e * mean_o - od * mean_e)

        total_tau = tau * (1.0 - (1.0 / mult) ** nscale) / (1.0 - 1.0 / mult)
        est_mean = total_tau * np.sqrt(np.pi / 2.0)
        est_sigma = total_tau * np.sqrt((4.0 - np.pi) / 2.0)
        noise_thresh = float(est_mean + k * est_sigma)
        energy = cp.maximum(energy - noise_thresh, 0.0)

        width = (sum_an / (max_an + _EPS) - 1.0) / (nscale - 1)
        weight = 1.0 / (1.0 + cp.exp((cut_off - width) * g))

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
    denom = cp.sqrt(covxy * covxy + (covx2 - covy2) ** 2) + _EPS
    M = (covy2 + covx2 + denom) * 0.5

    return cp.asnumpy(M).astype(np.float64), cp.asnumpy(EO), filter_bank
