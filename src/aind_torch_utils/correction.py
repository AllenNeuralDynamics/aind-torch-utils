"""Pure-numpy image corrections shared by the inference pipeline.

Kept dependency-light (numpy only, no torch/tensorstore/cupy) so the math can be
imported and unit-tested on hosts without the GPU/IO stack.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BackgroundField:
    """A coarse global background, sampled per-block at full resolution on demand.

    Storing the field at the *coarse* pyramid resolution (a few MB) and interpolating
    only the voxels a block needs avoids materializing a full-resolution background of
    the entire volume (which can be tens of GB). Because the field is global and is
    sampled by absolute coordinates, every block/overlap sees identical values -> the
    correction is seam-free.

    Attributes
    ----------
    field : np.ndarray
        Coarse background ``(Cz, Cy, Cx)`` in raw intensity units.
    scale : tuple of float
        Processing-level voxels per coarse voxel, per axis
        (``processing_size / coarse_size``).
    """

    field: np.ndarray
    scale: Tuple[float, float, float]

    @property
    def mean(self) -> float:
        """Mean intensity of the coarse field (~ the full-resolution mean)."""
        return float(self.field.mean())


def _axis_index_weights(start: int, stop: int, scale: float, n_coarse: int):
    """Linear-interpolation indices/weights mapping processing coords -> coarse coords.

    Maps absolute processing voxel indices ``[start, stop)`` onto the coarse grid with
    center alignment (``coarse = (proc + 0.5) / scale - 0.5``), clamped to valid range.
    """
    proc = np.arange(start, stop, dtype=np.float64)
    fc = np.clip((proc + 0.5) / scale - 0.5, 0.0, n_coarse - 1)
    i0 = np.floor(fc).astype(np.intp)
    i1 = np.minimum(i0 + 1, n_coarse - 1)
    w = (fc - i0).astype(np.float32)
    return i0, i1, w


def sample_background(
    field: np.ndarray,
    scale: Tuple[float, float, float],
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Trilinearly sample a coarse background over an absolute block bbox.

    Returns a ``(z1-z0, y1-y0, x1-x0)`` ``float32`` array. Implemented as three
    separable 1-D interpolations (cheap: the coarse axes stay small), depends only on
    absolute coordinates, so adjacent blocks agree exactly on shared voxels.
    """
    cz, cy, cx = field.shape
    f = field.astype(np.float32, copy=False)

    iz0, iz1, wz = _axis_index_weights(z0, z1, scale[0], cz)
    a = f[iz0] * (1.0 - wz[:, None, None]) + f[iz1] * wz[:, None, None]  # (Dz,Cy,Cx)

    iy0, iy1, wy = _axis_index_weights(y0, y1, scale[1], cy)
    b = a[:, iy0] * (1.0 - wy[None, :, None]) + a[:, iy1] * wy[None, :, None]

    ix0, ix1, wx = _axis_index_weights(x0, x1, scale[2], cx)
    out = b[:, :, ix0] * (1.0 - wx[None, None, :]) + b[:, :, ix1] * wx[None, None, :]
    return out.astype(np.float32, copy=False)


def normalize_global(block, lower: float, upper: float, eps: float = 1e-6):
    """Clip to ``[lower, upper]`` then affine-scale to ``[0, 1]`` (global norm).

    Mirrors the ``normalize="global"`` branch of ``PrepWorker.run`` so the pipeline and
    the tuner share one implementation. Uses the array's own ``.clip`` method, so it
    works on both numpy and cupy arrays without importing either.

    Parameters
    ----------
    block : np.ndarray or cupy.ndarray
        Raw-intensity array.
    lower, upper : float
        Global intensity bounds (e.g. dataset-wide percentiles).
    eps : float
        Floor for the divisor when ``upper == lower``.

    Returns
    -------
    Same array type as ``block``, normalized to ~``[0, 1]``.
    """
    scale = max(upper - lower, eps)
    return (block.clip(lower, upper) - lower) / scale


def fill_no_data(volume, empty_threshold: float = 0.0):
    """Replace no-data voxels with the valid-voxel median (for background estimation).

    Empty/zero regions of a fused volume (value ``<= empty_threshold``) otherwise drag a
    morphological background estimate *down* near the empty/tissue interface -- a
    grey-opening's erosion reaches into the zeros -- so a white top-hat (``subtract``)
    correction leaves a bright residual rim exactly at the tissue boundary. Replacing
    the no-data voxels with a representative tissue-background level (the median of the
    valid voxels) before the opening/smoothing keeps the estimate flat across it.
    The empty region still corrects to ~0 (raw 0 minus a positive background, clamped).

    Parameters
    ----------
    volume : np.ndarray
        Raw-intensity array (e.g. the coarse flat-field level).
    empty_threshold : float
        Voxels ``<= empty_threshold`` are treated as no-data. ``0.0`` treats only
        exactly-zero (the typical fused fill) as empty.

    Returns
    -------
    np.ndarray
        A copy with no-data voxels filled, or ``volume`` unchanged when every voxel is
        valid or none are.
    """
    valid = volume > empty_threshold
    n_valid = int(valid.sum())
    if n_valid == 0 or n_valid == volume.size:
        return volume
    out = volume.copy()
    out[~valid] = np.median(volume[valid])
    return out


def scale_params(
    smooth_sigma: Tuple[float, float, float],
    open_iterations: int,
    factor: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], int]:
    """Rescale voxel-unit mask params from a tune level to a target level.

    ``smooth_sigma`` and ``open_iterations`` are measured in voxels, so a value tuned on
    a coarse level must be multiplied by ``factor = scale_tune / scale_target`` (>= 1
    when the target level is finer) to represent the same physical size at the target
    resolution. ``open_iterations`` is an in-plane radius, scaled by the mean of the
    Y/X factors and rounded. This is approximate (anisotropy, integer rounding).

    Returns
    -------
    (scaled_smooth_sigma, scaled_open_iterations)
    """
    scaled_sigma = tuple(float(s) * float(f) for s, f in zip(smooth_sigma, factor))
    yx = (float(factor[1]) + float(factor[2])) / 2.0
    scaled_open = int(round(open_iterations * yx))
    return scaled_sigma, scaled_open


def apply_flatfield(
    block: np.ndarray,
    background: np.ndarray,
    mode: str = "subtract",
    eps: float = 1e-6,
    bg_mean: float = 1.0,
) -> np.ndarray:
    """Flat-field correct a block in raw intensity units.

    The correction is purely point-wise: each voxel depends only on its own value and
    the background at the *same* coordinate. Since the background is a single global
    field sampled by absolute coordinates, two blocks that overlap (or are adjacent
    via halo) produce identical corrected values on shared voxels -- i.e. the
    correction is seam-free by construction.

    Parameters
    ----------
    block : np.ndarray
        Raw-intensity block to correct.
    background : np.ndarray
        Background estimate aligned to ``block`` (same shape).
    mode : str
        ``"subtract"`` for a white top-hat (``max(block - background, 0)``) or
        ``"divide"`` for a multiplicative shading/gain correction
        (``block / max(background, eps) * bg_mean``).
    eps : float
        Small floor for the divisor in ``"divide"`` mode.
    bg_mean : float
        Mean of the global background, used in ``"divide"`` mode to preserve the
        overall intensity scale.

    Returns
    -------
    np.ndarray
        Corrected ``float32`` block, same shape as ``block``.
    """
    block = block.astype(np.float32, copy=False)
    if mode == "subtract":
        return np.maximum(block - background, 0.0).astype(np.float32, copy=False)
    if mode == "divide":
        denom = np.maximum(background, eps)
        return (block / denom * bg_mean).astype(np.float32, copy=False)
    raise ValueError(f"Unknown flatfield mode: {mode!r}")
