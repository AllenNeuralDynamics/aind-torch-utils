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
