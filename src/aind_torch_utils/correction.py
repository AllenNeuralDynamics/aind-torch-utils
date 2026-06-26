"""Pure-numpy image corrections shared by the inference pipeline.

Kept dependency-light (numpy only, no torch/tensorstore/cupy) so the math can be
imported and unit-tested on hosts without the GPU/IO stack.
"""
import numpy as np


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
