"""GPU-accelerated post-processing functions that plug into the inference pipeline.

This module mirrors a CPU scikit-image / scipy.ndimage masking routine on the GPU
using `cupy <https://cupy.dev/>`_ and `cuCIM <https://github.com/rapidsai/cucim>`_.

The GPU stack (``cupy`` + ``cucim``) is imported at module top, so importing this
module requires a working CUDA + cuCIM installation. The module is therefore imported
lazily from the registry loader (see :func:`aind_torch_utils.models.load_gfp_mask`)
rather than at package import time, keeping ``import aind_torch_utils`` usable on hosts
without a GPU.
"""
import json
from typing import Optional, Tuple

import cupy as cp
import cupyx.scipy.ndimage as cndi
import torch
from cucim.skimage import exposure, filters, morphology
from torch import nn

# Keys a params JSON may carry that configure the *pipeline* (PrepWorker
# normalization), not the mask model. ``read_pipeline_params`` consumes these and
# ``GfpMaskModel.from_json`` ignores them, so both live in one config file.
PIPELINE_PARAM_KEYS = ("normalize", "norm_lower", "norm_upper")


def read_pipeline_params(path: Optional[str]) -> dict:
    """Return the pipeline-normalization keys present in a params JSON.

    Returns an empty dict if ``path`` is ``None`` or the file carries none of
    :data:`PIPELINE_PARAM_KEYS`. These configure ``InferenceConfig`` normalization,
    separate from the mask-model params consumed by :meth:`GfpMaskModel.from_json`.
    """
    if not path:
        return {}
    with open(path) as f:
        params = json.load(f)
    return {k: params[k] for k in PIPELINE_PARAM_KEYS if k in params}


def create_gfp_mask_gpu(
    img,
    intensity_percentiles,
    min_intensity: Optional[float] = None,
    background_sigma: Tuple[float, float, float] = (2, 10, 10),
    min_object_size: int = 20,
    hole_size: int = 64,
    low_thresh: float = 0.05,
    high_thresh: float = 0.20,
) -> "cp.ndarray":
    """Create a binary GFP mask for a single 3D volume on the GPU.

    GPU port of the CPU ``create_gfp_mask`` routine. Operates on one ``(Z, Y, X)``
    volume; the result is a ``uint8`` cupy array of the same shape with values in
    ``{0, 1}``.

    Parameters
    ----------
    img : cupy.ndarray
        Input volume of shape ``(Z, Y, X)``.
    intensity_percentiles : tuple of float
        ``(low, high)`` intensity values used to rescale the background-subtracted
        image to ``[0, 1]`` (computed once per tile/volume by the caller).
    min_intensity : float, optional
        If provided, voxels with raw intensity below this value are forced to 0.
    background_sigma : tuple of float
        Anisotropic Gaussian sigma for background estimation (light-sheet).
    min_object_size : int
        Minimum connected-component size to keep.
    hole_size : int
        Maximum hole area to fill.
    low_thresh, high_thresh : float
        Low/high thresholds for hysteresis thresholding (in ``[0, 1]`` space).

    Returns
    -------
    cupy.ndarray
        ``uint8`` mask of shape ``(Z, Y, X)`` with values in ``{0, 1}``.
    """
    raw = img.astype(cp.float32)
    smooth = _preprocess(
        raw,
        intensity_percentiles,
        background_sigma=tuple(background_sigma),
        smooth_sigma=(0.5, 1, 1),
    )
    return _threshold_and_clean(
        smooth,
        raw,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        hole_size=hole_size,
        min_object_size=min_object_size,
        min_intensity=min_intensity,
    )


def _preprocess(vol, intensity_percentiles, background_sigma, smooth_sigma):
    """Background-subtract, rescale, and smooth (the cheap, per-voxel ops).

    Works on an N-D float32 array. ``background_sigma``/``smooth_sigma`` must match
    the array's rank; pass ``0`` on any axis that must not be blurred (e.g. the batch
    and channel axes of a ``(B, 1, Z, Y, X)`` array), which makes the batched result
    identical to processing each volume independently.
    """
    bg = cndi.gaussian_filter(vol, sigma=background_sigma)
    img_bs = cp.clip(vol - bg, 0, None)
    img_clean = exposure.rescale_intensity(
        img_bs,
        in_range=(intensity_percentiles[0], intensity_percentiles[1]),
        out_range=(0, 1),
    )
    return cndi.gaussian_filter(img_clean, sigma=smooth_sigma)


def _threshold_and_clean(
    smooth,
    raw,
    low_thresh,
    high_thresh,
    hole_size,
    min_object_size,
    min_intensity,
) -> "cp.ndarray":
    """Hysteresis threshold + morphological cleanup for a single 3D volume.

    These steps use connected-component labeling, so they must run per-volume (never
    on a stacked batch, which would connect adjacent volumes across the batch axis).
    """
    # Hysteresis threshold (consistent across all blocks).
    mask = filters.apply_hysteresis_threshold(smooth, low=low_thresh, high=high_thresh)

    # Cleanup — fill holes, close to bridge spine necks, THEN remove small.
    mask = morphology.remove_small_holes(mask, max_size=hole_size)
    mask = morphology.binary_closing(mask, morphology.ball(1))
    mask = morphology.remove_small_objects(mask, max_size=min_object_size)

    if min_intensity is not None:
        mask[raw < min_intensity] = 0

    return mask.astype(cp.uint8)


class GfpMaskModel(nn.Module):
    """Pipeline "model" that produces a GFP mask for each volume in a batch.

    The cheap per-voxel ops (background subtraction, rescale, smoothing) run once over
    the whole ``(B, 1, Z, Y, X)`` batch; only the labeling/morphology cleanup loops
    per-volume. The result equals :func:`create_gfp_mask_gpu` applied to each volume.

    Conforms to the inference pipeline's model contract: ``forward`` takes a CUDA
    tensor of shape ``(B, 1, Z, Y, X)`` (normalized to ~``[0, 1]`` by the prep
    worker) and returns a CUDA tensor of shape ``(B, 1, Z, Y, X)`` of ``uint8``
    masks. There are no learnable parameters, so ``deepcopy`` and ``.to(device)``
    (used by the GPU workers) are effectively no-ops.

    Parameters
    ----------
    intensity_percentiles : tuple of float
        Passed to :func:`create_gfp_mask_gpu`. With pipeline percentile
        normalization the input is already in ``[0, 1]``, so ``(0.0, 1.0)`` makes
        the rescale a near-identity and keeps thresholds in their tuned range.
    background_sigma, min_object_size, hole_size, low_thresh, high_thresh,
    min_intensity
        Forwarded to :func:`create_gfp_mask_gpu`.
    """

    def __init__(
        self,
        intensity_percentiles: Tuple[float, float] = (0.0, 1.0),
        background_sigma: Tuple[float, float, float] = (2, 10, 10),
        min_object_size: int = 20,
        hole_size: int = 64,
        low_thresh: float = 0.05,
        high_thresh: float = 0.20,
        min_intensity: Optional[float] = None,
    ):
        """Store masking parameters (no learnable state)."""
        super().__init__()
        self.intensity_percentiles = intensity_percentiles
        self.background_sigma = background_sigma
        self.min_object_size = min_object_size
        self.hole_size = hole_size
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.min_intensity = min_intensity

    @classmethod
    def from_json(cls, path: str) -> "GfpMaskModel":
        """Build a model from a JSON file of masking parameters.

        Keys mirror the ``__init__`` arguments; missing keys fall back to
        defaults and an unrecognized key raises ``TypeError`` (catches typos).
        Pipeline-normalization keys (:data:`PIPELINE_PARAM_KEYS`) are ignored here so
        the same file can also configure the pipeline.

        Parameters
        ----------
        path : str
            Path to a JSON object of parameters.

        Returns
        -------
        GfpMaskModel
        """
        with open(path) as f:
            params = json.load(f)
        # Drop pipeline-only keys; the rest are mask-model params.
        params = {k: v for k, v in params.items() if k not in PIPELINE_PARAM_KEYS}
        # JSON arrays -> tuples for the sequence-valued params.
        for key in ("intensity_percentiles", "background_sigma"):
            if params.get(key) is not None:
                params[key] = tuple(params[key])
        return cls(**params)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GFP mask to each volume in the batch.

        Parameters
        ----------
        x : torch.Tensor
            CUDA tensor of shape ``(B, 1, Z, Y, X)``.

        Returns
        -------
        torch.Tensor
            CUDA ``uint8`` tensor of shape ``(B, 1, Z, Y, X)`` on the same device.
        """
        # Honor the worker's device (e.g. cuda:1) so cupy doesn't default to GPU 0.
        idx = x.device.index if x.device.index is not None else 0
        with cp.cuda.Device(idx):
            raw = cp.from_dlpack(x)  # zero-copy view, (B, 1, Z, Y, X), [0, 1]
            # Cheap per-voxel ops run ONCE over the whole batch. Zero sigma on the
            # batch/channel axes prevents any bleed across volumes, so this is
            # numerically identical to processing each volume independently.
            smooth = _preprocess(
                raw.astype(cp.float32),
                self.intensity_percentiles,
                background_sigma=(0, 0) + tuple(self.background_sigma),
                smooth_sigma=(0, 0, 0.5, 1, 1),
            )
            # Sync-heavy labeling/morphology stays per-volume.
            outs = [
                _threshold_and_clean(
                    smooth[b, 0],
                    raw[b, 0],
                    low_thresh=self.low_thresh,
                    high_thresh=self.high_thresh,
                    hole_size=self.hole_size,
                    min_object_size=self.min_object_size,
                    min_intensity=self.min_intensity,
                )
                for b in range(raw.shape[0])
            ]
            mask = cp.stack(outs)[:, None]  # (B, 1, Z, Y, X) uint8
            # Clone into a torch-owned tensor so the cupy memory pool can't reclaim
            # the buffer while the writer's async D2H copy is still in flight.
            # torch.from_dlpack consumes cupy's __dlpack__ directly (no .toDlpack()).
            return torch.from_dlpack(mask).clone()
