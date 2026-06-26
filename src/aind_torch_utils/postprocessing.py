"""GPU-accelerated GFP masking that plugs into the inference pipeline.

Produces a binary GFP mask by light Gaussian smoothing, a global intensity threshold,
and an optional morphological opening that removes tiny specks. The input is assumed
already normalized to ~``[0, 1]`` by the pipeline (e.g. ``normalize="global"`` with
dataset-wide ``norm_lower``/``norm_upper``), so the threshold is a single value in
``[0, 1]``. The opening is a *local* erosion->dilation (not connected-component
labeling), so the whole op stays elementwise + separable Gaussian + local morphology
and runs batched with no device->host syncs, biased toward oversegmentation.

``cupy`` is imported at module top, so importing this module requires a working CUDA
install. The module is therefore imported lazily from the registry loader (see
:func:`aind_torch_utils.models.load_gfp_mask`) rather than at package import time,
keeping ``import aind_torch_utils`` usable on hosts without a GPU.

Note: when the pipeline runs with flat-field correction enabled (``cfg.flatfield``;
see :data:`PIPELINE_PARAM_KEYS` and the example), the model's input is the
background-flattened signal rather than raw normalized intensity, so ``threshold`` must
be re-tuned for that flattened range (subtraction compresses the dynamic range).
"""
import json
from typing import Optional, Tuple

import cupy as cp
import cupyx.scipy.ndimage as cndi
import torch
from torch import nn

# Keys a params JSON may carry that configure the *pipeline* (PrepWorker
# normalization + flat-field correction), not the mask model. ``read_pipeline_params``
# consumes these and ``GfpMaskModel.from_json`` ignores them, so both live in one file.
PIPELINE_PARAM_KEYS = (
    "normalize",
    "norm_lower",
    "norm_upper",
    "flatfield",
    "flatfield_mode",
    "flatfield_level",
    "flatfield_opening_radius",
    "flatfield_sigma",
)


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
    threshold: float = 0.1,
    smooth_sigma: Optional[Tuple[float, float, float]] = (0.5, 1, 1),
    open_iterations: int = 1,
) -> "cp.ndarray":
    """Create a binary GFP mask for a single 3D volume on the GPU.

    Light Gaussian smooth + global threshold + optional morphological opening to
    drop tiny specks (still biased toward oversegmentation). The input is assumed
    normalized to ~``[0, 1]`` by the pipeline, so ``threshold`` is a single value in
    ``[0, 1]``.

    Parameters
    ----------
    img : cupy.ndarray
        Input volume of shape ``(Z, Y, X)``, normalized to ~``[0, 1]``.
    threshold : float
        Intensity threshold in ``[0, 1]``; voxels ``>= threshold`` are foreground.
        Lower values oversegment more.
    smooth_sigma : tuple of float, optional
        Gaussian sigma (Z, Y, X) for denoising before thresholding. ``None`` skips
        smoothing.
    open_iterations : int
        Iterations of binary opening (erosion->dilation) with a 3x3x3 cube to remove
        tiny dots after thresholding. ``0`` disables cleanup; higher values remove
        larger specks. This is a local op (no connected-component labeling).

    Returns
    -------
    cupy.ndarray
        ``uint8`` mask of shape ``(Z, Y, X)`` with values in ``{0, 1}``.
    """
    x = img.astype(cp.float32)
    if smooth_sigma is not None:
        x = cndi.gaussian_filter(x, sigma=tuple(smooth_sigma))
    mask = x >= threshold
    if open_iterations > 0:
        struct = cp.ones((3, 3, 3), dtype=bool)  # full 3x3x3 cube
        mask = cndi.binary_opening(mask, structure=struct, iterations=open_iterations)
    return mask.astype(cp.uint8)


class GfpMaskModel(nn.Module):
    """Pipeline "model" that produces a GFP mask for each volume in a batch.

    Light Gaussian smooth + global threshold + optional morphological opening, run
    once over the whole ``(B, 1, Z, Y, X)`` batch (no per-volume loop, no
    connected-component labeling). The result equals :func:`create_gfp_mask_gpu`
    applied to each volume.

    Conforms to the inference pipeline's model contract: ``forward`` takes a CUDA
    tensor of shape ``(B, 1, Z, Y, X)`` (normalized to ~``[0, 1]`` by the prep
    worker) and returns a CUDA tensor of shape ``(B, 1, Z, Y, X)`` of ``uint8``
    masks. There are no learnable parameters, so ``deepcopy`` and ``.to(device)``
    (used by the GPU workers) are effectively no-ops.

    Parameters
    ----------
    threshold : float
        Intensity threshold in ``[0, 1]`` (input is globally normalized). Voxels
        ``>= threshold`` are foreground; lower values oversegment more.
    smooth_sigma : tuple of float, optional
        Gaussian sigma (Z, Y, X) for denoising before thresholding. ``None`` skips it.
    open_iterations : int
        Iterations of binary opening (erosion->dilation) with a 3x3x3 cube to remove
        tiny dots after thresholding. ``0`` disables cleanup; higher values remove
        larger specks.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        smooth_sigma: Optional[Tuple[float, float, float]] = (0.5, 1, 1),
        open_iterations: int = 1,
    ):
        """Store masking parameters (no learnable state)."""
        super().__init__()
        self.threshold = threshold
        self.smooth_sigma = smooth_sigma
        self.open_iterations = open_iterations

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
        if params.get("smooth_sigma") is not None:
            params["smooth_sigma"] = tuple(params["smooth_sigma"])
        return cls(**params)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mask the whole batch in one shot (smooth + threshold).

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
            vol = cp.from_dlpack(x).astype(cp.float32)  # (B, 1, Z, Y, X), ~[0, 1]
            if self.smooth_sigma is not None:
                # Zero sigma on the batch/channel axes prevents bleed across volumes,
                # so the batched smooth is identical to processing each independently.
                vol = cndi.gaussian_filter(vol, sigma=(0, 0) + tuple(self.smooth_sigma))
            mask = vol >= self.threshold  # bool, (B, 1, Z, Y, X)
            if self.open_iterations > 0:
                # Unit extent on the batch/channel axes confines the opening to each
                # volume's (Z, Y, X), so the batched result matches per-volume.
                struct = cp.ones((1, 1, 3, 3, 3), dtype=bool)
                mask = cndi.binary_opening(
                    mask, structure=struct, iterations=self.open_iterations
                )
            mask = mask.astype(cp.uint8)
            # Clone into a torch-owned tensor so the cupy memory pool can't reclaim
            # the buffer while the writer's async D2H copy is still in flight.
            # torch.from_dlpack consumes cupy's __dlpack__ directly (no .toDlpack()).
            return torch.from_dlpack(mask).clone()
