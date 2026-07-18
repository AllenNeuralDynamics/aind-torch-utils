"""Checkpoint-aware denoise-net workflow recipe."""

from __future__ import annotations

import math
from importlib import import_module
from numbers import Real
from typing import Any, Callable, Dict, Tuple

from aind_torch_utils.transforms import IntensityTransformAdapter
from aind_torch_utils.workflow import Workflow, WorkflowRegistry


_SOURCE_MODULE = "aind_exaspim_image_compression.inference"
_INSTALL_HINT = (
    "Install or update the optional dependency with "
    "`pip install 'aind-torch-utils[denoise-net]'`."
)


def _load_source_api() -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Import the checkpoint-aware source API only when the recipe is built."""
    try:
        source = import_module(_SOURCE_MODULE)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "The 'denoise-net' workflow requires the optional "
            "aind-exaspim-image-compression package. " + _INSTALL_HINT
        ) from exc

    missing = [
        name
        for name in ("load_model", "build_volume_transform")
        if not callable(getattr(source, name, None))
    ]
    if missing:
        raise ImportError(
            "The installed aind-exaspim-image-compression package does not "
            "provide the checkpoint-aware transform API "
            f"({', '.join(missing)}). " + _INSTALL_HINT
        )
    return source.load_model, source.build_volume_transform


def _validate_params(params: Dict[str, Any]) -> Tuple[str, Any]:
    """Validate the deliberately small denoise-net parameter schema."""
    if not isinstance(params, dict):
        raise TypeError("denoise-net workflow parameters must be a dictionary.")

    unknown = sorted(set(params) - {"checkpoint_path", "offset"})
    if unknown:
        raise ValueError(
            "Unknown denoise-net workflow parameter(s): " + ", ".join(unknown)
        )

    checkpoint_path = params.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError(
            "denoise-net workflow parameter 'checkpoint_path' is required "
            "and must be a non-empty string."
        )

    offset = params.get("offset")
    if offset is not None:
        if isinstance(offset, bool) or not isinstance(offset, Real):
            raise TypeError(
                "denoise-net workflow parameter 'offset' must be a finite number."
            )
        offset = float(offset)
        if not math.isfinite(offset):
            raise ValueError(
                "denoise-net workflow parameter 'offset' must be a finite number."
            )
    return checkpoint_path, offset


@WorkflowRegistry.register("denoise-net")
def build_denoise_net(params: Dict[str, Any]) -> Workflow:
    """Load a U-Net and its serialized intensity transform from a checkpoint.

    The source package reconstructs the architecture, weights, and transform
    configuration on CPU. Device placement remains the generic runtime's job.
    """
    checkpoint_path, offset = _validate_params(params)
    load_model, build_volume_transform = _load_source_api()

    loaded = load_model(checkpoint_path, device="cpu")
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ImportError(
            "The installed aind-exaspim-image-compression checkpoint loader "
            "does not return (model, transform). " + _INSTALL_HINT
        )
    model, transform = loaded

    # An explicit volume offset composes around the frozen checkpoint mapping;
    # omitting it deliberately preserves the exact transform returned by the
    # checkpoint loader.
    if offset is not None:
        transform = build_volume_transform(transform, offset=offset)

    try:
        preprocess = IntensityTransformAdapter(transform)
    except TypeError as exc:
        raise ImportError(
            "The transform returned by aind-exaspim-image-compression does "
            "not provide the required forward/inverse API. " + _INSTALL_HINT
        ) from exc

    return Workflow(processor=model, preprocess=preprocess)
