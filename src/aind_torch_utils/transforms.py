"""Block-scoped input transforms injected into the tiled-inference runtime.

A **preprocessor** is an input-domain operation applied to each block *before* it is
tiled into patches (normalization, calibration correction, ...). The runtime treats a
preprocessor as an opaque object with two contracts:

``forward(block, ctx) -> (block, state)``
    Runs in the prep stage. ``state`` is an opaque, **per-block** blob (see
    :class:`~aind_torch_utils.workers.Batch.transform_state`); return ``None`` when
    nothing needs to travel to the writer.

``inverse(block, state, ctx) -> block`` *(invertible transforms only)*
    Runs in the writer stage. Normalization is the canonical invertible transform:
    the affine denormalization is the mathematical inverse of the affine
    normalization, so both halves live in **one** class and the state schema is that
    class's private business rather than a contract two classes must secretly share.

Keeping forward + inverse in one object is the point of
:class:`InvertibleBlockTransform`: splitting them would re-introduce the coupling
this refactor removes (issue #25 §3.2).

The concrete normalizers here reproduce the exact arithmetic the ``PrepWorker`` used
to inline, so wiring them in (PR2b) leaves existing runs byte-for-byte identical.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aind_torch_utils.context import BlockContext


# "after_finalize": invert once per block after merge (correct + fast for linear
#   inverses; the default). "before_accumulate": invert each patch before merge
#   (needed only when a nonlinear inverse must not commute with averaging).
InverseStage = str


@runtime_checkable
class BlockPreprocessor(Protocol):
    """Input-domain op applied per block. Returns ``(block, state)``.

    ``state`` is opaque, per-block, and travels to the writer only for invertible
    transforms; a one-way preprocessor returns ``None``.
    """

    def forward(
        self, block: np.ndarray, ctx: "BlockContext"
    ) -> Tuple[np.ndarray, Any]:
        """Apply the transform to ``block`` and return ``(block, state)``."""
        ...


@runtime_checkable
class InvertibleBlockTransform(BlockPreprocessor, Protocol):
    """A preprocessor whose effect on model outputs can be undone in the writer.

    ``inverse_stage`` declares where the writer must apply :meth:`inverse` relative
    to seam merging (see :data:`InverseStage`).
    """

    inverse_stage: InverseStage

    def inverse(
        self, block: np.ndarray, state: Any, ctx: "BlockContext"
    ) -> np.ndarray:
        """Undo :meth:`forward` on a finalized output block using ``state``."""
        ...


def is_invertible(obj: Any) -> bool:
    """True if ``obj`` exposes an ``inverse`` method (duck-typed, not isinstance)."""
    return callable(getattr(obj, "inverse", None))


class _AffineNormalizer:
    """Shared affine inverse for the linear normalizers.

    ``state`` is the ``(mn, mx)`` pair each subclass records in :meth:`forward`; the
    inverse maps a normalized value ``v`` back to ``v * max(mx - mn, eps) + mn`` in
    float32 -- exactly what the writer used to compute per patch.
    """

    inverse_stage: InverseStage = "after_finalize"

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def inverse(
        self, block: np.ndarray, state: Any, ctx: "BlockContext"
    ) -> np.ndarray:
        mn, mx = state
        scale = max(mx - mn, self.eps)
        return (block * np.float32(scale) + np.float32(mn)).astype(
            np.float32, copy=False
        )


class PercentileNormalizer(_AffineNormalizer):
    """Normalize a block to [0, 1] using its own ``[lower, upper]`` percentiles.

    Mirrors the legacy ``normalize="percentile"`` path: percentiles are computed over
    the whole (expanded) block and the block is affinely rescaled **in place** on the
    float32 array the prep stage owns.
    """

    def __init__(self, lower: float, upper: float, eps: float = 1e-6):
        super().__init__(eps=eps)
        self.lower = float(lower)
        self.upper = float(upper)

    def forward(
        self, block: np.ndarray, ctx: "BlockContext"
    ) -> Tuple[np.ndarray, Any]:
        block = _as_float32(block)
        mn, mx = np.percentile(block, [self.lower, self.upper])
        scale = max(mx - mn, self.eps)
        block -= mn
        block /= scale
        return block, (float(mn), float(mx))


class GlobalNormalizer(_AffineNormalizer):
    """Clip a block to a fixed ``[lower, upper]`` range, then rescale to [0, 1].

    Mirrors the legacy ``normalize="global"`` path (clip first, then normalize).
    """

    def __init__(self, lower: float, upper: float, eps: float = 1e-6):
        super().__init__(eps=eps)
        self.lower = float(lower)
        self.upper = float(upper)

    def forward(
        self, block: np.ndarray, ctx: "BlockContext"
    ) -> Tuple[np.ndarray, Any]:
        lo, hi = self.lower, self.upper
        scale = max(hi - lo, self.eps)
        out = np.clip(block, lo, hi)
        out = (out - lo) / scale
        return out.astype(np.float32, copy=False), (float(lo), float(hi))


class IdentityTransform:
    """No-op invertible transform (legacy ``normalize=False``).

    Carries no state and its inverse returns the block unchanged, so an output that
    "inverts" through it is written exactly as the model produced it.
    """

    inverse_stage: InverseStage = "after_finalize"

    def forward(
        self, block: np.ndarray, ctx: "BlockContext"
    ) -> Tuple[np.ndarray, Any]:
        return block, None

    def inverse(
        self, block: np.ndarray, state: Any, ctx: "BlockContext"
    ) -> np.ndarray:
        return block


class Clip:
    """One-way clip of the (normalized) block to ``[lo, hi]``. Non-invertible.

    Mirrors the legacy ``clip_norm`` step, which ran after normalization. It only
    affects the model input, so it produces no state and defines no ``inverse``.
    """

    def __init__(self, lo: float, hi: float):
        self.lo = float(lo)
        self.hi = float(hi)

    def forward(
        self, block: np.ndarray, ctx: "BlockContext"
    ) -> Tuple[np.ndarray, Any]:
        return np.clip(block, self.lo, self.hi), None


class Sequential:
    """Compose preprocessors left-to-right; invert the invertible members in reverse.

    The composite ``state`` is the tuple of member states (opaque to the runtime).
    :meth:`inverse` walks the members back-to-front and applies ``inverse`` only on
    those that define it, so non-invertible steps (e.g. :class:`Clip`) are skipped.
    """

    def __init__(self, steps: List[BlockPreprocessor]):
        self.steps = list(steps)

    @property
    def inverse_stage(self) -> InverseStage:
        stages = {
            s.inverse_stage
            for s in self.steps
            if is_invertible(s) and hasattr(s, "inverse_stage")
        }
        if len(stages) > 1:
            raise ValueError(
                f"Sequential members disagree on inverse_stage: {stages}. "
                "Compose transforms that invert in the same stage."
            )
        return stages.pop() if stages else "after_finalize"

    def forward(
        self, block: np.ndarray, ctx: "BlockContext"
    ) -> Tuple[np.ndarray, Any]:
        states: List[Any] = []
        for step in self.steps:
            block, state = step.forward(block, ctx)
            states.append(state)
        return block, tuple(states)

    def inverse(
        self, block: np.ndarray, state: Any, ctx: "BlockContext"
    ) -> np.ndarray:
        for step, step_state in zip(reversed(self.steps), reversed(state)):
            if is_invertible(step):
                block = step.inverse(block, step_state, ctx)
        return block


def _as_float32(block: np.ndarray) -> np.ndarray:
    """Return ``block`` as float32, without copying when it already is.

    The prep stage passes a float32 block it owns, so callers there get in-place
    normalization (matching the legacy hot path); other callers get a safe copy.
    """
    if block.dtype == np.float32:
        return block
    return block.astype(np.float32)


def from_config(
    normalize: Any,
    norm_lower: float,
    norm_upper: float,
    eps: float,
    clip_norm: Any,
) -> BlockPreprocessor:
    """Synthesize the default preprocessor from legacy config fields.

    This is the bridge that keeps ``run()`` backward-compatible: when no explicit
    ``preprocess`` is injected, the runtime builds the transform these fields used to
    describe. The result is byte-for-byte equivalent to the old inline prep path.

    Parameters
    ----------
    normalize : {"percentile", "global"} or False
        Normalization strategy (``InferenceConfig.normalize``).
    norm_lower, norm_upper : float
        Percentiles (percentile mode) or the clip range (global mode).
    eps : float
        Division epsilon.
    clip_norm : bool or (float, float)
        ``True`` clips the normalized block to [0, 1]; a pair clips to that range;
        ``False`` disables clipping. Applied after normalization, as before.
    """
    if normalize == "percentile":
        base: BlockPreprocessor = PercentileNormalizer(norm_lower, norm_upper, eps)
    elif normalize == "global":
        base = GlobalNormalizer(norm_lower, norm_upper, eps)
    else:  # False / disabled
        base = IdentityTransform()

    clip: Optional[Clip] = None
    if clip_norm:
        if clip_norm is True:
            clip = Clip(0.0, 1.0)
        else:
            lo, hi = clip_norm
            clip = Clip(lo, hi)

    if clip is None:
        return base
    return Sequential([base, clip])
