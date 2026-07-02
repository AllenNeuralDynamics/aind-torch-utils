"""Per-output specification and output-domain post-processors.

A multi-output model returns a stacked ``(B, N, Z, Y, X)`` tensor, and different
outputs want different handling: a mask channel wants max-merge + threshold + uint8,
while a distance-transform channel wants mean-merge + identity + float32. So merge,
post-processing, inversion, and dtype are bound **per output** in an
:class:`OutputSpec` (issue #25 §3.5), not fixed globally on the writer.

A :class:`BlockPostProcessor` is an output-domain transform (threshold, argmax, label
cleanup, morphology) run on the finalized *expanded* block -- doing it before the halo
crop gives neighborhood ops their context, provided their radius is within the halo.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import numpy as np

from aind_torch_utils.accumulators import BlockAccumulatorFactory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aind_torch_utils.context import BlockContext


@runtime_checkable
class BlockPostProcessor(Protocol):
    """Output-domain transform on the finalized (expanded) block."""

    def __call__(self, block: np.ndarray, ctx: "BlockContext") -> np.ndarray:
        """Return the post-processed block (same spatial shape)."""
        ...


@dataclass
class OutputSpec:
    """One model output channel: where it goes and how it is finalized.

    Attributes
    ----------
    store : ts.TensorStore
        Destination store. Its dtype drives the final cast (integer stores clip).
    accumulator_factory : BlockAccumulatorFactory
        How overlapping patches merge for *this* output.
    postprocess : BlockPostProcessor, optional
        Output-domain transform applied on the finalized expanded block.
    invert : bool
        Whether the shared input transform's inverse is applied to this output
        (only some outputs live in input-intensity space).
    """

    store: Any
    accumulator_factory: BlockAccumulatorFactory
    postprocess: Optional[BlockPostProcessor] = None
    invert: bool = False


class Threshold:
    """Threshold a block to two levels: ``block > thresh ? above : below``.

    numpy-only; the default (``above=1, below=0``) yields a binary mask.
    """

    def __init__(self, thresh: float = 0.0, above: float = 1.0, below: float = 0.0):
        self.thresh = float(thresh)
        self.above = float(above)
        self.below = float(below)

    def __call__(self, block: np.ndarray, ctx: "BlockContext") -> np.ndarray:
        return np.where(block > self.thresh, self.above, self.below).astype(
            np.float32, copy=False
        )


class ThresholdThenOpen:
    """Threshold to a binary mask, then binary-open it (removes speckle).

    Morphology needs SciPy, which is not a core dependency, so it is imported
    lazily and this class raises a clear error at construction if SciPy is absent.
    Runs on the expanded block, so opening sees halo context (keep the structuring
    radius within the halo to avoid re-introducing block-edge artifacts).
    """

    def __init__(self, thresh: float = 0.0, open_iters: int = 1):
        try:
            from scipy import ndimage
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "ThresholdThenOpen requires scipy; install scipy or use Threshold."
            ) from exc
        self._ndi = ndimage
        self.thresh = float(thresh)
        self.open_iters = int(open_iters)

    def __call__(self, block: np.ndarray, ctx: "BlockContext") -> np.ndarray:
        mask = block > self.thresh
        opened = self._ndi.binary_opening(mask, iterations=self.open_iters)
        return opened.astype(np.float32, copy=False)
