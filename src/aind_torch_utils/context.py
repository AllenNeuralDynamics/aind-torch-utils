"""Per-block context passed to injected pre/post-processing hooks.

A :class:`BlockContext` describes *where* a block lives in the full volume. It is the
information an injected ``BlockPreprocessor`` / ``BlockPostProcessor`` needs but the
runtime carriers (:class:`~aind_torch_utils.workers.Batch` /
:class:`~aind_torch_utils.workers.Preds`) do not otherwise expose in one place.

The **absolute** ``expanded_bbox`` is load-bearing: corrections that sample a coarse
*global* field (flat-field background, adaptive local statistics) must index that
field by absolute coordinate so two blocks that overlap via the halo agree exactly on
shared voxels -> the correction is seam-free. Any hook that does this needs the
absolute expanded bounding box, which is why it is a first-class field here.

This module is intentionally dependency-light (dataclass + typing only) so it can be
imported by the workers, the (future) transform objects, and unit tests without pulling
in torch / tensorstore.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from aind_torch_utils.workers import Preds


@dataclass(frozen=True, slots=True)
class BlockContext:
    """Absolute placement of one processing block within the full volume.

    Attributes
    ----------
    block_idx : tuple of int
        Block grid index ``(iz, iy, ix)``.
    core_bbox : tuple of slice
        Absolute ``(z, y, x)`` slices of the block's *core* region (the part that
        is written out).
    expanded_bbox : tuple of slice
        Absolute ``(z, y, x)`` slices of the block's *core + halo* region (the part
        that is read and processed). Sampling global fields by these absolute
        coordinates is what keeps halo-overlapping blocks seam-free.
    halo_left : tuple of int
        Halo width actually added on the ``-z, -y, -x`` sides (0 at volume borders).
    full_shape : tuple of int
        Full volume spatial shape ``(Z, Y, X)``.
    t_idx, c_idx : int
        Time / channel indices being processed.
    """

    block_idx: Tuple[int, int, int]
    core_bbox: Tuple[slice, slice, slice]
    expanded_bbox: Tuple[slice, slice, slice]
    halo_left: Tuple[int, int, int]
    full_shape: Tuple[int, int, int]
    t_idx: int
    c_idx: int

    @classmethod
    def from_block(
        cls,
        block_idx: Tuple[int, int, int],
        core_bbox: Tuple[slice, slice, slice],
        expanded_bbox: Tuple[slice, slice, slice],
        full_shape: Tuple[int, int, int],
        t_idx: int,
        c_idx: int,
    ) -> "BlockContext":
        """Build a context from the prep stage, deriving ``halo_left``.

        ``PrepWorker`` already computes both the core and expanded bounding boxes, so
        the left-halo widths are just their per-axis start offsets.
        """
        (zsl, ysl, xsl), (ezsl, eysl, exsl) = core_bbox, expanded_bbox
        halo_left = (
            zsl.start - ezsl.start,
            ysl.start - eysl.start,
            xsl.start - exsl.start,
        )
        return cls(
            block_idx=block_idx,
            core_bbox=core_bbox,
            expanded_bbox=expanded_bbox,
            halo_left=halo_left,
            full_shape=full_shape,
            t_idx=t_idx,
            c_idx=c_idx,
        )

    @classmethod
    def from_preds(
        cls,
        preds: "Preds",
        full_shape: Tuple[int, int, int],
        t_idx: int,
        c_idx: int,
    ) -> "BlockContext":
        """Reconstruct the context in the writer stage from a :class:`Preds`.

        ``Preds`` carries the core bbox, the left-halo widths, and the expanded
        ``acc_shape`` but not the expanded bbox itself; recover it as
        ``expanded_start = core_start - halo_left`` and
        ``expanded_stop = expanded_start + acc_shape``. ``full_shape`` is a
        volume-level constant supplied by the caller (it does not ride on every
        per-block carrier).
        """
        (zsl, ysl, xsl) = preds.block_bbox
        lz, ly, lx = preds.halo_left
        bz, by, bx = preds.acc_shape
        z0e, y0e, x0e = zsl.start - lz, ysl.start - ly, xsl.start - lx
        expanded_bbox = (
            slice(z0e, z0e + bz),
            slice(y0e, y0e + by),
            slice(x0e, x0e + bx),
        )
        return cls(
            block_idx=preds.block_idx,
            core_bbox=preds.block_bbox,
            expanded_bbox=expanded_bbox,
            halo_left=preds.halo_left,
            full_shape=full_shape,
            t_idx=t_idx,
            c_idx=c_idx,
        )
