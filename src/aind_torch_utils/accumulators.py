"""Block accumulators: how overlapping patches merge into one block.

The correct merge depends on what the output *means*. Averaging is right for
continuous intensities but wrong for masks/labels: a binary mask averaged at a seam
yields 0.5, which floors to 0 (an eroded mask at every patch boundary). So the merge
strategy is a first-class, injectable choice (issue #25 §3.3), supplied as a
**factory** because one accumulator is created per block per output.

All accumulators expose the same tiny contract (:class:`BlockAccumulator`): ``add`` a
patch, then ``finalize`` to the **expanded** (core + halo) block; ``count``/``total``
let the writer detect block completion.
"""
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aind_torch_utils.context import BlockContext


@runtime_checkable
class BlockAccumulator(Protocol):
    """Merges patches for one block/output. Created once per block per output."""

    total: int
    count: int

    def add(
        self,
        pred_patch: np.ndarray,
        start: Tuple[int, int, int],
        valid: Tuple[int, int, int],
    ) -> None:
        """Merge one patch at ``start`` with ``valid`` (dz, dy, dx) extent."""
        ...

    def finalize(self) -> np.ndarray:
        """Return the merged **expanded** (core + halo) block."""
        ...


class BlockAccumulatorFactory(Protocol):
    """Builds a fresh :class:`BlockAccumulator` for a block of ``shape``."""

    def __call__(
        self, shape: Tuple[int, int, int], ctx: "BlockContext"
    ) -> BlockAccumulator:
        """Create an accumulator sized to the expanded block ``shape``."""
        ...


class WeightedAverageAccumulator:
    """Accumulates predicted patches into a single block, handling seams.

    This is the historical default merge: ``trim`` (last-write on cropped margins)
    or ``blend`` (edge-aware weighted average). Correct for continuous intensities.
    """

    def __init__(
        self,
        block_shape: Tuple[int, int, int],
        eps: float,
        overlap: int,
        seam_mode: str,
        trim_voxels: Optional[int],
        min_blend_weight: float,
    ):
        """
        Initialize the accumulator.

        Parameters
        ----------
        block_shape : Tuple[int, int, int]
            The shape of the block to accumulate.
        eps : float
            A small value to avoid division by zero.
        overlap : int
            The overlap between patches.
        seam_mode : str
            The seam handling mode, either 'trim' or 'blend'.
        trim_voxels : Optional[int]
            The number of voxels to trim from each patch edge.
        min_blend_weight : float
            The minimum weight for blending.
        """
        self.block_shape = block_shape
        self.acc = np.zeros(block_shape, dtype=np.float32)
        self.wacc = np.zeros(block_shape, dtype=np.float32)
        self.eps = eps
        self.count = 0
        self.total = 0
        self.overlap = overlap
        self.seam_mode = seam_mode
        self.trim_voxels = (
            trim_voxels if trim_voxels is not None else max(overlap // 2, 0)
        )
        self.min_blend_weight = float(min_blend_weight)

    def _edge_aware_weights(
        self, start: Tuple[int, int, int], valid: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Build per-patch weights that taper only on sides that actually overlap
        within the current block. On sides that touch the block border, weight=1.
        Avoid strictly zero weights by flooring at min_blend_weight.

        Parameters
        ----------
        start : Tuple[int, int, int]
            The starting coordinates of the patch within the block.
        valid : Tuple[int, int, int]
            The valid dimensions of the patch.

        Returns
        -------
        np.ndarray
            The weights for the patch.
        """
        bz, by, bx = self.block_shape
        sz, sy, sx = start
        dz, dy, dx = valid
        o = self.overlap

        def _axis_weights(L_block, s, v):
            w = np.ones(v, dtype=np.float32)
            # left overlap exists if s > 0
            left = min(o, s)
            if left > 0:
                t = np.arange(left, dtype=np.float32)
                ramp = 0.5 - 0.5 * np.cos(np.pi * (t + 1) / (left + 1))  # (0,1)
                w[:left] *= ramp
            # right overlap exists if s + v < L_block
            right = min(o, L_block - (s + v))
            if right > 0:
                t = np.arange(right, dtype=np.float32)
                ramp = 0.5 - 0.5 * np.cos(np.pi * (t + 1) / (right + 1))
                w[v - right :] *= ramp[::-1]
            np.maximum(w, self.min_blend_weight, out=w)
            return w

        wz = _axis_weights(bz, sz, dz)
        wy = _axis_weights(by, sy, dy)
        wx = _axis_weights(bx, sx, dx)
        W = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
        return W.astype(np.float32, copy=False)

    def _trim_slices(
        self, start: Tuple[int, int, int], valid: Tuple[int, int, int]
    ) -> Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice]]:
        """
        Calculate the slices for trimming the patch and the corresponding
        block region.

        Parameters
        ----------
        start : Tuple[int, int, int]
            The starting coordinates of the patch within the block.
        valid : Tuple[int, int, int]
            The valid dimensions of the patch.

        Returns
        -------
        Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice]]
            A tuple containing the patch slices and block slices.
        """
        sz, sy, sx = start
        dz, dy, dx = valid
        t = int(self.trim_voxels)
        bz, by, bx = self.block_shape

        # Do not trim on the left if the patch starts at the block boundary
        lz = 0 if sz == 0 else min(t, dz)
        ly = 0 if sy == 0 else min(t, dy)
        lx = 0 if sx == 0 else min(t, dx)

        # Do not trim on the right if the patch ends at the block boundary
        rz = 0 if sz + dz == bz else min(t, dz - lz)
        ry = 0 if sy + dy == by else min(t, dy - ly)
        rx = 0 if sx + dx == bx else min(t, dx - lx)

        patch_sl = (slice(lz, dz - rz), slice(ly, dy - ry), slice(lx, dx - rx))
        block_sl = (
            slice(sz + lz, sz + dz - rz),
            slice(sy + ly, sy + dy - ry),
            slice(sx + lx, sx + dx - rx),
        )
        return patch_sl, block_sl

    def add(
        self,
        pred_patch: np.ndarray,
        start: Tuple[int, int, int],
        valid: Tuple[int, int, int],
    ) -> None:
        """
        Add a predicted patch to the accumulator.

        Parameters
        ----------
        pred_patch : np.ndarray
            The predicted patch.
        start : Tuple[int, int, int]
            The starting coordinates of the patch within the block.
        valid : Tuple[int, int, int]
            The valid dimensions of the patch.
        """
        if self.seam_mode == "trim":
            patch_sl, block_sl = self._trim_slices(start, valid)
            # last-write-wins
            self.acc[block_sl] = np.asarray(pred_patch[patch_sl], dtype=np.float32)
            self.wacc[block_sl] = 1.0
        else:
            sz, sy, sx = start
            dz, dy, dx = valid
            # edge-aware blending
            W = self._edge_aware_weights(start, valid).astype(np.float32, copy=False)
            pp = np.asarray(pred_patch, dtype=np.float32, copy=False)
            self.acc[sz : sz + dz, sy : sy + dy, sx : sx + dx] += pp[:dz, :dy, :dx] * W
            self.wacc[sz : sz + dz, sy : sy + dy, sx : sx + dx] += W

        self.count += 1

    def finalize(self) -> np.ndarray:
        out = self.acc / np.maximum(self.wacc, self.eps)
        return out


class _RegionAccumulator:
    """Shared plumbing for accumulators that write into a single block buffer."""

    def __init__(self, block_shape: Tuple[int, int, int], fill: float = 0.0):
        self.block_shape = block_shape
        self.acc = np.full(block_shape, fill, dtype=np.float32)
        self.count = 0
        self.total = 0

    def _region(self, start, valid):
        sz, sy, sx = start
        dz, dy, dx = valid
        return (
            (slice(sz, sz + dz), slice(sy, sy + dy), slice(sx, sx + dx)),
            (slice(0, dz), slice(0, dy), slice(0, dx)),
        )

    def finalize(self) -> np.ndarray:
        return self.acc


class LastWriteAccumulator(_RegionAccumulator):
    """Overwrite each patch's full region; the last patch to cover a voxel wins.

    Unlike ``trim``, no margins are cropped -- useful when patches are already
    disjoint or a deterministic write order is acceptable.
    """

    def add(self, pred_patch, start, valid):
        block_sl, patch_sl = self._region(start, valid)
        self.acc[block_sl] = np.asarray(pred_patch[patch_sl], dtype=np.float32)
        self.count += 1


class MaxAccumulator(_RegionAccumulator):
    """Element-wise maximum across overlapping patches (order-independent).

    Good for foreground masks/logits: any patch that calls a voxel foreground wins.
    Voxels no patch ever covers finalize to 0.
    """

    def __init__(self, block_shape: Tuple[int, int, int]):
        super().__init__(block_shape, fill=-np.inf)

    def add(self, pred_patch, start, valid):
        block_sl, patch_sl = self._region(start, valid)
        region = self.acc[block_sl]
        np.maximum(
            region, np.asarray(pred_patch[patch_sl], dtype=np.float32), out=region
        )
        self.count += 1

    def finalize(self) -> np.ndarray:
        self.acc[np.isneginf(self.acc)] = 0.0
        return self.acc


class SumAccumulator(_RegionAccumulator):
    """Sum overlapping contributions (e.g. vote counts, densities)."""

    def add(self, pred_patch, start, valid):
        block_sl, patch_sl = self._region(start, valid)
        self.acc[block_sl] += np.asarray(pred_patch[patch_sl], dtype=np.float32)
        self.count += 1


class MajorityVoteAccumulator:
    """Per-voxel majority vote across overlapping patches (labels / instance IDs).

    Averaging discrete labels is meaningless; this counts votes per label and
    finalizes to the most-voted label (ties broken toward the smaller label).
    Memory scales with the number of distinct labels seen in the block.
    """

    def __init__(self, block_shape: Tuple[int, int, int]):
        self.block_shape = block_shape
        self.votes: Dict[Any, np.ndarray] = {}
        self.count = 0
        self.total = 0

    def add(self, pred_patch, start, valid):
        sz, sy, sx = start
        dz, dy, dx = valid
        patch = np.asarray(pred_patch[:dz, :dy, :dx])
        region = (slice(sz, sz + dz), slice(sy, sy + dy), slice(sx, sx + dx))
        for label in np.unique(patch):
            arr = self.votes.get(label)
            if arr is None:
                arr = np.zeros(self.block_shape, dtype=np.float32)
                self.votes[label] = arr
            arr[region] += patch == label
        self.count += 1

    def finalize(self) -> np.ndarray:
        if not self.votes:
            return np.zeros(self.block_shape, dtype=np.float32)
        labels = sorted(self.votes.keys())
        stack = np.stack([self.votes[label] for label in labels], axis=0)
        winner = np.asarray(labels)[np.argmax(stack, axis=0)]
        return winner.astype(np.float32)


def weighted_average_factory(
    eps: float,
    overlap: int,
    seam_mode: str,
    trim_voxels: Optional[int],
    min_blend_weight: float,
) -> BlockAccumulatorFactory:
    """Return the default factory: a :class:`WeightedAverageAccumulator` per block.

    Reproduces the historical trim/blend merge from the runtime config, so a run
    that does not inject a factory behaves exactly as before.
    """

    def factory(
        shape: Tuple[int, int, int], ctx: "BlockContext"
    ) -> BlockAccumulator:
        return WeightedAverageAccumulator(
            shape,
            eps,
            overlap=overlap,
            seam_mode=seam_mode,
            trim_voxels=trim_voxels,
            min_blend_weight=min_blend_weight,
        )

    return factory
