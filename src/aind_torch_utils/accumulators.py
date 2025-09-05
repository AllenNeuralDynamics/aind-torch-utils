from typing import Optional, Tuple

import numpy as np


class BlockAccumulator:
    """
    Accumulates predicted patches into a single block, handling seams.
    """

    def __init__(
        self,
        block_shape: Tuple[int, int, int],
        patch: Tuple[int, int, int],
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
        patch : Tuple[int, int, int]
            The shape of the patches being added.
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
        self.patch = patch
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
