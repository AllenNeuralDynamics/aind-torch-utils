from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from aind_torch_utils.utils import ceil_div


BlockIndex = Tuple[int, int, int]
Slice3D = Tuple[slice, slice, slice]


@dataclass(frozen=True)
class ShardSpec:
    """
    Description of the spatial region assigned to a worker shard.

    Attributes
    ----------
    index : int
        Zero-based shard index.
    count : int
        Total number of shards.
    strategy : str
        Partitioning strategy identifier (e.g., 'contiguous-z', 'stride').
    block_start : Tuple[int, int, int]
        Inclusive start index of the block grid handled by this shard.
    block_stop : Tuple[int, int, int]
        Exclusive stop index of the block grid handled by this shard.
        tiles_per_axis : Tuple[int, int, int]
        How many shards the grid is split into along (z, y, x).
    """

    index: int
    count: int
    strategy: str
    block_start: BlockIndex
    block_stop: BlockIndex
    grid_shape: BlockIndex
    tile_index: BlockIndex
    tiles_per_axis: BlockIndex


def block_grid_shape(full_zyx: BlockIndex, block_shape: BlockIndex) -> BlockIndex:
    """
    Compute the number of blocks along each axis for the given volume.
    """
    return tuple(ceil_div(f, b) for f, b in zip(full_zyx, block_shape))


def _compute_tiles_per_axis(
    grid_shape: BlockIndex, shard_count: int, axis_order: Tuple[int, int, int]
) -> BlockIndex:
    """
    Determine how many shards to place along each axis.
    """
    tiles = [1, 1, 1]
    total_tiles = 1
    max_tiles = grid_shape
    total_blocks = math.prod(grid_shape)

    if shard_count > total_blocks:
        raise ValueError(
            f"shard_count ({shard_count}) cannot exceed total blocks ({total_blocks})."
        )

    while total_tiles < shard_count:
        progressed = False
        for axis in axis_order:
            if total_tiles >= shard_count:
                break
            if tiles[axis] < max_tiles[axis]:
                tiles[axis] += 1
                total_tiles = math.prod(tiles)
                progressed = True
            if total_tiles >= shard_count:
                break
        if not progressed:
            break

    if total_tiles < shard_count:
        raise ValueError(
            f"Unable to allocate {shard_count} shards across block grid {grid_shape}."
        )

    return tuple(tiles)  # type: ignore[return-value]


def _axis_bounds(length: int, parts: int, index: int) -> Tuple[int, int]:
    """
    Evenly partition a 1D length into `parts` segments and return the bounds for `index`.
    """
    base = length // parts
    remainder = length % parts
    start = index * base + min(index, remainder)
    size = base + (1 if index < remainder else 0)
    return start, start + size


def _tile_from_index(index: int, tiles_per_axis: BlockIndex) -> BlockIndex:
    """
    Convert a linear shard index into (z, y, x) tile coordinates.
    """
    tz, ty, tx = tiles_per_axis
    total_tiles = tz * ty * tx
    if index >= total_tiles:
        raise ValueError(
            f"Shard index {index} is out of range for tile grid {tiles_per_axis}"
        )
    yz = ty * tx
    tile_z = index // yz
    rem = index % yz
    tile_y = rem // tx if tx else 0
    tile_x = rem % tx if tx else 0
    return tile_z, tile_y, tile_x


def make_shard_spec(
    full_zyx: BlockIndex,
    block_shape: BlockIndex,
    shard_count: int,
    shard_index: int,
    strategy: str,
) -> ShardSpec:
    """
    Produce a ShardSpec describing the spatial region owned by a shard.
    """
    grid_shape = block_grid_shape(full_zyx, block_shape)
    if strategy == "stride":
        tiles_per_axis: BlockIndex = (1, 1, 1)
        tile_index: BlockIndex = (0, 0, 0)
        block_start: BlockIndex = (0, 0, 0)
        block_stop: BlockIndex = grid_shape
    elif strategy == "contiguous-z":
        tiles_per_axis = _compute_tiles_per_axis(grid_shape, shard_count, (0, 1, 2))
        tile_index = _tile_from_index(shard_index, tiles_per_axis)
        block_start = tuple(
            _axis_bounds(axis_len, tiles, idx)[0]
            for axis_len, tiles, idx in zip(grid_shape, tiles_per_axis, tile_index)
        )
        block_stop = tuple(
            _axis_bounds(axis_len, tiles, idx)[1]
            for axis_len, tiles, idx in zip(grid_shape, tiles_per_axis, tile_index)
        )
    else:
        raise ValueError(f"Unsupported shard_strategy '{strategy}'")

    return ShardSpec(
        index=shard_index,
        count=shard_count,
        strategy=strategy,
        block_start=block_start,
        block_stop=block_stop,
        grid_shape=grid_shape,
        tile_index=tile_index,
        tiles_per_axis=tiles_per_axis,
    )
