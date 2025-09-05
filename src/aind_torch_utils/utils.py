from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Tuple, Union

import tensorstore as ts


def ceil_div(a: int, b: int) -> int:
    """Ceiling division.

    Parameters
    ----------
    a : int
        Dividend.
    b : int
        Divisor (must be non-zero).

    Returns
    -------
    int
        Smallest integer >= a / b.

    Examples
    --------
    >>> ceil_div(10, 3)
    4
    >>> ceil_div(9, 3)
    3
    """
    return (a + b - 1) // b


def open_ts_spec(path_or_json: Union[str, Dict[str, Any]]) -> Any:
    """Open a TensorStore from a JSON spec.

    Parameters
    ----------
    path_or_json : str or dict
        Either a filesystem path to a JSON spec file or an in-memory
        dictionary representing a TensorStore spec.

    Returns
    -------
    Any
        A opened (read-ready) TensorStore object (future already resolved).

    Notes
    -----
    This is a light wrapper that loads JSON when a string path is provided
    and immediately resolves the TensorStore future with ``.result()``.

    Examples
    --------
    Open from a path::

        ts_obj = open_ts_spec('my_spec.json')

    Open from an in-memory dict::

        spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": "data.zarr"}}
        ts_obj = open_ts_spec(spec)
    """
    if isinstance(path_or_json, str):
        with open(path_or_json, "r", encoding="utf-8") as f:
            spec = json.load(f)
    else:
        spec = path_or_json
    return ts.open(spec).result()


def iter_blocks_zyx(
    full_zyx: Tuple[int, int, int], block: Tuple[int, int, int]
) -> Iterator[Tuple[Tuple[int, int, int], Tuple[slice, slice, slice]]]:
    """Iterate over equally sized (except edges) 3D blocks in Z, Y, X order.

    Parameters
    ----------
    full_zyx : tuple of int
        Full volume shape as (Z, Y, X).
    block : tuple of int
        Desired nominal block shape (bz, by, bx). Edge blocks may be smaller.

    Yields
    ------
    (tuple of int, tuple of slice)
        A tuple ``(index, bbox)`` where ``index`` is the (iz, iy, ix) block
        grid index and ``bbox`` is a tuple of slice objects usable for
        indexing a NumPy-like array: ``(slice_z, slice_y, slice_x)``.

    Examples
    --------
    >>> list(iter_blocks_zyx((5, 4, 3), (2, 2, 2)))  # doctest: +ELLIPSIS
    [((0, 0, 0), (slice(0, 2, None), slice(0, 2, None), slice(0, 2, None))), ...]
    """
    Z, Y, X = full_zyx
    bz, by, bx = block
    nz, ny, nx = ceil_div(Z, bz), ceil_div(Y, by), ceil_div(X, bx)
    for iz in range(nz):
        z0, z1 = iz * bz, min((iz + 1) * bz, Z)
        for iy in range(ny):
            y0, y1 = iy * by, min((iy + 1) * by, Y)
            for ix in range(nx):
                x0, x1 = ix * bx, min((ix + 1) * bx, X)
                idx = (iz, iy, ix)
                bbox = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
                yield idx, bbox


def iter_patch_starts(
    block_shape: Tuple[int, int, int],
    patch: Tuple[int, int, int],
    overlap: int,
) -> List[Tuple[int, int, int]]:
    """Compute patch start coordinates that guarantee full coverage.

    Produces starting (z, y, x) coordinates (inclusive) for sliding patches
    inside a block such that the final patch along an axis ends exactly at
    ``L - P`` when there is a remainder, ensuring coverage of the block to
    its boundary. Step size along each axis is ``max(P - overlap, 1)``.

    Parameters
    ----------
    block_shape : tuple of int
        Shape of the enclosing block (bz, by, bx).
    patch : tuple of int
        Patch size (pz, py, px).
    overlap : int
        Desired overlap in voxels along each axis (uniform). Must be < patch
        size to have effect; values >= patch size degrade to step size 1.

    Returns
    -------
    list of tuple of int
        List of starting coordinates (z, y, x).

    Examples
    --------
    >>> iter_patch_starts((10, 10, 10), (4, 4, 4), 1)[:3]
    [(0, 0, 0), (0, 0, 3), (0, 0, 6)]
    """
    bz, by, bx = block_shape
    pz, py, px = patch
    sz, sy, sx = (
        max(pz - overlap, 1),
        max(py - overlap, 1),
        max(px - overlap, 1),
    )

    def axis_starts(L: int, P: int, S: int) -> List[int]:
        if L <= P:
            return [0]
        starts = list(range(0, L - P + 1, S))
        last = L - P
        if starts[-1] != last:
            starts.append(last)
        return starts

    zs = axis_starts(bz, pz, sz)
    ys = axis_starts(by, py, sy)
    xs = axis_starts(bx, px, sx)
    return [(z, y, x) for z in zs for y in ys for x in xs]
