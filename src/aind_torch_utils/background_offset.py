"""Background-offset calibration for multiscale Zarr inputs."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def load_json_arg(value: str | dict[str, Any]) -> dict[str, Any]:
    """Load JSON from an inline value, file path, or already-loaded dict."""
    if isinstance(value, dict):
        return copy.deepcopy(value)
    stripped = value.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        data = json.loads(value)
    else:
        with open(value, "r", encoding="utf-8") as stream:
            data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    return data


def replace_final_path_component(path: str, replacement: str) -> str:
    """Replace the final slash-delimited component of a path or URL."""
    replacement = str(replacement).strip("/")
    if not replacement:
        raise ValueError("Replacement path component must not be empty.")
    stripped = path.rstrip("/")
    if not stripped:
        raise ValueError("Cannot replace the final component of an empty path.")
    head, separator, _tail = stripped.rpartition("/")
    return f"{head}/{replacement}" if separator else replacement


def replace_kvstore_resolution(kvstore: Any, resolution: str) -> Any:
    """Copy a TensorStore kvstore with its final path set to a resolution."""
    if isinstance(kvstore, str):
        return replace_final_path_component(kvstore, resolution)
    if not isinstance(kvstore, dict):
        raise TypeError(f"Unsupported kvstore type: {type(kvstore).__name__}")
    rewritten = copy.deepcopy(kvstore)
    if isinstance(rewritten.get("path"), str):
        rewritten["path"] = replace_final_path_component(
            rewritten["path"], resolution
        )
        return rewritten
    if isinstance(rewritten.get("url"), str):
        rewritten["url"] = replace_final_path_component(
            rewritten["url"], resolution
        )
        return rewritten
    if "base" in rewritten:
        rewritten["base"] = replace_kvstore_resolution(
            rewritten["base"], resolution
        )
        return rewritten
    raise ValueError(
        "Cannot identify the path component in kvstore; expected a string, "
        "a dict with 'path' or 'url', or a dict with 'base'."
    )


def spec_with_stats_resolution(
    input_spec: dict[str, Any], stats_resolution: str
) -> dict[str, Any]:
    """Copy an input TensorStore spec and select the statistics resolution."""
    stats_spec = copy.deepcopy(input_spec)
    if stats_spec.get("driver") not in {"zarr", "zarr2", "zarr3"}:
        raise ValueError(
            f"Expected a Zarr TensorStore driver, got {stats_spec.get('driver')!r}."
        )
    if "kvstore" not in stats_spec:
        raise ValueError("Input TensorStore spec is missing 'kvstore'.")
    stats_spec["kvstore"] = replace_kvstore_resolution(
        stats_spec["kvstore"], str(stats_resolution)
    )
    return stats_spec


def _file_url_to_path(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return url
    if parsed.netloc and parsed.netloc not in {"localhost", ""}:
        return f"//{parsed.netloc}{unquote(parsed.path)}"
    return unquote(parsed.path)


def _join_url_or_path(base: str, child: str) -> str:
    child = str(child).strip("/")
    return f"{base.rstrip('/')}/{child}" if child else base


def kvstore_to_zarr_url(kvstore: Any) -> str:
    """Convert common TensorStore kvstores into a Dask Zarr URL/path."""
    if isinstance(kvstore, str):
        return _file_url_to_path(kvstore)
    if not isinstance(kvstore, dict):
        raise TypeError(f"Unsupported kvstore type: {type(kvstore).__name__}")
    if "base" in kvstore:
        return _join_url_or_path(
            kvstore_to_zarr_url(kvstore["base"]), kvstore.get("path", "")
        )
    driver = kvstore.get("driver")
    if driver == "file":
        path = kvstore.get("path")
        if not isinstance(path, str):
            raise ValueError("File kvstore is missing string field 'path'.")
        return path
    if driver == "s3":
        bucket = kvstore.get("bucket") or kvstore.get("bucket_name")
        if not bucket:
            raise ValueError("S3 kvstore is missing field 'bucket'.")
        path = str(kvstore.get("path", "")).strip("/")
        return f"s3://{bucket}/{path}" if path else f"s3://{bucket}"
    if driver in {"gcs", "google_cloud_storage"}:
        bucket = kvstore.get("bucket") or kvstore.get("bucket_name")
        if not bucket:
            raise ValueError("GCS kvstore is missing field 'bucket'.")
        path = str(kvstore.get("path", "")).strip("/")
        return f"gs://{bucket}/{path}" if path else f"gs://{bucket}"
    if isinstance(kvstore.get("url"), str):
        return _file_url_to_path(kvstore["url"])
    if isinstance(kvstore.get("path"), str):
        return kvstore["path"]
    raise ValueError(f"Unsupported kvstore driver: {driver!r}")


def zarr_location_from_ts_spec(spec: dict[str, Any]) -> tuple[str, str | None]:
    """Return a Dask Zarr URL and optional component from a TensorStore spec."""
    url = kvstore_to_zarr_url(spec["kvstore"])
    component = spec.get("path")
    if component is not None and not isinstance(component, str):
        raise ValueError("TensorStore spec field 'path' must be a string.")
    return url, component


def _import_dask_array():
    try:
        import dask.array as da
    except ImportError as exc:
        raise RuntimeError(
            "Dask array support is required; install aind-torch-utils[beaker]."
        ) from exc
    return da


def _open_dask_client(n_workers: int, threads_per_worker: int, processes: bool):
    try:
        from distributed import Client, LocalCluster
    except ImportError as exc:
        raise RuntimeError(
            "dask.distributed is required; install aind-torch-utils[beaker]."
        ) from exc
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        dashboard_address=None,
    )
    return Client(cluster), cluster


def _percentile_tdigest(sample: Any, percentile: float) -> float:
    da = _import_dask_array()
    try:
        result = da.percentile(
            sample,
            [percentile],
            method="linear",
            internal_method="tdigest",
        ).compute()
    except ImportError as exc:
        raise RuntimeError(
            "Dask t-digest percentiles require the crick package."
        ) from exc
    return float(result[0])


def _validate_index(name: str, index: int, size: int) -> None:
    if index < 0 or index >= size:
        raise IndexError(f"{name}={index} is out of bounds for axis size {size}.")


def compute_background_offset(
    array: Any,
    t_idx: int,
    c_idx: int,
    percentile: float,
    ignore_zeros: bool = True,
) -> float:
    """Estimate the background from a low percentile of one T,C volume."""
    if array.ndim != 5:
        raise ValueError(f"Expected a 5D T,C,Z,Y,X array; got shape {array.shape}.")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"Offset percentile must be in [0, 100]; got {percentile}.")
    _validate_index("t_idx", t_idx, array.shape[0])
    _validate_index("c_idx", c_idx, array.shape[1])
    volume = array[t_idx, c_idx, :, :, :].ravel()
    sample = volume[volume > 0] if ignore_zeros else volume
    offset = _percentile_tdigest(sample, percentile)
    if ignore_zeros and not math.isfinite(offset):
        offset = _percentile_tdigest(volume, percentile)
    if not math.isfinite(offset):
        raise ValueError(f"Computed non-finite background offset: {offset}.")
    return offset


def build_workflow_params(checkpoint_path: str, offset: float) -> dict[str, Any]:
    """Build validated denoise-net workflow parameters."""
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError("checkpoint_path must be a non-empty string.")
    offset = float(offset)
    if not math.isfinite(offset):
        raise ValueError("offset must be a finite number.")
    return {"checkpoint_path": checkpoint_path, "offset": offset}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with deterministic human-readable formatting."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run_prepass(
    *,
    in_spec: str | dict[str, Any],
    checkpoint_path: str,
    stats_resolution: str,
    offset_percentile: float,
    config: str | dict[str, Any] | None = None,
    ignore_zeros: bool = True,
    workers: int | None = None,
    threads_per_worker: int = 1,
    processes: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute an offset and return workflow parameters plus calibration stats."""
    da = _import_dask_array()
    input_spec = load_json_arg(in_spec)
    config_data = load_json_arg(config) if config is not None else {}
    stats_spec = spec_with_stats_resolution(input_spec, str(stats_resolution))
    zarr_url, component = zarr_location_from_ts_spec(stats_spec)
    n_workers = int(workers or min(16, os.cpu_count() or 1))
    if n_workers < 1:
        raise ValueError("workers must be at least 1.")
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be at least 1.")
    t_idx = int(config_data.get("t_idx", 0))
    c_idx = int(config_data.get("c_idx", 0))

    started = time.perf_counter()
    client, cluster = _open_dask_client(n_workers, threads_per_worker, processes)
    try:
        array = da.from_zarr(zarr_url, component=component)
        offset = compute_background_offset(
            array, t_idx, c_idx, offset_percentile, ignore_zeros
        )
        shape = list(array.shape)
        chunks = [list(axis_chunks) for axis_chunks in array.chunks]
        dtype = str(array.dtype)
    finally:
        client.close()
        cluster.close()

    workflow_params = build_workflow_params(checkpoint_path, offset)
    stats = {
        "stats_resolution": str(stats_resolution),
        "zarr_url": zarr_url,
        "component": component,
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "t_idx": t_idx,
        "c_idx": c_idx,
        "offset_percentile": float(offset_percentile),
        "ignore_zeros": bool(ignore_zeros),
        "percentile_method": "tdigest",
        "workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "processes": processes,
        "offset": offset,
        "checkpoint_path": checkpoint_path,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return workflow_params, stats
