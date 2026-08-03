"""Durable, per-block completion state for resumable inference runs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Set, Tuple
from urllib.parse import urlparse

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.distributed.sharding import ShardSpec

logger = logging.getLogger(__name__)

_RUN_ID_VERSION = 2
_TENSORSTORE_LIFECYCLE_KEYS = {"create", "delete_existing", "open"}
_OUTPUT_AFFECTING_CONFIG_FIELDS = (
    "patch",
    "overlap",
    "block",
    "t_idx",
    "c_idx",
    "amp",
    "use_tf32",
    "cudnn_benchmark",
    "use_compile",
    "compile_mode",
    "compile_dynamic",
    "seam_mode",
    "trim_voxels",
    "halo",
    "min_blend_weight",
    "output_denormalize",
    "eps",
    "norm_lower",
    "norm_upper",
    "normalize",
    "clip_norm",
)


@dataclass(frozen=True, slots=True)
class BlockKey:
    """Stable identifier for a single output block."""

    t: int
    c: int
    z: int
    y: int
    x: int
    linear_k: int

    @property
    def coords(self) -> Tuple[int, int, int, int, int]:
        """Return the time, channel, and spatial grid coordinates."""
        return self.t, self.c, self.z, self.y, self.x


@dataclass(frozen=True, slots=True)
class BlockLease:
    """Claim returned by a work store for a block this worker may process."""

    block: BlockKey
    token: Optional[str] = None


class BlockWorkStore(Protocol):
    """Backend-independent block completion interface."""

    def prepare(self, shard_spec: ShardSpec) -> None:
        """Load backend state before worker threads start."""
        ...

    def claim_block(self, block: BlockKey) -> Optional[BlockLease]:
        """Return a lease, or ``None`` when the block is already complete."""
        ...

    def complete_block(self, lease: BlockLease) -> None:
        """Mark a block complete after every output write commits."""
        ...

    def fail_block(self, lease: BlockLease, exc: BaseException) -> None:
        """Record a block failure if the backend supports it."""
        ...


class NoopBlockWorkStore:
    """Existing non-resumable behavior: every block is processed."""

    def prepare(self, shard_spec: ShardSpec) -> None:
        """Do nothing."""
        return None

    def claim_block(self, block: BlockKey) -> BlockLease:
        """Always grant a lease."""
        return BlockLease(block)

    def complete_block(self, lease: BlockLease) -> None:
        """Do nothing."""
        return None

    def fail_block(self, lease: BlockLease, exc: BaseException) -> None:
        """Do nothing."""
        return None


class S3MarkerBlockWorkStore:
    """Use sidecar S3 marker objects as durable completion state."""

    _KEY_RE = re.compile(r"/z=(?P<z>\d+)/y=(?P<y>\d+)/x=(?P<x>\d+)\.done$")

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        t_idx: int,
        c_idx: int,
        run_id: str,
        s3_client: Optional[Any] = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = _strip_slashes(prefix)
        self.t_idx = int(t_idx)
        self.c_idx = int(c_idx)
        self.run_id = run_id
        self._completed: Set[Tuple[int, int, int, int, int]] = set()
        self._lock = threading.Lock()
        self._s3_client = s3_client

    @property
    def marker_prefix(self) -> str:
        """Return the S3 key prefix for this time/channel pair."""
        return f"{self.prefix}/t={self.t_idx}/c={self.c_idx}/"

    def prepare(self, shard_spec: ShardSpec) -> None:
        """List existing markers once and cache completed coordinates."""
        del shard_spec  # marker identity is independent of shard topology
        paginator = self._client().get_paginator("list_objects_v2")
        loaded = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.marker_prefix):
            for obj in page.get("Contents", []):
                coords = self._coords_from_key(obj.get("Key", ""))
                if coords is None:
                    continue
                with self._lock:
                    self._completed.add(coords)
                loaded += 1
        logger.info(
            "Loaded %d completed block markers from s3://%s/%s",
            loaded,
            self.bucket,
            self.marker_prefix,
        )

    def claim_block(self, block: BlockKey) -> Optional[BlockLease]:
        """Skip blocks represented by an existing completion marker."""
        with self._lock:
            if block.coords in self._completed:
                return None
        return BlockLease(block)

    def complete_block(self, lease: BlockLease) -> None:
        """Write a completion marker after the block's outputs commit."""
        block = lease.block
        body = json.dumps(
            {
                "schema_version": _RUN_ID_VERSION,
                "run_id": self.run_id,
                "t": block.t,
                "c": block.c,
                "z": block.z,
                "y": block.y,
                "x": block.x,
                "linear_k": block.linear_k,
                "completed_at_unix": time.time(),
            },
            sort_keys=True,
        ).encode("utf-8")
        self._client().put_object(
            Bucket=self.bucket,
            Key=self._key_for_block(block),
            Body=body,
            ContentType="application/json",
        )
        with self._lock:
            self._completed.add(block.coords)

    def fail_block(self, lease: BlockLease, exc: BaseException) -> None:
        """Log a failed claim without creating durable completion state."""
        logger.warning(
            "Block %s failed before its completion marker was written: %s",
            lease.block,
            exc,
        )

    def _client(self) -> Any:
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3  # type: ignore
        except ImportError as exc:  # pragma: no cover - integration-only path
            raise RuntimeError(
                "S3 marker resume requires boto3. Install with "
                "`pip install aind-torch-utils[aws]`."
            ) from exc
        self._s3_client = boto3.client("s3")
        return self._s3_client

    def _key_for_block(self, block: BlockKey) -> str:
        return (
            f"{self.prefix}/t={block.t}/c={block.c}/z={block.z}/"
            f"y={block.y}/x={block.x}.done"
        )

    def _coords_from_key(self, key: str) -> Optional[Tuple[int, int, int, int, int]]:
        match = self._KEY_RE.search(key)
        if match is None:
            return None
        return (
            self.t_idx,
            self.c_idx,
            int(match.group("z")),
            int(match.group("y")),
            int(match.group("x")),
        )


def build_block_work_store(
    *,
    cfg: InferenceConfig,
    input_store: Any,
    input_spec: Optional[Mapping[str, Any]] = None,
    output_specs: Optional[Sequence[Mapping[str, Any]]] = None,
    output_stores: Optional[Sequence[Any]] = None,
    workload: Optional[Mapping[str, Any]] = None,
    output_spec: Optional[Mapping[str, Any]] = None,
    output_store: Optional[Any] = None,
    model_type: Optional[str] = None,
    weights_path: Optional[str] = None,
) -> BlockWorkStore:
    """Build a store from current or legacy orchestration metadata.

    ``output_spec``/``output_store`` and the model fields retain the original
    single-output API from ``feat-resumability``. New orchestration should pass
    the plural output arguments plus a canonical ``workload`` descriptor.
    """
    if not cfg.resume:
        return NoopBlockWorkStore()
    if output_specs is not None and output_spec is not None:
        raise ValueError("Pass output_specs or output_spec, not both")
    if output_stores is not None and output_store is not None:
        raise ValueError("Pass output_stores or output_store, not both")

    resolved_specs = list(
        output_specs or ([] if output_spec is None else [output_spec])
    )
    resolved_stores = list(
        output_stores or ([] if output_store is None else [output_store])
    )
    if len(resolved_specs) != len(resolved_stores):
        raise ValueError(
            "Resume requires one opened output store per output TensorStore spec"
        )
    resolved_workload = dict(
        workload
        or {
            "kind": "model",
            "model_type": model_type,
            "weights_path": weights_path,
        }
    )

    validate_resume_output_specs(cfg, resolved_specs)
    if cfg.work_store != "s3-markers":
        raise ValueError(f"Unsupported work_store backend: {cfg.work_store}")

    bucket, base_prefix = _resolve_marker_location(cfg, resolved_specs)
    run_id = cfg.resume_run_id or derive_run_id(
        cfg=cfg,
        input_spec=input_spec or {},
        output_specs=resolved_specs,
        input_store=input_store,
        output_stores=resolved_stores,
        workload=resolved_workload,
    )
    marker_prefix = _join_s3_key(
        base_prefix,
        ".aind_torch_utils",
        "resume",
        f"v{_RUN_ID_VERSION}",
        run_id,
    )
    logger.info("Using S3 resume markers at s3://%s/%s", bucket, marker_prefix)
    return S3MarkerBlockWorkStore(
        bucket=bucket,
        prefix=marker_prefix,
        t_idx=cfg.t_idx,
        c_idx=cfg.c_idx,
        run_id=run_id,
    )


def validate_resume_output_specs(
    cfg: InferenceConfig, output_specs: Sequence[Mapping[str, Any]]
) -> None:
    """Reject destructive or ambiguous output specs before opening them."""
    if not cfg.resume:
        return
    if not output_specs:
        raise ValueError("Resume requires at least one output TensorStore spec")
    for index, spec in enumerate(output_specs):
        if not isinstance(spec, Mapping):
            raise ValueError(f"Resume requires output spec {index} to be a JSON object")
        if bool(spec.get("delete_existing", False)):
            raise ValueError(
                "Resume cannot be used with an output spec that has "
                "`delete_existing: true`; that would delete existing output."
            )
    if len(output_specs) > 1 and cfg.resume_marker_prefix is None:
        raise ValueError(
            "Multi-output resume requires an explicit resume_marker_prefix"
        )


def validate_resume_output_spec(
    cfg: InferenceConfig, output_spec: Mapping[str, Any]
) -> None:
    """Validate one output spec using the original resumability API."""
    validate_resume_output_specs(cfg, [output_spec])


def derive_run_id(
    *,
    cfg: InferenceConfig,
    input_spec: Mapping[str, Any],
    output_specs: Sequence[Mapping[str, Any]],
    input_store: Any,
    output_stores: Sequence[Any],
    workload: Mapping[str, Any],
) -> str:
    """Derive a deterministic identity for output-equivalent retries."""
    fields = {
        "schema_version": _RUN_ID_VERSION,
        "input_spec": _canonical_store_spec(input_spec),
        "output_specs": [_canonical_store_spec(spec) for spec in output_specs],
        "input_store": _store_signature(input_store),
        "output_stores": [_store_signature(store) for store in output_stores],
        "workload": dict(workload),
        "config": {
            name: getattr(cfg, name) for name in _OUTPUT_AFFECTING_CONFIG_FIELDS
        },
    }
    encoded = json.dumps(fields, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _canonical_store_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in dict(spec).items()
        if key not in _TENSORSTORE_LIFECYCLE_KEYS
    }


def _store_signature(store: Any) -> Dict[str, Any]:
    domain = getattr(store, "domain", None)
    shape = getattr(domain, "shape", getattr(store, "shape", None))
    dtype = getattr(getattr(store, "dtype", None), "numpy_dtype", None)
    return {
        "shape": tuple(shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
    }


def _resolve_marker_location(
    cfg: InferenceConfig, output_specs: Sequence[Mapping[str, Any]]
) -> Tuple[str, str]:
    if cfg.resume_marker_prefix:
        return _parse_s3_uri(cfg.resume_marker_prefix)
    return _extract_s3_kvstore_root(output_specs[0])


def _extract_s3_kvstore_root(spec: Mapping[str, Any]) -> Tuple[str, str]:
    kvstore = spec.get("kvstore")
    if isinstance(kvstore, str) and kvstore.startswith("s3://"):
        return _parse_s3_uri(kvstore)
    if isinstance(kvstore, Mapping) and kvstore.get("driver") == "s3":
        bucket = kvstore.get("bucket")
        if not bucket:
            raise ValueError("S3 output kvstore is missing `bucket`")
        return str(bucket), _strip_slashes(str(kvstore.get("path", "")))
    raise ValueError(
        "S3 marker resume requires an S3 output kvstore or "
        "`resume_marker_prefix='s3://bucket/prefix'`."
    )


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/prefix URI, got {uri!r}")
    return parsed.netloc, _strip_slashes(parsed.path)


def _join_s3_key(*parts: str) -> str:
    return "/".join(_strip_slashes(part) for part in parts if _strip_slashes(part))


def _strip_slashes(value: str) -> str:
    return value.strip("/")
