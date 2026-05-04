from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Set, Tuple
from urllib.parse import urlparse

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.distributed.sharding import ShardSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BlockKey:
    """
    Stable identifier for a single output block.
    """

    t: int
    c: int
    z: int
    y: int
    x: int
    linear_k: int

    @property
    def coords(self) -> Tuple[int, int, int, int, int]:
        return self.t, self.c, self.z, self.y, self.x


@dataclass(frozen=True, slots=True)
class BlockLease:
    """
    Claim returned by a work store for a block this worker may process.
    """

    block: BlockKey
    token: Optional[str] = None


class BlockWorkStore(Protocol):
    """
    Backend-independent block completion/claim interface.

    The S3 marker backend uses this as a skip/completion marker API. A future
    DynamoDB backend can use the same calls for conditional claim and complete
    state transitions.
    """

    def prepare(self, shard_spec: ShardSpec) -> None:
        """Load any backend state needed before worker threads start."""
        ...

    def claim_block(self, block: BlockKey) -> Optional[BlockLease]:
        """Return a lease for blocks to process, or None when already complete."""
        ...

    def complete_block(self, lease: BlockLease) -> None:
        """Mark a block complete after the output write succeeds."""
        ...

    def fail_block(self, lease: BlockLease, exc: BaseException) -> None:
        """Record a block failure, if the backend supports it."""
        ...


class NoopBlockWorkStore:
    """
    Existing behavior: every block is processed and no completion state is kept.
    """

    def prepare(self, shard_spec: ShardSpec) -> None:
        return None

    def claim_block(self, block: BlockKey) -> BlockLease:
        return BlockLease(block)

    def complete_block(self, lease: BlockLease) -> None:
        return None

    def fail_block(self, lease: BlockLease, exc: BaseException) -> None:
        return None


class S3MarkerBlockWorkStore:
    """
    Block work store that treats sidecar S3 marker objects as completion state.

    The output block is written first. Only after that write succeeds does this
    backend write the small ``.done`` marker object.
    """

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
        return f"{self.prefix}/t={self.t_idx}/c={self.c_idx}/"

    def prepare(self, shard_spec: ShardSpec) -> None:
        client = self._client()
        paginator = client.get_paginator("list_objects_v2")
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
        with self._lock:
            if block.coords in self._completed:
                return None
        return BlockLease(block)

    def complete_block(self, lease: BlockLease) -> None:
        block = lease.block
        key = self._key_for_block(block)
        body = json.dumps(
            {
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
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        with self._lock:
            self._completed.add(block.coords)

    def fail_block(self, lease: BlockLease, exc: BaseException) -> None:
        logger.warning(
            "Block %s failed before completion marker was written: %s",
            lease.block,
            exc,
        )

    def _client(self) -> Any:
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised by integration use
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
    output_spec: Dict[str, Any],
    input_store: Any,
    output_store: Any,
    model_type: Optional[str] = None,
    weights_path: Optional[str] = None,
) -> BlockWorkStore:
    """
    Build the configured block work store for CLI/programmatic orchestration.
    """

    if not cfg.resume:
        return NoopBlockWorkStore()
    validate_resume_output_spec(cfg, output_spec)
    if cfg.work_store == "none":
        raise ValueError("cfg.resume=True requires a non-'none' work_store backend")
    if cfg.work_store != "s3-markers":
        raise ValueError(f"Unsupported work_store backend: {cfg.work_store}")

    bucket, base_prefix = _resolve_marker_location(cfg, output_spec)
    run_id = cfg.resume_run_id or _derive_run_id(
        cfg=cfg,
        input_store=input_store,
        output_store=output_store,
        output_spec=output_spec,
        model_type=model_type,
        weights_path=weights_path,
    )
    marker_prefix = _join_s3_key(
        base_prefix,
        ".aind_torch_utils",
        "resume",
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


def validate_resume_output_spec(cfg: InferenceConfig, output_spec: Any) -> None:
    """
    Reject output specs that would destroy resumable state before opening them.
    """

    if not cfg.resume:
        return
    if not isinstance(output_spec, dict):
        raise ValueError("Resume requires the output TensorStore spec to be an object")
    if _delete_existing_enabled(output_spec):
        raise ValueError(
            "Resume cannot be used with an output spec that has "
            "`delete_existing: true`; that would delete existing output and markers."
        )


def _delete_existing_enabled(spec: Dict[str, Any]) -> bool:
    return bool(spec.get("delete_existing", False))


def _resolve_marker_location(
    cfg: InferenceConfig, output_spec: Dict[str, Any]
) -> Tuple[str, str]:
    if cfg.resume_marker_prefix:
        return _parse_s3_uri(cfg.resume_marker_prefix)
    return _extract_s3_kvstore_root(output_spec)


def _extract_s3_kvstore_root(spec: Dict[str, Any]) -> Tuple[str, str]:
    kvstore = spec.get("kvstore")
    if isinstance(kvstore, str) and kvstore.startswith("s3://"):
        return _parse_s3_uri(kvstore)
    if isinstance(kvstore, dict) and kvstore.get("driver") == "s3":
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


def _derive_run_id(
    *,
    cfg: InferenceConfig,
    input_store: Any,
    output_store: Any,
    output_spec: Dict[str, Any],
    model_type: Optional[str],
    weights_path: Optional[str],
) -> str:
    fields = {
        "input_shape": tuple(input_store.domain.shape),
        "input_dtype": str(input_store.dtype.numpy_dtype),
        "output_shape": tuple(output_store.domain.shape),
        "output_dtype": str(output_store.dtype.numpy_dtype),
        "output_path": output_spec.get("path"),
        "model_type": model_type,
        "weights_path": weights_path,
        "config": {
            "patch": cfg.patch,
            "overlap": cfg.overlap,
            "block": cfg.block,
            "t_idx": cfg.t_idx,
            "c_idx": cfg.c_idx,
            "seam_mode": cfg.seam_mode,
            "trim_voxels": cfg.trim_voxels,
            "halo": cfg.halo,
            "min_blend_weight": cfg.min_blend_weight,
            "eps": cfg.eps,
            "norm_lower": cfg.norm_lower,
            "norm_upper": cfg.norm_upper,
            "normalize": cfg.normalize,
            "clip_norm": cfg.clip_norm,
            "amp": cfg.amp,
            "use_tf32": cfg.use_tf32,
        },
    }
    encoded = json.dumps(fields, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _join_s3_key(*parts: str) -> str:
    return "/".join(_strip_slashes(part) for part in parts if _strip_slashes(part))


def _strip_slashes(value: str) -> str:
    return value.strip("/")
