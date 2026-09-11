import json
import queue
import threading

import numpy as np
import pytest
import torch

from aind_torch_utils.accumulators import weighted_average_factory
from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.context import BlockContext
from aind_torch_utils.distributed.sharding import ShardSpec
from aind_torch_utils.execution import ExecutionPolicy
from aind_torch_utils.outputs import OutputSpec
from aind_torch_utils.transforms import IdentityTransform
from aind_torch_utils.work_state import (
    BlockKey,
    BlockLease,
    NoopBlockWorkStore,
    S3MarkerBlockWorkStore,
    build_block_work_store,
    derive_run_id,
    validate_resume_output_specs,
)
from aind_torch_utils.workers import Preds, PrepWorker, WriterWorker


def _shard_spec():
    return ShardSpec(
        index=0,
        count=1,
        strategy="stride",
        block_start=(0, 0, 0),
        block_stop=(2, 2, 2),
        grid_shape=(2, 2, 2),
        tile_index=(0, 0, 0),
        tiles_per_axis=(1, 1, 1),
    )


def _cfg(**kwargs):
    values = {
        "patch": (4, 4, 4),
        "overlap": 0,
        "block": (8, 8, 8),
        "batch_size": 1,
        "devices": ["cpu"],
        "amp": False,
        "seam_mode": "trim",
        "trim_voxels": 0,
        "halo": 1,
        "normalize": False,
        "output_denormalize": False,
    }
    values.update(kwargs)
    return InferenceConfig(**values)


class _Paginator:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        return self.pages


class _S3Client:
    def __init__(self, pages=()):
        self.paginator = _Paginator(pages)
        self.puts = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self.paginator

    def put_object(self, **kwargs):
        self.puts.append(kwargs)


def test_noop_work_store_claims_every_block():
    store = NoopBlockWorkStore()
    block = BlockKey(t=0, c=0, z=1, y=2, x=3, linear_k=42)

    assert store.claim_block(block) == BlockLease(block)


def test_s3_marker_store_loads_and_writes_versioned_markers():
    prefix = "out/.aind_torch_utils/resume/v2/run"
    client = _S3Client(
        [
            {
                "Contents": [
                    {"Key": f"{prefix}/t=0/c=0/z=1/y=2/x=3.done"},
                    {"Key": f"{prefix}/t=0/c=0/not-a-marker"},
                ]
            }
        ]
    )
    store = S3MarkerBlockWorkStore(
        bucket="bucket",
        prefix=prefix,
        t_idx=0,
        c_idx=0,
        run_id="run",
        s3_client=client,
    )

    store.prepare(_shard_spec())

    completed = BlockKey(t=0, c=0, z=1, y=2, x=3, linear_k=99)
    missing = BlockKey(t=0, c=0, z=1, y=2, x=4, linear_k=100)
    assert store.claim_block(completed) is None
    lease = store.claim_block(missing)
    assert lease == BlockLease(missing)

    store.complete_block(lease)

    assert client.paginator.calls == [
        {"Bucket": "bucket", "Prefix": f"{prefix}/t=0/c=0/"}
    ]
    assert client.puts[0]["Key"].endswith("/z=1/y=2/x=4.done")
    assert json.loads(client.puts[0]["Body"])["schema_version"] == 2
    assert store.claim_block(missing) is None


class _Domain:
    shape = (1, 1, 8, 8, 8)


class _DType:
    numpy_dtype = np.dtype("float32")


class _Store:
    domain = _Domain()
    dtype = _DType()


def _input_spec(path="input.zarr"):
    return {"driver": "zarr", "kvstore": f"s3://bucket/{path}"}


def _output_spec(path="output.zarr", **kwargs):
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "s3", "bucket": "bucket", "path": path},
        "path": "0",
    }
    spec.update(kwargs)
    return spec


def test_build_store_derives_single_output_location_and_fresh_namespace():
    store = build_block_work_store(
        cfg=_cfg(resume=True),
        input_spec=_input_spec(),
        output_specs=[_output_spec()],
        input_store=_Store(),
        output_stores=[_Store()],
        workload={"kind": "model", "model_type": "dummy"},
    )

    assert isinstance(store, S3MarkerBlockWorkStore)
    assert store.bucket == "bucket"
    assert store.prefix.startswith("output.zarr/.aind_torch_utils/resume/v2/")


def test_build_store_accepts_original_single_output_api():
    store = build_block_work_store(
        cfg=_cfg(resume=True),
        output_spec=_output_spec(),
        input_store=_Store(),
        output_store=_Store(),
        model_type="dummy",
        weights_path="weights.pth",
    )

    assert isinstance(store, S3MarkerBlockWorkStore)


def test_resume_rejects_destructive_and_ambiguous_multi_output_specs():
    cfg = _cfg(resume=True)
    with pytest.raises(ValueError, match="delete_existing"):
        validate_resume_output_specs(
            cfg, [_output_spec(), _output_spec("other.zarr", delete_existing=True)]
        )
    with pytest.raises(ValueError, match="resume_marker_prefix"):
        validate_resume_output_specs(cfg, [_output_spec(), _output_spec("other.zarr")])


def test_explicit_prefix_supports_multi_output_resume():
    cfg = _cfg(
        resume=True,
        resume_marker_prefix="s3://marker-bucket/shared/checkpoints",
    )
    store = build_block_work_store(
        cfg=cfg,
        input_spec=_input_spec(),
        output_specs=[_output_spec(), _output_spec("other.zarr")],
        input_store=_Store(),
        output_stores=[_Store(), _Store()],
        workload={"kind": "workflow", "name": "multi", "params": {}},
    )

    assert isinstance(store, S3MarkerBlockWorkStore)
    assert store.bucket == "marker-bucket"
    assert store.prefix.startswith("shared/checkpoints/.aind_torch_utils/resume/v2/")


def test_run_id_ignores_lifecycle_and_sharding_but_tracks_workload():
    common = {
        "input_spec": _input_spec(),
        "input_store": _Store(),
        "output_stores": [_Store()],
    }
    first = derive_run_id(
        cfg=_cfg(resume=True, shard_count=1, shard_index=0),
        output_specs=[_output_spec(create=True, open=False)],
        workload={"kind": "workflow", "name": "denoise", "params": {"x": 1}},
        **common,
    )
    retry = derive_run_id(
        cfg=_cfg(
            resume=True,
            shard_count=2,
            shard_index=1,
            read_timeout_s=123,
            write_timeout_s=234,
            progress_timeout_s=345,
            startup_timeout_s=456,
            shutdown_timeout_s=12,
            max_shard_retries=4,
            retry_backoff_s=0,
            diagnostic_timeout_s=3,
            diagnostics_dir="/tmp/diagnostics",
        ),
        output_specs=[_output_spec(create=False, open=True)],
        workload={"kind": "workflow", "name": "denoise", "params": {"x": 1}},
        **common,
    )
    changed = derive_run_id(
        cfg=_cfg(resume=True, shard_count=2, shard_index=0),
        output_specs=[_output_spec(open=True)],
        workload={"kind": "workflow", "name": "denoise", "params": {"x": 2}},
        **common,
    )

    assert first == retry
    assert changed != first


class _SkippingWorkStore:
    def __init__(self):
        self.claimed = []

    def claim_block(self, block):
        self.claimed.append(block)
        return None


class _ReaderThatMustNotRead:
    shape = (1, 1, 16, 16, 16)

    def __getitem__(self, item):
        raise AssertionError("completed blocks must be skipped before input reads")


def test_prep_worker_skips_completed_blocks_before_input_read():
    store = _SkippingWorkStore()
    worker = PrepWorker(
        _cfg(resume=True),
        _ReaderThatMustNotRead(),
        queue.Queue(),
        (4, 4, 4),
        IdentityTransform(),
        ExecutionPolicy(),
        shard_spec=_shard_spec(),
        work_store=store,
    )

    worker.run(threading.Event())

    assert len(store.claimed) == 8


class _RecordingWorkStore:
    def __init__(self, events, completion_error=None):
        self.events = events
        self.completion_error = completion_error
        self.completed = []
        self.failed = []

    def complete_block(self, lease):
        self.events.append(("complete", lease.block.coords))
        self.completed.append(lease)
        if self.completion_error is not None:
            raise self.completion_error

    def fail_block(self, lease, exc):
        self.events.append(("fail", str(exc)))
        self.failed.append((lease, exc))


class _WriteFuture:
    def __init__(self, events, label, fail=False):
        self.events = events
        self.label = label
        self.fail = fail

    def result(self, timeout=None):
        self.events.append(("result", self.label))
        if self.fail:
            raise RuntimeError(f"{self.label} failed")

    def done(self):
        return False


class _WriteView:
    def __init__(self, store):
        self.store = store

    def write(self, array):
        self.store.events.append(("write", self.store.label))
        self.store.sources.append(array)
        return _WriteFuture(self.store.events, self.store.label, fail=self.store.fail)


class _OutputStore:
    dtype = _DType()

    def __init__(self, events, label, fail=False):
        self.events = events
        self.label = label
        self.fail = fail
        self.sources = []

    def __getitem__(self, item):
        return _WriteView(self)


def _preds(lease):
    bbox = (slice(0, 2),) * 3
    ctx = BlockContext.from_block(
        block_idx=(0, 0, 0),
        core_bbox=bbox,
        expanded_bbox=bbox,
        full_shape=(2, 2, 2),
        t_idx=0,
        c_idx=0,
    )
    return Preds(
        block_idx=(0, 0, 0),
        block_bbox=bbox,
        writer_key=0,
        starts_in_block=[(0, 0, 0)],
        host_out=torch.ones((1, 2, 2, 2, 2), dtype=torch.float32),
        valid_sizes=[(2, 2, 2)],
        transform_state=None,
        total_patches_in_block=1,
        acc_shape=(2, 2, 2),
        halo_left=(0, 0, 0),
        ctx=ctx,
        lease=lease,
    )


def _output(store):
    return OutputSpec(
        store=store,
        accumulator_factory=weighted_average_factory(1e-6, 0, "trim", 0, 0.05),
        invert=False,
    )


def test_writer_completes_only_after_every_output_future_commits():
    events = []
    work_store = _RecordingWorkStore(events)
    stores = [_OutputStore(events, "first"), _OutputStore(events, "second")]
    block = BlockKey(0, 0, 0, 0, 0, 0)
    write_q = queue.Queue()
    write_q.put(_preds(BlockLease(block)))
    write_q.put(None)

    WriterWorker(
        _cfg(max_pending_writes=2),
        [_output(store) for store in stores],
        write_q,
        IdentityTransform(),
        work_store,
    ).run(threading.Event())

    assert events == [
        ("write", "first"),
        ("write", "second"),
        ("result", "first"),
        ("result", "second"),
        ("complete", block.coords),
    ]
    assert len(work_store.completed) == 1


def test_writer_never_completes_a_partially_failed_multi_output_block():
    events = []
    work_store = _RecordingWorkStore(events)
    stores = [
        _OutputStore(events, "first"),
        _OutputStore(events, "second", fail=True),
    ]
    block = BlockKey(0, 0, 0, 0, 0, 0)
    write_q = queue.Queue()
    write_q.put(_preds(BlockLease(block)))
    write_q.put(None)

    with pytest.raises(RuntimeError, match="second failed"):
        WriterWorker(
            _cfg(max_pending_writes=2),
            [_output(store) for store in stores],
            write_q,
            IdentityTransform(),
            work_store,
        ).run(threading.Event())

    assert work_store.completed == []
    assert len(work_store.failed) == 1


def test_writer_propagates_completion_marker_failure():
    events = []
    work_store = _RecordingWorkStore(
        events, completion_error=RuntimeError("marker write failed")
    )
    stores = [_OutputStore(events, "first"), _OutputStore(events, "second")]
    block = BlockKey(0, 0, 0, 0, 0, 0)
    write_q = queue.Queue()
    write_q.put(_preds(BlockLease(block)))
    write_q.put(None)

    with pytest.raises(RuntimeError, match="marker write failed"):
        WriterWorker(
            _cfg(max_pending_writes=2),
            [_output(store) for store in stores],
            write_q,
            IdentityTransform(),
            work_store,
        ).run(threading.Event())

    assert events[-1] == ("complete", block.coords)
