import queue
import threading

import numpy as np
import pytest
import torch

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.distributed.sharding import ShardSpec
from aind_torch_utils.work_state import (
    BlockKey,
    BlockLease,
    NoopBlockWorkStore,
    S3MarkerBlockWorkStore,
    build_block_work_store,
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
    base = dict(
        patch=(4, 4, 4),
        overlap=0,
        block=(8, 8, 8),
        batch_size=1,
        t_idx=0,
        c_idx=0,
        devices=["cuda:0"],
        amp=False,
        seam_mode="trim",
        trim_voxels=0,
        halo=0,
        normalize=False,
    )
    base.update(kwargs)
    return InferenceConfig(**base)


class FakePaginator:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        return self.pages


class FakeS3Client:
    def __init__(self, pages=None):
        self.paginator = FakePaginator(pages or [])
        self.puts = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self.paginator

    def put_object(self, **kwargs):
        self.puts.append(kwargs)


def test_noop_work_store_claims_all_blocks():
    store = NoopBlockWorkStore()
    block = BlockKey(t=0, c=0, z=1, y=2, x=3, linear_k=42)

    lease = store.claim_block(block)

    assert lease == BlockLease(block)
    store.complete_block(lease)
    store.fail_block(lease, RuntimeError("ignored"))


def test_s3_marker_store_lists_once_and_skips_completed_blocks():
    s3 = FakeS3Client(
        pages=[
            {
                "Contents": [
                    {
                        "Key": (
                            "out/.aind_torch_utils/resume/run/t=0/c=0/"
                            "z=1/y=2/x=3.done"
                        )
                    }
                ]
            }
        ]
    )
    store = S3MarkerBlockWorkStore(
        bucket="bucket",
        prefix="out/.aind_torch_utils/resume/run",
        t_idx=0,
        c_idx=0,
        run_id="run",
        s3_client=s3,
    )

    store.prepare(_shard_spec())

    assert s3.paginator.calls == [
        {
            "Bucket": "bucket",
            "Prefix": "out/.aind_torch_utils/resume/run/t=0/c=0/",
        }
    ]
    completed = BlockKey(t=0, c=0, z=1, y=2, x=3, linear_k=99)
    missing = BlockKey(t=0, c=0, z=1, y=2, x=4, linear_k=100)
    assert store.claim_block(completed) is None
    assert store.claim_block(missing) == BlockLease(missing)


def test_s3_marker_store_writes_marker_after_completion():
    s3 = FakeS3Client()
    store = S3MarkerBlockWorkStore(
        bucket="bucket",
        prefix="/out/.aind_torch_utils/resume/run/",
        t_idx=0,
        c_idx=0,
        run_id="run",
        s3_client=s3,
    )
    block = BlockKey(t=0, c=0, z=1, y=2, x=3, linear_k=42)

    store.complete_block(BlockLease(block))

    assert s3.puts[0]["Bucket"] == "bucket"
    assert (
        s3.puts[0]["Key"] == "out/.aind_torch_utils/resume/run/t=0/c=0/z=1/y=2/x=3.done"
    )
    assert s3.puts[0]["ContentType"] == "application/json"
    assert store.claim_block(block) is None


class SkippingStore:
    def __init__(self):
        self.claimed = []

    def prepare(self, shard_spec):
        return None

    def claim_block(self, block):
        self.claimed.append(block)
        return None

    def complete_block(self, lease):
        raise AssertionError("completed skipped block")

    def fail_block(self, lease, exc):
        raise AssertionError("failed skipped block")


class ReaderThatMustNotRead:
    shape = (1, 1, 16, 16, 16)

    def __getitem__(self, item):
        raise AssertionError("completed blocks should be skipped before input reads")


def test_prep_worker_skips_completed_blocks_before_input_read():
    store = SkippingStore()
    worker = PrepWorker(
        _cfg(resume=True),
        ReaderThatMustNotRead(),
        queue.Queue(),
        (4, 4, 4),
        _shard_spec(),
        store,
    )

    worker.run(threading.Event())

    assert len(store.claimed) == 8


class RecordingWorkStore:
    def __init__(self):
        self.completed = []
        self.failed = []

    def prepare(self, shard_spec):
        return None

    def claim_block(self, block):
        return BlockLease(block)

    def complete_block(self, lease):
        self.completed.append(lease)

    def fail_block(self, lease, exc):
        self.failed.append((lease, exc))


class FakeFuture:
    def __init__(self, exc=None):
        self.exc = exc

    def result(self):
        if self.exc is not None:
            raise self.exc


class FakeWriteView:
    def __init__(self, writer):
        self.writer = writer

    def write(self, arr):
        self.writer.writes.append(np.asarray(arr))
        return FakeFuture(self.writer.exc)


class FakeDType:
    numpy_dtype = np.dtype("float32")


class FakeWriter:
    dtype = FakeDType()

    def __init__(self, exc=None):
        self.exc = exc
        self.writes = []

    def __getitem__(self, item):
        self.last_item = item
        return FakeWriteView(self)


def _preds():
    block = BlockKey(t=0, c=0, z=0, y=0, x=0, linear_k=0)
    return Preds(
        block_idx=(0, 0, 0),
        block_bbox=(slice(0, 4), slice(0, 4), slice(0, 4)),
        linear_k=0,
        starts_in_block=[(0, 0, 0)],
        host_out=torch.ones((1, 1, 4, 4, 4), dtype=torch.float32),
        valid_sizes=[(4, 4, 4)],
        per_block_minmax=[(0.0, 1.0)],
        total_patches_in_block=1,
        acc_shape=(4, 4, 4),
        halo_left=(0, 0, 0),
        lease=BlockLease(block),
    )


def test_writer_marks_block_complete_after_output_write_succeeds():
    write_q = queue.Queue()
    write_q.put(_preds())
    write_q.put(None)
    work_store = RecordingWorkStore()
    writer = FakeWriter()

    WriterWorker(_cfg(), writer, write_q, work_store).run(threading.Event())

    assert len(writer.writes) == 1
    assert np.all(writer.writes[0] == 1.0)
    assert len(work_store.completed) == 1
    assert work_store.failed == []


def test_writer_does_not_mark_complete_when_output_write_fails():
    write_q = queue.Queue()
    write_q.put(_preds())
    work_store = RecordingWorkStore()
    writer = FakeWriter(exc=RuntimeError("write failed"))

    with pytest.raises(RuntimeError, match="write failed"):
        WriterWorker(_cfg(), writer, write_q, work_store).run(threading.Event())

    assert work_store.completed == []
    assert len(work_store.failed) == 1


class FakeDomain:
    shape = (1, 1, 8, 8, 8)


class FakeStore:
    domain = FakeDomain()
    dtype = FakeDType()


def test_build_work_store_rejects_delete_existing_for_resume():
    cfg = _cfg(resume=True)

    with pytest.raises(ValueError, match="delete_existing"):
        build_block_work_store(
            cfg=cfg,
            output_spec={
                "driver": "zarr",
                "kvstore": {
                    "driver": "s3",
                    "bucket": "bucket",
                    "path": "out.zarr",
                },
                "delete_existing": True,
            },
            input_store=FakeStore(),
            output_store=FakeStore(),
            model_type="model",
            weights_path="weights.pth",
        )
