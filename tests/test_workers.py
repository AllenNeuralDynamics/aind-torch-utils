import queue
import threading
import time
import weakref
from collections import Counter
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import tensorstore as ts
import torch

from aind_torch_utils.accumulators import weighted_average_factory
from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.context import BlockContext
from aind_torch_utils.distributed.sharding import make_shard_spec
from aind_torch_utils.execution import ExecutionPolicy
from aind_torch_utils.outputs import OutputSpec, Threshold
from aind_torch_utils.recovery import PipelineProgress, PipelineTimeoutError
from aind_torch_utils.transforms import (
    GlobalNormalizer,
    IdentityTransform,
    IntensityTransformAdapter,
    from_config,
)
from aind_torch_utils.utils import iter_blocks_zyx
from aind_torch_utils.work_state import BlockKey, BlockLease
from aind_torch_utils.workers import (
    _ABANDONED_WRITES,
    GpuWorker,
    Preds,
    PrepWorker,
    WriterWorker,
    _BlockWriteGroup,
    _BoundedPendingWrites,
    writer_for_key,
)


def test_input_read_deadline_reports_coordinates_before_any_batches(caplog):
    promise, future = ts.Promise.new()

    class Reader:
        shape = (1, 1, 32, 32, 32)

        def __getitem__(self, key):
            return SimpleNamespace(read=lambda: future)

    cfg = _prep_cfg(False).model_copy(update={"read_timeout_s": 0.1})
    prepared = queue.Queue()
    worker = PrepWorker(
        cfg,
        Reader(),
        prepared,
        cfg.patch,
        _default_preprocess(cfg),
        _default_execution(cfg),
    )
    worker.progress = PipelineProgress()
    with pytest.raises(PipelineTimeoutError, match="input block BlockKey"):
        worker.run(threading.Event())
    assert prepared.empty()
    assert "elapsed=" in caplog.text and "bbox=" in caplog.text
    operation = next(iter(worker.progress.snapshot()["outstanding"].values()))
    assert operation["block"] == (0, 0, 0, 0, 0)
    promise.set_result(np.zeros((32, 32, 32)))


def test_stalled_write_is_bounded_retains_source_and_never_marks_complete():
    promise, future = ts.Promise.new()
    completed = []
    work_store = SimpleNamespace(
        complete_block=completed.append,
        fail_block=lambda *args: None,
    )
    group = _BlockWriteGroup(BlockLease(BlockKey(0, 0, 0, 0, 0, 0)), work_store)
    source = np.ones((2, 2, 2))
    ref = weakref.ref(source)
    pending = _BoundedPendingWrites(1, timeout_s=0.1)
    pending.submit(SimpleNamespace(write=lambda array: future), source, group)
    group.seal()
    del source
    started = time.monotonic()
    try:
        with pytest.raises(PipelineTimeoutError):
            pending.flush()
        assert time.monotonic() - started < 1
        assert ref() is not None
        assert completed == [] and group.failed
    finally:
        promise.set_result(None)
        _ABANDONED_WRITES[:] = [p for p in _ABANDONED_WRITES if p.group is not group]


def test_committed_write_is_reaped_without_waiting_for_another_block():
    promise, future = ts.Promise.new()
    group = _BlockWriteGroup(None, SimpleNamespace())
    progress = PipelineProgress()
    pending = _BoundedPendingWrites(1, progress=progress)
    pending.submit(SimpleNamespace(write=lambda array: future), np.ones(1), group)
    group.seal()
    assert progress.snapshot()["outstanding"]
    promise.set_result(None)
    pending.poll()
    assert group.remaining == 0
    assert not progress.snapshot()["outstanding"]


def _default_preprocess(cfg):
    """The same transform run() synthesizes from config."""
    return from_config(
        cfg.normalize, cfg.norm_lower, cfg.norm_upper, cfg.eps, cfg.clip_norm
    )


def _default_execution(cfg):
    """The same execution policy run() synthesizes from config."""
    return ExecutionPolicy.from_config(
        cfg.amp, cfg.use_compile, cfg.compile_mode, cfg.compile_dynamic
    )


def _default_factory(cfg):
    """The same accumulator factory run() synthesizes from config."""
    return weighted_average_factory(
        cfg.eps, cfg.overlap, cfg.seam_mode, cfg.trim_voxels, cfg.min_blend_weight
    )


def _make_input_store(shape):
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "memory"},
        "metadata": {
            "shape": shape,
            "chunks": (1, 1, 16, 16, 16),
            "dtype": "<u2",
        },
    }
    store = ts.open(spec, create=True).result()
    arr = np.random.randint(1, 65535, size=shape, dtype=np.uint16)
    store.write(arr).result()
    return store


def _prep_cfg(use_compile):
    return InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        seam_mode="trim",
        block=(32, 32, 32),
        batch_size=4,
        devices=["cpu"],
        amp=False,
        normalize=False,
        use_compile=use_compile,
    )


def _drain(prep_q):
    batches = []
    while not prep_q.empty():
        batches.append(prep_q.get_nowait())
    return batches


class _FakeResult:
    def result(self, timeout=None):
        return None

    def done(self):
        return True


class _FakeSlice:
    def __init__(self, store):
        self._store = store

    def write(self, arr):
        self._store.written = np.asarray(arr)
        return _FakeResult()


class _FakeDtype:
    def __init__(self, np_dtype):
        self.numpy_dtype = np.dtype(np_dtype)


class _FakeStore:
    """Minimal TensorStore stand-in: records the last array written."""

    def __init__(self, np_dtype=np.float32):
        self._dtype = _FakeDtype(np_dtype)
        self.written = None

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key):
        return _FakeSlice(self)


class _TrackedWriteFuture:
    """Controllable write future that does not retain its source array."""

    def __init__(self, store, write_index, source, fail):
        self._store = store
        self._write_index = write_index
        self._source_ref = weakref.ref(source)
        self._fail = fail

    def result(self, timeout=None):
        self._store.events.append(("result", self._write_index))
        self._store.source_alive_at_result.append(self._source_ref() is not None)
        self._store.inflight -= 1
        if self._fail:
            raise RuntimeError(f"asynchronous write {self._write_index} failed")

    def done(self):
        return False


class _TrackedAsyncSlice:
    def __init__(self, store):
        self._store = store

    def write(self, arr):
        write_index = self._store.write_count
        self._store.write_count += 1
        self._store.inflight += 1
        self._store.max_inflight = max(self._store.max_inflight, self._store.inflight)
        self._store.events.append(("write", write_index))
        return _TrackedWriteFuture(
            self._store,
            write_index,
            arr,
            write_index in self._store.fail_indices,
        )


class _TrackedAsyncStore:
    """TensorStore stand-in for checking asynchronous writer behavior."""

    def __init__(self, fail_indices=()):
        self._dtype = _FakeDtype(np.float32)
        self.fail_indices = set(fail_indices)
        self.events = []
        self.source_alive_at_result = []
        self.write_count = 0
        self.inflight = 0
        self.max_inflight = 0

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key):
        return _TrackedAsyncSlice(self)


def _block_ctx(extent=2, full_shape=(2, 2, 2)):
    """A no-halo BlockContext for a block spanning [0, extent) on each axis."""
    bbox = (slice(0, extent),) * 3
    return BlockContext.from_block(
        block_idx=(0, 0, 0),
        core_bbox=bbox,
        expanded_bbox=bbox,
        full_shape=full_shape,
        t_idx=0,
        c_idx=0,
    )


def _single_patch_preds(host_out, transform_state):
    """A Preds for a 2x2x2 single-patch block that completes on arrival."""
    return Preds(
        block_idx=(0, 0, 0),
        block_bbox=(slice(0, 2), slice(0, 2), slice(0, 2)),
        writer_key=0,
        starts_in_block=[(0, 0, 0)],
        host_out=host_out,
        valid_sizes=[(2, 2, 2)],
        transform_state=transform_state,
        total_patches_in_block=1,
        acc_shape=(2, 2, 2),
        halo_left=(0, 0, 0),
        ctx=_block_ctx(),
        ready_event=None,
    )


def _spec(cfg, store, postprocess=None, accumulator_factory=None):
    """A single OutputSpec mirroring the run() default (invert=output_denormalize)."""
    return OutputSpec(
        store=store,
        accumulator_factory=accumulator_factory or _default_factory(cfg),
        postprocess=postprocess,
        invert=cfg.output_denormalize,
    )


def _run_writer_once(cfg, store, preds, preprocess):
    write_q: "queue.Queue[Optional[Preds]]" = queue.Queue()
    write_q.put(preds)
    write_q.put(None)  # sentinel closes the writer loop
    WriterWorker(
        cfg=cfg,
        outputs=[_spec(cfg, store)],
        write_q=write_q,
        preprocess=preprocess,
    ).run(threading.Event())


def _run_single_patch_blocks(cfg, store, block_count):
    """Run several independently accumulated blocks through one writer."""
    write_q: "queue.Queue[Optional[Preds]]" = queue.Queue()
    for block_index in range(block_count):
        preds = _single_patch_preds(
            torch.full((1, 1, 2, 2, 2), float(block_index)),
            transform_state=None,
        )
        preds.block_idx = (block_index, 0, 0)
        preds.writer_key = block_index
        write_q.put(preds)
    write_q.put(None)
    WriterWorker(
        cfg=cfg,
        outputs=[_spec(cfg, store)],
        write_q=write_q,
        preprocess=IdentityTransform(),
    ).run(threading.Event())


def test_prep_worker_pads_tail_batch_to_constant_shape_when_compiling():
    """With torch.compile, every batch must have exactly batch_size rows so
    the compiled model never sees a varying input shape; padded rows must be
    zero."""
    store = _make_input_store((1, 1, 32, 32, 32))
    cfg = _prep_cfg(use_compile=True)
    prep_q = queue.Queue()
    PrepWorker(
        cfg, store, prep_q, cfg.patch, _default_preprocess(cfg), _default_execution(cfg)
    ).run(threading.Event())

    batches = _drain(prep_q)
    assert batches

    saw_partial = False
    total_real = 0
    for b in batches:
        assert b.host_in.shape == (cfg.batch_size, 1, *cfg.patch)
        n_real = len(b.starts_in_block)
        assert 0 < n_real <= cfg.batch_size
        assert len(b.valid_sizes) == n_real
        # transform_state is per-block (produced by the injected transform), not
        # per-patch. With normalize=False (IdentityTransform) there is no state.
        assert b.transform_state is None
        total_real += n_real
        if n_real < cfg.batch_size:
            saw_partial = True
            assert torch.all(b.host_in[n_real:] == 0)

    # 32^3 block, patch 16, overlap 4 -> 3 starts per axis -> 27 patches
    assert total_real == batches[0].total_patches_in_block == 27
    assert {b.writer_key for b in batches} == {0}
    assert saw_partial, "geometry should produce a partial tail batch"


def test_prep_worker_does_not_pad_in_eager_mode():
    """Without torch.compile there is no constant-shape requirement, so the
    tail batch keeps its true row count instead of wasting compute and copy
    bandwidth on zero padding (matching the pre-compile behavior)."""
    store = _make_input_store((1, 1, 32, 32, 32))
    cfg = _prep_cfg(use_compile=False)
    prep_q = queue.Queue()
    PrepWorker(
        cfg, store, prep_q, cfg.patch, _default_preprocess(cfg), _default_execution(cfg)
    ).run(threading.Event())

    batches = _drain(prep_q)
    assert batches

    saw_partial = False
    for b in batches:
        n_real = len(b.starts_in_block)
        # No padding: the allocation matches the real number of patches.
        assert b.host_in.shape == (n_real, 1, *cfg.patch)
        if n_real < cfg.batch_size:
            saw_partial = True
    assert saw_partial, "geometry should produce a partial tail batch"


def test_prep_worker_uses_execution_input_dtype():
    """Host patch dtype comes from the ExecutionPolicy, not cfg.amp (decoupled)."""
    store = _make_input_store((1, 1, 32, 32, 32))
    cfg = _prep_cfg(use_compile=False)  # cfg.amp is False
    prep_q = queue.Queue()
    # Policy asks for float16 even though cfg.amp is False.
    execution = ExecutionPolicy.from_config(
        amp=True, use_compile=False, compile_mode="default", compile_dynamic=None
    )
    PrepWorker(cfg, store, prep_q, cfg.patch, _default_preprocess(cfg), execution).run(
        threading.Event()
    )

    batches = _drain(prep_q)
    assert batches
    assert all(b.host_in.dtype == torch.float16 for b in batches)


class _ShapeOnlyStore:
    """Reader stand-in for testing block ownership without reading data."""

    def __init__(self, full_zyx):
        self.shape = (1, 1, *full_zyx)


def _routing_prep_worker(cfg, full_zyx, shard_spec, worker_id, local_prep):
    """Construct a PrepWorker used only for its shard-local routing logic."""
    return PrepWorker(
        cfg,
        _ShapeOnlyStore(full_zyx),
        queue.Queue(),
        cfg.patch,
        _default_preprocess(cfg),
        _default_execution(cfg),
        shard_spec=shard_spec,
        worker_id=worker_id,
        num_workers=local_prep,
        global_worker_offset=shard_spec.index * local_prep,
        global_worker_count=shard_spec.count * local_prep,
    )


def test_stride_writer_keys_are_dense_and_balance_current_configuration():
    """The 4-shard/4-prep/8-writer case must use every writer equally."""
    full_zyx = (4096, 2660, 3548)
    local_prep = 4
    shard_count = 4
    num_writers = 8
    cfg = InferenceConfig(
        patch=(64, 64, 64),
        overlap=10,
        block=(256, 256, 256),
        batch_size=32,
        devices=["cpu"],
        amp=False,
        shard_count=shard_count,
        shard_strategy="stride",
    )
    blocks = list(iter_blocks_zyx(full_zyx, cfg.block))
    global_prep = local_prep * shard_count

    for shard_index in range(shard_count):
        shard_spec = make_shard_spec(
            full_zyx, cfg.block, shard_count, shard_index, "stride"
        )
        workers = [
            _routing_prep_worker(cfg, full_zyx, shard_spec, i, local_prep)
            for i in range(local_prep)
        ]
        keys = []
        offset = shard_index * local_prep
        for global_block_index, (block_idx, _) in enumerate(blocks):
            global_slot = global_block_index % global_prep
            if offset <= global_slot < offset + local_prep:
                worker = workers[global_slot - offset]
                keys.append(worker._writer_key_for_block(global_block_index, block_idx))

        assert sorted(keys) == list(range(616))
        counts = Counter(writer_for_key(key, num_writers) for key in keys)
        assert [counts[i] for i in range(num_writers)] == [77] * num_writers


def test_contiguous_writer_keys_are_dense_and_balanced():
    """Contiguous shards use dense local row-major keys despite global gaps."""
    full_zyx = (7, 5, 3)
    local_prep = 4
    shard_count = 4
    num_writers = 5
    cfg = InferenceConfig(
        patch=(1, 1, 1),
        overlap=0,
        block=(2, 2, 2),
        batch_size=1,
        devices=["cpu"],
        amp=False,
        seam_mode="trim",
        trim_voxels=0,
        halo=1,
        shard_count=shard_count,
        shard_strategy="contiguous-z",
    )
    blocks = list(iter_blocks_zyx(full_zyx, cfg.block))

    for shard_index in range(shard_count):
        shard_spec = make_shard_spec(
            full_zyx, cfg.block, shard_count, shard_index, "contiguous-z"
        )
        workers = [
            _routing_prep_worker(cfg, full_zyx, shard_spec, i, local_prep)
            for i in range(local_prep)
        ]
        keys = []
        for global_block_index, (block_idx, _) in enumerate(blocks):
            if workers[0]._block_in_shard(block_idx):
                worker = workers[global_block_index % local_prep]
                keys.append(worker._writer_key_for_block(global_block_index, block_idx))

        assert sorted(keys) == list(range(len(keys)))
        counts = Counter(writer_for_key(key, num_writers) for key in keys)
        writer_counts = [counts[i] for i in range(num_writers)]
        assert max(writer_counts) - min(writer_counts) <= 1


def test_writer_raises_on_mismatched_output_channels_and_writers():
    cfg = InferenceConfig(devices=["cpu"])
    write_q: "queue.Queue[Optional[Preds]]" = queue.Queue()

    # Single writer, but model output has N=2 channels.
    worker = WriterWorker(
        cfg=cfg,
        outputs=[_spec(cfg, object())],
        write_q=write_q,
        preprocess=IdentityTransform(),
    )

    preds = Preds(
        block_idx=(0, 0, 0),
        block_bbox=(slice(0, 2), slice(0, 2), slice(0, 2)),
        writer_key=0,
        starts_in_block=[(0, 0, 0)],
        host_out=torch.zeros((1, 2, 2, 2, 2), dtype=torch.float32),
        valid_sizes=[(2, 2, 2)],
        transform_state=(0.0, 1.0),
        total_patches_in_block=1,
        acc_shape=(2, 2, 2),
        halo_left=(0, 0, 0),
        ctx=_block_ctx(),
        ready_event=None,
    )

    write_q.put(preds)

    with pytest.raises(ValueError, match="Mismatch between model output channels"):
        worker.run(stop_event=threading.Event())


def test_writer_applies_block_level_transform_state_inverse():
    """output_denormalize=True applies the transform's inverse from transform_state."""
    cfg = InferenceConfig(devices=["cpu"], output_denormalize=True)
    store = _FakeStore(np.float32)

    # An affine (Global) normalizer: inverse(v, (10, 20)) = v * 10 + 10.
    # Normalized output 0.5 -> 0.5 * 10 + 10 = 15.
    preprocess = GlobalNormalizer(10.0, 20.0, eps=cfg.eps)
    host_out = torch.full((1, 1, 2, 2, 2), 0.5, dtype=torch.float32)
    _run_writer_once(
        cfg,
        store,
        _single_patch_preds(host_out, transform_state=(10.0, 20.0)),
        preprocess,
    )

    assert store.written is not None
    assert store.written.shape == (2, 2, 2)
    np.testing.assert_allclose(store.written, 15.0)


def test_writer_ignores_transform_state_when_denorm_disabled():
    """output_denormalize=False writes outputs as-is and never inverts."""
    cfg = InferenceConfig(devices=["cpu"], output_denormalize=False)
    store = _FakeStore(np.float32)

    host_out = torch.full((1, 1, 2, 2, 2), 0.5, dtype=torch.float32)
    # An invertible transform is supplied, but output_denormalize=False skips it.
    _run_writer_once(
        cfg,
        store,
        _single_patch_preds(host_out, transform_state=(10.0, 20.0)),
        GlobalNormalizer(10.0, 20.0, eps=cfg.eps),
    )

    np.testing.assert_allclose(store.written, 0.5)


def test_writer_bounds_asynchronous_writes_and_retains_sources_until_completion():
    cfg = InferenceConfig(
        devices=["cpu"],
        output_denormalize=False,
        max_pending_writes=2,
    )
    store = _TrackedAsyncStore()

    _run_single_patch_blocks(cfg, store, block_count=3)

    # Two writes are submitted without waiting; the third applies backpressure.
    assert store.events[:3] == [("write", 0), ("write", 1), ("result", 0)]
    assert store.max_inflight == cfg.max_pending_writes
    # The sentinel flushes both writes that remain after backpressure releases.
    assert [event for event in store.events if event[0] == "result"] == [
        ("result", 0),
        ("result", 1),
        ("result", 2),
    ]
    assert store.inflight == 0
    assert all(store.source_alive_at_result)


def test_writer_flushes_all_pending_writes_and_propagates_async_error():
    cfg = InferenceConfig(
        devices=["cpu"],
        output_denormalize=False,
        max_pending_writes=3,
    )
    store = _TrackedAsyncStore(fail_indices={0})

    with pytest.raises(RuntimeError, match="asynchronous write 0 failed"):
        _run_single_patch_blocks(cfg, store, block_count=3)

    # One failed future must not prevent the remaining writes from being awaited.
    assert [event for event in store.events if event[0] == "result"] == [
        ("result", 0),
        ("result", 1),
        ("result", 2),
    ]
    assert store.inflight == 0


def test_nonlinear_intensity_inverse_runs_once_after_patch_accumulation():
    """Match source inference: average transformed predictions, then invert."""
    cfg = InferenceConfig(
        devices=["cpu"],
        output_denormalize=True,
        seam_mode="blend",
        trim_voxels=None,
    )
    store = _FakeStore(np.float32)

    class SquareInverse:
        def __init__(self):
            self.inverse_calls = 0

        def forward(self, array):
            return np.sqrt(array)

        def inverse(self, array):
            self.inverse_calls += 1
            return np.square(array)

    source = SquareInverse()
    preprocess = IntensityTransformAdapter(source)
    # Both patches cover the same block. Their transformed-space values average
    # to 2, so post-merge inversion produces 4. Inverting first would produce 5.
    host_out = torch.stack(
        [
            torch.full((1, 2, 2, 2), 1.0),
            torch.full((1, 2, 2, 2), 3.0),
        ]
    )
    preds = Preds(
        block_idx=(0, 0, 0),
        block_bbox=(slice(0, 2), slice(0, 2), slice(0, 2)),
        writer_key=0,
        starts_in_block=[(0, 0, 0), (0, 0, 0)],
        host_out=host_out,
        valid_sizes=[(2, 2, 2), (2, 2, 2)],
        transform_state=None,
        total_patches_in_block=2,
        acc_shape=(2, 2, 2),
        halo_left=(0, 0, 0),
        ctx=_block_ctx(),
        ready_event=None,
    )

    _run_writer_once(cfg, store, preds, preprocess)

    np.testing.assert_allclose(store.written, 4.0)
    assert source.inverse_calls == 1


def test_writer_per_output_merge_post_and_dtype():
    """Two outputs from one (B, 2, ...) tensor take different merge/post/dtype.

    Channel 0: max-merge + threshold(0) + uint8 (a mask). Channel 1: weighted
    average + no post + float32 (a raw field). A single block-sized patch means
    merge is a no-op, isolating the per-output post/dtype/invert wiring.
    """
    from aind_torch_utils.accumulators import MaxAccumulator

    cfg = InferenceConfig(devices=["cpu"], output_denormalize=False)
    mask_store = _FakeStore(np.uint8)
    field_store = _FakeStore(np.float32)

    mask_spec = OutputSpec(
        store=mask_store,
        accumulator_factory=lambda shp, ctx: MaxAccumulator(shp),
        postprocess=Threshold(thresh=0.0),
        invert=False,
    )
    field_spec = OutputSpec(
        store=field_store,
        accumulator_factory=_default_factory(cfg),
        postprocess=None,
        invert=False,
    )

    # channel 0 = logits (mixed sign), channel 1 = a raw field.
    logits = np.array([-2.0, 3.0, -1.0, 4.0, 0.5, -0.5, 9.0, -9.0], np.float32)
    field = np.arange(8, dtype=np.float32)
    host_out = torch.from_numpy(
        np.stack([logits.reshape(2, 2, 2), field.reshape(2, 2, 2)])[None]
    )  # (1, 2, 2, 2, 2)

    preds = Preds(
        block_idx=(0, 0, 0),
        block_bbox=(slice(0, 2), slice(0, 2), slice(0, 2)),
        writer_key=0,
        starts_in_block=[(0, 0, 0)],
        host_out=host_out,
        valid_sizes=[(2, 2, 2)],
        transform_state=None,
        total_patches_in_block=1,
        acc_shape=(2, 2, 2),
        halo_left=(0, 0, 0),
        ctx=_block_ctx(),
        ready_event=None,
    )

    write_q: "queue.Queue[Optional[Preds]]" = queue.Queue()
    write_q.put(preds)
    write_q.put(None)
    WriterWorker(
        cfg=cfg,
        outputs=[mask_spec, field_spec],
        write_q=write_q,
        preprocess=IdentityTransform(),
    ).run(threading.Event())

    # Mask: threshold(logit > 0) -> uint8 binary.
    assert mask_store.written.dtype == np.uint8
    np.testing.assert_array_equal(
        mask_store.written.ravel(), (logits > 0).astype(np.uint8)
    )
    # Field: raw values as float32.
    assert field_store.written.dtype == np.float32
    np.testing.assert_allclose(field_store.written.ravel(), field)


def _make_compile_worker():
    """Build a GpuWorker shell without running __init__ (which needs CUDA),
    wired with just the attributes _compile_model touches."""
    worker = object.__new__(GpuWorker)
    cfg = InferenceConfig(devices=["cpu"], use_compile=True)
    worker.cfg = cfg
    worker.device = torch.device("cpu")
    worker.model = torch.nn.Identity()
    worker.execution = _default_execution(cfg)
    return worker


def test_compile_model_falls_back_to_eager_when_compile_call_raises(monkeypatch):
    worker = _make_compile_worker()
    eager = worker.model

    def boom(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(torch, "compile", boom)

    worker._compile_model()

    assert worker.model is eager


def test_compile_model_falls_back_to_eager_when_warmup_raises(monkeypatch):
    """The real failure mode: torch.compile returns lazily, then tracing
    blows up on the first forward inside warmup."""
    worker = _make_compile_worker()
    eager = worker.model

    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: torch.nn.Identity())

    def warmup_boom(self):
        raise RuntimeError("Guard failed on the same frame it was created")

    monkeypatch.setattr(GpuWorker, "_warmup_compiled_model", warmup_boom)

    worker._compile_model()

    assert worker.model is eager


def test_compile_model_keeps_compiled_module_on_success(monkeypatch):
    worker = _make_compile_worker()
    compiled = torch.nn.Identity()

    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: compiled)
    monkeypatch.setattr(GpuWorker, "_warmup_compiled_model", lambda self: None)

    worker._compile_model()

    assert worker.model is compiled
