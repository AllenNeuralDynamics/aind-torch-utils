import queue
import threading
from typing import Optional

import numpy as np
import pytest
import tensorstore as ts
import torch

from aind_torch_utils.accumulators import weighted_average_factory
from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.outputs import OutputSpec, Threshold
from aind_torch_utils.transforms import (
    GlobalNormalizer,
    IdentityTransform,
    from_config,
)
from aind_torch_utils.workers import GpuWorker, Preds, PrepWorker, WriterWorker


def _default_preprocess(cfg):
    """The same transform run() synthesizes from config."""
    return from_config(
        cfg.normalize, cfg.norm_lower, cfg.norm_upper, cfg.eps, cfg.clip_norm
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
    def result(self):
        return None


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


def _single_patch_preds(host_out, transform_state):
    """A Preds for a 2x2x2 single-patch block that completes on arrival."""
    return Preds(
        block_idx=(0, 0, 0),
        block_bbox=(slice(0, 2), slice(0, 2), slice(0, 2)),
        linear_k=0,
        starts_in_block=[(0, 0, 0)],
        host_out=host_out,
        valid_sizes=[(2, 2, 2)],
        transform_state=transform_state,
        total_patches_in_block=1,
        acc_shape=(2, 2, 2),
        halo_left=(0, 0, 0),
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


def _run_writer_once(cfg, store, preds, preprocess, full_shape=(2, 2, 2)):
    write_q: "queue.Queue[Optional[Preds]]" = queue.Queue()
    write_q.put(preds)
    write_q.put(None)  # sentinel closes the writer loop
    WriterWorker(
        cfg=cfg,
        outputs=[_spec(cfg, store)],
        write_q=write_q,
        preprocess=preprocess,
        full_shape=full_shape,
    ).run(threading.Event())


def test_prep_worker_pads_tail_batch_to_constant_shape_when_compiling():
    """With torch.compile, every batch must have exactly batch_size rows so
    the compiled model never sees a varying input shape; padded rows must be
    zero."""
    store = _make_input_store((1, 1, 32, 32, 32))
    cfg = _prep_cfg(use_compile=True)
    prep_q = queue.Queue()
    PrepWorker(
        cfg, store, prep_q, cfg.patch, _default_preprocess(cfg)
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
    assert saw_partial, "geometry should produce a partial tail batch"


def test_prep_worker_does_not_pad_in_eager_mode():
    """Without torch.compile there is no constant-shape requirement, so the
    tail batch keeps its true row count instead of wasting compute and copy
    bandwidth on zero padding (matching the pre-compile behavior)."""
    store = _make_input_store((1, 1, 32, 32, 32))
    cfg = _prep_cfg(use_compile=False)
    prep_q = queue.Queue()
    PrepWorker(
        cfg, store, prep_q, cfg.patch, _default_preprocess(cfg)
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


def test_writer_raises_on_mismatched_output_channels_and_writers():
    cfg = InferenceConfig(devices=["cpu"])
    write_q: "queue.Queue[Optional[Preds]]" = queue.Queue()

    # Single writer, but model output has N=2 channels.
    worker = WriterWorker(
        cfg=cfg,
        outputs=[_spec(cfg, object())],
        write_q=write_q,
        preprocess=IdentityTransform(),
        full_shape=(2, 2, 2),
    )

    preds = Preds(
        block_idx=(0, 0, 0),
        block_bbox=(slice(0, 2), slice(0, 2), slice(0, 2)),
        linear_k=0,
        starts_in_block=[(0, 0, 0)],
        host_out=torch.zeros((1, 2, 2, 2, 2), dtype=torch.float32),
        valid_sizes=[(2, 2, 2)],
        transform_state=(0.0, 1.0),
        total_patches_in_block=1,
        acc_shape=(2, 2, 2),
        halo_left=(0, 0, 0),
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
        linear_k=0,
        starts_in_block=[(0, 0, 0)],
        host_out=host_out,
        valid_sizes=[(2, 2, 2)],
        transform_state=None,
        total_patches_in_block=1,
        acc_shape=(2, 2, 2),
        halo_left=(0, 0, 0),
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
        full_shape=(2, 2, 2),
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
    worker.cfg = InferenceConfig(devices=["cpu"], use_compile=True)
    worker.device = torch.device("cpu")
    worker.model = torch.nn.Identity()
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
