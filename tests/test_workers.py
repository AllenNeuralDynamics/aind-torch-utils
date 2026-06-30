import queue
import threading
from typing import Optional

import numpy as np
import pytest
import tensorstore as ts
import torch

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.workers import GpuWorker, Preds, PrepWorker, WriterWorker


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


def test_prep_worker_pads_tail_batch_to_constant_shape_when_compiling():
    """With torch.compile, every batch must have exactly batch_size rows so
    the compiled model never sees a varying input shape; padded rows must be
    zero."""
    store = _make_input_store((1, 1, 32, 32, 32))
    cfg = _prep_cfg(use_compile=True)
    prep_q = queue.Queue()
    PrepWorker(cfg, store, prep_q, cfg.patch).run(threading.Event())

    batches = _drain(prep_q)
    assert batches

    saw_partial = False
    total_real = 0
    for b in batches:
        assert b.host_in.shape == (cfg.batch_size, 1, *cfg.patch)
        n_real = len(b.starts_in_block)
        assert 0 < n_real <= cfg.batch_size
        assert len(b.valid_sizes) == n_real
        assert len(b.per_block_minmax) == n_real
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
    PrepWorker(cfg, store, prep_q, cfg.patch).run(threading.Event())

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
    worker = WriterWorker(cfg=cfg, writers=[object()], write_q=write_q)

    preds = Preds(
        block_idx=(0, 0, 0),
        block_bbox=(slice(0, 2), slice(0, 2), slice(0, 2)),
        linear_k=0,
        starts_in_block=[(0, 0, 0)],
        host_out=torch.zeros((1, 2, 2, 2, 2), dtype=torch.float32),
        valid_sizes=[(2, 2, 2)],
        per_block_minmax=[(0.0, 1.0)],
        total_patches_in_block=1,
        acc_shape=(2, 2, 2),
        halo_left=(0, 0, 0),
        ready_event=None,
    )

    write_q.put(preds)

    with pytest.raises(ValueError, match="Mismatch between model output channels"):
        worker.run(stop_event=threading.Event())


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
