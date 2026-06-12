import queue
import threading

import numpy as np
import tensorstore as ts
import torch

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.workers import PrepWorker


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


def test_prep_worker_pads_tail_batch_to_constant_shape():
    """Every batch must have exactly batch_size rows so compiled models
    never see a varying input shape; padded rows must be zero."""
    store = _make_input_store((1, 1, 32, 32, 32))
    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        seam_mode="trim",
        block=(32, 32, 32),
        batch_size=4,
        devices=["cpu"],
        amp=False,
        normalize=False,
    )
    prep_q = queue.Queue()
    worker = PrepWorker(cfg, store, prep_q, cfg.patch)
    worker.run(threading.Event())

    batches = []
    while not prep_q.empty():
        batches.append(prep_q.get_nowait())
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
