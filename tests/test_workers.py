import queue
import threading
from types import SimpleNamespace
from unittest import mock

import torch

from aind_torch_utils import workers
from aind_torch_utils.workers import GpuWorker


def _fake_torch_backends():
    return SimpleNamespace(
        fp32_precision=None,
        cuda=SimpleNamespace(
            matmul=SimpleNamespace(fp32_precision=None),
        ),
        cudnn=SimpleNamespace(fp32_precision=None),
    )


def test_configure_cuda_tf32_enabled_uses_tf32_precision(monkeypatch):
    backends = _fake_torch_backends()
    monkeypatch.setattr(workers.torch, "backends", backends)

    workers._configure_cuda_tf32(use_tf32=True)

    assert backends.fp32_precision == "ieee"
    assert backends.cuda.matmul.fp32_precision == "tf32"
    assert backends.cudnn.fp32_precision == "tf32"


def test_configure_cuda_tf32_disabled_uses_ieee_precision(monkeypatch):
    backends = _fake_torch_backends()
    monkeypatch.setattr(workers.torch, "backends", backends)

    workers._configure_cuda_tf32(use_tf32=False)

    assert backends.fp32_precision == "ieee"
    assert backends.cuda.matmul.fp32_precision == "ieee"
    assert backends.cudnn.fp32_precision == "ieee"


def test_gpu_worker_run_does_not_set_cuda_device():
    prep_q = queue.Queue()
    prep_q.put(None)

    worker = object.__new__(GpuWorker)
    worker.cfg = SimpleNamespace(amp=False)
    worker.prep_q = prep_q
    worker.device = torch.device("cuda:0")

    with mock.patch("aind_torch_utils.workers.torch.cuda.set_device") as set_device:
        worker.run(threading.Event())

    set_device.assert_not_called()
