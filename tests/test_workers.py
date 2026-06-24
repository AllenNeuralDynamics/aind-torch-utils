import queue
import threading
from typing import Optional

import pytest
import torch

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.workers import Preds, WriterWorker


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
