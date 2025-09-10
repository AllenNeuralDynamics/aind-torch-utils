import unittest.mock

import numpy as np
import pytest
import tensorstore as ts
import torch
from torch import nn

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.run import run


class DummyModel(nn.Module):
    """A dummy model that returns its input."""

    def forward(self, x):
        return x


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


@pytest.fixture
def dummy_data(temp_dir):
    """Create dummy data for testing."""
    shape = (1, 1, 32, 32, 32)
    dtype = "<u2"
    input_path = temp_dir / "input.zarr"
    output_path = temp_dir / "output.zarr"

    # Create dummy input data
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(input_path)},
        "metadata": {
            "shape": shape,
            "chunks": (1, 1, 16, 16, 16),
            "dtype": dtype,
        },
    }
    data = ts.open(spec, create=True).result()
    # Min value 1 to identify gaps in output due to incorrect stitching
    arr = np.random.randint(1, 65535, size=shape, dtype=np.uint16)
    data.write(arr).result()

    # Prepare output spec
    output_spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(output_path)},
        "metadata": {
            "shape": shape,
            "chunks": (1, 1, 16, 16, 16),
            "dtype": dtype,
        },
    }
    ts.open(output_spec, create=True).result()

    return ts.open(spec).result(), ts.open(output_spec).result()


def test_run_pipeline(temp_dir, dummy_data):
    """
    Test the run function with a dummy model and data.
    This is an integration test for the pipeline.
    """
    input_store, output_store = dummy_data
    metrics_json = temp_dir / "metrics.json"

    # Use CPU for testing if no CUDA devices are available
    devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]

    # If no CUDA devices, we need to patch GpuWorker to not use cuda streams
    if not torch.cuda.is_available():
        with unittest.mock.patch("aind_torch_utils.run.GpuWorker") as mock_gpu_worker:
            # The mock needs to have a run method that can be called in a thread
            mock_gpu_worker.return_value.run.side_effect = lambda stop_event: None
            _run_test_logic(
                input_store, output_store, metrics_json, devices, DummyModel()
            )
    else:
        _run_test_logic(input_store, output_store, metrics_json, devices, DummyModel())

    # Assertions
    assert metrics_json.exists()

    # Verify output data is same as input for DummyModel
    input_data = input_store.read().result()
    output_data = output_store.read().result()

    # The mock GpuWorker does not process data, so we only check for equality if cuda is available
    if torch.cuda.is_available():
        np.testing.assert_array_equal(input_data, output_data)
        # assert all values are > 0 to check for gaps during stitching
        assert np.all(output_data > 0)
    else:
        # If no cuda, output will be empty
        assert np.all(output_data == 0)


def _run_test_logic(input_store, output_store, metrics_json, devices, model):
    """Helper function to run the test logic."""
    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        seam_mode="trim",
        block=(32, 32, 32),
        batch_size=4,
        t_idx=0,
        c_idx=0,
        devices=devices,
        amp=False,
        max_inflight_batches=10,
        normalize=False,  # so we get the same data out as we put in
    )

    run(
        model=model,
        input_store=input_store,
        output_store=output_store,
        cfg=cfg,
        metrics_json=str(metrics_json),
        metrics_interval=0.1,
        num_prep_workers=1,
        num_writer_workers=1,
    )
