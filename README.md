[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-74.2%25-red)
![Coverage](https://img.shields.io/badge/coverage-18%25-red)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

<p align="left">
  <img src="assets/logo.png" alt="Project Logo" width="200">
</p>

Generic, queue-based, multi-GPU PyTorch inference pipeline for large volumetric (3D or 5D T,C,Z,Y,X) data.

## Features
- Block + patch based tiled volume processing with optional overlap / halo
- Multi-threaded CPU stages (prep / writer) feeding multi-GPU inference workers
- Seam handling via trimming or weighted blending
- Optional global or per-block percentile normalization
- Queue & system monitoring (JSON metrics export)
- Model registry for plug‑and‑play custom architectures
- Optional AMP, TF32, and `torch.compile` acceleration
- Backed by `tensorstore` for flexible IO backends

## Installation
User (editable):
```bash
pip install -e .
```
Development (linting, docs, tests):
```bash
python3 -m pip install --upgrade pip
pip install -e . --group dev
```
Optional extras (e.g. UNet dependency):
```bash
pip install -e .[denoise-net]
```
S3-backed resumability:
```bash
pip install -e .[aws]
```

## CLI Example (Full Parameter Set)
```bash
python -m aind_torch_utils.run \
    --in-spec '{"driver": "zarr", "kvstore": "s3://my-bucket/in.zarr/0"}' \
    --out-spec '{"driver":"zarr","kvstore":{"driver":"s3","bucket":"my-bucket","path":"out.zarr"},"path":"0","metadata":{"shape":[1,1,1024,1024,1024],"chunks":[1,1,256,256,256],"dtype":"<u2"},"create":true,"delete_existing":true}' \
    --model-type denoise-net \
    --weights /data/BM4DNet-20250905-169-0.0073.pth \
    --t 0 --c 0 \
    --patch 64 64 64 \
    --overlap 12 \
    --block 256 256 256 \
    --batch 64 \
    --devices cuda:0 cuda:1 \
    --seam-mode trim \
    --trim-voxels 6 \
    --halo 8 \
    --max-inflight-batches 64 \
    --norm-lower 0.5 --norm-upper 99.9 \
    --min-blend-weight 0.05 \
    --prep-workers 4 \
    --writer-workers 4 \
    --metrics-json metrics.json \
    --metrics-interval 0.5
```

## Minimal CLI
```bash
python -m aind_torch_utils.run \
    --in-spec "data/in_spec.json" \
    --out-spec "data/out_spec.json" \
    --model-type denoise-net \
    --weights /data/BM4DNet-20250905-169-0.0073.pth
```

## Checkpoint-Aware `denoise-net` Workflow

For checkpoints that contain both the U-Net configuration and its serialized
intensity transform, put the workflow parameters in `denoise-params.json`:

```json
{"checkpoint_path": "/data/model.pth", "offset": 73.5}
```

Then use the registered workflow:

```bash
python -m aind_torch_utils.run \
    --in-spec "data/in_spec.json" \
    --out-spec "data/out_spec.json" \
    --workflow denoise-net \
    --workflow-params denoise-params.json
```

`checkpoint_path` is required. `offset` is optional; omit it to use the saved
checkpoint transform unchanged. The default output inversion restores count-space
predictions. Use `--no-output-denormalize` only when transformed-space output is
intentional. Generic `--normalize` settings do not override a workflow-provided
transform. `--model-type denoise-net --weights ...` remains the legacy raw-model
path.

## Resuming Interrupted Runs

Block-level resumability uses small S3 sidecar markers. Put the following in
the inference JSON passed with `--config`:

```json
{
  "resume": true,
  "work_store": "s3-markers",
  "resume_marker_prefix": "s3://my-bucket/checkpoints/my-output"
}
```

For a single S3 output, `resume_marker_prefix` may be omitted and the marker
root is derived from the output kvstore. Multi-output runs must provide one
explicit shared prefix. The runtime skips marked blocks before reading their
input and writes a marker only after every asynchronous output write for the
block commits.

Do not set `delete_existing: true` on any output spec when resuming; the CLI
and Ray launcher reject that combination before opening the output. Markers use
the versioned path
`.aind_torch_utils/resume/v2/<run-id>/t=<t>/c=<c>/z=<z>/y=<y>/x=<x>.done`.
Set `resume_run_id` to choose the namespace explicitly; otherwise it is derived
from the workload, TensorStore specs, and output-affecting configuration.

## Programmatic Usage
```python
from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.run import run
from aind_torch_utils.model_registry import ModelRegistry
from aind_torch_utils.utils import open_ts_spec

in_spec = {
    "driver": "zarr",
    "kvstore": "s3://bucket/path/input.zarr/0",
}

out_spec = {
    "driver": "zarr",
    "kvstore": {
        "driver": "s3",
        "bucket": "bucket",
        "path": "ns/out.zarr"
    },
    "path": "0",
    "metadata": {
        "shape": [1, 1, 256, 1024, 1024],
        "chunks": [1, 1, 256, 256, 256],
        "dtype": "<u2"
    },
    "create": True,
    "delete_existing": True
}

input_store = open_ts_spec(in_spec)
output_store = open_ts_spec(out_spec)

cfg = InferenceConfig(
    patch=(64,64,64),
    overlap=10,
    block=(256,256,256),
    batch_size=16,
    devices=["cuda:0","cuda:1"],
    seam_mode="trim",
    trim_voxels=5,
)

model = ModelRegistry.load_model("denoise-net", weights_path="weights.pth")

run(
    model=model,
    input_store=input_store,
    output_store=output_store,
    cfg=cfg,
    metrics_json="metrics.json",
    num_prep_workers=4,
    num_writer_workers=2,
)
```

The checkpoint-aware equivalent is:

```python
import aind_torch_utils.recipes  # register bundled recipes
from aind_torch_utils.run import run_workflow
from aind_torch_utils.workflow import WorkflowRegistry

workflow = WorkflowRegistry.build(
    "denoise-net",
    {"checkpoint_path": "/data/model.pth", "offset": 73.5},
)
run_workflow(workflow, input_store, output_store, cfg)
```

Leave out `offset` to preserve the checkpoint mapping. The workflow leaves output
specification and execution policy unset, so `InferenceConfig` continues to control
output inversion, AMP, compilation, seam handling, devices, and stores.

> Configuration Note: The example above only sets a subset of available fields. For the complete list of parameters, validation rules, and detailed descriptions, open `src/aind_torch_utils/config.py` and review the `InferenceConfig` class doc/Field metadata.

## Monitoring
If `--metrics-json` provided:
- Queue depths over time (prep & writer queues)
- System metrics (CPU %, RAM, GPU (if implemented))
Use to diagnose stalls (e.g., GPU idle while prep queue empty => increase prep workers / decrease IO latency).

## Beaker

The production-shaped smoke deployment is one 8-GPU Beaker node running eight
local Ray shards against S3 TensorStores. After configuring the checked-in base
profile once, generate and submit a run directly from an input Zarr:

```bash
pip install -e '.[beaker]'
aind-beaker-submit s3://aind-open-data/path/to/fused.zarr
```

The submitter derives both TensorStore specs, computes the background offset,
and persists the rendered experiment without creating a per-run dataset. See
[`beaker/README.md`](beaker/README.md) for the immutable image build, checkpoint
dataset, secrets, v2 experiment template, safety constraints, and verification
procedure.

## Custom Model Registration
Add your model directly to `src/aind_torch_utils/models.py` so it is automatically available when the package is imported.

Steps:
1. Open `src/aind_torch_utils/models.py`.
2. Define your `nn.Module` subclass.
3. Add a loader function that returns an instance (optionally loading weights) and decorate it with `@ModelRegistry.register("your-name")`.

Snippet (showing how to append under the existing UNet registration):
```python
# ... existing imports and UNet registration ...

from typing import Optional
import torch
from torch import nn


class MySimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 3, padding=1)

    def forward(self, x):  # x: (N, C=1, Z, Y, X)
        return self.conv(x)


@ModelRegistry.register("simple-net")
def load_simple_net(weights_path: Optional[str] = None) -> nn.Module:
    model = MySimpleNet()
    if weights_path:
        sd = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(sd)
    return model
```

You can now run:
```bash
python -m aind_torch_utils.run --model-type simple-net ...
```

## Listing Available Models
From Python:
```python
from aind_torch_utils.model_registry import ModelRegistry
print(ModelRegistry.list_models())
```

## Contributing
- Run formatting: `black . && isort .`
- Run tests: `pytest -q`
- Add docstrings; keep public API minimal.

## License
MIT
