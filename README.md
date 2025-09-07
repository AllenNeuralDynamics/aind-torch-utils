# aind-torch-utils

Generic, queue-based, multi-GPU PyTorch inference pipeline for large volumetric (3D or 5D T,C,Z,Y,X) data.

## Features
- Block + patch based tiled volume processing with optional overlap / halo
- Multi-threaded CPU stages (prep / writer) feeding multi-GPU inference workers
- Seam handling via trimming or weighted blending
- Optional per-patch percentile normalization
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

> Configuration Note: The example above only sets a subset of available fields. For the complete list of parameters, validation rules, and detailed descriptions, open `src/aind_torch_utils/config.py` and review the `InferenceConfig` class doc/Field metadata.

## Monitoring
If `--metrics-json` provided:
- Queue depths over time (prep & writer queues)
- System metrics (CPU %, RAM, GPU (if implemented))
Use to diagnose stalls (e.g., GPU idle while prep queue empty => increase prep workers / decrease IO latency).

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