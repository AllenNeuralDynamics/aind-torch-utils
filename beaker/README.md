# Single-node Ray smoke run on Beaker

This is the deliberately small Beaker deployment path: one task replica reserves
8 GPUs and starts local Ray, which schedules eight shards with 1 GPU and 8 CPUs
each. TensorStore reads and writes S3 directly. Beaker mounts only the run
configuration/checkpoint at `/config` and captures shard metrics from `/results`.

## 1. Build and upload an immutable image

From the repository root:

```bash
docker build --platform linux/amd64 -f Dockerfile.beaker \
  -t aind-torch-utils-beaker:$(git rev-parse --short HEAD) .

docker run --rm aind-torch-utils-beaker:$(git rev-parse --short HEAD) \
  python -c "import importlib.metadata as m; import torch, ray, tensorstore; import aind_torch_utils.recipes; print(torch.__version__, ray.__version__, m.version('tensorstore'))"

beaker image create \
  --name "aind-torch-utils-$(git rev-parse --short HEAD)" \
  "aind-torch-utils-beaker:$(git rev-parse --short HEAD)"
```

Resolve the uploaded image ID and put that immutable ID in
`single-node-ray.yaml`; do not use the mutable image name for a smoke run.
The image contains the CUDA 12.8 runtime. The selected cluster's NVIDIA host
driver must support CUDA 12.8; verify this with the cluster owner or a short
`nvidia-smi` job before the paid run.

## 2. Create the run-assets dataset

Copy `smoke-config/` to a new staging directory, replace every angle-bracket
placeholder, and add the real checkpoint as `checkpoint.pth`. The input must be
a zero-origin 5D `T,C,Z,Y,X` Zarr with spatial shape `128×256×256`. With `64³`
blocks this produces 32 blocks, so all eight shards receive work.

```bash
cp -R beaker/smoke-config /tmp/aind-ray-smoke-assets
# Edit input.json and output.json, then:
cp /path/to/checkpoint.pth /tmp/aind-ray-smoke-assets/checkpoint.pth
beaker dataset create --name aind-ray-smoke-assets \
  /tmp/aind-ray-smoke-assets
```

Use a fresh, nonexistent S3 output prefix for every attempt. The output spec has
`delete_existing: true`; reusing a prefix destroys the previous output. Give the
AWS credentials read access to the input and write access only to that unique
output prefix.

Create Beaker secrets for `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and, when
using temporary credentials, `AWS_SESSION_TOKEN`. Replace the three secret-name
placeholders in the template. For long-lived credentials that have no session
token, remove the `AWS_SESSION_TOKEN` entry instead of creating an empty secret.
Set both region placeholders to the input/output bucket's region. Never put
credential values in the dataset, YAML, image, shell history, or Git.

## 3. Customize and submit

Replace the budget, cluster, immutable image ID, run-assets dataset ID, secret
names, and region in `single-node-ray.yaml`, then inspect the expanded file and
submit it:

```bash
beaker experiment create -f beaker/single-node-ray.yaml
beaker experiment logs --follow <EXPERIMENT_ID>
beaker experiment get <EXPERIMENT_ID>
```

The smoke template enables `--thread-dump-interval 1800` and disables Ray log
deduplication. While inference is running, every shard therefore writes all
Python thread stacks to its stderr log every five minutes. This is diagnostic
output, not a failure signal: healthy long-running shards also emit it. Set the
interval to `0` or remove the argument after investigating a stall.

The task reserves 8 GPUs, 64 CPUs, 512 GiB RAM, and 64 GiB shared memory. Ray
receives eight tasks at 1 GPU and 8 CPUs each, exactly consuming the advertised
GPU/CPU allocation. Each Ray worker sees its assigned physical GPU as logical
`cuda:0`, which is why `inference.json` contains only `devices: ["cuda:0"]`.
There is one task replica and no `--ray-address`, so Ray remains local to the
node.

The template uses `preemptible: false`, zero task retries, and a two-hour task
timeout. It deliberately does not set Beaker's newer `autoResume` field: current
Beaker APIs reject a context containing both legacy `preemptible` and
`autoResume`, and a non-preemptible task has nothing to auto-resume. If a
workspace requires the newer scheduling fields, replace `preemptible: false`
with the workspace-approved non-preemptible policy and keep `autoResume: false`;
do not combine the legacy and new fields.

## 4. Verify the smoke result

A successful result dataset contains exactly:

```text
metrics_shard0.json
metrics_shard1.json
...
metrics_shard7.json
```

Inspect all eight Ray task start/finish messages and the per-GPU utilization
graphs in Beaker to confirm every GPU participated. Download the result dataset
from the experiment page or with the dataset ID shown by `beaker experiment
get`.

Open the S3 output with TensorStore and verify:

- domain origin `(0, 0, 0, 0, 0)`;
- shape `(1, 1, 128, 256, 256)`;
- dtype `uint16`;
- exact value equality with a one-shard reference produced from the same input,
  checkpoint, workflow parameters, and inference config except
  `--num-shards 1`.

For example, after changing the two paths:

```bash
python - <<'PY'
import numpy as np
import tensorstore as ts

def open_zarr(path):
    return ts.open({"driver": "zarr", "kvstore": path}).result()

actual = open_zarr("s3://<bucket>/<eight-shard-output>/output.zarr")
reference = open_zarr("s3://<bucket>/<one-shard-output>/output.zarr")
assert tuple(actual.domain.inclusive_min) == (0, 0, 0, 0, 0)
assert tuple(actual.domain.shape) == (1, 1, 128, 256, 256)
assert str(actual.dtype) == "dtype(\"uint16\")"
np.testing.assert_array_equal(actual.read().result(), reference.read().result())
PY
```

## Zarr safety and failure checks

Concurrent shards are safe only when every spatial Zarr chunk dimension divides
the corresponding `block` dimension and the chunk grid aligns with the
zero-based block grid. The supplied `64³` chunks and blocks satisfy both rules.
The launcher checks declared metadata before a create/delete open, then checks
the effective opened TensorStore again before `ray.init()`.

To verify fail-fast behavior, make a copy of `output.json`, change one spatial
chunk from `64` to `96`, and run the image command with `--num-shards 8
--dry-run`. It must fail with an unsafe-layout error before touching S3 or
starting Ray.

There is no durable resume protocol in this milestone. A preemption, process
failure, or interrupt makes the task fail after cleanup; start the next attempt
with a new S3 output prefix. Multi-node Ray bootstrap, automatic retries/resume,
and a Python Beaker submitter are intentionally deferred.
