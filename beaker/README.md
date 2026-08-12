# Automated single-node Ray runs on Beaker

`aind-beaker-submit` turns one S3 Zarr URI into a complete denoising experiment.
It discovers input metadata, derives the TensorStore specs, estimates the
background offset from a coarse resolution, saves a reproducible local run
record, and submits the rendered YAML. The only Beaker dataset mounted at
`/config` is the model checkpoint.

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
`single-node-ray.yaml`; do not use a mutable image name. The image contains the
CUDA 13.0 runtime and the GPU-metrics wrapper at
`/opt/aind-torch-utils/beaker/run-with-gpu-metrics.sh`.

## 2. Create the checkpoint dataset once

Create a committed dataset containing one root-level model checkpoint. The
submitter discovers its filename automatically; use `--checkpoint-name` when a
dataset deliberately contains multiple checkpoints.

```bash
checkpoint_dir=$(mktemp -d)
cp /path/to/checkpoint.pth "$checkpoint_dir/model.pth"
beaker dataset create --name aind-denoise-checkpoint "$checkpoint_dir"
```

Put the immutable dataset ID in `single-node-ray.yaml`. Existing datasets with
additional legacy files remain usable during migration, but generated
experiments reference only the checkpoint.

Create Beaker secrets for `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and, when
using temporary credentials, `AWS_SESSION_TOKEN`. Replace the three secret-name
placeholders in the template. For long-lived credentials that have no session
token, remove the `AWS_SESSION_TOKEN` entry instead of creating an empty secret.
Set both region placeholders to the input/output bucket's region. Never put
credential values in the dataset, YAML, image, shell history, or Git.

## 3. Install and submit

Install the local calibration and rendering dependencies:

```bash
pip install -e '.[beaker]'
```

Configure the image, checkpoint dataset, budget, cluster, secret names, and AWS
region once in `single-node-ray.yaml`. Then submit with only the input Zarr root:

```bash
aind-beaker-submit s3://aind-open-data/path/to/fused.zarr
```

Metadata inspection and offset calibration run locally, so private inputs also
require ambient local AWS credentials (for example, an AWS profile or standard
AWS environment variables). Credential values are never copied into run
artifacts; the generated experiment retains only the Beaker secret references.

Input resolution defaults to `0` and background calibration defaults to level
`5`. Input Zarr v2 and v3 are auto-detected; output is Zarr v2 at level `0`.
The output defaults to a sibling `-denoised.zarr`, or can be overridden:

```bash
aind-beaker-submit INPUT \
  --output-uri s3://bucket/path/custom-denoised.zarr
```

Input chunks are copied to the output and must divide the configured inference
block. Use `--output-chunks Z Y X` when the source chunks are incompatible. If
the output already exists, submission stops before calibration; pass
`--resume-existing` to validate and reuse it. `--no-submit` renders without
creating an experiment.

Every invocation saves `input.json`, `output.json`, `workflow.json`,
`inference.json`, `offset-stats.json`, `experiment.yaml`, and the submission
response beneath `beaker/runs/`. It never creates a per-run Beaker dataset.
The generated task and experiment name is `denoised-<s3-prefix>`, where
`<s3-prefix>` is the input's top-level S3 key prefix; `--name` overrides it.

The base template enables `--thread-dump-interval 3600` and disables Ray log
deduplication. While inference is running, every shard therefore writes all
Python thread stacks to its stderr log every hour. This is diagnostic
output, not a failure signal: healthy long-running shards also emit it. Set the
interval to `0` or remove the argument after investigating a stall.

The task command runs the image's metrics wrapper, which starts one node-wide
`nvidia-smi` sampler before the Ray launcher and stops it when the launcher
exits. `GPU_METRICS_INTERVAL` controls the sampling interval and defaults to one
second. The wrapper preserves the launcher's exit status.

The task reserves 8 GPUs, 192 CPUs, 2000 GB RAM, and 64 GiB shared memory. Ray
receives eight tasks at 1 GPU and 24 CPUs each, exactly consuming the advertised
GPU/CPU allocation. Each Ray worker sees its assigned physical GPU as logical
`cuda:0`, which is why `inference.json` contains only `devices: ["cuda:0"]`.
There is one task replica and no `--ray-address`, so Ray remains local to the
node.

The template allows five task retries and has a 72-hour timeout. It deliberately
leaves `autoResume` unset because block-level S3 markers provide the inference
resume mechanism.

## 4. Verify the smoke result

A successful result dataset contains:

```text
gpu_metrics.csv
metrics_shard0.json
metrics_shard1.json
...
metrics_shard7.json
```

Inspect all eight Ray task start/finish messages and Beaker's resource graphs to
confirm every GPU participated. Download the result dataset from the experiment
page or with the dataset ID shown by `beaker experiment get`.

After downloading it, plot GPU utilization and VRAM with:

```bash
python benchmarking/plot_gpu_metrics.py \
  /path/to/gpu_metrics.csv \
  --out /path/to/gpu_metrics
```

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
