Single-node Ray on Beaker
=========================

The supported Beaker smoke path uses one 8-GPU node and eight local Ray shards.
It reads and writes TensorStore Zarr arrays in S3, mounts configuration and a
checkpoint at ``/config``, and writes per-shard metrics under ``/results``.

Deployment files
----------------

- ``Dockerfile.beaker`` pins the PyTorch 2.13 / CUDA 13.0 runtime image by digest.
- ``beaker/single-node-ray.yaml`` is the Beaker v2 experiment template.
- ``beaker/smoke-config/`` contains copyable input, output, inference, and
  workflow JSON examples.
- ``beaker/README.md`` is the complete build, upload, secret, submission,
  verification, and failure-check runbook.

The launcher must be invoked without ``--ray-address`` for this path. It creates
local Ray and schedules eight tasks at one GPU and eight CPUs apiece. Ray's GPU
isolation maps each task's assigned physical GPU to logical ``cuda:0``.

Output safety
-------------

For a multi-shard Zarr output, every spatial chunk dimension must divide the
corresponding inference block dimension, and the chunk grid must align with the
zero-based block grid. The launcher rejects unsafe or indeterminate layouts
before Ray starts. When output metadata is declared in the spec, validation
happens before any destructive create/delete open.

See the repository's ``beaker/README.md`` before submitting a run. For retries,
enable S3 marker resume in the inference config, keep ``delete_existing`` false,
and reuse the same output and marker prefixes. Completion markers provide
restart safety; they do not coordinate independent concurrent launches.
