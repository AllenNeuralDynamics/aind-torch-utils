CLI
===

Minimal Example
---------------

.. code-block:: bash

   python -m aind_torch_utils.run \
       --in-spec "data/in_spec.json" \
       --out-spec "data/out_spec.json" \
       --model-type denoise-net \
       --weights /path/to/weights.pth

Checkpoint-Aware Denoise Workflow
---------------------------------

Install the optional dependency and create a workflow-parameter file:

.. code-block:: bash

   pip install -e '.[denoise-net]'
   printf '%s\n' '{"checkpoint_path": "/data/model.pth", "offset": 73.5}' \
       > denoise-params.json

Then run the registered recipe:

.. code-block:: bash

   python -m aind_torch_utils.run \
       --in-spec "data/in_spec.json" \
       --out-spec "data/out_spec.json" \
       --workflow denoise-net \
       --workflow-params denoise-params.json

Workflow parameters may be supplied as either a JSON file path or an inline JSON
object. ``checkpoint_path`` is required. ``offset`` is optional; omit it to use the
checkpoint's saved intensity transform unchanged. When supplied, the source
package composes the offset around the trained transform without changing its
normalization denominator.

Output inversion is enabled by default, so this workflow writes count-space
predictions. Add ``--no-output-denormalize`` to intentionally write predictions
in the transformed domain. Generic ``--normalize``, ``--norm-lower``,
``--norm-upper``, and ``--clip-norm`` settings do not override the transform
provided by a workflow. The older ``--model-type denoise-net --weights ...``
form remains available as the legacy raw-model path and does not reconstruct
checkpoint transform metadata.


Resume Configuration
--------------------

Resumability is configured in the JSON passed with ``--config`` rather than by
separate CLI flags. Install ``.[aws]`` and set ``resume`` to true. A single S3
output can derive its marker root from the output TensorStore spec; multi-output
runs require an explicit shared ``resume_marker_prefix``. All output specs must
set ``delete_existing`` to false or omit it.

Completion markers are stored under
``.aind_torch_utils/resume/v2/<run-id>/t=<t>/c=<c>/z=<z>/y=<y>/x=<x>.done``.
Use ``resume_run_id`` to select a namespace explicitly. Otherwise the runtime
derives one from the input/output specs, workload, and output-affecting
configuration; changing the number or arrangement of Ray shards does not change
that identity.


Full Parameter Set (example)
----------------------------

.. code-block:: bash

   python -m aind_torch_utils.run \
       --in-spec '{"driver": "zarr", "kvstore": "s3://my-bucket/in.zarr/0"}' \
       --out-spec '{"driver":"zarr","kvstore":{"driver":"s3","bucket":"my-bucket","path":"out.zarr"},"path":"0","metadata":{"shape":[1,1,1024,1024,1024],"chunks":[1,1,256,256,256],"dtype":"<u2"},"create":true,"delete_existing":true}' \
       --model-type denoise-net \
       --weights /data/model.pth \
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


Options Overview
----------------

- Input/Output: ``--in-spec``, ``--out-spec`` (TensorStore JSON specs)
- Model/workflow: ``--model-type`` and ``--weights`` for a raw registered model,
  or ``--workflow`` and ``--workflow-params`` for a registered recipe
- Geometry: ``--t``, ``--c``, ``--patch``, ``--overlap``, ``--block``, ``--batch``
- Devices/Precision: ``--devices``, ``--no-amp``, ``--tf32``,
  ``--cudnn-benchmark``, ``--compile``, ``--compile-mode``,
  ``--no-compile-dynamic``
- Queues: ``--max-inflight-batches``, ``--prep-workers``, ``--writer-workers``
- Seam handling: ``--seam-mode {trim,blend}``, ``--trim-voxels``, ``--halo``,
  ``--min-blend-weight``
- Normalization: ``--normalize {percentile,global,false}``, ``--norm-lower``,
  ``--norm-upper``, ``--clip-norm [LO HI]``, ``--no-output-denormalize``
- Monitoring: ``--metrics-json``, ``--metrics-interval``
- Resume (config JSON): ``resume``, ``work_store``, ``resume_marker_prefix``,
  ``resume_run_id``

See ``src/aind_torch_utils/run.py`` for authoritative CLI definitions and
``src/aind_torch_utils/config.py`` for detailed field descriptions.
