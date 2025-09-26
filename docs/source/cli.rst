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
- Model: ``--model-type`` (registry key), ``--weights`` (optional path)
- Geometry: ``--t``, ``--c``, ``--patch``, ``--overlap``, ``--block``, ``--batch``
- Devices/Precision: ``--devices``, ``--no-amp``, ``--no-tf32``, ``--compile``,
  ``--compile-mode``, ``--no-compile-dynamic``
- Queues: ``--max-inflight-batches``, ``--prep-workers``, ``--writer-workers``
- Seam handling: ``--seam-mode {trim,blend}``, ``--trim-voxels``, ``--halo``,
  ``--min-blend-weight``
- Normalization: ``--normalize {percentile,global,false}``, ``--norm-lower``,
  ``--norm-upper``, ``--clip-norm [LO HI]``
- Monitoring: ``--metrics-json``, ``--metrics-interval``

See ``src/aind_torch_utils/run.py`` for authoritative CLI definitions and
``src/aind_torch_utils/config.py`` for detailed field descriptions.

