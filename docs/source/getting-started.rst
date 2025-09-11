Getting Started
===============

Installation
------------

User (editable):

.. code-block:: bash

   pip install -e .

Development (linting, docs, tests):

.. code-block:: bash

   python3 -m pip install --upgrade pip
   pip install -e . --group dev

Optional extras (e.g., UNet dependency):

.. code-block:: bash

   pip install -e .[denoise-net]


Minimal Programmatic Usage
--------------------------

.. code-block:: python

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
       "kvstore": {"driver": "s3", "bucket": "bucket", "path": "ns/out.zarr"},
       "path": "0",
       "metadata": {
           "shape": [1, 1, 256, 1024, 1024],
           "chunks": [1, 1, 256, 256, 256],
           "dtype": "<u2",
       },
       "create": True,
       "delete_existing": True,
   }

   input_store = open_ts_spec(in_spec)
   output_store = open_ts_spec(out_spec)

   cfg = InferenceConfig(
       patch=(64, 64, 64),
       overlap=10,
       block=(256, 256, 256),
       batch_size=16,
       devices=["cuda:0", "cuda:1"],
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

Configuration note: for the complete list of parameters, see
``src/aind_torch_utils/config.py`` (``InferenceConfig`` field metadata).

