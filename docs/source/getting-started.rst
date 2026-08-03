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

Install the AWS extra to use S3 completion markers:

.. code-block:: bash

   pip install -e .[aws]


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


Checkpoint-Aware Denoise Workflow
---------------------------------

The ``denoise-net`` workflow loads the U-Net and serialized intensity transform
together through the source package's checkpoint loader:

.. code-block:: python

   import aind_torch_utils.recipes  # registers bundled workflow recipes
   from aind_torch_utils.config import InferenceConfig
   from aind_torch_utils.run import run_workflow
   from aind_torch_utils.workflow import WorkflowRegistry

   workflow = WorkflowRegistry.build(
       "denoise-net",
       {
           "checkpoint_path": "/data/model.pth",
           "offset": 73.5,  # optional; omit to retain the checkpoint mapping
       },
   )
   cfg = InferenceConfig(devices=["cuda:0"])
   run_workflow(workflow, input_store, output_store, cfg)

The source checkpoint loader runs on CPU; the inference runtime copies and moves
the model to ``cfg.devices``. Outputs, execution policy, compilation, AMP, seam
handling, and storage remain controlled by ``InferenceConfig``. In particular,
``output_denormalize=True`` (the default) restores count-space output, while
``output_denormalize=False`` writes transformed-space predictions.

Because this workflow provides its own transform, the generic ``normalize`` and
related normalization fields do not override it. The legacy
``ModelRegistry.load_model("denoise-net", ...)`` path remains unchanged and loads
only the raw model/weights.


Resuming Interrupted Runs
-------------------------

Enable block-level resumability in the inference JSON supplied to the CLI or
Ray launcher:

.. code-block:: json

   {
     "resume": true,
     "work_store": "s3-markers",
     "resume_marker_prefix": "s3://my-bucket/checkpoints/my-output"
   }

The marker prefix is optional for a single S3 output and required when a model
has multiple outputs. Never combine resume mode with ``delete_existing: true``;
the output is validated before it is opened. A block is marked complete only
after all of its output writes commit, so an interrupted or partially written
block is processed again on the next run.
