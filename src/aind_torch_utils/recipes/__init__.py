"""Workflow recipes.

Each module here owns one workflow's math: its processor, preprocessing/correction,
per-output specs, and execution policy, registered via
``@WorkflowRegistry.register(...)``. Importing this package registers the recipes it
contains, mirroring how :mod:`aind_torch_utils.models` registers models on import.

This is the home for workflow-specific code that must not live in the generic core
(issue #25 §4.6/§4.7). Optional recipe dependencies are imported lazily by each
builder, so importing this package only performs registration.
"""

from aind_torch_utils.recipes import denoise_net  # noqa: F401
