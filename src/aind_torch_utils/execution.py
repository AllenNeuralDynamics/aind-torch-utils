"""Execution policy for the GPU processor stage.

The processor (the ``nn.Module`` slot in :class:`~aind_torch_utils.workers.GpuWorker`)
already lets a workflow run arbitrary tensor-in/tensor-out GPU code. What was still
global was *how* it runs: AMP, ``torch.compile``, input dtype, memory format. Those are
processor constraints, not runtime knobs, so they belong on an ``ExecutionPolicy``
attached to the processor/workflow rather than on ``InferenceConfig`` (issue #25 §3.1).

The default is synthesized from the legacy config fields, so a run that does not inject
a policy behaves exactly as before (AMP on by default).
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ExecutionPolicy:
    """How the GPU processor is executed.

    Attributes
    ----------
    input_dtype : torch.dtype
        Host dtype the prep stage allocates patches in (what the model receives).
    autocast : bool
        Wrap the forward in ``torch.autocast('cuda', float16)``.
    inference_mode : bool
        Wrap the forward in ``torch.inference_mode()``.
    compile : bool
        Compile the processor with ``torch.compile``.
    compile_mode : str
        ``torch.compile`` mode (e.g. "default", "reduce-overhead").
    compile_dynamic : Optional[bool]
        ``torch.compile`` dynamic-shapes setting (None = auto).
    channels_last : bool
        Run the processor in channels-last (3D) memory format.
    """

    input_dtype: torch.dtype = torch.float32
    autocast: bool = False
    inference_mode: bool = True
    compile: bool = False
    compile_mode: str = "default"
    compile_dynamic: Optional[bool] = None
    channels_last: bool = False

    @classmethod
    def from_config(
        cls,
        amp: bool,
        use_compile: bool,
        compile_mode: str,
        compile_dynamic: Optional[bool],
    ) -> "ExecutionPolicy":
        """Synthesize the default policy from the legacy config fields.

        AMP drives both autocast and the float16 host dtype, matching the old prep +
        GpuWorker behavior exactly.
        """
        return cls(
            input_dtype=torch.float16 if amp else torch.float32,
            autocast=amp,
            inference_mode=True,
            compile=use_compile,
            compile_mode=compile_mode,
            compile_dynamic=compile_dynamic,
            channels_last=False,
        )
