import os
from typing import Optional

import torch
from torch import nn

from aind_torch_utils.model_registry import ModelRegistry


# Register the UNet model
@ModelRegistry.register("denoise-net")
def load_unet(weights_path: Optional[str] = None) -> nn.Module:
    """Load UNet model with optional weights.
    
    The import is done inside the function to avoid requiring the dependency
    unless this specific model is being used.
    """
    from aind_exaspim_image_compression.machine_learning.unet3d import UNet

    model = UNet()
    if weights_path and os.path.exists(weights_path):
        sd = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(sd)
    return model
