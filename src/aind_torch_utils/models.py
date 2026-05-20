"""
Module defining PyTorch models and related utilities.
"""
import os
from typing import List, Optional

import torch
from torch import nn

from aind_torch_utils.model_registry import ModelRegistry


class SharedEncoderModel(nn.Module):
    """Wraps a shared encoder and N independent decoders.

    The encoder runs once per batch; all decoders receive the same features.
    Outputs are stacked along dim 1, producing shape (B, N, *spatial).

    Parameters
    ----------
    encoder : nn.Module
        Encoder module. Its forward() output is passed directly to each decoder.
        May return a tensor or any structure (tuple of tensors for skip
        connections), as long as each decoder accepts that same structure.
    decoders : list of nn.Module
        N decoder modules. Each must accept the encoder output and return a
        tensor of shape (B, 1, Z, Y, X).
    """

    def __init__(self, encoder: nn.Module, decoders: List[nn.Module]):
        """Initialize the SharedEncoderModel.

        Parameters
        ----------
        encoder : nn.Module
            The shared encoder module.
        decoders : list of nn.Module
            A list of N decoder modules. Each should accept the encoder's output
            and produce a tensor of shape (B, 1, Z, Y, X).
        """
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)

    def forward(
        self,
        x: torch.Tensor,
        apply_sigmoid: Optional[bool] = False,
    ) -> torch.Tensor:
        """Forward pass through the shared encoder and all decoders.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, Z, Y, X).
        apply_sigmoid : bool, optional
            Whether to apply a sigmoid activation to the outputs. Default is False.
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, Z, Y, X),
            where N is the number of decoders.
        """
        features = self.encoder(x)
        outputs = [dec(features) for dec in self.decoders]
        # Each output: (B, 1, Z, Y, X) -> stack -> (B, N, 1, Z, Y, X) -> squeeze
        outputs = torch.stack(outputs, dim=1).squeeze(2)  # (B, N, Z, Y, X)
        if apply_sigmoid:
            outputs = torch.clamp(torch.sigmoid(outputs), min=0.01, max=0.99)
        return outputs


# Register the UNet model
@ModelRegistry.register("denoise-net")
def load_unet(weights_path: Optional[str] = None) -> nn.Module:
    """Load UNet model with optional weights.

    The import is done inside the function to avoid requiring the dependency
    unless this specific model is being used.
    """
    from aind_exaspim_image_compression.machine_learning.unet3d import UNet

    model = UNet()
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights file not found: '{weights_path}'. "
                "Provide a valid path or omit 'weights_path' to load an uninitialized model."
            )
        sd = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(sd)
    return model
