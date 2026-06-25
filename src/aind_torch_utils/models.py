"""
Module defining PyTorch models and related utilities.
"""
import os
from typing import List, Optional, Tuple

import torch
from torch import nn

from aind_torch_utils.model_registry import ModelRegistry


class SharedEncoderModel(nn.Module):
    """Wraps a shared encoder and N independent decoders.

    This class supports two encoder/decoder interfaces:
    1) Proteomics-style encoder returning tuple outputs consumed by decoder
        heads via ``dec(x=..., latent=..., hidden_states_out=...)``.
    2) Generic encoder whose output is passed directly to each decoder as
        ``dec(features)``.

    Outputs are stacked along dim 1, producing shape (B, N, Z, Y, X).

    Parameters
    ----------
    encoder : nn.Module
        Encoder module. Its forward() output is passed directly to each decoder.
        May return a tensor or any structure (tuple of tensors for skip
        connections), as long as each decoder accepts that same structure.
    decoders : list of nn.Module
        N decoder modules.
    recover_layers : tuple of int
        Encoder layer indices used by proteomics-style encoders.
    apply_sigmoid : bool
        If True, apply sigmoid (clamped to [0.01, 0.99]) to all outputs.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoders: List[nn.Module],
        recover_layers: Tuple[int, int] = (2, 5),
        apply_sigmoid: bool = False,
    ):
        """Initialize the SharedEncoderModel.

        Parameters
        ----------
        encoder : nn.Module
            The shared encoder module.
        decoders : list of nn.Module
            A list of N decoder modules.
        recover_layers : tuple of int
            Encoder layer indices for skip connections in proteomics-style
            encoder/decoder architectures.
        apply_sigmoid : bool
            Whether to apply sigmoid activation to outputs.
        """
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.recover_layers = recover_layers
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the shared encoder and all decoders.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, Z, Y, X).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, Z, Y, X),
            where N is the number of decoders.
        """
        try:
            enc_out = self.encoder(
                x=x,
                mask_ratio=0.0,
                recover_layers=self.recover_layers,
            )
            if isinstance(enc_out, tuple) and len(enc_out) >= 5:
                latent = enc_out[0]
                feature_maps = enc_out[4]
                outputs = [
                    dec(x=x, latent=latent, hidden_states_out=feature_maps)
                    for dec in self.decoders
                ]
            else:
                outputs = [dec(enc_out) for dec in self.decoders]
        except TypeError:
            features = self.encoder(x)
            outputs = [dec(features) for dec in self.decoders]

        # Each output: (B, 1, Z, Y, X) -> stack -> (B, N, 1, Z, Y, X) -> squeeze
        outputs = torch.stack(outputs, dim=1).squeeze(2)  # (B, N, Z, Y, X)
        if self.apply_sigmoid:
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


# Register the GPU GFP-masking "model"
@ModelRegistry.register("gfp-mask")
def load_gfp_mask(weights_path: Optional[str] = None) -> nn.Module:
    """Load the GPU GFP-masking model.

    The import is done inside the function so the GPU stack (``cupy``) is only
    required when this model is actually selected, keeping the package importable
    on hosts without a GPU. The model has no learnable weights, so
    ``weights_path`` is instead treated as a path to a JSON file of masking
    parameters (see :meth:`GfpMaskModel.from_json`); if omitted, defaults are used.
    """
    from aind_torch_utils.postprocessing import GfpMaskModel

    if weights_path:
        return GfpMaskModel.from_json(weights_path)
    return GfpMaskModel()
