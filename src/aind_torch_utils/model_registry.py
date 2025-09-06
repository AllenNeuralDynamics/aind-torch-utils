from typing import Callable, Dict, Optional

from torch import nn


class ModelRegistry:
    """Registry for managing different model architectures and their loaders."""

    _registry: Dict[str, Callable[[Optional[str]], nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a model loader function.
        
        Parameters
        ----------
        name : str
            Name to register the model under.
            
        Returns
        -------
        Callable
            Decorator function.
            
        Example
        -------
        @ModelRegistry.register("unet")
        def load_unet(weights_path: Optional[str] = None) -> nn.Module:
            from aind_exaspim_image_compression.machine_learning.unet3d import UNet
            model = UNet()
            if weights_path:
                model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            return model
        """
        def decorator(func: Callable[[Optional[str]], nn.Module]) -> Callable:
            cls._registry[name] = func
            return func
        return decorator
    
    @classmethod
    def load_model(cls, model_type: str, weights_path: Optional[str] = None) -> nn.Module:
        """Load a model from the registry.
        
        Parameters
        ----------
        model_type : str
            Type of model to load (must be registered).
        weights_path : Optional[str]
            Path to model weights file.
            
        Returns
        -------
        nn.Module
            The loaded model.
            
        Raises
        ------
        KeyError
            If model_type is not registered.
        """
        if model_type not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Model type '{model_type}' not found in registry. "
                f"Available models: {available}"
            )
        
        return cls._registry[model_type](weights_path)
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered model types.
        
        Returns
        -------
        list
            List of registered model names.
        """
        return list(cls._registry.keys())
