import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable

class ActivationHook:
    """
    A context-managed hook for extracting activations from a specific layer.
    Ensures activations are detached and moved to CPU to prevent VRAM leakage.
    """
    def __init__(self, module: nn.Module, callback: Callable[[torch.Tensor], None]):
        self.module = module
        self.callback = callback
        self.handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        # Handle cases where output might be a tuple (e.g., in some transformer layers)
        if isinstance(output, tuple):
            output = output[0]
        
        # Detach and move to CPU to avoid VRAM growth and graph retention
        activation = output.detach().cpu()
        self.callback(activation)

    def remove(self):
        self.handle.remove()

class HookManager:
    """
    Manages multiple activation hooks across a model.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: Dict[str, ActivationHook] = {}
        self._latest_activations: Dict[str, torch.Tensor] = {}

    def register_layer(self, layer_path: str):
        """
        Registers a hook for a layer specified by its named path (e.g., 'model.layers.12').
        """
        if layer_path in self.hooks:
            return

        try:
            module = dict(self.model.named_modules())[layer_path]
        except KeyError:
            raise ValueError(f"Layer path '{layer_path}' not found in model.")

        def callback(activation: torch.Tensor):
            self._latest_activations[layer_path] = activation

        self.hooks[layer_path] = ActivationHook(module, callback)

    def get_activation(self, layer_path: str) -> Optional[torch.Tensor]:
        return self._latest_activations.get(layer_path)

    def remove_all(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self._latest_activations.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()
