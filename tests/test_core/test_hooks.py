import torch
import torch.nn as nn
import pytest
from src.core.hooks import HookManager

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def test_hook_manager_captures_activation():
    model = SimpleModel()
    manager = HookManager(model)
    manager.register_layer('layer1')

    # Create dummy input
    input_data = torch.randn(1, 10)
    
    # Run a forward pass
    _ = model(input_data)

    # Check if activation was captured
    activation = manager.get_activation('layer1')
    assert activation is not None
    assert activation.shape == (1, 20)
    assert not activation.is_cuda # Even if model was on CUDA, hook moves to CPU
    assert activation.grad_fn is None # Should be detached

def test_hook_manager_removes_hooks():
    model = SimpleModel()
    manager = HookManager(model)
    manager.register_layer('layer1')
    
    # Before removal, there should be a hook handle
    assert len(model.layer1._forward_hooks) == 1

    manager.remove_all()
    
    # After removal, hook handle should be gone
    assert len(model.layer1._forward_hooks) == 0

def test_hook_manager_context_manager():
    model = SimpleModel()
    with HookManager(model) as manager:
        manager.register_layer('layer1')
        assert len(model.layer1._forward_hooks) == 1
    
    # Context exit should remove hooks
    assert len(model.layer1._forward_hooks) == 0

def test_vram_leak_check():
    # This is a simplified check. In a real scenario, we'd check torch.cuda.memory_allocated()
    model = SimpleModel()
    manager = HookManager(model)
    manager.register_layer('layer1')

    input_data = torch.randn(1, 10)
    
    # Simulate many forward passes
    for _ in range(100):
        _ = model(input_data)
        activation = manager.get_activation('layer1')
        # If not detached/moved to CPU, this would keep the graph alive
        assert activation.grad_fn is None

    manager.remove_all()
