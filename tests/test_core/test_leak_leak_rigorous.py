import torch
import torch.nn as nn
import gc
from src.core.hooks import HookManager

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Large layer to make any leakage more obvious
        self.layer1 = nn.Linear(1000, 1000)

    def forward(self, x):
        return self.layer1(x)

def test_no_graph_retention_over_time():
    """
    Verifies that running many iterations does not accumulate memory
    by ensuring each activation is detached and old ones are cleared.
    """
    model = LargeModel()
    manager = HookManager(model)
    manager.register_layer('layer1')

    # Warm up
    input_data = torch.randn(1, 1000, requires_grad=True)
    
    initial_mem = 0
    # We can't easily check VRAM on CPU, but we can check if the graph is kept alive
    
    for i in range(100):
        # Forward pass
        output = model(input_data)
        # Calculate loss and backward to create a graph
        loss = output.sum()
        loss.backward(retain_graph=True) # Intentionally retain graph to see if hook avoids it
        
        activation = manager.get_activation('layer1')
        assert activation is not None
        assert activation.grad_fn is None
        
        # Manually clear output and loss
        del output
        del loss
        # The hook's callback should overwrite the old activation
        # and since it's detached, it shouldn't hold the graph.

    manager.remove_all()
    gc.collect()

def test_hook_removes_from_module():
    """
    Ensures that when a hook is removed, the module's _forward_hooks dict is actually empty.
    """
    model = LargeModel()
    manager = HookManager(model)
    manager.register_layer('layer1')
    
    assert len(model.layer1._forward_hooks) == 1
    manager.remove_all()
    assert len(model.layer1._forward_hooks) == 0
