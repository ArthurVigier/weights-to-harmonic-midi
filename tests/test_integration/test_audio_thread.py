import torch
import torch.nn as nn
import time
import pytest
from unittest.mock import MagicMock
from src.core.bridge import AsyncAudioBridge

class SlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
    
    def forward(self, x):
        # Simulate a heavy, blocking computation
        time.sleep(0.5)
        return self.layer1(x)

def test_bridge_continues_during_blocking_model():
    """
    Simulates a blocking LLM task while verifying that the 
    AsyncAudioBridge continues to process the queue in its background thread.
    """
    model = SlowModel()
    hallucination_vector = torch.randn(20)
    
    # We use a very small update interval for the bridge loop
    bridge = AsyncAudioBridge(
        model=model,
        layer_path='layer1',
        hallucination_vector=hallucination_vector,
        update_interval=0.01
    )
    
    # Mock update_parameters to track calls
    bridge.audio_engine.update_parameters = MagicMock()
    
    with bridge:
        # Run a forward pass (this blocks the main thread for 0.5s)
        _ = model(torch.randn(1, 10))
        
        # After forward pass, let's wait a bit for the bridge thread to catch up
        time.sleep(0.1)
        
        # Even if model was blocking, the bridge should have updated the engine
        # at least once after the activation was captured.
        # (Note: Capture happens in forward_hook at the START of the forward pass
        # depending on the hook type, but for Linear it's usually at the end.
        # Let's ensure it was called.)
        assert bridge.audio_engine.update_parameters.called

def test_bridge_does_not_leak_on_stop():
    """Verify that stopping the bridge terminates the thread."""
    model = SlowModel()
    bridge = AsyncAudioBridge(
        model=model,
        layer_path='layer1',
        hallucination_vector=torch.randn(20)
    )
    
    bridge.start()
    assert bridge._thread.is_alive()
    
    bridge.stop()
    assert not bridge._thread.is_alive()
