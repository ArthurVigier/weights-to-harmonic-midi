import torch
import torch.nn as nn
import time
from unittest.mock import MagicMock
from src.core.bridge import AudioBridge

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
    def forward(self, x):
        return self.layer1(x)

def test_bridge_step_updates_audio_engine():
    model = SimpleModel()
    hallucination_vector = torch.randn(20)
    
    # Initialize bridge with a small update interval for testing
    bridge = AudioBridge(
        model=model,
        layer_path='layer1',
        hallucination_vector=hallucination_vector,
        update_interval=0.0
    )
    
    # Mock the audio engine's update_parameters method
    bridge.audio_engine.update_parameters = MagicMock()
    
    # Run a forward pass to populate activation
    input_data = torch.randn(1, 10)
    _ = model(input_data)
    
    # Run bridge step
    bridge.step()
    
    # Verify update_parameters was called
    assert bridge.audio_engine.update_parameters.called
    args, _ = bridge.audio_engine.update_parameters.call_args
    freq, score = args
    assert isinstance(freq, float)
    assert 0.0 <= score <= 1.0

def test_bridge_rate_limiting():
    model = SimpleModel()
    bridge = AudioBridge(
        model=model,
        layer_path='layer1',
        hallucination_vector=torch.randn(20),
        update_interval=1.0 # 1 second interval
    )
    bridge.audio_engine.update_parameters = MagicMock()
    
    # First step should work
    _ = model(torch.randn(1, 10))
    bridge.step()
    assert bridge.audio_engine.update_parameters.call_count == 1
    
    # Second step immediately after should NOT call update_parameters
    bridge.step()
    assert bridge.audio_engine.update_parameters.call_count == 1

def test_bridge_cleanup():
    model = SimpleModel()
    bridge = AudioBridge(
        model=model,
        layer_path='layer1',
        hallucination_vector=torch.randn(20)
    )
    
    # Mock removal
    bridge.hook_manager.remove_all = MagicMock()
    bridge.audio_engine.stop = MagicMock()
    
    bridge.stop()
    
    assert bridge.hook_manager.remove_all.called
    assert bridge.audio_engine.stop.called
