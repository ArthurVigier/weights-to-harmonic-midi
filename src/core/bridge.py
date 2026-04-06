import time
import queue
import threading
import torch
from typing import Optional
from src.audio.dsp_engine import AudioEngine
from src.audio.harmonic_math import get_quantized_frequency
from src.core.steering import HallucinationSteerer
from src.core.hooks import HookManager

class AsyncAudioBridge:
    """
    Consumes activations in a background thread to update the DSP engine.
    Ensures zero audio glitches even if the model thread is heavy or blocking.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        layer_path: str,
        hallucination_vector: torch.Tensor,
        update_interval: float = 0.05,
        smoothing_factor: float = 0.9,
        alert_threshold: float = 0.5
    ):
        self.hook_manager = HookManager(model)
        self.hook_manager.register_layer(layer_path)
        self.layer_path = layer_path
        self.steerer = HallucinationSteerer(hallucination_vector, smoothing_factor)
        self.audio_engine = AudioEngine()
        
        self.update_interval = update_interval
        self.alert_threshold = alert_threshold
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the background consumption loop."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Background loop consuming latest activations from the HookManager."""
        while not self._stop_event.is_set():
            activation = self.hook_manager.get_activation(self.layer_path)
            
            if activation is not None:
                # 1. Calculate Score
                score = self.steerer.compute_score(activation)
                
                # 2. Get Energy/Freq
                energy = torch.abs(activation).mean().item()
                freq = get_quantized_frequency(
                    activation_value=energy,
                    dissonance_score=score,
                    alert_threshold=self.alert_threshold
                )
                
                # 3. Update Audio (Internal Pyo thread handles the smoothing/playing)
                self.audio_engine.update_parameters(freq, score)
            
            time.sleep(self.update_interval)

    def stop(self):
        """Cleanly stops the thread and audio engine."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        self.hook_manager.remove_all()
        self.audio_engine.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
