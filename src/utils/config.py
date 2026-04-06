from dataclasses import dataclass, field
from typing import List

@dataclass
class AudioConfig:
    sr: int = 44100
    nchnls: int = 2
    update_interval: float = 0.05
    smoothing_factor: float = 0.9
    alert_threshold: float = 0.5

@dataclass
class HarmonicConfig:
    base_freq: float = 261.63 # C4
    min_hz: float = 60.0
    max_hz: float = 2000.0
    pentatonic_intervals: List[int] = field(default_factory=lambda: [0, 2, 4, 7, 9])
    dissonant_intervals: List[int] = field(default_factory=lambda: [1, 6])

@dataclass
class ModelConfig:
    layer_path: str = "model.layers.12"
    model_name: str = "google/gemma-2b" # Placeholder
    max_new_tokens: int = 100

@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    harmonic: HarmonicConfig = field(default_factory=HarmonicConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
