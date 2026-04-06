import numpy as np
from typing import List, Tuple

# Frequencies for Pentatonic scale (C Major Pentatonic as example: C, D, E, G, A)
# Intervals in semitones: 0, 2, 4, 7, 9
PENTATONIC_INTERVALS: List[int] = [0, 2, 4, 7, 9]

# Dissonant intervals: Minor Second (1), Tritone (6)
DISSONANT_INTERVALS: List[int] = [1, 6]

BASE_FREQ: float = 261.63  # C4

def mido_to_hz(mido_note: float) -> float:
    """Converts a MIDI note number to Frequency in Hz."""
    return 440.0 * (2.0 ** ((mido_note - 69.0) / 12.0))

def get_quantized_frequency(
    activation_value: float,
    dissonance_score: float,
    alert_threshold: float = 0.5,
    min_hz: float = 60.0,
    max_hz: float = 2000.0
) -> float:
    """
    Maps an activation value to a frequency quantized to a scale.
    If dissonance_score < alert_threshold, restricts to Pentatonic.
    If dissonance_score >= alert_threshold, allows Dissonant intervals.
    """
    # Handle NaNs and Infs safely
    if not np.isfinite(activation_value):
        activation_value = 0.0
    if not np.isfinite(dissonance_score):
        dissonance_score = 0.0

    # Map activation value (assumed normalized or centered) to a MIDI range (e.g., 48 to 84)
    # Using a simple linear mapping for now; activations might need normalization first.
    base_note = 48  # C3
    note_range = 36 # 3 octaves
    
    # We use a sigmoid-like squash or just clip to ensure it stays in range
    normalized_activation = float(1.0 / (1.0 + np.exp(-activation_value)))
    target_note = base_note + (normalized_activation * note_range)
    
    # Octave and scale quantization
    octave = int(target_note // 12)
    note_in_octave = target_note % 12
    
    allowed_intervals = PENTATONIC_INTERVALS.copy()
    if dissonance_score >= alert_threshold:
        allowed_intervals.extend(DISSONANT_INTERVALS)
    
    # Find the nearest allowed interval
    best_interval = min(allowed_intervals, key=lambda x: abs(x - note_in_octave))
    
    final_note = (octave * 12) + best_interval
    final_hz = mido_to_hz(float(final_note))
    
    # Final safety clip
    return float(np.clip(final_hz, min_hz, max_hz))
