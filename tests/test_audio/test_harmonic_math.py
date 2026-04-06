from hypothesis import given, strategies as st
import numpy as np
import pytest
from src.audio.harmonic_math import (
    get_quantized_frequency, 
    PENTATONIC_INTERVALS,
    DISSONANT_INTERVALS
)

@given(
    activation=st.floats(allow_nan=True, allow_infinity=True),
    dissonance=st.floats(allow_nan=True, allow_infinity=True, min_value=-10.0, max_value=10.0),
    min_hz=st.floats(min_value=20.0, max_value=100.0),
    max_hz=st.floats(min_value=1000.0, max_value=5000.0)
)
def test_frequency_is_always_valid_float(activation, dissonance, min_hz, max_hz):
    """Verifies that the function always returns a valid float within bounds."""
    freq = get_quantized_frequency(activation, dissonance, min_hz=min_hz, max_hz=max_hz)
    assert isinstance(freq, float)
    assert min_hz <= freq <= max_hz
    assert np.isfinite(freq)

@given(
    activation=st.floats(min_value=-1000.0, max_value=1000.0),
    dissonance=st.floats(min_value=0.0, max_value=0.49) # Below threshold (0.5)
)
def test_pentatonic_constraint_below_threshold(activation, dissonance):
    """
    Verifies that when dissonance is low, the output frequency matches 
    a note in the pentatonic scale.
    """
    freq = get_quantized_frequency(activation, dissonance, alert_threshold=0.5)
    
    # Reverse Hz to MIDI Note
    mido_note = 12 * np.log2(freq / 440.0) + 69
    
    # Check if the note in octave (rounding to handle float precision)
    note_in_octave = round(mido_note) % 12
    
    assert note_in_octave in PENTATONIC_INTERVALS

@given(
    activation=st.floats(min_value=-1000.0, max_value=1000.0),
    dissonance=st.floats(min_value=0.5, max_value=1.0) # Above/at threshold
)
def test_allowed_intervals_above_threshold(activation, dissonance):
    """
    Verifies that when dissonance is high, the output frequency is in 
    either Pentatonic or Dissonant intervals.
    """
    freq = get_quantized_frequency(activation, dissonance, alert_threshold=0.5)
    
    mido_note = 12 * np.log2(freq / 440.0) + 69
    note_in_octave = round(mido_note) % 12
    
    allowed = PENTATONIC_INTERVALS + DISSONANT_INTERVALS
    assert note_in_octave in allowed
