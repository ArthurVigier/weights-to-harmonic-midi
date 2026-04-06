import pyo
import time
from typing import Optional

class AudioEngine:
    def __init__(self, sr: int = 44100, nchnls: int = 2):
        self.server = pyo.Server(sr=sr, nchnls=nchnls).boot()
        self.server.start()
        
        # Base FM Synth setup
        # carrier frequency, ratio, index
        self.carrier_freq = pyo.Sig(440)
        self.ratio = pyo.Sig(0.5) # Rhodes-like ratio
        self.index = pyo.Sig(2)   # Modulation depth
        
        self.synth = pyo.FM(carrier=self.carrier_freq, ratio=self.ratio, index=self.index, mul=0.2)
        
        # Low Pass Filter
        self.cutoff = pyo.Sig(1000)
        self.q = pyo.Sig(1)
        self.lpf = pyo.Biquad(self.synth, freq=self.cutoff, q=self.q, type=0)
        
        # Distortion
        self.dist_amt = pyo.Sig(0)
        self.dist = pyo.Distort(self.lpf, drive=self.dist_amt, slope=0.5, mul=0.8)
        
        # Reverb for some "Lo-Fi" vibe
        self.rev = pyo.Freeverb(self.dist, size=0.8, damp=0.5, bal=0.3).out()

    def update_parameters(self, freq: float, dissonance_score: float):
        """
        Updates the DSP parameters based on the current frequency and dissonance.
        - freq: The target frequency from harmonic_math.py
        - dissonance_score: 0.0 to 1.0 mapping to LPF and Distortion.
        """
        # Smooth transitions using .setFreq or by assigning to the Sig objects
        self.carrier_freq.value = freq
        
        # As dissonance increases:
        # 1. LPF opens up (freq increases)
        # 2. Distortion increases
        # 3. FM Index increases for more "bite"
        
        target_cutoff = 1000 + (dissonance_score * 8000) # Range 1k to 9k
        self.cutoff.value = target_cutoff
        
        self.dist_amt.value = dissonance_score * 0.9 # Range 0 to 0.9
        self.index.value = 2 + (dissonance_score * 10) # More harmonic complexity

    def stop(self):
        self.server.stop()
        self.server.shutdown()

if __name__ == "__main__":
    # Quick test
    engine = AudioEngine()
    try:
        for i in range(10):
            d = i / 10.0
            print(f"Testing dissonance: {d}")
            engine.update_parameters(220 + (i * 20), d)
            time.sleep(0.5)
    finally:
        engine.stop()
