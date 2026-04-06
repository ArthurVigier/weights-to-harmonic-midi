# 🎵 LLM Auditory Monitor — Real-Time Latent Space Sonification

> **What does a language model *sound* like while it thinks?**

## 🔄 Project Pivot: From Offline MIDI to Real-Time DSP

**Notice:** This project has fundamentally shifted its architecture. 

Previously, the project focused on *offline* sonification: downloading model weights, analyzing them statistically (SVD, PCA), generating MIDI files, and rendering them via external tools like FluidSynth or a DAW (see the `legacy/` directory for these experiments).

**The new direction is a Standalone, Real-Time Auditory Monitor.**

The goal is to build a "Plug n Play" auditory dashboard for Mechanistic Interpretability. Instead of static weight analysis, the system now hooks directly into a live PyTorch model during inference. It sonifies the *activations* (the latent space) in real-time, operating entirely within Python using an internal Digital Signal Processing (DSP) engine. **No external DAWs, MIDI software, or offline rendering are required.**

## 🎯 Core Concept: Sonifying Hallucinations

The engine acts as an auditory monitor for the LLM's internal state, specifically designed to detect when the model diverges from typical behavior (e.g., hallucinating or refusing a prompt).

*   **Normal State (Chill/Lo-Fi):** While the model is processing normally, the DSP engine generates pleasant, ambient generative arpeggios restricted to a strict **Pentatonic scale**. The sound profile is a smooth, Rhodes-like FM synthesizer running through a low-pass filter.
*   **Divergent State (Hallucination/Refusal):** As the projection of the current token's activation onto a predefined "Hallucination/Refusal Steering Vector" increases (calculated via cosine similarity):
    1.  The mathematical mapping breaks the pentatonic rule, introducing **dissonant intervals** (minor seconds, tritones).
    2.  The low-pass filter dynamically opens up, increasing the brightness and harshness of the sound.
    3.  A wave-shaping distortion effect is applied, physically altering the texture of the audio output.

## 🏗️ Architecture

The system is designed for high performance, utilizing a multi-threaded architecture to ensure that the heavy matrix multiplications of the LLM do not cause audio dropouts.

```text
llm_auditory_monitor/
├── src/
│   ├── core/
│   │   ├── model_runner.py     # Asynchronous LLM management (Text generation)
│   │   ├── hooks.py            # Leak-proof PyTorch forward_hooks for activation capture
│   │   ├── steering.py         # Vector projection math (Cosine similarity)
│   │   └── bridge.py           # Multi-threaded bridge syncing PyTorch with Audio
│   ├── audio/
│   │   ├── dsp_engine.py       # REAL-TIME SYNTHESIS ENGINE (using `pyo`)
│   │   └── harmonic_math.py    # Tensor -> Frequency/Pentatonic/Dissonance logic
│   └── utils/
│       └── config.py           # Centralized hyperparameters
├── tests/                      # Strict TDD suite (Hypothesis, Memory Leak checks)
├── Makefile                    # Developer commands (test, lint, typecheck)
└── pyproject.toml              # Dependencies
```

### Key Technical Achievements

1.  **Standalone Real-Time DSP (Zero DAW):** Built using the `pyo` library. The audio engine runs in its own background thread. An asynchronous queue bridges the PyTorch generation thread and the audio server, ensuring zero stuttering or buffer under-runs.
2.  **Zero Memory Leaks:** PyTorch forward hooks are notoriously prone to causing massive VRAM leaks if computational graphs are accidentally retained. The `HookManager` immediately detaches tensors and moves them to the CPU. Rigorous testing (`test_leak_leak_rigorous.py`) ensures a flat memory profile over extended sessions.
3.  **Mathematically Proven Harmonics:** Property-based testing via `hypothesis` mathematically proves that the `harmonic_math.py` module will *never* crash when fed extreme tensor values (NaNs, Infs) and guarantees strict adherence to the Pentatonic scale unless the dissonance threshold is explicitly breached.

## 🚀 Quick Start

### Installation

The project requires Python 3.10+ and a system capable of running PortAudio (required by `pyo`).

```bash
# Clone the repository
git clone https://github.com/your-username/weights-to-harmonic-midi.git
cd weights-to-harmonic-midi

# Install the package and its dependencies in editable mode
pip install -e ".[test]"
```

### Developer Commands

The project uses a `Makefile` for standardizing development tasks. We enforce strict static typing and comprehensive test coverage.

```bash
# Run the test suite (includes memory leak checks and property-based math tests)
make test

# Run the linter (Ruff)
make lint

# Run strict static type checking (MyPy)
make typecheck

# Run all checks
make all
```

## 📁 Legacy Files

The initial exploratory notebooks (`.ipynb`), scripts, and rendered `.wav` samples related to the offline SVD weight sonification and MIDI generation have been archived in the [`legacy/`](legacy/) directory for historical reference.
