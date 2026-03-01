# 🎵 LLM Sonification — Turning Neural Network Weights & Activations into Music

> **What does a language model *sound* like?**
>
> This project extracts mathematical structures from transformer models (SVD spectra, weight distributions, activation dynamics, attention entropy) and maps them to musical parameters — producing orchestral compositions, acid basslines, house DJ sets, and full activation albums. No randomness for the sake of randomness: every note is derived from a real tensor.

⚠️ **Language note:** Comments and markdown cells inside the notebooks are written in **French**. The code, variable names, and this README are in English.

**note**  WIP, solo indie, feedback welcome
---

## 🎧 Audio Samples

Pre-rendered WAV files are available in [`audio/samples/`](audio/samples/):

| File | Source | Description |
|------|--------|-------------|
| `llm-house.wav` | Notebook 02 | House track (128 BPM) — bassline from `down_proj` row norms, chords from `q_proj` SVD |
| `gemma-3-270m-acid.wav` | Notebook 01 | Acid TB-303 pattern — resonance from `v_proj` kurtosis, cutoff from `gate_proj` std |
| `gemma-3-270m-acid-v2.wav` | Notebook 01 | Acid variant with different layer progression |
| `llm-thinking-sonification.wav` | Notebook 02 | Layer-by-layer sonification of a single forward pass through Gemma-3-270M |

---

## 📓 Notebooks

### [`01_llm_orchestra_midi_sf.ipynb`](notebooks/01_llm_orchestra_midi_sf.ipynb) — LLM Orchestra (MIDI + SoundFonts)

**Frozen weights as score.** Loads a model (Gemma-3-270M), decomposes every weight matrix via SVD, and translates the spectral structure into a multi-instrument MIDI composition rendered with FluidSynth.

Three genres are generated from the same model:

- **Opera (4 acts, 66 BPM):** Overture with embedding SVD → strings; fuguette canon where q/k/v/o_proj singular values enter as soprano/alto/tenor/bass voices; dramatic climax driven by attention-vs-MLP tension ratio; epilogue from `final_norm`.
- **House (128 bars, 128 BPM):** Chord progression from `q_proj` SVD, bassline from `down_proj` row norms, drums gated by `gate_proj` statistics. Full intro→buildup→drop→outro structure.
- **Acid (138 BPM):** One TB-303 pattern per layer — pitch sequence from `down_proj` row norms, accent pattern from value magnitude, slide from inter-layer SVD delta.

**Key mappings:**

| Track | GM Instrument | Weight source |
|-------|--------------|---------------|
| Violins | Violin (40) | `q_proj` SVD |
| Violas | Viola (41) | `k_proj` SVD |
| Cellos | Cello (42) | `v_proj` SVD |
| Contrabasses | Contrabass (43) | `down_proj` row norms |
| French Horns | Horn (60) | `gate_proj` histograms |
| Oboe/Clarinet | Oboe/Clarinet (68/71) | `up_proj` histograms |
| Harp | Harp (46) | LayerNorm γ |
| Organ | Organ (19) | `final_norm` |
| Synth Bass | Synth Bass (38) | `down_proj` (acid/house) |
| Drums | Channel 10 | `gate_proj` statistics |

**Tonality detection:** PCA on LayerNorm γ vectors across layers → PC2 skewness determines major/minor, PC3 kurtosis selects harmonic minor vs. dorian.

---

### [`02_llm_house_acid_sonification.ipynb`](notebooks/02_llm_house_acid_sonification.ipynb) — Standalone Sonifications

A multi-part notebook containing several independent sonification approaches:

1. **Activation extraction pipeline** — Extracts hidden states from Gemma-3-270M on Dolly-15k prompts, with masked mean pooling. Includes PCA/t-SNE/UMAP visualizations.
2. **Full-dataset sinusoidal sonification** — 2000 prompts × 640 dims → each dimension is a fixed-frequency sinusoid, amplitude = |activation|, phase = sign. Dimensions selected by variance.
3. **Layer-by-layer thinking sonification** — Single prompt through all 19 layers. Each token = one frequency (log-spaced), amplitude = L2 norm of hidden vector, phase = PCA angle. You *hear* the residual stream transform layer by layer.
4. **Operatic version** — Same pipeline but with Dorian scale quantization, additive synthesis (6 harmonics with formant-inspired weights), ADSR envelopes, vibrato, convolution reverb, and stereo panning.
5. **Frozen-weight DNA sonification (5 movements):**
   - I. *Prologue* — Embedding SVD → Lydian organ
   - II. *Anatomy* — Attention matrices SVD per layer → Dorian strings
   - III. *Memory* — MLP weights (gate/up/down) → Phrygian brass
   - IV. *Balance* — LayerNorm γ → Aeolian flute
   - V. *Epilogue* — Frobenius norms in order → fade to silence
6. **House track from weights** — Full-length house track (intro/buildup/drop/outro) with section-specific modal scales (Lydian→Dorian→Phrygian→Aeolian), Karplus-Strong string synthesis, FM bass, and multiband mastering.

---

### [`03_activation_album.ipynb`](notebooks/03_activation_album.ipynb) — Activation Album (Runtime Sonification)

**Runtime activations, not frozen weights.** Runs the model on actual prompts and sonifies the *computation*, not the parameters.

Architecture:

```
Prompt → tokenize → forward pass → hidden_states [NL, NT, D]
                                  → attention weights → entropy [NL]

Per layer:
  block_norms(h[layer]) → outer product → Markov transition matrix M[SL×SL]
  SVD(h[layer]) → Vt[:4] → micro-LSTM weights (W_i, W_f, W_o, W_c)
  dot(h[l,t], r̂) → refusal direction projection (injected into LSTM input)
  → generate sequence of (degree, pitch, velocity, duration) per token
```

**Refusal direction r̂:** Computed via PCA on `diff(hidden_harmful − hidden_harmless)` using 10 contrastive pairs, pooled over the final 50% of layers. The first principal component captures the most discriminative direction between harmful/harmless content.

**Four modes:**
- **Solo** — Single prompt → token-by-token tension profile + audio
- **Contrast** — Two synchronized MIDI files (safe vs. unsafe prompt). The unsafe voice activates the organ channel and shifts toward chromaticism.
- **Dialogue** — 11 prompts from JailbreakBench sorted by tension (calm→peak→resolution), forming a dramatic arc with tempo and mode modulation.
- **Album** — All 100 JBB behaviors grouped by category → one track per behavior, concatenated into a full album with per-track heatmaps.
- **Bonus: Abliteration signature** — Compares dot(h, r̂) between a base model and its abliterated counterpart, layer by layer. Sonifies the delta — you can *hear* which layers lost the refusal direction.

**MIDI channels:**

| Channel | Instrument | Source |
|---------|-----------|--------|
| 0 | Lead Saw (81) | Markov-LSTM token-level |
| 1 | Strings (48) | Markov-LSTM layer-level |
| 7 | Organ (19) | dot(h, r̂) — refusal tension |
| 12 | Celesta (8) | MLP sparsity — micro texture |

---

### [`04_jepa_house_train.ipynb`](notebooks/04_jepa_house_train.ipynb) — JEPA House Training

**Learned alignment between LLM weight statistics and house music.** Trains a Joint Embedding Predictive Architecture (BYOL-style) to align two heterogeneous spaces:

- **Enc_stats:** `SV[q,k,v,gate,down] × NL` + α-stable index + bimodality + Wasserstein distances + QK spectral coherence → `z_stats [256]`
- **Enc_music:** Windows of 32 MIDI notes (pitch, duration, velocity, interval, beat_pos) from Lakh MIDI Dataset (house-filtered: 115-142 BPM, 4/4, ≥32 bars) → `z_music [256]`
- **Predictor:** `z_stats → z_pred` in the space of `z_music`
- **Loss:** `1 − cosine(z_pred, sg(z_music_target))` + variance regularization + temporal continuity

**Training (3 phases):**
1. Music encoder pre-training (alone)
2. Cross-modal alignment (music encoder frozen, EMA target)
3. Joint fine-tuning

**Curriculum:** Pairing guided by tension — impulsive weights (α < 1.3) → high rhythmic density windows; gaussian weights (α > 1.8) → legato windows.

At inference, the JEPA predictor replaces the untrained micro-LSTM from the other notebooks, producing musically informed note distributions: `p_final = w_markov × M[current] + (1 − w_markov) × p_jepa`.

---

## 🔬 Technical Concepts

### Weight → Music Mappings

| Mathematical object | Musical parameter |
|---|---|
| Singular values (SVD) | Pitch sequence (log-scaled) |
| Weight distribution kurtosis | Filter resonance / harmonic content |
| Weight distribution std | Amplitude / filter cutoff |
| Row norms | Step sequencer pattern |
| LayerNorm γ | Melodic contour |
| Attention/MLP norm ratio | Dramatic tension |
| Layer depth | Reverb amount / temporal position |
| PCA of norms | Key/mode selection |

### Activation → Music Mappings

| Activation feature | Musical parameter |
|---|---|
| Hidden state norms | Note amplitude |
| dot(h, r̂) — refusal projection | Organ intensity / dissonance level |
| Attention entropy | Note duration (legato ↔ staccato) |
| MLP sparsity | Ornament density |
| Block norms outer product | Markov transition matrix |
| SVD of activation matrix | LSTM gate weights |

---

## 🚀 Quick Start

### Requirements

```bash
pip install transformers accelerate scikit-learn scipy numpy pretty_midi datasets
apt-get install fluidsynth fluid-soundfont-gm  # for WAV rendering
pip install pyfluidsynth
```

### Running

1. Open any notebook in **Google Colab** (GPU recommended for activation notebooks)
2. Set your HuggingFace token in the first cell (needed for gated models)
3. Run all cells — MIDI files are generated first, then rendered to WAV via FluidSynth

The orchestra and house notebooks work with **Gemma-3-270M** (~500MB, runs on CPU). The activation album uses **Phi-3.5-mini-instruct** (for demonstration — configurable to any causal LM).

---

## 📁 Repository Structure

```
llm-sonification/
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   ├── 01_llm_orchestra_midi_sf.ipynb   # Frozen weights → Opera/House/Acid MIDI
│   ├── 02_llm_house_acid_sonification.ipynb  # Standalone sonifications (6 approaches)
│   ├── 03_activation_album.ipynb        # Runtime activations → Album (4 modes + abliteration)
│   └── 04_jepa_house_train.ipynb        # JEPA training for LLM↔House alignment
├── audio/
│   └── samples/
│       ├── gemma-3-270m-acid.wav
│       ├── gemma-3-270m-acid-v2.wav
│       ├── llm-house.wav
│       └── llm-thinking-sonification.wav
└── scripts/                             # (future: CLI extraction scripts)
```

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

- **Models:** [Gemma-3-270M](https://huggingface.co/google/gemma-3-270m) (Google), [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) (Microsoft)
- **Datasets:** [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (Databricks), [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) (Chao et al., 2024), [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (Raffel, 2016)
- **Synthesis:** FluidSynth + GeneralUser GS SoundFont, Karplus-Strong and FM synthesis implementations
- **Refusal direction:** Inspired by [Arditi et al., 2024](https://arxiv.org/abs/2406.11717) — *Refusal in Language Models Is Mediated by a Single Direction*
