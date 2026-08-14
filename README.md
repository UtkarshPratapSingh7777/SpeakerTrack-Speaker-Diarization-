# Language-Agnostic Speaker Identification & Diarization

A production-grade pipeline that answers two questions for any audio recording, **in any language**: *who is speaking*, and *when*. It segments speech, derives a fixed-dimensional "voiceprint" for every speaker using an **ECAPA-TDNN** embedding network, and clusters those voiceprints into discrete speaker identities using vectorized cosine-similarity analytics — outputting an industry-standard **RTTM** or **JSON** diarization timeline.

It is "language-agnostic" by construction, not by claim: every stage of the pipeline operates on the acoustic signal — energy, spectral shape, pitch, formant structure — and never on text, a lexicon, or a phonetic alignment. Two people speaking different languages should still embed far apart by *voice*; the same person switching languages mid-recording should still embed close together.

```
┌─────────────┐     ┌──────────────┐     ┌────────────────────┐     ┌───────────────────┐     ┌─────────────┐
│   audio_     │────▶│     vad      │────▶│   embedder          │────▶│    clusterer        │────▶│  pipeline    │
│   loader     │     │  (Silero /   │     │  (ECAPA-TDNN:        │     │  (cosine distance,  │     │ (windowing,  │
│  decode,     │     │   energy     │     │   pretrained or      │     │   agglomerative /   │     │  smoothing,  │
│  resample,   │     │   fallback)  │     │   native PyTorch)     │     │   spectral)         │     │  RTTM/JSON)  │
│  log-Mel     │     │              │     │                       │     │                     │     │              │
└─────────────┘     └──────────────┘     └────────────────────┘     └───────────────────┘     └─────────────┘
```

---

## Table of contents

- [Key features](#key-features)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration reference](#configuration-reference)
- [Algorithmic deep dive](#algorithmic-deep-dive)
- [Output formats](#output-formats)
- [Testing](#testing)
- [Performance & complexity](#performance--complexity)
- [Production considerations & limitations](#production-considerations--limitations)
- [License](#license)

---

## Key features

- **Language-agnostic by design.** No ASR, no lexicon, no phonetic alignment anywhere in the path — speaker identity is derived purely from acoustic timbre, pitch, and vocal-tract characteristics.
- **Native PyTorch ECAPA-TDNN.** The full architecture (SE-Res2Net blocks, multi-layer feature aggregation, attentive statistics pooling with global context) is implemented from scratch — not just imported — with an optional pretrained `speechbrain/spkrec-ecapa-voxceleb` backend for production accuracy.
- **Triple-redundant graceful degradation.** Every external dependency (Silero VAD via `torch.hub`, SpeechBrain pretrained weights via the HuggingFace Hub, even the `torchaudio` decode backend) has a fully-offline fallback, so the pipeline **never hard-fails purely on network/model availability** — it degrades, logs why, and keeps running. This isn't theoretical: all three fallbacks were observed firing for real in the sandbox this project was validated in.
- **Vectorized clustering math.** Pairwise speaker-distance computation is a single `O(n²d)` matrix operation (scikit-learn's BLAS-backed `cosine_distances`), never a Python loop over embedding pairs. Automatic speaker-count estimation uses a partial eigendecomposition (`scipy.linalg.eigh(..., subset_by_index=...)`) rather than a full `O(n³)` factorization.
- **Correct overlapping-window math.** Speech regions are sliced into fixed-length, overlapping windows with a final window anchored to each region's tail, guaranteeing full coverage with no dropped audio at region boundaries — then resolved back into a continuous, smoothed speaker timeline.
- **Typed exception hierarchy + structured logging.** Corrupted audio, too-short clips, VAD/embedding/clustering failures all raise specific, catchable exception types instead of bare `Exception`s or opaque tracebacks.
- **Decoupled YAML configuration.** Every hyperparameter — sample rate, window sizes, VAD thresholds, clustering method, speaker-count bounds — lives in `config/config.yaml`, not in code.
- **Real unit test coverage** for the clustering math, VAD fallback, and audio loader (`pytest`), not just a demo script.

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch, Torchaudio |
| Audio engineering | Librosa, Silero VAD (torch.hub) / custom energy-VAD |
| Speaker embeddings | ECAPA-TDNN (native PyTorch + optional SpeechBrain pretrained) |
| Clustering & analytics | scikit-learn (Agglomerative / Spectral clustering, cosine similarity), NumPy, SciPy |
| Config & CLI | PyYAML, argparse |
| Testing | pytest |

## Project structure

```
speaker_diarization_pipeline/
├── config/
│   └── config.yaml          # All hyperparameters (sample rate, VAD, embedding, clustering)
├── src/
│   ├── __init__.py
│   ├── audio_loader.py      # Dual-backend decode, resampling, normalization, log-Mel features
│   ├── vad.py                # Silero VAD with an offline energy+ZCR fallback
│   ├── embedder.py           # Native ECAPA-TDNN + SpeechBrain pretrained backend
│   ├── clusterer.py          # Vectorized cosine-distance clustering + eigengap speaker-count estimation
│   ├── pipeline.py            # End-to-end orchestration: windowing, smoothing, pitch enrichment
│   ├── exceptions.py          # Typed exception hierarchy
│   ├── logger.py              # Structured logging configuration
│   └── utils.py                # Seeding / device-resolution helpers
├── tests/                      # pytest unit tests (clusterer, VAD, audio loader)
├── main.py                     # CLI entry point -> RTTM / JSON
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

> `exceptions.py`, `logger.py`, `utils.py`, and `tests/` are additions beyond a bare-minimum scaffold — standard ML-engineering practice for anything meant to run in production, and called out here so the rationale is explicit rather than implied.

## Installation

```bash
cd speaker_diarization_pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional, for running the test suite:
pip install -r requirements-dev.txt
```

`speechbrain` (for pretrained ECAPA-TDNN weights) is in `requirements.txt` by default. If you'd rather not pull it in, remove it — the pipeline will log a warning and use the native architecture instead.

## Usage

```bash
# Basic run, RTTM output (default), speaker count auto-estimated
python main.py path/to/audio.wav

# Both RTTM and JSON, custom output directory
python main.py path/to/audio.wav --format both --output-dir results/

# Force a known speaker count instead of auto-estimating it
python main.py path/to/audio.wav --num-speakers 3

# Verbose (DEBUG) logging, custom config
python main.py path/to/audio.wav --config config/config.yaml -v
```

CLI options:

| Flag | Description |
|---|---|
| `audio_path` | Path to the input audio file (positional) |
| `--config` | Path to the YAML config (default `config/config.yaml`) |
| `--output-dir` | Output directory (default from config, `outputs/`) |
| `--format` | `rttm` \| `json` \| `both` |
| `--num-speakers` | Force a fixed speaker count (skips auto-estimation) |
| `--uri` | Recording label used in RTTM lines (default `stream`) |
| `-v`, `--verbose` | DEBUG-level logging |

## Configuration reference

`config/config.yaml` is organized by pipeline stage:

- **`audio`** — target sample rate (16 kHz, required by Silero VAD), mono-mixing, peak normalization, minimum duration.
- **`features`** — log-Mel filterbank parameters (`n_mels`, FFT/window/hop sizes, frequency range) shared by the native ECAPA-TDNN backend.
- **`vad`** — `backend: silero | energy`, Silero's threshold/duration/padding parameters, and the energy-fallback's frame size, dB threshold, and zero-crossing-rate ceiling.
- **`embedding`** — `backend: speechbrain | native`, the pretrained source/checkpoint, ECAPA-TDNN architecture dimensions (channels, embedding dim, Res2Net scale, SE bottleneck, attention channels), device, batch size, and window duration/step for the diarization sliding window.
- **`clustering`** — `method: agglomerative | spectral`, distance threshold (used when speaker count isn't fixed or auto-estimated), `min_speakers`/`max_speakers` bounds for auto-estimation, and post-clustering smoothing (`min_segment_duration_s`, `collar_s`).
- **`output`** — default format and output directory.
- **`logging`** — level, optional file sink, optional JSON formatting.

## Algorithmic deep dive

### Voice Activity Detection

[Silero VAD](https://github.com/snakers4/silero-vad) is a small recurrent/convolutional network trained across dozens of languages on raw waveform energy patterns, fetched once via `torch.hub`. If it can't be fetched, `EnergyZCRVAD` takes over: short-time RMS energy is computed in **fully vectorized form** via NumPy fancy-indexing framing (no per-sample Python loop), expressed in dB relative to the clip's own peak (with an absolute noise-floor guard so pure silence — where every frame is equally near-zero — isn't misread as 100% speech), and combined with a zero-crossing-rate ceiling to reject broadband-noise frames energy alone would misclassify.

### ECAPA-TDNN speaker embeddings

Implemented from the original architecture (Desplanques et al., 2020):

1. **Stem** — a single dilated Conv1D + BatchNorm over the log-Mel filterbank.
2. **Three SE-Res2Net blocks** (dilations 2, 3, 4) — each splits its channels into 8 groups (**Res2Net**, the "Propagation" in ECAPA) processed at hierarchically increasing dilation for multi-scale temporal context, then re-weights channels via a learned **Squeeze-and-Excitation** gate (the "Emphasized Channel Attention").
3. **Multi-layer feature aggregation** — the three blocks' outputs are concatenated, giving the pooling stage access to every scale at once rather than just the final layer's.
4. **Attentive statistics pooling with global context** — rather than a plain average over time, a learned attention map (conditioned on the whole utterance's mean/std, appended as context) weights *which frames* contribute to the pooled mean and standard deviation per channel — emphasizing stable, speaker-characteristic frames (e.g. steady-state vowels) over transients.
5. **Projection** — a final linear layer + BatchNorm produces a 192-dim embedding, L2-normalized before clustering so cosine similarity reduces to a dot product.

By default this runs against the official pretrained `speechbrain/spkrec-ecapa-voxceleb` checkpoint (`embedding.backend: speechbrain`). If SpeechBrain isn't installed or its weights can't be fetched, the pipeline **automatically falls back** to the native PyTorch implementation above — which is fully functional and architecturally correct, but, with no checkpoint supplied, starts from random weights and is *not yet speaker-discriminative* until trained or pointed at a checkpoint via `embedding.checkpoint_path`. This is logged loudly rather than silently producing misleadingly bad clusters.

### Clustering

Embeddings are L2-normalized, so pairwise **cosine distance** is computed once as a single vectorized matrix (`sklearn.metrics.pairwise.cosine_distances`) and fed to either:

- **Agglomerative clustering** (`metric="precomputed"`, average linkage) — either with a fixed `n_clusters`, or via `distance_threshold` when the count isn't known.
- **Spectral clustering** on the cosine-affinity graph.

When the speaker count is neither supplied nor fixed by threshold, it's estimated via the **eigengap heuristic**: the normalized graph Laplacian of the affinity matrix has one near-zero eigenvalue per well-separated cluster, so the largest gap between consecutive sorted eigenvalues (within `[min_speakers, max_speakers]`) is taken as the speaker count. Only the lowest `max_speakers + 1` eigenvalues are computed via `scipy.linalg.eigh(..., subset_by_index=...)` rather than the full spectrum, which matters once the number of embedding windows grows into the thousands.

### Windowing & timeline reconstruction

Each VAD speech region is sliced into fixed-length, overlapping windows (`window_duration_s` / `window_step_s`); a region shorter than one window is zero-padded rather than dropped, and a final window is anchored to each region's *end* so the sliding stride never leaves a tail uncovered. After clustering, overlapping per-window labels are resolved into a single timeline on a fine time grid (default 100 ms), where each grid point inherits the label of whichever covering window's center is closest — smoothing speaker-change boundaries through the overlap rather than trusting any one window's prediction outright. The result is run-length encoded, small gaps between same-speaker segments are merged (`collar_s`), and anything left shorter than `min_segment_duration_s` is dropped.

### Acoustic enrichment

Each final segment is also annotated with a mean fundamental frequency (`mean_pitch_hz`) via `librosa.pyin` — a purely acoustic statistic with no linguistic content, included to make the "language-agnostic acoustic footprint" claim something you can see in the output, not just trust about the embedding.

## Output formats

**RTTM** (NIST Rich Transcription Time Marked format):

```
SPEAKER stream 1 0.000 2.600 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER stream 1 3.000 2.200 <NA> <NA> SPEAKER_01 <NA> <NA>
```

Columns: `SPEAKER <uri> <channel> <start> <duration> <NA> <NA> <speaker-label> <NA> <NA>`.

**JSON:**

```json
{
  "source": "audio.wav",
  "duration_seconds": 10.3,
  "num_speakers": 2,
  "processing_time_seconds": 2.9,
  "segments": [
    {"start": 0.0, "end": 2.6, "duration": 2.6, "speaker": "SPEAKER_00", "mean_pitch_hz": 120.1},
    {"start": 3.0, "end": 5.2, "duration": 2.2, "speaker": "SPEAKER_01", "mean_pitch_hz": 220.3}
  ]
}
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

The suite covers: vectorized distance-matrix correctness (symmetry, zero diagonal), agglomerative/spectral clustering on synthetic well-separated embeddings (both fixed-`k` and eigengap auto-estimation), the offline energy/ZCR VAD on synthetic tone/silence patterns (including the pure-silence edge case), and the audio loader's resampling, normalization, and typed-exception paths.

## Performance & complexity

- **Embedding extraction** is batched (`embedding.batch_size`) and runs on GPU automatically when available (`embedding.device: auto`).
- **Clustering is `O(n²)`** in the number of embedding windows (the pairwise distance/affinity matrix), which is standard for this family of window-then-cluster diarization systems. For a typical conversation-length recording (minutes, not many hours) at the default 0.75 s window step, this is a small fraction of total runtime relative to embedding extraction.
- **Speaker-count estimation uses a partial eigendecomposition**, not a full `O(n³)` one, specifically to keep this stage from becoming the bottleneck as `n` grows.
- For very long recordings (multi-hour), consider chunking the input and diarizing per-chunk, or swapping in an online/incremental clustering strategy — the windowed-embedding architecture here is the same one production systems like `pyannote.audio` use, and shares the same scaling characteristics.

## Production considerations & limitations

- **Pretrained vs. native embeddings.** Real speaker-discrimination accuracy depends on the SpeechBrain pretrained checkpoint (or a checkpoint you've trained for the native architecture). The native backend with random weights is architecturally correct and useful for offline development/testing of the *pipeline*, but is not a speaker-recognition model until trained.
- **VAD/audio-decode fallbacks change behavior, not just availability.** The energy-VAD fallback is a much cruder speech detector than Silero (it doesn't know what speech *sounds* like, only that it's louder than the background) — expect more false positives on non-speech-but-loud audio when it's in use; this is logged clearly so it's never a silent quality regression.
- **8/16 kHz only for Silero VAD.** This is a hard constraint of the upstream model, which is why `audio.target_sample_rate` defaults to 16000 and the pipeline resamples everything to it regardless of source rate.
- **No speaker re-identification across files.** Each run's `SPEAKER_00`, `SPEAKER_01`, ... labels are local to that file (assigned in order of first appearance); matching "the same person" across separate recordings would require persisting and comparing embeddings, which this pipeline doesn't currently do.

## License

MIT — use, modify, and ship this freely; attribution appreciated but not required.
