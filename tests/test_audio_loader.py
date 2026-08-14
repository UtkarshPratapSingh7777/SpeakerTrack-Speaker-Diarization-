from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from src.audio_loader import AudioConfig, AudioLoader, FeatureConfig, LogMelFeatureExtractor
from src.exceptions import AudioTooShortError, CorruptedAudioError, UnsupportedAudioFormatError

def _write_sine_wav(path: Path, duration_s: float, sample_rate: int=22050, freq: float=220.0) -> None:
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), signal, sample_rate)

def test_load_resamples_to_target_rate(tmp_path: Path) -> None:
    wav_path = tmp_path / 'tone.wav'
    _write_sine_wav(wav_path, duration_s=1.0, sample_rate=22050)
    loader = AudioLoader(AudioConfig(target_sample_rate=16000, min_duration_seconds=0.1))
    audio = loader.load(wav_path)
    assert audio.sample_rate == 16000
    assert audio.waveform.shape[0] == 1
    assert abs(audio.duration_seconds - 1.0) < 0.05

def test_load_raises_on_too_short_audio(tmp_path: Path) -> None:
    wav_path = tmp_path / 'blip.wav'
    _write_sine_wav(wav_path, duration_s=0.1, sample_rate=16000)
    loader = AudioLoader(AudioConfig(target_sample_rate=16000, min_duration_seconds=1.0))
    with pytest.raises(AudioTooShortError):
        loader.load(wav_path)

def test_load_raises_on_missing_file(tmp_path: Path) -> None:
    loader = AudioLoader(AudioConfig())
    with pytest.raises(CorruptedAudioError):
        loader.load(tmp_path / 'does_not_exist.wav')

def test_load_raises_on_obviously_unsupported_extension(tmp_path: Path) -> None:
    fake_path = tmp_path / 'notes.txt'
    fake_path.write_text('not audio')
    loader = AudioLoader(AudioConfig())
    with pytest.raises(UnsupportedAudioFormatError):
        loader.load(fake_path)

def test_peak_normalize_keeps_amplitude_within_bounds(tmp_path: Path) -> None:
    wav_path = tmp_path / 'loud.wav'
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    signal = (5.0 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    sf.write(str(wav_path), signal, 16000)
    loader = AudioLoader(AudioConfig(target_sample_rate=16000, peak_normalize=True, min_duration_seconds=0.1))
    audio = loader.load(wav_path)
    assert audio.waveform.abs().max().item() <= 1.0

def test_log_mel_feature_extractor_output_shape() -> None:
    sample_rate = 16000
    extractor = LogMelFeatureExtractor(FeatureConfig(n_mels=40), sample_rate=sample_rate)
    waveform_batch = np.random.randn(2, sample_rate).astype(np.float32)
    import torch
    features = extractor(torch.from_numpy(waveform_batch))
    assert features.shape[0] == 2
    assert features.shape[2] == 40
    assert features.shape[1] > 1