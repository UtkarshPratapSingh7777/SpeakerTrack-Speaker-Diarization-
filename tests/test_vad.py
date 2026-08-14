from __future__ import annotations
import numpy as np
import torch
from src.vad import EnergyZCRVAD, VADConfig

def _tone_silence_tone(sample_rate: int=16000) -> torch.Tensor:
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    tone = (0.8 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    silence = np.zeros(sample_rate, dtype=np.float32)
    signal = np.concatenate([tone, silence, tone])
    return torch.from_numpy(signal).unsqueeze(0)

def test_energy_vad_detects_two_speech_regions() -> None:
    waveform = _tone_silence_tone(sample_rate=16000)
    vad = EnergyZCRVAD(VADConfig(backend='energy', energy_threshold_db=-35.0, min_speech_duration_ms=200, min_silence_duration_ms=200, speech_pad_ms=0))
    segments = vad.detect(waveform, sample_rate=16000)
    assert len(segments) == 2
    for seg in segments:
        assert seg.duration > 0.5

def test_energy_vad_on_pure_silence_returns_no_segments() -> None:
    waveform = torch.zeros(1, 16000)
    vad = EnergyZCRVAD(VADConfig(backend='energy'))
    segments = vad.detect(waveform, sample_rate=16000)
    assert segments == []

def test_energy_vad_handles_audio_shorter_than_one_frame() -> None:
    waveform = torch.zeros(1, 10)
    vad = EnergyZCRVAD(VADConfig(backend='energy', frame_length_ms=25.0))
    segments = vad.detect(waveform, sample_rate=16000)
    assert segments == []