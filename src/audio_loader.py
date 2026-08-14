from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from src.exceptions import AudioTooShortError, CorruptedAudioError, UnsupportedAudioFormatError
logger = logging.getLogger(__name__)
_KNOWN_NON_AUDIO_EXTENSIONS = {'.txt', '.json', '.csv', '.tsv', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.docx', '.xlsx', '.pptx', '.zip', '.tar', '.gz', '.py', '.md'}

@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int = 16000
    mono: bool = True
    peak_normalize: bool = True
    min_duration_seconds: float = 0.5

@dataclass(frozen=True)
class FeatureConfig:
    n_mels: int = 80
    n_fft: int = 400
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    f_min: float = 20.0
    f_max: float = 7600.0

@dataclass
class AudioTensor:
    waveform: torch.Tensor
    sample_rate: int
    duration_seconds: float
    source_path: Path

class AudioLoader:

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    def load(self, path: Union[str, Path]) -> AudioTensor:
        path = Path(path)
        if not path.exists():
            raise CorruptedAudioError(f'Audio file not found: {path}')
        if path.suffix.lower() in _KNOWN_NON_AUDIO_EXTENSIONS:
            raise UnsupportedAudioFormatError(f"'{path.suffix}' is not a supported audio format: {path}")
        waveform, sample_rate = self._read_waveform(path)
        if waveform.numel() == 0:
            raise CorruptedAudioError(f'Decoded zero-length waveform from: {path}')
        waveform = self._to_mono_or_first_channel(waveform)
        waveform, sample_rate = self._resample_if_needed(waveform, sample_rate)
        waveform = self._normalize(waveform)
        duration_seconds = waveform.shape[-1] / float(sample_rate)
        if duration_seconds < self.config.min_duration_seconds:
            raise AudioTooShortError(f'Audio duration {duration_seconds:.3f}s is below the configured minimum of {self.config.min_duration_seconds:.3f}s: {path}')
        waveform = waveform.contiguous().to(torch.float32)
        return AudioTensor(waveform=waveform, sample_rate=sample_rate, duration_seconds=duration_seconds, source_path=path)

    def _read_waveform(self, path: Path) -> Tuple[torch.Tensor, int]:
        try:
            waveform, sample_rate = torchaudio.load(str(path))
            return (waveform, sample_rate)
        except Exception as primary_exc:
            logger.warning("torchaudio failed to decode '%s' (%s); retrying with librosa.", path, primary_exc)
        try:
            samples, sample_rate = librosa.load(str(path), sr=None, mono=False)
            if samples.ndim == 1:
                samples = samples[np.newaxis, :]
            waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
            return (waveform, int(sample_rate))
        except Exception as fallback_exc:
            raise CorruptedAudioError(f"Could not decode audio file '{path}' with either torchaudio or librosa: {fallback_exc}") from fallback_exc

    def _to_mono_or_first_channel(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] == 1:
            return waveform
        if self.config.mono:
            return waveform.mean(dim=0, keepdim=True)
        return waveform[:1, :]

    def _resample_if_needed(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        target = self.config.target_sample_rate
        if sample_rate == target:
            return (waveform, sample_rate)
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target)
        return (resampler(waveform), target)

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.config.peak_normalize:
            return waveform
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak * 0.95
        return waveform

class LogMelFeatureExtractor:

    def __init__(self, config: FeatureConfig, sample_rate: int) -> None:
        self.config = config
        self.sample_rate = sample_rate
        win_length = max(1, int(round(config.win_length_ms / 1000.0 * sample_rate)))
        hop_length = max(1, int(round(config.hop_length_ms / 1000.0 * sample_rate)))
        f_max = min(config.f_max, sample_rate / 2.0 - 1.0)
        self._transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=config.n_fft, win_length=win_length, hop_length=hop_length, n_mels=config.n_mels, f_min=config.f_min, f_max=f_max, power=2.0)

    def to(self, device: torch.device) -> 'LogMelFeatureExtractor':
        self._transform = self._transform.to(device)
        return self

    def __call__(self, waveform_batch: torch.Tensor) -> torch.Tensor:
        mel = self._transform(waveform_batch)
        log_mel = torch.log(mel.clamp(min=1e-06))
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True, unbiased=False)
        normalized = (log_mel - mean) / (std + 1e-06)
        return normalized.transpose(1, 2)