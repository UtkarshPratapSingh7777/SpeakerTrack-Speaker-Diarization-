from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from src.exceptions import VADProcessingError
logger = logging.getLogger(__name__)
SILERO_SUPPORTED_SAMPLE_RATES = {8000, 16000}

@dataclass(frozen=True)
class VADConfig:
    backend: str = 'silero'
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 100
    window_size_samples: int = 512
    energy_threshold_db: float = -40.0
    zcr_threshold: float = 0.5
    frame_length_ms: float = 25.0
    hop_length_ms: float = 10.0

@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

class BaseVAD:

    def detect(self, waveform: torch.Tensor, sample_rate: int) -> List[SpeechSegment]:
        raise NotImplementedError

class SileroVAD(BaseVAD):

    def __init__(self, config: VADConfig) -> None:
        self.config = config
        self._model, self._get_speech_timestamps = self._load_model()

    def _load_model(self):
        try:
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False, trust_repo=True, verbose=False)
            model.eval()
            get_speech_timestamps = utils[0]
            return (model, get_speech_timestamps)
        except Exception as exc:
            raise VADProcessingError(f'Failed to load Silero VAD from torch.hub: {exc}') from exc

    def detect(self, waveform: torch.Tensor, sample_rate: int) -> List[SpeechSegment]:
        if sample_rate not in SILERO_SUPPORTED_SAMPLE_RATES:
            raise VADProcessingError(f'Silero VAD only supports sample rates {SILERO_SUPPORTED_SAMPLE_RATES}, got {sample_rate} Hz.')
        audio_1d = waveform.squeeze(0).to(torch.float32)
        try:
            timestamps = self._get_speech_timestamps(audio_1d, self._model, threshold=self.config.threshold, sampling_rate=sample_rate, min_speech_duration_ms=self.config.min_speech_duration_ms, min_silence_duration_ms=self.config.min_silence_duration_ms, speech_pad_ms=self.config.speech_pad_ms, window_size_samples=self.config.window_size_samples, return_seconds=False)
        except Exception as exc:
            raise VADProcessingError(f'Silero VAD inference failed: {exc}') from exc
        return [SpeechSegment(start=ts['start'] / sample_rate, end=ts['end'] / sample_rate) for ts in timestamps]

class EnergyZCRVAD(BaseVAD):

    def __init__(self, config: VADConfig) -> None:
        self.config = config

    def detect(self, waveform: torch.Tensor, sample_rate: int) -> List[SpeechSegment]:
        signal = waveform.squeeze(0).to(torch.float64).numpy()
        frame_len = max(1, int(round(self.config.frame_length_ms / 1000.0 * sample_rate)))
        hop_len = max(1, int(round(self.config.hop_length_ms / 1000.0 * sample_rate)))
        frames = self._frame_signal(signal, frame_len, hop_len)
        if frames.size == 0:
            return []
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        peak_rms = float(rms.max())
        if peak_rms < 1e-07:
            logger.debug('Energy VAD: clip is effectively silent (peak rms=%.2e); no speech detected.', peak_rms)
            return []
        energy_db = 20.0 * np.log10(np.clip(rms / peak_rms, 1e-12, None))
        sign_changes = np.diff(np.sign(frames), axis=1) != 0
        zcr = np.mean(sign_changes, axis=1)
        speech_flags = (energy_db > self.config.energy_threshold_db) & (zcr <= self.config.zcr_threshold)
        logger.debug('Energy VAD: %d/%d frames flagged as speech (mean energy=%.1fdB, mean zcr=%.3f)', int(speech_flags.sum()), len(speech_flags), float(energy_db.mean()), float(zcr.mean()))
        return self._frames_to_segments(speech_flags, hop_len, sample_rate)

    @staticmethod
    def _frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
        n_samples = signal.shape[0]
        if n_samples < frame_len:
            return np.empty((0, frame_len))
        n_frames = 1 + (n_samples - frame_len) // hop_len
        indices = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
        return signal[indices]

    def _frames_to_segments(self, speech_flags: np.ndarray, hop_len: int, sample_rate: int) -> List[SpeechSegment]:
        hop_seconds = hop_len / sample_rate
        min_speech_frames = max(1, int(round(self.config.min_speech_duration_ms / 1000.0 / hop_seconds)))
        min_silence_frames = max(1, int(round(self.config.min_silence_duration_ms / 1000.0 / hop_seconds)))
        pad_seconds = self.config.speech_pad_ms / 1000.0
        raw_segments: List[Tuple[int, int]] = []
        in_speech = False
        seg_start = 0
        for i, flag in enumerate(speech_flags):
            if flag and (not in_speech):
                in_speech, seg_start = (True, i)
            elif not flag and in_speech:
                in_speech = False
                raw_segments.append((seg_start, i))
        if in_speech:
            raw_segments.append((seg_start, len(speech_flags)))
        merged: List[Tuple[int, int]] = []
        for start, end in raw_segments:
            if merged and start - merged[-1][1] <= min_silence_frames:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return [SpeechSegment(start=max(0.0, start * hop_seconds - pad_seconds), end=end * hop_seconds + pad_seconds) for start, end in merged if end - start >= min_speech_frames]

def build_vad(config: VADConfig) -> BaseVAD:
    if config.backend == 'silero':
        try:
            return SileroVAD(config)
        except VADProcessingError as exc:
            logger.warning('Silero VAD unavailable (%s); falling back to the offline energy + zero-crossing-rate VAD.', exc)
            return EnergyZCRVAD(config)
    if config.backend == 'energy':
        return EnergyZCRVAD(config)
    raise VADProcessingError(f"Unknown VAD backend: '{config.backend}'")