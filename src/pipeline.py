from __future__ import annotations
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, Union
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from src.audio_loader import AudioConfig, AudioLoader
from src.clusterer import ClusteringConfig, SpeakerClusterer
from src.embedder import EmbeddingConfig, SpeakerEmbedder
from src.vad import SpeechSegment, VADConfig, build_vad
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DiarizedSegment:
    start: float
    end: float
    speaker: str
    mean_pitch_hz: Optional[float] = None

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

@dataclass(frozen=True)
class DiarizationResult:
    source_path: str
    duration_seconds: float
    num_speakers: int
    segments: List[DiarizedSegment]
    processing_time_seconds: float

@dataclass(frozen=True)
class PipelineConfig:
    audio: AudioConfig
    vad: VADConfig
    embedding: EmbeddingConfig
    clustering: ClusteringConfig
    window_duration_s: float = 1.5
    window_step_s: float = 0.75
    min_window_duration_s: float = 0.5
    min_segment_duration_s: float = 0.5
    collar_s: float = 0.25
    num_speakers: Optional[int] = None

class SpeakerDiarizationPipeline:

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.audio_loader = AudioLoader(config.audio)
        self.vad = build_vad(config.vad)
        self.embedder = SpeakerEmbedder(config.embedding, sample_rate=config.audio.target_sample_rate)
        self.clusterer = SpeakerClusterer(config.clustering)

    def run(self, audio_path: Union[str, Path]) -> DiarizationResult:
        start_time = time.perf_counter()
        audio_path = Path(audio_path)
        logger.info('Loading audio: %s', audio_path)
        audio = self.audio_loader.load(audio_path)
        logger.info('Loaded %.2fs of audio at %d Hz', audio.duration_seconds, audio.sample_rate)
        logger.info('Running voice activity detection...')
        speech_segments = self.vad.detect(audio.waveform, audio.sample_rate)
        speech_total = sum((s.duration for s in speech_segments))
        logger.info('Detected %d speech region(s) totaling %.2fs', len(speech_segments), speech_total)
        if not speech_segments:
            logger.warning('No speech detected in %s', audio_path)
            return self._empty_result(audio_path, audio.duration_seconds, start_time)
        windows, window_bounds = self._build_windows(audio.waveform, audio.sample_rate, speech_segments)
        if not windows:
            logger.warning('Speech regions in %s were too short to form any embedding window.', audio_path)
            return self._empty_result(audio_path, audio.duration_seconds, start_time)
        logger.info('Built %d overlapping embedding window(s)', len(windows))
        logger.info('Extracting speaker embeddings (backend=%s)...', self.embedder.backend)
        embeddings = self.embedder.embed_batch(windows)
        logger.info('Clustering speaker embeddings (method=%s)...', self.config.clustering.method)
        labels = self.clusterer.cluster(embeddings, num_speakers=self.config.num_speakers)
        raw_segments = self._labels_to_segments(window_bounds, labels)
        smoothed_segments = self._smooth_segments(raw_segments)
        enriched_segments = self._attach_pitch_statistics(smoothed_segments, audio.waveform, audio.sample_rate)
        num_speakers = len({seg.speaker for seg in enriched_segments})
        elapsed = time.perf_counter() - start_time
        logger.info('Diarization completed in %.2fs: %d speaker(s), %d segment(s)', elapsed, num_speakers, len(enriched_segments))
        return DiarizationResult(source_path=str(audio_path), duration_seconds=audio.duration_seconds, num_speakers=num_speakers, segments=enriched_segments, processing_time_seconds=elapsed)

    def _build_windows(self, waveform: torch.Tensor, sample_rate: int, speech_segments: List[SpeechSegment]) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        window_samples = max(1, int(round(self.config.window_duration_s * sample_rate)))
        step_samples = max(1, int(round(self.config.window_step_s * sample_rate)))
        min_samples = max(1, int(round(self.config.min_window_duration_s * sample_rate)))
        total_samples = waveform.shape[-1]
        windows: List[torch.Tensor] = []
        bounds: List[Tuple[float, float]] = []
        for segment in speech_segments:
            seg_start = max(0, int(round(segment.start * sample_rate)))
            seg_end = min(total_samples, int(round(segment.end * sample_rate)))
            seg_len = seg_end - seg_start
            if seg_len < min_samples:
                continue
            if seg_len <= window_samples:
                chunk = waveform[:, seg_start:seg_end]
                pad = window_samples - chunk.shape[-1]
                if pad > 0:
                    chunk = F.pad(chunk, (0, pad))
                windows.append(chunk)
                bounds.append((seg_start / sample_rate, seg_end / sample_rate))
                continue
            cursor = seg_start
            last_window_end = seg_start
            while cursor + window_samples <= seg_end:
                windows.append(waveform[:, cursor:cursor + window_samples])
                bounds.append((cursor / sample_rate, (cursor + window_samples) / sample_rate))
                last_window_end = cursor + window_samples
                cursor += step_samples
            if last_window_end < seg_end:
                tail_start = max(seg_start, seg_end - window_samples)
                chunk = waveform[:, tail_start:seg_end]
                pad = window_samples - chunk.shape[-1]
                if pad > 0:
                    chunk = F.pad(chunk, (0, pad))
                windows.append(chunk)
                bounds.append((tail_start / sample_rate, seg_end / sample_rate))
        return (windows, bounds)

    def _labels_to_segments(self, window_bounds: List[Tuple[float, float]], labels: np.ndarray, frame_resolution_s: float=0.1) -> List[Tuple[float, float, int]]:
        if not window_bounds:
            return []
        starts = np.array([b[0] for b in window_bounds])
        ends = np.array([b[1] for b in window_bounds])
        centers = (starts + ends) / 2.0
        t0, t1 = (float(starts.min()), float(ends.max()))
        n_frames = max(1, int(np.ceil((t1 - t0) / frame_resolution_s)))
        frame_centers = t0 + (np.arange(n_frames) + 0.5) * frame_resolution_s
        frame_labels = np.full(n_frames, fill_value=-1, dtype=int)
        for i, t in enumerate(frame_centers):
            covering = np.where((starts <= t) & (ends >= t))[0]
            if covering.size == 0:
                continue
            closest = covering[np.argmin(np.abs(centers[covering] - t))]
            frame_labels[i] = labels[closest]
        return self._runlength_encode(frame_labels, t0, frame_resolution_s)

    @staticmethod
    def _runlength_encode(frame_labels: np.ndarray, t0: float, resolution: float) -> List[Tuple[float, float, int]]:
        segments: List[Tuple[float, float, int]] = []
        current_label: Optional[int] = None
        seg_start = t0
        for i, label in enumerate(frame_labels):
            t = t0 + i * resolution
            if label == -1:
                if current_label is not None:
                    segments.append((seg_start, t, current_label))
                    current_label = None
                continue
            if label != current_label:
                if current_label is not None:
                    segments.append((seg_start, t, current_label))
                current_label, seg_start = (int(label), t)
        if current_label is not None:
            segments.append((seg_start, t0 + len(frame_labels) * resolution, current_label))
        return segments

    def _smooth_segments(self, raw_segments: List[Tuple[float, float, int]]) -> List[DiarizedSegment]:
        if not raw_segments:
            return []
        merged: List[List] = []
        for start, end, label in raw_segments:
            if merged and merged[-1][2] == label and (start - merged[-1][1] <= self.config.collar_s):
                merged[-1][1] = end
            else:
                merged.append([start, end, label])
        filtered = [m for m in merged if m[1] - m[0] >= self.config.min_segment_duration_s]
        label_map: dict = {}
        result: List[DiarizedSegment] = []
        for start, end, label in filtered:
            if label not in label_map:
                label_map[label] = f'SPEAKER_{len(label_map):02d}'
            result.append(DiarizedSegment(start=round(start, 3), end=round(end, 3), speaker=label_map[label]))
        return result

    def _attach_pitch_statistics(self, segments: List[DiarizedSegment], waveform: torch.Tensor, sample_rate: int) -> List[DiarizedSegment]:
        if not segments:
            return segments
        audio_np = waveform.squeeze(0).cpu().numpy()
        enriched: List[DiarizedSegment] = []
        for seg in segments:
            start_sample = max(0, int(seg.start * sample_rate))
            end_sample = min(audio_np.shape[0], int(seg.end * sample_rate))
            pitch = self._estimate_mean_pitch(audio_np[start_sample:end_sample], sample_rate)
            enriched.append(replace(seg, mean_pitch_hz=pitch))
        return enriched

    @staticmethod
    def _estimate_mean_pitch(clip: np.ndarray, sample_rate: int) -> Optional[float]:
        if clip.size < int(sample_rate * 0.05):
            return None
        try:
            f0, voiced_flag, _ = librosa.pyin(clip.astype(np.float32), fmin=float(librosa.note_to_hz('C2')), fmax=float(librosa.note_to_hz('C7')), sr=sample_rate)
            if voiced_flag is not None and np.any(voiced_flag):
                voiced_f0 = f0[voiced_flag]
            else:
                voiced_f0 = f0[~np.isnan(f0)]
            if voiced_f0.size == 0:
                return None
            return float(np.nanmean(voiced_f0))
        except Exception as exc:
            logger.debug('Pitch estimation skipped for a segment: %s', exc)
            return None

    @staticmethod
    def _empty_result(audio_path: Path, duration: float, start_time: float) -> DiarizationResult:
        return DiarizationResult(source_path=str(audio_path), duration_seconds=duration, num_speakers=0, segments=[], processing_time_seconds=time.perf_counter() - start_time)