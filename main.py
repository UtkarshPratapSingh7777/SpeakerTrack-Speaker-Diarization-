from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from src.audio_loader import AudioConfig
from src.clusterer import ClusteringConfig
from src.embedder import EmbeddingConfig
from src.exceptions import DiarizationPipelineError
from src.logger import configure_logging
from src.pipeline import DiarizationResult, PipelineConfig, SpeakerDiarizationPipeline
from src.utils import set_global_seed
from src.vad import VADConfig
logger = logging.getLogger(__name__)

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}

def build_pipeline_config(raw: Dict[str, Any], num_speakers_override: Optional[int]) -> PipelineConfig:
    audio_cfg = AudioConfig(**raw.get('audio', {}))
    features_raw = raw.get('features', {})
    vad_raw = dict(raw.get('vad', {}))
    energy_fallback = vad_raw.pop('energy_fallback', {}) or {}
    vad_cfg = VADConfig(backend=vad_raw.get('backend', 'silero'), threshold=vad_raw.get('threshold', 0.5), min_speech_duration_ms=vad_raw.get('min_speech_duration_ms', 250), min_silence_duration_ms=vad_raw.get('min_silence_duration_ms', 100), speech_pad_ms=vad_raw.get('speech_pad_ms', 100), window_size_samples=vad_raw.get('window_size_samples', 512), energy_threshold_db=energy_fallback.get('energy_threshold_db', -40.0), zcr_threshold=energy_fallback.get('zcr_threshold', 0.5), frame_length_ms=energy_fallback.get('frame_length_ms', 25.0), hop_length_ms=energy_fallback.get('hop_length_ms', 10.0))
    emb_raw = raw.get('embedding', {})
    embedding_cfg = EmbeddingConfig(backend=emb_raw.get('backend', 'speechbrain'), pretrained_source=emb_raw.get('pretrained_source', 'speechbrain/spkrec-ecapa-voxceleb'), checkpoint_path=emb_raw.get('checkpoint_path'), cache_dir=emb_raw.get('cache_dir', '.cache/ecapa_tdnn'), channels=emb_raw.get('channels', 512), embedding_dim=emb_raw.get('embedding_dim', 192), res2net_scale=emb_raw.get('res2net_scale', 8), se_bottleneck=emb_raw.get('se_bottleneck', 128), attention_channels=emb_raw.get('attention_channels', 128), n_mels=features_raw.get('n_mels', 80), device=emb_raw.get('device', 'auto'), batch_size=emb_raw.get('batch_size', 32))
    clust_raw = raw.get('clustering', {})
    clustering_cfg = ClusteringConfig(method=clust_raw.get('method', 'agglomerative'), metric=clust_raw.get('metric', 'cosine'), distance_threshold=clust_raw.get('distance_threshold', 0.45), linkage=clust_raw.get('linkage', 'average'), min_speakers=clust_raw.get('min_speakers', 1), max_speakers=clust_raw.get('max_speakers', 8), auto_estimate_speakers=clust_raw.get('auto_estimate_speakers', True))
    smoothing = clust_raw.get('smoothing', {}) or {}
    return PipelineConfig(audio=audio_cfg, vad=vad_cfg, embedding=embedding_cfg, clustering=clustering_cfg, window_duration_s=emb_raw.get('window_duration_s', 1.5), window_step_s=emb_raw.get('window_step_s', 0.75), min_window_duration_s=emb_raw.get('min_window_duration_s', 0.5), min_segment_duration_s=smoothing.get('min_segment_duration_s', 0.5), collar_s=smoothing.get('collar_s', 0.25), num_speakers=num_speakers_override)

def result_to_rttm(result: DiarizationResult, uri: str='stream') -> str:
    lines = [f'SPEAKER {uri} 1 {seg.start:.3f} {seg.duration:.3f} <NA> <NA> {seg.speaker} <NA> <NA>' for seg in result.segments]
    return '\n'.join(lines) + ('\n' if lines else '')

def result_to_json(result: DiarizationResult) -> str:
    payload = {'source': result.source_path, 'duration_seconds': round(result.duration_seconds, 3), 'num_speakers': result.num_speakers, 'processing_time_seconds': round(result.processing_time_seconds, 3), 'segments': [{'start': seg.start, 'end': seg.end, 'duration': seg.duration, 'speaker': seg.speaker, 'mean_pitch_hz': round(seg.mean_pitch_hz, 1) if seg.mean_pitch_hz is not None else None} for seg in result.segments]}
    return json.dumps(payload, indent=2)

def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='speaker-diarization', description='Language-Agnostic Speaker Identification & Diarization Pipeline')
    parser.add_argument('audio_path', type=str, help='Path to the input audio file')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to YAML config')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to write output files')
    parser.add_argument('--format', type=str, choices=['rttm', 'json', 'both'], default=None)
    parser.add_argument('--num-speakers', type=int, default=None, help='Force a fixed number of speakers')
    parser.add_argument('--uri', type=str, default='stream', help='Recording URI/label used in RTTM output')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable DEBUG logging')
    return parser.parse_args(argv)

def main(argv: Optional[list]=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    config_path = Path(args.config)
    if not config_path.exists():
        print(f'[FATAL] Config file not found: {config_path}', file=sys.stderr)
        return 2
    raw_config = load_yaml_config(config_path)
    log_cfg = raw_config.get('logging', {})
    configure_logging(level='DEBUG' if args.verbose else log_cfg.get('level', 'INFO'), log_to_file=log_cfg.get('log_to_file', False), log_file_path=log_cfg.get('log_file_path'), json_format=log_cfg.get('json_format', False))
    set_global_seed(int(raw_config.get('seed', 42)))
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        logger.error('Input audio file does not exist: %s', audio_path)
        return 2
    output_cfg = raw_config.get('output', {})
    output_dir = Path(args.output_dir or output_cfg.get('output_dir', 'outputs'))
    output_format = args.format or output_cfg.get('format', 'rttm')
    try:
        pipeline_config = build_pipeline_config(raw_config, args.num_speakers)
        pipeline = SpeakerDiarizationPipeline(pipeline_config)
        result = pipeline.run(audio_path)
    except DiarizationPipelineError as exc:
        logger.error('Diarization failed: %s', exc)
        return 1
    except Exception:
        logger.exception('Unexpected error during diarization')
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    if output_format in ('rttm', 'both'):
        rttm_path = output_dir / f'{stem}.rttm'
        rttm_path.write_text(result_to_rttm(result, uri=args.uri), encoding='utf-8')
        logger.info('RTTM written to %s', rttm_path)
    if output_format in ('json', 'both'):
        json_path = output_dir / f'{stem}.json'
        json_path.write_text(result_to_json(result), encoding='utf-8')
        logger.info('JSON written to %s', json_path)
    print(f'\nDiarization complete: {result.num_speakers} speaker(s) detected in {result.duration_seconds:.2f}s of audio (processed in {result.processing_time_seconds:.2f}s).')
    for seg in result.segments:
        pitch_str = f', ~{seg.mean_pitch_hz:.0f} Hz' if seg.mean_pitch_hz else ''
        print(f'  [{seg.start:7.2f}s -> {seg.end:7.2f}s] {seg.speaker}{pitch_str}')
    return 0
if __name__ == '__main__':
    sys.exit(main())