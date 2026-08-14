from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st
st.set_page_config(page_title='Speaker Diarization', page_icon='🎙️', layout='wide', initial_sidebar_state='expanded')
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'config' / 'config.yaml'

@st.cache_resource(show_spinner=False)
def _load_pipeline_modules():
    from main import build_pipeline_config, load_yaml_config, result_to_json, result_to_rttm
    from src.logger import configure_logging
    from src.pipeline import SpeakerDiarizationPipeline
    from src.utils import set_global_seed
    return {'build_pipeline_config': build_pipeline_config, 'load_yaml_config': load_yaml_config, 'result_to_json': result_to_json, 'result_to_rttm': result_to_rttm, 'configure_logging': configure_logging, 'SpeakerDiarizationPipeline': SpeakerDiarizationPipeline, 'set_global_seed': set_global_seed}

def _speaker_color(speaker: str) -> str:
    palette = ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B', '#EECA3B', '#B279A2', '#FF9DA6']
    try:
        idx = int(speaker.split('_')[-1]) % len(palette)
    except (ValueError, IndexError):
        idx = abs(hash(speaker)) % len(palette)
    return palette[idx]

def _segments_to_dataframe(result) -> pd.DataFrame:
    rows = []
    for seg in result.segments:
        rows.append({'Speaker': seg.speaker, 'Start (s)': round(seg.start, 3), 'End (s)': round(seg.end, 3), 'Duration (s)': seg.duration, 'Mean Pitch (Hz)': round(seg.mean_pitch_hz, 1) if seg.mean_pitch_hz is not None else None})
    return pd.DataFrame(rows)

def _timeline_html(result, total_duration: float) -> str:
    if not result.segments or total_duration <= 0:
        return "<p style='color:#888'>No speech segments to display.</p>"
    bars = []
    for seg in result.segments:
        left = 100.0 * seg.start / total_duration
        width = max(0.4, 100.0 * (seg.end - seg.start) / total_duration)
        color = _speaker_color(seg.speaker)
        title = f'{seg.speaker}: {seg.start:.2f}s – {seg.end:.2f}s'
        bars.append(f'<div title="{title}" style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;height:100%;background:{color};border-radius:3px;opacity:0.9;"></div>')
    legend_items = []
    seen = set()
    for seg in result.segments:
        if seg.speaker in seen:
            continue
        seen.add(seg.speaker)
        color = _speaker_color(seg.speaker)
        legend_items.append(f'<span style="display:inline-flex;align-items:center;margin-right:14px;"><span style="width:12px;height:12px;background:{color};border-radius:2px;margin-right:6px;"></span>{seg.speaker}</span>')
    return f'\n    <div style="margin:8px 0 4px 0;font-size:0.85rem;color:#555;">\n      {''.join(legend_items)}\n    </div>\n    <div style="position:relative;width:100%;height:36px;background:#e8e8e8;\n                border-radius:6px;overflow:hidden;border:1px solid #ddd;">\n      {''.join(bars)}\n    </div>\n    <div style="display:flex;justify-content:space-between;font-size:0.75rem;\n                color:#888;margin-top:2px;">\n      <span>0.0 s</span>\n      <span>{total_duration:.1f} s</span>\n    </div>\n    '

def render_sidebar() -> Dict[str, Any]:
    st.sidebar.title('🎙️ Settings')
    st.sidebar.markdown('### Input')
    uploaded = st.sidebar.file_uploader('Upload audio', type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma'])
    st.sidebar.markdown('### Speakers')
    force_speakers = st.sidebar.checkbox('Force number of speakers', value=False, help='Override automatic estimation (recommended for short clips)')
    num_speakers: Optional[int] = None
    if force_speakers:
        num_speakers = st.sidebar.slider('Number of speakers', 1, 8, 2)
    st.sidebar.markdown('### Clustering')
    method = st.sidebar.selectbox('Method', ['agglomerative', 'spectral'], index=0)
    distance_threshold = st.sidebar.slider('Distance threshold', 0.15, 0.8, 0.4, 0.01, help='Lower → more speakers. Used when speaker count is not forced.')
    auto_estimate = st.sidebar.checkbox('Auto-estimate speakers (eigengap)', value=not force_speakers, disabled=force_speakers)
    st.sidebar.markdown('### Windows')
    window_duration = st.sidebar.slider('Window duration (s)', 0.5, 3.0, 1.5, 0.1)
    window_step = st.sidebar.slider('Window step (s)', 0.25, 2.0, 0.75, 0.05)
    st.sidebar.markdown('### Output')
    output_format = st.sidebar.radio('Download format', ['both', 'rttm', 'json'], index=0)
    st.sidebar.markdown('---')
    st.sidebar.caption('ECAPA-TDNN · Silero VAD · Language-agnostic')
    return {'uploaded': uploaded, 'num_speakers': num_speakers, 'method': method, 'distance_threshold': distance_threshold, 'auto_estimate': auto_estimate and (not force_speakers), 'window_duration': window_duration, 'window_step': window_step, 'output_format': output_format}

def run_diarization(audio_bytes: bytes, filename: str, opts: Dict[str, Any]):
    mods = _load_pipeline_modules()
    mods['configure_logging'](level='WARNING')
    raw = mods['load_yaml_config'](CONFIG_PATH)
    mods['set_global_seed'](int(raw.get('seed', 42)))
    raw.setdefault('clustering', {})
    raw['clustering']['method'] = opts['method']
    raw['clustering']['distance_threshold'] = opts['distance_threshold']
    raw['clustering']['auto_estimate_speakers'] = opts['auto_estimate']
    raw.setdefault('embedding', {})
    raw['embedding']['window_duration_s'] = opts['window_duration']
    raw['embedding']['window_step_s'] = opts['window_step']
    pipeline_config = mods['build_pipeline_config'](raw, opts['num_speakers'])
    pipeline = mods['SpeakerDiarizationPipeline'](pipeline_config)
    suffix = Path(filename).suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        result = pipeline.run(tmp_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    return (result, mods)

def main() -> None:
    st.title('🎙️ Speaker Identification & Diarization')
    st.markdown('Upload any audio — the pipeline finds **who** spoke and **when**, in any language.')
    opts = render_sidebar()
    uploaded = opts['uploaded']
    col_run, _ = st.columns([1, 3])
    with col_run:
        run_btn = st.button('▶ Run Diarization', type='primary', disabled=uploaded is None, use_container_width=True)
    if uploaded is None:
        st.info('← Upload an audio file in the sidebar to get started.')
        st.markdown('\n            **Tips**\n            - Short clips (< 15 s) → enable **Force number of speakers**\n            - Similar voices → lower distance threshold (0.30–0.35)\n            - Prefer WAV / FLAC over heavy MP3 compression\n            ')
        return
    st.audio(uploaded.getvalue())
    if not run_btn:
        st.caption('Click **Run Diarization** when ready.')
        return
    with st.spinner('Running VAD → embeddings → clustering…'):
        try:
            result, mods = run_diarization(uploaded.getvalue(), uploaded.name, opts)
        except Exception as exc:
            name = type(exc).__name__
            if any((k in name for k in ('Diarization', 'Audio', 'VAD', 'Embedding', 'Clustering', 'Pipeline'))):
                st.error(f'Pipeline error: {exc}')
            else:
                st.exception(exc)
            return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Speakers', result.num_speakers)
    m2.metric('Segments', len(result.segments))
    m3.metric('Audio duration', f'{result.duration_seconds:.1f} s')
    m4.metric('Processing time', f'{result.processing_time_seconds:.1f} s')
    st.subheader('Timeline')
    st.markdown(_timeline_html(result, result.duration_seconds), unsafe_allow_html=True)
    st.subheader('Segments')
    df = _segments_to_dataframe(result)
    if df.empty:
        st.warning('No speech segments detected.')
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.subheader('Download')
    stem = Path(uploaded.name).stem
    fmt = opts['output_format']
    dl_cols = st.columns(2)
    if fmt in ('rttm', 'both'):
        rttm_text = mods['result_to_rttm'](result, uri=stem)
        dl_cols[0].download_button('⬇ RTTM', data=rttm_text, file_name=f'{stem}.rttm', mime='text/plain', use_container_width=True)
    if fmt in ('json', 'both'):
        json_text = mods['result_to_json'](result)
        dl_cols[1].download_button('⬇ JSON', data=json_text, file_name=f'{stem}.json', mime='application/json', use_container_width=True)
    with st.expander('Raw JSON'):
        st.code(mods['result_to_json'](result), language='json')
if __name__ == '__main__':
    main()