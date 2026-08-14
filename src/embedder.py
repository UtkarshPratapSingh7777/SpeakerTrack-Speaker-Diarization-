from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.audio_loader import FeatureConfig, LogMelFeatureExtractor
from src.exceptions import EmbeddingExtractionError
from src.utils import resolve_device
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = 'speechbrain'
    pretrained_source: str = 'speechbrain/spkrec-ecapa-voxceleb'
    checkpoint_path: Optional[str] = None
    cache_dir: str = '.cache/ecapa_tdnn'
    channels: int = 512
    embedding_dim: int = 192
    res2net_scale: int = 8
    se_bottleneck: int = 128
    attention_channels: int = 128
    n_mels: int = 80
    device: str = 'auto'
    batch_size: int = 32

class SEModule(nn.Module):

    def __init__(self, channels: int, bottleneck: int=128) -> None:
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, bottleneck, kernel_size=1), nn.ReLU(inplace=True), nn.Conv1d(bottleneck, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)

class Res2NetBlock(nn.Module):

    def __init__(self, channels: int, scale: int=8, kernel_size: int=3, dilation: int=1) -> None:
        super().__init__()
        if channels % scale != 0:
            raise ValueError(f'channels ({channels}) must be divisible by scale ({scale})')
        self.scale = scale
        self.width = channels // scale
        padding = dilation * (kernel_size - 1) // 2
        self.convs = nn.ModuleList([nn.Conv1d(self.width, self.width, kernel_size=kernel_size, dilation=dilation, padding=padding) for _ in range(scale - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = torch.split(x, self.width, dim=1)
        out = [splits[0]]
        for i in range(1, self.scale):
            sp = splits[i] if i == 1 else splits[i] + out[-1]
            sp = self.convs[i - 1](sp)
            sp = F.relu(self.bns[i - 1](sp), inplace=True)
            out.append(sp)
        return torch.cat(out, dim=1)

class SERes2NetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, scale: int=8, se_bottleneck: int=128) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.res2net = Res2NetBlock(out_channels, scale=scale, kernel_size=kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.se = SEModule(out_channels, bottleneck=se_bottleneck)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.res2net(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        out = self.se(out)
        return out + residual

class AttentiveStatsPool(nn.Module):

    def __init__(self, channels: int, attention_channels: int=128, global_context: bool=True) -> None:
        super().__init__()
        self.global_context = global_context
        in_dim = channels * 3 if global_context else channels
        self.attention = nn.Sequential(nn.Conv1d(in_dim, attention_channels, kernel_size=1), nn.ReLU(inplace=True), nn.BatchNorm1d(attention_channels), nn.Tanh(), nn.Conv1d(attention_channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_context:
            t = x.size(-1)
            global_mean = x.mean(dim=-1, keepdim=True).expand(-1, -1, t)
            global_std = x.std(dim=-1, keepdim=True, unbiased=False).expand(-1, -1, t)
            attn_input = torch.cat([x, global_mean, global_std], dim=1)
        else:
            attn_input = x
        alpha = torch.softmax(self.attention(attn_input), dim=-1)
        mean = torch.sum(alpha * x, dim=-1)
        residual = torch.sum(alpha * x ** 2, dim=-1) - mean ** 2
        std = torch.sqrt(residual.clamp(min=1e-08))
        return torch.cat([mean, std], dim=1)

class ECAPA_TDNN(nn.Module):

    def __init__(self, input_dim: int=80, channels: int=512, embedding_dim: int=192, scale: int=8, se_bottleneck: int=128, attention_channels: int=128) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(input_dim, channels, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.BatchNorm1d(channels))
        self.layer2 = SERes2NetBlock(channels, channels, kernel_size=3, dilation=2, scale=scale, se_bottleneck=se_bottleneck)
        self.layer3 = SERes2NetBlock(channels, channels, kernel_size=3, dilation=3, scale=scale, se_bottleneck=se_bottleneck)
        self.layer4 = SERes2NetBlock(channels, channels, kernel_size=3, dilation=4, scale=scale, se_bottleneck=se_bottleneck)
        cat_channels = channels * 3
        self.layer5 = nn.Sequential(nn.Conv1d(cat_channels, cat_channels, kernel_size=1), nn.ReLU(inplace=True))
        self.pooling = AttentiveStatsPool(cat_channels, attention_channels=attention_channels, global_context=True)
        self.bn_pool = nn.BatchNorm1d(cat_channels * 2)
        self.fc = nn.Linear(cat_channels * 2, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x1 + x2)
        x4 = self.layer4(x1 + x2 + x3)
        multi_scale = torch.cat([x2, x3, x4], dim=1)
        multi_scale = self.layer5(multi_scale)
        pooled = self.pooling(multi_scale)
        pooled = self.bn_pool(pooled)
        embedding = self.bn_fc(self.fc(pooled))
        return embedding

class SpeakerEmbedder:

    def __init__(self, config: EmbeddingConfig, sample_rate: int) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self.device = resolve_device(config.device)
        self._backend = 'native'
        self._speechbrain_classifier = None
        self._native_model: Optional[ECAPA_TDNN] = None
        self._feature_extractor: Optional[LogMelFeatureExtractor] = None
        if config.backend == 'speechbrain':
            self._init_speechbrain()
        if self._backend == 'native':
            self._init_native()

    @property
    def backend(self) -> str:
        return self._backend

    def _init_speechbrain(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError:
            logger.warning('`speechbrain` is not installed; using the native PyTorch ECAPA-TDNN backend instead. Install `speechbrain` (see requirements.txt) for pretrained, production-grade accuracy.')
            return
        try:
            self._speechbrain_classifier = EncoderClassifier.from_hparams(source=self.config.pretrained_source, savedir=self.config.cache_dir, run_opts={'device': str(self.device)}, local_strategy=LocalStrategy.COPY)
            self._backend = 'speechbrain'
            logger.info("Loaded pretrained ECAPA-TDNN via SpeechBrain ('%s').", self.config.pretrained_source)
        except Exception as exc:
            logger.warning('Could not download/load SpeechBrain pretrained weights (%s). Falling back to the native PyTorch ECAPA-TDNN backend.', exc)

    def _init_native(self) -> None:
        self._native_model = ECAPA_TDNN(input_dim=self.config.n_mels, channels=self.config.channels, embedding_dim=self.config.embedding_dim, scale=self.config.res2net_scale, se_bottleneck=self.config.se_bottleneck, attention_channels=self.config.attention_channels).to(self.device)
        self._native_model.eval()
        if self.config.checkpoint_path:
            try:
                state_dict = torch.load(self.config.checkpoint_path, map_location=self.device)
                self._native_model.load_state_dict(state_dict)
                logger.info('Loaded native ECAPA-TDNN checkpoint: %s', self.config.checkpoint_path)
            except Exception as exc:
                raise EmbeddingExtractionError(f"Failed to load checkpoint at '{self.config.checkpoint_path}': {exc}") from exc
        else:
            logger.warning('Native ECAPA-TDNN initialized with random weights (no `checkpoint_path` configured). Embeddings will be architecturally valid but not yet speaker-discriminative until trained or a pretrained checkpoint is supplied.')
        feature_config = FeatureConfig(n_mels=self.config.n_mels)
        self._feature_extractor = LogMelFeatureExtractor(feature_config, self.sample_rate).to(self.device)

    @torch.no_grad()
    def embed_batch(self, windows: List[torch.Tensor]) -> np.ndarray:
        if not windows:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)
        batch_size = max(1, self.config.batch_size)
        chunks: List[np.ndarray] = []
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            try:
                chunks.append(self._embed_one_batch(batch))
            except Exception as exc:
                raise EmbeddingExtractionError(f'Embedding extraction failed: {exc}') from exc
        embeddings = np.concatenate(chunks, axis=0)
        return self._l2_normalize(embeddings)

    def _embed_one_batch(self, batch: List[torch.Tensor]) -> np.ndarray:
        waveform_batch = torch.cat(batch, dim=0).to(self.device)
        if self._backend == 'speechbrain' and self._speechbrain_classifier is not None:
            embeddings = self._speechbrain_classifier.encode_batch(waveform_batch)
            embeddings = embeddings.squeeze(1)
            return embeddings.detach().cpu().numpy().astype(np.float32)
        assert self._native_model is not None and self._feature_extractor is not None
        mel_features = self._feature_extractor(waveform_batch)
        embeddings = self._native_model(mel_features)
        return embeddings.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return embeddings / norms