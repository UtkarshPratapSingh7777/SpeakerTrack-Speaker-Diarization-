from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from src.exceptions import ClusteringError
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ClusteringConfig:
    method: str = 'agglomerative'
    metric: str = 'cosine'
    distance_threshold: float = 0.45
    linkage: str = 'average'
    min_speakers: int = 1
    max_speakers: int = 8
    auto_estimate_speakers: bool = True

class SpeakerClusterer:

    def __init__(self, config: ClusteringConfig) -> None:
        self.config = config
        if config.metric != 'cosine':
            logger.warning("Only the 'cosine' metric is currently implemented; ignoring configured metric '%s'.", config.metric)

    def cluster(self, embeddings: np.ndarray, num_speakers: Optional[int]=None) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ClusteringError(f'Expected a 2D embedding matrix, got shape {embeddings.shape}')
        n = embeddings.shape[0]
        if n == 0:
            raise ClusteringError('No embeddings were provided for clustering.')
        if n == 1:
            return np.zeros(1, dtype=int)
        distance_matrix = self._compute_distance_matrix(embeddings)
        if num_speakers is None and self.config.auto_estimate_speakers:
            num_speakers = self._estimate_num_speakers(embeddings, distance_matrix)
            logger.info('Auto-estimated number of speakers: %d', num_speakers)
        if self.config.method == 'agglomerative':
            return self._cluster_agglomerative(distance_matrix, num_speakers)
        if self.config.method == 'spectral':
            return self._cluster_spectral(embeddings, num_speakers)
        raise ClusteringError(f"Unknown clustering method: '{self.config.method}'")

    @staticmethod
    def _compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
        distance_matrix = cosine_distances(embeddings)
        np.fill_diagonal(distance_matrix, 0.0)
        return distance_matrix

    def _cluster_agglomerative(self, distance_matrix: np.ndarray, num_speakers: Optional[int]) -> np.ndarray:
        if num_speakers is not None:
            model = AgglomerativeClustering(n_clusters=int(num_speakers), metric='precomputed', linkage=self.config.linkage)
        else:
            model = AgglomerativeClustering(n_clusters=None, distance_threshold=self.config.distance_threshold, metric='precomputed', linkage=self.config.linkage)
        return model.fit_predict(distance_matrix)

    def _cluster_spectral(self, embeddings: np.ndarray, num_speakers: Optional[int]) -> np.ndarray:
        affinity = np.clip(cosine_similarity(embeddings), 0.0, 1.0)
        np.fill_diagonal(affinity, 1.0)
        k = num_speakers or self._estimate_num_speakers(embeddings, 1.0 - affinity)
        model = SpectralClustering(n_clusters=int(k), affinity='precomputed', assign_labels='kmeans', random_state=42)
        return model.fit_predict(affinity)

    def _estimate_num_speakers(self, embeddings: np.ndarray, distance_matrix: np.ndarray) -> int:
        n = embeddings.shape[0]
        min_k = max(1, self.config.min_speakers)
        max_k = min(self.config.max_speakers, n)
        if max_k <= min_k:
            return min_k
        affinity = np.clip(1.0 - distance_matrix, 0.0, 1.0)
        np.fill_diagonal(affinity, 0.0)
        degree = affinity.sum(axis=1)
        degree = np.where(degree <= 1e-10, 1e-10, degree)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        laplacian = np.eye(n) - d_inv_sqrt @ affinity @ d_inv_sqrt
        laplacian = (laplacian + laplacian.T) / 2.0
        search_upper = min(max_k + 1, n)
        try:
            eigenvalues = eigh(laplacian, subset_by_index=[0, search_upper - 1], eigvals_only=True)
        except (TypeError, ValueError):
            eigenvalues = eigh(laplacian, eigvals_only=True)[:search_upper]
        eigenvalues = np.sort(eigenvalues)
        gaps = np.diff(eigenvalues)
        candidate_gaps = gaps[min_k - 1:max_k]
        if candidate_gaps.size == 0:
            return min_k
        best_k = min_k + int(np.argmax(candidate_gaps))
        best_k = int(np.clip(best_k, min_k, max_k))
        logger.debug('Eigengap heuristic: eigenvalues=%s -> k=%d', np.round(eigenvalues, 4), best_k)
        return best_k