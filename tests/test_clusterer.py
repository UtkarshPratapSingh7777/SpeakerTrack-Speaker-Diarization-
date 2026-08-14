from __future__ import annotations
import numpy as np
import pytest
from src.clusterer import ClusteringConfig, SpeakerClusterer
from src.exceptions import ClusteringError

def _two_speaker_embeddings(n_per_speaker: int=8, dim: int=16, seed: int=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center_a = np.zeros(dim)
    center_a[0] = 1.0
    center_b = np.zeros(dim)
    center_b[1] = 1.0
    cluster_a = center_a + 0.05 * rng.standard_normal((n_per_speaker, dim))
    cluster_b = center_b + 0.05 * rng.standard_normal((n_per_speaker, dim))
    embeddings = np.vstack([cluster_a, cluster_b]).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def test_distance_matrix_is_symmetric_with_zero_diagonal() -> None:
    embeddings = _two_speaker_embeddings()
    distance_matrix = SpeakerClusterer._compute_distance_matrix(embeddings)
    assert np.allclose(distance_matrix, distance_matrix.T, atol=1e-06)
    assert np.allclose(np.diag(distance_matrix), 0.0, atol=1e-06)

def test_agglomerative_clustering_with_known_speaker_count() -> None:
    embeddings = _two_speaker_embeddings()
    clusterer = SpeakerClusterer(ClusteringConfig(method='agglomerative', auto_estimate_speakers=False))
    labels = clusterer.cluster(embeddings, num_speakers=2)
    assert len(set(labels.tolist())) == 2
    assert len(set(labels[:8].tolist())) == 1
    assert len(set(labels[8:].tolist())) == 1
    assert labels[0] != labels[8]

def test_agglomerative_clustering_auto_estimates_two_speakers() -> None:
    embeddings = _two_speaker_embeddings()
    clusterer = SpeakerClusterer(ClusteringConfig(method='agglomerative', auto_estimate_speakers=True, min_speakers=1, max_speakers=5))
    labels = clusterer.cluster(embeddings, num_speakers=None)
    assert len(set(labels.tolist())) == 2

def test_spectral_clustering_with_known_speaker_count() -> None:
    embeddings = _two_speaker_embeddings()
    clusterer = SpeakerClusterer(ClusteringConfig(method='spectral', auto_estimate_speakers=False))
    labels = clusterer.cluster(embeddings, num_speakers=2)
    assert len(set(labels.tolist())) == 2

def test_single_embedding_returns_single_label() -> None:
    embeddings = np.random.randn(1, 16).astype(np.float32)
    clusterer = SpeakerClusterer(ClusteringConfig())
    labels = clusterer.cluster(embeddings)
    assert labels.tolist() == [0]

def test_empty_embeddings_raises_clustering_error() -> None:
    clusterer = SpeakerClusterer(ClusteringConfig())
    with pytest.raises(ClusteringError):
        clusterer.cluster(np.empty((0, 16)))

def test_unknown_method_raises_clustering_error() -> None:
    embeddings = _two_speaker_embeddings()
    clusterer = SpeakerClusterer(ClusteringConfig(method='not_a_real_method', auto_estimate_speakers=False))
    with pytest.raises(ClusteringError):
        clusterer.cluster(embeddings, num_speakers=2)