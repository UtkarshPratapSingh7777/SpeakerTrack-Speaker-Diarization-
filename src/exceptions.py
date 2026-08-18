from __future__ import annotations
class DiarizationPipelineError(Exception):
    pass

class ConfigurationError(DiarizationPipelineError):
    pass

class AudioProcessingError(DiarizationPipelineError):
    pass

class UnsupportedAudioFormatError(AudioProcessingError):
    pass

class CorruptedAudioError(AudioProcessingError):
    pass

class AudioTooShortError(AudioProcessingError):
    pass

class VADProcessingError(DiarizationPipelineError):
    pass

class EmbeddingExtractionError(DiarizationPipelineError):
    pass

class ClusteringError(DiarizationPipelineError):
    pass

class PipelineError(DiarizationPipelineError):
    pass