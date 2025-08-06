"""
Service layer for SEMAMORPH API.

This module provides centralized access to all processing services.
"""

from .episode_processing_service import EpisodeProcessingService, ProcessingResult
from .background_job_service import BackgroundJobService, ProcessingJob, ProcessingRequest, JobStatus
from .processing_pipeline import ProcessingPipeline
from .exceptions import (
    ProcessingError,
    SRTFileNotFoundError,
    EntityExtractionError,
    NarrativeExtractionError,
    SeasonSummaryError,
    SemanticSegmentationError,
    PronounReplacementError,
    PlotGenerationError,
    ConfigurationError,
    ValidationError
)

__all__ = [
    'EpisodeProcessingService',
    'ProcessingResult',
    'BackgroundJobService', 
    'ProcessingJob',
    'ProcessingRequest',
    'JobStatus',
    'ProcessingPipeline',
    'ProcessingError',
    'SRTFileNotFoundError',
    'EntityExtractionError',
    'NarrativeExtractionError',
    'SeasonSummaryError',
    'SemanticSegmentationError',
    'PronounReplacementError',
    'PlotGenerationError',
    'ConfigurationError',
    'ValidationError'
]
