"""
Recap Generator Module

This module provides functionality for generating recap videos from TV episode data.
It combines narrative arc analysis, vector search, and video processing to create
contextually relevant episode recaps.
"""

from .models.recap_models import (
    RecapEvent,
    RecapClip,
    RecapConfiguration,
    RecapMetadata
)

from .models.event_models import (
    VectorEvent,
    SubtitleSequence,
    EventRanking
)

from .exceptions.recap_exceptions import (
    RecapGenerationError,
    MissingInputFilesError,
    VideoProcessingError,
    SubtitleProcessingError
)

__all__ = [
    # Models
    "RecapEvent",
    "RecapClip", 
    "RecapConfiguration",
    "RecapMetadata",
    "VectorEvent",
    "SubtitleSequence", 
    "EventRanking",
    # Exceptions
    "RecapGenerationError",
    "MissingInputFilesError",
    "VideoProcessingError",
    "SubtitleProcessingError",
]
