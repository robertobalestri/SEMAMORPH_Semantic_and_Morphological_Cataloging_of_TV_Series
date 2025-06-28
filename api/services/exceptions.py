"""
Custom exceptions for SEMAMORPH processing pipeline.

This module defines the exception hierarchy for processing operations,
providing structured error handling with detailed context.
"""

from typing import Optional, Dict, Any


class ProcessingError(Exception):
    """Base exception for all processing-related errors."""
    
    def __init__(
        self, 
        message: str, 
        step: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize processing error.
        
        Args:
            message: Human-readable error description
            step: Processing step where error occurred
            context: Additional context information
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.step = step
        self.context = context or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """Return formatted error message."""
        base_msg = super().__str__()
        if self.step:
            base_msg = f"[{self.step}] {base_msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (context: {context_str})"
        return base_msg


class SRTFileNotFoundError(ProcessingError):
    """Raised when SRT subtitle file is not found."""
    
    def __init__(self, file_path: str, series: str, season: str, episode: str):
        super().__init__(
            f"SRT file not found: {file_path}",
            step="SRT_PROCESSING",
            context={
                "file_path": file_path,
                "series": series,
                "season": season,
                "episode": episode
            }
        )


class EntityExtractionError(ProcessingError):
    """Raised when entity extraction fails."""
    
    def __init__(self, message: str, plot_file: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(
            f"Entity extraction failed: {message}",
            step="ENTITY_EXTRACTION",
            context={"plot_file": plot_file} if plot_file else {},
            cause=cause
        )


class NarrativeExtractionError(ProcessingError):
    """Raised when narrative arc extraction fails."""
    
    def __init__(self, message: str, semantic_segments_file: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(
            f"Narrative arc extraction failed: {message}",
            step="NARRATIVE_EXTRACTION", 
            context={"semantic_segments_file": semantic_segments_file} if semantic_segments_file else {},
            cause=cause
        )


class SeasonSummaryError(ProcessingError):
    """Raised when season summary creation fails."""
    
    def __init__(self, message: str, episode_path: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(
            f"Season summary creation failed: {message}",
            step="SEASON_SUMMARY",
            context={"episode_path": episode_path} if episode_path else {},
            cause=cause
        )


class SemanticSegmentationError(ProcessingError):
    """Raised when semantic segmentation fails."""
    
    def __init__(self, message: str, text_length: Optional[int] = None, cause: Optional[Exception] = None):
        super().__init__(
            f"Semantic segmentation failed: {message}",
            step="SEMANTIC_SEGMENTATION",
            context={"text_length": text_length} if text_length else {},
            cause=cause
        )


class PronounReplacementError(ProcessingError):
    """Raised when pronoun replacement fails."""
    
    def __init__(self, message: str, text_length: Optional[int] = None, cause: Optional[Exception] = None):
        super().__init__(
            f"Pronoun replacement failed: {message}",
            step="PRONOUN_REPLACEMENT",
            context={"text_length": text_length} if text_length else {},
            cause=cause
        )


class PlotGenerationError(ProcessingError):
    """Raised when plot generation from SRT fails."""
    
    def __init__(self, message: str, srt_file: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(
            f"Plot generation failed: {message}",
            step="PLOT_GENERATION",
            context={"srt_file": srt_file} if srt_file else {},
            cause=cause
        )


class ConfigurationError(ProcessingError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            f"Configuration error: {message}",
            step="CONFIGURATION",
            context={"config_key": config_key} if config_key else {}
        )


class ValidationError(ProcessingError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[str] = None):
        super().__init__(
            f"Validation error: {message}",
            step="VALIDATION",
            context={"field": field, "value": value} if field else {}
        )
