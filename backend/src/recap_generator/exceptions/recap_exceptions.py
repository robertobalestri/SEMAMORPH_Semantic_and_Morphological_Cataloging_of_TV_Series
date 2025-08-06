"""
Exception classes for recap generation.

This module defines custom exceptions used throughout the recap generation pipeline
to provide clear error handling and debugging information.
"""

from typing import List, Optional, Dict, Any


class RecapGenerationError(Exception):
    """Base exception for recap generation errors."""
    
    def __init__(self, message: str, series: str = None, season: str = None, episode: str = None):
        self.message = message
        self.series = series
        self.season = season
        self.episode = episode
        super().__init__(message)
    
    @property
    def episode_identifier(self) -> str:
        """Get episode identifier string."""
        if all([self.series, self.season, self.episode]):
            return f"{self.series}{self.season}{self.episode}"
        return "Unknown Episode"


class MissingInputFilesError(RecapGenerationError):
    """Raised when required input files are missing for recap generation."""
    
    def __init__(self, missing_files: List[str], series: str = None, season: str = None, episode: str = None):
        self.missing_files = missing_files
        files_str = ", ".join(missing_files)
        message = f"Missing required input files: {files_str}"
        super().__init__(message, series, season, episode)
    
    @property
    def critical_files_missing(self) -> bool:
        """Check if any critical files are missing."""
        critical_files = [
            "_plot_possible_speakers.txt",
            "_present_running_plotlines.json", 
            "_possible_speakers.srt",
            ".mp4"
        ]
        return any(any(critical in file for critical in critical_files) for file in self.missing_files)


class VideoProcessingError(RecapGenerationError):
    """Raised when video processing operations fail."""
    
    def __init__(self, 
                 message: str, 
                 operation: str = None,
                 ffmpeg_command: str = None,
                 return_code: int = None,
                 stderr_output: str = None,
                 series: str = None, 
                 season: str = None, 
                 episode: str = None):
        self.operation = operation
        self.ffmpeg_command = ffmpeg_command
        self.return_code = return_code
        self.stderr_output = stderr_output
        
        if operation:
            message = f"Video processing failed during {operation}: {message}"
        
        super().__init__(message, series, season, episode)
    
    @property
    def is_ffmpeg_error(self) -> bool:
        """Check if this is an FFmpeg-related error."""
        return self.ffmpeg_command is not None
    
    @property
    def debug_info(self) -> Dict[str, Any]:
        """Get debug information for troubleshooting."""
        return {
            "operation": self.operation,
            "ffmpeg_command": self.ffmpeg_command,
            "return_code": self.return_code,
            "stderr_output": self.stderr_output,
            "episode": self.episode_identifier
        }


class SubtitleProcessingError(RecapGenerationError):
    """Raised when subtitle processing operations fail."""
    
    def __init__(self, 
                 message: str,
                 subtitle_file: str = None,
                 line_number: int = None,
                 timestamp_range: tuple = None,
                 series: str = None,
                 season: str = None, 
                 episode: str = None):
        self.subtitle_file = subtitle_file
        self.line_number = line_number
        self.timestamp_range = timestamp_range
        
        if subtitle_file:
            message = f"Subtitle processing failed for {subtitle_file}: {message}"
        if line_number:
            message += f" (line {line_number})"
        if timestamp_range:
            message += f" (timestamp range: {timestamp_range[0]}-{timestamp_range[1]})"
            
        super().__init__(message, series, season, episode)
    
    @property
    def has_location_info(self) -> bool:
        """Check if error has specific location information."""
        return self.line_number is not None or self.timestamp_range is not None


class LLMServiceError(RecapGenerationError):
    """Raised when LLM service operations fail."""
    
    def __init__(self, 
                 message: str,
                 service_name: str = None,
                 model_name: str = None,
                 prompt_tokens: int = None,
                 response_tokens: int = None,
                 retry_count: int = 0,
                 original_error: Exception = None,
                 series: str = None,
                 season: str = None,
                 episode: str = None):
        self.service_name = service_name
        self.model_name = model_name
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.retry_count = retry_count
        self.original_error = original_error
        
        if service_name:
            message = f"LLM service '{service_name}' failed: {message}"
        if retry_count > 0:
            message += f" (after {retry_count} retries)"
            
        super().__init__(message, series, season, episode)
    
    @property
    def token_info(self) -> Dict[str, int]:
        """Get token usage information."""
        return {
            "prompt_tokens": self.prompt_tokens or 0,
            "response_tokens": self.response_tokens or 0,
            "total_tokens": (self.prompt_tokens or 0) + (self.response_tokens or 0)
        }


class VectorSearchError(RecapGenerationError):
    """Raised when vector database search operations fail."""
    
    def __init__(self, 
                 message: str,
                 query: str = None,
                 collection_name: str = None,
                 filter_criteria: Dict = None,
                 original_error: Exception = None,
                 series: str = None,
                 season: str = None,
                 episode: str = None):
        self.query = query
        self.collection_name = collection_name
        self.filter_criteria = filter_criteria
        self.original_error = original_error
        
        if collection_name:
            message = f"Vector search failed in collection '{collection_name}': {message}"
        if query:
            message += f" (query: '{query[:100]}...' if len(query) > 100 else query)"
            
        super().__init__(message, series, season, episode)


class ConfigurationError(RecapGenerationError):
    """Raised when recap configuration is invalid."""
    
    def __init__(self, 
                 message: str,
                 parameter_name: str = None,
                 parameter_value: Any = None,
                 valid_range: tuple = None):
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.valid_range = valid_range
        
        if parameter_name:
            message = f"Invalid configuration for '{parameter_name}': {message}"
        if parameter_value is not None:
            message += f" (value: {parameter_value})"
        if valid_range:
            message += f" (valid range: {valid_range[0]}-{valid_range[1]})"
            
        super().__init__(message)


class OutputQualityError(RecapGenerationError):
    """Raised when generated recap doesn't meet quality standards."""
    
    def __init__(self, 
                 message: str,
                 quality_metrics: Dict[str, float] = None,
                 failed_checks: List[str] = None,
                 series: str = None,
                 season: str = None,
                 episode: str = None):
        self.quality_metrics = quality_metrics or {}
        self.failed_checks = failed_checks or []
        
        if failed_checks:
            checks_str = ", ".join(failed_checks)
            message = f"Recap quality checks failed ({checks_str}): {message}"
            
        super().__init__(message, series, season, episode)
    
    @property
    def has_quality_data(self) -> bool:
        """Check if quality metrics are available."""
        return bool(self.quality_metrics)


class ResourceConstraintError(RecapGenerationError):
    """Raised when system resources are insufficient for recap generation."""
    
    def __init__(self, 
                 message: str,
                 resource_type: str = None,
                 required_amount: str = None,
                 available_amount: str = None,
                 series: str = None,
                 season: str = None,
                 episode: str = None):
        self.resource_type = resource_type
        self.required_amount = required_amount
        self.available_amount = available_amount
        
        if resource_type:
            message = f"Insufficient {resource_type}: {message}"
        if required_amount and available_amount:
            message += f" (required: {required_amount}, available: {available_amount})"
            
        super().__init__(message, series, season, episode)
