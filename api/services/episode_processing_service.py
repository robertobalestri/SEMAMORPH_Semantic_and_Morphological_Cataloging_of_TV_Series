"""
Episode processing service for SEMAMORPH API.

This service provides a high-level interface for processing episodes,
handling validation, error management, and progress reporting.
"""

import asyncio
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

from backend.src.config import Config
from backend.src.utils.logger_utils import setup_logging
from .processing_pipeline import ProcessingPipeline
from .exceptions import ProcessingError, ValidationError


class ProcessingStatus(str, Enum):
    """Status of episode processing."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """Result of episode processing operation."""
    series: str
    season: str
    episode: str
    status: ProcessingStatus
    message: str
    files_created: list[str]
    entities_extracted: int
    narrative_arcs_found: int
    steps_completed: list[str]
    error_details: Optional[Dict[str, Any]] = None


class EpisodeProcessingService:
    """
    High-level service for processing individual episodes.
    
    This service wraps the ProcessingPipeline with additional features:
    - Input validation
    - Progress reporting
    - Error handling and recovery
    - Result formatting
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the episode processing service.
        
        Args:
            config: Configuration instance (uses default if None)
        """
        self.config = config or Config()
        self.logger = setup_logging(self.__class__.__name__)
        self.pipeline = ProcessingPipeline(self.config)
        
    async def process_episode(
        self,
        series: str,
        season: str,
        episode: str,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> ProcessingResult:
        """
        Process a single episode through the complete pipeline.
        
        Args:
            series: Series name (e.g., "GA")
            season: Season name (e.g., "S01")
            episode: Episode name (e.g., "E01")
            progress_callback: Optional callback for progress updates
            
        Returns:
            ProcessingResult with operation details and status
            
        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing fails
        """
        try:
            # Validate inputs
            self._validate_episode_inputs(series, season, episode)
            
            self.logger.info(f"🚀 Starting episode processing: {series} {season} {episode}")
            
            if progress_callback:
                progress_callback("VALIDATION", "Input validation completed")
            
            # Process the episode
            results = await self.pipeline.process_episode_complete(
                series, season, episode, progress_callback
            )
            
            # Create successful result
            result = ProcessingResult(
                series=series,
                season=season,
                episode=episode,
                status=ProcessingStatus.COMPLETED,
                message=f"Successfully processed {series} {season} {episode}",
                files_created=results.get("files_created", []),
                entities_extracted=results.get("entities_extracted", 0),
                narrative_arcs_found=results.get("narrative_arcs_found", 0),
                steps_completed=results.get("steps_completed", [])
            )
            
            self.logger.info(f"✅ Episode processing completed: {series} {season} {episode}")
            return result
            
        except ValidationError as e:
            self.logger.error(f"❌ Validation failed for {series} {season} {episode}: {e}")
            return ProcessingResult(
                series=series,
                season=season,
                episode=episode,
                status=ProcessingStatus.FAILED,
                message=str(e),
                files_created=[],
                entities_extracted=0,
                narrative_arcs_found=0,
                steps_completed=[],
                error_details={
                    "error_type": "ValidationError",
                    "step": e.step,
                    "context": e.context
                }
            )
            
        except ProcessingError as e:
            self.logger.error(f"❌ Processing failed for {series} {season} {episode}: {e}")
            return ProcessingResult(
                series=series,
                season=season,
                episode=episode,
                status=ProcessingStatus.FAILED,
                message=str(e),
                files_created=[],
                entities_extracted=0,
                narrative_arcs_found=0,
                steps_completed=[],
                error_details={
                    "error_type": type(e).__name__,
                    "step": e.step,
                    "context": e.context
                }
            )
            
        except Exception as e:
            error_msg = f"Unexpected error processing {series} {season} {episode}: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(
                series=series,
                season=season,
                episode=episode,
                status=ProcessingStatus.FAILED,
                message=error_msg,
                files_created=[],
                entities_extracted=0,
                narrative_arcs_found=0,
                steps_completed=[],
                error_details={
                    "error_type": "UnexpectedError",
                    "step": "UNKNOWN",
                    "context": {"exception": str(e)}
                }
            )
    
    def _validate_episode_inputs(self, series: str, season: str, episode: str) -> None:
        """
        Validate episode processing inputs.
        
        Args:
            series: Series name
            season: Season name
            episode: Episode name
            
        Raises:
            ValidationError: If any input is invalid
        """
        if not series or not series.strip():
            raise ValidationError("Series name cannot be empty", field="series", value=series)
            
        if not season or not season.strip():
            raise ValidationError("Season name cannot be empty", field="season", value=season)
            
        if not episode or not episode.strip():
            raise ValidationError("Episode name cannot be empty", field="episode", value=episode)
            
        # Validate format patterns
        if not season.startswith('S'):
            raise ValidationError(
                "Season must start with 'S' (e.g., S01)", 
                field="season", 
                value=season
            )
            
        if not episode.startswith('E'):
            raise ValidationError(
                "Episode must start with 'E' (e.g., E01)", 
                field="episode", 
                value=episode
            )
        
        # Validate season number format
        try:
            season_num = int(season[1:])
            if season_num < 1 or season_num > 99:
                raise ValidationError(
                    "Season number must be between 1 and 99",
                    field="season",
                    value=season
                )
        except ValueError:
            raise ValidationError(
                "Season must be in format S01, S02, etc.",
                field="season", 
                value=season
            )
        
        # Validate episode number format  
        try:
            episode_num = int(episode[1:])
            if episode_num < 1 or episode_num > 99:
                raise ValidationError(
                    "Episode number must be between 1 and 99",
                    field="episode",
                    value=episode
                )
        except ValueError:
            raise ValidationError(
                "Episode must be in format E01, E02, etc.",
                field="episode",
                value=episode
            )
    
    def validate_episode_inputs(self, series: str, season: str, episode: str) -> bool:
        """
        Public method to validate inputs without raising exceptions.
        
        Args:
            series: Series name
            season: Season name  
            episode: Episode name
            
        Returns:
            True if inputs are valid, False otherwise
        """
        try:
            self._validate_episode_inputs(series, season, episode)
            return True
        except ValidationError:
            return False
    
    async def check_episode_requirements(self, series: str, season: str, episode: str) -> Dict[str, Any]:
        """
        Check if episode has the required files for processing.
        
        Args:
            series: Series name
            season: Season name
            episode: Episode name
            
        Returns:
            Dictionary with requirement check results
        """
        from backend.src.path_handler import PathHandler
        import os
        
        try:
            path_handler = PathHandler(series, season, episode)
            
            requirements = {
                "srt_file": {
                    "path": path_handler.get_srt_file_path(),
                    "exists": False,
                    "required": True
                },
                "plot_file": {
                    "path": path_handler.get_raw_plot_file_path(),
                    "exists": False,
                    "required": False
                }
            }
            
            # Check file existence
            for req_name, req_info in requirements.items():
                req_info["exists"] = os.path.exists(req_info["path"])
            
            # Determine overall readiness
            missing_required = [
                name for name, info in requirements.items() 
                if info["required"] and not info["exists"]
            ]
            
            return {
                "ready_for_processing": len(missing_required) == 0,
                "missing_required_files": missing_required,
                "requirements": requirements
            }
            
        except Exception as e:
            self.logger.error(f"Error checking episode requirements: {e}")
            return {
                "ready_for_processing": False,
                "missing_required_files": ["unknown"],
                "requirements": {},
                "error": str(e)
            }
