"""
Recap Orchestrator - Main coordinator for recap generation pipeline.

This module orchestrates the entire recap generation process, from analysis
to final video assembly, with comprehensive error handling and progress tracking.
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..utils.logger_utils import setup_logging
from .exceptions.recap_exceptions import (
    RecapGenerationError, MissingInputFilesError, VideoProcessingError,
    LLMServiceError, ConfigurationError
)
from .models.recap_models import RecapConfiguration, RecapMetadata
from .utils.validation_utils import ValidationUtils
from ..path_handler import PathHandler

# Import recap generation services
from .query_generator import QueryGeneratorService
from .event_retrieval_service import EventRetrievalService
from .subtitle_processor import SubtitleProcessorService
from .video_clip_extractor import VideoClipExtractor
from .recap_assembler import RecapAssembler

logger = setup_logging(__name__)


class RecapOrchestrator:
    """Main orchestrator for the recap generation pipeline."""
    
    def __init__(self, series: str, season: str, episode: str, 
                 base_dir: str = "data", config: Optional[RecapConfiguration] = None):
        """
        Initialize the recap orchestrator.
        
        Args:
            series: Series identifier
            season: Season identifier  
            episode: Episode identifier
            base_dir: Base data directory
            config: Recap configuration (uses defaults if not provided)
        """
        self.series = series
        self.season = season
        self.episode = episode
        self.path_handler = PathHandler(series, season, episode, base_dir)
        
        # Use provided config or create default
        self.config = config or RecapConfiguration()
        
        # Initialize services (lazy loading)
        self._query_generator = None
        self._event_retrieval = None
        self._subtitle_processor = None
        self._clip_extractor = None
        self._assembler = None
        
        # Processing state
        self.processing_metadata = {
            'start_time': None,
            'llm_queries_count': 0,
            'vector_search_count': 0,
            'ffmpeg_operations_count': 0,
            'steps_completed': [],
            'errors': [],
            'warnings': []
        }
    
    def generate_recap(self) -> RecapMetadata:
        """
        Execute the complete recap generation pipeline.
        
        Returns:
            RecapMetadata with generation results and file paths
        """
        try:
            logger.info(f"🎬 Starting recap generation for {self.series}{self.season}{self.episode}")
            self.processing_metadata['start_time'] = datetime.now()
            
            # Step 1: Validate prerequisites
            self._validate_input_files()
            self._mark_step_completed("input_validation")
            
            # Step 2: Generate queries for each narrative arc
            arc_queries = self._generate_queries_for_narrative_arcs()
            self._mark_step_completed("query_generation")
            
            # Step 3: Search vector database and rank events
            query_results = self._search_vector_database(arc_queries)
            event_rankings = self._rank_events_by_relevance(query_results)
            selected_events = self._select_final_events(event_rankings)
            self._mark_step_completed("event_selection")
            
            # Step 4: Process subtitles for selected events
            subtitle_sequences = self._process_event_subtitles(selected_events)
            optimized_sequences = self._optimize_sequence_timing(subtitle_sequences)
            self._mark_step_completed("subtitle_processing")
            
            # Step 5: Extract video clips
            extracted_clips = self._extract_video_clips(selected_events, optimized_sequences)
            normalized_clips = self._normalize_audio_levels(extracted_clips)
            self._mark_step_completed("clip_extraction")
            
            # Step 6: Assemble final recap
            self.processing_metadata['recap_events'] = [event.dict() for event in selected_events]
            final_metadata = self._assemble_final_recap(normalized_clips, selected_events)
            self._mark_step_completed("final_assembly")
            
            # Step 7: Final validation
            validation_results = self._validate_final_output(final_metadata)
            self._mark_step_completed("final_validation")
            
            # Update metadata with processing information
            final_metadata.processing_metadata = self.processing_metadata
            final_metadata.validation_results = validation_results
            
            total_time = (datetime.now() - self.processing_metadata['start_time']).total_seconds()
            logger.info(f"✅ Recap generation completed successfully in {total_time:.2f}s")
            logger.info(f"📁 Final video: {final_metadata.file_paths.get('final_video', 'Unknown')}")
            
            return final_metadata
            
        except Exception as e:
            # Log error and create error metadata
            total_time = 0
            if self.processing_metadata['start_time']:
                total_time = (datetime.now() - self.processing_metadata['start_time']).total_seconds()
            
            self.processing_metadata['errors'].append({
                'error': str(e),
                'step': len(self.processing_metadata['steps_completed']),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.error(f"❌ Recap generation failed: {e}")
            
            # Create error metadata
            error_metadata = RecapMetadata(
                series=self.series,
                season=self.season,
                episode=self.episode,
                configuration=self.config,
                events=[],
                clips=[],
                total_duration=0,
                processing_time_seconds=total_time,
                success=False,
                error_message=str(e),
                processing_metadata=self.processing_metadata
            )
            
            return error_metadata
    
    def validate_episode_prerequisites(self) -> Dict[str, Any]:
        """
        Validate episode prerequisites without running full generation.
        
        Returns:
            Dictionary with detailed validation results
        """
        try:
            logger.info(f"🔍 Validating prerequisites for {self.series}{self.season}{self.episode}")
            
            validation_utils = ValidationUtils()
            prerequisites = validation_utils.validate_episode_prerequisites(self.path_handler)
            
            # Add configuration validation
            config_validation = validation_utils.validate_recap_configuration(self.config)
            prerequisites['configuration_validation'] = config_validation
            
            return prerequisites
            
        except Exception as e:
            logger.error(f"❌ Prerequisites validation failed: {e}")
            return {
                'ready_for_recap': False,
                'error': str(e),
                'issues_found': True
            }
    
    # Private methods for pipeline steps
    
    def _validate_input_files(self) -> None:
        """Validate that all required input files exist and are valid."""
        logger.info("🔍 STEP 1: VALIDATING INPUT FILES")
        
        validation_results = self.validate_episode_prerequisites()
        
        if not validation_results.get('ready_for_recap', False):
            missing_files = []
            for file_info in validation_results.get('required_files', {}).get('missing', []):
                missing_files.append(file_info['path'])
            
            raise MissingInputFilesError(
                missing_files,
                series=self.series,
                season=self.season,
                episode=self.episode
            )
        
        # Check configuration validity
        config_validation = validation_results.get('configuration_validation', {})
        if not config_validation.get('valid', True):
            issues = config_validation.get('issues', [])
            raise ConfigurationError(
                f"Invalid configuration: {'; '.join(issues)}",
                parameter_name="recap_configuration"
            )
        
        logger.info("✅ Input validation completed successfully")
    
    def _generate_queries_for_narrative_arcs(self) -> List[Dict[str, Any]]:
        """Generate targeted vector database queries for each narrative arc."""
        logger.info("🎯 STEP 2: GENERATING QUERIES FOR NARRATIVE ARCS")
        
        query_generator = self._get_query_generator()
        arc_queries = query_generator.generate_queries_for_narrative_arcs()
        
        # Count total LLM queries (one per narrative arc)
        total_queries = sum(len(arc['queries']) for arc in arc_queries)
        self.processing_metadata['llm_queries_count'] += len(arc_queries)  # One LLM call per arc
        
        logger.info(f"✅ Generated queries for {len(arc_queries)} narrative arcs, "
                   f"total of {total_queries} vector database queries")
        
        return arc_queries
    
    def _search_vector_database(self, arc_queries: List[Dict[str, Any]]) -> List[Any]:
        """Execute vector database searches for all narrative arcs."""
        logger.info("🔍 STEP 3: SEARCHING VECTOR DATABASE")
        
        # Flatten arc queries into individual queries for the event retrieval service
        all_queries = []
        for arc_query_data in arc_queries:
            queries = arc_query_data.get('queries', [])
            all_queries.extend(queries)
        
        event_retrieval = self._get_event_retrieval()
        query_results = event_retrieval.search_vector_database(all_queries)
        
        self.processing_metadata['vector_search_count'] += len(all_queries)
        
        total_events = sum(len(result.events) for result in query_results)
        logger.info(f"✅ Vector search completed: {total_events} events found across {len(all_queries)} queries "
                   f"from {len(arc_queries)} narrative arcs")
        
        return query_results
    
    def _rank_events_by_relevance(self, query_results: List[Any]) -> List[Any]:
        """Rank events by relevance and importance."""
        logger.info("📊 STEP 3b: RANKING EVENTS BY RELEVANCE")
        
        event_retrieval = self._get_event_retrieval()
        event_rankings = event_retrieval.rank_events_by_relevance(
            query_results, 
            max_events=self.config.max_events * 2  # Get more for better selection
        )
        
        self.processing_metadata['llm_queries_count'] += 1  # LLM used for ranking
        
        logger.info(f"✅ Event ranking completed: {len(event_rankings)} events ranked")
        
        return event_rankings
    
    def _select_final_events(self, event_rankings: List[Any]) -> List[Any]:
        """Select final set of events for recap."""
        logger.info("🎯 STEP 3c: SELECTING FINAL EVENTS")
        
        event_retrieval = self._get_event_retrieval()
        selection_result = event_retrieval.select_final_events(
            event_rankings,
            target_count=self.config.max_events,
            min_score=self.config.relevance_threshold
        )
        
        selected_events = selection_result.selected_events
        arc_distribution = selection_result.arc_distribution
        
        logger.info(f"✅ Event selection completed: {len(selected_events)} events selected")
        logger.info(f"📊 Arc distribution: {arc_distribution}")
        
        return selected_events
    
    def _process_event_subtitles(self, selected_events: List[Any]) -> List[Any]:
        """Process subtitles for selected events."""
        logger.info("📝 STEP 4: PROCESSING EVENT SUBTITLES")
        
        subtitle_processor = self._get_subtitle_processor()
        subtitle_sequences = subtitle_processor.process_event_subtitles(selected_events)
        
        logger.info(f"✅ Subtitle processing completed: {len(subtitle_sequences)} sequences extracted")
        
        return subtitle_sequences
    
    def _optimize_sequence_timing(self, subtitle_sequences: List[Any]) -> List[Any]:
        """Optimize subtitle sequence timing."""
        logger.info("⚙️ STEP 4b: OPTIMIZING SEQUENCE TIMING")
        
        subtitle_processor = self._get_subtitle_processor()
        optimized_sequences = subtitle_processor.optimize_sequence_timing(subtitle_sequences)
        
        self.processing_metadata['llm_queries_count'] += len(subtitle_sequences)  # LLM used per sequence
        
        total_duration = sum(seq.duration for seq in optimized_sequences)
        logger.info(f"✅ Sequence timing optimization completed: {total_duration:.1f}s total duration")
        
        return optimized_sequences
    
    def _extract_video_clips(self, selected_events: List[Any], subtitle_sequences: List[Any]) -> List[Any]:
        """Extract video clips for events."""
        logger.info("🎬 STEP 5: EXTRACTING VIDEO CLIPS")
        
        clip_extractor = self._get_clip_extractor()
        extracted_clips = clip_extractor.extract_clips_for_events(selected_events, subtitle_sequences)
        
        self.processing_metadata['ffmpeg_operations_count'] += len(extracted_clips)
        
        logger.info(f"✅ Video clip extraction completed: {len(extracted_clips)} clips extracted")
        
        return extracted_clips
    
    def _normalize_audio_levels(self, clips: List[Any]) -> List[Any]:
        """Normalize audio levels across clips."""
        logger.info("🔊 STEP 5b: NORMALIZING AUDIO LEVELS")
        
        clip_extractor = self._get_clip_extractor()
        normalized_clips = clip_extractor.normalize_audio_levels(clips)
        
        self.processing_metadata['ffmpeg_operations_count'] += len(clips)  # Audio processing per clip
        
        logger.info(f"✅ Audio normalization completed")
        
        return normalized_clips
    
    def _assemble_final_recap(self, clips: List[Any], events: List[Any]) -> RecapMetadata:
        """Assemble final recap video."""
        logger.info("🎬 STEP 6: ASSEMBLING FINAL RECAP")
        
        assembler = self._get_assembler()
        final_metadata = assembler.assemble_final_recap(clips, events, self.processing_metadata)
        
        self.processing_metadata['ffmpeg_operations_count'] += 3  # Subtitle overlay + concatenation + optimization
        
        logger.info(f"✅ Final recap assembly completed")
        
        return final_metadata
    
    def _validate_final_output(self, metadata: RecapMetadata) -> Dict[str, Any]:
        """Validate final output quality."""
        logger.info("🔍 STEP 7: VALIDATING FINAL OUTPUT")
        
        assembler = self._get_assembler()
        validation_results = assembler.validate_final_output(metadata)
        
        # Also use validation utils for comprehensive checks
        validation_utils = ValidationUtils()
        quality_results = validation_utils.validate_output_quality(metadata)
        
        # Combine results
        combined_results = {
            'assembly_validation': validation_results,
            'quality_validation': quality_results,
            'overall_success': validation_results.get('overall_success', False) and 
                              quality_results.get('overall_quality_score', 0) >= 0.6
        }
        
        logger.info(f"✅ Final output validation completed: {combined_results['overall_success']}")
        
        return combined_results
    
    def _mark_step_completed(self, step_name: str) -> None:
        """Mark a pipeline step as completed."""
        self.processing_metadata['steps_completed'].append({
            'step': step_name,
            'timestamp': datetime.now().isoformat()
        })
    
    # Service getters (lazy initialization)
    
    def _get_query_generator(self) -> QueryGeneratorService:
        """Get or create query generator service."""
        if self._query_generator is None:
            self._query_generator = QueryGeneratorService(self.path_handler)
        return self._query_generator
    
    def _get_event_retrieval(self) -> EventRetrievalService:
        """Get or create event retrieval service."""
        if self._event_retrieval is None:
            # Import vector store service
            from ..narrative_storage_management.vector_store_service import VectorStoreService
            vector_store = VectorStoreService()
            self._event_retrieval = EventRetrievalService(self.path_handler, vector_store)
        return self._event_retrieval
    
    def _get_subtitle_processor(self) -> SubtitleProcessorService:
        """Get or create subtitle processor service."""
        if self._subtitle_processor is None:
            self._subtitle_processor = SubtitleProcessorService(self.path_handler)
        return self._subtitle_processor
    
    def _get_clip_extractor(self) -> VideoClipExtractor:
        """Get or create clip extractor service."""
        if self._clip_extractor is None:
            self._clip_extractor = VideoClipExtractor(self.path_handler, self.config)
        return self._clip_extractor
    
    def _get_assembler(self) -> RecapAssembler:
        """Get or create recap assembler service."""
        if self._assembler is None:
            self._assembler = RecapAssembler(self.path_handler, self.config)
        return self._assembler
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get current progress information.
        
        Returns:
            Dictionary with progress details
        """
        total_steps = 7  # Total number of main steps
        completed_steps = len(self.processing_metadata['steps_completed'])
        
        progress_percent = (completed_steps / total_steps) * 100
        
        current_step = None
        if completed_steps < total_steps:
            step_names = [
                "Input Validation",
                "Episode Analysis & Query Generation", 
                "Event Search & Selection",
                "Subtitle Processing",
                "Video Clip Extraction",
                "Final Assembly",
                "Output Validation"
            ]
            current_step = step_names[completed_steps] if completed_steps < len(step_names) else "Unknown"
        
        elapsed_time = 0
        if self.processing_metadata['start_time']:
            elapsed_time = (datetime.now() - self.processing_metadata['start_time']).total_seconds()
        
        return {
            'progress_percent': progress_percent,
            'completed_steps': completed_steps,
            'total_steps': total_steps,
            'current_step': current_step,
            'elapsed_time_seconds': elapsed_time,
            'steps_completed': [step['step'] for step in self.processing_metadata['steps_completed']],
            'errors_count': len(self.processing_metadata['errors']),
            'warnings_count': len(self.processing_metadata['warnings']),
            'llm_queries_count': self.processing_metadata['llm_queries_count'],
            'vector_searches_count': self.processing_metadata['vector_search_count'],
            'ffmpeg_operations_count': self.processing_metadata['ffmpeg_operations_count']
        }
