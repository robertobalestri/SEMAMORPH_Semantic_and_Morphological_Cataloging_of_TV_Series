"""
Processing pipeline service for SEMAMORPH.

This module contains the core processing logic migrated from main.py,
organized as a service with proper error handling and async support.
"""

import os
import json
import asyncio
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path

from backend.src.config import Config
from backend.src.utils.logger_utils import setup_logging
from backend.src.path_handler import PathHandler
from backend.src.ai_models.ai_models import get_llm, LLMType
from backend.src.plot_processing.plot_processing_models import EntityLink, EntityLinkEncoder
from backend.src.plot_processing.subtitle_processing import (
    generate_plot_from_subtitles, 
    map_scenes_to_timestamps,
    save_plot_files,
    save_scene_timestamps,
    load_previous_season_summary,
    PlotScene
)
from backend.src.utils.subtitle_utils import parse_srt_file
from backend.src.plot_processing.plot_summarizing import create_or_update_season_summary
from backend.src.utils.text_utils import load_text
from backend.src.plot_processing.plot_text_processing import replace_pronouns_with_names
from backend.src.plot_processing.plot_ner_entity_extraction import (
    extract_and_refine_entities_with_path_handler,
    substitute_appellations_with_names,
    normalize_entities_names_to_best_appellation
)
from backend.src.langgraph_narrative_arcs_extraction.narrative_arc_graph import extract_narrative_arcs
from backend.src.plot_processing.process_suggested_arcs import process_suggested_arcs

from .exceptions import (
    ProcessingError,
    SRTFileNotFoundError,
    EntityExtractionError,
    NarrativeExtractionError,
    SeasonSummaryError,
    SemanticSegmentationError,
    PronounReplacementError,
    PlotGenerationError
)


class ProcessingPipeline:
    """
    Core processing pipeline for episode processing.
    
    This service encapsulates all the processing steps from the original main.py,
    providing a structured, async-capable interface with proper error handling.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the processing pipeline.
        
        Args:
            config: Configuration instance (uses default if None)
        """
        self.config = config or Config()
        self.logger = setup_logging(self.__class__.__name__)
        
        # Initialize database manager
        try:
            from backend.src.narrative_storage_management.repositories import DatabaseSessionManager
            self.db_manager = DatabaseSessionManager()
            self.logger.info("✅ Database connection initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database connection: {e}")
            self.logger.error("This will cause database operations to fail. Please check your .env file and database configuration.")
            self.db_manager = None
        
    async def process_episode_complete(
        self,
        series: str,
        season: str, 
        episode: str,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a complete episode through the entire pipeline.
        
        Args:
            series: Series name (e.g., "GA")
            season: Season name (e.g., "S01") 
            episode: Episode name (e.g., "E01")
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results and statistics
            
        Raises:
            ProcessingError: If any step in the pipeline fails
        """
        try:
            path_handler = PathHandler(series, season, episode)
            self.logger.info(f"🔄 Starting complete processing for {series} {season} {episode}")
            
            # Report progress
            if progress_callback:
                progress_callback("STARTING", f"Initializing processing for {series} {season} {episode}")
            
            # Initialize LLM services
            if progress_callback:
                progress_callback("INIT_LLM", "Initializing language models")
            
            llm_intelligent = get_llm(LLMType.INTELLIGENT)
            llm_cheap = get_llm(LLMType.CHEAP)
            
            results = {
                "series": series,
                "season": season,
                "episode": episode,
                "steps_completed": [],
                "files_created": [],
                "entities_extracted": 0,
                "narrative_arcs_found": 0
            }
            
            # Step 0: Transcription and alignment (if SRT is provided)
            if progress_callback:
                progress_callback("TRANSCRIPTION", "Checking for transcription and alignment needs")
            
            transcription_result = await self._handle_transcription_and_alignment(path_handler, progress_callback)
            if transcription_result:
                results["files_created"].append(transcription_result.get("srt_path", ""))
                results["steps_completed"].append("TRANSCRIPTION_ALIGNMENT")
            
            # Step 1: Generate plot from SRT
            if progress_callback:
                progress_callback("PLOT_GENERATION", "Generating plot from SRT subtitles")
            
            plot_path = await self._generate_plot_from_srt(path_handler, llm_intelligent, progress_callback)
            if plot_path:
                results["files_created"].append(plot_path)
                results["steps_completed"].append("PLOT_GENERATION")
            
            # Step 2: Map scenes to timestamps
            if progress_callback:
                progress_callback("TIMESTAMP_MAPPING", "Mapping scenes to subtitle timestamps")
            
            timestamps_path = await self._map_scene_timestamps(path_handler, llm_cheap, progress_callback)
            if timestamps_path:
                results["files_created"].append(timestamps_path)
                results["steps_completed"].append("TIMESTAMP_MAPPING")
                
            # Step 3: Pronoun replacement and entity processing
            if progress_callback:
                progress_callback("PRONOUN_REPLACEMENT", "Replacing pronouns with character names")
                
            named_plot_path, entities = await self._process_pronouns_and_entities(
                path_handler, llm_intelligent, llm_cheap, progress_callback
            )
            if named_plot_path:
                results["files_created"].append(named_plot_path)
                results["entities_extracted"] = len(entities)
                results["steps_completed"].append("PRONOUN_REPLACEMENT")
                results["steps_completed"].append("ENTITY_EXTRACTION")
            
            # Step 4: Semantic segmentation
            if progress_callback:
                progress_callback("SEMANTIC_SEGMENTATION", "Performing semantic segmentation")
                
            segments_path = await self._perform_semantic_segmentation(
                path_handler, llm_intelligent, progress_callback
            )
            if segments_path:
                results["files_created"].append(segments_path)
                results["steps_completed"].append("SEMANTIC_SEGMENTATION")
            
            # Step 5: Season summary update (MOVED EARLIER)
            if progress_callback:
                progress_callback("SEASON_SUMMARY", "Updating season summary")
                
            summary_path = await self._update_season_summary(
                path_handler, llm_intelligent, progress_callback
            )
            if summary_path:
                results["files_created"].append(summary_path)
                results["steps_completed"].append("SEASON_SUMMARY")
            
            # Step 6: Narrative arc extraction (MOVED TO END WITH DELAY)
            if progress_callback:
                progress_callback("NARRATIVE_EXTRACTION", "Waiting before narrative arcs extraction...")
            
            import asyncio
            await asyncio.sleep(100000)  # Almost infinite delay for debugging
            
            if progress_callback:
                progress_callback("NARRATIVE_EXTRACTION", "Extracting narrative arcs")
                
            arcs_count = await self._extract_narrative_arcs(
                path_handler, series, season, episode, progress_callback
            )
            results["narrative_arcs_found"] = arcs_count
            results["steps_completed"].append("NARRATIVE_EXTRACTION")
            if summary_path:
                results["files_created"].append(summary_path)
                results["steps_completed"].append("SEASON_SUMMARY")
            
            if progress_callback:
                progress_callback("COMPLETED", f"✅ Processing completed successfully")
                
            self.logger.info(f"✅ Complete processing finished for {series} {season} {episode}")
            return results
            
        except Exception as e:
            error_msg = f"Complete processing failed for {series} {season} {episode}: {str(e)}"
            self.logger.error(error_msg)
            if progress_callback:
                progress_callback("ERROR", error_msg)
            raise ProcessingError(error_msg, step="COMPLETE_PROCESSING", cause=e)
    
    async def _handle_transcription_and_alignment(
        self,
        path_handler: PathHandler,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[Dict]:
        """
        Handle transcription and alignment if needed.
        
        This method checks if an SRT file exists and if transcription/alignment is needed.
        If an aligned SRT already exists, it uses that. Otherwise, it runs transcription.
        
        Args:
            path_handler: Path handler for the episode
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with transcription results or None if not needed
        """
        try:
            # Check if SRT already exists (now tautologically aligned since transcription and alignment are done during initial step)
            srt_path = path_handler.get_srt_file_path()
            if os.path.exists(srt_path):
                self.logger.info(f"✅ SRT already exists: {srt_path}")
                if progress_callback:
                    progress_callback("TRANSCRIPTION", "Using existing SRT")
                return {
                    "status": "success",
                    "message": "Using existing SRT",
                    "srt_path": srt_path
                }
            
            # Check if SRT exists
            if not os.path.exists(srt_path):
                self.logger.info("📝 No SRT file found - skipping transcription")
                if progress_callback:
                    progress_callback("TRANSCRIPTION", "No SRT file found - skipping transcription")
                return None
            
            # Check if audio file exists for transcription
            audio_path = path_handler.get_audio_file_path()
            video_path = path_handler.get_video_file_path()
            
            # If neither audio nor video exists, we can't do transcription
            if not os.path.exists(audio_path) and not os.path.exists(video_path):
                self.logger.info("🎵 No audio or video file found - using existing SRT")
                if progress_callback:
                    progress_callback("TRANSCRIPTION", "No audio or video file found - using existing SRT")
                return {
                    "status": "success",
                    "message": "Using existing SRT (no audio or video for transcription)",
                    "srt_path": srt_path
                }
            
            # Run transcription and alignment
            if progress_callback:
                progress_callback("TRANSCRIPTION", "Running WhisperX transcription and alignment")
            
            # Import transcription workflow
            from backend.src.subtitle_speaker_identification.transcription_workflow import TranscriptionWorkflow
            
            transcription_workflow = TranscriptionWorkflow(
                path_handler.get_series(),
                path_handler.get_season(),
                path_handler.get_episode(),
                path_handler.base_dir,
                path_handler=path_handler
            )
            
            # Check prerequisites
            prerequisites = transcription_workflow.check_prerequisites()
            if not prerequisites["all_passed"]:
                self.logger.warning("⚠️ Prerequisites not met for transcription - using existing SRT")
                if progress_callback:
                    progress_callback("TRANSCRIPTION", "Prerequisites not met - using existing SRT")
                return {
                    "status": "warning",
                    "message": "Prerequisites not met - using existing SRT",
                    "srt_path": original_srt_path,
                    "prerequisites": prerequisites
                }
            
            # Run transcription workflow
            result = transcription_workflow.run_transcription_and_alignment(
                language="en",
                model_size="base",
                compute_type="float16",
                force_regenerate=False
            )
            
            if result["status"] == "success":
                self.logger.info("✅ Transcription and alignment completed successfully")
                if progress_callback:
                    progress_callback("TRANSCRIPTION", "Transcription and alignment completed")
                return result
            else:
                self.logger.warning(f"⚠️ Transcription failed: {result['message']} - using existing SRT")
                if progress_callback:
                    progress_callback("TRANSCRIPTION", "Transcription failed - using existing SRT")
                return {
                    "status": "warning",
                    "message": f"Transcription failed: {result['message']} - using existing SRT",
                    "srt_path": srt_path,
                    "transcription_error": result
                }
                
        except Exception as e:
            self.logger.error(f"❌ Transcription handling failed: {e}")
            if progress_callback:
                progress_callback("TRANSCRIPTION", f"Transcription failed: {str(e)}")
            return None

    async def _generate_plot_from_srt(
        self,
        path_handler: PathHandler, 
        llm_intelligent,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[str]:
        """Generate plot from SRT file if it doesn't exist (preferring aligned SRT)."""
        try:
            raw_plot_path = path_handler.get_raw_plot_file_path()
            
            if os.path.exists(raw_plot_path):
                self.logger.info(f"Plot file already exists: {raw_plot_path}")
                return raw_plot_path
                
            # Use SRT file (now tautologically aligned since transcription and alignment are done during initial step)
            srt_path = path_handler.get_srt_file_path()
            
            if os.path.exists(srt_path):
                self.logger.info(f"Using SRT: {srt_path}")
            else:
                raise SRTFileNotFoundError(
                    srt_path, 
                    path_handler.get_series(), 
                    path_handler.get_season(), 
                    path_handler.get_episode()
                )
            
            self.logger.info("Generating plot from SRT subtitles")
            
            # Parse SRT file
            subtitles = parse_srt_file(srt_path)
            
            # Load previous season summary for context
            season_summary_path = path_handler.get_season_summary_path()
            previous_season_summary = load_previous_season_summary(season_summary_path)
            
            # Generate plot from subtitles
            plot_data = generate_plot_from_subtitles(subtitles, llm_intelligent, previous_season_summary)
            
            # Save plot files
            txt_path, scenes_json_path = save_plot_files(plot_data, path_handler)
            
            self.logger.info(f"Generated plot saved to: {txt_path}")
            return txt_path
            
        except Exception as e:
            if isinstance(e, (SRTFileNotFoundError, PlotGenerationError)):
                raise
            raise PlotGenerationError(str(e), srt_path, cause=e)
    
    async def _map_scene_timestamps(
        self,
        path_handler: PathHandler,
        llm_cheap,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[str]:
        """Map scenes to timestamps if file doesn't exist."""
        try:
            timestamps_path = path_handler.get_scene_timestamps_path()
            
            if os.path.exists(timestamps_path):
                self.logger.info(f"Scene timestamps already exist: {timestamps_path}")
                return timestamps_path
            
            self.logger.info("Mapping scenes to subtitle timestamps")
            
            # Parse SRT file for timestamp mapping
            srt_path = path_handler.get_srt_file_path()
            if not os.path.exists(srt_path):
                self.logger.error(f"SRT file not found for timestamp mapping: {srt_path}")
                return None
            
            subtitles = parse_srt_file(srt_path)
            
            # Load plot scenes data
            scenes_json_path = path_handler.get_plot_scenes_json_path()
            if not os.path.exists(scenes_json_path):
                self.logger.error(f"Plot scenes JSON not found: {scenes_json_path}")
                return None
                
            with open(scenes_json_path, 'r') as f:
                plot_data = json.load(f)
            
            scenes = []
            for scene_data in plot_data.get("scenes", []):
                scene = PlotScene(
                    scene_number=scene_data.get("scene_number", len(scenes) + 1),
                    plot_segment=scene_data.get("plot_segment", "")
                )
                scenes.append(scene)
            
            # Map scenes to timestamps using new simplified boundary detection
            mapped_scenes = map_scenes_to_timestamps(scenes, subtitles, llm_cheap)
            
            # Save scene timestamps
            saved_path = save_scene_timestamps(mapped_scenes, path_handler)
            
            self.logger.info(f"Scene timestamps saved to: {saved_path}")
            return saved_path
            
        except Exception as e:
            self.logger.error(f"Error mapping scene timestamps: {e}")
            # Non-critical error, don't fail the entire process
            return None
    
    async def _process_pronouns_and_entities(
        self,
        path_handler: PathHandler,
        llm_intelligent,
        llm_cheap,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> tuple[Optional[str], List[EntityLink]]:
        """Process pronouns and extract entities."""
        try:
            # Load the raw plot
            raw_plot_path = path_handler.get_raw_plot_file_path()
            raw_plot = load_text(raw_plot_path)
            
            # Process pronouns
            named_file_path = path_handler.get_named_plot_file_path()
            
            if not os.path.exists(named_file_path):
                self.logger.info("Replacing pronouns with names")
                named_plot = replace_pronouns_with_names(
                    text=raw_plot, 
                    intelligent_llm=llm_intelligent, 
                    cheap_llm=llm_cheap
                )
                
                with open(named_file_path, "w") as named_file:
                    named_file.write(named_plot)
            else:
                self.logger.info(f"Loading named plot from: {named_file_path}")
                with open(named_file_path, "r") as named_file:
                    named_plot = named_file.read()
            
            # Extract entities
            entities = await self._extract_entities(path_handler, named_plot, llm_intelligent)
            
            # Entity substitution and normalization
            await self._process_entity_substitution(path_handler, named_plot, entities, llm_intelligent)
            
            return named_file_path, entities
            
        except Exception as e:
            raise PronounReplacementError(str(e), cause=e)
    
    async def _extract_entities(
        self, 
        path_handler: PathHandler, 
        named_plot: str, 
        llm_intelligent
    ) -> List[EntityLink]:
        """Extract and refine entities from the named plot."""
        try:
            episode_entities_path = path_handler.get_episode_refined_entities_path()
            
            if not os.path.exists(episode_entities_path):
                self.logger.info("Extracting and refining entities from named plot")
                entities = extract_and_refine_entities_with_path_handler(
                    path_handler, 
                    path_handler.get_series(),
                    self.db_manager  # Add db_manager parameter
                )
            else:
                self.logger.info(f"Loading existing entities from: {episode_entities_path}")
                with open(episode_entities_path, "r") as entities_file:
                    entities_data = json.load(entities_file)
                    entities = [EntityLink(**entity) for entity in entities_data]
            
            return entities
            
        except Exception as e:
            raise EntityExtractionError(str(e), episode_entities_path, cause=e)
    
    async def _process_entity_substitution(
        self,
        path_handler: PathHandler,
        named_plot: str,
        entities: List[EntityLink],
        llm_intelligent
    ) -> None:
        """Process entity substitution and normalization."""
        try:
            entity_substituted_plot_path = path_handler.get_entity_substituted_plot_file_path()
            entity_normalized_plot_path = path_handler.get_entity_normalized_plot_file_path()
            
            # Entity substitution
            if not os.path.exists(entity_substituted_plot_path):
                entity_substituted_plot = substitute_appellations_with_names(
                    named_plot, entities, llm_intelligent
                )
                with open(entity_substituted_plot_path, "w") as file:
                    file.write(entity_substituted_plot)
            else:
                with open(entity_substituted_plot_path, "r") as file:
                    entity_substituted_plot = file.read()
            
            # Entity normalization
            if not os.path.exists(entity_normalized_plot_path):
                entity_normalized_plot = normalize_entities_names_to_best_appellation(
                    entity_substituted_plot, entities
                )
                with open(entity_normalized_plot_path, "w") as file:
                    file.write(entity_normalized_plot)
            
        except Exception as e:
            raise EntityExtractionError(f"Entity substitution failed: {str(e)}", cause=e)
    
    async def _extract_narrative_arcs(
        self,
        path_handler: PathHandler,
        series: str,
        season: str,
        episode: str,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> int:
        """Extract narrative arcs and update database."""
        try:
            suggested_episode_arc_path = path_handler.get_suggested_episode_arc_path()
            
            if not os.path.exists(suggested_episode_arc_path):
                # Prepare file paths for narrative arc extraction
                file_paths_for_graph = {
                    "episode_plot_path": path_handler.get_raw_plot_file_path(),
                    "seasonal_narrative_analysis_output_path": path_handler.get_season_narrative_analysis_path(),
                    "episode_narrative_analysis_output_path": path_handler.get_episode_narrative_analysis_path(),
                    "episode_narrative_arcs_output_path": path_handler.get_episode_narrative_arcs_path(),
                    "season_narrative_arcs_output_path": path_handler.get_season_narrative_arcs_path(),
                    "episode_entities_path": path_handler.get_episode_refined_entities_path(),
                    "season_entities_path": path_handler.get_season_extracted_refined_entities_path(),
                    "suggested_episode_arc_path": suggested_episode_arc_path
                }
                
                # Extract narrative arcs using LangGraph
                self.logger.info("Extracting narrative arcs")
                extract_narrative_arcs(file_paths_for_graph, series, season, episode)
            
            # Process suggested arcs and update database
            self.logger.info("Processing suggested arcs and updating database")
            updated_arcs = process_suggested_arcs(
                suggested_episode_arc_path,
                series,
                season,
                episode
            )
            
            arc_count = len(updated_arcs)
            self.logger.info(f"Updated {arc_count} arcs in the database")
            return arc_count
            
        except Exception as e:
            raise NarrativeExtractionError(str(e), suggested_episode_arc_path, cause=e)
    
    async def _update_season_summary(
        self,
        path_handler: PathHandler,
        llm_intelligent,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[str]:
        """Create or update season summary using speaker-identified content when available."""
        try:
            self.logger.info("Creating/updating season summary")
            
            # Prefer speaker-identified plot, fallback to raw plot
            speaker_plot_path = path_handler.get_plot_possible_speakers_path()
            raw_plot_path = path_handler.get_raw_plot_file_path()
            
            if os.path.exists(speaker_plot_path):
                episode_plot_path = speaker_plot_path
                self.logger.info("✅ Using speaker-identified plot for season summary")
            else:
                episode_plot_path = raw_plot_path
                self.logger.warning("⚠️ Speaker-identified plot not available, using raw plot for season summary")
            
            season_summary_path = path_handler.get_season_summary_path()
            episode_summary_path = path_handler.get_episode_summary_path()
            
            # Create/update season summary
            season_summary = create_or_update_season_summary(
                episode_plot_path,
                season_summary_path,
                episode_summary_path,
                llm_intelligent
            )
            
            if season_summary:
                self.logger.info("✅ Season summary updated successfully")
                return season_summary_path
            else:
                self.logger.warning("⚠️ Season summary creation failed")
                return None
                
        except Exception as e:
            # Don't fail the entire process if summary creation fails
            self.logger.error(f"❌ Error creating season summary: {e}")
            return None
