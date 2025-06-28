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
    parse_srt_file, 
    generate_plot_from_subtitles, 
    map_scene_to_timestamps, 
    save_plot_files,
    save_scene_timestamps,
    load_previous_season_summary,
    PlotScene
)
from backend.src.plot_processing.plot_summarizing import create_or_update_season_summary
from backend.src.utils.text_utils import load_text
from backend.src.plot_processing.plot_text_processing import replace_pronouns_with_names
from backend.src.plot_processing.plot_semantic_processing import semantic_split
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
            
            # Step 5: Narrative arc extraction
            if progress_callback:
                progress_callback("NARRATIVE_EXTRACTION", "Extracting narrative arcs")
                
            arcs_count = await self._extract_narrative_arcs(
                path_handler, series, season, episode, progress_callback
            )
            results["narrative_arcs_found"] = arcs_count
            results["steps_completed"].append("NARRATIVE_EXTRACTION")
            
            # Step 6: Season summary update
            if progress_callback:
                progress_callback("SEASON_SUMMARY", "Updating season summary")
                
            summary_path = await self._update_season_summary(
                path_handler, llm_intelligent, progress_callback
            )
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
    
    async def _generate_plot_from_srt(
        self,
        path_handler: PathHandler, 
        llm_intelligent,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[str]:
        """Generate plot from SRT file if it doesn't exist."""
        try:
            raw_plot_path = path_handler.get_raw_plot_file_path()
            
            if os.path.exists(raw_plot_path):
                self.logger.info(f"Plot file already exists: {raw_plot_path}")
                return raw_plot_path
                
            # Look for SRT file
            srt_path = path_handler.get_srt_file_path()
            if not os.path.exists(srt_path):
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
            episode_prefix = f"{path_handler.get_series()}{path_handler.get_season()}{path_handler.get_episode()}"
            episode_dir = os.path.dirname(raw_plot_path)
            txt_path, scenes_json_path = save_plot_files(plot_data, episode_dir, episode_prefix)
            
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
            
            # Map each scene to timestamps
            mapped_scenes = []
            for scene in scenes:
                mapped_scene = map_scene_to_timestamps(scene, subtitles, llm_cheap)
                mapped_scenes.append(mapped_scene)
            
            # Save scene timestamps
            episode_prefix = f"{path_handler.get_series()}{path_handler.get_season()}{path_handler.get_episode()}"
            episode_dir = os.path.dirname(timestamps_path)
            saved_path = save_scene_timestamps(mapped_scenes, episode_dir, episode_prefix)
            
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
                    path_handler.get_series()
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
    
    async def _perform_semantic_segmentation(
        self,
        path_handler: PathHandler,
        llm_intelligent,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[str]:
        """Perform semantic segmentation of the plot."""
        try:
            semantic_segments_path = path_handler.get_semantic_segments_path()
            
            if os.path.exists(semantic_segments_path):
                self.logger.info(f"Loading semantic segments from: {semantic_segments_path}")
                return semantic_segments_path
            
            # Load entity normalized plot
            entity_normalized_plot_path = path_handler.get_entity_normalized_plot_file_path()
            with open(entity_normalized_plot_path, "r") as file:
                entity_normalized_plot = file.read()
            
            self.logger.info("Performing semantic splitting")
            semantic_segments = semantic_split(text=entity_normalized_plot, llm=llm_intelligent)
            
            with open(semantic_segments_path, "w", encoding='utf-8') as file:
                json.dump(semantic_segments, file, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Semantic splitting complete. Results saved to {semantic_segments_path}")
            return semantic_segments_path
            
        except Exception as e:
            raise SemanticSegmentationError(str(e), cause=e)
    
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
        """Create or update season summary."""
        try:
            self.logger.info("Creating/updating season summary")
            
            episode_plot_path = path_handler.get_raw_plot_file_path()
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
