"""
Main recap generator class - simplified and streamlined.

This module provides the core RecapGenerator class that orchestrates
the entire recap generation process following the specified LLM workflow.
"""

import logging
import os
import sys
import json
from typing import Optional

# Add the src directory to Python path to enable proper imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from path_handler import PathHandler
from .models import RecapResult
from .utils import load_episode_inputs, select_events_round_robin, search_vector_database
from .llm_services import generate_arc_queries, rank_events_per_arc, extract_key_dialogue
from .video_processor import extract_video_clips, assemble_final_recap

logger = logging.getLogger(__name__)


class RecapGenerator:
    """
    Simplified recap generator that follows the core LLM workflow:
    
    1. LLM #1: Query generation per narrative arc
    2. Vector DB search for historical events
    3. LLM #2: Event ranking per arc (loop)
    4. Round-robin final event selection
    5. LLM #3: Subtitle pruning (round robin)
    6. FFmpeg video clip extraction
    7. Final recap assembly
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        
    def generate_recap(self, series: str, season: str, episode: str) -> RecapResult:
        """
        Generate a "Previously On" recap for the specified episode.
        
        Args:
            series: Series identifier (e.g., "GreysAnatomy")
            season: Season identifier (e.g., "S01")
            episode: Episode identifier (e.g., "E09")
            
        Returns:
            RecapResult with video path and metadata
        """
        try:
            logger.info(f"🎬 Starting recap generation for {series} {season} {episode}")
            
            # Step 1: Load episode inputs
            logger.info("📖 Loading episode inputs...")
            inputs = load_episode_inputs(series, season, episode, self.base_dir)
            
            if not inputs['narrative_arcs']:
                return RecapResult(
                    video_path="", events=[], clips=[], 
                    total_duration=0.0, success=False,
                    error_message="No narrative arcs found"
                )
            
            # Step 2: LLM #1 - Generate queries per narrative arc
            logger.info("🎯 Generating vector database queries...")
            arc_queries = generate_arc_queries(
                inputs['season_summary'],
                inputs['episode_plot'], 
                inputs['narrative_arcs']
            )
            
            if not arc_queries:
                return RecapResult(
                    video_path="", events=[], clips=[],
                    total_duration=0.0, success=False,
                    error_message="No queries generated"
                )
            
            # Step 3: Search vector database
            logger.info("🔍 Searching vector database...")
            events_by_arc = search_vector_database(arc_queries, series, season, episode)
            
            if not events_by_arc:
                return RecapResult(
                    video_path="", events=[], clips=[],
                    total_duration=0.0, success=False,
                    error_message="No events found in vector database"
                )
            
            # Step 4: LLM #2 - Rank events per arc (loop)
            logger.info("📊 Ranking events per narrative arc...")
            ranked_events = rank_events_per_arc(events_by_arc, inputs['episode_plot'])
            logger.info("📊 Event ranking complete.")
            
            # Step 5: Round-robin final event selection
            logger.info("🎯 Selecting final events (round-robin)...")
            selected_events = select_events_round_robin(ranked_events, max_events=8)
            
            if not selected_events:
                return RecapResult(
                    video_path="", events=[], clips=[],
                    total_duration=0.0, success=False,
                    error_message="No events selected"
                )
            
            # Step 6: LLM #3 - Extract key dialogue with consecutive subtitles and fallback
            logger.info("📝 Extracting key dialogue...")
            key_dialogue = extract_key_dialogue(selected_events, inputs['subtitle_data'], ranked_events, inputs['subtitle_data'])

            # Save a JSON spec of selected events and pruned subtitles for transparency
            try:
                path_handler = PathHandler(series, season, episode, self.base_dir)
                recap_dir = path_handler.get_recap_files_dir()
                os.makedirs(recap_dir, exist_ok=True)
                recap_spec_path = path_handler.get_recap_clips_json_path()

                # Build export structure grouped by arc
                export = {
                    "series": series,
                    "season": season,
                    "episode": episode,
                    "ranked_events_by_arc": {arc_id: [ev.to_dict() for ev in ev_list] for arc_id, ev_list in ranked_events.items()},
                    "selected_events": []
                }
                for ev in selected_events:
                    export["selected_events"].append({
                        "event_id": ev.id,
                        "arc_title": ev.arc_title,
                        "narrative_arc_id": getattr(ev, 'narrative_arc_id', ''),
                        "source_episode": f"{ev.series}{ev.season}{ev.episode}",
                        "start_time": ev.start_time,
                        "end_time": ev.end_time,
                        "content": ev.content,
                        "selected_subtitles": key_dialogue.get(ev.id, {}).get('lines', []),
                        "debug_info": key_dialogue.get(ev.id, {}).get('debug', {})
                    })

                with open(recap_spec_path, 'w', encoding='utf-8') as f:
                    json.dump(export, f, ensure_ascii=False, indent=2)
                logger.info(f"📝 Saved recap clips JSON: {recap_spec_path}")
            except Exception as e:
                logger.warning(f"Couldn't write recap clips JSON: {e}")
            
            # Step 7: Extract video clips
            logger.info("🎬 Extracting video clips...")
            video_clips = extract_video_clips(selected_events, key_dialogue, self.base_dir)
            
            if not video_clips:
                return RecapResult(
                    video_path="", events=selected_events, clips=[],
                    total_duration=0.0, success=False,
                    error_message="No video clips extracted"
                )
            
            # Step 8: Assemble final recap
            logger.info("🎞️ Assembling final recap...")
            path_handler = PathHandler(series, season, episode, self.base_dir)
            output_dir = path_handler.get_recap_files_dir()  # This will be used by assemble_final_recap
            final_video_path = assemble_final_recap(video_clips, output_dir, series, season, episode)
            
            # Calculate total duration
            total_duration = sum(clip.duration for clip in video_clips)
            
            logger.info(f"✅ Recap generation completed successfully!")
            logger.info(f"📁 Final video: {final_video_path}")
            logger.info(f"⏱️ Total duration: {total_duration:.1f}s")
            logger.info(f"🎬 Clips: {len(video_clips)} from {len(selected_events)} events")
            
            return RecapResult(
                video_path=final_video_path,
                events=selected_events,
                clips=video_clips,
                total_duration=total_duration,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Recap generation failed: {e}")
            return RecapResult(
                video_path="", events=[], clips=[],
                total_duration=0.0, success=False,
                error_message=str(e)
            )
    
    def validate_prerequisites(self, series: str, season: str, episode: str) -> dict:
        """
        Validate that all required files exist for recap generation.
        
        Returns:
            Dictionary with validation results
        """
        path_handler = PathHandler(series, season, episode, self.base_dir)
        
        results = {
            'ready': True,
            'missing_files': [],
            'warnings': []
        }
        
        # Required files using PathHandler
        required_files = [
            ("plot_possible_speakers", path_handler.get_plot_possible_speakers_path()),
            ("present_running_plotlines", path_handler.get_present_running_plotlines_path()),
            ("video_file", path_handler.get_video_file_path())
        ]
        
        for file_type, file_path in required_files:
            if not os.path.exists(file_path):
                results['ready'] = False
                results['missing_files'].append(f"{file_type}: {file_path}")
        
        # Optional but recommended files
        season_summary = path_handler.get_season_summary_path()
        if not os.path.exists(season_summary):
            results['warnings'].append(f"Season summary not found: {season_summary}")
        
        return results
