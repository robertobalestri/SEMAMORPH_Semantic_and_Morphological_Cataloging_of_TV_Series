"""
Main pipeline for subtitle-based speaker identification and face clustering.
Orchestrates the complete workflow from SRT parsing to speaker resolution.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

from ..path_handler import PathHandler
from ..utils.subtitle_utils import SubtitleEntry
from ..plot_processing.subtitle_processing import load_previous_season_summary, map_scenes_to_timestamps, PlotScene
from ..ai_models.ai_models import get_llm, LLMType
from ..narrative_storage_management.repositories import DatabaseSessionManager, CharacterRepository
from ..narrative_storage_management.character_service import CharacterService
from ..utils.logger_utils import setup_logging
from ..config import config
from ..config_validator import validate_config

from .srt_parser import SRTParser
from .speaker_identifier import SpeakerIdentifier  
from .subtitle_face_extractor import SubtitleFaceExtractor
from .subtitle_face_embedder import SubtitleFaceEmbedder
from .face_cluster_visualizer import FaceClusterVisualizer
from .face_embedding_vector_store import FaceEmbeddingVectorStore
from .face_clustering_system import FaceClusteringSystem

logger = setup_logging(__name__)

class SpeakerIdentificationPipeline:
    """Main pipeline for speaker identification and face clustering."""
    
    def __init__(self, series: str, season: str, episode: str, base_dir: str = "data"):
        self.path_handler = PathHandler(series, season, episode, base_dir)
        self.series = series
        self.season = season
        self.episode = episode
        self.llm = get_llm(LLMType.INTELLIGENT)
        self.config = config  # Add config reference
        self.face_vector_store = FaceEmbeddingVectorStore(self.path_handler)
        
        # Initialize components
        self.srt_parser = SRTParser()
        self.speaker_identifier = SpeakerIdentifier(self.llm, series)  # Pass series to speaker identifier
        self.face_extractor = SubtitleFaceExtractor(self.path_handler)
        self.face_embedder = SubtitleFaceEmbedder(self.path_handler)
        self.face_clustering_system = FaceClusteringSystem(self.path_handler, self.face_vector_store)
        
        # Initialize debug tracker
        self.debug_tracker = SpeakerDebugTracker(self.path_handler)
    
    def run_complete_pipeline(
        self,
        force_regenerate: bool = False,
        face_similarity_threshold: float = 0.8,
        embedding_model: str = "Facenet512",
        face_detector: str = "retinaface"
    ) -> Dict:
        """
        Run the complete speaker identification pipeline.
        
        Args:
            force_regenerate: If True, regenerate all intermediate files
            face_similarity_threshold: Threshold for face-based speaker resolution
            embedding_model: DeepFace embedding model to use
            face_detector: DeepFace detector backend
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        # Boolean confidence approach - no threshold needed
        
        # Validate configuration before starting pipeline
        logger.info("🔍 Validating configuration...")
        validation_report = validate_config()
        if validation_report['status'] == 'FAIL':
            error_msg = f"Configuration validation failed with {validation_report['error_count']} errors. Please fix configuration issues."
            logger.error(f"❌ {error_msg}")
            for error in validation_report['errors']:
                logger.error(f"   • {error['field']}: {error['message']}")
            raise ValueError(error_msg)
        elif validation_report['warnings']:
            logger.warning(f"⚠️ Configuration has {validation_report['warning_count']} warnings (pipeline will continue):")
            for warning in validation_report['warnings']:
                logger.warning(f"   • {warning['field']}: {warning['message']}")
        else:
            logger.info("✅ Configuration validation passed")
        
        logger.info(f"🚀 Starting Speaker Identification Pipeline for {self.path_handler.get_episode_code()}")
        
        results = {
            'episode_code': self.path_handler.get_episode_code(),
            'series': self.path_handler.get_series(),
            'season': self.path_handler.get_season(),
            'episode': self.path_handler.get_episode(),
            'pipeline_steps': {}
        }
        
        try:
            # Step 1: Parse SRT and generate dialogue JSON
            logger.info("📄 Step 1: Parsing SRT file")
            dialogue_lines = self._parse_subtitles(force_regenerate)
            
            # Check if we have any dialogue lines
            if not dialogue_lines:
                logger.warning("⚠️ No dialogue lines found in SRT file")
                results['pipeline_steps']['srt_parsing'] = {
                    'status': 'warning',
                    'dialogue_count': 0,
                    'message': 'No dialogue lines found in SRT file'
                }
                # Create empty enhanced SRT file
                enhanced_srt_path = self.path_handler.get_enhanced_srt_path()
                with open(enhanced_srt_path, 'w', encoding='utf-8') as f:
                    f.write("# No dialogue lines found in SRT file\n")
                    f.write("# Enhanced SRT generation skipped\n")
                
                results['pipeline_steps']['enhanced_srt'] = {
                    'status': 'success',
                    'enhanced_srt_path': enhanced_srt_path,
                    'message': 'Empty enhanced SRT file created'
                }
                results['overall_stats'] = {
                    'total_dialogue_lines': 0,
                    'faces_extracted': 0,
                    'speakers_identified': 0,
                    'confidence_rate': 0,
                    'processing_time': 0
                }
                logger.info("✅ Pipeline completed with empty subtitle file")
                return results
            
            results['pipeline_steps']['srt_parsing'] = {
                'status': 'success',
                'dialogue_count': len(dialogue_lines)
            }
            
            # Step 2: Load plot scenes for context
            logger.info("📖 Step 2: Loading plot scenes")
            plot_scenes = self._load_plot_scenes()
            results['pipeline_steps']['plot_loading'] = {
                'status': 'success',
                'scene_count': len(plot_scenes)
            }
            
            # Step 2.5: Generate scene timestamps if missing
            logger.info("⏰ Step 2.5: Generating scene timestamps")
            plot_scenes_with_timestamps = self._ensure_scene_timestamps(
                plot_scenes, 
                dialogue_lines, 
                force_regenerate
            )
            results['pipeline_steps']['timestamp_mapping'] = {
                'status': 'success',
                'scenes_with_timestamps': len([s for s in plot_scenes_with_timestamps if 'start_seconds' in s])
            }
            
            # Step 3: Identify speakers using LLM (or load from checkpoint)
            logger.info("🎭 Step 3: Identifying speakers with LLM")
            dialogue_with_speakers = self._identify_speakers_or_load_checkpoint(
                dialogue_lines, 
                plot_scenes_with_timestamps, 
                force_regenerate
            )
            
            # Track LLM assignments for debug
            self.debug_tracker.track_llm_assignments(dialogue_with_speakers)
            
            confident_count = sum(
                1 for line in dialogue_with_speakers 
                if line.is_llm_confident
            )
            
            results['pipeline_steps']['speaker_identification'] = {
                'status': 'success',
                'total_dialogue': len(dialogue_with_speakers),
                'confident_speakers': confident_count,
                'confidence_rate': (confident_count / len(dialogue_with_speakers)) * 100 if dialogue_with_speakers else 0
            }
            
            # Step 3.5: Save LLM assignments checkpoint (before expensive face processing)
            logger.info("💾 Step 3.5: Saving LLM speaker assignments checkpoint")
            self._save_llm_checkpoint(dialogue_with_speakers, results['pipeline_steps']['speaker_identification'])
            
            # Step 4: Extract faces from video
            logger.info("👤 Step 4: Extracting faces from video")
            df_faces = self._extract_faces(dialogue_with_speakers, face_detector, force_regenerate)
            results['pipeline_steps']['face_extraction'] = {
                'status': 'success',
                'faces_extracted': len(df_faces)
            }
            
            # Step 5: Generate face embeddings
            logger.info("🧠 Step 5: Generating face embeddings")
            df_faces_with_embeddings = self._generate_embeddings(
                df_faces, 
                embedding_model, 
                force_regenerate
            )
            results['pipeline_steps']['embedding_generation'] = {
                'status': 'success',
                'embeddings_generated': len(df_faces_with_embeddings)
            }
            
            # Step 6: Save embeddings to vector store
            logger.info("💾 Step 6: Saving embeddings to vector store")
            self._save_to_vector_store(df_faces_with_embeddings)
            results['pipeline_steps']['vector_store_save'] = {
                'status': 'success'
            }
            
            # Step 7-8: Run face clustering system for speaker assignment
            logger.info("🎭 Step 7-8: Running face clustering system for speaker assignment")
            final_dialogue, cluster_info, character_clusters, df_faces_clustered = self.face_clustering_system.run_face_clustering_pipeline(
                df_faces_with_embeddings,
                dialogue_with_speakers,
                expected_embedding_dim=config.face_embedding_dimension
            )
            
            # Track face clustering results for debug
            self.debug_tracker.track_face_clusters(cluster_info, df_faces_clustered)
            
            # Calculate final statistics
            final_confident_count = sum(
                1 for line in final_dialogue 
                if line.is_llm_confident
            )
            
            resolved_count = final_confident_count - confident_count
            
            results['pipeline_steps']['face_clustering'] = {
                'status': 'success',
                'total_clusters': len(cluster_info),
                'character_clusters': len(character_clusters),
                'speakers_resolved': resolved_count,
                'final_confidence_rate': (final_confident_count / len(final_dialogue)) * 100 if final_dialogue else 0,
                'cluster_info': self._strip_embeddings_from_clusters(cluster_info),
                'character_clusters': character_clusters
            }
            
            # Step 9: Create cluster visualizations
            logger.info("📊 Step 9: Creating cluster visualizations")
            visualization_paths = self._create_visualizations(df_faces_clustered)
            results['pipeline_steps']['visualization'] = {
                'status': 'success',
                'plots_created': len(visualization_paths),
                'visualization_paths': visualization_paths
            }
            
            # Step 10: Attach face data to dialogue lines
            logger.info("🖼️ Step 10: Attaching face data to dialogue lines")
            final_dialogue_with_faces = self._attach_face_data_to_dialogue(final_dialogue, df_faces_with_embeddings)
            results['pipeline_steps']['attach_face_data'] = {
                'status': 'success'
            }
            
            # Step 11: Save final results
            logger.info("💾 Step 11: Saving final results")
            self._save_final_results(final_dialogue_with_faces, character_clusters, results)
            results['pipeline_steps']['save_results'] = {
                'status': 'success'
            }
            
            # Step 12: Generate enhanced SRT with speaker identification
            logger.info("📝 Step 12: Generating enhanced SRT with speaker identification")
            logger.info(f"🔍 [DEBUG] Calling _generate_enhanced_srt from run_complete_pipeline")
            
            # Debug: Check dialogue objects before SRT generation
            character_median_count = sum(1 for d in final_dialogue_with_faces if getattr(d, 'resolution_method', None) == "character_median_direct")
            face_recognition_count = sum(1 for d in final_dialogue_with_faces if getattr(d, 'resolution_method', None) in ["face_clustering_single", "face_clustering_multi"])
            logger.info(f"🔍 [DEBUG] Before SRT generation: {character_median_count} character_median_direct, {face_recognition_count} face recognition dialogues")
            
            # Debug: Check a few sample dialogues
            for i, dialogue in enumerate(final_dialogue_with_faces[:5]):
                resolution_method = getattr(dialogue, 'resolution_method', None)
                speaker = getattr(dialogue, 'speaker', None)
                is_confident = getattr(dialogue, 'is_llm_confident', None)
                logger.info(f"🔍 [DEBUG] Dialogue {dialogue.index}: speaker='{speaker}', resolution_method='{resolution_method}', is_confident={is_confident}")
            
            # Debug: Count resolution methods
            resolution_methods = {}
            for dialogue in final_dialogue_with_faces:
                method = getattr(dialogue, 'resolution_method', None)
                if method:
                    resolution_methods[method] = resolution_methods.get(method, 0) + 1
            logger.info(f"🔍 [DEBUG] Resolution method distribution: {resolution_methods}")
            
            enhanced_srt_path = self._generate_enhanced_srt(final_dialogue_with_faces)
            results['pipeline_steps']['enhanced_srt'] = {
                'status': 'success',
                'enhanced_srt_path': enhanced_srt_path
            }
            
            # Track final resolution methods and confidence stats for debug
            self.debug_tracker.track_resolution_methods(final_dialogue_with_faces)
            self.debug_tracker.track_confidence_stats(final_dialogue_with_faces)
            
            # Save debug file
            debug_file_path = self.debug_tracker.save_debug_file()
            logger.info(f"📄 Debug file saved to: {debug_file_path}")
            
            # Overall pipeline statistics
            results['overall_stats'] = {
                'total_dialogue_lines': len(final_dialogue_with_faces),
                'faces_extracted': len(df_faces_with_embeddings),
                'speakers_identified': len(character_clusters),
                'confident_dialogue': final_confident_count,
                'final_confidence_rate': results['pipeline_steps']['face_clustering']['final_confidence_rate']
            }
            
            # Add debug file path to results
            results['debug_file_path'] = debug_file_path
            
            logger.info(f"✅ Pipeline completed successfully!")
            logger.info(f"📊 Final stats: {final_confident_count}/{len(final_dialogue_with_faces)} dialogue lines with confident speakers ({results['overall_stats']['final_confidence_rate']:.1f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            results['pipeline_steps']['error'] = {
                'status': 'failed',
                'error': str(e)
            }
            return results
    
    def _parse_subtitles(self, force_regenerate: bool) -> List:
        """Parse SRT file and return dialogue lines."""
        srt_path = self.path_handler.get_srt_file_path()
        dialogue_json_path = self.path_handler.get_dialogue_json_path()
        
        if not force_regenerate and os.path.exists(dialogue_json_path):
            logger.info(f"📂 Loading existing dialogue from: {dialogue_json_path}")
            return self.srt_parser.load_dialogue_json(dialogue_json_path)
        
        if not os.path.exists(srt_path):
            logger.error(f"❌ SRT file not found: {srt_path}")
            # Create empty enhanced SRT with error message
            enhanced_srt_path = self.path_handler.get_enhanced_srt_path()
            with open(enhanced_srt_path, 'w', encoding='utf-8') as f:
                f.write(f"# Error: SRT file not found: {srt_path}\n")
                f.write("# Please ensure the SRT file exists before running speaker identification\n")
            
            raise FileNotFoundError(f"SRT file not found: {srt_path}")
        
        # Parse SRT and save as JSON
        dialogue_lines = self.srt_parser.parse(srt_path, dialogue_json_path)
        
        # Check if parsing resulted in empty dialogue list
        if not dialogue_lines:
            logger.warning(f"⚠️ SRT file exists but contains no valid dialogue lines: {srt_path}")
        
        return dialogue_lines
    
    def _load_plot_scenes(self) -> List[Dict]:
        """Load plot scenes from JSON file."""
        scenes_path = self.path_handler.get_plot_scenes_json_path()
        
        if not os.path.exists(scenes_path):
            logger.warning(f"⚠️ Plot scenes file not found: {scenes_path}")
            return []
        
        try:
            with open(scenes_path, 'r', encoding='utf-8') as f:
                scenes_data = json.load(f)
                
            if isinstance(scenes_data, list):
                return scenes_data
            elif isinstance(scenes_data, dict) and 'scenes' in scenes_data:
                return scenes_data['scenes']
            else:
                logger.warning("⚠️ Invalid plot scenes format")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error loading plot scenes: {e}")
            return []
    
    def _identify_speakers(
        self, 
        dialogue_lines: List, 
        plot_scenes: List[Dict], 
        force_regenerate: bool
    ) -> List:
        """Identify speakers using LLM analysis."""
        speaker_analysis_path = self.path_handler.get_speaker_analysis_path()
        
        if not force_regenerate and os.path.exists(speaker_analysis_path):
            logger.info(f"📂 Loading existing speaker analysis from: {speaker_analysis_path}")
            try:
                with open(speaker_analysis_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                
                # Reconstruct dialogue lines with speaker info
                updated_lines = []
                for line_data in speaker_data.get('dialogue_lines', []):
                    # Fix: Create DialogueLine from dict, not from load_dialogue_json
                    from ..narrative_storage_management.narrative_models import DialogueLine
                    line = DialogueLine.from_dict(line_data)
                    if line:
                        updated_lines.append(line)
                
                if updated_lines:
                    return updated_lines
                    
            except Exception as e:
                logger.warning(f"⚠️ Error loading existing speaker analysis: {e}")
        
        # Get character context from previous episodes/season
        character_context = self._get_character_context()
        
        # Get episode entities from database for validation using CharacterService
        episode_entities = self._get_episode_entities()
        
        # Get episode plot for better validation context
        episode_plot = self._get_episode_plot()
        
        # Identify speakers
        dialogue_with_speakers = self.speaker_identifier.identify_speakers_for_episode(
            plot_scenes,
            dialogue_lines,
            character_context,
            episode_entities,
            episode_plot
        )
        
        # Save results
        speaker_data = {
            'episode_code': self.path_handler.get_episode_code(),
            'dialogue_lines': [line.to_dict() for line in dialogue_with_speakers],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(speaker_analysis_path), exist_ok=True)
        with open(speaker_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_data, f, indent=4, ensure_ascii=False)
        
        return dialogue_with_speakers
    
    def _get_character_context(self) -> Optional[str]:
        """Get character context from previous season summary."""
        try:
            season_summary_path = self.path_handler.get_season_summary_path()
            return load_previous_season_summary(season_summary_path)
        except Exception as e:
            logger.debug(f"Could not load character context: {e}")
            return None
    
    def _get_episode_entities(self) -> Optional[List[Dict]]:
        """Get entities that were extracted for this episode from the database as data dictionaries."""
        try:
            db_manager = DatabaseSessionManager()
            with db_manager.session_scope() as session:
                character_service = CharacterService(CharacterRepository(session))
                characters = character_service.get_episode_entities(self.series)
                
                # Convert to plain data dictionaries while session is active
                characters_data = []
                for char in characters:
                    # Force load relationships while session is active
                    appellations_data = [app.appellation for app in char.appellations]
                    
                    char_data = {
                        'entity_name': char.entity_name,
                        'best_appellation': char.best_appellation,
                        'appellations': appellations_data,
                        'series': char.series,
                        'biological_sex': char.biological_sex  # NEW: Include biological sex
                    }
                    characters_data.append(char_data)
                
                logger.info(f"📚 Loaded {len(characters_data)} episode entities as data dictionaries")
                return characters_data
        except Exception as e:
            logger.warning(f"⚠️ Could not load episode entities from database: {e}")
            return None
    
    def _get_episode_plot(self) -> Optional[str]:
        """Get the full episode plot for validation context."""
        try:
            # Fix: Use get_raw_plot_file_path instead of get_plot_file_path
            plot_path = self.path_handler.get_raw_plot_file_path()
            if os.path.exists(plot_path):
                with open(plot_path, 'r', encoding='utf-8') as f:
                    episode_plot = f.read().strip()
                logger.info(f"📖 Loaded episode plot ({len(episode_plot)} characters) for validation context")
                return episode_plot
            else:
                logger.debug(f"📖 Episode plot file not found: {plot_path}")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Could not load episode plot: {e}")
            return None
    
    def _ensure_scene_timestamps(
        self, 
        plot_scenes: List[Dict], 
        dialogue_lines: List, 
        force_regenerate: bool
    ) -> List[Dict]:
        """Ensure scenes have timestamp information."""
        if not plot_scenes:
            logger.warning("⚠️ No plot scenes available")
            return plot_scenes
        
        scene_timestamps_path = self.path_handler.get_scene_timestamps_path()
        
        # Check if timestamps already exist and we're not forcing regeneration
        if not force_regenerate and os.path.exists(scene_timestamps_path):
            logger.info(f"📂 Loading existing scene timestamps from: {scene_timestamps_path}")
            try:
                with open(scene_timestamps_path, 'r', encoding='utf-8') as f:
                    timestamp_data = json.load(f)
                
                # Merge timestamp data with plot scenes
                scenes_with_timestamps = []
                scene_timestamp_map = {
                    scene['scene_number']: scene 
                    for scene in timestamp_data.get('scenes', [])
                }
                
                for scene in plot_scenes:
                    scene_num = scene.get('scene_number')
                    if scene_num in scene_timestamp_map:
                        # Merge plot scene with timestamp data
                        merged_scene = {**scene, **scene_timestamp_map[scene_num]}
                        scenes_with_timestamps.append(merged_scene)
                    else:
                        scenes_with_timestamps.append(scene)
                
                # Check if all scenes have timestamps
                scenes_with_ts = [s for s in scenes_with_timestamps if 'start_seconds' in s and 'end_seconds' in s]
                if len(scenes_with_ts) == len(plot_scenes):
                    logger.info(f"✅ All {len(plot_scenes)} scenes have timestamps")
                    return scenes_with_timestamps
                else:
                    logger.warning(f"⚠️ Only {len(scenes_with_ts)}/{len(plot_scenes)} scenes have timestamps, regenerating...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error loading scene timestamps: {e}, regenerating...")
        
        # Generate timestamps using LLM with FORCED gap closure
        logger.info("🤖 Generating scene timestamps using LLM with FORCED gap closure")
        
        # Convert dialogue lines to SubtitleEntry format for mapping
        subtitle_entries = [
            SubtitleEntry(
                index=line.index,
                start_time=self._seconds_to_srt_timestamp(line.start_time),
                end_time=self._seconds_to_srt_timestamp(line.end_time),
                text=line.text,
                start_seconds=line.start_time,
                end_seconds=line.end_time
            )
            for line in dialogue_lines
        ]
        
        # Convert plot scenes to PlotScene objects for batch processing
        plot_scene_objects = []
        for scene in plot_scenes:
            plot_scene = PlotScene(
                scene_number=scene.get('scene_number', 0),
                plot_segment=scene.get('plot_segment', ''),
                start_time=scene.get('start_time'),
                end_time=scene.get('end_time')
            )
            plot_scene_objects.append(plot_scene)
        
        # Map all scenes to timestamps with FORCED gap closure
        logger.info(f"🕒 Mapping {len(plot_scene_objects)} scenes to timestamps with FORCED gap closure")
        try:
            # Import the new forced gap closure function
            from backend.src.plot_processing.subtitle_processing import map_scenes_to_timestamps
            
            mapped_scenes = map_scenes_to_timestamps(
                plot_scene_objects, subtitle_entries, self.llm
            )
            
            # Convert back to dict format
            scenes_with_timestamps = []
            for original_scene, mapped_scene in zip(plot_scenes, mapped_scenes):
                scene_with_timestamps = {
                    **original_scene,  # Keep original scene data
                    'start_seconds': mapped_scene.start_seconds,
                    'end_seconds': mapped_scene.end_seconds,
                    'start_time': mapped_scene.start_time,
                    'end_time': mapped_scene.end_time
                }
                scenes_with_timestamps.append(scene_with_timestamps)
                
            logger.info(f"✅ Successfully mapped all scenes with FORCED gap closure")
            
        except Exception as e:
            logger.error(f"❌ Failed to map scenes with forced gap closure: {e}")
            # Fallback to individual mapping without correction
            logger.info("🔄 Falling back to individual scene mapping")
            scenes_with_timestamps = []
            for scene in plot_scenes:
                plot_scene = PlotScene(
                    scene_number=scene.get('scene_number', 0),
                    plot_segment=scene.get('plot_segment', ''),
                    start_time=scene.get('start_time'),
                    end_time=scene.get('end_time')
                )
                
                try:
                    from backend.src.plot_processing.subtitle_processing import map_scenes_to_timestamps
                    # For single scene, use the new simplified approach
                    mapped_scenes = map_scenes_to_timestamps([plot_scene], subtitle_entries, self.llm)
                    mapped_scene = mapped_scenes[0] if mapped_scenes else plot_scene
                    scene_with_timestamps = {
                        **scene,
                        'start_seconds': mapped_scene.start_seconds,
                        'end_seconds': mapped_scene.end_seconds,
                        'start_time': mapped_scene.start_time,
                        'end_time': mapped_scene.end_time
                    }
                    scenes_with_timestamps.append(scene_with_timestamps)
                except Exception as scene_e:
                    logger.error(f"❌ Failed to map scene {scene.get('scene_number')}: {scene_e}")
                    scenes_with_timestamps.append(scene)
        
        # Save timestamp mappings
        self._save_scene_timestamps(scenes_with_timestamps)
        
        return scenes_with_timestamps
    
    def _save_scene_timestamps(self, scenes_with_timestamps: List[Dict]):
        """Save scene timestamp mappings to file."""
        scene_timestamps_path = self.path_handler.get_scene_timestamps_path()
        
        # Debug: Check what scenes have timestamps
        scenes_with_ts = [s for s in scenes_with_timestamps if 'start_seconds' in s and 'end_seconds' in s]
        scenes_without_ts = [s for s in scenes_with_timestamps if 'start_seconds' not in s or 'end_seconds' not in s]
        
        logger.info(f"💾 Preparing to save timestamps: {len(scenes_with_ts)} scenes have timestamps, {len(scenes_without_ts)} scenes missing timestamps")
        
        if scenes_without_ts:
            for scene in scenes_without_ts:
                logger.warning(f"⚠️ Scene {scene.get('scene_number')} missing timestamps: {scene.keys()}")
        
        timestamp_data = {
            'episode_code': self.path_handler.get_episode_code(),
            'scenes': [
                {
                    'scene_number': scene.get('scene_number'),
                    'start_seconds': scene.get('start_seconds'),
                    'end_seconds': scene.get('end_seconds'),
                    'plot_segment': scene.get('plot_segment', '')
                }
                for scene in scenes_with_timestamps
                if 'start_seconds' in scene and 'end_seconds' in scene and 
                   scene.get('start_seconds') is not None and scene.get('end_seconds') is not None
            ],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"💾 Saving {len(timestamp_data['scenes'])} scenes with valid timestamps to: {scene_timestamps_path}")
        
        os.makedirs(os.path.dirname(scene_timestamps_path), exist_ok=True)
        with open(scene_timestamps_path, 'w', encoding='utf-8') as f:
            json.dump(timestamp_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✅ Successfully saved scene timestamps file: {scene_timestamps_path}")
    
    def _seconds_to_srt_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _extract_faces(self, dialogue_lines: List, detector: str, force_regenerate: bool):
        """Extract faces from video at dialogue midpoints."""
        return self.face_extractor.extract_faces_from_subtitles(
            dialogue_lines,
            detector=detector,
            force_extract=force_regenerate
        )
    
    def _generate_embeddings(self, df_faces, model: str, force_regenerate: bool):
        """Generate face embeddings."""
        return self.face_embedder.generate_embeddings(
            df_faces,
            model=model,
            force_regenerate=force_regenerate
        )
    
    def _save_to_vector_store(self, df_faces):
        """Save embeddings to vector store."""
        self.face_vector_store.save_embeddings_to_vector_store(
            df_faces,
            embedding_model_name="Facenet512"
        )
    
    def _create_visualizations(self, df_faces: pd.DataFrame) -> Dict[str, str]:
        """Create cluster visualizations using ChromaDB-based clustering."""
        try:
            visualizer = FaceClusterVisualizer(self.path_handler)
            
            # Use ChromaDB-based visualization if faces have embeddings
            if 'embedding' in df_faces.columns and not df_faces.empty:
                visualization_paths = visualizer.create_cluster_visualizations_with_chromadb(
                    df_faces=df_faces,
                    embedding_model_name="Facenet512",
                    cosine_similarity_threshold=0.65,
                    min_cluster_size=5,
                    merge_threshold=0.85,
                    output_format="both",
                    save_plots=True
                )
            else:
                # Fallback to standard visualization
                visualization_paths = visualizer.create_cluster_visualizations(
                    df_faces=df_faces,
                    output_format="both",
                    save_plots=True
                )
            
            logger.info(f"📊 Created {len(visualization_paths)} visualization plots")
            return visualization_paths
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create visualizations: {e}")
            return {}
    
    def _strip_embeddings_from_clusters(self, cluster_info):
        """Remove embedding arrays from cluster info to reduce file size."""
        if not isinstance(cluster_info, dict):
            return cluster_info
        
        stripped_clusters = {}
        for cluster_id, cluster_data in cluster_info.items():
            if isinstance(cluster_data, dict):
                # Create a copy without the median_embedding field
                stripped_cluster = cluster_data.copy()
                if 'median_embedding' in stripped_cluster:
                    # Keep only metadata about the embedding, not the data itself
                    embedding_dim = stripped_cluster.get('embedding_dimension', 'unknown')
                    stripped_cluster['median_embedding'] = f"<embedding_array_excluded_{embedding_dim}_dims>"
                stripped_clusters[cluster_id] = stripped_cluster
            else:
                stripped_clusters[cluster_id] = cluster_data
        
        return stripped_clusters
    
    def _save_final_results(self, dialogue_lines: List, character_clusters: Dict, results: Dict):
        """Save final pipeline results."""
        
        def is_numpy_integer(obj):
            """Check if object is a numpy integer type."""
            return isinstance(obj, np.integer) or str(type(obj)).startswith('<class \'numpy.int')
        
        def is_numpy_float(obj):
            """Check if object is a numpy float type."""
            return isinstance(obj, np.floating) or str(type(obj)).startswith('<class \'numpy.float')
        
        def is_numpy_bool(obj):
            """Check if object is a numpy boolean type."""
            return isinstance(obj, np.bool_) or str(type(obj)).startswith('<class \'numpy.bool')

        def convert_for_json(obj):
            """Convert numpy/pandas types to JSON-serializable Python types."""
            if isinstance(obj, dict):
                # Convert both keys AND values
                converted_dict = {}
                for k, v in obj.items():
                    # Convert key - handle numpy/pandas types in keys
                    if hasattr(k, 'item'):  # numpy scalar
                        converted_key = k.item()
                    elif is_numpy_integer(k):
                        converted_key = int(k)
                    elif is_numpy_float(k):
                        converted_key = float(k)
                    elif is_numpy_bool(k):
                        converted_key = bool(k)
                    else:
                        converted_key = k
                    
                    # Convert value recursively
                    converted_value = convert_for_json(v)
                    converted_dict[converted_key] = converted_value
                return converted_dict
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif pd.isna(obj):  # pandas NaN/None
                return None
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif str(type(obj)).startswith('<class \'numpy.'):  # numpy types
                logger.debug(f"🔍 Converting numpy type {type(obj)}: {obj}")
                return obj.item() if hasattr(obj, 'item') else obj
            elif str(type(obj)).startswith('<class \'pandas.'):  # pandas types
                logger.debug(f"🔍 Converting pandas type {type(obj)}: {obj}")
                return obj.item() if hasattr(obj, 'item') else obj
            elif is_numpy_integer(obj):  # numpy integers
                logger.debug(f"🔍 Converting numpy integer {type(obj)}: {obj}")
                return int(obj)
            elif is_numpy_float(obj):  # numpy floats
                logger.debug(f"🔍 Converting numpy float {type(obj)}: {obj}")
                return float(obj)
            elif is_numpy_bool(obj):  # numpy bool
                logger.debug(f"🔍 Converting numpy bool {type(obj)}: {obj}")
                return bool(obj)
            else:
                # Log potentially problematic types
                obj_type = type(obj)
                if 'int64' in str(obj_type) or 'numpy' in str(obj_type) or 'pandas' in str(obj_type):
                    logger.warning(f"⚠️ Potentially problematic type not converted: {obj_type} = {obj}")
                return obj        # Save updated dialogue with resolved speakers
        final_dialogue_path = self.path_handler.get_dialogue_json_path().replace('.json', '_final.json')
        
        logger.info("🔍 Preparing final dialogue data for JSON serialization...")
        logger.info(f"   dialogue_lines count: {len(dialogue_lines)}")
        logger.info(f"   character_clusters type: {type(character_clusters)}, keys: {list(character_clusters.keys()) if character_clusters else 'None'}")
        logger.info(f"   results type: {type(results)}, keys: {list(results.keys()) if results else 'None'}")
        
        # Check a sample dialogue line for problematic types
        if dialogue_lines:
            sample_line = dialogue_lines[0]
            logger.info(f"   Sample dialogue line type: {type(sample_line)}")
            if hasattr(sample_line, 'to_dict'):
                sample_dict = sample_line.to_dict()
                logger.info(f"   Sample dialogue dict keys: {list(sample_dict.keys())}")
                for key, value in sample_dict.items():
                    if hasattr(value, '__dict__') or 'int64' in str(type(value)) or 'numpy' in str(type(value)):
                        logger.info(f"     {key}: {type(value)} = {value}")
        
        final_dialogue_data = convert_for_json([line.to_dict() for line in dialogue_lines])
        
        try:
            logger.info("🔍 Attempting final results JSON serialization...")
            with open(final_dialogue_path, 'w', encoding='utf-8') as f:
                json.dump(final_dialogue_data, f, indent=4, ensure_ascii=False)
            logger.info(f"💾 Saved final results to: {final_dialogue_path}")
        except Exception as e:
            logger.error(f"❌ Error saving final results: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            
            # Try to identify the problematic items
            logger.info("🔍 Analyzing final_dialogue_data contents for problematic types...")
            if isinstance(final_dialogue_data, dict):
                for key, value in final_dialogue_data.items():
                    try:
                        json.dumps({key: value})
                        logger.debug(f"   ✅ {key}: {type(value)} - OK")
                    except Exception as field_error:
                        logger.error(f"   ❌ {key}: {type(value)} - ERROR: {field_error}")
                        if isinstance(value, list) and value:
                            logger.error(f"      List sample types: {[type(item) for item in value[:3]]}")
                            if hasattr(value[0], '__dict__'):
                                logger.error(f"      First item attributes: {vars(value[0])}")
            elif isinstance(final_dialogue_data, list):
                for i, item in enumerate(final_dialogue_data[:5]):  # Check first 5 items
                    try:
                        json.dumps(item)
                        logger.debug(f"   ✅ Item {i}: {type(item)} - OK")
                    except Exception as field_error:
                        logger.error(f"   ❌ Item {i}: {type(item)} - ERROR: {field_error}")
                        if hasattr(item, '__dict__'):
                            logger.error(f"      Item attributes: {vars(item)}")
            raise
    
    def _generate_enhanced_srt(self, dialogue_lines: List) -> str:
        """
        Generate enhanced SRT file with speaker identification and confidence metadata.
        
        Format examples:
        - High confidence LLM: "MEREDITH: they say a person either has what it takes..."
        - Face recognition single: "DEREK (FACE_REC, original_uncertain: Dr. Yang): I think we should..."
        - Face recognition ambiguous: "DEREK/MEREDITH (FACE_REC, original_uncertain: Unknown): What about..."
        - Spatial outlier: "Unknown (FACE_REC_UNCERTAIN, original_uncertain: Dr. Bailey): Everyone listen..."
        """
        # Initialize enhanced SRT logger if available
        enhanced_srt_logger = None
        try:
            from .enhanced_srt_logger import EnhancedSRTLogger
            series = self.path_handler.get_series()
            season = self.path_handler.get_season()
            episode = self.path_handler.get_episode()
            enhanced_srt_logger = EnhancedSRTLogger(series, season, episode)
            enhanced_srt_logger.log_enhanced_srt_generation_start(len(dialogue_lines))
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize enhanced SRT logger: {e}")
        
        try:
            # Check if dialogue_lines is None or empty
            if not dialogue_lines:
                logger.warning("⚠️ No dialogue lines provided for enhanced SRT generation")
                enhanced_srt_path = self.path_handler.get_enhanced_srt_path()
                # Create empty enhanced SRT file
                with open(enhanced_srt_path, 'w', encoding='utf-8') as f:
                    f.write("# No dialogue lines provided for enhanced SRT generation\n")
                
                if enhanced_srt_logger:
                    enhanced_srt_logger.log_enhanced_srt_generation_complete(enhanced_srt_path)
                return enhanced_srt_path
                
            # Check for duplicate dialogue indices
            indices = [d.index for d in dialogue_lines if d]
            unique_indices = set(indices)
            if len(indices) != len(unique_indices):
                logger.warning(f"⚠️ Found duplicate dialogue indices in enhanced SRT generation!")
                logger.warning(f"   Total dialogue lines: {len(indices)}")
                logger.warning(f"   Unique indices: {len(unique_indices)}")
                
                # Find and log duplicates
                from collections import Counter
                index_counts = Counter(indices)
                duplicates = {idx: count for idx, count in index_counts.items() if count > 1}
                logger.warning(f"   Duplicate indices: {duplicates}")
                
                # Remove duplicates by keeping first occurrence
                seen_indices = set()
                unique_dialogue_lines = []
                for dialogue in dialogue_lines:
                    if dialogue and dialogue.index not in seen_indices:
                        unique_dialogue_lines.append(dialogue)
                        seen_indices.add(dialogue.index)
                
                dialogue_lines = unique_dialogue_lines
                logger.warning(f"   After deduplication: {len(dialogue_lines)} unique dialogue lines")
                
            # Boolean confidence approach - no threshold needed
            enhanced_srt_path = self.path_handler.get_enhanced_srt_path()
            
            logger.info(f"📝 Generating enhanced SRT: {enhanced_srt_path}")
            logger.info(f"   Processing {len(dialogue_lines)} dialogue lines")
            
            # Track SRT generation statistics
            srt_stats = {
                'total_dialogues': 0,
                'high_confidence_llm': 0,
                'face_recognition_single': 0,
                'face_recognition_multi': 0,
                'face_recognition_all': 0,
                'face_recognition_low': 0,
                'face_recognition_weak': 0,
                'character_median_direct': 0,
                'cluster_assigned': 0,
                'llm_fallback': 0,
                'unknown': 0,
                'face_recognition_below_threshold': 0,
                'llm_alternatives': 0,
                'llm_uncertain': 0
            }
            
            with open(enhanced_srt_path, 'w', encoding='utf-8') as f:
                for dialogue in dialogue_lines:
                    # Skip if dialogue is None
                    if not dialogue:
                        continue
                    
                    srt_stats['total_dialogues'] += 1
                    
                    # Write subtitle index
                    f.write(f"{dialogue.index}\n")
                    
                    # Write timestamp in proper SRT format (HH:MM:SS,mmm)
                    start_timestamp = self._seconds_to_srt_timestamp(dialogue.start_time)
                    end_timestamp = self._seconds_to_srt_timestamp(dialogue.end_time)
                    f.write(f"{start_timestamp} --> {end_timestamp}\n")
                    
                    # Generate speaker prefix based on boolean confidence and resolution method
                    speaker_prefix = self._generate_speaker_prefix(dialogue)  # Boolean confidence doesn't need threshold
                    
                    # Debug logging for statistics tracking
                    resolution_method = getattr(dialogue, 'resolution_method', None)
                    is_confident = getattr(dialogue, 'is_llm_confident', None)
                    speaker = getattr(dialogue, 'speaker', None)
                    candidate_speakers = getattr(dialogue, 'candidate_speakers', None) or []
                    if resolution_method == "character_median_direct":
                        logger.debug(f"🔍 [SRT_STATS] Dialogue {dialogue.index}: resolution_method={resolution_method}, is_confident={is_confident}, speaker='{speaker}', candidates={candidate_speakers}, speaker_prefix='{speaker_prefix}'")
                    elif resolution_method:
                        logger.debug(f"🔍 [SRT_STATS] Dialogue {dialogue.index}: resolution_method={resolution_method}, is_confident={is_confident}, speaker='{speaker}', candidates={candidate_speakers}, speaker_prefix='{speaker_prefix}'")
                    
                    # Track SRT prefix types
                    if "FACE_REC_MULTI" in speaker_prefix:
                        srt_stats['face_recognition_multi'] += 1
                    elif "FACE_REC_ALL" in speaker_prefix:
                        srt_stats['face_recognition_all'] += 1
                    elif "FACE_REC_LOW" in speaker_prefix:
                        srt_stats['face_recognition_low'] += 1
                    elif "FACE_REC_WEAK" in speaker_prefix:
                        srt_stats['face_recognition_weak'] += 1
                    elif "CHAR_MEDIAN_MULTI" in speaker_prefix:
                        srt_stats['character_median_direct'] += 1
                    elif "CHAR_MEDIAN" in speaker_prefix:
                        srt_stats['character_median_direct'] += 1
                    elif "CLUSTER_ASSIGNED" in speaker_prefix:
                        srt_stats['cluster_assigned'] += 1
                    elif "FACE_REC_BELOW_THRESHOLD" in speaker_prefix:
                        srt_stats['face_recognition_below_threshold'] += 1
                    elif "LLM_ALTERNATIVES" in speaker_prefix:
                        srt_stats['llm_alternatives'] += 1
                    elif "LLM_UNCERTAIN" in speaker_prefix:
                        srt_stats['llm_uncertain'] += 1
                    elif "FACE_REC" in speaker_prefix:
                        srt_stats['face_recognition_single'] += 1
                    elif dialogue.is_llm_confident:
                        srt_stats['high_confidence_llm'] += 1
                    elif dialogue.speaker == "Unknown":
                        srt_stats['unknown'] += 1
                    else:
                        srt_stats['llm_fallback'] += 1
                    
                    # Write enhanced subtitle text
                    f.write(f"{speaker_prefix}{dialogue.text}\n\n")
            
            # Log SRT generation statistics
            logger.info(f"📊 Enhanced SRT Statistics:")
            logger.info(f"   🎬 Total Dialogues: {srt_stats['total_dialogues']}")
            logger.info(f"   🧠 High Confidence LLM: {srt_stats['high_confidence_llm']}")
            logger.info(f"   👤 Face Recognition Single: {srt_stats['face_recognition_single']}")
            logger.info(f"   🎭 Face Recognition Multi: {srt_stats['face_recognition_multi']}")
            logger.info(f"   📊 Face Recognition All: {srt_stats['face_recognition_all']}")
            logger.info(f"   🔍 Face Recognition Low: {srt_stats['face_recognition_low']}")
            logger.info(f"   💪 Face Recognition Weak: {srt_stats['face_recognition_weak']}")
            logger.info(f"   🎯 Character Median Direct: {srt_stats['character_median_direct']}")
            logger.info(f"   🔗 Cluster Assigned: {srt_stats['cluster_assigned']}")
            logger.info(f"   🔄 LLM Fallback: {srt_stats['llm_fallback']}")
            logger.info(f"   ❓ Unknown: {srt_stats['unknown']}")
            logger.info(f"   🔍 Face Recognition Below Threshold: {srt_stats['face_recognition_below_threshold']}")
            logger.info(f"   🔄 LLM Alternatives: {srt_stats['llm_alternatives']}")
            logger.info(f"   🔄 LLM Uncertain: {srt_stats['llm_uncertain']}")
            
            logger.info(f"✅ Enhanced SRT saved: {enhanced_srt_path}")
            
            # Log completion if logger is available
            if enhanced_srt_logger:
                enhanced_srt_logger.log_enhanced_srt_generation_complete(enhanced_srt_path)
            
            return enhanced_srt_path
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced SRT: {e}")
            import traceback
            traceback.print_exc()
            # Return path to empty file on error
            enhanced_srt_path = self.path_handler.get_enhanced_srt_path()
            with open(enhanced_srt_path, 'w', encoding='utf-8') as f:
                f.write(f"# Error generating enhanced SRT: {e}\n")
            
            # Log completion if logger is available
            if enhanced_srt_logger:
                enhanced_srt_logger.log_enhanced_srt_generation_complete(enhanced_srt_path)
            
            return enhanced_srt_path
    
    def _generate_speaker_prefix(self, dialogue) -> str:
        """
        Generate speaker prefix for SRT format based on confidence and resolution method.
        
        Handles various confidence levels and resolution methods including:
        - High confidence LLM assignments
        - Low confidence LLM assignments
        - Face clustering resolutions
        - Multiple faces detected.
        
        Returns speaker prefix with appropriate formatting and metadata.
        """
        speaker = getattr(dialogue, 'speaker', None) or "Unknown"
        is_llm_confident = getattr(dialogue, 'is_llm_confident', None) or False
        resolution_method = getattr(dialogue, 'resolution_method', None)
        original_llm_speaker = getattr(dialogue, 'original_llm_speaker', None)
        candidate_speakers = getattr(dialogue, 'candidate_speakers', None) or []
        face_similarities = getattr(dialogue, 'face_similarities', None) or []
        
        # Ensure is_llm_confident is a boolean
        if is_llm_confident is not None:
            is_llm_confident = bool(is_llm_confident)
        else:
            is_llm_confident = False
        
        # High confidence LLM assignment (boolean)
        if is_llm_confident:
            return f"{speaker.upper()}: "
        
        # Low confidence - check resolution method and face data
        if resolution_method == "face_clustering_single":
            # Single face candidate from uncertain dialogue
            if original_llm_speaker:
                return f"{speaker.upper()} (FACE_REC, original_uncertain: {original_llm_speaker.upper()}): "
            else:
                return f"{speaker.upper()} (FACE_REC): "
        
        elif resolution_method == "face_clustering_multi":
            # Multi-face resolution
            if candidate_speakers and len(candidate_speakers) > 1:
                candidates_str = "/".join([name.upper() for name in candidate_speakers])
                return f"{speaker.upper()} (MULTI_FACE: {candidates_str}): "
            else:
                return f"{speaker.upper()} (MULTI_FACE): "
        
        elif resolution_method == "character_median_direct":
            # Direct character median matching
            if original_llm_speaker:
                return f"{speaker.upper()} (CHAR_MEDIAN, original: {original_llm_speaker.upper()}): "
            else:
                return f"{speaker.upper()} (CHAR_MEDIAN): "
        
        elif resolution_method == "database_validation":
            # Database validation (LLM speaker corrected)
            if original_llm_speaker:
                return f"{speaker.upper()} (DB_VALIDATED, original: {original_llm_speaker.upper()}): "
            else:
                return f"{speaker.upper()} (DB_VALIDATED): "
        
        elif resolution_method == "llm_direct":
            # Direct LLM assignment (no face processing)
            if original_llm_speaker:
                return f"{original_llm_speaker.upper()} (LLM_UNCERTAIN): "
            else:
                return f"{speaker.upper()} (LLM_UNCERTAIN): "
        
        # Fallback - low confidence LLM without face processing
        elif not is_llm_confident:
            return f"{speaker.upper()} (LOW_CONF): "
        
        # Default case
        else:
            return f"{speaker.upper()}: "
    
    def _attach_face_data_to_dialogue(self, dialogue_lines: List, df_faces: pd.DataFrame) -> List:
        """
        Attach face image paths and frame paths to dialogue lines based on dialogue index.
        
        Args:
            dialogue_lines: List of DialogueLine objects
            df_faces: DataFrame with face data including image_path and frame_path
            
        Returns:
            Updated dialogue lines with face image data
        """
        logger.info(f"🖼️ Attaching face data to {len(dialogue_lines)} dialogue lines")
        
        if df_faces.empty:
            logger.warning("⚠️ No face data available to attach")
            return dialogue_lines
        
        # Create a mapping from dialogue_index to face data
        face_data_map = {}
        for _, row in df_faces.iterrows():
            dialogue_idx = row['dialogue_index']
            if dialogue_idx not in face_data_map:
                face_data_map[dialogue_idx] = {
                    'face_image_paths': [],
                    'frame_image_paths': []
                }
            
            # Add face and frame paths (can be multiple faces per dialogue line)
            face_data_map[dialogue_idx]['face_image_paths'].append(row['image_path'])
            face_data_map[dialogue_idx]['frame_image_paths'].append(row['frame_path'])
        
        # Attach face data to dialogue lines
        faces_attached_count = 0
        for dialogue in dialogue_lines:
            if dialogue.index in face_data_map:
                dialogue.face_image_paths = face_data_map[dialogue.index]['face_image_paths']
                dialogue.frame_image_paths = face_data_map[dialogue.index]['frame_image_paths']
                faces_attached_count += 1
        
        logger.info(f"✅ Attached face data to {faces_attached_count}/{len(dialogue_lines)} dialogue lines")
        return dialogue_lines
    
    def _save_llm_checkpoint(self, dialogue_lines: List, llm_stats: Dict) -> None:
        """
        Save LLM speaker assignments to checkpoint file.
        
        This allows resuming face clustering without re-running expensive LLM calls.
        """
        checkpoint_path = self.path_handler.get_llm_checkpoint_path()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint_data = {
            'episode_code': self.path_handler.get_episode_code(),
            'timestamp': pd.Timestamp.now().isoformat(),
            'llm_stats': llm_stats,
            'dialogue_lines': [dialogue.to_dict() for dialogue in dialogue_lines]
        }
        
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved LLM checkpoint to: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Error saving LLM checkpoint: {e}")
    
    def _load_llm_checkpoint(self) -> Optional[List]:
        """
        Load LLM speaker assignments from checkpoint file.
        
        Returns:
            List of DialogueLine objects with LLM assignments, or None if checkpoint doesn't exist
        """
        checkpoint_path = self.path_handler.get_llm_checkpoint_path()
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # Convert back to DialogueLine objects
            from ..narrative_storage_management.narrative_models import DialogueLine
            dialogue_lines = [
                DialogueLine.from_dict(dialogue_dict) 
                for dialogue_dict in checkpoint_data['dialogue_lines']
            ]
            
            # Validate checkpoint data and count issues
            total_lines = len(dialogue_lines)
            null_speaker_count = sum(1 for line in dialogue_lines if line.speaker is None)
            null_confident_count = sum(1 for line in dialogue_lines if line.is_llm_confident is None)
            
            logger.info(f"📂 Loaded LLM checkpoint with {total_lines} dialogue lines")
            logger.info(f"📊 LLM Stats: {checkpoint_data.get('llm_stats', {})}")
            
            if null_speaker_count > 0:
                logger.warning(f"⚠️ Found {null_speaker_count}/{total_lines} dialogue lines with null speakers")
            
            if null_confident_count > 0:
                logger.warning(f"⚠️ Found {null_confident_count}/{total_lines} dialogue lines with null confidence")
            
            # If more than 50% have null speakers, checkpoint might be corrupted
            if null_speaker_count > total_lines * 0.5:
                logger.error(f"❌ LLM checkpoint appears corrupted: {null_speaker_count}/{total_lines} have null speakers")
                logger.info("🔧 Consider regenerating LLM assignments with force_regenerate=True")
            
            return dialogue_lines
            
        except Exception as e:
            logger.error(f"❌ Error loading LLM checkpoint: {e}")
            return None
    
    def validate_and_fix_llm_checkpoint(self) -> bool:
        """
        Validate and potentially fix issues with the LLM checkpoint.
        
        Returns:
            True if checkpoint is valid or was fixed, False if it needs regeneration
        """
        checkpoint_path = self.path_handler.get_llm_checkpoint_path()
        
        if not os.path.exists(checkpoint_path):
            logger.info("📂 No LLM checkpoint found - will need to run full LLM processing")
            return False
        
        try:
            dialogue_lines = self._load_llm_checkpoint()
            if not dialogue_lines:
                logger.error("❌ Could not load dialogue lines from checkpoint")
                return False
            
            total_lines = len(dialogue_lines)
            null_speaker_count = sum(1 for line in dialogue_lines if line.speaker is None)
            null_confident_count = sum(1 for line in dialogue_lines if line.is_llm_confident is None)
            
            logger.info(f"🔍 Checkpoint validation:")
            logger.info(f"   Total lines: {total_lines}")
            logger.info(f"   Null speakers: {null_speaker_count} ({null_speaker_count/total_lines*100:.1f}%)")
            logger.info(f"   Null confidence: {null_confident_count} ({null_confident_count/total_lines*100:.1f}%)")
            
            # Fix null confidence values
            fixed_confidence_count = 0
            for line in dialogue_lines:
                if line.is_llm_confident is None:
                    line.is_llm_confident = False
                    fixed_confidence_count += 1
            
            if fixed_confidence_count > 0:
                logger.info(f"🔧 Fixed {fixed_confidence_count} null confidence values")
                
                # Save the fixed checkpoint
                self._save_llm_checkpoint(dialogue_lines, {
                    'status': 'success',
                    'total_dialogue': total_lines,
                    'confident_speakers': sum(1 for line in dialogue_lines if line.is_llm_confident),
                    'confidence_rate': sum(1 for line in dialogue_lines if line.is_llm_confident) / total_lines * 100
                })
                logger.info("💾 Saved fixed checkpoint")
            
            # Determine if checkpoint is usable
            usable_threshold = 0.3  # At least 30% should have speaker assignments
            assigned_count = sum(1 for line in dialogue_lines if line.speaker is not None)
            assignment_rate = assigned_count / total_lines
            
            if assignment_rate >= usable_threshold:
                logger.info(f"✅ Checkpoint is usable: {assigned_count}/{total_lines} ({assignment_rate*100:.1f}%) have speaker assignments")
                return True
            else:
                logger.warning(f"⚠️ Checkpoint has low assignment rate: {assigned_count}/{total_lines} ({assignment_rate*100:.1f}%)")
                logger.info("💡 Consider regenerating LLM assignments for better results")
                return True  # Still usable, but user should consider regenerating
                
        except Exception as e:
            logger.error(f"❌ Error validating checkpoint: {e}")
            return False
    
    def _identify_speakers_or_load_checkpoint(
        self, 
        dialogue_lines: List, 
        plot_scenes: List, 
        force_regenerate: bool
    ) -> List:
        """
        Identify speakers using LLM or load from checkpoint if available.
        
        Args:
            dialogue_lines: Parsed dialogue lines
            plot_scenes: Plot scenes with timestamps
            force_regenerate: Force LLM regeneration even if checkpoint exists
            
        Returns:
            Dialogue lines with speaker assignments
        """
        if not force_regenerate:
            # Try to load from checkpoint first
            checkpoint_dialogue = self._load_llm_checkpoint()
            if checkpoint_dialogue is not None:
                logger.info("✅ Using LLM assignments from checkpoint (skipping expensive LLM calls)")
                return checkpoint_dialogue
        
        # No checkpoint or force regeneration - run LLM identification
        logger.info("🤖 Running LLM speaker identification (no checkpoint available)")
        return self._identify_speakers(dialogue_lines, plot_scenes, force_regenerate)
    
    def run_face_clustering_only(
        self,
        similarity_threshold: float = 0.8,
        embedding_model: str = "Facenet512",
        face_detector: str = "retinaface"
    ) -> Dict:
        """
        Run only the face clustering part of the pipeline (resume from LLM checkpoint).
        
        This is useful when you want to re-run face clustering with different parameters
        without making expensive LLM calls again.
        
        Args:
            similarity_threshold: Threshold for face-based speaker resolution
            embedding_model: DeepFace embedding model to use
            face_detector: DeepFace detector backend
            
        Returns:
            Dictionary with pipeline results
        """
            
        logger.info(f"🔄 Running face clustering only for {self.path_handler.get_episode_code()}")
        
        # Load dialogue with LLM assignments from checkpoint
        dialogue_with_speakers = self._load_llm_checkpoint()
        if dialogue_with_speakers is None:
            raise FileNotFoundError(
                f"No LLM checkpoint found at {self.path_handler.get_llm_checkpoint_path()}. "
                "Please run the full pipeline first to create the checkpoint."
            )
        
        results = {
            'episode_code': self.path_handler.get_episode_code(),
            'pipeline_type': 'face_clustering_only',
            'pipeline_steps': {}
        }
        
        try:
            # Step 4: Extract faces from video
            logger.info("👤 Step 4: Extracting faces from video")
            df_faces = self._extract_faces(dialogue_with_speakers, face_detector, False)
            results['pipeline_steps']['face_extraction'] = {
                'status': 'success',
                'faces_extracted': len(df_faces)
            }
            
            # Step 5: Generate face embeddings
            logger.info("🧠 Step 5: Generating face embeddings")
            df_faces_with_embeddings = self._generate_embeddings(df_faces, embedding_model, False)
            results['pipeline_steps']['face_embeddings'] = {
                'status': 'success',
                'embeddings_generated': len(df_faces_with_embeddings)
            }
            
            # Step 6: Save embeddings to vector store
            logger.info("💾 Step 6: Saving embeddings to vector store")
            if 'embedding' in df_faces_with_embeddings.columns and not df_faces_with_embeddings.empty:
                self.face_vector_store.save_embeddings_to_vector_store(df_faces_with_embeddings)
                results['pipeline_steps']['vector_store'] = {'status': 'success'}
            
            # Step 7-8: Run face clustering system for speaker assignment
            logger.info("🎭 Step 7-8: Running face clustering system for speaker assignment")
            final_dialogue, cluster_info, character_clusters, df_faces_clustered = self.face_clustering_system.run_face_clustering_pipeline(
                df_faces_with_embeddings,
                dialogue_with_speakers,
                expected_embedding_dim=config.face_embedding_dimension
            )
            
            # Count final confident speakers for statistics
            final_confident_count = len([
                line for line in final_dialogue 
                if line.is_llm_confident
            ])
            
            results['pipeline_steps']['face_clustering'] = {
                'status': 'success',
                'total_clusters': len(cluster_info),
                'character_clusters': len(character_clusters),
                'uncertain_speakers_processed': len(final_dialogue),
                'final_confident_count': final_confident_count,
                'final_confidence_rate': (final_confident_count / len(final_dialogue)) * 100 if final_dialogue else 0,
                'cluster_info': self._strip_embeddings_from_clusters(cluster_info),
                'character_clusters': character_clusters
            }
            
            # Step 9: Create visualizations
            logger.info("📊 Step 9: Creating cluster visualizations")
            visualization_paths = self._create_visualizations(df_faces_clustered)
            results['pipeline_steps']['visualization'] = {
                'status': 'success',
                'plots_created': len(visualization_paths),
                'visualization_paths': visualization_paths
            }
            
            # Step 10: Attach face data to dialogue lines
            logger.info("🖼️ Step 10: Attaching face data to dialogue lines")
            final_dialogue_with_faces = self._attach_face_data_to_dialogue(final_dialogue, df_faces_with_embeddings)
            results['pipeline_steps']['attach_face_data'] = {
                'status': 'success'
            }
            
            # Step 11: Save final results
            logger.info("💾 Step 11: Saving final results")
            self._save_final_results(final_dialogue_with_faces, character_clusters, results)
            results['pipeline_steps']['save_results'] = {
                'status': 'success'
            }
            
            # Step 12: Generate enhanced SRT with speaker identification
            logger.info("📝 Step 12: Generating enhanced SRT with speaker identification")
            logger.info(f"🔍 [DEBUG] Calling _generate_enhanced_srt from run_face_clustering_only")
            enhanced_srt_path = self._generate_enhanced_srt(final_dialogue_with_faces)
            results['pipeline_steps']['enhanced_srt'] = {
                'status': 'success',
                'enhanced_srt_path': enhanced_srt_path
            }
            
            logger.info(f"✅ Face clustering completed successfully for {self.path_handler.get_episode_code()}")
            
        except Exception as e:
            logger.error(f"❌ Face clustering pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            raise
        
        return results


def run_speaker_identification_pipeline(
    series: str,
    season: str, 
    episode: str,
    base_dir: str = "data",
    force_regenerate: bool = False,
    **kwargs
) -> Dict:
    """
    Convenience function to run the complete speaker identification pipeline.
    
    Args:
        series: Series name (e.g., "GA")
        season: Season name (e.g., "S01") 
        episode: Episode name (e.g., "E01")
        base_dir: Base data directory
        force_regenerate: If True, regenerate all intermediate files
        **kwargs: Additional parameters for pipeline configuration
        
    Returns:
        Dictionary with pipeline results
    """
    pipeline = SpeakerIdentificationPipeline(series, season, episode, base_dir)
    return pipeline.run_complete_pipeline(force_regenerate=force_regenerate, **kwargs)


def run_face_clustering_only(
    series: str,
    season: str,
    episode: str,
    base_dir: str = "data",
    **kwargs
) -> Dict:
    """
    Convenience function to run only face clustering (resume from LLM checkpoint).
    
    This skips expensive LLM calls and only runs face extraction, clustering, and resolution.
    Useful when you want to re-run face clustering with different parameters.
    
    Args:
        series: Series name (e.g., "GA")
        season: Season name (e.g., "S01") 
        episode: Episode name (e.g., "E01")
        base_dir: Base data directory
        **kwargs: Additional parameters for face clustering
        
    Returns:
        Dictionary with pipeline results
    """
    pipeline = SpeakerIdentificationPipeline(series, season, episode, base_dir)
    return pipeline.run_face_clustering_only(**kwargs)
