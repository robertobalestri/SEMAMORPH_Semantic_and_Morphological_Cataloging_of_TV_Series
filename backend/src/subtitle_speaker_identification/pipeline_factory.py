"""
Pipeline factory for speaker identification.
Creates the appropriate pipeline based on configuration mode.
"""

import logging
from typing import List, Dict
import os
import json

from .base_pipeline import BaseSpeakerIdentificationPipeline, DialogueLine, SpeakerIdentificationConfig
from .audio_only_pipeline import AudioOnlyPipeline
from .face_only_pipeline import FaceOnlyPipeline
from .complete_pipeline import CompletePipeline
from ..path_handler import PathHandler
from ..utils.logger_utils import setup_logging

logger = setup_logging(__name__)

class SpeakerIdentificationPipelineFactory:
    """Factory for creating speaker identification pipelines."""
    
    @staticmethod
    def create_pipeline(
        path_handler: PathHandler,
        config: SpeakerIdentificationConfig,
        llm=None
    ) -> BaseSpeakerIdentificationPipeline:
        """Create the appropriate pipeline based on configuration."""
        
        mode = config.get_pipeline_mode()
        
        if mode == "audio_only":
            logger.info("🎵 Creating audio-only pipeline")
            return AudioOnlyPipeline(path_handler, config, llm)
        
        elif mode == "face_only":
            logger.info("👤 Creating face-only pipeline")
            return FaceOnlyPipeline(path_handler, config, llm)
        
        elif mode == "complete":
            logger.info("🎯 Creating complete pipeline")
            return CompletePipeline(path_handler, config, llm)
        
        else:
            logger.warning(f"⚠️ Unknown pipeline mode '{mode}', defaulting to complete")
            return CompletePipeline(path_handler, config, llm)
    
    @staticmethod
    def validate_configuration(config: SpeakerIdentificationConfig) -> Dict:
        """Validate pipeline configuration."""
        issues = []
        
        # Validate pipeline mode
        mode = config.get_pipeline_mode()
        if mode not in ['audio_only', 'face_only', 'complete']:
            issues.append(f"Invalid pipeline mode: {mode}")
        
        # Validate component enablement
        if mode == 'audio_only' and not config.is_audio_enabled():
            issues.append("Audio-only mode requires audio_enabled = true")
        
        if mode == 'face_only' and not config.is_face_enabled():
            issues.append("Face-only mode requires face_enabled = true")
        
        if mode == 'complete' and not (config.is_audio_enabled() and config.is_face_enabled()):
            issues.append("Complete mode requires both audio_enabled = true and face_enabled = true")
        
        # Validate audio settings
        if config.is_audio_enabled():
            if not config.get_auth_token():
                issues.append("Audio processing requires HuggingFace auth_token")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'mode': mode
        }

def run_speaker_identification_pipeline(
    series: str,
    season: str,
    episode: str,
    video_path: str,
    dialogue_lines: List[DialogueLine],
    episode_entities: List[Dict],
    plot_scenes: List[Dict], # Added plot_scenes parameter
    base_dir: str = "data",
    force_regenerate: bool = False,
    llm=None
) -> Dict:
    """
    Run speaker identification pipeline with the appropriate mode.
    
    Args:
        series: Series name
        season: Season number
        episode: Episode number
        video_path: Path to video file
        dialogue_lines: List of dialogue lines to process
        episode_entities: List of episode entities for character mapping
        base_dir: Base directory for data
        force_regenerate: Force regeneration of intermediate files
        
    Returns:
        Dictionary with pipeline results and statistics
    """
    
    logger.info(f"🎯 Starting main pipeline for {series} {season} {episode}")
    logger.info(f"📋 Parameters: video_path={video_path}, dialogue_lines_count={len(dialogue_lines)}, episode_entities_count={len(episode_entities)}, plot_scenes_count={len(plot_scenes)}, force_regenerate={force_regenerate}")
    
    try:
        # Create path handler and config
        logger.info("🔧 Creating path handler and config...")
        path_handler = PathHandler(series, season, episode, base_dir)
        from ..config import config
        pipeline_config = SpeakerIdentificationConfig(config)
        logger.info(f"✅ Path handler created: {path_handler.get_episode_code()}")
        logger.info(f"✅ Pipeline config created with mode: {pipeline_config.get_pipeline_mode()}")
        
        # Validate configuration
        logger.info("🔍 Validating configuration...")
        validation_result = SpeakerIdentificationPipelineFactory.validate_configuration(pipeline_config)
        logger.info(f"✅ Configuration validation result: {validation_result}")
        if not validation_result['valid']:
            error_msg = f"Configuration validation failed: {validation_result['issues']}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Create pipeline
        logger.info("🏭 Creating pipeline...")
        pipeline = SpeakerIdentificationPipelineFactory.create_pipeline(path_handler, pipeline_config, llm)
        logger.info(f"✅ Pipeline created: {type(pipeline).__name__}")
        
        # Run pipeline
        logger.info("🚀 About to run pipeline...")
        logger.info(f"📋 Pipeline run parameters: video_path={video_path}, dialogue_lines_count={len(dialogue_lines)}, episode_entities_count={len(episode_entities)}")
        
        result_dialogues = pipeline.run_pipeline(video_path, dialogue_lines, episode_entities, plot_scenes)
        
        logger.info(f"✅ Pipeline run completed. Result dialogues count: {len(result_dialogues)}")
        
        # Calculate final statistics
        logger.info("📊 Calculating final statistics...")
        statistics = pipeline._calculate_statistics(result_dialogues)
        logger.info(f"✅ Statistics calculated: {statistics}")
        
        logger.info("🎉 Main pipeline completed successfully!")
        return {
            'episode_code': path_handler.get_episode_code(),
            'series': path_handler.get_series(),
            'season': path_handler.get_season(),
            'episode': path_handler.get_episode(),
            'pipeline_mode': validation_result['mode'],
            'dialogue_lines': result_dialogues,
            'statistics': statistics,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        logger.error(f"❌ Exception type: {type(e)}")
        logger.error(f"❌ Exception args: {e.args}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            'episode_code': path_handler.get_episode_code() if 'path_handler' in locals() else f"{series}{season}{episode}",
            'series': path_handler.get_series() if 'path_handler' in locals() else series,
            'season': path_handler.get_season() if 'path_handler' in locals() else season,
            'episode': path_handler.get_episode() if 'path_handler' in locals() else episode,
            'pipeline_mode': validation_result['mode'] if 'validation_result' in locals() else 'unknown',
            'dialogue_lines': dialogue_lines,
            'statistics': {},
            'success': False,
            'error': str(e)
        }


def run_speaker_identification_pipeline_compat(
    series: str,
    season: str,
    episode: str,
    base_dir: str = "data",
    force_regenerate: bool = False,
    face_similarity_threshold: float = 0.8,
    embedding_model: str = "Facenet512",
    face_detector: str = "retinaface",
    **kwargs
) -> Dict:
    """
    Compatibility wrapper for the old API that accepts the old parameters
    and converts them to the new pipeline format.
    
    This function maintains backward compatibility with the existing API
    while using our new Pyannote pipeline internally.
    """
    
    logger.info(f"🔄 Running compatibility wrapper for {series} {season} {episode}")
    logger.info(f"📋 Parameters: base_dir={base_dir}, force_regenerate={force_regenerate}, face_similarity_threshold={face_similarity_threshold}, embedding_model={embedding_model}, face_detector={face_detector}")
    
    try:
        # Create path handler
        logger.info("🔧 Creating path handler...")
        path_handler = PathHandler(series, season, episode, base_dir)
        logger.info(f"✅ Path handler created for episode: {path_handler.get_episode_code()}")
        
        # Get video path
        logger.info("🎬 Getting video path...")
        video_path = path_handler.get_video_file_path()
        logger.info(f"🎬 Video path: {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        logger.info("✅ Video file exists")
        
        # Parse SRT to get dialogue lines
        logger.info("📝 Parsing SRT file...")
        srt_path = path_handler.get_srt_file_path()
        logger.info(f"📝 SRT path: {srt_path}")
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"SRT file not found: {srt_path}")
        
        from .srt_parser import SRTParser
        srt_parser = SRTParser()
        dialogue_lines = srt_parser.parse(srt_path)
        logger.info(f"✅ Parsed {len(dialogue_lines)} dialogue lines from SRT")
        
        # Get episode entities
        logger.info("👥 Getting episode entities...")
        episode_entities_path = path_handler.get_episode_refined_entities_path()
        logger.info(f"👥 Episode entities path: {episode_entities_path}")
        episode_entities = []
        if os.path.exists(episode_entities_path):
            with open(episode_entities_path, 'r') as f:
                episode_entities = json.load(f)
            logger.info(f"✅ Loaded {len(episode_entities)} episode entities")
        else:
            logger.info("⚠️ No episode entities file found, using empty list")
        
        # Get LLM for character mapping
        logger.info("🧠 Getting LLM...")
        from ..ai_models.ai_models import get_llm, LLMType
        try:
            llm = get_llm(LLMType.INTELLIGENT)
            logger.info("✅ LLM initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ LLM initialization failed: {e}, using None")
            llm = None  # Fallback if LLM not available
        
        # Get plot scenes
        logger.info("📖 Getting plot scenes...")
        plot_scenes_path = path_handler.get_plot_scenes_json_path()
        logger.info(f"📖 Plot scenes path: {plot_scenes_path}")
        plot_scenes = []
        if os.path.exists(plot_scenes_path):
            with open(plot_scenes_path, 'r') as f:
                plot_scenes_data = json.load(f)
                plot_scenes = plot_scenes_data.get("scenes", [])
            logger.info(f"✅ Loaded {len(plot_scenes)} plot scenes")
        else:
            logger.info("⚠️ No plot scenes file found, using empty list")

        # Run the new pipeline
        logger.info("🚀 About to call run_speaker_identification_pipeline...")
        logger.info(f"📋 Pipeline parameters: series={series}, season={season}, episode={episode}, video_path={video_path}, dialogue_lines_count={len(dialogue_lines)}, episode_entities_count={len(episode_entities)}, plot_scenes_count={len(plot_scenes)}")
        
        result = run_speaker_identification_pipeline(
            series=series,
            season=season,
            episode=episode,
            video_path=video_path,
            dialogue_lines=dialogue_lines,
            episode_entities=episode_entities,
            plot_scenes=plot_scenes, # Pass plot_scenes here
            base_dir=base_dir,
            force_regenerate=force_regenerate,
            llm=llm
        )
        
        logger.info(f"✅ Pipeline completed. Result type: {type(result)}")
        logger.info(f"✅ Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        logger.info(f"✅ Success: {result.get('success', 'Unknown')}")
        
        # Convert result to old API format for compatibility
        if result['success']:
            logger.info("📊 Converting successful result to old API format...")
            # Calculate statistics in old format
            dialogue_lines = result['dialogue_lines']
            total_lines = len(dialogue_lines)
            speakers_identified = sum(1 for d in dialogue_lines if d.speaker)
            confident_dialogue = sum(1 for d in dialogue_lines if getattr(d, 'is_llm_confident', False))
            
            logger.info(f"📊 Statistics: total_lines={total_lines}, speakers_identified={speakers_identified}, confident_dialogue={confident_dialogue}")
            
            # Convert to old API response format
            return {
                'episode_code': result['episode_code'],
                'overall_stats': {
                    'total_dialogue_lines': total_lines,
                    'faces_extracted': 0,  # Will be updated by face processing
                    'speakers_identified': speakers_identified,
                    'high_confidence_dialogue': confident_dialogue,
                    'final_confidence_rate': (confident_dialogue / total_lines * 100) if total_lines > 0 else 0
                },
                'pipeline_steps': {
                    'pyannote_integration': {
                        'status': 'success',
                        'mode': result['pipeline_mode']
                    }
                },
                'dialogue_lines': [d.to_dict() for d in dialogue_lines],
                'success': True
            }
        else:
            logger.error(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            return {
                'episode_code': result['episode_code'],
                'overall_stats': {
                    'total_dialogue_lines': 0,
                    'faces_extracted': 0,
                    'speakers_identified': 0,
                    'high_confidence_dialogue': 0,
                    'final_confidence_rate': 0
                },
                'pipeline_steps': {
                    'pyannote_integration': {
                        'status': 'failed',
                        'error': result.get('error', 'Unknown error')
                    }
                },
                'success': False,
                'error': result.get('error', 'Unknown error')
            }
            
    except Exception as e:
        logger.error(f"❌ Compatibility wrapper failed: {e}")
        logger.error(f"❌ Exception type: {type(e)}")
        logger.error(f"❌ Exception args: {e.args}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise 