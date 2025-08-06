"""
Recap Assembler for generating final recap videos.

This module handles assembling individual video clips into a cohesive final recap,
with transitions, subtitles, and quality optimization.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.logger_utils import setup_logging
from .exceptions.recap_exceptions import VideoProcessingError
from .models.recap_models import RecapClip, RecapConfiguration, RecapMetadata
from .models.event_models import VectorEvent
from .utils.ffmpeg_utils import FFmpegUtils
from ..path_handler import PathHandler

logger = setup_logging(__name__)


class RecapAssembler:
    """Service for assembling individual clips into final recap video."""
    
    def __init__(self, path_handler: PathHandler, config: RecapConfiguration):
        self.path_handler = path_handler
        self.config = config
        self.ffmpeg_utils = FFmpegUtils()
        
    def assemble_final_recap(self, 
                           clips: List[RecapClip],
                           events: List[VectorEvent],
                           processing_metadata: Dict[str, Any]) -> RecapMetadata:
        """
        Assemble individual clips into final recap video with transitions and optimization.
        
        Args:
            clips: List of extracted and validated clips
            events: Original events for context
            processing_metadata: Metadata from previous processing steps
            
        Returns:
            RecapMetadata object with assembly results
        """
        try:
            logger.info(f"🎬 Assembling final recap from {len(clips)} clips")
            start_time = datetime.now()
            
            # Validate inputs
            if not clips:
                raise VideoProcessingError("No clips provided for assembly")
            
            # Order clips by narrative logic
            ordered_clips = self._order_clips_by_narrative(clips, events)
            
            # Add subtitle overlays if enabled
            if self.config.enable_subtitles:
                subtitled_clips = self._add_subtitle_overlays(ordered_clips)
            else:
                subtitled_clips = ordered_clips
            
            # Concatenate clips with transitions
            final_video_path = self._concatenate_clips_with_transitions(subtitled_clips)
            
            # Optimize final output
            optimized_path = self._optimize_final_output(final_video_path)
            
            # Generate clip specifications JSON
            clips_json_path = self._save_clips_specifications(ordered_clips)
            
            # Create metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metadata = RecapMetadata(
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode(),
                configuration=self.config,
                events=processing_metadata.get('recap_events', []),
                clips=ordered_clips,
                total_duration=sum(clip.duration for clip in ordered_clips),
                processing_time_seconds=processing_time,
                llm_queries_count=processing_metadata.get('llm_queries_count', 0),
                vector_search_count=processing_metadata.get('vector_search_count', 0),
                ffmpeg_operations_count=processing_metadata.get('ffmpeg_operations_count', 0),
                success=True,
                file_paths={
                    'final_video': optimized_path,
                    'clips_specification': clips_json_path,
                    'clips_directory': self.path_handler.get_recap_clips_dir()
                }
            )
            
            # Save metadata
            metadata_path = self._save_recap_metadata(metadata)
            metadata.file_paths['metadata'] = metadata_path
            
            logger.info(f"✅ Recap assembly completed successfully in {processing_time:.2f}s")
            logger.info(f"📁 Final video: {optimized_path}")
            logger.info(f"⏱️ Total duration: {metadata.total_duration:.1f}s")
            
            return metadata
            
        except Exception as e:
            # Create error metadata
            error_metadata = RecapMetadata(
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode(),
                configuration=self.config,
                events=[],
                clips=clips,
                total_duration=0,
                success=False,
                error_message=str(e)
            )
            
            raise VideoProcessingError(
                f"Recap assembly failed: {e}",
                operation="assemble_final_recap",
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
    
    def _order_clips_by_narrative(self, clips: List[RecapClip], events: List[VectorEvent]) -> List[RecapClip]:
        """Order clips by narrative logic for optimal viewing experience."""
        
        try:
            logger.info("📑 Ordering clips by narrative logic")
            
            # Create event lookup for additional context
            event_map = {event.id: event for event in events}
            
            # Create clip-event pairs for analysis
            clip_event_pairs = []
            for clip in clips:
                event = event_map.get(clip.event_id)
                if event:
                    clip_event_pairs.append((clip, event))
                else:
                    # Clip without matching event - add at end
                    clip_event_pairs.append((clip, None))
                    logger.warning(f"⚠️ Clip {clip.id} has no matching event")
            
            # Sort by multiple criteria
            def narrative_sort_key(pair):
                clip, event = pair
                
                # Primary: Arc type priority (Soap > Genre-Specific > Anthology)
                arc_priority = 0
                if event:
                    if event.arc_type == 'Soap Arc':
                        arc_priority = 3
                    elif event.arc_type == 'Genre-Specific Arc':
                        arc_priority = 2
                    elif event.arc_type == 'Anthology Arc':
                        arc_priority = 1
                
                # Secondary: Ordinal position within arc
                ordinal = event.ordinal_position if event else 999
                
                # Tertiary: Original timestamp to maintain chronological order
                timestamp = clip.start_time
                
                return (-arc_priority, ordinal, timestamp)
            
            # Sort clips
            sorted_pairs = sorted(clip_event_pairs, key=narrative_sort_key)
            ordered_clips = [pair[0] for pair in sorted_pairs]
            
            # Log ordering decision
            order_info = []
            for i, (clip, event) in enumerate(sorted_pairs):
                if event:
                    order_info.append(f"{i+1}. {event.arc_type}: {event.arc_title} (pos {event.ordinal_position})")
                else:
                    order_info.append(f"{i+1}. Unknown event")
            
            logger.debug(f"📋 Clip order:\\n" + "\\n".join(order_info))
            
            return ordered_clips
            
        except Exception as e:
            logger.warning(f"⚠️ Narrative ordering failed: {e}, using original order")
            return clips
    
    def _add_subtitle_overlays(self, clips: List[RecapClip]) -> List[RecapClip]:
        """Add subtitle overlays to clips if enabled."""
        
        try:
            logger.info(f"📝 Adding subtitle overlays to {len(clips)} clips")
            
            overlaid_clips = []
            
            for i, clip in enumerate(clips):
                if not clip.subtitle_lines or not clip.file_path:
                    overlaid_clips.append(clip)
                    continue
                
                try:
                    # Create output path for subtitled clip
                    base_path, ext = os.path.splitext(clip.file_path)
                    subtitled_path = f"{base_path}_subtitled{ext}"
                    
                    # Add subtitle overlay
                    success = self.ffmpeg_utils.add_subtitle_overlay(
                        input_video=clip.file_path,
                        output_path=subtitled_path,
                        subtitle_lines=clip.subtitle_lines,
                        start_time=0,  # Relative to clip start
                        end_time=clip.duration,
                        font_size=24,
                        font_color="white",
                        outline_color="black",
                        outline_width=2
                    )
                    
                    if success and os.path.exists(subtitled_path):
                        # Update clip to point to subtitled version
                        updated_clip = clip.copy()
                        updated_clip.file_path = subtitled_path
                        overlaid_clips.append(updated_clip)
                        logger.debug(f"✅ Clip {i+1}: Subtitles added")
                    else:
                        logger.warning(f"⚠️ Clip {i+1}: Subtitle overlay failed, using original")
                        overlaid_clips.append(clip)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Clip {i+1} subtitle overlay failed: {e}")
                    overlaid_clips.append(clip)
            
            logger.info("✅ Subtitle overlay processing completed")
            return overlaid_clips
            
        except Exception as e:
            logger.warning(f"⚠️ Subtitle overlay process failed: {e}, using original clips")
            return clips
    
    def _concatenate_clips_with_transitions(self, clips: List[RecapClip]) -> str:
        """Concatenate clips into final video with transitions."""
        
        try:
            logger.info(f"🔗 Concatenating {len(clips)} clips with transitions")
            
            # Prepare clip paths
            clip_paths = []
            for clip in clips:
                if clip.file_path and os.path.exists(clip.file_path):
                    clip_paths.append(clip.file_path)
                else:
                    logger.warning(f"⚠️ Clip file not found: {clip.file_path}")
            
            if not clip_paths:
                raise VideoProcessingError("No valid clip files found for concatenation")
            
            # Output path for concatenated video
            final_video_path = self.path_handler.get_final_recap_video_path()
            
            # Concatenate with transitions
            success = self.ffmpeg_utils.concatenate_clips(
                clip_paths=clip_paths,
                output_path=final_video_path,
                add_transitions=self.config.enable_transitions,
                transition_duration=0.5
            )
            
            if not success:
                raise VideoProcessingError("Video concatenation failed")
            
            if not os.path.exists(final_video_path):
                raise VideoProcessingError(f"Final video file not created: {final_video_path}")
            
            logger.info(f"✅ Video concatenation completed: {final_video_path}")
            return final_video_path
            
        except Exception as e:
            raise VideoProcessingError(
                f"Video concatenation failed: {e}",
                operation="concatenate_clips_with_transitions"
            )
    
    def _optimize_final_output(self, video_path: str) -> str:
        """Optimize final video for quality and file size."""
        
        try:
            logger.info("⚙️ Optimizing final video output")
            
            # Create optimized output path
            base_path, ext = os.path.splitext(video_path)
            optimized_path = f"{base_path}_optimized{ext}"
            
            # Get video info for optimization decisions
            video_info = self.ffmpeg_utils.get_video_info(video_path)
            
            # Build optimization command
            import subprocess
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-c:v', 'libx264',  # Force standard H.264 codec
                '-c:a', 'aac',      # Force standard AAC audio codec
                '-preset', 'medium', # Standard preset for compatibility
                '-profile:v', 'high', # H.264 High profile for broad compatibility
                '-level', '4.1',    # H.264 level 4.1 for 1080p compatibility
                '-pix_fmt', 'yuv420p',  # Standard pixel format for maximum compatibility
                '-crf', '23',       # Good quality/size balance
                '-movflags', '+faststart',  # Enable web streaming
                '-r', '30',         # Standard frame rate
                optimized_path
            ]
            
            # Add resolution optimization if needed
            duration = video_info.get('duration', 0)
            if duration > 0:
                # For recap videos, ensure reasonable bitrate
                target_bitrate = max(800, min(2500, int(duration * 150)))  # 800-2500 kbps for better quality
                cmd.extend(['-b:v', f'{target_bitrate}k'])
            
            logger.debug(f"🔧 Optimization command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ Optimization failed: {result.stderr}")
                logger.info("📁 Using unoptimized version")
                return video_path
            
            # Verify optimized file is valid
            if os.path.exists(optimized_path) and os.path.getsize(optimized_path) > 1024:
                optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
                original_size = os.path.getsize(video_path) / (1024 * 1024)
                
                logger.info(f"✅ Optimization completed: {original_size:.1f}MB → {optimized_size:.1f}MB")
                
                # Remove unoptimized version to save space
                try:
                    os.remove(video_path)
                except OSError:
                    pass
                
                return optimized_path
            else:
                logger.warning("⚠️ Optimized file invalid, using original")
                return video_path
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Optimization timed out, using original video")
            return video_path
        except Exception as e:
            logger.warning(f"⚠️ Optimization failed: {e}, using original video")
            return video_path
    
    def _save_clips_specifications(self, clips: List[RecapClip]) -> str:
        """Save clip specifications to JSON file."""
        
        try:
            clips_json_path = self.path_handler.get_recap_clips_json_path()
            
            # Create specifications data
            specifications = {
                'episode': {
                    'series': self.path_handler.get_series(),
                    'season': self.path_handler.get_season(),
                    'episode': self.path_handler.get_episode()
                },
                'generated_at': datetime.now().isoformat(),
                'configuration': self.config.dict(),
                'clips': [clip.dict() for clip in clips],
                'total_clips': len(clips),
                'total_duration': sum(clip.duration for clip in clips),
                'clip_order': [clip.id for clip in clips]
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(clips_json_path), exist_ok=True)
            
            # Save JSON
            with open(clips_json_path, 'w', encoding='utf-8') as f:
                json.dump(specifications, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 Clip specifications saved: {clips_json_path}")
            return clips_json_path
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save clip specifications: {e}")
            return ""
    
    def _save_recap_metadata(self, metadata: RecapMetadata) -> str:
        """Save recap metadata to JSON file."""
        
        try:
            metadata_path = self.path_handler.get_recap_metadata_path()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            # Save metadata as JSON
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.dict(), f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📋 Recap metadata saved: {metadata_path}")
            return metadata_path
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save recap metadata: {e}")
            return ""
    
    def validate_final_output(self, metadata: RecapMetadata) -> Dict[str, Any]:
        """
        Validate the final recap output against quality standards.
        
        Args:
            metadata: RecapMetadata with assembly results
            
        Returns:
            Dictionary with validation results
        """
        try:
            logger.info("🔍 Validating final recap output")
            
            validation_results = {
                'overall_success': True,
                'issues': [],
                'warnings': [],
                'metrics': {}
            }
            
            # Check final video file
            final_video = metadata.file_paths.get('final_video')
            if not final_video or not os.path.exists(final_video):
                validation_results['issues'].append("Final video file not found")
                validation_results['overall_success'] = False
            else:
                # Video file metrics
                file_size = os.path.getsize(final_video) / (1024 * 1024)  # MB
                validation_results['metrics']['file_size_mb'] = file_size
                
                if file_size < 1:
                    validation_results['warnings'].append(f"Small file size: {file_size:.1f}MB")
                elif file_size > 50:
                    validation_results['warnings'].append(f"Large file size: {file_size:.1f}MB")
            
            # Duration validation
            target_duration = metadata.configuration.target_duration_seconds
            actual_duration = metadata.total_duration
            duration_diff = abs(actual_duration - target_duration)
            
            validation_results['metrics']['target_duration'] = target_duration
            validation_results['metrics']['actual_duration'] = actual_duration
            validation_results['metrics']['duration_difference'] = duration_diff
            
            if duration_diff > target_duration * 0.3:  # More than 30% off
                validation_results['warnings'].append(
                    f"Duration significantly off target: {actual_duration:.1f}s vs {target_duration}s"
                )
            
            # Clip count validation
            expected_clips = min(len(metadata.events), metadata.configuration.max_events)
            actual_clips = len(metadata.clips)
            
            validation_results['metrics']['expected_clips'] = expected_clips
            validation_results['metrics']['actual_clips'] = actual_clips
            
            if actual_clips < expected_clips * 0.7:  # Less than 70% of expected
                validation_results['warnings'].append(
                    f"Fewer clips than expected: {actual_clips} vs {expected_clips}"
                )
            
            # Processing efficiency
            processing_time = metadata.processing_time_seconds or 0
            if processing_time > 600:  # More than 10 minutes
                validation_results['warnings'].append(f"Long processing time: {processing_time:.1f}s")
            
            validation_results['metrics']['processing_time_seconds'] = processing_time
            
            # Quality assessment
            if metadata.clips:
                avg_video_quality = sum(
                    clip.video_quality_score or 0 for clip in metadata.clips
                ) / len(metadata.clips)
                
                avg_audio_quality = sum(
                    clip.audio_quality_score or 0 for clip in metadata.clips
                ) / len(metadata.clips)
                
                validation_results['metrics']['average_video_quality'] = avg_video_quality
                validation_results['metrics']['average_audio_quality'] = avg_audio_quality
                
                quality_threshold = metadata.configuration.quality_threshold
                if avg_video_quality < quality_threshold or avg_audio_quality < quality_threshold:
                    validation_results['warnings'].append(
                        f"Average quality below threshold: video={avg_video_quality:.2f}, audio={avg_audio_quality:.2f}"
                    )
            
            # Overall assessment
            if len(validation_results['issues']) == 0:
                if len(validation_results['warnings']) == 0:
                    validation_results['status'] = 'excellent'
                elif len(validation_results['warnings']) <= 2:
                    validation_results['status'] = 'good'
                else:
                    validation_results['status'] = 'acceptable'
            else:
                validation_results['status'] = 'poor'
                validation_results['overall_success'] = False
            
            logger.info(f"✅ Final output validation: {validation_results['status']}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Final output validation failed: {e}")
            return {
                'overall_success': False,
                'status': 'error',
                'issues': [f"Validation error: {e}"],
                'warnings': [],
                'metrics': {}
            }
