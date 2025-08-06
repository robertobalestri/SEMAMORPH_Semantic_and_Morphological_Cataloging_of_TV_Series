"""
Video Clip Extractor for recap generation.

This module handles extracting video clips from source episodes based on
selected events and their subtitle sequences, with quality validation.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.logger_utils import setup_logging
from .exceptions.recap_exceptions import VideoProcessingError
from .models.recap_models import RecapClip, RecapConfiguration
from .models.event_models import VectorEvent, SubtitleSequence
from .utils.ffmpeg_utils import FFmpegUtils, validate_clip_quality
from ..path_handler import PathHandler

logger = setup_logging(__name__)


class VideoClipExtractor:
    """Service for extracting video clips from source episodes."""
    
    def __init__(self, path_handler: PathHandler, config: RecapConfiguration):
        self.path_handler = path_handler
        self.config = config
        self.ffmpeg_utils = FFmpegUtils()
        
    def extract_clips_for_events(self, 
                                selected_events: List[VectorEvent],
                                subtitle_sequences: List[SubtitleSequence]) -> List[RecapClip]:
        """
        Extract video clips for all selected events based on their subtitle sequences.
        
        Args:
            selected_events: Events selected for recap inclusion
            subtitle_sequences: Optimized subtitle sequences for each event
            
        Returns:
            List of RecapClip objects with extraction metadata
        """
        try:
            logger.info(f"🎬 Extracting video clips for {len(selected_events)} events")
            
            # Validate inputs
            if len(selected_events) != len(subtitle_sequences):
                logger.warning(f"⚠️ Event count ({len(selected_events)}) != sequence count ({len(subtitle_sequences)})")
                # Match by event_id
                sequence_map = {seq.event_id: seq for seq in subtitle_sequences}
                matched_sequences = [sequence_map.get(event.id) for event in selected_events]
                subtitle_sequences = [seq for seq in matched_sequences if seq is not None]
                logger.info(f"📊 Matched {len(subtitle_sequences)} sequences to events")
            
            # Ensure clips directory exists
            clips_dir = self.path_handler.get_recap_clips_dir()
            os.makedirs(clips_dir, exist_ok=True)
            
            extracted_clips = []
            total_duration = 0
            
            for i, (event, sequence) in enumerate(zip(selected_events, subtitle_sequences)):
                logger.debug(f"🎥 Extracting clip {i+1}/{len(selected_events)}: {event.arc_title}")
                
                try:
                    clip = self._extract_single_clip(event, sequence, i + 1)
                    if clip:
                        extracted_clips.append(clip)
                        total_duration += clip.duration
                        logger.debug(f"✅ Clip {i+1}: {clip.duration:.1f}s extracted successfully")
                    else:
                        logger.warning(f"⚠️ Clip {i+1}: Extraction failed")
                        
                except Exception as e:
                    logger.error(f"❌ Clip {i+1} extraction failed: {e}")
                    continue
            
            logger.info(f"✅ Clip extraction completed: {len(extracted_clips)}/{len(selected_events)} clips, {total_duration:.1f}s total")
            
            # Validate total duration against target
            target_duration = self.config.target_duration_seconds
            if total_duration > target_duration * 1.3:
                logger.warning(f"⚠️ Total duration ({total_duration:.1f}s) significantly exceeds target ({target_duration}s)")
            
            return extracted_clips
            
        except Exception as e:
            raise VideoProcessingError(
                f"Clip extraction failed: {e}",
                operation="extract_clips_for_events",
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
    
    def normalize_audio_levels(self, clips: List[RecapClip]) -> List[RecapClip]:
        """
        Normalize audio levels across all clips for consistent playback.
        
        Args:
            clips: List of extracted clips
            
        Returns:
            List of clips with normalized audio
        """
        try:
            logger.info(f"🔊 Normalizing audio levels for {len(clips)} clips")
            
            normalized_clips = []
            
            for i, clip in enumerate(clips):
                logger.debug(f"🎵 Normalizing audio for clip {i+1}/{len(clips)}")
                
                try:
                    normalized_clip = self._normalize_single_clip_audio(clip)
                    normalized_clips.append(normalized_clip)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Audio normalization failed for clip {i+1}: {e}, using original")
                    normalized_clips.append(clip)
            
            logger.info("✅ Audio normalization completed")
            return normalized_clips
            
        except Exception as e:
            raise VideoProcessingError(
                f"Audio normalization failed: {e}",
                operation="normalize_audio_levels",
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
    
    def _extract_single_clip(self, 
                           event: VectorEvent,
                           sequence: SubtitleSequence,
                           clip_number: int) -> Optional[RecapClip]:
        """Extract a single video clip for an event."""
        
        try:
            # Determine source video based on event metadata
            source_video = self._get_source_video_for_event(event)
            if not source_video or not os.path.exists(source_video):
                logger.error(f"❌ Source video not found for event {event.id}: {source_video}")
                return None
            
            # Audio/Video sync verification logging
            logger.debug(f"🔍 Audio/Video Sync Verification:")
            logger.debug(f"  📺 Event: {event.series} {event.season} {event.episode}")
            logger.debug(f"  🎬 Source video: {source_video}")
            logger.debug(f"  ⏰ Subtitle timing: {sequence.start_time:.3f}s - {sequence.end_time:.3f}s")
            logger.debug(f"  🎯 Duration: {sequence.duration:.3f}s")
            logger.debug(f"  📝 Lines: {len(sequence.lines)} subtitle lines")
            
            logger.debug(f"📹 Using source video: {source_video}")
            
            # Generate clip ID and output path
            clip_id = f"{event.id}_{clip_number:02d}"
            output_path = self.path_handler.get_individual_clip_path(clip_id)
            
            # Validate timing constraints
            duration = sequence.duration
            if duration < self.config.min_event_duration:
                logger.warning(f"⚠️ Clip duration ({duration:.1f}s) below minimum ({self.config.min_event_duration}s)")
                # Extend duration to meet minimum
                extension = self.config.min_event_duration - duration
                sequence.end_time += extension
                duration = self.config.min_event_duration
            
            if duration > self.config.max_event_duration:
                logger.warning(f"⚠️ Clip duration ({duration:.1f}s) exceeds maximum ({self.config.max_event_duration}s)")
                # Trim to maximum
                sequence.end_time = sequence.start_time + self.config.max_event_duration
                duration = self.config.max_event_duration
            
            # Extract clip using FFmpeg
            success = self.ffmpeg_utils.extract_clip(
                input_video=source_video,
                output_path=output_path,
                start_time=sequence.start_time,
                end_time=sequence.end_time,
                preset=self.config.ffmpeg_preset,
                video_codec=self.config.video_codec,
                audio_codec=self.config.audio_codec
            )
            
            if not success:
                logger.error(f"❌ FFmpeg extraction failed for clip {clip_id}")
                return None
            
            # Validate clip quality
            quality_info = validate_clip_quality(output_path)
            
            if not quality_info['valid']:
                logger.error(f"❌ Clip quality validation failed: {quality_info.get('error', 'Unknown error')}")
                return None
            
            # Create RecapClip object
            clip = RecapClip(
                id=clip_id,
                event_id=event.id,
                start_time=sequence.start_time,
                end_time=sequence.end_time,
                duration=duration,
                subtitle_lines=sequence.lines,
                video_quality_score=self._assess_video_quality(quality_info),
                audio_quality_score=self._assess_audio_quality(quality_info),
                has_dialogue=sequence.has_dialogue,
                main_speakers=sequence.speakers,
                file_path=output_path,
                extraction_method="ffmpeg"
            )
            
            return clip
            
        except Exception as e:
            logger.error(f"❌ Single clip extraction failed: {e}")
            return None
    
    def _normalize_single_clip_audio(self, clip: RecapClip) -> RecapClip:
        """Normalize audio for a single clip."""
        
        if not clip.file_path or not os.path.exists(clip.file_path):
            logger.warning(f"⚠️ Clip file not found for normalization: {clip.file_path}")
            return clip
        
        try:
            # Create normalized output path
            base_path, ext = os.path.splitext(clip.file_path)
            normalized_path = f"{base_path}_normalized{ext}"
            
            # Use FFmpeg to normalize audio
            # This is a simplified approach - more sophisticated methods could be used
            import subprocess
            
            cmd = [
                'ffmpeg', '-y',
                '-i', clip.file_path,
                '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # EBU R128 loudness normalization
                '-c:v', 'copy',  # Copy video without re-encoding
                normalized_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ Audio normalization command failed: {result.stderr}")
                return clip
            
            # Verify normalized file exists and is valid
            if os.path.exists(normalized_path) and os.path.getsize(normalized_path) > 1024:
                # Replace original with normalized version
                os.replace(normalized_path, clip.file_path)
                
                # Update audio quality score
                updated_clip = clip.copy()
                updated_clip.audio_quality_score = min(1.0, (clip.audio_quality_score or 0.7) + 0.1)
                
                return updated_clip
            else:
                logger.warning(f"⚠️ Normalized audio file invalid or too small")
                return clip
                
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Audio normalization timed out for clip {clip.id}")
            return clip
        except Exception as e:
            logger.warning(f"⚠️ Audio normalization failed for clip {clip.id}: {e}")
            return clip
    
    def _assess_video_quality(self, quality_info: Dict[str, Any]) -> float:
        """Assess video quality based on validation results."""
        
        score = 0.5  # Base score
        
        # File size assessment
        file_size_mb = quality_info.get('file_size_mb', 0)
        if file_size_mb > 5:  # Good size for a short clip
            score += 0.2
        elif file_size_mb > 1:
            score += 0.1
        
        # Duration assessment
        duration = quality_info.get('duration', 0)
        if 3 <= duration <= 15:  # Reasonable duration
            score += 0.2
        elif 1 <= duration <= 20:
            score += 0.1
        
        # Stream presence
        if quality_info.get('has_video', False):
            score += 0.2
        if quality_info.get('has_audio', False):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_audio_quality(self, quality_info: Dict[str, Any]) -> float:
        """Assess audio quality based on validation results."""
        
        score = 0.5  # Base score
        
        # Audio stream presence
        if quality_info.get('has_audio', False):
            score += 0.3
        else:
            return 0.1  # Very low score if no audio
        
        # File size (indirectly indicates audio quality)
        file_size_mb = quality_info.get('file_size_mb', 0)
        if file_size_mb > 2:  # Good size suggests good audio quality
            score += 0.2
        
        return min(1.0, score)
    
    def validate_extracted_clips(self, clips: List[RecapClip]) -> Dict[str, Any]:
        """
        Validate all extracted clips and return quality report.
        
        Args:
            clips: List of extracted clips
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        try:
            logger.info(f"🔍 Validating {len(clips)} extracted clips")
            
            validation_results = {
                'total_clips': len(clips),
                'valid_clips': 0,
                'invalid_clips': 0,
                'quality_issues': [],
                'total_duration': 0,
                'average_video_quality': 0,
                'average_audio_quality': 0,
                'clips_below_threshold': 0
            }
            
            quality_threshold = self.config.quality_threshold
            
            for clip in clips:
                validation_results['total_duration'] += clip.duration
                
                # Check file existence
                if not clip.file_path or not os.path.exists(clip.file_path):
                    validation_results['invalid_clips'] += 1
                    validation_results['quality_issues'].append(f"Clip {clip.id}: File not found")
                    continue
                
                # Check quality scores
                video_quality = clip.video_quality_score or 0
                audio_quality = clip.audio_quality_score or 0
                
                validation_results['average_video_quality'] += video_quality
                validation_results['average_audio_quality'] += audio_quality
                
                # Check quality threshold
                overall_quality = (video_quality + audio_quality) / 2
                if overall_quality < quality_threshold:
                    validation_results['clips_below_threshold'] += 1
                    validation_results['quality_issues'].append(
                        f"Clip {clip.id}: Quality {overall_quality:.2f} below threshold {quality_threshold}"
                    )
                
                validation_results['valid_clips'] += 1
            
            # Calculate averages
            if clips:
                validation_results['average_video_quality'] /= len(clips)
                validation_results['average_audio_quality'] /= len(clips)
            
            # Overall assessment
            success_rate = validation_results['valid_clips'] / len(clips) if clips else 0
            
            if success_rate >= 0.9 and validation_results['clips_below_threshold'] == 0:
                validation_results['overall_status'] = 'excellent'
            elif success_rate >= 0.8 and validation_results['clips_below_threshold'] <= 1:
                validation_results['overall_status'] = 'good'
            elif success_rate >= 0.6:
                validation_results['overall_status'] = 'acceptable'
            else:
                validation_results['overall_status'] = 'poor'
            
            logger.info(f"✅ Clip validation completed: {validation_results['overall_status']} quality")
            logger.info(f"📊 Valid clips: {validation_results['valid_clips']}/{validation_results['total_clips']}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Clip validation failed: {e}")
            return {
                'total_clips': len(clips),
                'valid_clips': 0,
                'invalid_clips': len(clips),
                'overall_status': 'error',
                'error': str(e)
            }
    
    def _get_source_video_for_event(self, event: VectorEvent) -> Optional[str]:
        """
        Determine the source video file path for an event based on its metadata.
        
        Args:
            event: The VectorEvent with series/season/episode metadata
            
        Returns:
            Path to the source video file for this event
        """
        # Get event's episode information
        event_series = getattr(event, 'series', self.path_handler.get_series())
        event_season = getattr(event, 'season', self.path_handler.get_season())
        event_episode = getattr(event, 'episode', self.path_handler.get_episode())
        
        # Construct path to the event's source video
        source_video_path = os.path.join(
            self.path_handler.base_dir,
            event_series,
            event_season,
            event_episode,
            f"{event_series}{event_season}{event_episode}.mp4"
        )
        
        logger.debug(f"🎯 Event {event.id} source: {event_series} {event_season} {event_episode}")
        logger.debug(f"📁 Source video path: {source_video_path}")
        
        return source_video_path
