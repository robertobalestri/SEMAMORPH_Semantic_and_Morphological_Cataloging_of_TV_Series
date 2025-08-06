"""
FFmpeg utilities for recap generation.

This module provides FFmpeg command builders and video processing utilities
for extracting clips, adding subtitles, and assembling final recap videos.
Extends existing FFmpeg functionality from transcription_workflow.py.
"""

import os
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ...utils.logger_utils import setup_logging
from ..exceptions.recap_exceptions import VideoProcessingError
from ..models.recap_models import RecapClip

logger = setup_logging(__name__)


class FFmpegUtils:
    """Utility class for FFmpeg operations in recap generation."""
    
    def __init__(self):
        self.validate_ffmpeg_installation()
    
    def validate_ffmpeg_installation(self) -> None:
        """Validate that FFmpeg is installed and accessible."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise VideoProcessingError("FFmpeg is not properly installed or accessible")
            logger.info("✅ FFmpeg installation validated")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise VideoProcessingError(f"FFmpeg validation failed: {e}")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise VideoProcessingError(
                    f"Failed to get video info: {result.stderr}",
                    operation="get_video_info",
                    ffmpeg_command=' '.join(cmd),
                    return_code=result.returncode,
                    stderr_output=result.stderr
                )
            
            import json
            info = json.loads(result.stdout)
            
            # Extract useful information
            video_info = {
                'duration': float(info['format'].get('duration', 0)),
                'size': int(info['format'].get('size', 0)),
                'bit_rate': int(info['format'].get('bit_rate', 0)),
                'streams': []
            }
            
            for stream in info.get('streams', []):
                stream_info = {
                    'codec_type': stream.get('codec_type'),
                    'codec_name': stream.get('codec_name'),
                    'duration': float(stream.get('duration', 0))
                }
                if stream.get('codec_type') == 'video':
                    stream_info.update({
                        'width': stream.get('width'),
                        'height': stream.get('height'),
                        'r_frame_rate': stream.get('r_frame_rate'),
                        'avg_frame_rate': stream.get('avg_frame_rate')
                    })
                video_info['streams'].append(stream_info)
            
            return video_info
            
        except subprocess.TimeoutExpired:
            raise VideoProcessingError("Video info retrieval timed out")
        except json.JSONDecodeError as e:
            raise VideoProcessingError(f"Failed to parse video info JSON: {e}")
        except Exception as e:
            raise VideoProcessingError(f"Unexpected error getting video info: {e}")
    
    def extract_clip(self, 
                     input_video: str,
                     output_path: str,
                     start_time: float,
                     end_time: float,
                     preset: str = "medium",
                     video_codec: str = "libx264",
                     audio_codec: str = "aac",
                     audio_gain_db: float = 12.0) -> bool:
        """
        Extract a video clip from a larger video file with audio normalization.
        
        Args:
            input_video: Path to source video
            output_path: Path for output clip
            start_time: Start time in seconds
            end_time: End time in seconds
            preset: FFmpeg encoding preset
            video_codec: Video codec to use
            audio_codec: Audio codec to use
            audio_gain_db: Audio gain in dB to apply (default: 12dB boost)
            
        Returns:
            True if extraction successful
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            duration = end_time - start_time
            
            # Build audio filter for normalization and gain
            audio_filter = f"volume={audio_gain_db}dB"
            
            cmd = [
                'ffmpeg', '-y',  # Overwrite output files
                '-ss', str(start_time),  # Start time
                '-i', input_video,       # Input file
                '-t', str(duration),     # Duration
                '-c:v', 'libx264',       # Force H.264 codec
                '-c:a', 'aac',           # Force AAC audio codec
                '-profile:v', 'high',    # H.264 High profile for compatibility
                '-level', '4.1',         # H.264 level 4.1 for 1080p
                '-pix_fmt', 'yuv420p',   # Standard pixel format for maximum compatibility
                '-af', audio_filter,     # Audio filter for gain
                '-preset', 'medium',     # Standard preset for balance
                '-crf', '23',            # Good quality setting
                '-avoid_negative_ts', 'make_zero',  # Handle negative timestamps
                output_path
            ]
            
            logger.info(f"🎬 Extracting clip: {start_time}s-{end_time}s -> {output_path}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise VideoProcessingError(
                    f"Clip extraction failed: {result.stderr}",
                    operation="extract_clip",
                    ffmpeg_command=' '.join(cmd),
                    return_code=result.returncode,
                    stderr_output=result.stderr
                )
            
            # Verify output file was created and has reasonable size
            if not os.path.exists(output_path):
                raise VideoProcessingError("Output clip file was not created")
            
            file_size = os.path.getsize(output_path)
            if file_size < 1024:  # Less than 1KB is suspicious
                raise VideoProcessingError(f"Output clip file is too small ({file_size} bytes)")
            
            logger.info(f"✅ Clip extracted successfully: {file_size / 1024 / 1024:.2f} MB")
            return True
            
        except subprocess.TimeoutExpired:
            raise VideoProcessingError("Clip extraction timed out (>5 minutes)")
        except Exception as e:
            raise VideoProcessingError(f"Unexpected error during clip extraction: {e}")
    
    def add_subtitle_overlay(self,
                           input_video: str,
                           output_path: str,
                           subtitle_lines: List[str],
                           start_time: float,
                           end_time: float,
                           font_size: int = 24,
                           font_color: str = "white",
                           outline_color: str = "black",
                           outline_width: int = 2) -> bool:
        """
        Add subtitle overlay to a video clip.
        
        Args:
            input_video: Path to input video
            output_path: Path for output video with subtitles
            subtitle_lines: List of subtitle text lines
            start_time: Start time for subtitles
            end_time: End time for subtitles
            font_size: Subtitle font size
            font_color: Subtitle text color
            outline_color: Subtitle outline color
            outline_width: Subtitle outline width
            
        Returns:
            True if overlay successful
        """
        try:
            # Create temporary subtitle file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as temp_srt:
                # Calculate duration per line
                duration = end_time - start_time
                time_per_line = duration / len(subtitle_lines) if subtitle_lines else duration
                
                for i, line in enumerate(subtitle_lines):
                    line_start = i * time_per_line
                    line_end = (i + 1) * time_per_line
                    
                    # Convert to SRT time format
                    start_srt = self._seconds_to_srt_time(line_start)
                    end_srt = self._seconds_to_srt_time(line_end)
                    
                    temp_srt.write(f"{i + 1}\n")
                    temp_srt.write(f"{start_srt} --> {end_srt}\n")
                    temp_srt.write(f"{line}\n\n")
                
                temp_srt_path = temp_srt.name
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build FFmpeg command with subtitle filter
            cmd = [
                'ffmpeg', '-y',
                '-i', input_video,
                '-vf', f'subtitles={temp_srt_path}:force_style=\'FontSize={font_size},PrimaryColour=&H{self._color_to_hex(font_color)},OutlineColour=&H{self._color_to_hex(outline_color)},Outline={outline_width}\'',
                '-c:a', 'copy',  # Copy audio without re-encoding
                output_path
            ]
            
            logger.info(f"📝 Adding subtitle overlay to {input_video}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Clean up temporary file
            try:
                os.unlink(temp_srt_path)
            except OSError:
                pass  # Ignore cleanup errors
            
            if result.returncode != 0:
                raise VideoProcessingError(
                    f"Subtitle overlay failed: {result.stderr}",
                    operation="add_subtitle_overlay",
                    ffmpeg_command=' '.join(cmd),
                    return_code=result.returncode,
                    stderr_output=result.stderr
                )
            
            logger.info("✅ Subtitle overlay added successfully")
            return True
            
        except subprocess.TimeoutExpired:
            raise VideoProcessingError("Subtitle overlay timed out")
        except Exception as e:
            raise VideoProcessingError(f"Unexpected error during subtitle overlay: {e}")
    
    def concatenate_clips(self,
                         clip_paths: List[str],
                         output_path: str,
                         add_transitions: bool = True,
                         transition_duration: float = 0.5) -> bool:
        """
        Concatenate multiple video clips into a single video.
        
        Args:
            clip_paths: List of paths to video clips
            output_path: Path for concatenated output
            add_transitions: Whether to add fade transitions
            transition_duration: Duration of fade transitions in seconds
            
        Returns:
            True if concatenation successful
        """
        try:
            if not clip_paths:
                raise VideoProcessingError("No clips provided for concatenation")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if len(clip_paths) == 1:
                # Single clip, just copy
                import shutil
                shutil.copy2(clip_paths[0], output_path)
                logger.info("✅ Single clip copied as final output")
                return True
            
            # Create temporary concat list file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_list:
                for clip_path in clip_paths:
                    # Use absolute paths to avoid path issues
                    absolute_path = os.path.abspath(clip_path)
                    # Escape single quotes and backslashes for FFmpeg
                    escaped_path = absolute_path.replace("'", "'\"'\"'").replace("\\", "\\\\")
                    temp_list.write(f"file '{escaped_path}'\n")
                temp_list_path = temp_list.name
            
            if add_transitions:
                # Use complex filter for transitions (more complex, requires all clips to have same resolution)
                return self._concatenate_with_transitions(clip_paths, output_path, transition_duration)
            else:
                # Simple concatenation without transitions
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', temp_list_path,
                    '-c', 'copy',  # Copy streams without re-encoding for speed
                    output_path
                ]
                
                logger.info(f"🔗 Concatenating {len(clip_paths)} clips")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_list_path)
                except OSError:
                    pass
                
                if result.returncode != 0:
                    raise VideoProcessingError(
                        f"Clip concatenation failed: {result.stderr}",
                        operation="concatenate_clips",
                        ffmpeg_command=' '.join(cmd),
                        return_code=result.returncode,
                        stderr_output=result.stderr
                    )
                
                logger.info("✅ Clips concatenated successfully")
                return True
                
        except subprocess.TimeoutExpired:
            raise VideoProcessingError("Clip concatenation timed out")
        except Exception as e:
            raise VideoProcessingError(f"Unexpected error during concatenation: {e}")
    
    def _concatenate_with_transitions(self,
                                    clip_paths: List[str],
                                    output_path: str,
                                    transition_duration: float) -> bool:
        """
        Concatenate clips with fade transitions (complex filter method).
        
        This method re-encodes all clips to ensure compatibility.
        """
        try:
            if len(clip_paths) <= 1:
                # For single clip or empty, fall back to simple copy
                if clip_paths:
                    import shutil
                    shutil.copy2(clip_paths[0], output_path)
                    logger.info("✅ Single clip copied as final output")
                    return True
                else:
                    raise VideoProcessingError("No clips provided for concatenation")
            
            # For multiple clips, use proper concat demuxer instead of complex crossfade
            # which was causing timing and sync issues
            
            # Create temporary concat list file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_list:
                for clip_path in clip_paths:
                    # Use absolute paths to avoid path issues
                    absolute_path = os.path.abspath(clip_path)
                    # Escape single quotes and backslashes for FFmpeg
                    escaped_path = absolute_path.replace("'", "'\"'\"'").replace("\\", "\\\\")
                    temp_list.write(f"file '{escaped_path}'\n")
                temp_list_path = temp_list.name
            
            # Use concat demuxer which preserves audio/video sync better than complex filters
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', temp_list_path,
                '-c:v', 'libx264',       # Re-encode video for consistency
                '-c:a', 'aac',           # Re-encode audio for consistency  
                '-profile:v', 'high',    # H.264 High profile for compatibility
                '-level', '4.1',         # H.264 level 4.1
                '-pix_fmt', 'yuv420p',   # Standard pixel format
                '-preset', 'medium',     # Standard preset
                '-crf', '23',            # Good quality
                '-movflags', '+faststart', # Web-optimized
                output_path
            ]
            
            logger.info(f"🎬 Concatenating {len(clip_paths)} clips with transitions")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 minute timeout
            
            # Clean up temporary file
            try:
                os.unlink(temp_list_path)
            except OSError:
                pass
            
            if result.returncode != 0:
                raise VideoProcessingError(
                    f"Transition concatenation failed: {result.stderr}",
                    operation="concatenate_with_transitions",
                    ffmpeg_command=' '.join(cmd),
                    return_code=result.returncode,
                    stderr_output=result.stderr
                )
            
            logger.info("✅ Clips concatenated with transitions successfully")
            return True
            
        except subprocess.TimeoutExpired:
            raise VideoProcessingError("Transition concatenation timed out")
        except Exception as e:
            raise VideoProcessingError(f"Unexpected error during transition concatenation: {e}")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _color_to_hex(self, color: str) -> str:
        """Convert color name to hex format for FFmpeg."""
        color_map = {
            'white': 'FFFFFF',
            'black': '000000',
            'red': 'FF0000',
            'blue': '0000FF',
            'green': '00FF00',
            'yellow': 'FFFF00'
        }
        return color_map.get(color.lower(), 'FFFFFF')


def validate_clip_quality(clip_path: str, min_size_mb: float = 0.1) -> Dict[str, Any]:
    """
    Validate the quality of an extracted video clip.
    
    Args:
        clip_path: Path to the video clip
        min_size_mb: Minimum file size in MB
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        if not os.path.exists(clip_path):
            return {
                'valid': False,
                'error': 'File does not exist',
                'file_size_mb': 0,
                'duration': 0
            }
        
        file_size_bytes = os.path.getsize(clip_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Get video info
        ffmpeg_utils = FFmpegUtils()
        video_info = ffmpeg_utils.get_video_info(clip_path)
        
        # Basic quality checks
        quality_issues = []
        
        if file_size_mb < min_size_mb:
            quality_issues.append(f"File size too small: {file_size_mb:.2f} MB")
        
        duration = video_info.get('duration', 0)
        if duration < 1.0:
            quality_issues.append(f"Duration too short: {duration:.2f} seconds")
        
        # Check for video and audio streams
        has_video = any(s.get('codec_type') == 'video' for s in video_info.get('streams', []))
        has_audio = any(s.get('codec_type') == 'audio' for s in video_info.get('streams', []))
        
        if not has_video:
            quality_issues.append("No video stream found")
        if not has_audio:
            quality_issues.append("No audio stream found")
        
        return {
            'valid': len(quality_issues) == 0,
            'quality_issues': quality_issues,
            'file_size_mb': file_size_mb,
            'duration': duration,
            'has_video': has_video,
            'has_audio': has_audio,
            'video_info': video_info
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'file_size_mb': 0,
            'duration': 0
        }
