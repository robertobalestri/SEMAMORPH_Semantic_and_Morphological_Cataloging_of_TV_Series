"""
Video processing utilities for recap generation.

This module handles FFmpeg operations for video clip extraction and assembly.
"""

import os
import subprocess
import logging
import sys
from typing import List, Dict, Any

# Add the src directory to Python path to enable proper imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from path_handler import PathHandler
from .models import Event, VideoClip

logger = logging.getLogger(__name__)


def extract_video_clips(events: List[Event], key_dialogue: Dict[str, List[str]], base_data_dir: str = "data") -> List[VideoClip]:
    """
    Extract video clips for selected events using FFmpeg.
    
    Args:
        events: List of selected events
        key_dialogue: Dictionary mapping event_id to dialogue lines
        base_data_dir: Base directory for episode data
        
    Returns:
        List of extracted video clips
    """
    clips = []
    
    for event in events:
        try:
            # Create PathHandler for this event's episode
            path_handler = PathHandler(event.series, event.season, event.episode, base_data_dir)
            
            # Find source video file
            video_path = path_handler.get_video_file_path()
            if not os.path.exists(video_path):
                logger.warning(f"No video file found for {event.series}{event.season}{event.episode}")
                continue
            
            # Get dialogue info, including new start/end times
            dialogue_info = key_dialogue.get(event.id)
            if not dialogue_info or not dialogue_info.get('lines'):
                logger.warning(f"No dialogue found for event {event.id}, skipping clip")
                continue

            # Calculate clip timestamps from dialogue
            start_seconds = _parse_timestamp_to_seconds(dialogue_info['start_time'])
            end_seconds = _parse_timestamp_to_seconds(dialogue_info['end_time'])
            duration = end_seconds - start_seconds
            
            # Create output directory using PathHandler
            output_dir = path_handler.get_recap_clip_dir()
            os.makedirs(output_dir, exist_ok=True)

            # Generate output filename using PathHandler helper (stable naming)
            clip_id = event.id  # Use full event ID
            clip_path = path_handler.get_individual_clip_path(clip_id)
            
            # Extract clip using FFmpeg
            success = _extract_clip_ffmpeg(video_path, clip_path, start_seconds, duration)
            
            if success:
                clips.append(VideoClip(
                    event_id=event.id,
                    file_path=clip_path,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    duration=duration,
                    subtitle_lines=dialogue_info['lines'],
                    arc_title=event.arc_title
                ))
                logger.info(f"Extracted clip: {event.arc_title} ({duration:.1f}s)")
            
        except Exception as e:
            logger.warning(f"Failed to extract clip for event {event.id}: {e}")
    
    logger.info(f"Successfully extracted {len(clips)} video clips")
    return clips


def assemble_final_recap(clips: List[VideoClip], output_dir: str, series: str, season: str, episode: str) -> str:
    """
    Assemble individual clips into final recap video.
    
    Args:
        clips: List of video clips to assemble
        output_dir: Directory to save final recap (will use PathHandler)
        series: Series identifier
        season: Season identifier
        episode: Episode identifier
        
    Returns:
        Path to final recap video
    """
    if not clips:
        raise ValueError("No clips provided for assembly")
    
    # Use PathHandler for proper path management
    path_handler = PathHandler(series, season, episode, "data")  # Use data as base_dir
    
    # Create output directory using PathHandler
    recap_dir = path_handler.get_recap_files_dir()
    os.makedirs(recap_dir, exist_ok=True)
    
    # Final recap filename using PathHandler
    recap_path = path_handler.get_final_recap_video_path()
    
    try:
        # Create file list for FFmpeg concatenation
        filelist_path = os.path.join(recap_dir, "clips_list.txt")
        with open(filelist_path, 'w') as f:
            for clip in clips:
                f.write(f"file '{os.path.abspath(clip.file_path)}'\n")
        
        # Concatenate clips using FFmpeg
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy',
            recap_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully assembled recap: {recap_path}")
            
            # Clean up temporary files
            # os.remove(filelist_path)
            
            return recap_path
        else:
            raise RuntimeError(f"FFmpeg concatenation failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to assemble final recap: {e}")
        raise


def _find_episode_video(series: str, season: str, episode: str, base_dir: str) -> str:
    """Find the video file for a given episode using PathHandler."""
    path_handler = PathHandler(series, season, episode, base_dir)
    video_path = path_handler.get_video_file_path()
    
    if os.path.exists(video_path):
        return video_path
    
    # Fallback: look for other video files in the episode directory
    episode_dir = f"{base_dir}/{series}/{season}/{episode}"
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    
    if os.path.exists(episode_dir):
        for file in os.listdir(episode_dir):
            for ext in video_extensions:
                if file.endswith(ext) and not file.startswith('recap_'):
                    return os.path.join(episode_dir, file)
    
    return None


def _extract_clip_ffmpeg(input_path: str, output_path: str, start_seconds: float, duration: float) -> bool:
    """Extract a video clip using FFmpeg."""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-ss', str(start_seconds),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-avoid_negative_ts', 'make_zero',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            logger.error(f"FFmpeg extraction failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"FFmpeg command failed: {e}")
        return False


def _parse_timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS,mmm timestamp to seconds."""
    try:
        if ',' in timestamp:
            time_part, ms_part = timestamp.split(',')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part)
            return h * 3600 + m * 60 + s + ms / 1000.0
        else:
            h, m, s = map(int, timestamp.split(':'))
            return h * 3600 + m * 60 + s
    except:
        return 0.0
