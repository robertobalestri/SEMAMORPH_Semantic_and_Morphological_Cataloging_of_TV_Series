"""
Utilities for handling recap content filtering throughout the processing pipeline.
"""

import json
import os
from typing import List, Dict, Any
from ..utils.logger_utils import setup_logging

logger = setup_logging(__name__)

def filter_non_recap_content(plot_content: str) -> str:
    """
    Filter out recap scenes from plot content text.
    
    This function is used to ensure that recap content doesn't
    contaminate narrative arc extraction and other analysis steps.
    
    Args:
        plot_content: Raw plot content text
        
    Returns:
        Plot content with recap scenes removed
    """
    # This function works with the _plot.txt files which already have
    # recap content filtered out by save_plot_files()
    # So we can return the content as-is for now
    return plot_content

def load_plot_with_recap_filtering(plot_file_path: str) -> str:
    """
    Load plot content that has already been filtered of recap scenes.
    
    Args:
        plot_file_path: Path to the _plot.txt file
        
    Returns:
        Plot content without recap scenes
    """
    if not os.path.exists(plot_file_path):
        logger.warning(f"Plot file not found: {plot_file_path}")
        return ""
    
    with open(plot_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    logger.info(f"Loaded filtered plot content: {len(content)} characters (recap scenes already excluded)")
    return content

def get_non_recap_scenes_from_json(scenes_json_path: str) -> List[Dict[str, Any]]:
    """
    Load scene data and filter out recap scenes.
    
    Args:
        scenes_json_path: Path to the scenes JSON file
        
    Returns:
        List of non-recap scenes
    """
    if not os.path.exists(scenes_json_path):
        logger.warning(f"Scenes JSON file not found: {scenes_json_path}")
        return []
    
    with open(scenes_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    non_recap_scenes = [scene for scene in scenes if not scene.get("is_recap", False)]
    recap_scenes = [scene for scene in scenes if scene.get("is_recap", False)]
    
    logger.info(f"Filtered scenes: {len(non_recap_scenes)} non-recap, {len(recap_scenes)} recap")
    
    return non_recap_scenes

def get_recap_scene_count(scenes_json_path: str) -> int:
    """
    Get count of recap scenes in an episode.
    
    Args:
        scenes_json_path: Path to the scenes JSON file
        
    Returns:
        Number of recap scenes
    """
    if not os.path.exists(scenes_json_path):
        return 0
    
    try:
        with open(scenes_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenes = data.get("scenes", [])
        recap_count = sum(1 for scene in scenes if scene.get("is_recap", False))
        
        return recap_count
    except Exception as e:
        logger.error(f"Error counting recap scenes: {e}")
        return 0

def create_non_recap_plot_content(scenes_json_path: str) -> str:
    """
    Create plot content string from non-recap scenes only.
    
    This is useful for steps that need plot content but want to
    ensure recap scenes are excluded.
    
    Args:
        scenes_json_path: Path to the scenes JSON file
        
    Returns:
        Plot content string with only non-recap scenes
    """
    non_recap_scenes = get_non_recap_scenes_from_json(scenes_json_path)
    
    plot_lines = []
    for scene in non_recap_scenes:
        scene_number = scene.get("scene_number", "Unknown")
        plot_segment = scene.get("plot_segment", "")
        plot_lines.append(f"Scene {scene_number}: {plot_segment}")
    
    content = "\n\n".join(plot_lines)
    logger.info(f"Created non-recap plot content: {len(content)} characters from {len(non_recap_scenes)} scenes")
    
    return content
