"""
New simplified scene boundary detection using LLM to identify subtitle boundaries.
This replaces the complex timestamp-based approach with a more reliable subtitle-number-based approach.
"""

import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..utils.logger_utils import setup_logging
from ..ai_models.ai_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from ..utils.llm_utils import clean_llm_json_response
from ..utils.subtitle_utils import SubtitleEntry

logger = setup_logging(__name__)

# Note: PlotScene is imported from the parent module to avoid circular imports

def format_subtitles_with_indices(subtitles: List[SubtitleEntry]) -> str:
    """Format subtitles with their indices for LLM processing."""
    formatted_lines = []
    for i, sub in enumerate(subtitles):
        # Clean text of formatting tags
        clean_text = sub.text.strip()
        formatted_lines.append(f"#{i+1}: [{sub.start_time} --> {sub.end_time}] {clean_text}")
    
    return '\n'.join(formatted_lines)

def find_scene_boundary(
    scene1,  # PlotScene
    scene2,  # PlotScene  
    subtitles: List[SubtitleEntry],
    llm: AzureChatOpenAI
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the subtitle boundary between two consecutive scenes.
    
    Args:
        scene1: The first scene
        scene2: The second scene
        subtitles: List of all available subtitle entries
        llm: LLM for boundary detection
        
    Returns:
        Tuple of (scene1_end_subtitle_index, scene2_start_subtitle_index) or (None, None) if failed
    """
    logger.info(f"🔍 Finding boundary between scene {scene1.scene_number} and scene {scene2.scene_number}")
    
    # Format subtitles with indices
    subtitle_text = format_subtitles_with_indices(subtitles)
    
    prompt = f"""You need to identify the exact subtitle boundary between two consecutive TV scenes.

**SCENE {scene1.scene_number} CONTENT:**
{scene1.plot_segment}

**SCENE {scene2.scene_number} CONTENT:**
{scene2.plot_segment}

**SUBTITLES WITH INDICES:**
{subtitle_text}

**TASK:**
Analyze the scene content and subtitles to determine:
1. The LAST subtitle that belongs to Scene {scene1.scene_number} (return its #number)
2. The FIRST subtitle that belongs to Scene {scene2.scene_number} (return its #number)

**REQUIREMENTS:**
- Scene {scene1.scene_number} ends with subtitle #X (the last subtitle of that scene)
- Scene {scene2.scene_number} starts with subtitle #Y (the first subtitle of that scene) 
- Y must be greater than X (scene 2 starts after scene 1 ends)
- Look for natural content transitions between the scenes
- The boundary should make narrative sense

**EXAMPLE:**
If Scene 1 is about "characters discussing plans" and Scene 2 is about "characters executing the plan", 
find where the discussion ends (e.g., subtitle #15) and where the action begins (e.g., subtitle #16).

Return ONLY a JSON object:
{{
    "scene_{scene1.scene_number}_end_subtitle": 15,
    "scene_{scene2.scene_number}_start_subtitle": 16,
    "reasoning": "Brief explanation of the boundary choice"
}}"""
    
    try:
        logger.info(f"🔄 Sending boundary detection request to LLM...")
        response = llm.invoke([HumanMessage(content=prompt)])
        
        boundary_data = clean_llm_json_response(response.content)
        logger.info(f"📋 LLM boundary response: {boundary_data}")
        
        if isinstance(boundary_data, list) and len(boundary_data) > 0:
            boundary_data = boundary_data[0]
        
        if not isinstance(boundary_data, dict):
            logger.error(f"Invalid response format: {boundary_data}")
            return None, None
        
        scene1_end_key = f"scene_{scene1.scene_number}_end_subtitle"
        scene2_start_key = f"scene_{scene2.scene_number}_start_subtitle"
        
        scene1_end_subtitle = boundary_data.get(scene1_end_key)
        scene2_start_subtitle = boundary_data.get(scene2_start_key)
        reasoning = boundary_data.get("reasoning", "No reasoning provided")
        
        if scene1_end_subtitle is None or scene2_start_subtitle is None:
            logger.error(f"Missing subtitle indices in response: {boundary_data}")
            return None, None
        
        # Convert to 0-based indexing
        scene1_end_index = int(scene1_end_subtitle) - 1
        scene2_start_index = int(scene2_start_subtitle) - 1
        
        # Validate indices
        if not (0 <= scene1_end_index < len(subtitles)):
            logger.error(f"Invalid scene1 end index: {scene1_end_index} (max: {len(subtitles)-1})")
            return None, None
            
        if not (0 <= scene2_start_index < len(subtitles)):
            logger.error(f"Invalid scene2 start index: {scene2_start_index} (max: {len(subtitles)-1})")
            return None, None
        
        # Validate logical order
        if scene1_end_index >= scene2_start_index:
            logger.error(f"Invalid boundary: scene1 ends at #{scene1_end_index+1} but scene2 starts at #{scene2_start_index+1}")
            return None, None
        
        logger.info(f"✅ Boundary found: Scene {scene1.scene_number} ends at subtitle #{scene1_end_index+1}, Scene {scene2.scene_number} starts at subtitle #{scene2_start_index+1}")
        logger.info(f"📝 Reasoning: {reasoning}")
        
        return scene1_end_index, scene2_start_index
        
    except Exception as e:
        logger.error(f"❌ Error in boundary detection: {e}")
        return None, None

def assign_timestamps_from_subtitle_indices(
    scene,  # PlotScene
    start_subtitle_index: Optional[int],
    end_subtitle_index: Optional[int],
    subtitles: List[SubtitleEntry]
):  # Returns PlotScene
    """
    Assign timestamps to a scene based on subtitle indices.
    
    Args:
        scene: The scene to update
        start_subtitle_index: Index of the first subtitle in the scene
        end_subtitle_index: Index of the last subtitle in the scene
        subtitles: List of subtitle entries
        
    Returns:
        Updated scene with timestamps
    """
    if start_subtitle_index is not None and end_subtitle_index is not None:
        if 0 <= start_subtitle_index < len(subtitles) and 0 <= end_subtitle_index < len(subtitles):
            start_subtitle = subtitles[start_subtitle_index]
            end_subtitle = subtitles[end_subtitle_index]
            
            scene.start_time = start_subtitle.start_time
            scene.end_time = end_subtitle.end_time
            scene.start_seconds = start_subtitle.start_seconds
            scene.end_seconds = end_subtitle.end_seconds
            scene.start_subtitle_index = start_subtitle_index
            scene.end_subtitle_index = end_subtitle_index
            
            logger.info(f"✅ Scene {scene.scene_number} timestamps: {scene.start_time} --> {scene.end_time}")
        else:
            logger.error(f"❌ Invalid subtitle indices for scene {scene.scene_number}: start={start_subtitle_index}, end={end_subtitle_index}")
    else:
        logger.warning(f"⚠️ No subtitle indices provided for scene {scene.scene_number}")
    
    return scene

def map_scenes_to_timestamps_simple(
    scenes: List,  # List[PlotScene]
    subtitles: List[SubtitleEntry],
    llm: AzureChatOpenAI
) -> List:  # List[PlotScene]
    """
    Map scenes to timestamps using the simplified boundary detection approach.
    
    Args:
        scenes: List of scenes to map
        subtitles: List of subtitle entries
        llm: LLM for boundary detection
        
    Returns:
        List of scenes with timestamps
    """
    logger.info(f"🗺️ Starting simplified scene timestamp mapping for {len(scenes)} scenes")
    
    if not scenes:
        return []
    
    if len(scenes) == 1:
        # For single scene, use all subtitles
        scene = scenes[0]
        scene = assign_timestamps_from_subtitle_indices(scene, 0, len(subtitles) - 1, subtitles)
        return [scene]
    
    mapped_scenes = []
    processed_subtitle_count = 0
    
    for i in range(len(scenes)):
        scene = scenes[i]
        
        if i == 0:
            # First scene: find boundary with second scene
            if len(scenes) > 1:
                # Use all subtitles for boundary detection
                available_subtitles = subtitles
                
                end_index, next_start_index = find_scene_boundary(
                    scene, scenes[i + 1], available_subtitles, llm
                )
                
                if end_index is not None:
                    # First scene starts at subtitle 0, ends at found boundary
                    scene = assign_timestamps_from_subtitle_indices(scene, 0, end_index, subtitles)
                    processed_subtitle_count = end_index + 1
                else:
                    # Fallback: assign reasonable portion
                    fallback_end = min(len(subtitles) // len(scenes), len(subtitles) - 1)
                    scene = assign_timestamps_from_subtitle_indices(scene, 0, fallback_end, subtitles)
                    processed_subtitle_count = fallback_end + 1
            else:
                # Only one scene, use all subtitles
                scene = assign_timestamps_from_subtitle_indices(scene, 0, len(subtitles) - 1, subtitles)
                
        elif i == len(scenes) - 1:
            # Last scene: use remaining subtitles
            scene = assign_timestamps_from_subtitle_indices(
                scene, 
                processed_subtitle_count, 
                len(subtitles) - 1, 
                subtitles
            )
            
        else:
            # Middle scene: find boundary with next scene
            # Use only unprocessed subtitles for LLM analysis
            available_subtitles = subtitles[processed_subtitle_count:]
            
            if available_subtitles:
                # Find boundary within available subtitles
                end_index_relative, next_start_index_relative = find_scene_boundary(
                    scene, scenes[i + 1], available_subtitles, llm
                )
                
                if end_index_relative is not None:
                    # Convert relative index to absolute index
                    end_index_absolute = processed_subtitle_count + end_index_relative
                    
                    scene = assign_timestamps_from_subtitle_indices(
                        scene, 
                        processed_subtitle_count, 
                        end_index_absolute, 
                        subtitles
                    )
                    processed_subtitle_count = end_index_absolute + 1
                else:
                    # Fallback: assign reasonable portion of remaining subtitles
                    remaining_scenes = len(scenes) - i
                    remaining_subtitles = len(subtitles) - processed_subtitle_count
                    scene_portion = remaining_subtitles // remaining_scenes
                    
                    fallback_end = processed_subtitle_count + max(1, scene_portion) - 1
                    fallback_end = min(fallback_end, len(subtitles) - 1)
                    
                    scene = assign_timestamps_from_subtitle_indices(
                        scene, 
                        processed_subtitle_count, 
                        fallback_end, 
                        subtitles
                    )
                    processed_subtitle_count = fallback_end + 1
            else:
                logger.warning(f"⚠️ No subtitles available for scene {scene.scene_number}")
        
        mapped_scenes.append(scene)
    
    # Log final mapping summary
    logger.info("✅ Scene mapping completed:")
    for scene in mapped_scenes:
        if scene.start_time and scene.end_time:
            logger.info(f"   Scene {scene.scene_number}: {scene.start_time} --> {scene.end_time}")
        else:
            logger.warning(f"   Scene {scene.scene_number}: No timestamps assigned")
    
    return mapped_scenes
