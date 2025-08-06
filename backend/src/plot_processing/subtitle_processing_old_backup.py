"""
Subtitle processing utilities for SEMAMORPH.
Handles SRT file parsing and subtitle-to-plot conversion.
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from ..utils.logger_utils import setup_logging
from ..ai_models.ai_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from ..utils.llm_utils import clean_llm_json_response
from ..utils.subtitle_utils import parse_srt_time_to_seconds, SubtitleEntry

logger = setup_logging(__name__)

@dataclass
class PlotScene:
    """Represents a plot scene with content and timing."""
    scene_number: int
    plot_segment: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    start_subtitle_index: Optional[int] = None  # New: index of first subtitle in scene
    end_subtitle_index: Optional[int] = None    # New: index of last subtitle in scene

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def format_subtitles_for_llm(subtitles: List[SubtitleEntry]) -> str:
    """Format subtitles for LLM processing."""
    formatted_lines = []
    for sub in subtitles:
        # Clean text of formatting tags
        clean_text = re.sub(r'<[^>]+>', '', sub.text)
        clean_text = re.sub(r'\{[^}]+\}', '', clean_text)
        clean_text = clean_text.strip()
        
        if clean_text:
            formatted_lines.append(f"[{sub.start_time} --> {sub.end_time}] {clean_text}")
    
    return '\n'.join(formatted_lines)

def generate_plot_from_subtitles(subtitles: List[SubtitleEntry], llm: AzureChatOpenAI, previous_season_summary: Optional[str] = None) -> Dict:
    """Generate detailed plot from subtitles using LLM with optional season context."""
    logger.info("Generating detailed plot from subtitles")
    
    # Format subtitles for LLM
    subtitle_text = format_subtitles_for_llm(subtitles)
    
    # Base prompt for plot generation
    base_prompt = """You are a script analyst that understands character names, relationships, and storylines from subtitles. You are also trained in ambiguity handling and know how to recognize when attribution is uncertain. 

Your task is to convert the provided subtitles into a detailed, scene-based plot summary in JSON. Be conservative in speaker attribution — never rely on stereotypes, character tropes, or assumed patterns.

**CRITICAL SPEAKER IDENTIFICATION RULES:**
- Only use specific names when the speaker is clear from the dialogue or the context and you feel confident.
- If the speaker is not clear in the text, use generic references like "a character", "someone", or "a voice".
- Do not assign actions based on behavior typical of a character.

**Bias Avoidance Principles:**
- Do not infer character behavior based on prior knowledge, stereotypes, or training corpus frequency.
- Do not assume that humiliating, comic, or dramatic actions belong to characters who often fill those roles unless **clearly** stated in the text.
- When multiple characters are present and the speaker isn’t identified, use:  
  - "one of the group of footballers"  
  - "a male character"  
  - "an unidentified person"  
  as needed.

**If unsure, you MUST default to generic labeling** — uncertainty is better than incorrect confidence.
**Output Format (JSON only):**
[
  {{
    "scene_number": 1,
    "plot_segment": "..."
  }},
  {{
    "scene_number": 2,
    "plot_segment": "..."
  }}
]

**Plot Segment Requirements:**
- Extract every narrative event from the subtitles in chronological order
- Use clear, concise sentences (no run-on sentences)
- Include all dialogue, actions, and story developments
- Maintain objective tone - report what happens, don't interpret
- Number scenes based on natural story breaks or setting changes
- Correctly separate scenes, trying to not make big large sequences of scenes but rather smaller and focused scenes.
- Make the plot segment detailed and specific, indicating also the character present in the scene.

REMEMBER ALWAYS: Don't be overly confident, if you are not certain about the speaker, use generic references. 

"""



    # Add season context if available
    if previous_season_summary:
        context_prompt = f"""
                                **Previous Season Context:**
                                Use the following summary of previous episodes to better understand character relationships, ongoing storylines, and narrative context. This context helps you interpret the subtitles more accurately but should NOT be included in your plot output - only use it for understanding.

                                {previous_season_summary}

                                **Important:** Your plot output should ONLY describe what happens in the current episode subtitles. The context is provided to help you better understand character names, relationships, and ongoing storylines when interpreting the subtitles.
                                """
        
        full_prompt = f"{base_prompt}\n{context_prompt}"
        
        logger.info("Using previous season context for plot generation")
    else:
        full_prompt = base_prompt
    
    full_prompt += f"\n\nSubtitles:\n{subtitle_text}"
    
    logger.info(f"Sending plot generation request to LLM (subtitle length: {len(subtitle_text)} chars)")
    logger.info(f"Prompt total length: {len(full_prompt)} chars")
    logger.info(f"Using LLM type: {type(llm).__name__}")
    logger.debug(f"Full prompt preview (first 500 chars): {full_prompt[:500]}...")
    logger.debug(f"LLM configuration: {getattr(llm, 'model_name', 'Unknown')} | Temperature: {getattr(llm, 'temperature', 'Unknown')}")
    
    try:
        logger.info("🔄 Invoking LLM for plot generation...")
        
        # Add timing information
        import time
        start_time = time.time()
        
        response = llm.invoke([HumanMessage(content=full_prompt)])
        
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"✅ LLM response received successfully in {elapsed:.2f} seconds")
        logger.debug(f"Response preview (first 200 chars): {response.content[:200]}...")
    except Exception as e:
        logger.error(f"❌ LLM invocation failed: {e}")
        logger.error(f"LLM type: {type(llm)}")
        logger.error(f"Prompt length: {len(full_prompt)}")
        raise
    
    try:
        # Clean and parse JSON response
        plot_data = clean_llm_json_response(response.content)
        
        if isinstance(plot_data, list):
            # Convert list format to scenes dict
            scenes_dict = {"scenes": plot_data}
        elif isinstance(plot_data, dict) and "scenes" in plot_data:
            scenes_dict = plot_data
        else:
            # Assume the dict itself contains scenes
            scenes_dict = {"scenes": list(plot_data.values()) if isinstance(plot_data, dict) else plot_data}
        
        logger.info(f"Generated plot with {len(scenes_dict.get('scenes', []))} scenes")
        return scenes_dict
        
    except Exception as e:
        logger.error(f"Error parsing LLM response for plot generation: {e}")
        logger.error(f"Raw response: {response.content[:500]}...")
        raise

def cascade_timeline_corrections(
    affected_scenes: List[PlotScene],
    anchor_scene: PlotScene,
    current_scene: PlotScene,
    subtitles: List[SubtitleEntry]
) -> None:
    """
    Cascade timeline corrections for intermediate scenes between anchor and current scene.
    
    Args:
        affected_scenes: List of scenes between anchor and current that need correction
        anchor_scene: The scene that was just corrected (new end boundary)
        current_scene: The scene being processed (new start boundary)
        subtitles: List of subtitle entries for finding valid timestamps
    """
    if not affected_scenes:
        return
        
    logger.info(f"🔄 Cascading timeline corrections for {len(affected_scenes)} intermediate scenes")
    
    # Sort scenes by scene number to process sequentially
    affected_scenes.sort(key=lambda s: s.scene_number)
    
    # Start from the anchor scene's new end time
    current_end_time = anchor_scene.end_time
    current_end_seconds = anchor_scene.end_seconds
    
    for i, scene in enumerate(affected_scenes):
        logger.info(f"🔧 Correcting timeline for scene {scene.scene_number}")
        
        # Find the next valid start time after the previous scene's end
        next_start_time = find_next_available_start_time(current_end_seconds, subtitles)
        if next_start_time:
            scene.start_time = next_start_time
            scene.start_seconds = parse_srt_time_to_seconds(next_start_time)
            logger.info(f"   Updated start: {next_start_time}")
        
        # For the end time, either use the original or adjust to fit before next scene
        if i == len(affected_scenes) - 1:
            # Last intermediate scene - must end before current scene starts
            max_end_seconds = current_scene.start_seconds - 0.001  # Tiny gap to prevent overlap
            
            # Find a suitable end time that doesn't exceed the limit
            suitable_end_sub = None
            for sub in reversed(subtitles):
                if (sub.end_seconds <= max_end_seconds and 
                    sub.end_seconds > scene.start_seconds):
                    suitable_end_sub = sub
                    break
            
            if suitable_end_sub:
                scene.end_time = suitable_end_sub.end_time
                scene.end_seconds = suitable_end_sub.end_seconds
                logger.info(f"   Updated end: {scene.end_time}")
            else:
                logger.warning(f"⚠️ Could not find suitable end time for scene {scene.scene_number}")
        else:
            # Keep original end time or adjust if needed for next intermediate scene
            # This will be handled in the next iteration
            pass
        
        # Update current_end for next iteration
        if scene.end_time and scene.end_seconds:
            current_end_time = scene.end_time
            current_end_seconds = scene.end_seconds


def disambiguate_overlapping_scenes(
    scene1: PlotScene,
    scene2: PlotScene,
    subtitles: List[SubtitleEntry],
    llm: AzureChatOpenAI
) -> Tuple[Optional[str], Optional[str]]:
    """
    Use LLM to disambiguate overlapping scenes and find exact boundary timestamps.
    
    Args:
        scene1: The first scene (already mapped)
        scene2: The second scene (conflicting)
        subtitles: List of subtitle entries
        llm: LLM for disambiguation
        
    Returns:
        Tuple of (scene1_end_time, scene2_start_time) or (None, None) if failed
    """
    logger.info(f"🔍 Disambiguating overlap between scenes {scene1.scene_number} and {scene2.scene_number}")
    
    # Get subtitles in the overlapping region and some context around it
    overlap_start = min(scene1.start_seconds or 0, scene2.start_seconds or 0) if scene1.start_seconds and scene2.start_seconds else 0
    overlap_end = max(scene1.end_seconds or 9999, scene2.end_seconds or 9999) if scene1.end_seconds and scene2.end_seconds else 9999
    
    # Include some context before and after the overlap
    context_start = overlap_start - 30  # 30 seconds before
    context_end = overlap_end + 30      # 30 seconds after
    
    relevant_subtitles = [
        sub for sub in subtitles 
        if context_start <= sub.start_seconds <= context_end
    ]
    
    # Format subtitles for LLM
    subtitle_context = "\n".join([
        f"[{sub.start_time} --> {sub.end_time}] {sub.text}" 
        for sub in relevant_subtitles
    ])
    
    prompt = f"""You need to find the exact boundary between two overlapping TV scenes by analyzing their content and the subtitles.

**SCENE OVERLAP SITUATION:**
- Scene {scene1.scene_number} currently mapped to: {scene1.start_time} --> {scene1.end_time}
- Scene {scene2.scene_number} trying to map to overlapping timestamps
- You need to find where Scene {scene1.scene_number} should END and Scene {scene2.scene_number} should START

**SCENE {scene1.scene_number} CONTENT:**
{scene1.plot_segment}

**SCENE {scene2.scene_number} CONTENT:**
{scene2.plot_segment}

**SUBTITLES IN THE OVERLAPPING REGION:**
{subtitle_context}

**TASK:**
Analyze the scene content and subtitles to determine:
1. Where Scene {scene1.scene_number} naturally ends (find the EXACT end timestamp from subtitles)
2. Where Scene {scene2.scene_number} naturally begins (find the EXACT start timestamp from subtitles)

**REQUIREMENTS:**
- Scene {scene1.scene_number} end_time must be the END timestamp of a subtitle (e.g., "00:15:27,890")
- Scene {scene2.scene_number} start_time must be the START timestamp of the NEXT subtitle (e.g., "00:15:28,000")
- The timestamps must appear EXACTLY in the subtitle list above
- Scene {scene1.scene_number} must end BEFORE Scene {scene2.scene_number} starts (NO overlap or shared timestamps)
- Look for natural content transitions between the scenes
- Ensure there's a clear temporal separation between the scenes

**EXAMPLE LOGIC:**
- If Scene {scene1.scene_number} is about "Christina and Meredith discuss Katie's pageant" 
- And Scene {scene2.scene_number} is about "Katie performs her routine"
- Find where discussion ends (e.g., subtitle ending at 00:15:27,890)
- Next scene starts with next subtitle (e.g., subtitle starting at 00:15:28,000)
- This creates proper temporal separation with no overlap

**CRITICAL:** Scene {scene1.scene_number}_end_time must be < Scene {scene2.scene_number}_start_time

**EXAMPLE LOGIC:**
- If Scene {scene1.scene_number} is about "Christina and Meredith discuss Katie's pageant" 
- And Scene {scene2.scene_number} is about "Katie performs her routine"
- Find the subtitle where the discussion ends and the performance begins

Return ONLY a JSON object:
{{
    "scene_{scene1.scene_number}_end_time": "HH:MM:SS,mmm",
    "scene_{scene2.scene_number}_start_time": "HH:MM:SS,mmm",
    "boundary_reasoning": "Explanation of where you found the natural transition"
}}

Use EXACT timestamps from the subtitle list above."""
    
    try:
        logger.info(f"🔄 Sending overlap disambiguation request to LLM...")
        response = llm.invoke([HumanMessage(content=prompt)])
        
        boundary_data = clean_llm_json_response(response.content)
        logger.info(f"📋 LLM disambiguation response: {boundary_data}")
        
        if isinstance(boundary_data, list) and len(boundary_data) > 0:
            boundary_data = boundary_data[0]
        
        if not isinstance(boundary_data, dict):
            logger.error(f"Invalid disambiguation response format: {boundary_data}")
            return None, None
        
        scene1_end_key = f"scene_{scene1.scene_number}_end_time"
        scene2_start_key = f"scene_{scene2.scene_number}_start_time"
        
        scene1_end_time = boundary_data.get(scene1_end_key)
        scene2_start_time = boundary_data.get(scene2_start_key)
        reasoning = boundary_data.get("boundary_reasoning", "No reasoning provided")
        
        if not scene1_end_time or not scene2_start_time:
            logger.error(f"Missing timestamps in disambiguation response: {boundary_data}")
            return None, None
        
        # Validate timestamps exist in subtitles
        valid_end_times = {sub.end_time for sub in subtitles}
        valid_start_times = {sub.start_time for sub in subtitles}
        
        if scene1_end_time not in valid_end_times:
            logger.warning(f"⚠️ Scene {scene1.scene_number} end time not found in subtitles: {scene1_end_time}")
            # Find closest valid end time
            target_seconds = parse_srt_time_to_seconds(scene1_end_time)
            closest_end_time, _ = find_closest_scene_end_timestamp(target_seconds, subtitles)
            logger.info(f"🔧 Corrected to closest valid end time: {closest_end_time}")
            scene1_end_time = closest_end_time
        
        if scene2_start_time not in valid_start_times:
            logger.warning(f"⚠️ Scene {scene2.scene_number} start time not found in subtitles: {scene2_start_time}")
            # Find closest valid start time
            target_seconds = parse_srt_time_to_seconds(scene2_start_time)
            closest_start_time, _ = find_closest_scene_start_timestamp(target_seconds, subtitles)
            logger.info(f"🔧 Corrected to closest valid start time: {closest_start_time}")
            scene2_start_time = closest_start_time
        
        # Validate logical order (scene1 ends BEFORE scene2 starts - no overlap or shared timestamps)
        scene1_end_seconds = parse_srt_time_to_seconds(scene1_end_time)
        scene2_start_seconds = parse_srt_time_to_seconds(scene2_start_time)
        
        if scene1_end_seconds >= scene2_start_seconds:
            logger.error(f"❌ Invalid boundary: Scene {scene1.scene_number} ends at or after Scene {scene2.scene_number} starts")
            logger.error(f"   Scene {scene1.scene_number} end: {scene1_end_time} ({scene1_end_seconds:.3f}s)")
            logger.error(f"   Scene {scene2.scene_number} start: {scene2_start_time} ({scene2_start_seconds:.3f}s)")
            logger.error(f"   Scenes must have strict temporal separation (end < start)")
            return None, None
        
        logger.info(f"✅ Overlap disambiguation successful:")
        logger.info(f"   Scene {scene1.scene_number} will end at: {scene1_end_time}")
        logger.info(f"   Scene {scene2.scene_number} will start at: {scene2_start_time}")
        logger.info(f"   Reasoning: {reasoning}")
        
        return scene1_end_time, scene2_start_time
        
    except Exception as e:
        logger.error(f"❌ Error during overlap disambiguation: {e}")
        return None, None

def map_scene_to_timestamps(scene: PlotScene, subtitles: List[SubtitleEntry], llm: AzureChatOpenAI, previously_mapped_scenes: List[PlotScene] = None) -> PlotScene:
    """Map a plot scene to subtitle timestamps using LLM with validation and context of previously mapped scenes."""
    logger.info(f"Mapping scene {scene.scene_number} to timestamps")
    
    # Format subtitles with timestamps
    subtitle_text = format_subtitles_for_llm(subtitles)
    
    # Create sets of valid start and end times for validation
    valid_start_times = {sub.start_time for sub in subtitles}
    valid_end_times = {sub.end_time for sub in subtitles}
    valid_start_seconds = {sub.start_seconds for sub in subtitles}
    valid_end_seconds = {sub.end_seconds for sub in subtitles}
    
    # Build context of previously mapped scenes to avoid conflicts
    previously_mapped_context = ""
    available_range_hint = ""
    if previously_mapped_scenes:
        mapped_timestamps = []
        last_end_seconds = 0
        
        for prev_scene in previously_mapped_scenes:
            if prev_scene.start_time and prev_scene.end_time:
                mapped_timestamps.append(f"Scene {prev_scene.scene_number}: {prev_scene.start_time} --> {prev_scene.end_time}")
                if prev_scene.end_seconds and prev_scene.end_seconds > last_end_seconds:
                    last_end_seconds = prev_scene.end_seconds
        
        if mapped_timestamps:
            # Find available timestamp range for guidance
            available_subtitles = [
                sub for sub in subtitles 
                if sub.start_seconds > last_end_seconds
            ]
            
            if available_subtitles:
                earliest_available = available_subtitles[0].start_time
                latest_available = available_subtitles[-1].end_time
                next_start_after_conflict, _ = find_next_available_start_time(last_end_seconds, subtitles)
                
                available_range_hint = f"""
AVAILABLE TIMESTAMP RANGE FOR THIS SCENE:
- Previous scene ended at: {seconds_to_srt_time(last_end_seconds)}
- Next available start time: {next_start_after_conflict or earliest_available}
- Range available: {earliest_available} to {latest_available}

⚠️ CRITICAL: Your scene must start AFTER {seconds_to_srt_time(last_end_seconds)} to avoid conflicts!
Use the START timestamp of a subtitle that begins after the previous scene's end.
"""
            
            previously_mapped_context = f"""
ALREADY MAPPED SCENES (DO NOT CONFLICT WITH THESE):
{chr(10).join(mapped_timestamps)}

⚠️ CRITICAL: You MUST choose timestamps that do NOT overlap with the scenes already mapped above!
{available_range_hint}
"""
    
    prompt = f"""Given the following subtitles with timestamps and the plot scene description, identify the EXACT start and end timestamps for this scene.

{previously_mapped_context}

Plot Scene:
{scene.plot_segment}

Subtitles:
{subtitle_text}

CRITICAL REQUIREMENTS: 
- For start_time: Use the EXACT start timestamp of the FIRST subtitle in the scene (e.g., "00:15:23,456")
- For end_time: Use the EXACT end timestamp of the LAST subtitle in the scene (e.g., "00:15:27,890")  
- You MUST use timestamps that appear EXACTLY in the subtitle list above
- Do NOT modify or approximate the timestamps - copy them exactly as shown
- TEMPORAL SEPARATION: If previous scenes exist, this scene must start AFTER they end (no overlap)
{f"- AVOID conflicts with already mapped scenes listed above" if previously_mapped_context else ""}

**SCENE BOUNDARY RULES:**
- Each scene covers a continuous time period
- Scenes must NOT overlap in time  
- Scene A ends at subtitle X's END time, Scene B starts at subtitle Y's START time (where Y > X)
- There should be clear temporal separation between consecutive scenes

Return only a JSON object with:
{{
    "start_time": "HH:MM:SS,mmm",
    "end_time": "HH:MM:SS,mmm"
}}

Use the exact timestamp format from the subtitles."""
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        logger.info(f"Sending timestamp mapping request to LLM for scene {scene.scene_number} (attempt {retry_count + 1}/{max_retries})")
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"Received LLM response for scene {scene.scene_number}: {response.content[:200]}...")
            
            timing_data = clean_llm_json_response(response.content)
            logger.info(f"Parsed timing data for scene {scene.scene_number}: {timing_data}")
            
            # Handle case where clean_llm_json_response returns a list
            if isinstance(timing_data, list) and len(timing_data) > 0:
                timing_data = timing_data[0]
            
            # Ensure timing_data is a dictionary
            if not isinstance(timing_data, dict):
                logger.error(f"Expected dict but got {type(timing_data)}: {timing_data}")
                retry_count += 1
                continue
            
            start_time = timing_data.get("start_time")
            end_time = timing_data.get("end_time")
            
            if not start_time or not end_time:
                logger.warning(f"Missing start_time or end_time in response: {timing_data}")
                retry_count += 1
                continue
            
            # VALIDATION: Find closest valid timestamps if exact match fails
            start_seconds = parse_srt_time_to_seconds(start_time)
            end_seconds = parse_srt_time_to_seconds(end_time)
            
            # Check for conflicts with previously mapped scenes
            if previously_mapped_scenes:
                conflicting_scenes = []
                
                # Find ALL scenes that overlap with proposed timestamps
                for prev_scene in previously_mapped_scenes:
                    if prev_scene.start_seconds and prev_scene.end_seconds:
                        # Check if proposed timestamps overlap with existing scene
                        if (start_seconds < prev_scene.end_seconds and end_seconds > prev_scene.start_seconds):
                            logger.warning(f"⚠️ Timestamp conflict detected with scene {prev_scene.scene_number}")
                            conflicting_scenes.append(prev_scene)
                
                if conflicting_scenes:
                    # Sort conflicting scenes by scene number to handle them sequentially
                    conflicting_scenes.sort(key=lambda s: s.scene_number)
                    
                    # Find the LAST conflicting scene (chronologically) to disambiguate with
                    last_conflicting_scene = max(conflicting_scenes, key=lambda s: s.scene_number)
                    
                    logger.info(f"🔧 Found {len(conflicting_scenes)} conflicting scenes: {[s.scene_number for s in conflicting_scenes]}")
                    logger.info(f"🔧 Disambiguating with the last chronological conflict: Scene {last_conflicting_scene.scene_number}")
                    
                    # Check if there are intermediate scenes that need cascading updates
                    intermediate_scenes = [
                        s for s in conflicting_scenes 
                        if s.scene_number > last_conflicting_scene.scene_number and s.scene_number < scene.scene_number
                    ]
                    
                    if intermediate_scenes:
                        logger.warning(f"⚠️ Intermediate scenes detected: {[s.scene_number for s in intermediate_scenes]}")
                        logger.warning(f"⚠️ These scenes may need timeline correction after disambiguation")
                
                    conflicting_scene = last_conflicting_scene
                    logger.info(f"🔧 Attempting overlap disambiguation between scenes {conflicting_scene.scene_number} and {scene.scene_number}")
                    
                    # Use LLM to disambiguate the overlap
                    disambiguated_end, disambiguated_start = disambiguate_overlapping_scenes(
                        conflicting_scene, scene, subtitles, llm
                    )
                    
                    if disambiguated_end and disambiguated_start:
                        # Update the conflicting scene's end time
                        conflicting_scene.end_time = disambiguated_end
                        conflicting_scene.end_seconds = parse_srt_time_to_seconds(disambiguated_end)
                        
                        # Use the disambiguated start time for current scene
                        start_time = disambiguated_start
                        start_seconds = parse_srt_time_to_seconds(disambiguated_start)
                        
                        logger.info(f"✅ Overlap resolved:")
                        logger.info(f"   Scene {conflicting_scene.scene_number} now ends at: {disambiguated_end}")
                        logger.info(f"   Scene {scene.scene_number} will start at: {disambiguated_start}")
                        
                        # Apply cascading corrections to intermediate scenes if any exist
                        if intermediate_scenes:
                            logger.info(f"🔄 Applying cascading corrections to intermediate scenes: {[s.scene_number for s in intermediate_scenes]}")
                            cascade_timeline_corrections(
                                intermediate_scenes, 
                                conflicting_scene, 
                                scene, 
                                subtitles
                            )
                            logger.info(f"✅ Cascading corrections completed")
                        
                        # Continue with the disambiguated timestamps
                    else:
                        # Disambiguation failed, try next iteration
                        logger.warning(f"⚠️ Disambiguation failed, retrying with different timestamps")
                        retry_count += 1
                        continue
            
            start_valid = start_time in valid_start_times and start_seconds in valid_start_seconds
            end_valid = end_time in valid_end_times and end_seconds in valid_end_seconds
            
            # Debug logging for timestamp validation
            logger.debug(f"🔍 Validating timestamps for scene {scene.scene_number}:")
            logger.debug(f"   LLM start: {start_time} ({start_seconds:.3f}s) - in valid_start_times: {start_time in valid_start_times}, in valid_start_seconds: {start_seconds in valid_start_seconds}")
            logger.debug(f"   LLM end: {end_time} ({end_seconds:.3f}s) - in valid_end_times: {end_time in valid_end_times}, in valid_end_seconds: {end_seconds in valid_end_seconds}")
            
            # If exact match fails, find closest valid timestamps
            if not start_valid or not end_valid:
                logger.info(f"🔍 Exact timestamps not found, finding closest valid matches for scene {scene.scene_number}")
                logger.info(f"   start_valid: {start_valid}, end_valid: {end_valid}")
                
                # Find closest start timestamp
                if not start_valid:
                    closest_start_sub = min(subtitles, key=lambda s: abs(s.start_seconds - start_seconds))
                    corrected_start_time = closest_start_sub.start_time
                    corrected_start_seconds = closest_start_sub.start_seconds
                    logger.info(f"   🔧 Corrected start: {start_time} → {corrected_start_time} (diff: {abs(start_seconds - corrected_start_seconds):.3f}s)")
                else:
                    corrected_start_time = start_time
                    corrected_start_seconds = start_seconds
                
                # Find closest end timestamp
                if not end_valid:
                    closest_end_sub = min(subtitles, key=lambda s: abs(s.end_seconds - end_seconds))
                    corrected_end_time = closest_end_sub.end_time
                    corrected_end_seconds = closest_end_sub.end_seconds
                    logger.info(f"   🔧 Corrected end: {end_time} → {corrected_end_time} (diff: {abs(end_seconds - corrected_end_seconds):.3f}s)")
                else:
                    corrected_end_time = end_time
                    corrected_end_seconds = end_seconds
                
                # Validate corrected timestamps make sense (start before end)
                if corrected_start_seconds <= corrected_end_seconds:
                    scene.start_time = corrected_start_time
                    scene.end_time = corrected_end_time
                    scene.start_seconds = corrected_start_seconds
                    scene.end_seconds = corrected_end_seconds
                    
                    logger.info(f"✅ Scene {scene.scene_number} mapped with corrected timestamps: {corrected_start_time} --> {corrected_end_time}")
                    return scene
                else:
                    logger.warning(f"⚠️ Corrected timestamps invalid: start ({corrected_start_seconds}) > end ({corrected_end_seconds})")
                    retry_count += 1
                    continue
            
            if start_valid and end_valid:
                # SUCCESS: Both timestamps are valid
                scene.start_time = start_time
                scene.end_time = end_time
                scene.start_seconds = start_seconds
                scene.end_seconds = end_seconds
                
                logger.info(f"✅ Scene {scene.scene_number} mapped successfully to {start_time} --> {end_time}")
                return scene
            else:
                # VALIDATION FAILED: Log details and retry
                error_details = []
                if not start_valid:
                    error_details.append(f"start_time '{start_time}' not found in valid subtitle start times")
                if not end_valid:
                    error_details.append(f"end_time '{end_time}' not found in valid subtitle end times")
        
                retry_count += 1
                
        except Exception as e:
            logger.error(f"Error mapping scene {scene.scene_number} to timestamps (attempt {retry_count + 1}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"🔄 Retrying due to error...")
    
    # All retries failed - apply intelligent fallback
    logger.error(f"❌ Failed to map scene {scene.scene_number} to valid timestamps after {max_retries} attempts")
    logger.error(f"Scene content: {scene.plot_segment[:100]}...")
    
    # Apply intelligent fallback: find next available timestamp range
    if previously_mapped_scenes:
        logger.info(f"🔧 Applying intelligent fallback for scene {scene.scene_number}")
        
        # Find the last mapped scene's end time
        last_mapped_scene = None
        for prev_scene in reversed(previously_mapped_scenes):
            if prev_scene.end_seconds is not None:
                last_mapped_scene = prev_scene
                break
        
        if last_mapped_scene:
            # Find the next available subtitle after the last mapped scene
            available_subtitles = [
                sub for sub in subtitles 
                if sub.start_seconds > last_mapped_scene.end_seconds
            ]
            
            if available_subtitles:
                # Assign a reasonable range from available subtitles
                start_subtitle = available_subtitles[0]
                
                # Try to find a reasonable end point (look for natural break or use multiple subtitles)
                scene_duration_estimate = 60.0  # Assume 1 minute default scene length
                target_end_seconds = start_subtitle.start_seconds + scene_duration_estimate
                
                # Find the subtitle closest to our target end time
                end_subtitle = min(
                    available_subtitles,
                    key=lambda s: abs(s.end_seconds - target_end_seconds)
                )
                
                # Ensure we don't go beyond available subtitles
                if end_subtitle.end_seconds > available_subtitles[-1].end_seconds:
                    end_subtitle = available_subtitles[-1]
                
                # Apply fallback timestamps
                scene.start_time = start_subtitle.start_time
                scene.end_time = end_subtitle.end_time
                scene.start_seconds = start_subtitle.start_seconds
                scene.end_seconds = end_subtitle.end_seconds
                
                logger.info(f"🔧 Fallback assigned for scene {scene.scene_number}: {scene.start_time} --> {scene.end_time}")
                logger.info(f"   Duration: {scene.end_seconds - scene.start_seconds:.2f}s")
                return scene
    
    # If no intelligent fallback possible, return scene without timestamps
    logger.warning(f"⚠️ No fallback possible for scene {scene.scene_number} - returned without timestamps")
    return scene

def find_next_available_start_time(after_seconds: float, subtitles: List[SubtitleEntry]) -> Tuple[Optional[str], Optional[float]]:
    """
    Find the next available subtitle start time after a given time.
    
    Args:
        after_seconds: Time in seconds after which to find the next start
        subtitles: List of subtitle entries
        
    Returns:
        Tuple of (next_start_time, next_start_seconds) or (None, None) if not found
    """
    # Find the first subtitle that starts after the given time
    next_subtitles = [
        sub for sub in subtitles 
        if sub.start_seconds > after_seconds
    ]
    
    if next_subtitles:
        # Sort by start time and return the earliest
        next_subtitle = min(next_subtitles, key=lambda s: s.start_seconds)
        return next_subtitle.start_time, next_subtitle.start_seconds
    
    return None, None

def find_closest_scene_start_timestamp(target_seconds: float, subtitles: List[SubtitleEntry]) -> Tuple[str, float]:
    """
    Find the closest subtitle START time to the target time.
    
    Args:
        target_seconds: Target start time in seconds
        subtitles: List of subtitle entries
        
    Returns:
        Tuple of (closest_start_time, start_seconds)
    """
    closest_start_sub = min(subtitles, key=lambda s: abs(s.start_seconds - target_seconds))
    return (closest_start_sub.start_time, closest_start_sub.start_seconds)

def find_closest_scene_end_timestamp(target_seconds: float, subtitles: List[SubtitleEntry]) -> Tuple[str, float]:
    """
    Find the subtitle that should end the scene closest to target time.
    This finds the subtitle whose END time is closest to or just before the target.
    
    Args:
        target_seconds: Target end time in seconds
        subtitles: List of subtitle entries
        
    Returns:
        Tuple of (closest_end_time, end_seconds)
    """
    # Find subtitle whose end time is closest to target
    # Prefer subtitles that end at or before the target time
    valid_subs = []
    
    for sub in subtitles:
        # Prefer subtitles that end at or before target
        if sub.end_seconds <= target_seconds:
            valid_subs.append((sub, abs(sub.end_seconds - target_seconds)))
        else:
            # If subtitle ends after target, penalize it slightly
            valid_subs.append((sub, abs(sub.end_seconds - target_seconds) + 0.1))
    
    if not valid_subs:
        # Fallback to closest subtitle end time
        closest_sub = min(subtitles, key=lambda s: abs(s.end_seconds - target_seconds))
        return (closest_sub.end_time, closest_sub.end_seconds)
    
    # Return the subtitle with minimum distance (accounting for preference)
    closest_sub = min(valid_subs, key=lambda x: x[1])[0]
    return (closest_sub.end_time, closest_sub.end_seconds)

def find_closest_valid_timestamps(
    target_start_seconds: float, 
    target_end_seconds: float, 
    subtitles: List[SubtitleEntry]
) -> Tuple[str, str, float, float]:
    """
    Find the closest valid subtitle timestamps to the target times.
    
    Args:
        target_start_seconds: Target start time in seconds
        target_end_seconds: Target end time in seconds
        subtitles: List of subtitle entries
        
    Returns:
        Tuple of (closest_start_time, closest_end_time, start_seconds, end_seconds)
    """
    # Find closest start timestamp
    closest_start_time, start_seconds = find_closest_scene_start_timestamp(target_start_seconds, subtitles)
    
    # Find closest end timestamp  
    closest_end_time, end_seconds = find_closest_scene_end_timestamp(target_end_seconds, subtitles)
    
    return (closest_start_time, closest_end_time, start_seconds, end_seconds)

def detect_gaps_and_overlaps(scenes: List[PlotScene], subtitles: List[SubtitleEntry]) -> List[Dict]:
    """
    Detect gaps and overlaps between consecutive scenes.
    
    Args:
        scenes: List of mapped scenes with timestamps
        subtitles: List of subtitle entries for reference
        
    Returns:
        List of issues found with details for correction
    """
    logger.info("🔍 Detecting gaps and overlaps between scenes")
    
    issues = []
    subtitle_times = sorted([(sub.start_seconds, sub.end_seconds, sub.index) for sub in subtitles])
    
    for i in range(len(scenes) - 1):
        current_scene = scenes[i]
        next_scene = scenes[i + 1]
        
        # Skip scenes without valid timestamps
        if (not current_scene.end_seconds or not next_scene.start_seconds):
            continue
            
        gap_or_overlap = next_scene.start_seconds - current_scene.end_seconds
        
        if abs(gap_or_overlap) > 0.1:  # More than 0.1 second difference
            if gap_or_overlap > 0:
                # GAP: Find subtitles that fall in the gap
                gap_subtitles = [
                    sub for sub in subtitles 
                    if current_scene.end_seconds < sub.start_seconds < next_scene.start_seconds
                ]
                
                if gap_subtitles:
                    issue = {
                        "type": "gap",
                        "scene_before": current_scene.scene_number,
                        "scene_after": next_scene.scene_number,
                        "gap_start": current_scene.end_seconds,
                        "gap_end": next_scene.start_seconds,
                        "gap_duration": gap_or_overlap,
                        "missed_subtitles": len(gap_subtitles),
                        "gap_subtitle_indices": [sub.index for sub in gap_subtitles]
                    }
                    issues.append(issue)
                    logger.warning(f"⚠️ GAP detected between scenes {current_scene.scene_number}-{next_scene.scene_number}: {gap_or_overlap:.2f}s, missing {len(gap_subtitles)} subtitles")
                    
            else:  # gap_or_overlap < 0
                # OVERLAP: Scenes overlap in time
                overlap_duration = abs(gap_or_overlap)
                overlapped_subtitles = [
                    sub for sub in subtitles 
                    if next_scene.start_seconds <= sub.start_seconds <= current_scene.end_seconds
                ]
                
                issue = {
                    "type": "overlap",
                    "scene_before": current_scene.scene_number,
                    "scene_after": next_scene.scene_number,
                    "overlap_start": next_scene.start_seconds,
                    "overlap_end": current_scene.end_seconds,
                    "overlap_duration": overlap_duration,
                    "overlapped_subtitles": len(overlapped_subtitles),
                    "overlap_subtitle_indices": [sub.index for sub in overlapped_subtitles]
                }
                issues.append(issue)
                logger.warning(f"⚠️ OVERLAP detected between scenes {current_scene.scene_number}-{next_scene.scene_number}: {overlap_duration:.2f}s, affecting {len(overlapped_subtitles)} subtitles")
    
    logger.info(f"Found {len(issues)} gap/overlap issues")
    return issues

def fix_scene_boundaries(
    scenes: List[PlotScene], 
    issues: List[Dict], 
    subtitles: List[SubtitleEntry], 
    llm: AzureChatOpenAI
) -> List[PlotScene]:
    """
    Fix gaps and overlaps between scenes using LLM to determine proper boundaries.
    Now includes context about all previously assigned timestamps to avoid conflicts.
    
    Args:
        scenes: List of scenes with potential boundary issues
        issues: List of detected gaps/overlaps
        subtitles: List of subtitle entries
        llm: LLM for boundary correction
        
    Returns:
        List of corrected scenes
    """
    logger.info("🔧 Fixing scene boundaries using LLM with timestamp context")
    
    if not issues:
        logger.info("No boundary issues found - returning original scenes")
        return scenes
    
    corrected_scenes = scenes.copy()
    
    for issue in issues:
        scene_before_idx = next(i for i, s in enumerate(corrected_scenes) if s.scene_number == issue["scene_before"])
        scene_after_idx = next(i for i, s in enumerate(corrected_scenes) if s.scene_number == issue["scene_after"])
        
        scene_before = corrected_scenes[scene_before_idx]
        scene_after = corrected_scenes[scene_after_idx]
        
        logger.info(f"🔧 Fixing {issue['type']} between scenes {scene_before.scene_number} and {scene_after.scene_number}")
        
        # Build context of ALL previously assigned timestamps
        timestamp_context = []
        for i, scene in enumerate(corrected_scenes):
            if scene.start_time and scene.end_time:
                timestamp_context.append(
                    f"Scene {scene.scene_number}: {scene.start_time} --> {scene.end_time}"
                )
        
        timestamp_context_str = "\n".join(timestamp_context)
        
        # Get relevant subtitles for context
        if issue["type"] == "gap":
            # Include subtitles around the gap
            relevant_subtitles = [
                sub for sub in subtitles 
                if (scene_before.end_seconds - 10) <= sub.start_seconds <= (scene_after.start_seconds + 10)
            ]
        else:  # overlap
            # Include subtitles in the overlap region
            relevant_subtitles = [
                sub for sub in subtitles 
                if (scene_before.start_seconds) <= sub.start_seconds <= (scene_after.end_seconds)
            ]
        
        # Format subtitles for LLM
        subtitle_context = "\n".join([
            f"[{sub.start_time} --> {sub.end_time}] {sub.text}" 
            for sub in relevant_subtitles
        ])
        
        prompt = f"""You need to fix a {issue["type"]} between two consecutive scenes. You must respect the timestamps already assigned to other scenes.

CURRENT SITUATION:
- Scene {scene_before.scene_number} currently ends at: {scene_before.end_time} ({scene_before.end_seconds:.2f}s)
- Scene {scene_after.scene_number} currently starts at: {scene_after.start_time} ({scene_after.start_seconds:.2f}s)
- Issue: {issue["type"].upper()} of {issue.get("gap_duration", issue.get("overlap_duration", 0)):.2f} seconds

CRITICAL: EXISTING TIMESTAMP ASSIGNMENTS FOR ALL SCENES:
{timestamp_context_str}

⚠️ WARNING: You MUST NOT assign timestamps that conflict with other scenes already assigned above!

SCENE CONTENT:
Scene {scene_before.scene_number}: {scene_before.plot_segment[:200]}...

Scene {scene_after.scene_number}: {scene_after.plot_segment[:200]}...

RELEVANT SUBTITLES WITH TIMESTAMPS:
{subtitle_context}

TASK: Determine the exact boundary between these scenes. You must:
1. Choose a subtitle timestamp that naturally separates the two scenes
2. The first scene should end at the END timestamp of a subtitle
3. The second scene should start at the START timestamp of the next subtitle
4. Ensure no subtitles are missed and no subtitles are duplicated
5. Use EXACT timestamps from the subtitle list above
6. ⚠️ CRITICAL: Do NOT conflict with timestamps already assigned to other scenes listed above

Return ONLY a JSON object:
{{
    "scene_before_end_time": "HH:MM:SS,mmm",
    "scene_after_start_time": "HH:MM:SS,mmm",
    "reasoning": "Brief explanation of why this boundary was chosen and how it avoids conflicts"
}}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            boundary_data = clean_llm_json_response(response.content)
            
            if isinstance(boundary_data, list) and len(boundary_data) > 0:
                boundary_data = boundary_data[0]
            
            if not isinstance(boundary_data, dict):
                logger.error(f"Invalid response format for boundary correction: {boundary_data}")
                continue
                
            new_end_time = boundary_data.get("scene_before_end_time")
            new_start_time = boundary_data.get("scene_after_start_time")
            reasoning = boundary_data.get("reasoning", "No reasoning provided")
            
            if not new_end_time or not new_start_time:
                logger.error(f"Missing timestamps in boundary correction response: {boundary_data}")
                continue
            
            # Validate that new timestamps don't conflict with other scenes
            new_end_seconds = parse_srt_time_to_seconds(new_end_time)
            new_start_seconds = parse_srt_time_to_seconds(new_start_time)
            
            # Check for conflicts with other scenes (excluding the two being fixed)
            conflict_found = False
            for i, other_scene in enumerate(corrected_scenes):
                if i in [scene_before_idx, scene_after_idx]:
                    continue  # Skip the scenes being fixed
                    
                if other_scene.start_seconds and other_scene.end_seconds:
                    # Check if new boundaries overlap with existing scene
                    if (new_end_seconds > other_scene.start_seconds and new_end_seconds < other_scene.end_seconds) or \
                       (new_start_seconds > other_scene.start_seconds and new_start_seconds < other_scene.end_seconds):
                        logger.warning(f"⚠️ Timestamp conflict detected with scene {other_scene.scene_number}: proposed boundaries overlap")
                        conflict_found = True
                        break
            
            if conflict_found:
                logger.warning(f"⚠️ Skipping boundary fix due to conflicts with other scenes")
                continue
            
            # Use flexible timestamp matching instead of exact validation
            try:
                # Find closest valid timestamps using proper scene boundary logic
                closest_end_time, closest_end_seconds = find_closest_scene_end_timestamp(new_end_seconds, subtitles)
                closest_start_time, closest_start_seconds = find_closest_scene_start_timestamp(new_start_seconds, subtitles)
                
                # Log if we made corrections
                if abs(new_end_seconds - closest_end_seconds) > 0.1:
                    logger.info(f"   🔧 Adjusted end time: {new_end_time} → {closest_end_time} (diff: {abs(new_end_seconds - closest_end_seconds):.3f}s)")
                    new_end_time = closest_end_time
                    
                if abs(new_start_seconds - closest_start_seconds) > 0.1:
                    logger.info(f"   🔧 Adjusted start time: {new_start_time} → {closest_start_time} (diff: {abs(new_start_seconds - closest_start_seconds):.3f}s)")
                    new_start_time = closest_start_time
                
                # Apply corrections
                corrected_scenes[scene_before_idx].end_time = new_end_time
                corrected_scenes[scene_before_idx].end_seconds = parse_srt_time_to_seconds(new_end_time)
                
                corrected_scenes[scene_after_idx].start_time = new_start_time
                corrected_scenes[scene_after_idx].start_seconds = parse_srt_time_to_seconds(new_start_time)
                
            except Exception as correction_error:
                logger.error(f"❌ Failed to apply timestamp corrections: {correction_error}")
                continue
            
            logger.info(f"✅ Fixed {issue['type']} between scenes {scene_before.scene_number}-{scene_after.scene_number}")
            logger.info(f"   Scene {scene_before.scene_number} now ends at: {new_end_time}")
            logger.info(f"   Scene {scene_after.scene_number} now starts at: {new_start_time}")
            logger.info(f"   Reasoning: {reasoning}")
            
        except Exception as e:
            logger.error(f"❌ Failed to fix {issue['type']} between scenes {scene_before.scene_number}-{scene_after.scene_number}: {e}")
            continue
    
    return corrected_scenes

def map_scenes_to_timestamps_old(
    scenes: List[PlotScene], 
    subtitles: List[SubtitleEntry], 
    llm: AzureChatOpenAI,
    correction_strategy: str = "intelligent"
) -> List[PlotScene]:
    """
    Map scenes to timestamps with configurable correction strategies.
    
    Args:
        scenes: List of scenes to map
        subtitles: List of subtitle entries
        llm: LLM for timestamp mapping and boundary correction
        correction_strategy: "intelligent" (LLM-based) or "forced" (heuristic gap closure)
        
    Returns:
        List of scenes with corrected timestamps
    """
    logger.info(f"🗺️ Starting sequential scene timestamp mapping with {correction_strategy} correction")
    
    # Step 1: Map each scene to timestamps sequentially with context of previous scenes
    mapped_scenes = []
    for i, scene in enumerate(scenes):
        # Pass previously mapped scenes as context to avoid conflicts
        previously_mapped = mapped_scenes.copy() if mapped_scenes else None
        
        logger.info(f"🔄 Mapping scene {scene.scene_number} ({i+1}/{len(scenes)}) with context of {len(mapped_scenes)} previous scenes")
        mapped_scene = map_scene_to_timestamps(scene, subtitles, llm, previously_mapped)
        mapped_scenes.append(mapped_scene)
    
    # Step 2: Detect gaps and overlaps
    issues = detect_gaps_and_overlaps(mapped_scenes, subtitles)
    
    # Step 3: Apply correction strategy based on preference
    if issues:
        logger.info(f"Found {len(issues)} boundary issues - applying {correction_strategy} corrections")
        
        if correction_strategy == "intelligent":
            # Use LLM-based intelligent boundary correction
            corrected_scenes = fix_scene_boundaries(mapped_scenes, issues, subtitles, llm)
        else:
            # Use forced gap closure (heuristic approach)
            corrected_scenes = force_gap_closure(mapped_scenes, subtitles)
            corrected_scenes = apply_final_forced_closure(corrected_scenes, subtitles)
        
        # Step 4: Verify fixes worked
        remaining_issues = detect_gaps_and_overlaps(corrected_scenes, subtitles)
        if remaining_issues:
            logger.warning(f"⚠️ {len(remaining_issues)} boundary issues remain after {correction_strategy} correction")
            
            # If intelligent correction failed, fallback to forced closure
            if correction_strategy == "intelligent":
                logger.info("🔄 Falling back to forced gap closure")
                corrected_scenes = force_gap_closure(mapped_scenes, subtitles)
                corrected_scenes = apply_final_forced_closure(corrected_scenes, subtitles)
                
                final_issues = detect_gaps_and_overlaps(corrected_scenes, subtitles)
                if final_issues:
                    logger.warning(f"⚠️ {len(final_issues)} boundary issues remain after forced closure")
                else:
                    logger.info("✅ All boundary issues resolved with forced closure fallback")
        else:
            logger.info(f"✅ All boundary issues resolved with {correction_strategy} correction")
            
        return corrected_scenes
    else:
        logger.info("✅ No boundary issues detected - timestamps are properly aligned")
        return mapped_scenes

def force_gap_closure(
    scenes: List[PlotScene], 
    subtitles: List[SubtitleEntry]
) -> List[PlotScene]:
    """
    Force closure of gaps between consecutive scenes by extending either the previous or next scene.
    
    Args:
        scenes: List of scenes with timestamps
        subtitles: List of subtitle entries
        
    Returns:
        List of scenes with gaps closed
    """
    if len(scenes) < 2:
        return scenes
    
    logger.info("🔧 Applying forced gap closure")
    corrected_scenes = scenes.copy()
    
    for i in range(len(corrected_scenes) - 1):
        current_scene = corrected_scenes[i]
        next_scene = corrected_scenes[i + 1]
        
        # Check if there's a gap between current and next scene
        if (current_scene.end_seconds is not None and 
            next_scene.start_seconds is not None and
            current_scene.end_seconds < next_scene.start_seconds):
            
            gap_duration = next_scene.start_seconds - current_scene.end_seconds
            logger.info(f"🔧 Gap detected between scenes {i+1} and {i+2}: {gap_duration:.2f}s")
            
            # Find the nearest subtitle to the gap
            gap_center = (current_scene.end_seconds + next_scene.start_seconds) / 2
            
            # Find subtitle closest to gap center
            closest_subtitle = min(subtitles, key=lambda s: abs(s.start_seconds - gap_center))
            
            # Decide whether to extend previous or next scene based on which is closer
            if abs(closest_subtitle.start_seconds - current_scene.end_seconds) < abs(closest_subtitle.start_seconds - next_scene.start_seconds):
                # Extend previous scene
                current_scene.end_seconds = next_scene.start_seconds
                current_scene.end_time = next_scene.start_time
                logger.info(f"🔧 Extended scene {i+1} to close gap")
            else:
                # Extend next scene
                next_scene.start_seconds = current_scene.end_seconds
                next_scene.start_time = current_scene.end_time
                logger.info(f"🔧 Extended scene {i+2} to close gap")
    
    return corrected_scenes

def get_llm_gap_decision(
    gap_info: Dict, 
    llm: AzureChatOpenAI
) -> str:
    """
    Ask LLM to decide how to close a gap between scenes.
    
    Args:
        gap_info: Dictionary with gap information
        llm: LLM for decision making
        
    Returns:
        Decision: "extend_previous" or "extend_next"
    """
    prompt = f"""You are analyzing a gap between two consecutive scenes in a TV episode.

**Gap Information:**
- Gap duration: {gap_info['gap_duration']:.2f} seconds
- Previous scene ends at: {gap_info['previous_end']:.2f}s
- Next scene starts at: {gap_info['next_start']:.2f}s
- Gap center: {gap_info['gap_center']:.2f}s

**Available subtitles in the gap:**
{gap_info['gap_subtitles']}

**Decision needed:**
Should the gap be closed by:
1. Extending the previous scene to include the gap content
2. Extending the next scene to include the gap content

Consider:
- Which scene the gap content belongs to thematically
- Natural story flow and continuity
- Timing and pacing

Return only: "extend_previous" or "extend_next"
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()
        
        if decision in ["extend_previous", "extend_next"]:
            return decision
        else:
            logger.warning(f"⚠️ Invalid LLM decision: {decision}, using heuristic fallback")
            return "extend_previous"  # Default fallback
    except Exception as e:
        logger.error(f"❌ Error getting LLM gap decision: {e}")
        return "extend_previous"  # Default fallback

def force_heuristic_decision(
    gap_info: Dict
) -> str:
    """
    Apply heuristic decision when LLM fails to decide.
    
    Args:
        gap_info: Dictionary with gap information
        
    Returns:
        Decision: "extend_previous" or "extend_next"
    """
    # Simple heuristic: extend the scene that's closer to the gap center
    gap_center = gap_info['gap_center']
    previous_end = gap_info['previous_end']
    next_start = gap_info['next_start']
    
    if abs(gap_center - previous_end) < abs(gap_center - next_start):
        return "extend_previous"
    else:
        return "extend_next"

def apply_final_forced_closure(
    scenes: List[PlotScene], 
    subtitles: List[SubtitleEntry]
) -> List[PlotScene]:
    """
    Apply final forced closure to any remaining gaps.
    
    Args:
        scenes: List of scenes with timestamps
        subtitles: List of subtitle entries
        
    Returns:
        List of scenes with all gaps closed
    """
    logger.info("🔧 Applying final forced gap closure")
    
    if len(scenes) < 2:
        return scenes
    
    corrected_scenes = scenes.copy()
    
    for i in range(len(corrected_scenes) - 1):
        current_scene = corrected_scenes[i]
        next_scene = corrected_scenes[i + 1]
        
        # Check for any remaining gaps
        if (current_scene.end_seconds is not None and 
            next_scene.start_seconds is not None and
            current_scene.end_seconds < next_scene.start_seconds):
            
            # Force closure by extending the previous scene
            current_scene.end_seconds = next_scene.start_seconds
            current_scene.end_time = next_scene.start_time
            logger.info(f"🔧 Final forced closure: extended scene {i+1}")
    
    return corrected_scenes

def save_plot_files(plot_data: Dict, output_dir: str, episode_prefix: str) -> Tuple[str, str]:
    """Save plot as both TXT (full plot) and JSON (scenes) files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate full plot text
    scenes = plot_data.get("scenes", [])
    full_plot_lines = []
    
    for scene in scenes:
        if isinstance(scene, dict):
            plot_segment = scene.get("plot_segment", "")
        else:
            plot_segment = str(scene)
        
        if plot_segment:  # Only add non-empty segments
            full_plot_lines.append(plot_segment)
    
    # Join with proper sentence separation
    full_plot = " ".join(full_plot_lines)
    
    # Ensure we have some content
    if not full_plot.strip():
        logger.error("Generated plot is empty!")
        raise ValueError("Generated plot text is empty")
    
    # Save TXT file
    txt_path = output_path / f"{episode_prefix}_plot.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_plot)
    
    # Save JSON file
    json_path = output_path / f"{episode_prefix}_plot_scenes.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(plot_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved plot files: {txt_path} and {json_path}")
    return str(txt_path), str(json_path)

def save_scene_timestamps(scenes: List[PlotScene], output_dir: str, episode_prefix: str) -> str:
    """Save scene timestamp mappings to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp_data = {
        "scenes": [
            {
                "scene_number": scene.scene_number,
                "plot_segment": scene.plot_segment,
                "start_time": scene.start_time,
                "end_time": scene.end_time,
                "start_seconds": scene.start_seconds,
                "end_seconds": scene.end_seconds
            }
            for scene in scenes
        ]
    }
    
    json_path = output_path / f"{episode_prefix}_scene_timestamps.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(timestamp_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved scene timestamps: {json_path}")
    return str(json_path)


def load_previous_season_summary(season_summary_path: str) -> Optional[str]:
    """
    Load the previous season summary if it exists.
    
    Args:
        season_summary_path (str): Path to the season summary file
        
    Returns:
        Optional[str]: The season summary content, or None if not found
    """
    try:
        if Path(season_summary_path).exists():
            from backend.src.utils.text_utils import load_text
            summary = load_text(season_summary_path)
            if summary.strip():
                logger.info(f"Loaded previous season summary from: {season_summary_path}")
                return summary
        
        logger.info("No previous season summary found")
        return None
        
    except Exception as e:
        logger.error(f"Error loading previous season summary: {e}")
        return None

# ================================
# NEW SIMPLIFIED SCENE BOUNDARY DETECTION
# ================================

# Import the new simplified approach
from .scene_boundary_detection import map_scenes_to_timestamps_simple

def map_scenes_to_timestamps_v2(
    scenes: List[PlotScene], 
    subtitles: List[SubtitleEntry], 
    llm: AzureChatOpenAI
) -> List[PlotScene]:
    """
    NEW SIMPLIFIED APPROACH: Map scenes to timestamps using subtitle boundary detection.
    
    This function replaces the complex timestamp-based approach with a simpler method:
    1. LLM identifies subtitle numbers where scenes begin/end
    2. Timestamps are extracted mechanically from SRT data
    3. No complex overlap resolution needed
    
    Args:
        scenes: List of scenes to map
        subtitles: List of subtitle entries
        llm: LLM for boundary detection
        
    Returns:
        List of scenes with timestamps
    """
    logger.info("🆕 Using new simplified scene boundary detection approach")
    return map_scenes_to_timestamps_simple(scenes, subtitles, llm)

# ================================
# DEPRECATED FUNCTIONS - TO BE REMOVED
# ================================

def map_scenes_to_timestamps_deprecated(
    scenes: List[PlotScene], 
    subtitles: List[SubtitleEntry], 
    llm: AzureChatOpenAI,
    correction_strategy: str = "intelligent"
) -> List[PlotScene]:
    """
    DEPRECATED: Old complex timestamp mapping approach.
    Use map_scenes_to_timestamps_v2 instead.
    """
    logger.warning("⚠️ Using deprecated scene mapping function. Consider switching to map_scenes_to_timestamps_v2")
    return map_scenes_to_timestamps_old(scenes, subtitles, llm, correction_strategy)

# New main function that uses the simplified approach
def map_scenes_to_timestamps(
    scenes: List[PlotScene], 
    subtitles: List[SubtitleEntry], 
    llm: AzureChatOpenAI,
    correction_strategy: str = "intelligent"
) -> List[PlotScene]:
    """
    Map scenes to timestamps using the new simplified boundary detection approach.
    
    This function uses LLM to identify subtitle boundaries between scenes,
    then extracts timestamps mechanically from the SRT data.
    
    Args:
        scenes: List of scenes to map
        subtitles: List of subtitle entries
        llm: LLM for boundary detection
        correction_strategy: Kept for compatibility, but not used in new approach
        
    Returns:
        List of scenes with timestamps
    """
    logger.info("🆕 Using new simplified scene boundary detection approach")
    return map_scenes_to_timestamps_simple(scenes, subtitles, llm)
