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

def map_scene_to_timestamps(scene: PlotScene, subtitles: List[SubtitleEntry], llm: AzureChatOpenAI) -> PlotScene:
    """Map a plot scene to subtitle timestamps using LLM with validation and retry."""
    logger.info(f"Mapping scene {scene.scene_number} to timestamps")
    
    # Format subtitles with timestamps
    subtitle_text = format_subtitles_for_llm(subtitles)
    
    # Create sets of valid start and end times for validation
    valid_start_times = {sub.start_time for sub in subtitles}
    valid_end_times = {sub.end_time for sub in subtitles}
    valid_start_seconds = {sub.start_seconds for sub in subtitles}
    valid_end_seconds = {sub.end_seconds for sub in subtitles}
    
    prompt = f"""Given the following subtitles with timestamps and the plot scene description, identify the EXACT start and end timestamps for this scene.

Plot Scene:
{scene.plot_segment}

Subtitles:
{subtitle_text}

CRITICAL REQUIREMENTS: 
- For start_time: Use the EXACT start timestamp of the FIRST subtitle in the scene (e.g., "00:15:23,456")
- For end_time: Use the EXACT end timestamp of the LAST subtitle in the scene (e.g., "00:15:27,890")
- You MUST use timestamps that appear EXACTLY in the subtitle list above
- Do NOT modify or approximate the timestamps - copy them exactly as shown

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
    
    # All retries failed
    logger.error(f"❌ Failed to map scene {scene.scene_number} to valid timestamps after {max_retries} attempts")
    logger.error(f"Scene content: {scene.plot_segment[:100]}...")
    return scene

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
    
    Args:
        scenes: List of scenes with potential boundary issues
        issues: List of detected gaps/overlaps
        subtitles: List of subtitle entries
        llm: LLM for boundary correction
        
    Returns:
        List of corrected scenes
    """
    logger.info("🔧 Fixing scene boundaries using LLM")
    
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
        
        prompt = f"""You need to fix a {issue["type"]} between two consecutive scenes. Determine the exact boundary where the first scene should end and the second scene should start.

CURRENT SITUATION:
- Scene {scene_before.scene_number} currently ends at: {scene_before.end_time}
- Scene {scene_after.scene_number} currently starts at: {scene_after.start_time}
- Issue: {issue["type"].upper()} of {issue.get("gap_duration", issue.get("overlap_duration", 0)):.2f} seconds

SCENE CONTENT:
Scene {scene_before.scene_number}: {scene_before.plot_segment}

Scene {scene_after.scene_number}: {scene_after.plot_segment}

RELEVANT SUBTITLES WITH TIMESTAMPS:
{subtitle_context}

TASK: Determine the exact boundary between these scenes. You must:
1. Choose a subtitle timestamp that naturally separates the two scenes
2. The first scene should end at the END timestamp of a subtitle
3. The second scene should start at the START timestamp of the next subtitle
4. Ensure no subtitles are missed and no subtitles are duplicated
5. Use EXACT timestamps from the subtitle list above

Return ONLY a JSON object:
{{
    "scene_before_end_time": "HH:MM:SS,mmm",
    "scene_after_start_time": "HH:MM:SS,mmm",
    "reasoning": "Brief explanation of why this boundary was chosen"
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
            
            # Use flexible timestamp matching instead of exact validation
            try:
                new_end_seconds = parse_srt_time_to_seconds(new_end_time)
                new_start_seconds = parse_srt_time_to_seconds(new_start_time)
                
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

def map_scenes_to_timestamps_with_boundary_correction(
    scenes: List[PlotScene], 
    subtitles: List[SubtitleEntry], 
    llm: AzureChatOpenAI
) -> List[PlotScene]:
    """
    Map scenes to timestamps and fix any gaps or overlaps between consecutive scenes.
    
    Args:
        scenes: List of scenes to map
        subtitles: List of subtitle entries
        llm: LLM for timestamp mapping and boundary correction
        
    Returns:
        List of scenes with corrected timestamps
    """
    logger.info("🗺️ Starting scene timestamp mapping with boundary correction")
    
    # Step 1: Map each scene to timestamps individually
    mapped_scenes = []
    for scene in scenes:
        mapped_scene = map_scene_to_timestamps(scene, subtitles, llm)
        mapped_scenes.append(mapped_scene)
    
    # Step 2: Detect gaps and overlaps
    issues = detect_gaps_and_overlaps(mapped_scenes, subtitles)
    
    # Step 3: Fix boundary issues if any exist
    if issues:
        logger.info(f"Found {len(issues)} boundary issues - applying corrections")
        corrected_scenes = fix_scene_boundaries(mapped_scenes, issues, subtitles, llm)
        
        # Step 4: Verify fixes worked
        remaining_issues = detect_gaps_and_overlaps(corrected_scenes, subtitles)
        if remaining_issues:
            logger.warning(f"⚠️ {len(remaining_issues)} boundary issues remain after correction")
        else:
            logger.info("✅ All boundary issues resolved successfully")
            
        return corrected_scenes
    else:
        logger.info("✅ No boundary issues detected - timestamps are properly aligned")
        return mapped_scenes

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
