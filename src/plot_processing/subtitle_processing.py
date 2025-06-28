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

from src.utils.logger_utils import setup_logging
from src.ai_models.ai_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from src.utils.llm_utils import clean_llm_json_response

logger = setup_logging(__name__)

@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry."""
    index: int
    start_time: str
    end_time: str
    text: str
    start_seconds: float
    end_seconds: float

@dataclass
class PlotScene:
    """Represents a plot scene with content and timing."""
    scene_number: int
    plot_segment: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None

def parse_srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS,mmm) to seconds."""
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def parse_srt_file(srt_path: str) -> List[SubtitleEntry]:
    """Parse an SRT subtitle file and return list of subtitle entries."""
    logger.info(f"Parsing SRT file: {srt_path}")
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(srt_path, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Split into subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        try:
            # Parse index
            index = int(lines[0])
            
            # Parse time range
            time_line = lines[1]
            time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', time_line)
            if not time_match:
                continue
                
            start_time = time_match.group(1)
            end_time = time_match.group(2)
            
            # Parse subtitle text (may span multiple lines)
            text = '\n'.join(lines[2:]).strip()
            
            # Convert times to seconds
            start_seconds = parse_srt_time_to_seconds(start_time)
            end_seconds = parse_srt_time_to_seconds(end_time)
            
            subtitles.append(SubtitleEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text,
                start_seconds=start_seconds,
                end_seconds=end_seconds
            ))
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse subtitle block: {block[:50]}... Error: {e}")
            continue
    
    logger.info(f"Parsed {len(subtitles)} subtitle entries")
    return subtitles

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
    base_prompt = """You are a script analyst. Convert the provided subtitles into a detailed plot summary using the exact JSON format below.

**Output Format (JSON only):**
```json
[
  {
    "scene_number": 1,
    "plot_segment": "..."
  },
  {
    "scene_number": 2,
    "plot_segment": "..."
  }
]
```

**Plot Segment Requirements:**
- Extract every narrative event from the subtitles in chronological order
- Use clear, concise sentences (no run-on sentences)
- Include all dialogue, actions, and story developments
- Maintain objective tone - report what happens, don't interpret
- For unidentified speakers, use descriptive terms ("a voice," "someone," "the speaker"). Remember: if you are not totally sure, do not invent the names of the character doing the speaking.
- Number scenes based on natural story breaks or setting changes"""

    # Add season context if available
    if previous_season_summary:
        context_prompt = f"""
**Previous Season Context:**
Use the following summary of previous episodes to better understand character relationships, ongoing storylines, and narrative context. This context helps you interpret the subtitles more accurately but should NOT be included in your plot output - only use it for understanding.

{previous_season_summary}

**Important:** Your plot output should ONLY describe what happens in the current episode subtitles. The context is provided to help you better understand character names, relationships, and ongoing storylines when interpreting the subtitles."""
        
        full_prompt = f"{base_prompt}\n{context_prompt}\n\n**Critical Rules:**\n- Output ONLY the JSON array - no explanations, titles, or additional text\n- Base content exclusively on subtitle text - add nothing extra\n- Use the context to better understand characters and relationships but don't include previous events\n- If subtitles are unclear about speaker identity, use generic references\n- Ensure each plot_segment is comprehensive yet concise"
        
        logger.info("Using previous season context for plot generation")
    else:
        full_prompt = f"{base_prompt}\n\n**Critical Rules:**\n- Output ONLY the JSON array - no explanations, titles, or additional text\n- Base content exclusively on subtitle text - add nothing extra\n- If subtitles are unclear about speaker identity, use generic references\n- Ensure each plot_segment is comprehensive yet concise"
    
    full_prompt += f"\n\nSubtitles:\n{subtitle_text}"
    
    logger.info(f"Sending plot generation request to LLM (subtitle length: {len(subtitle_text)} chars)")
    
    response = llm.invoke([HumanMessage(content=full_prompt)])
    
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
    """Map a plot scene to subtitle timestamps using LLM."""
    logger.info(f"Mapping scene {scene.scene_number} to timestamps")
    
    # Format subtitles with timestamps
    subtitle_text = format_subtitles_for_llm(subtitles)
    
    prompt = f"""Given the following subtitles with timestamps and the plot scene description, identify the EXACT start and end timestamps for this scene.

Plot Scene:
{scene.plot_segment}

Subtitles:
{subtitle_text}

Return only a JSON object with:
{{
    "start_time": "HH:MM:SS,mmm",
    "end_time": "HH:MM:SS,mmm"
}}

Use the exact timestamp format from the subtitles."""
    
    logger.info(f"Sending timestamp mapping request to LLM for scene {scene.scene_number}")
    
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
            return scene
        
        start_time = timing_data.get("start_time")
        end_time = timing_data.get("end_time")
        
        if start_time and end_time:
            scene.start_time = start_time
            scene.end_time = end_time
            scene.start_seconds = parse_srt_time_to_seconds(start_time)
            scene.end_seconds = parse_srt_time_to_seconds(end_time)
            
            logger.info(f"Scene {scene.scene_number} mapped to {start_time} --> {end_time}")
        else:
            logger.warning(f"Could not map scene {scene.scene_number} to timestamps - missing start_time or end_time")
            logger.warning(f"Received timing data: {timing_data}")
            
    except Exception as e:
        logger.error(f"Error mapping scene {scene.scene_number} to timestamps: {e}")
        logger.error(f"Raw LLM response: {getattr(response, 'content', 'No response')}")
    
    return scene

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

def create_or_update_season_summary(
    episode_plot_path: str, 
    season_summary_path: str, 
    episode_summary_path: str,
    llm: AzureChatOpenAI
) -> str:
    """
    Create or update season summary after processing an episode.
    
    This function:
    1. Creates an episode summary from the detailed plot
    2. Updates the cumulative season summary with the new episode
    
    Args:
        episode_plot_path (str): Path to the current episode's detailed plot
        season_summary_path (str): Path to the cumulative season summary
        episode_summary_path (str): Path where episode summary will be saved
        llm (AzureChatOpenAI): The LLM to use for summarization
        
    Returns:
        str: The updated season summary content
    """
    from src.plot_processing.plot_summarizing import create_episode_summary
    from src.utils.text_utils import load_text
    
    try:
        # Step 1: Create episode summary
        episode_summary = create_episode_summary(episode_plot_path, llm, episode_summary_path)
        
        if not episode_summary:
            logger.error("Failed to create episode summary")
            return ""
        
        # Step 2: Handle cumulative season summary
        if Path(season_summary_path).exists():
            # Load existing season summary
            existing_summary = load_text(season_summary_path)
            if existing_summary.strip():
                # Create cumulative summary with previous context + new episode
                from textwrap import dedent
                from langchain_core.messages import HumanMessage
                from src.utils.llm_utils import clean_llm_text_response
                
                prompt = dedent(f"""You are an expert at creating cumulative narrative summaries for TV series.

You will receive TWO types of content that require different treatment:

1. PREVIOUSLY SUMMARIZED CONTENT: Already condensed material from past episodes
   - Do NOT heavily cut or reduce this content further
   - Preserve the existing narrative flow and key details
   - This content should be maintained with minimal changes

2. NEW EPISODE DETAILED SUMMARY: Fresh, summarized content from the latest episode
   - This should be integrated seamlessly with existing content
   - Focus on maintaining chronological flow and narrative continuity
   - Avoid over-summarization of already condensed content

Your task is to create a cumulative summary that:
- Preserves the narrative continuity from previous episodes
- Integrates the new episode content seamlessly
- Maintains chronological order
- Provides context for future episode generation without replacing detailed plot generation

PREVIOUSLY SUMMARIZED CONTENT (preserve with minimal changes):
{existing_summary}

NEW EPISODE SUMMARY (integrate seamlessly):
{episode_summary}

Create a cumulative summary that combines both sections while respecting their different treatment requirements:""")
                
                response = llm.invoke([HumanMessage(content=prompt)])
                cumulative_summary = clean_llm_text_response(response.content.strip())
                logger.info("Updated cumulative season summary with new episode")
            else:
                # Empty existing file, just use new episode summary
                cumulative_summary = episode_summary
                logger.info("Created new season summary from episode summary")
        else:
            # No existing summary, use episode summary as the initial season summary
            cumulative_summary = episode_summary
            logger.info("Created initial season summary from first episode")
        
        # Step 3: Save the updated season summary
        os.makedirs(os.path.dirname(season_summary_path), exist_ok=True)
        with open(season_summary_path, "w", encoding="utf-8") as output_file:
            output_file.write(cumulative_summary)
        
        logger.info(f"Saved updated season summary to: {season_summary_path}")
        return cumulative_summary
        
    except Exception as e:
        logger.error(f"Error creating/updating season summary: {e}")
        return ""

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
            from src.utils.text_utils import load_text
            summary = load_text(season_summary_path)
            if summary.strip():
                logger.info(f"Loaded previous season summary from: {season_summary_path}")
                return summary
        
        logger.info("No previous season summary found")
        return None
        
    except Exception as e:
        logger.error(f"Error loading previous season summary: {e}")
        return None
