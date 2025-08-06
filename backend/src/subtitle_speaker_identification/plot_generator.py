"""
Plot generation and scene timestamp correction based on possible speakers SRT.
"""

import json
import logging
import os
from typing import List, Dict, Optional
from pathlib import Path
from langchain_core.messages import HumanMessage

from ..utils.logger_utils import setup_logging
from ..path_handler import PathHandler
from ..ai_models.ai_models import get_llm, LLMType

logger = setup_logging(__name__)

class PlotGenerator:
    """Generate plots from possible speakers SRT and correct scene timestamps."""
    
    def __init__(self, path_handler: PathHandler, llm=None):
        self.path_handler = path_handler
        self.llm = llm or get_llm(LLMType.INTELLIGENT)
    
    def generate_plot_from_possible_speakers(self) -> str:
        """
        Generate a plot from the possible speakers SRT file.
        
        Returns:
            Path to the generated plot file
        """
        logger.info("📝 Generating plot from possible speakers SRT...")
        
        try:
            # Read the possible speakers SRT file
            srt_path = self.path_handler.get_possible_speakers_srt_path()
            if not os.path.exists(srt_path):
                raise FileNotFoundError(f"Possible speakers SRT file not found: {srt_path}")
            
            # Parse the SRT file
            srt_content = self._parse_srt_file(srt_path)
            
            # Generate plot using LLM
            plot_content = self._generate_plot_with_llm(srt_content)
            
            # Save the plot
            plot_path = self.path_handler.get_plot_possible_speakers_path()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            
            with open(plot_path, 'w', encoding='utf-8') as f:
                f.write(plot_content)
            
            logger.info(f"✅ Generated plot from possible speakers: {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.error(f"❌ Failed to generate plot from possible speakers: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise
    
    def correct_scene_timestamps(self, plot_path: str) -> str:
        """
        Correct scene timestamps using the new plot.
        
        Args:
            plot_path: Path to the new plot file
            
        Returns:
            Path to the corrected scene timestamps file
        """
        logger.info("🔄 Correcting scene timestamps with new plot...")
        
        try:
            # Read the original scene timestamps
            timestamps_path = self.path_handler.get_scene_timestamps_path()
            if not os.path.exists(timestamps_path):
                raise FileNotFoundError(f"Scene timestamps file not found: {timestamps_path}")
            
            with open(timestamps_path, 'r', encoding='utf-8') as f:
                timestamps_data = json.load(f)
            
            # Read the new plot
            with open(plot_path, 'r', encoding='utf-8') as f:
                new_plot = f.read()
            
            # Correct each scene
            corrected_scenes = []
            for scene in timestamps_data['scenes']:
                corrected_plot_segment = self._correct_scene_plot_segment(
                    scene, new_plot
                )
                
                corrected_scene = {
                    **scene,
                    'plot_segment': corrected_plot_segment
                }
                corrected_scenes.append(corrected_scene)
            
            # Create corrected timestamps data
            corrected_timestamps_data = {
                **timestamps_data,
                'scenes': corrected_scenes
            }
            
            # Save corrected timestamps
            corrected_path = self.path_handler.get_corrected_scene_timestamps_path()
            os.makedirs(os.path.dirname(corrected_path), exist_ok=True)
            
            with open(corrected_path, 'w', encoding='utf-8') as f:
                json.dump(corrected_timestamps_data, f, indent=2)
            
            logger.info(f"✅ Corrected scene timestamps: {corrected_path}")
            return corrected_path
            
        except Exception as e:
            logger.error(f"❌ Failed to correct scene timestamps: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise
    
    def _parse_srt_file(self, srt_path: str) -> str:
        """Parse SRT file and return content as text."""
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert SRT to readable text format
        lines = content.strip().split('\n')
        dialogue_text = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip subtitle number
            if line.isdigit():
                i += 1
                continue
            
            # Skip timestamp line
            if ' --> ' in line:
                i += 1
                continue
            
            # Get dialogue line
            if i < len(lines):
                dialogue_line = lines[i].strip()
                if dialogue_line:
                    dialogue_text.append(dialogue_line)
            
            i += 1
        
        return '\n'.join(dialogue_text)
    
    def _generate_plot_with_llm(self, srt_content: str) -> str:
        """Generate plot using LLM from SRT content."""
        
        prompt = f"""You are a script analyst that understands character names, relationships, and storylines from subtitles. 

Your task is to generate the plot of the episode based on the subtitles with possible speaker identifications.

**IMPORTANT CONTEXT ABOUT THE SPEAKER IDENTIFICATIONS:**
- When you see "Speaker / Speaker (PVM)", this means:
  - The first speaker is the possible name given by another episode analyzer without having access to voices but only text
  - The second speaker (after the slash) is the possible voice match from audio clustering
  - "(PVM)" stands for "Possible Voice Match"
  - The voice clustering is not 100% sure, so the names are just for context
  - You should choose the most appropriate speaker based on the context and dialogue

- When you see just "Speaker:", this means the speaker identification is confident

**Plot Summary Requirements:**
- Extract every narrative event from the subtitles in chronological order
- Use clear, concise sentences (no run-on sentences)
- Include all dialogue, actions, and story developments
- Maintain objective tone - report what happens, don't interpret
- Provide a comprehensive, flowing narrative of the entire episode
- Do not subdivide into scenes - create one continuous, detailed plot summary
- When you encounter "Speaker / Speaker (PVM)", choose the most appropriate speaker based on context and dialogue
- Use the speaker names that make the most sense for the plot flow
- **Do NOT invent characters that are not present in the subtitles**
- **Use the episode context to identify incorrect character assignments**
- **If a character follows a certain path during the episode and a subtitle seems to be assigned to another character but would fit better with the first character's storyline, consider that the assignment might be wrong**
- **Cross-reference character actions and dialogue patterns throughout the episode to ensure consistency**
- **Understand the whole episode context to make informed decisions about speaker assignments**

Here are the subtitles with possible speaker identifications:

{srt_content}

Please provide a comprehensive, detailed plot summary of the entire episode:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"❌ LLM plot generation failed: {e}")
            raise
    
    def _correct_scene_plot_segment(self, scene: Dict, new_plot: str) -> str:
        """Correct a single scene's plot segment using the new plot."""
        
        scene_number = scene['scene_number']
        original_plot = scene['plot_segment']
        start_time = scene['start_time']
        end_time = scene['end_time']
        
        prompt = f"""You are an expert TV series analyst. I need you to correct and enhance the plot description of a specific scene.

SCENE INFORMATION:
- Scene Number: {scene_number}
- Time Range: {start_time} to {end_time}
- Original Plot Description: {original_plot}

NEW EPISODE PLOT (with more detailed speaker information):
{new_plot}

**CRITICAL REQUIREMENTS:**
- Provide ONLY the corrected plot description for this specific scene
- Do NOT add any introductory phrases like "The episode opens", "In this episode", "This scene shows", etc.
- Do NOT add any formatting, headers, or explanatory text
- Do NOT include the scene number, time range, or any metadata in your response
- Write a direct, clean description of what happens in this scene
- Be more accurate based on the new plot information
- Include more specific character names and details
- Add any missing details that are now clear from the new plot
- Correct any inaccuracies in the original description
- Maintain the same level of detail but with improved accuracy
- The names of the characters should be the same as in the new plot, not the ones from the scene that needs to be corrected.

Provide ONLY the corrected plot description:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"❌ LLM scene correction failed for scene {scene_number}: {e}")
            # Return original if correction fails
            return original_plot 