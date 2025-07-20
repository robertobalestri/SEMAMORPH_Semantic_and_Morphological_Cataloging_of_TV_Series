"""
Speaker identification system for SEMAMORPH.
Uses LLM to identify speakers in dialogue based on plot context.
"""
import json
from typing import List, Dict, Optional
from ..narrative_storage_management.narrative_models import DialogueLine
from ..ai_models.ai_models import AzureChatOpenAI
from ..utils.llm_utils import clean_llm_json_response
from ..utils.logger_utils import setup_logging
from ..config import config
from langchain_core.messages import HumanMessage
from .speaker_character_validator import SpeakerCharacterValidator

logger = setup_logging(__name__)

class SpeakerIdentifier:
    """Identifies speakers in dialogue using LLM analysis."""
    
    def __init__(self, llm: AzureChatOpenAI, series: str):
        self.llm = llm
        self.series = series
        self.character_validator = SpeakerCharacterValidator(series, llm)
    
    def identify_speakers_for_scene(
        self, 
        scene_plot: str, 
        scene_dialogue_lines: List[DialogueLine],
        character_context: Optional[str] = None,
        episode_summary: Optional[str] = None,
        episode_entities: Optional[List[Dict]] = None,
        episode_plot: Optional[str] = None
    ) -> List[DialogueLine]:
        """
        Identify speakers for dialogue lines in a scene.
        
        Args:
            scene_plot: Plot description for the scene
            scene_dialogue_lines: List of dialogue lines in the scene
            character_context: Optional context about characters
            episode_summary: Optional episode summary for better context
            episode_entities: Optional list of character data dictionaries from current episode entity extraction
            episode_plot: Optional full episode plot for validation context
            
        Returns:
            Updated dialogue lines with speaker and confidence
        """
        logger.info(f"🎭 Identifying speakers for scene with {len(scene_dialogue_lines)} dialogue lines")
        
        if not scene_dialogue_lines:
            logger.warning("⚠️ No dialogue lines provided")
            return scene_dialogue_lines
        
        # Format dialogue for LLM
        dialogue_text = self._format_dialogue_for_llm(scene_dialogue_lines)
        
        # Create prompt
        prompt = self._create_speaker_identification_prompt(
            scene_plot, 
            dialogue_text, 
            character_context,
            episode_summary
        )
        
        logger.info(f"📤 Sending speaker identification request to LLM")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        logger.info(f"🔧 LLM instance: {type(self.llm)} - {self.llm}")
        
        try:
            # Get LLM response
            logger.info("🚀 Invoking LLM...")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"📥 LLM response received - Type: {type(response)}")
            
            response_content = response.content.strip()
            logger.info(f"📥 Response content length: {len(response_content)} characters")
            logger.info(f"📝 First 500 chars of response: {response_content[:500]}...")
            
            # Parse response
            speaker_data = self._parse_speaker_response(response_content)
            
            # Validate speakers against database and get corrected names
            proposed_speakers = list(set([entry["speaker"] for entry in speaker_data if entry.get("speaker")]))
            logger.info(f"🔍 Validating {len(proposed_speakers)} unique proposed speakers")
            
            speaker_mapping = self.character_validator.validate_and_process_speakers(
                proposed_speakers, 
                episode_entities,
                episode_plot
            )
            
            # Update dialogue lines with validated speakers
            updated_lines = self._update_dialogue_with_validated_speakers(
                scene_dialogue_lines, 
                speaker_data,
                speaker_mapping
            )
            
            # Log results
            confident_count = sum(
                1 for line in updated_lines 
                if line.is_llm_confident
            )
            
            logger.info(f"✅ Speaker identification complete: {confident_count}/{len(updated_lines)} with confident LLM assignments")
            
            return updated_lines
            
        except Exception as e:
            logger.error(f"❌ Error in speaker identification: {e}")
            return scene_dialogue_lines
    
    def _format_dialogue_for_llm(self, dialogue_lines: List[DialogueLine]) -> str:
        """Format dialogue lines for LLM processing."""
        formatted_lines = []
        for line in dialogue_lines:
            start_time = self._seconds_to_timestamp(line.start_time)
            end_time = self._seconds_to_timestamp(line.end_time)
            formatted_lines.append(
                f"[{line.index}] {start_time} - {end_time}: {line.text}"
            )
        return '\n'.join(formatted_lines)
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _create_speaker_identification_prompt(
        self, 
        scene_plot: str, 
        dialogue_text: str, 
        character_context: Optional[str] = None,
        episode_summary: Optional[str] = None
    ) -> str:
        """Create prompt for speaker identification."""
        
        base_prompt = """You are analyzing dialogue from a TV episode to identify speakers. Based on the scene context, episode summary, and dialogue content, identify the most likely speaker for each line.

{episode_summary_section}

**Scene Context:**
{scene_plot}

{character_context_section}

**Dialogue Lines:**
{dialogue_text}

**Instructions:**
1. For each dialogue line, identify the most likely speaker
2. Provide a boolean confidence (true/false) indicating if you are extremely confident about the speaker identification
3. Use character names consistently (e.g., if you identify "John" in one line, use "John" not "Johnny" in subsequent lines)
4. Set is_llm_confident to true ONLY if you are extremely confident (> 99 percent sure) about the speaker
5. Set is_llm_confident to false if you have ANY doubts, uncertainties, or if the speaker could be multiple characters
6. Be very conservative - it's better to be uncertain than wrong
7. Consider dialogue content, context, speaking patterns, and character relationships
8. If the dialogue could reasonably be spoken by multiple characters, set is_llm_confident to false
9. ALWAYS provide "other_possible_speakers" as an array of alternative character names who could reasonably speak this dialogue
10. If you are 100% certain about the speaker, "other_possible_speakers" can be an empty array []

**Confidence Guidelines:**
- Set to true ONLY if: The speaker is clearly identified by name, unique speech pattern, or unmistakable context
- Set to false if: Any ambiguity, multiple possible speakers, unclear context, or general uncertainty
- When in doubt, always set to false

**Output Format (JSON only):**
[
  {{
    "line_index": 1,
    "speaker": "Character Name",
    "is_llm_confident": false,
    "other_possible_speakers": ["Alternative Character 1", "Alternative Character 2"],
    "reasoning": "Could be multiple characters, not 100% certain"
  }},
  {{
    "line_index": 2,
    "speaker": "Another Character",
    "is_llm_confident": false,
    "other_possible_speakers": ["Alternative Character 1"],
    "reasoning": "Ambiguous context, multiple possible speakers"
  }},
  {{
    "line_index": 3,
    "speaker": "Clear Speaker",
    "is_llm_confident": true,
    "other_possible_speakers": [],
    "reasoning": "Speaker clearly identified by name in dialogue"
  }}
]

Return only the JSON array, no additional text."""

        # Add episode summary if provided
        episode_summary_section = ""
        if episode_summary:
            episode_summary_section = f"**Episode Summary:**\n{episode_summary}\n"
        
        # Add character context if provided
        character_context_section = ""
        if character_context:
            character_context_section = f"\n**Character Context:**\n{character_context}\n"
        
        return base_prompt.format(
            episode_summary_section=episode_summary_section,
            scene_plot=scene_plot,
            character_context_section=character_context_section,
            dialogue_text=dialogue_text
        )
    
    def _parse_speaker_response(self, response_content: str) -> List[Dict]:
        """Parse LLM response for speaker identification."""
        try:
            logger.info(f"🔍 Raw LLM response ({len(response_content)} chars):")
            logger.info(f"Raw response: {response_content[:1000]}{'...' if len(response_content) > 1000 else ''}")
            
            # Clean and parse JSON response - this function already returns List[Dict]
            speaker_data = clean_llm_json_response(response_content)
            logger.info(f"✅ Parsed response: {len(speaker_data)} items")
            
            if not isinstance(speaker_data, list):
                logger.error(f"❌ Speaker data is not a list, got: {type(speaker_data)}")
                return []
            
            # Validate each entry
            valid_data = []
            for entry in speaker_data:
                if (isinstance(entry, dict) and 
                    "line_index" in entry and 
                    "speaker" in entry and 
                    "is_llm_confident" in entry):
                    
                    # Force confidence to false if other possible speakers are present
                    original_confidence = entry.get("is_llm_confident", False)
                    other_speakers = entry.get("other_possible_speakers", [])
                    
                    # If there are alternative speakers, force confidence to false
                    if other_speakers and len(other_speakers) > 0:
                        entry["is_llm_confident"] = False
                        logger.debug(f"🔍 Line {entry.get('line_index')}: Forcing confidence to false due to alternatives: {other_speakers}")
                    
                    valid_data.append(entry)
                else:
                    logger.warning(f"⚠️ Invalid speaker entry: {entry}")
            
            logger.info(f"✅ Parsed {len(valid_data)} valid speaker identifications")
            return valid_data
            
        except ValueError as e:
            logger.error(f"❌ JSON parsing error in speaker response: {e}")
            logger.error(f"Raw response: {response_content[:1000]}...")
            return []
        except Exception as e:
            logger.error(f"❌ Error parsing speaker response: {e}")
            logger.error(f"Raw response: {response_content[:1000]}...")
            return []
    
    def _update_dialogue_with_speakers(
        self, 
        dialogue_lines: List[DialogueLine], 
        speaker_data: List[Dict]
    ) -> List[DialogueLine]:
        """Update dialogue lines with speaker information."""
        
        # Create index mapping for quick lookup
        speaker_map = {
            entry["line_index"]: entry 
            for entry in speaker_data
        }
        
        updated_lines = []
        for line in dialogue_lines:
            if line.index in speaker_map:
                speaker_info = speaker_map[line.index]
                line.speaker = speaker_info["speaker"]
                line.is_llm_confident = bool(speaker_info["is_llm_confident"])
                logger.debug(f"📝 Line {line.index}: {line.speaker} (confident: {line.is_llm_confident})")
            else:
                logger.debug(f"⚠️ No speaker info for line {line.index}")
            
            updated_lines.append(line)
        
        return updated_lines
    
    def _update_dialogue_with_validated_speakers(
        self, 
        dialogue_lines: List[DialogueLine], 
        speaker_data: List[Dict],
        speaker_mapping: Dict[str, str]
    ) -> List[DialogueLine]:
        """Update dialogue lines with validated speaker information."""
        
        # Create index mapping for quick lookup
        speaker_map = {
            entry["line_index"]: entry 
            for entry in speaker_data
        }
        
        updated_lines = []
        for line in dialogue_lines:
            if line.index in speaker_map:
                speaker_info = speaker_map[line.index]
                original_speaker = speaker_info["speaker"]
                original_is_confident = bool(speaker_info["is_llm_confident"])
                other_possible_speakers = speaker_info.get("other_possible_speakers", [])
                
                # Store original LLM assignment
                line.original_llm_speaker = original_speaker
                line.original_llm_is_confident = original_is_confident
                line.other_possible_speakers = other_possible_speakers
                
                # Use validated speaker name from database
                validated_speaker = speaker_mapping.get(original_speaker, original_speaker)
                
                line.speaker = validated_speaker
                line.is_llm_confident = original_is_confident
                
                # Track resolution method based on boolean confidence
                if original_speaker != validated_speaker:
                    line.resolution_method = "database_validation"
                    logger.debug(f"📝 Line {line.index}: {original_speaker} → {validated_speaker} (confident: {original_is_confident})")
                elif original_is_confident:
                    line.resolution_method = "llm_direct"
                    logger.debug(f"📝 Line {line.index}: {validated_speaker} (confident: {original_is_confident}) - HIGH CONFIDENCE")
                else:
                    line.resolution_method = "llm_direct"  # Will be updated to face_clustering if resolved later
                    logger.debug(f"📝 Line {line.index}: {validated_speaker} (confident: {original_is_confident}) - LOW CONFIDENCE (may be updated by face clustering)")
                    
                # Log alternative speakers if present
                if other_possible_speakers:
                    logger.debug(f"🔍 Line {line.index}: Alternative speakers: {other_possible_speakers}")
            else:
                logger.debug(f"⚠️ No speaker info for line {line.index}")
            
            updated_lines.append(line)
        
        return updated_lines

    def identify_speakers_for_episode(
        self,
        plot_scenes: List[Dict],
        dialogue_lines: List[DialogueLine],
        character_context: Optional[str] = None,
        episode_entities: Optional[List[Dict]] = None,
        episode_plot: Optional[str] = None
    ) -> List[DialogueLine]:
        """
        Identify speakers for an entire episode.
        
        Args:
            plot_scenes: List of scene dictionaries with plot segments
            dialogue_lines: All dialogue lines for the episode
            character_context: Optional character context
            episode_entities: Optional list of character data dictionaries from current episode entity extraction
            episode_plot: Optional full episode plot for validation context
            
        Returns:
            Updated dialogue lines with speaker information
        """
        logger.info(f"🎬 Identifying speakers for episode with {len(plot_scenes)} scenes")
        
        # Group dialogue lines by scene timestamps
        scene_dialogue_map = self._group_dialogue_by_scenes(plot_scenes, dialogue_lines)
        
        updated_lines = []
        
        for scene in plot_scenes:
            scene_num = scene.get("scene_number", 0)
            scene_plot = scene.get("plot_segment", "")
            scene_lines = scene_dialogue_map.get(scene_num, [])
            
            if scene_lines:
                logger.info(f"🎭 Processing scene {scene_num} with {len(scene_lines)} dialogue lines")
                
                # Identify speakers for this scene
                updated_scene_lines = self.identify_speakers_for_scene(
                    scene_plot,
                    scene_lines,
                    character_context,
                    episode_entities=episode_entities,
                    episode_plot=episode_plot
                )
                
                # Update scene number
                for line in updated_scene_lines:
                    line.scene_number = scene_num
                
                updated_lines.extend(updated_scene_lines)
            else:
                logger.debug(f"⚠️ No dialogue found for scene {scene_num}")
        
        # Sort by original index
        updated_lines.sort(key=lambda x: x.index)
        
        logger.info(f"✅ Completed speaker identification for episode")
        return updated_lines
    
    def _group_dialogue_by_scenes(
        self, 
        plot_scenes: List[Dict], 
        dialogue_lines: List[DialogueLine]
    ) -> Dict[int, List[DialogueLine]]:
        """Group dialogue lines by scene based on timestamps or equal distribution."""
        scene_dialogue_map = {}
        
        # Check if scenes have timestamp information
        has_timestamps = any(
            scene.get("start_seconds") is not None and scene.get("end_seconds") is not None
            for scene in plot_scenes
        )
        
        if has_timestamps:
            logger.info("📍 Using timestamp-based scene grouping")
            # Use timestamp-based grouping
            for scene in plot_scenes:
                scene_num = scene.get("scene_number", 0)
                start_seconds = scene.get("start_seconds", 0)
                end_seconds = scene.get("end_seconds", float('inf'))
                
                # Find dialogue lines that fall within this scene's timeframe
                scene_lines = [
                    line for line in dialogue_lines
                    if start_seconds <= line.start_time < end_seconds
                ]
                
                scene_dialogue_map[scene_num] = scene_lines
                logger.debug(f"Scene {scene_num}: {len(scene_lines)} dialogue lines ({start_seconds:.1f}s - {end_seconds:.1f}s)")
        else:
            logger.info("⚖️ No timestamps found, using equal distribution scene grouping")
            # Distribute dialogue lines equally among scenes
            total_lines = len(dialogue_lines)
            num_scenes = len(plot_scenes)
            
            if num_scenes == 0:
                logger.warning("⚠️ No scenes defined, putting all dialogue in scene 1")
                scene_dialogue_map[1] = dialogue_lines
            else:
                lines_per_scene = total_lines // num_scenes
                remainder = total_lines % num_scenes
                
                start_idx = 0
                for i, scene in enumerate(plot_scenes):
                    scene_num = scene.get("scene_number", i + 1)
                    
                    # Add extra line to first few scenes if there's a remainder
                    extra_line = 1 if i < remainder else 0
                    scene_size = lines_per_scene + extra_line
                    
                    end_idx = start_idx + scene_size
                    scene_lines = dialogue_lines[start_idx:end_idx]
                    
                    scene_dialogue_map[scene_num] = scene_lines
                    logger.debug(f"Scene {scene_num}: {len(scene_lines)} dialogue lines (lines {start_idx+1}-{end_idx})")
                    
                    start_idx = end_idx
        
        return scene_dialogue_map
