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
            
            if not speaker_data:
                logger.warning("⚠️ No speaker data parsed from LLM response, returning original lines")
                # Return original lines with default values
                for line in scene_dialogue_lines:
                    if line.speaker is None:
                        line.speaker = None
                    if line.is_llm_confident is None:
                        line.is_llm_confident = False
                return scene_dialogue_lines
            
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
            
            # Validate that all lines were processed
            processed_indices = {line.index for line in updated_lines}
            original_indices = {line.index for line in scene_dialogue_lines}
            missing_indices = original_indices - processed_indices
            
            if missing_indices:
                logger.warning(f"⚠️ LLM didn't process {len(missing_indices)} dialogue lines, adding them back")
                for line in scene_dialogue_lines:
                    if line.index in missing_indices:
                        # Add missing lines with default values
                        line.speaker = None
                        line.is_llm_confident = False
                        updated_lines.append(line)
            
            # Log results
            confident_count = sum(
                1 for line in updated_lines 
                if line.is_llm_confident
            )
            
            logger.info(f"✅ Speaker identification complete: {confident_count}/{len(updated_lines)} with confident LLM assignments")
            
            return updated_lines
            
        except Exception as e:
            logger.error(f"❌ Error in speaker identification: {e}")
            logger.warning("⚠️ Returning original dialogue lines with default speaker assignments")
            
            # Ensure all lines have proper default values
            for line in scene_dialogue_lines:
                if line.speaker is None:
                    line.speaker = None
                if line.is_llm_confident is None:
                    line.is_llm_confident = False
            
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
        base_prompt = """You are an expert at identifying speakers in TV show dialogue. Analyze the following dialogue and identify the most likely speaker for each line.

{episode_summary_section}**Scene Plot:**
{scene_plot}

{character_context_section}
**Dialogue to Analyze:**
{dialogue_text}

**Instructions:**
1. For each dialogue line, identify the most likely speaker
2. Provide a boolean confidence (true/false) indicating if you are extremely confident about the speaker identification
3. Set is_llm_confident to true ONLY if you are extremely confident (100 percent sure) about the speaker
4. Set is_llm_confident to false if you have ANY doubts, uncertainties, or if the speaker could be multiple characters
5. Be very conservative - it's better to be uncertain than wrong
6. Consider dialogue content, context, speaking patterns, and character relationships
7. If the dialogue could reasonably be spoken by multiple characters, set is_llm_confident to false


**Confidence Guidelines:**
- Set to true ONLY if: The speaker is clear and unambiguous
- Set to false if: Any ambiguity, multiple possible speakers, unclear context, or general uncertainty

**Output Format (JSON only):**
[
  {{
    "line_index": 1,
    "speaker": "Character Name",
    "other_possible_speakers": ["Alternative Character 1"],
    "reasoning": "Could be multiple characters, not 100 percent certain"
    "is_llm_confident": false,
  }},
  {{
    "line_index": 2,
    "speaker": "Another Character",
    "other_possible_speakers": ["Alternative Character 2"],
    "reasoning": "Ambiguous context, multiple possible speakers",
    "is_llm_confident": false,
  }},
  {{
    "line_index": 3,
    "speaker": "Clear Speaker",
    "other_possible_speakers": [],
    "reasoning": "Speaker clearly identified by name in dialogue"
    "is_llm_confident": true,
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
        if not response_content or not response_content.strip():
            logger.error("❌ Empty LLM response")
            return []
            
        try:
            logger.info(f"🔍 Raw LLM response ({len(response_content)} chars):")
            logger.info(f"Raw response: {response_content[:1000]}{'...' if len(response_content) > 1000 else ''}")
            
            # Clean and parse JSON response - this function already returns List[Dict]
            speaker_data = clean_llm_json_response(response_content)
            
            if speaker_data is None:
                logger.error("❌ JSON parsing returned None")
                return []
                
            logger.info(f"✅ Parsed response: {len(speaker_data) if isinstance(speaker_data, list) else 'not a list'} items")
            
            if not isinstance(speaker_data, list):
                logger.error(f"❌ Speaker data is not a list, got: {type(speaker_data)}")
                
                # Try to handle case where LLM returned a single object instead of array
                if isinstance(speaker_data, dict):
                    logger.info("🔧 Converting single object to list")
                    speaker_data = [speaker_data]
                else:
                    logger.error(f"❌ Speaker data is not a dictionary or list, got: {type(speaker_data)}")
                    return []
            
            # Validate each entry and fix common issues
            valid_data = []
            for i, entry in enumerate(speaker_data):
                if not isinstance(entry, dict):
                    logger.warning(f"⚠️ Entry {i} is not a dictionary: {type(entry)}")
                    continue
                
                # Check required fields
                required_fields = ["line_index", "speaker", "is_llm_confident"]
                missing_fields = [field for field in required_fields if field not in entry]
                
                if missing_fields:
                    logger.warning(f"⚠️ Entry {i} missing required fields {missing_fields}: {entry}")
                    continue
                
                # Validate and fix field types
                try:
                    # Ensure line_index is an integer
                    line_index = int(entry["line_index"])
                    
                    # Ensure speaker is a string (could be None)
                    speaker = entry["speaker"]
                    if speaker is not None and not isinstance(speaker, str):
                        speaker = str(speaker)
                    
                    # Ensure is_llm_confident is a boolean
                    is_confident = entry["is_llm_confident"]
                    if isinstance(is_confident, str):
                        is_confident = is_confident.lower() in ['true', '1', 'yes', 'confident']
                    else:
                        is_confident = bool(is_confident)
                    
                    # Create cleaned entry
                    cleaned_entry = {
                        "line_index": line_index,
                        "speaker": speaker,
                        "is_llm_confident": is_confident
                    }
                    
                    # Add optional fields if present
                    for optional_field in ["reasoning", "other_possible_speakers"]:
                        if optional_field in entry:
                            cleaned_entry[optional_field] = entry[optional_field]
                    
                    valid_data.append(cleaned_entry)
                    logger.debug(f"✅ Valid entry {i}: line {line_index} → {speaker} (confident: {is_confident})")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Could not parse entry {i}: {e} - {entry}")
                    continue
            
            logger.info(f"✅ Parsed {len(valid_data)} valid speaker identifications out of {len(speaker_data)} total entries")
            
            if len(valid_data) == 0 and len(speaker_data) > 0:
                logger.error("❌ No valid speaker identifications parsed despite having data entries")
                logger.error(f"Sample entry: {speaker_data[0] if speaker_data else 'None'}")
            
            return valid_data
            
        except ValueError as e:
            logger.error(f"❌ JSON parsing error in speaker response: {e}")
            logger.error(f"Raw response: {response_content[:1000]}...")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing speaker response: {e}")
            logger.error(f"Raw response: {response_content[:1000]}...")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
                
                # Store original LLM assignment
                line.original_llm_speaker = original_speaker
                line.original_llm_is_confident = original_is_confident
                
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
                # The other_possible_speakers field is removed from the prompt and parsing,
                # so we don't log it here.
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
        failed_scenes = []
        
        for scene in plot_scenes:
            scene_num = scene.get("scene_number", 0)
            scene_plot = scene.get("plot_segment", "")
            scene_lines = scene_dialogue_map.get(scene_num, [])
            
            if scene_lines:
                logger.info(f"🎭 Processing scene {scene_num} with {len(scene_lines)} dialogue lines")
                
                try:
                    # Identify speakers for this scene with retry logic
                    updated_scene_lines = self._identify_speakers_for_scene_with_retry(
                    scene_plot,
                    scene_lines,
                    character_context,
                        episode_entities,
                        episode_plot,
                        scene_num
                    )
                
                    # Update scene number
                    for line in updated_scene_lines:
                        line.scene_number = scene_num
                    
                    updated_lines.extend(updated_scene_lines)
                    logger.info(f"✅ Successfully processed scene {scene_num}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process scene {scene_num}: {e}")
                    failed_scenes.append(scene_num)
                    
                    # Add the original scene lines without speaker assignments as fallback
                    for line in scene_lines:
                        line.scene_number = scene_num
                        # Ensure these fields are set even if processing failed
                        if line.speaker is None:
                            line.speaker = None
                        if line.is_llm_confident is None:
                            line.is_llm_confident = False
                    
                    updated_lines.extend(scene_lines)
            else:
                logger.debug(f"⚠️ No dialogue found for scene {scene_num}")
        
        # Handle failed scenes by processing them individually or in smaller batches
        if failed_scenes:
            logger.warning(f"⚠️ Retrying {len(failed_scenes)} failed scenes individually")
            updated_lines = self._retry_failed_scenes(
                failed_scenes, 
                scene_dialogue_map, 
                plot_scenes,
                updated_lines,
                character_context,
                episode_entities,
                episode_plot
            )
        
        # Sort by original index
        updated_lines.sort(key=lambda x: x.index)
        
        # Final validation - ensure no dialogue lines are missing
        processed_indices = {line.index for line in updated_lines}
        original_indices = {line.index for line in dialogue_lines}
        missing_indices = original_indices - processed_indices
        
        if missing_indices:
            logger.warning(f"⚠️ Found {len(missing_indices)} missing dialogue lines, adding them back")
            for line in dialogue_lines:
                if line.index in missing_indices:
                    # Add missing lines with default values
                    line.speaker = None
                    line.is_llm_confident = False
                    line.scene_number = 1  # Default scene
                    updated_lines.append(line)
            
            # Sort again after adding missing lines
            updated_lines.sort(key=lambda x: x.index)
        
        logger.info(f"✅ Completed speaker identification for episode: {len(updated_lines)} total dialogue lines")
        
        # Report statistics
        confident_count = sum(1 for line in updated_lines if line.is_llm_confident)
        assigned_count = sum(1 for line in updated_lines if line.speaker is not None)
        logger.info(f"📊 Results: {confident_count} confident assignments, {assigned_count} total assignments, {len(updated_lines) - assigned_count} unassigned")
        
        return updated_lines
    
    def _identify_speakers_for_scene_with_retry(
        self,
        scene_plot: str,
        scene_lines: List[DialogueLine],
        character_context: Optional[str] = None,
        episode_entities: Optional[List[Dict]] = None,
        episode_plot: Optional[str] = None,
        scene_num: int = 0,
        max_retries: int = 3
    ) -> List[DialogueLine]:
        """
        Identify speakers for a scene with retry logic.
        """
        for attempt in range(max_retries):
            try:
                logger.debug(f"🔄 Attempt {attempt + 1}/{max_retries} for scene {scene_num}")
                
                updated_lines = self.identify_speakers_for_scene(
                    scene_plot,
                    scene_lines,
                    character_context,
                    episode_entities=episode_entities,
                    episode_plot=episode_plot
                )
                
                # Validate that we got responses for all lines
                processed_indices = {line.index for line in updated_lines}
                original_indices = {line.index for line in scene_lines}
                
                if processed_indices == original_indices:
                    logger.debug(f"✅ Scene {scene_num} processed successfully on attempt {attempt + 1}")
                    return updated_lines
                else:
                    missing_count = len(original_indices - processed_indices)
                    logger.warning(f"⚠️ Scene {scene_num} missing {missing_count} dialogue lines on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        # On final attempt, fill in missing lines
                        return self._fill_missing_dialogue_lines(scene_lines, updated_lines)
                    
            except Exception as e:
                logger.warning(f"⚠️ Scene {scene_num} attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise  # Re-raise on final attempt
                
                # Wait before retry (exponential backoff)
                import time
                wait_time = 2 ** attempt
                logger.info(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        # This should never be reached due to the raise above
        raise Exception(f"Failed to process scene {scene_num} after {max_retries} attempts")
    
    def _retry_failed_scenes(
        self,
        failed_scenes: List[int],
        scene_dialogue_map: Dict[int, List[DialogueLine]],
        plot_scenes: List[Dict],
        updated_lines: List[DialogueLine],
        character_context: Optional[str] = None,
        episode_entities: Optional[List[Dict]] = None,
        episode_plot: Optional[str] = None
    ) -> List[DialogueLine]:
        """
        Retry failed scenes by processing dialogue lines individually.
        """
        scene_plot_map = {scene.get("scene_number", 0): scene.get("plot_segment", "") for scene in plot_scenes}
        
        for scene_num in failed_scenes:
            scene_lines = scene_dialogue_map.get(scene_num, [])
            scene_plot = scene_plot_map.get(scene_num, "")
            
            if not scene_lines:
                continue
                
            logger.info(f"🔄 Retrying scene {scene_num} with individual dialogue processing")
            
            # Remove the failed lines from updated_lines
            updated_lines = [line for line in updated_lines if line.scene_number != scene_num]
            
            # Process dialogue lines individually for this scene
            for i, line in enumerate(scene_lines):
                try:
                    logger.debug(f"🎭 Processing individual dialogue {line.index} in scene {scene_num}")
                    
                    # Process single dialogue line
                    single_line_result = self.identify_speakers_for_scene(
                        scene_plot,
                        [line],  # Single line
                        character_context,
                        episode_entities=episode_entities,
                        episode_plot=episode_plot
                    )
                    
                    if single_line_result:
                        processed_line = single_line_result[0]
                        processed_line.scene_number = scene_num
                        updated_lines.append(processed_line)
                        logger.debug(f"✅ Individual dialogue {line.index} processed successfully")
                    else:
                        # Fallback: add original line with no speaker assignment
                        line.scene_number = scene_num
                        line.speaker = None
                        line.is_llm_confident = False
                        updated_lines.append(line)
                        logger.debug(f"⚠️ Individual dialogue {line.index} fallback (no assignment)")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Individual dialogue {line.index} failed: {e}")
                    # Fallback: add original line with no speaker assignment
                    line.scene_number = scene_num
                    line.speaker = None
                    line.is_llm_confident = False
                    updated_lines.append(line)
        
        return updated_lines
    
    def _fill_missing_dialogue_lines(
        self,
        original_lines: List[DialogueLine],
        processed_lines: List[DialogueLine]
    ) -> List[DialogueLine]:
        """
        Fill in any missing dialogue lines that weren't processed.
        """
        processed_indices = {line.index for line in processed_lines}
        result_lines = list(processed_lines)
        
        for line in original_lines:
            if line.index not in processed_indices:
                # Add missing line with no speaker assignment
                line.speaker = None
                line.is_llm_confident = False
                result_lines.append(line)
                logger.debug(f"🔧 Added missing dialogue line {line.index}")
        
        return result_lines
    
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
