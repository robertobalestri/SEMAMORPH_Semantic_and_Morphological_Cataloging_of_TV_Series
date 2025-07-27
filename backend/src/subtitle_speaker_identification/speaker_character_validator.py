"""
Speaker Character Validator for SEMAMORPH.
Validates LLM-proposed speakers against the character database and handles new character creation.
"""
import json
from typing import List, Dict, Optional, Tuple
from ..narrative_storage_management.speaker_character_service import SpeakerCharacterService
from ..ai_models.ai_models import AzureChatOpenAI
from ..utils.llm_utils import clean_llm_json_response
from ..utils.logger_utils import setup_logging
from langchain_core.messages import HumanMessage

logger = setup_logging(__name__)

class SpeakerCharacterValidator:
    """Validates speaker names against character database and creates new characters when appropriate."""
    
    def __init__(self, series: str, llm: AzureChatOpenAI):
        self.series = series
        self.llm = llm
        self.speaker_character_service = SpeakerCharacterService()
        
    def validate_and_process_speakers(
        self, 
        proposed_speakers: List[str], 
        episode_entities: Optional[List[Dict]] = None,
        episode_plot: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Validate LLM-proposed speakers against database and process new characters.
        
        Args:
            proposed_speakers: List of speaker names proposed by LLM
            episode_entities: Optional list of character data dictionaries from current episode entity extraction
            episode_plot: Optional episode plot text for better context
            
        Returns:
            Dictionary mapping original LLM speaker names to their best appellations from database
        """
        logger.info(f"🔍 Validating {len(proposed_speakers)} proposed speakers against database")
        
        # Get existing characters from database as plain data (no session dependencies)
        existing_characters = self.speaker_character_service.get_all_characters_as_data(self.series)
        
        # Build mapping using the plain data
        appellation_to_character = self.speaker_character_service.build_appellation_mapping(existing_characters)
        
        # Include episode entities if provided - now already in safe format
        episode_characters_data = []
        if episode_entities:
            logger.info(f"📚 Processing {len(episode_entities)} episode entities for LLM context")
            episode_characters_data = episode_entities  # Already in data dictionary format
            
            # Add to appellation mapping
            for char_data in episode_characters_data:
                if char_data.get('best_appellation'):
                    appellation_to_character[char_data['best_appellation'].lower()] = char_data
                for appellation in char_data.get('appellations', []):
                    if appellation:
                        appellation_to_character[appellation.lower()] = char_data
            
            logger.info(f"📚 Successfully processed {len(episode_characters_data)} episode entities")
        
        logger.info(f"📖 Total known appellations: {len(appellation_to_character)}")
        
        # Process each proposed speaker
        speaker_mapping = {}
        new_characters_to_create = []
        
        for speaker in proposed_speakers:
            if not speaker or len(speaker.strip()) <= 1:
                continue
                
            speaker = speaker.strip()
            
            # Check if speaker already exists in appellations
            matched_character = self._find_matching_character(speaker, appellation_to_character)
            
            if matched_character:
                speaker_mapping[speaker] = matched_character['best_appellation']
                logger.info(f"✅ Found existing character: {speaker} → {matched_character['best_appellation']}")
            else:
                # Ask LLM if this should be associated with existing character or is new
                # Include both database characters and episode entities for full context
                all_characters_for_llm = existing_characters + episode_characters_data
                association_result = self._query_llm_for_character_association(
                    speaker, 
                    all_characters_for_llm,  # Use combined data
                    episode_plot
                )
                
                if association_result["action"] == "associate":
                    # Associate with existing character - search in both sources
                    target_character = self.speaker_character_service.find_character_by_entity_name(
                        association_result["entity_name"], 
                        all_characters_for_llm  # Search in combined data
                    )
                    if target_character:
                        speaker_mapping[speaker] = target_character['best_appellation']
                        
                        # Add this speaker as a new appellation to the existing character
                        success = self.speaker_character_service.add_appellation_to_character(
                            target_character['entity_name'], speaker, self.series
                        )
                        if success:
                            logger.info(f"🔗 Associated {speaker} with existing character {target_character['best_appellation']}")
                        else:
                            logger.warning(f"⚠️ Failed to add appellation for {speaker}")
                    else:
                        logger.warning(f"⚠️ LLM suggested association with {association_result['entity_name']} but character not found")
                        speaker_mapping[speaker] = speaker  # Keep original
                        
                elif association_result["action"] == "create":
                    # Create new character
                    new_character_data = association_result["character_data"]
                    if self._should_create_character(new_character_data):
                        new_characters_to_create.append(new_character_data)
                        speaker_mapping[speaker] = new_character_data["best_appellation"]
                        logger.info(f"🆕 Will create new character: {speaker} → {new_character_data['best_appellation']}")
                    else:
                        logger.info(f"⏭️ Skipping character creation for: {speaker} (generic/unnamed character)")
                        speaker_mapping[speaker] = speaker  # Keep original
                else:
                    # Keep original speaker name
                    speaker_mapping[speaker] = speaker
                    logger.info(f"🤷 Keeping original speaker name: {speaker}")
        
        # Create new characters in database
        if new_characters_to_create:
            success = self.speaker_character_service.create_new_characters(new_characters_to_create, self.series)
            if not success:
                logger.warning("⚠️ Some new characters failed to be created")
        
        logger.info(f"✅ Speaker validation complete. Mapping: {len(speaker_mapping)} speakers")
        return speaker_mapping
    
    def _find_matching_character(self, speaker: str, appellation_mapping: Dict[str, Dict]) -> Optional[Dict]:
        """Find a character data dict that matches the speaker name."""
        speaker_lower = speaker.lower()
        
        # Direct match
        if speaker_lower in appellation_mapping:
            return appellation_mapping[speaker_lower]
        
        # Fuzzy matching - check if speaker is contained in any appellation or vice versa
        for appellation, char_data in appellation_mapping.items():
            if self._names_are_similar(speaker_lower, appellation):
                return char_data
                
        return None
    
    def _names_are_similar(self, name1: str, name2: str) -> bool:
        """Check if two names are similar enough to be considered the same character."""
        # Simple similarity check - can be enhanced
        name1_parts = set(name1.split())
        name2_parts = set(name2.split())
        
        # If they share significant parts, consider them similar
        intersection = name1_parts.intersection(name2_parts)
        union = name1_parts.union(name2_parts)
        
        if len(union) == 0:
            return False
            
        similarity = len(intersection) / len(union)
        return similarity > 0.5  # 50% similarity threshold
    
    def _query_llm_for_character_association(
        self, 
        speaker_name: str, 
        existing_characters: List[Dict],
        episode_plot: Optional[str] = None
    ) -> Dict:
        """Ask LLM if the speaker should be associated with existing character or is new."""
        
        # Format existing characters for the prompt (using service)
        characters_info = self.speaker_character_service.build_characters_info_for_llm(existing_characters)
        
        # Build episode context section
        episode_context_section = ""
        if episode_plot:
            # Truncate plot if too long to avoid token limits
            truncated_plot = episode_plot[:2000] + "..." if len(episode_plot) > 2000 else episode_plot
            episode_context_section = f"""
**Episode Plot Context:**
{truncated_plot}

"""
        
        prompt = f"""You are analyzing a speaker name from TV episode dialogue to determine if it refers to an existing character or is a new character.

**Speaker Name from Dialogue:** "{speaker_name}"

{episode_context_section}

**Existing Characters in Database:**
{json.dumps(characters_info, indent=2)}

**Instructions:**
1. Determine if "{speaker_name}" refers to one of the existing characters (considering nicknames, variations, etc.)
2. Use the episode plot context to better understand who this speaker might be
3. If it matches an existing character, specify which one
4. If it's a new character, decide if it should be added to the database:
   - Add named characters (e.g., "John", "Mary Smith", "Dr. Wilson")
   - Do NOT add generic characters (e.g., "Nurse 1", "Guard", "Waitress", "Man in suit")
5. For new named characters, suggest entity_name, best_appellation, appellations, and biological_sex

**Output Format (JSON only):**
Return a single JSON object (not an array) with this exact structure:
{{
  "action": "associate|create|ignore",
  "reasoning": "Brief explanation of decision including episode context if relevant",
  "entity_name": "existing_entity_name_if_associating",
  "character_data": {{
    "entity_name": "normalized_name",
    "best_appellation": "display_name", 
    "appellations": ["name1", "name2"],
    "biological_sex": "M" or "F" or null
  }}
}}

For "associate": include entity_name of existing character
For "create": include character_data for new character (including biological_sex)
For "ignore": neither entity_name nor character_data needed

**Biological Sex Guidelines:**
- Use context clues, titles (Mr./Mrs./Ms./Miss), pronouns (he/she), and character names to determine biological sex
- Use 'M' for male, 'F' for female, or null for unknown/unclear cases
- If you're unsure, use null rather than guessing

**IMPORTANT:** Return ONLY a single JSON object, no additional text, no arrays."""

        try:
            logger.info(f"🤖 Querying LLM for character association: {speaker_name}")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content.strip()
            
            # Parse JSON response
            result = clean_llm_json_response(response_content)
            
            # Handle case where LLM returns an array instead of a single object
            if isinstance(result, list) and len(result) > 0:
                result = result[0]  # Take the first element
            
            if isinstance(result, dict) and "action" in result:
                logger.info(f"🎯 LLM decision for '{speaker_name}': {result['action']} - {result.get('reasoning', 'No reasoning provided')}")
                return result
            else:
                logger.error(f"❌ Invalid LLM response format: {result}")
                return {"action": "ignore", "reasoning": "Invalid LLM response"}
                
        except Exception as e:
            logger.error(f"❌ Error querying LLM for character association: {e}")
            return {"action": "ignore", "reasoning": f"LLM error: {e}"}
    
    def _should_create_character(self, character_data: Dict) -> bool:
        """Determine if a character should be created based on LLM suggestion."""
        if not character_data:
            return False
            
        entity_name = character_data.get("entity_name", "")
        best_appellation = character_data.get("best_appellation", "")
        
        # Must have meaningful name (more than 2 characters)
        return len(entity_name.strip()) > 2 and len(best_appellation.strip()) > 2
